"""YouTube engagement collector for TH-GEE.

Replaces the live Bluesky firehose with the YouTube Data API v3. Unlike a live
stream, a single search returns videos whose publication dates already span
months or years, so a temporal engagement trace is available immediately
instead of accumulating in real time. The former Bluesky collector is preserved
in ``collector_bluesky.py``.

Model (video-publication-with-engagement, mirroring the RedNote-Vibe analysis):
each matching video is treated like a timestamped post carrying its own final
engagement. At the video's publication time we emit, into the shared
``EngagementEvent`` table:

  * ``publication`` -- one event per video (magnitude 1);
  * ``like``        -- magnitude = the video's like count;
  * ``comment``     -- magnitude = the video's comment count.

Because every event is anchored to the video's *publication* time, the trace
reflects when content about a topic was actually produced and engaged with, over
the topic's whole history -- rather than clustering at recent dates the way a
newest-first comment scrape does. Optionally (YT_COLLECT_COMMENT_THREADS=1) the
collector also fetches individual timestamped comments.

Configuration (environment variables):
  YOUTUBE_API_KEY            required; a YouTube Data API v3 key.
  YT_MAX_VIDEOS              videos per query (default 50).
  YT_SEARCH_ORDER            'viewCount' (default, widest time span), 'relevance',
                             'date', 'rating'.
  YT_COLLECT_COMMENT_THREADS '1' to also fetch timestamped comments (default '0').
  YT_MAX_COMMENT_PAGES       comment pages (100 each) per video when enabled (5).
  YT_POLL_INTERVAL           seconds between passes; 0 (default) runs once.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from database import (
    EngagementEvent,
    SessionLocal,
    TrackedPost,
    TrackedQuery,
    initialize_database,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

MAX_VIDEOS = int(os.getenv("YT_MAX_VIDEOS", "50"))
SEARCH_ORDER = os.getenv("YT_SEARCH_ORDER", "viewCount")
COLLECT_COMMENT_THREADS = os.getenv("YT_COLLECT_COMMENT_THREADS", "0") == "1"
MAX_COMMENT_PAGES = int(os.getenv("YT_MAX_COMMENT_PAGES", "5"))
POLL_INTERVAL = int(os.getenv("YT_POLL_INTERVAL", "0"))


def build_client(api_key: str | None = None):
    api_key = api_key or os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit(
            "YOUTUBE_API_KEY is not set. Create a free YouTube Data API v3 key "
            "and export it, e.g. `export YOUTUBE_API_KEY=...`."
        )
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def active_queries(session) -> list[TrackedQuery]:
    return list(session.scalars(select(TrackedQuery).where(TrackedQuery.active.is_(True))))


def _add_event(pending: list, *, query_id, event_uri, target_uri, actor, etype, ts, magnitude):
    pending.append({
        "query_id": query_id,
        "event_uri": event_uri,
        "target_uri": target_uri,
        "actor_did": actor,
        "event_type": etype,
        "event_time": ts,
        "magnitude": float(magnitude),
        "observed_at": datetime.now(timezone.utc),
    })


def bulk_save(session, pending: list) -> int:
    """Insert events, skipping ones already present. One SELECT + one INSERT."""
    if not pending:
        return 0
    uris = [event["event_uri"] for event in pending]
    existing = set(session.execute(
        select(EngagementEvent.event_uri, EngagementEvent.event_type)
        .where(EngagementEvent.event_uri.in_(uris))
    ).all())
    fresh = [e for e in pending if (e["event_uri"], e["event_type"]) not in existing]
    if not fresh:
        return 0
    session.add_all([EngagementEvent(**event) for event in fresh])
    try:
        session.commit()
        return len(fresh)
    except IntegrityError:
        session.rollback()
        # Fall back to per-row inserts if a concurrent run raced us.
        written = 0
        for event in fresh:
            session.add(EngagementEvent(**event))
            try:
                session.commit()
                written += 1
            except IntegrityError:
                session.rollback()
        return written


def search_videos(yt, query: str, max_videos: int) -> list[str]:
    ids: list[str] = []
    page_token = None
    while len(ids) < max_videos:
        response = yt.search().list(
            q=query, part="id", type="video", order=SEARCH_ORDER,
            maxResults=min(50, max_videos - len(ids)), pageToken=page_token,
        ).execute()
        ids.extend(item["id"]["videoId"] for item in response.get("items", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return ids


def fetch_video_details(yt, video_ids: list[str]) -> list[dict]:
    details: list[dict] = []
    for start in range(0, len(video_ids), 50):
        chunk = video_ids[start:start + 50]
        response = yt.videos().list(id=",".join(chunk), part="snippet,statistics").execute()
        for item in response.get("items", []):
            snippet = item["snippet"]
            stats = item.get("statistics", {})
            details.append({
                "videoId": item["id"],
                "publishedAt": parse_time(snippet["publishedAt"]),
                "channelId": snippet.get("channelId", ""),
                "title": snippet.get("title", ""),
                "likeCount": int(stats.get("likeCount", 0) or 0),
                "commentCount": int(stats.get("commentCount", 0) or 0),
            })
    return details


def upsert_video(session, *, query_id: int, video: dict) -> None:
    if session.get(TrackedPost, video["videoId"]) is None:
        session.add(TrackedPost(
            uri=video["videoId"], cid=None, author_did=video["channelId"],
            text=video["title"][:500], created_at=video["publishedAt"],
            observed_at=datetime.now(timezone.utc), query_id=query_id,
        ))
        try:
            session.commit()
        except IntegrityError:
            session.rollback()


def collect_comment_threads(yt, query_id: int, video_id: str, pending: list) -> None:
    page_token = None
    for _ in range(MAX_COMMENT_PAGES):
        try:
            response = yt.commentThreads().list(
                videoId=video_id, part="snippet,replies", maxResults=100,
                order="time", textFormat="plainText", pageToken=page_token,
            ).execute()
        except HttpError as error:
            if error.resp.status in (403, 404):
                return
            raise
        for thread in response.get("items", []):
            top = thread["snippet"]["topLevelComment"]
            ts = parse_time(top["snippet"]["publishedAt"])
            _add_event(pending, query_id=query_id,
                       event_uri=f"q{query_id}:comment:{top['id']}", target_uri=video_id,
                       actor="", etype="comment", ts=ts,
                       magnitude=1 + int(top["snippet"].get("likeCount", 0) or 0))
        page_token = response.get("nextPageToken")
        if not page_token:
            return


def collect_query(yt, query: TrackedQuery, session) -> dict:
    logging.info("Searching YouTube for %r (id=%s)", query.query, query.id)
    video_ids = search_videos(yt, query.query, MAX_VIDEOS)
    videos = fetch_video_details(yt, video_ids)
    pending: list = []
    for video in videos:
        upsert_video(session, query_id=query.id, video=video)
        vid = video["videoId"]
        ts = video["publishedAt"]
        _add_event(pending, query_id=query.id, event_uri=f"q{query.id}:video:{vid}",
                   target_uri=vid, actor=video["channelId"], etype="publication",
                   ts=ts, magnitude=1.0)
        _add_event(pending, query_id=query.id, event_uri=f"q{query.id}:videolikes:{vid}",
                   target_uri=vid, actor=video["channelId"], etype="like",
                   ts=ts, magnitude=video["likeCount"])
        _add_event(pending, query_id=query.id, event_uri=f"q{query.id}:videocomments:{vid}",
                   target_uri=vid, actor=video["channelId"], etype="comment",
                   ts=ts, magnitude=video["commentCount"])
        if COLLECT_COMMENT_THREADS:
            collect_comment_threads(yt, query.id, vid, pending)
    written = bulk_save(session, pending)
    logging.info("Query %r: %d videos, %d new events", query.query, len(videos), written)
    return {"videos": len(videos), "events": written}


def collect_for_query(query_id: int, api_key: str | None = None) -> dict:
    """Collect one query by id (used by the Streamlit 'Collect' button)."""
    yt = build_client(api_key)
    with SessionLocal() as session:
        query = session.get(TrackedQuery, query_id)
        if query is None:
            return {"videos": 0, "events": 0}
        return collect_query(yt, query, session)


def run_once(yt) -> None:
    with SessionLocal() as session:
        queries = active_queries(session)
    if not queries:
        logging.info("No active queries. Add one in the app, then re-run.")
        return
    for query in queries:
        with SessionLocal() as session:
            try:
                collect_query(yt, query, session)
            except HttpError as error:
                logging.error("YouTube API error for query %r: %s", query.query, error)


def main() -> None:
    initialize_database()
    yt = build_client()
    try:
        while True:
            run_once(yt)
            if POLL_INTERVAL <= 0:
                break
            logging.info("Sleeping %d s before next pass.", POLL_INTERVAL)
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        logging.info("Collector stopped")


if __name__ == "__main__":
    main()
