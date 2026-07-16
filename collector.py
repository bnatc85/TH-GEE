from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from urllib.parse import urlencode

import websockets
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from database import (
    EngagementEvent,
    SessionLocal,
    TrackedPost,
    TrackedQuery,
    initialize_database,
)


JETSTREAM_HOST = "jetstream2.us-east.bsky.network"

COLLECTIONS = [
    "app.bsky.feed.post",
    "app.bsky.feed.like",
    "app.bsky.feed.repost",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def build_jetstream_uri() -> str:
    query_string = urlencode(
        [("wantedCollections", collection) for collection in COLLECTIONS]
    )
    return f"wss://{JETSTREAM_HOST}/subscribe?{query_string}"


def event_datetime(event: dict) -> datetime:
    time_us = event.get("time_us")

    if isinstance(time_us, int):
        return datetime.fromtimestamp(time_us / 1_000_000, tz=timezone.utc)

    return datetime.now(timezone.utc)


def at_uri(did: str, collection: str, rkey: str) -> str:
    return f"at://{did}/{collection}/{rkey}"


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def query_matches(text: str, query: TrackedQuery) -> bool:
    normalized_text = normalize_text(text)
    normalized_query = normalize_text(query.query)

    if not normalized_query:
        return False

    if query.query_type == "hashtag":
        hashtag = normalized_query
        if not hashtag.startswith("#"):
            hashtag = f"#{hashtag}"

        pattern = rf"(?<!\w){re.escape(hashtag)}(?!\w)"
        return re.search(pattern, normalized_text) is not None

    return normalized_query in normalized_text


def extract_subject_uri(record: dict) -> str | None:
    subject = record.get("subject")

    if isinstance(subject, dict):
        uri = subject.get("uri")
        return uri if isinstance(uri, str) else None

    return None


def extract_reply_parent(record: dict) -> str | None:
    reply = record.get("reply")

    if not isinstance(reply, dict):
        return None

    parent = reply.get("parent")

    if not isinstance(parent, dict):
        return None

    uri = parent.get("uri")
    return uri if isinstance(uri, str) else None


def extract_quote_uri(record: dict) -> str | None:
    embed = record.get("embed")

    if not isinstance(embed, dict):
        return None

    # app.bsky.embed.record
    embedded_record = embed.get("record")

    if isinstance(embedded_record, dict):
        uri = embedded_record.get("uri")
        if isinstance(uri, str):
            return uri

        nested_record = embedded_record.get("record")
        if isinstance(nested_record, dict):
            nested_uri = nested_record.get("uri")
            if isinstance(nested_uri, str):
                return nested_uri

    return None


def active_queries(session) -> list[TrackedQuery]:
    statement = select(TrackedQuery).where(TrackedQuery.active.is_(True))
    return list(session.scalars(statement))


def tracked_post_for_uri(session, uri: str | None) -> TrackedPost | None:
    if not uri:
        return None

    return session.get(TrackedPost, uri)


def save_event(
    session,
    *,
    query_id: int,
    event_uri: str,
    target_uri: str | None,
    actor_did: str,
    event_type: str,
    timestamp: datetime,
) -> None:
    event = EngagementEvent(
        query_id=query_id,
        event_uri=event_uri,
        target_uri=target_uri,
        actor_did=actor_did,
        event_type=event_type,
        event_time=timestamp,
        observed_at=datetime.now(timezone.utc),
    )

    session.add(event)

    try:
        session.commit()
    except IntegrityError:
        session.rollback()


def handle_matching_post(
    session,
    *,
    query: TrackedQuery,
    uri: str,
    cid: str | None,
    actor_did: str,
    text: str,
    created_at: datetime,
    observed_at: datetime,
) -> None:
    existing = session.get(TrackedPost, uri)

    if existing is None:
        post = TrackedPost(
            uri=uri,
            cid=cid,
            author_did=actor_did,
            text=text,
            created_at=created_at,
            observed_at=observed_at,
            query_id=query.id,
        )
        session.add(post)

        try:
            session.commit()
        except IntegrityError:
            session.rollback()

    save_event(
        session,
        query_id=query.id,
        event_uri=uri,
        target_uri=uri,
        actor_did=actor_did,
        event_type="publication",
        timestamp=created_at,
    )


def process_post_event(
    session,
    *,
    actor_did: str,
    uri: str,
    cid: str | None,
    record: dict,
    timestamp: datetime,
) -> None:
    text = record.get("text", "")
    if not isinstance(text, str):
        text = ""

    created_at_value = record.get("createdAt")
    try:
        created_at = datetime.fromisoformat(
            str(created_at_value).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        created_at = timestamp

    queries = active_queries(session)

    for query in queries:
        if query_matches(text, query):
            handle_matching_post(
                session,
                query=query,
                uri=uri,
                cid=cid,
                actor_did=actor_did,
                text=text,
                created_at=created_at,
                observed_at=timestamp,
            )

    reply_parent = extract_reply_parent(record)
    parent_post = tracked_post_for_uri(session, reply_parent)

    if parent_post is not None:
        save_event(
            session,
            query_id=parent_post.query_id,
            event_uri=uri,
            target_uri=reply_parent,
            actor_did=actor_did,
            event_type="reply",
            timestamp=created_at,
        )

    quote_uri = extract_quote_uri(record)
    quoted_post = tracked_post_for_uri(session, quote_uri)

    if quoted_post is not None:
        save_event(
            session,
            query_id=quoted_post.query_id,
            event_uri=uri,
            target_uri=quote_uri,
            actor_did=actor_did,
            event_type="quote",
            timestamp=created_at,
        )


def process_interaction_event(
    session,
    *,
    actor_did: str,
    event_uri: str,
    collection: str,
    record: dict,
    timestamp: datetime,
) -> None:
    target_uri = extract_subject_uri(record)
    target_post = tracked_post_for_uri(session, target_uri)

    if target_post is None:
        return

    event_type = {
        "app.bsky.feed.like": "like",
        "app.bsky.feed.repost": "repost",
    }.get(collection)

    if event_type is None:
        return

    save_event(
        session,
        query_id=target_post.query_id,
        event_uri=event_uri,
        target_uri=target_uri,
        actor_did=actor_did,
        event_type=event_type,
        timestamp=timestamp,
    )


def process_message(message: str) -> None:
    event = json.loads(message)

    if event.get("kind") != "commit":
        return

    did = event.get("did")
    commit = event.get("commit", {})

    if not isinstance(did, str) or not isinstance(commit, dict):
        return

    operation = commit.get("operation")
    collection = commit.get("collection")
    rkey = commit.get("rkey")
    record = commit.get("record")
    cid = commit.get("cid")

    if operation != "create":
        return

    if not isinstance(collection, str):
        return

    if not isinstance(rkey, str):
        return

    if not isinstance(record, dict):
        return

    uri = at_uri(did, collection, rkey)
    timestamp = event_datetime(event)

    with SessionLocal() as session:
        if collection == "app.bsky.feed.post":
            process_post_event(
                session,
                actor_did=did,
                uri=uri,
                cid=cid if isinstance(cid, str) else None,
                record=record,
                timestamp=timestamp,
            )
        else:
            process_interaction_event(
                session,
                actor_did=did,
                event_uri=uri,
                collection=collection,
                record=record,
                timestamp=timestamp,
            )


async def listen_forever() -> None:
    uri = build_jetstream_uri()
    reconnect_delay = 2

    while True:
        try:
            logging.info("Connecting to %s", uri)

            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
                max_size=None,
            ) as websocket:
                logging.info("Connected to Bluesky Jetstream")
                reconnect_delay = 2

                async for message in websocket:
                    try:
                        process_message(message)
                    except Exception:
                        logging.exception("Failed to process event")

        except asyncio.CancelledError:
            raise

        except Exception:
            logging.exception(
                "Connection failed. Reconnecting in %s seconds.",
                reconnect_delay,
            )
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)


def main() -> None:
    initialize_database()

    try:
        asyncio.run(listen_forever())
    except KeyboardInterrupt:
        logging.info("Collector stopped")


if __name__ == "__main__":
    main()