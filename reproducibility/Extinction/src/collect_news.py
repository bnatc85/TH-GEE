"""Collect English-language AI-catastrophe news metadata, one query per day.

Implements paper Sec. 5.1 and Appendix B.

Source choice. The paper's reproducible default is the GDELT DOC 2.0 API. That
endpoint refused every request from this environment with its shared-IP rate
limit, and its static GKG mirror is ~5 MB per 15-minute slice (~30 GB for one
event window), so neither is usable here. The collector therefore uses Google
News RSS, which accepts ``after:``/``before:`` date bounds and so reaches the
May 2023 replication window as well as the 2026 episode -- something the GDELT
DOC rolling archive cannot do. ``fetch_day`` is the only source-specific
function; swapping back to GDELT or Event Registry means replacing it alone.

Two properties of this source are recorded rather than hidden:

* **Right-censoring.** A query returns at most 100 items per day, and that cap
  binds on the highest-volume days. Daily story counts are therefore a censored
  index, not a census. Each day carries a ``censored`` flag. This is why the
  analysis uses frame *proportions* as the primary media construct (paper
  Sec. 5.1: "the substantive analysis emphasizes frame prevalence"), which are
  far more robust to a cap than raw counts.
* **Ranked selection.** Results are ranked, not exhaustive, so outlet
  composition is not a random sample of the press.

Running the base query together with the four mechanism-specific supplemental
queries of Appendix B raises effective daily coverage above any single query's
cap, because the union of five capped result sets is larger than one.

Usage:
    python src/collect_news.py                # both events
    python src/collect_news.py --event warning2026
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW, daterange, event_window, load_events, save_json, write_manifest

RSS = "https://news.google.com/rss/search"
BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/121.0 Safari/537.36")
REQUEST_SPACING = 1.5
PER_QUERY_CAP = 100          # observed hard cap on RSS results per query

# Paper Appendix B: the conceptual query.
BASE_QUERY = ('("artificial intelligence" OR AI OR AGI OR superintelligence OR '
              'Anthropic OR OpenAI) AND (extinction OR existential OR '
              'catastrophic OR "loss of control" OR "human survival")')

# Mechanism-specific supplemental queries (paper Appendix B). These raise recall
# of mechanism coverage; frame assignment is done by the classifier, not by
# which query retrieved a story.
SUPPLEMENTAL = {
    "infrastructure": '(AI OR "artificial intelligence") AND ("critical infrastructure" OR "power grid" OR "cyber attack")',
    "control": '(AI OR "artificial intelligence") AND ("kill switch" OR "emergency shutdown" OR "fail-safe" OR containment)',
    "bio": '(AI OR "artificial intelligence") AND ("biological weapon" OR biosecurity OR bioweapon)',
    "governance": '(AI OR "artificial intelligence") AND (regulation OR pause OR slowdown OR moratorium OR oversight) AND (risk OR safety)',
}

TITLE_SOURCE_SUFFIX = re.compile(r"\s+-\s+[^-]+$")


def normalise_title(t: str) -> str:
    t = TITLE_SOURCE_SUFFIX.sub("", t or "")
    t = re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", t.lower()).strip())
    return t


def canonical_url(u: str) -> str:
    try:
        p = urllib.parse.urlsplit(u)
    except ValueError:
        return u
    q = [(k, v) for k, v in urllib.parse.parse_qsl(p.query)
         if not k.lower().startswith(("utm_", "fbclid", "gclid", "ref", "oc"))]
    return urllib.parse.urlunsplit((p.scheme, p.netloc.lower().removeprefix("www."),
                                    p.path.rstrip("/"), urllib.parse.urlencode(q), ""))


def fetch_day(query: str, day: date, retries: int = 4) -> tuple[list[dict], bool]:
    """Articles for one day, plus whether the per-query cap binds."""
    params = {"q": f"{query} after:{day.isoformat()} "
                   f"before:{(day + timedelta(days=1)).isoformat()}",
              "hl": "en-US", "gl": "US", "ceid": "US:en"}
    url = f"{RSS}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})

    xml = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                xml = resp.read().decode("utf-8", errors="replace")
            break
        except Exception:                                    # noqa: BLE001
            time.sleep(3.0 * (attempt + 1))
    if xml is None:
        return [], False

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return [], False

    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src_el = it.find("source")
        items.append({
            "title": TITLE_SOURCE_SUFFIX.sub("", title),
            "raw_title": title,
            "url": (it.findtext("link") or "").strip(),
            "outlet": (src_el.text.strip() if src_el is not None and src_el.text
                       else ""),
            "published": (it.findtext("pubDate") or "").strip(),
            "date_only": day.isoformat(),
        })
    return items, len(items) >= PER_QUERY_CAP


def dedupe(articles: list[dict]) -> list[dict]:
    """Canonical-URL dedup, then normalised-title similarity within a day.

    The first copy seen is the retained story; later near-identical titles are
    kept but flagged ``syndicated``, so reach can be reported separately while
    frame prevalence counts each underlying story once (paper Sec. 5.1).
    """
    # A story retrieved by several queries is one story, but which queries found
    # it must survive: the base query defines the paper's corpus, and the
    # supplemental queries are separate mechanism-coverage indicators. Collapsing
    # to the first-seen copy alone would silently reassign stories between them.
    by_url: dict[str, dict] = {}
    for a in articles:
        cu = canonical_url(a["url"])
        key = cu or a["title"]
        if not key:
            continue
        if key in by_url:
            by_url[key]["queries"].append(a["query"])
        else:
            by_url[key] = {**a, "canonical_url": cu, "queries": [a["query"]]}

    by_day: dict[str, list[dict]] = {}
    out: list[dict] = []
    for art in by_url.values():
        art["queries"] = sorted(set(art["queries"]))
        art["norm_title"] = normalise_title(art["raw_title"])
        peers = by_day.setdefault(art["date_only"], [])
        twin = next((p for p in peers if art["norm_title"] and
                     SequenceMatcher(None, art["norm_title"],
                                     p["norm_title"]).ratio() > 0.88), None)
        art["syndicated"] = twin is not None
        if twin is not None:
            # Same story, different outlet: the retained copy inherits the
            # syndicated copy's query membership.
            twin["queries"] = sorted(set(twin["queries"]) | set(art["queries"]))
        peers.append(art)
        out.append(art)
    return out


def collect(event: dict, analysis: dict, today: date) -> dict:
    t0 = event["t0"]
    start, end = event_window(t0, analysis["pre_event_days"],
                              analysis["post_event_days"])
    end = min(end, today)
    days = list(daterange(start, end))
    queries = {"base": BASE_QUERY, **SUPPLEMENTAL}

    articles: list[dict] = []
    censored_days: dict[str, list[str]] = {}
    for name, q in queries.items():
        for i, day in enumerate(days, 1):
            items, censored = fetch_day(q, day)
            for a in items:
                a["query"] = name
            articles.extend(items)
            if censored:
                censored_days.setdefault(name, []).append(day.isoformat())
            if i % 10 == 0 or censored:
                print(f"  [{name}] {day} -> {len(items)}"
                      f"{' CAPPED' if censored else ''} ({i}/{len(days)})",
                      flush=True)
            time.sleep(REQUEST_SPACING)

    deduped = dedupe(articles)
    unique = [a for a in deduped if not a["syndicated"]]
    return {
        "event_id": event["id"], "t0": t0.isoformat(),
        "window": [start.isoformat(), end.isoformat()],
        "truncated_at_today": end < t0 + timedelta(days=analysis["post_event_days"]),
        "source": "Google News RSS",
        "per_query_cap": PER_QUERY_CAP,
        "queries": queries,
        "censored_days": censored_days,
        "n_raw": len(articles), "n_after_dedupe": len(deduped),
        "n_unique_stories": len(unique),
        "articles": deduped,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event")
    ap.add_argument("--today", default=date.today().isoformat())
    args = ap.parse_args()

    events_cfg = load_events()
    targets = [e for e in events_cfg["events"]
               if args.event is None or e["id"] == args.event]
    if not targets:
        raise SystemExit(f"unknown event {args.event!r}")

    today = date.fromisoformat(args.today)
    summary = {}
    for event in targets:
        print(f"[{event['id']}] collecting news ...", flush=True)
        payload = collect(event, events_cfg["analysis"], today)
        save_json(RAW / f"news_{event['id']}.json", payload)
        summary[event["id"]] = {
            "n_raw": payload["n_raw"],
            "n_unique_stories": payload["n_unique_stories"],
            "censored_days": {k: len(v) for k, v in payload["censored_days"].items()},
            "window": payload["window"],
        }
        print(f"[{event['id']}] {payload['n_unique_stories']} unique stories "
              f"from {payload['n_raw']} raw records", flush=True)

    write_manifest("collect_news", source="Google News RSS", events=summary)


if __name__ == "__main__":
    main()
