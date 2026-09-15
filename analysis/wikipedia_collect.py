"""Collect daily Wikipedia pageview series for the paper's 37 topics.

Why this exists: the YouTube application anchors engagement to *publication*
time, so a quarterly value is the eventual engagement of a publication cohort,
not audience activity during that quarter (paper Sections 4.2 and 6.3).
Wikimedia's per-article pageview API serves daily *audience* counts back to
2015-07-01, which is an event-level attention series for the same topics. This
script fetches those series so the temporal profiles can be recomputed on
event-level data and compared against the publication-cohort profiles.

Mapping policy (each topic records ``fidelity``, used to stratify the results):

  high    the article's subject is the topic (e.g. "pizzagate" ->
          "Pizzagate conspiracy theory")
  medium  the article's subject contains the topic but is broader, so the
          series carries off-topic traffic (e.g. "area 51 aliens" -> "Area 51",
          whose article is mostly about the airbase)
  low     the closest article is substantially broader than the search term
          (e.g. "cern portals" -> "CERN")

Redirect policy: Wikimedia records a pageview against the *requested* title, so
a redirect title has its own attention series. Where the topic name redirects to
an article on the same subject under a different name (e.g. "Phantom time
hypothesis" -> "Phantom time conspiracy theory") we use the target. Where it
redirects to a broader or different subject (e.g. "Mandela effect" -> "False
memory", "Project Blue Beam" -> "Serge Monast") we use the redirect title
itself, because its series is topic-specific; ``alt`` records the target so the
choice can be tested (see --alts).

API: no key, no quota. One GET per article. Days with zero recorded views are
omitted by the API and are back-filled with 0 here, so every series covers the
full horizon on a common daily grid.

Usage:
    python analysis/wikipedia_collect.py            # -> wikipedia_raw.json
    python analysis/wikipedia_collect.py --alts     # also fetch alternates
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_PATH = HERE / "wikipedia_raw.json"

# Frozen horizon. 2015-07-01 is the first day the pageviews API serves, and is
# also the first day the agent=user (bot-filtered) classification is available.
START = date(2015, 7, 1)
END = date(2026, 6, 30)

PROJECT = "en.wikipedia"
ACCESS = "all-access"   # desktop + mobile web + mobile app
AGENT = "user"          # excludes self-identified spiders and automated traffic
UA = "TH-GEE-research/1.0 (https://github.com/bnatc85/TH-GEE; bnatc85@gmail.com)"

# topic (exactly as keyed in youtube_raw.json) -> article, fidelity, alternate
TOPICS: dict[str, dict] = {
    # --- persistent -------------------------------------------------------
    "meditation music":        {"article": "Meditation music", "fidelity": "high"},
    "lofi hip hop":            {"article": "Lo-fi music", "fidelity": "high"},
    "yoga for beginners":      {"article": "Yoga as exercise", "fidelity": "medium",
                                "alt": "Yoga"},
    "minecraft tutorial":      {"article": "Minecraft", "fidelity": "medium"},
    "how to tie a tie":        {"article": "Necktie", "fidelity": "medium"},
    # --- acute ------------------------------------------------------------
    "harlem shake":            {"article": "Harlem Shake (meme)", "fidelity": "high"},
    "gangnam style":           {"article": "Gangnam Style", "fidelity": "high"},
    "ice bucket challenge":    {"article": "Ice Bucket Challenge", "fidelity": "high"},
    "fidget spinner":          {"article": "Fidget spinner", "fidelity": "high"},
    "mannequin challenge":     {"article": "Mannequin Challenge", "fidelity": "high"},
    # --- re-emergent ------------------------------------------------------
    "solar eclipse":           {"article": "Solar eclipse", "fidelity": "high"},
    "super bowl halftime":     {"article": "List of Super Bowl halftime shows",
                                "fidelity": "high", "alt": "Super Bowl halftime show"},
    "world cup final":         {"article": "List of FIFA World Cup finals",
                                "fidelity": "high", "alt": "FIFA World Cup final"},
    "summer olympics":         {"article": "Summer Olympic Games", "fidelity": "high"},
    "total lunar eclipse":     {"article": "Lunar eclipse", "fidelity": "high"},
    # --- recent -----------------------------------------------------------
    "chatgpt":                 {"article": "ChatGPT", "fidelity": "high"},
    "deepseek ai":             {"article": "DeepSeek", "fidelity": "high"},
    "threads app":             {"article": "Threads (social network)", "fidelity": "high"},
    # --- debunked ---------------------------------------------------------
    "flat earth":              {"article": "Flat Earth", "fidelity": "high"},
    "moon landing hoax":       {"article": "Moon landing conspiracy theories",
                                "fidelity": "high"},
    "5g coronavirus":          {"article": "5G misinformation", "fidelity": "high",
                                "alt": "COVID-19 misinformation"},
    "chemtrails":              {"article": "Chemtrail conspiracy theory", "fidelity": "high"},
    "pizzagate":               {"article": "Pizzagate conspiracy theory", "fidelity": "high"},
    "bigfoot sighting":        {"article": "Bigfoot", "fidelity": "high"},
    "loch ness monster":       {"article": "Loch Ness Monster", "fidelity": "high"},
    "area 51 aliens":          {"article": "Area 51", "fidelity": "medium"},
    "climate change hoax":     {"article": "Climate change denial", "fidelity": "high"},
    # redirect kept: target "False memory" is a broader cognitive-psychology article
    "mandela effect":          {"article": "Mandela effect", "fidelity": "high",
                                "alt": "False memory"},
    "birds aren't real":       {"article": "Birds Aren't Real", "fidelity": "high"},
    "hollow earth":            {"article": "Hollow Earth", "fidelity": "high"},
    "phantom time hypothesis": {"article": "Phantom time conspiracy theory",
                                "fidelity": "high"},
    # redirect kept: target "Serge Monast" is a biography of the claim's originator
    "project blue beam":       {"article": "Project Blue Beam", "fidelity": "high",
                                "alt": "Serge Monast"},
    "haarp weather control":   {"article": "High-frequency Active Auroral Research Program",
                                "fidelity": "medium"},
    "paul is dead conspiracy": {"article": "Paul is dead", "fidelity": "high"},
    "fluoride water poisoning": {"article": "Opposition to water fluoridation",
                                 "fidelity": "high", "alt": "Water fluoridation"},
    "mothman":                 {"article": "Mothman", "fidelity": "high"},
    "cern portals":            {"article": "CERN", "fidelity": "low",
                                "alt": "Large Hadron Collider"},
}

PATTERNS = {
    "persistent": ["meditation music", "lofi hip hop", "yoga for beginners",
                   "minecraft tutorial", "how to tie a tie"],
    "acute": ["harlem shake", "gangnam style", "ice bucket challenge",
              "fidget spinner", "mannequin challenge"],
    "reemergent": ["solar eclipse", "super bowl halftime", "world cup final",
                   "summer olympics", "total lunar eclipse"],
    "recent": ["chatgpt", "deepseek ai", "threads app"],
}
PATTERN_OF = {t: p for p, ts in PATTERNS.items() for t in ts}


RETRIES = 8        # attempts per range before it is bisected
MIN_SPAN = 20      # days; a range this short that still 404s is taken as no-data


def _fetch_chunk(article: str, start: date, end: date,
                 rng: random.Random, granularity: str = "daily") -> dict[str, int] | None:
    """Raw ``{YYYYMMDD: views}`` for one request, or None if the range keeps 404ing.

    The endpoint emits spurious 404s that the CDN then caches: a given URL 404s
    deterministically on every repeat, while both halves of the same range
    return data, and a *different* URL for the identical range succeeds. Retries
    therefore append a fresh throwaway query parameter, which changes the cache
    key and forces a new backend attempt; a plain retry would only re-read the
    cached failure. A 404 that survives all of that is left for the caller to
    bisect, and only a short range (see MIN_SPAN) is finally accepted as
    genuinely having no data -- which is a real state for articles created
    inside the horizon (ChatGPT, DeepSeek).
    """
    title = urllib.parse.quote(article.replace(" ", "_"), safe="")
    base = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"{PROJECT}/{ACCESS}/{AGENT}/{title}/{granularity}/"
            f"{start:%Y%m%d}/{end:%Y%m%d}")

    for attempt in range(RETRIES):
        url = base if attempt == 0 else f"{base}?cb={rng.randrange(1 << 30)}"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            payload = json.loads(urllib.request.urlopen(req, timeout=60).read())
            return {item["timestamp"][:8]: int(item["views"])
                    for item in payload["items"]}
        except urllib.error.HTTPError as exc:
            if exc.code not in (404, 429, 500, 502, 503, 504):
                raise
            if exc.code != 404 and attempt == RETRIES - 1:
                raise
            time.sleep(min(0.1 * 2 ** attempt, 1.0))
        except (urllib.error.URLError, TimeoutError):
            if attempt == RETRIES - 1:
                raise
            time.sleep(min(0.1 * 2 ** attempt, 1.0))
    return None


def _fetch_range(article: str, start: date, end: date, rng: random.Random,
                 by_day: dict[str, int], gaps: list[str]) -> None:
    """Fill ``by_day`` for [start, end], bisecting around ranges that keep 404ing.

    Bisecting blindly is expensive for the ranges that are *legitimately* empty
    -- an article created inside the horizon has years of genuine no-data, and
    halving all of it down to MIN_SPAN costs hundreds of requests. The monthly
    endpoint is used as an oracle first: it is a different URL (so unaffected by
    whatever poisoned the daily one) and answers the only question that matters
    before recursing, namely whether the range holds any data at all.
    """
    chunk = _fetch_chunk(article, start, end, rng)
    time.sleep(0.02)
    if chunk is not None:
        by_day.update(chunk)
        return

    monthly = _fetch_chunk(article, start, end, rng, granularity="monthly")
    if monthly is None or not any(monthly.values()):
        gaps.append(f"{start:%Y-%m-%d}..{end:%Y-%m-%d}")
        return

    if (end - start).days + 1 <= MIN_SPAN:
        gaps.append(f"{start:%Y-%m-%d}..{end:%Y-%m-%d}")
        return
    mid = start + timedelta(days=((end - start).days) // 2)
    _fetch_range(article, start, mid, rng, by_day, gaps)
    _fetch_range(article, mid + timedelta(days=1), end, rng, by_day, gaps)


def fetch_series(article: str, start: date, end: date) -> tuple[list[int], list[str]]:
    """Daily view counts for ``article``, zero-filled over [start, end].

    Requested one calendar year at a time, each year bisected as needed. Returns
    the series and the date ranges the API would not serve.
    """
    rng = random.Random(f"{article}|{start}|{end}")
    by_day: dict[str, int] = {}
    gaps: list[str] = []
    for year in range(start.year, end.year + 1):
        lo = max(start, date(year, 1, 1))
        hi = min(end, date(year, 12, 31))
        if lo <= hi:
            _fetch_range(article, lo, hi, rng, by_day, gaps)

    days = (end - start).days + 1
    series = [by_day.get(f"{start + timedelta(days=i):%Y%m%d}", 0) for i in range(days)]
    return series, gaps


def collect(include_alts: bool) -> None:
    raw = json.loads(RAW_PATH.read_text()) if RAW_PATH.exists() else {}
    meta = {"project": PROJECT, "access": ACCESS, "agent": AGENT,
            "start": START.isoformat(), "end": END.isoformat()}
    if raw.get("_meta") and raw["_meta"] != meta:
        raise SystemExit(f"cache was built for a different horizon: {raw['_meta']}")
    raw["_meta"] = meta

    wanted: list[tuple[str, str, str]] = []
    for topic, spec in TOPICS.items():
        wanted.append((topic, spec["article"], "primary"))
        if include_alts and "alt" in spec:
            wanted.append((topic, spec["alt"], "alt"))

    for topic, article, role in wanted:
        key = topic if role == "primary" else f"{topic}::alt"
        if key in raw:
            print(f"cached {key!r}")
            continue
        series, gaps = fetch_series(article, START, END)
        raw[key] = {
            "topic": topic,
            "article": article,
            "role": role,
            "pattern": PATTERN_OF.get(topic, "debunked"),
            "fidelity": TOPICS[topic]["fidelity"],
            "no_data_ranges": gaps,
            "views": series,
        }
        RAW_PATH.write_text(json.dumps(raw))
        nz = sum(1 for v in series if v > 0)
        note = f"  [no data: {', '.join(gaps)}]" if gaps else ""
        print(f"collected {key!r} <- {article!r}: {sum(series):,} views, "
              f"{nz}/{len(series)} nonzero days{note}")
        time.sleep(0.05)   # courtesy; the API imposes no quota

    bad = [k for k, v in raw.items()
           if k != "_meta" and v["role"] == "primary" and sum(v["views"]) == 0]
    if bad:
        print("WARNING: primary series with zero views (check the mapping):", bad)
    print("wrote", RAW_PATH)


def main() -> None:
    global RAW_PATH
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alts", action="store_true",
                    help="also fetch alternate articles for mapping-sensitivity checks")
    ap.add_argument("--out", type=Path, help=f"cache path (default {RAW_PATH.name})")
    args = ap.parse_args()
    if args.out:
        RAW_PATH = args.out
    collect(args.alts)


if __name__ == "__main__":
    main()
