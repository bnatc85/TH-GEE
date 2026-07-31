"""TH-GEE empirical analysis over curated YouTube topics (paper Results).

Two stages:

1. ``--collect`` queries the YouTube Data API v3 for a curated set of topics and
   caches, per topic, each matching video's publication time and final like and
   comment counts to ``youtube_raw.json``. This needs YOUTUBE_API_KEY. The raw
   cache contains only public counts and timestamps (no video content).

2. Analysis (default) reads the cache, builds a video-publication-with-engagement
   trace per topic, and computes the temporal profile. It also runs the
   deterministic synthetic construct tests (RQ1-RQ4). Results are written to
   ``youtube_results.json``. Analysis needs no API key, so the paper numbers
   reproduce from the committed cache.

Model: each video is a timestamped "post" (cf. RedNote-Vibe). At its publication
time we accumulate weighted engagement E_tau = sum(w_pub + w_like*likes +
w_comment*comments) / n_videos over 90-day (quarterly) windows, then compute
P_h, TH-GEE, and R_h. Anchoring engagement to publication time makes the trace
reflect a topic's whole history rather than clustering at the scrape date.

Usage:
    python analysis/youtube_analysis.py --collect   # refresh cache from API
    python analysis/youtube_analysis.py             # analyze cached data
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import geometric_persistence, thgee, reemergence_index  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW_PATH = HERE / "youtube_raw.json"
OUT_PATH = HERE / "youtube_results.json"

W = {"publication": 1.0, "like": 0.3, "comment": 0.7}
WINDOW = timedelta(days=90)  # quarterly: coarse enough for multi-year topics
GAMMA = 0.10
DORMANCY = 2
MAX_VIDEOS = 50  # primary specification; --max-videos overrides (see robustness.py)

# Curated topics grouped by the temporal pattern each is expected to display.
# Featured exemplars per pattern are the first in each list.
TOPICS = {
    "persistent": ["meditation music", "lofi hip hop", "yoga for beginners",
                   "minecraft tutorial", "how to tie a tie"],
    "acute": ["harlem shake", "gangnam style", "ice bucket challenge",
              "fidget spinner", "mannequin challenge"],
    "reemergent": ["solar eclipse", "super bowl halftime", "world cup final",
                   "summer olympics", "total lunar eclipse"],
    "recent": ["chatgpt", "deepseek ai", "threads app"],
    # Well-known, authoritatively debunked claims / conspiracy theories. Included
    # to show the framework applied to the influence-relevant domain. A search
    # returns a MIX of believer, debunking, and news content, so the trace
    # measures attention to the topic, not the false claim, and pattern says
    # nothing about veracity or intent.
    "debunked": ["flat earth", "moon landing hoax", "5g coronavirus",
                 "chemtrails", "pizzagate", "bigfoot sighting",
                 "loch ness monster", "area 51 aliens",
                 # Expanded set (paper's full debunked-claim list). "climate
                 # change hoax" queries the denialist narrative (climate change
                 # itself is not debunked); "paul is dead conspiracy" is the
                 # McCartney-lookalike claim under its named form.
                 "climate change hoax", "mandela effect", "birds aren't real",
                 "hollow earth", "phantom time hypothesis", "project blue beam",
                 "haarp weather control", "paul is dead conspiracy",
                 "fluoride water poisoning", "mothman", "cern portals"],
}


# --------------------------------------------------------------------- collect
def collect():
    """Incremental and resumable.

    A topic is refetched unless the cache already holds a full MAX_VIDEOS-sized
    sample for it, so raising MAX_VIDEOS re-collects the under-filled topics.
    The cache is flushed after every topic: search.list costs 100 units per
    50-result page, so the full topic set costs 200*len(topics) units against a
    default 10k units/day. A mid-run cutoff must not discard topics already paid
    for; rerun to resume.

    Bursting pages back-to-back also trips a short-window 'rateLimitExceeded'
    (distinct from the daily cap), so 429s are retried with backoff.
    """
    from collector import build_client, search_videos, fetch_video_details
    yt = build_client(os.environ.get("YOUTUBE_API_KEY"))
    raw = json.loads(RAW_PATH.read_text()) if RAW_PATH.exists() else {}

    def fetch(topic):
        for attempt in range(5):
            try:
                return fetch_video_details(yt, search_videos(yt, topic, MAX_VIDEOS))
            except Exception as exc:
                if "429" not in str(exc) or attempt == 4:
                    raise
                wait = 5 * 2 ** attempt
                print(f"  rate limited on {topic!r}, retrying in {wait}s")
                time.sleep(wait)

    for pattern, topics in TOPICS.items():
        for topic in topics:
            have = len(raw[topic]["videos"]) if topic in raw else 0
            if have >= MAX_VIDEOS:
                print(f"cached {topic!r}: {have} videos")
                continue
            try:
                videos = fetch(topic)
            except Exception as exc:  # quota exhaustion, transport errors
                print(f"STOPPED on {topic!r}: {exc}")
                print(f"cache holds {len(raw)} topics; rerun to resume")
                break
            raw[topic] = {
                "pattern": pattern,
                "videos": [
                    {"publishedAt": v["publishedAt"].isoformat(),
                     "like": v["likeCount"], "comment": v["commentCount"]}
                    for v in videos
                ],
            }
            RAW_PATH.write_text(json.dumps(raw, indent=2))
            print(f"collected {topic!r}: {len(videos)} videos (was {have})")
        else:
            continue
        break
    print("wrote", RAW_PATH)


# -------------------------------------------------------------------- analysis
def build_trace(videos):
    times = [datetime.fromisoformat(v["publishedAt"]) for v in videos]
    start, end = min(times), max(times)
    T = max(1, int(math.ceil((end - start) / WINDOW)))
    num = np.zeros(T); den = np.zeros(T)
    for v, t in zip(videos, times):
        idx = min(T - 1, int((t - start) / WINDOW))
        num[idx] += W["publication"] + W["like"] * v["like"] + W["comment"] * v["comment"]
        den[idx] += 1
    trace = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return trace, start, end


def profile(trace):
    active = trace[trace > 0]
    qh = float(np.percentile(active, 25)) if active.size else 1.0
    Ph = geometric_persistence(trace)
    G = thgee(trace, GAMMA)
    R, reacts = reemergence_index(trace, max(qh, 1e-9), DORMANCY)
    return {
        "Ph": Ph, "thgee": G, "recency_ratio": (G / Ph if Ph > 1e-9 else None),
        "R": R, "n_reactivations": len(reacts),
        "windows": int(trace.size), "active_windows": int((trace > 0).sum()),
    }


def synthetic_tests():
    out = {}
    persistent = np.array([10, 11, 9, 10, 12, 11], float)
    acute = np.zeros(6); acute[0] = persistent.sum()
    out["rq1"] = {"total_volume": float(persistent.sum()),
                  "persistent_Ph": geometric_persistence(persistent),
                  "acute_Ph": geometric_persistence(acute)}
    rng = np.random.default_rng(20260716)
    base = np.array([20, 10, 5, 0, 3, 8], float)
    perms, Ph_vals, thg = set(), [], []
    for _ in range(5000):
        p = tuple(rng.permutation(base))
        if p in perms:
            continue
        perms.add(p); a = np.array(p)
        Ph_vals.append(geometric_persistence(a)); thg.append(thgee(a, 0.5))
    out["rq2"] = {"n_unique_perms": len(perms),
                  "Ph_std": float(np.std(Ph_vals)), "thgee_std": float(np.std(thg)),
                  "thgee_range": [float(min(thg)), float(max(thg))],
                  "reduction_identity_max_abs_diff": float(max(
                      abs(geometric_persistence(np.array(p)) - thgee(np.array(p), 0.0))
                      for p in list(perms)[:500]))}
    out["rq3"] = {k: reemergence_index(np.array(v, float), 5.0, 3)[0] for k, v in {
        "continuous_R": [10, 11, 9, 10, 12, 11], "reemergent_R": [12, 10, 0, 0, 0, 14],
        "formerly_active_R": [12, 10, 8, 0, 0, 0], "late_first_R": [0, 0, 0, 0, 0, 14]}.items()}
    reem = np.array([12, 10, 0, 14, 0, 0, 0, 15], float)
    out["rq4_d"] = {dd: reemergence_index(reem, 5.0, dd)[0] for dd in [1, 2, 3, 4]}
    early = np.array([20, 12, 8, 5, 2, 1], float); late = early[::-1]
    grid = [0.0, 0.1, 0.5, 1.0, 2.0]
    out["rq4_gamma"] = {"early": {g: thgee(early, g) for g in grid},
                        "late": {g: thgee(late, g) for g in grid}}
    return out


def analyze():
    raw = json.loads(RAW_PATH.read_text())
    results = {"weights": W,
               "params": {"window_days": WINDOW.days, "gamma": GAMMA,
                          "dormancy": DORMANCY, "max_videos": MAX_VIDEOS,
                          "qh_rule": "per-topic 25th percentile of nonzero window scores"},
               "synthetic": synthetic_tests(), "topics": {}}
    for topic, data in raw.items():
        videos = data["videos"]
        if not videos:
            continue
        trace, start, end = build_trace(videos)
        prof = profile(trace)
        prof.update({"pattern": data["pattern"], "n_videos": len(videos),
                     "span_years": round((end - start).days / 365.25, 1),
                     "span": [start.date().isoformat(), end.date().isoformat()]})
        results["topics"][topic] = prof
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"{'topic':22} {'pattern':11} {'yrs':>4} {'P_h':>8} {'recency':>8} {'R_h':>6} {'nz/T':>7}")
    for topic, p in results["topics"].items():
        rr = p["recency_ratio"]
        print(f"{topic:22} {p['pattern']:11} {p['span_years']:4.1f} {p['Ph']:8.1f} "
              f"{(rr if rr else 0):8.2f} {p['R']:6.3f} "
              f"{p['active_windows']:3d}/{p['windows']:<3d}")
    print("wrote", OUT_PATH)


def main():
    global RAW_PATH, OUT_PATH, MAX_VIDEOS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect", action="store_true", help="refresh cache from the YouTube API")
    ap.add_argument("--raw", type=Path, help="raw cache path (default youtube_raw.json)")
    ap.add_argument("--out", type=Path, help="results path (default youtube_results.json)")
    ap.add_argument("--max-videos", type=int, default=MAX_VIDEOS,
                    help=f"videos retained per query (default {MAX_VIDEOS}, the "
                         "paper's primary specification)")
    args = ap.parse_args()
    MAX_VIDEOS = args.max_videos
    if args.raw:
        RAW_PATH = args.raw
    if args.out:
        OUT_PATH = args.out
    if args.collect:
        collect()
    analyze()


if __name__ == "__main__":
    main()
