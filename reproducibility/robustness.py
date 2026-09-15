"""Robustness check for the YouTube temporal profiles.

Two artifacts motivated this script, both traced to the same root cause: the
trace's time axis (and therefore how many windows are zero) is fixed by the
sample rather than by the topic.

  A. Deeper sampling adds recent low-view long-tail content, which fills
     previously empty quarters and inflates P_h -- so acute topics drift toward
     looking persistent as MAX_VIDEOS rises.
  B. Deeper sampling can also catch a single old, marginally-matching video,
     which stretches min/max backward and pads the trace with empty quarters --
     deleting one video out of 100 moves P_h by up to 33x.

Variants compared:

  v0_baseline    span = min..max of sample                  (what the paper runs)
  v1_robustspan  span = 2nd..98th percentile of dates       (fixes B)
  v2_decomposed  v1 + P_h split into coverage x intensity   (fixes A's conflation)

``coverage`` is the share of windows with any sampled video; ``intensity`` is the
shifted geometric mean over the ACTIVE windows only. v0/v1 P_h conflates the two,
and the conflation is what the zero-fill makes fragile.

The decisive test is depth stability: Spearman rank correlation between each
metric computed at MAX_VIDEOS=50 and at 100, over the topics common to both. A
metric describing the topic should rank topics the same way at either depth; one
describing the scrape depth should not.

Usage:
    python analysis/robustness.py
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# metrics.py sits in the same folder in this self-contained bundle.
from metrics import geometric_persistence, thgee, reemergence_index

W = {"publication": 1.0, "like": 0.3, "comment": 0.7}
WINDOW = timedelta(days=90)
GAMMA = 0.10
DORMANCY = 2
TRIM = 2.0  # percentile trimmed from each end of the date range in v1/v2


def _bounds(times, trim):
    if trim <= 0:
        return min(times), max(times)
    stamps = np.array([t.timestamp() for t in times])
    lo, hi = np.percentile(stamps, [trim, 100.0 - trim])
    tz = times[0].tzinfo
    return (datetime.fromtimestamp(lo, tz), datetime.fromtimestamp(hi, tz))


def build_trace(videos, trim):
    """Videos outside the trimmed range are clamped into the end windows, so no
    engagement is discarded -- only the time axis is made robust."""
    times = [datetime.fromisoformat(v["publishedAt"]) for v in videos]
    start, end = _bounds(times, trim)
    T = max(1, int(math.ceil((end - start) / WINDOW)))
    num = np.zeros(T)
    den = np.zeros(T)
    for v, t in zip(videos, times):
        idx = int((t - start) / WINDOW)
        idx = max(0, min(T - 1, idx))
        num[idx] += W["publication"] + W["like"] * v["like"] + W["comment"] * v["comment"]
        den[idx] += 1
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0), T


def profile(videos, trim):
    trace, T = build_trace(videos, trim)
    active = trace[trace > 0]
    qh = float(np.percentile(active, 25)) if active.size else 1.0
    R, reacts = reemergence_index(trace, max(qh, 1e-9), DORMANCY)
    Ph = geometric_persistence(trace)
    G = thgee(trace, GAMMA)
    return {
        "Ph": Ph,
        "thgee": G,
        "recency_ratio": (G / Ph if Ph > 1e-9 else 0.0),
        "R": R,
        "n_reactivations": len(reacts),
        "coverage": float(active.size / T),
        "intensity": geometric_persistence(active) if active.size else 0.0,
        "windows": T,
    }


VARIANTS = {"v0_baseline": 0.0, "v1_robustspan": TRIM, "v2_decomposed": TRIM}


def spearman(x, y):
    def rank(v):
        order = np.argsort(np.argsort(np.asarray(v, float)))
        return order.astype(float)
    a, b = rank(x), rank(y)
    a -= a.mean()
    b -= b.mean()
    denom = math.sqrt(float((a ** 2).sum()) * float((b ** 2).sum()))
    return float((a * b).sum() / denom) if denom else float("nan")


def main():
    raws = {
        50: json.loads((HERE / "youtube_raw.json").read_text()),
        100: json.loads((HERE / "youtube_raw_100.json").read_text()),
    }
    topics = [t for t in raws[50] if t in raws[100]]

    out = {}
    for name, trim in VARIANTS.items():
        out[name] = {d: {t: profile(raws[d][t]["videos"], trim) for t in topics}
                     for d in raws}

    patterns = {t: raws[50][t]["pattern"] for t in topics}
    keys = {"v0_baseline": ["Ph", "coverage", "recency_ratio", "R"],
            "v1_robustspan": ["Ph", "coverage", "recency_ratio", "R"],
            "v2_decomposed": ["coverage", "intensity", "R"]}

    print("=" * 78)
    print("DEPTH STABILITY -- Spearman rank corr. between 50-video and 100-video runs")
    print("(1.00 = deeper scraping does not change how topics rank)")
    print("=" * 78)
    for name in VARIANTS:
        for k in keys[name]:
            r = spearman([out[name][50][t][k] for t in topics],
                         [out[name][100][t][k] for t in topics])
            print(f"  {name:16} {k:10} rho = {r:5.2f}")

    print()
    print("=" * 78)
    print("SINGLE-VIDEO SENSITIVITY -- P_h after deleting each topic's oldest video")
    print("(max fold-change over all topics; 1.0 = no topic is hostage to one video)")
    print("=" * 78)
    for name, trim in VARIANTS.items():
        worst = (1.0, None)
        for t in topics:
            vs = sorted(raws[100][t]["videos"], key=lambda v: v["publishedAt"])
            p0 = profile(vs, trim)["Ph"]
            p1 = profile(vs[1:], trim)["Ph"]
            if p0 > 0:
                fold = max(p1 / p0, p0 / p1)
                if fold > worst[0]:
                    worst = (fold, t)
        print(f"  {name:16} worst = {worst[0]:5.1f}x  ({worst[1]})")

    print()
    print("=" * 78)
    print("GROUP MEDIANS")
    print("=" * 78)
    order = ["persistent", "acute", "reemergent", "recent", "debunked"]
    for name in VARIANTS:
        print(f"\n-- {name} --")
        hdr = keys[name]
        print(f"  {'pattern':12}{'n':>3}" + "".join(
            f"{k + ' @50':>14}{'@100':>11}" for k in hdr))
        for pat in order:
            ts = [t for t in topics if patterns[t] == pat]
            row = f"  {pat:12}{len(ts):3d}"
            for k in hdr:
                m50 = float(np.median([out[name][50][t][k] for t in ts]))
                m100 = float(np.median([out[name][100][t][k] for t in ts]))
                row += f"{m50:14.3f}{m100:11.3f}"
            print(row)

    sep = {}
    print()
    print("=" * 78)
    print("SEPARATION persistent vs acute (ratio of group medians; >1 = expected direction)")
    print("=" * 78)
    for name in VARIANTS:
        for k in keys[name]:
            vals = []
            for d in (50, 100):
                p = float(np.median([out[name][d][t][k] for t in topics if patterns[t] == "persistent"]))
                a = float(np.median([out[name][d][t][k] for t in topics if patterns[t] == "acute"]))
                vals.append(p / a if a else float("nan"))
            sep[f"{name}.{k}"] = vals
            print(f"  {name:16} {k:10} @50 = {vals[0]:7.2f}x   @100 = {vals[1]:7.2f}x")

    (HERE / "robustness_results.json").write_text(json.dumps(
        {"variants": {n: {str(d): v for d, v in out[n].items()} for n in out},
         "separation_persistent_over_acute": sep}, indent=2))
    print("\nwrote", HERE / "robustness_results.json")


if __name__ == "__main__":
    main()
