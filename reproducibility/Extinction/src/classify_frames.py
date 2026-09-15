"""Assign media frames and sentiment to the collected news corpus.

Implements paper Sec. 5.1. The corpus is classified into the paper's seven
nonexclusive frames, with the governance frame measured as two components
(governance, control) so that H3's directional expectation about *control*
framing is testable; their union reproduces the paper's frame 5.

Classifier. A transparent keyword-dictionary classifier over the headline.
Frames are nonexclusive, so each is an independent binary decision. The paper
requires a human-coded validation sample and a macro-F1 threshold before a frame
is used; ``--export-coding-sample`` writes the stratified sample for that step,
and until human labels exist the reported validation fields stay null rather
than being filled with self-assessment.

Sentiment uses VADER as a transparent baseline. Because negative polarity is
near-automatic in extinction coverage, a separate threat-intensity dictionary is
scored alongside it, and the substantive analysis leans on frame prevalence.

Scope limit. Only headlines are available from the RSS source; the paper
anticipates body text "where licensing permits". Headline-only classification
raises precision and lowers recall relative to full text, which is recorded in
the manifest.

Usage:
    python src/classify_frames.py
    python src/classify_frames.py --export-coding-sample 400
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DERIVED, RAW, load_events, save_json, write_manifest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------- dictionaries
# Each frame is a list of regex alternatives matched case-insensitively against
# the headline. Patterns are word-bounded to avoid substring false positives
# (e.g. "AI" inside "said").
FRAMES: dict[str, list[str]] = {
    # 1. existential harm or loss of control
    "existential": [
        r"extinction", r"existential", r"end of humanity", r"wipe out human",
        r"kill (?:us all|all humans|humanity)", r"human survival", r"doom",
        r"loss of control", r"out of control", r"rogue ai", r"takeover",
        r"annihilat", r"catastroph", r"apocalyp", r"threat to humanity",
    ],
    # 2. cyber operations, critical infrastructure, cascading failure
    "infrastructure": [
        r"critical infrastructure", r"power grid", r"\bgrid\b", r"pipeline",
        r"water system", r"cyber ?attack", r"cyber ?security", r"hacking",
        r"ransomware", r"blackout", r"cascading failure", r"supply chain",
        r"telecom", r"data ?cent(?:er|re)", r"utilit(?:y|ies)",
    ],
    # 3. biological or chemical misuse
    "bio": [
        r"biological weapon", r"bioweapon", r"biosecurity", r"pathogen",
        r"chemical weapon", r"pandemic", r"virus design", r"bioterror",
    ],
    # 4. economic or labor disruption
    "economic": [
        r"\bjobs?\b", r"unemploy", r"labou?r market", r"workforce", r"layoff",
        r"wages?", r"economic disruption", r"automat(?:e|ion|ing)",
        r"displace(?:d|ment)? workers?", r"\bbubble\b", r"market(?:s)? (?:rout|selloff|plunge)",
    ],
    # 5a. governance, regulation, slowdown
    "governance": [
        r"regulat", r"legislat", r"\bbill\b", r"\blaw\b", r"congress",
        r"lawmaker", r"senat", r"polic(?:y|ymaker)", r"governance", r"oversight",
        r"moratorium", r"\bpause\b", r"slow ?down", r"ban\b", r"treaty",
        r"white house", r"eu ai act", r"safety institute", r"accountab",
    ],
    # 5b. emergency shutdown / kill-switch control mechanisms
    "control": [
        r"kill ?switch", r"emergency shutdown", r"shut(?: it)? down",
        r"off switch", r"fail-?safe", r"containment", r"interruptib",
        r"circuit breaker", r"capability control", r"guardrail",
    ],
    # 6. technical capability, superintelligence, self-improvement
    "capability": [
        r"superintelligen", r"\bagi\b", r"artificial general intelligence",
        r"self-?improv", r"recursive", r"frontier model", r"scaling",
        r"breakthrough", r"more powerful", r"capabilit", r"autonomous agent",
        r"reasoning model",
    ],
    # 7. skepticism, hype, dismissal, geopolitical competition
    "skeptic": [
        r"hype", r"overblown", r"skeptic", r"sceptic", r"dismiss", r"nonsense",
        r"fear ?mongering", r"doomer", r"hysteri", r"distraction", r"overstat",
        r"china", r"arms race", r"compet(?:e|ition) with", r"race with",
        r"rejects? warning", r"pushes? back",
    ],
}

# Threat-intensity dictionary, scored separately from polarity (paper Sec. 5.1).
THREAT_TERMS = [
    r"extinction", r"annihilat", r"catastroph", r"apocalyp", r"doom",
    r"existential", r"kill", r"death", r"die\b", r"destroy", r"wipe out",
    r"threat", r"danger", r"disaster", r"collapse", r"crisis", r"emergency",
]

COMPILED = {f: [re.compile(p, re.I) for p in pats] for f, pats in FRAMES.items()}
THREAT_RE = [re.compile(p, re.I) for p in THREAT_TERMS]


def classify(text: str) -> dict[str, int]:
    return {f: int(any(r.search(text) for r in regexes))
            for f, regexes in COMPILED.items()}


def threat_intensity(text: str) -> int:
    """Count of distinct threat-dictionary terms present."""
    return sum(1 for r in THREAT_RE if r.search(text))


def article_date(a: dict) -> str | None:
    """True publication date from the item's own timestamp.

    The collector queries one day at a time, but Google News honours
    ``after:``/``before:`` only coarsely: consecutive day-queries return largely
    the same ranked set, and the items they return carry publication dates
    spread over surrounding days. The query day is therefore not the article's
    date. Every article is re-dated here from its own ``pubDate``, and anything
    falling outside the event window is dropped.
    """
    try:
        return parsedate_to_datetime(a["published"]).date().isoformat()
    except (TypeError, ValueError, KeyError):
        return None


def build(event_id: str, analyzer: SentimentIntensityAnalyzer) -> dict | None:
    path = RAW / f"news_{event_id}.json"
    if not path.exists():
        print(f"  no news file for {event_id}; skipping")
        return None
    payload = json.loads(path.read_text())
    lo, hi = payload["window"]

    kept, dropped_undated, dropped_outside = [], 0, 0
    for a in payload["articles"]:
        d = article_date(a)
        if d is None:
            dropped_undated += 1
            continue
        if not (lo <= d <= hi):
            dropped_outside += 1
            continue
        kept.append({**a, "query_day": a["date_only"], "date_only": d})

    rows = []
    for a in kept:
        text = a["title"]
        frames = classify(text)
        vs = analyzer.polarity_scores(text)
        rows.append({
            "date": a["date_only"], "query_day": a["query_day"],
            "outlet": a["outlet"], "title": text,
            "url": a.get("canonical_url", a.get("url", "")),
            "queries": "|".join(a.get("queries", [a.get("query", "")])),
            "in_base_corpus": int("base" in a.get("queries",
                                                  [a.get("query", "")])),
            "syndicated": a["syndicated"],
            "vader_compound": vs["compound"],
            "threat_intensity": threat_intensity(text),
            **frames,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return None

    # Frame prevalence M_ft uses unique stories only; syndicated copies are kept
    # for reach (paper Sec. 5.1).
    #
    # Denominator. The paper defines the corpus by its narrow conceptual query.
    # That query cannot supply a daily series from this source: it matches a
    # small candidate pool, so its ranked top-100 is nearly identical from one
    # day-query to the next, and after URL deduplication it yields a median of
    # zero stories per day. The five-query union does vary day to day and its
    # publication dates spread properly across the window, so it is the
    # denominator here. The query design is identical on every day, which is
    # what makes day-to-day movement in a frame's *share* interpretable; the
    # absolute level is a property of the query set, not of the press, and the
    # daily count is a ranked sample rather than a census of coverage.
    uniq = df[~df["syndicated"]].copy()
    frame_cols = list(FRAMES)
    base = uniq[uniq["in_base_corpus"] == 1]

    daily = uniq.groupby("date").agg(
        n_stories=("title", "size"),
        vader_mean=("vader_compound", "mean"),
        threat_intensity_mean=("threat_intensity", "mean"),
        **{f: (f, "mean") for f in frame_cols},
    ).reset_index()

    # Kept as a diagnostic: how much of each day came from the paper's narrow
    # conceptual query rather than the mechanism-specific supplements.
    if not base.empty:
        b_daily = base.groupby("date").size().rename("n_base_corpus").reset_index()
        daily = daily.merge(b_daily, on="date", how="outer")
        daily["n_base_corpus"] = daily["n_base_corpus"].fillna(0)
    reach = df.groupby("date").size().rename("n_with_syndication").reset_index()
    daily = daily.merge(reach, on="date", how="outer").sort_values("date")
    daily["n_stories"] = daily["n_stories"].fillna(0)

    save_json(DERIVED / f"news_daily_{event_id}.json",
              daily.to_dict(orient="records"))
    df.to_csv(DERIVED / f"news_classified_{event_id}.csv", index=False)

    return {
        "event_id": event_id,
        "n_records": int(len(df)),
        "n_unique_stories": int(len(uniq)),
        "n_days": int(daily.shape[0]),
        "dropped_undated": dropped_undated,
        "dropped_outside_window": dropped_outside,
        "corpus_used_for_daily_series": "five-query union (see build docstring)",
        "n_base_corpus_stories": int(len(base)),
        "frame_prevalence_overall": {f: float(uniq[f].mean()) for f in frame_cols},
        "n_outlets": int(uniq["outlet"].nunique()),
        "validation": {
            "note": ("Requires the human-coded sample: run "
                     "--export-coding-sample, have two coders label it, then "
                     "compute Krippendorff's alpha and macro-F1 against these "
                     "automated labels (paper Sec. 5.1)."),
            "krippendorff_alpha": None,
            "macro_f1": None,
            "coded_sample_size": None,
        },
    }


def export_coding_sample(n: int, seed: int = 20260909) -> Path:
    """Stratified sample for double human coding (paper Sec. 5.1)."""
    frames = list(FRAMES)
    parts = []
    for event_id in ("statement2023", "warning2026"):
        p = DERIVED / f"news_classified_{event_id}.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["event_id"] = event_id
            parts.append(d)
    if not parts:
        raise SystemExit("no classified corpus; run without --export-coding-sample first")
    df = pd.concat(parts, ignore_index=True)
    df = df[~df["syndicated"]]

    rng = np.random.default_rng(seed)
    # Stratify by event and by predicted frame so rare frames are represented.
    per_stratum = max(1, n // (len(frames) * len(parts) + len(parts)))
    picks = []
    for event_id in df["event_id"].unique():
        sub = df[df["event_id"] == event_id]
        for f in frames:
            pool = sub[sub[f] == 1]
            if len(pool):
                take = min(per_stratum, len(pool))
                picks.append(pool.iloc[rng.choice(len(pool), take, replace=False)])
        none_pool = sub[sub[frames].sum(axis=1) == 0]
        if len(none_pool):
            take = min(per_stratum, len(none_pool))
            picks.append(none_pool.iloc[rng.choice(len(none_pool), take, replace=False)])

    sample = pd.concat(picks).drop_duplicates(subset=["url", "title"])
    if len(sample) > n:
        sample = sample.iloc[rng.choice(len(sample), n, replace=False)]
    # Blank columns for the two coders; automated labels are kept in *_auto.
    out = sample[["event_id", "date", "outlet", "title", "url"]].copy()
    for f in frames:
        out[f"{f}_auto"] = sample[f].values
        out[f"{f}_coder1"] = ""
        out[f"{f}_coder2"] = ""
    path = DERIVED / "frame_coding_sample.csv"
    out.to_csv(path, index=False)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-coding-sample", type=int, metavar="N")
    args = ap.parse_args()

    analyzer = SentimentIntensityAnalyzer()
    summary = {}
    for event in load_events()["events"]:
        res = build(event["id"], analyzer)
        if res:
            summary[event["id"]] = res
            print(f"[{event['id']}] {res['n_unique_stories']} unique stories, "
                  f"{res['n_days']} days, {res['n_outlets']} outlets")
            print("   frame prevalence: " + ", ".join(
                f"{k}={v:.2f}" for k, v in res["frame_prevalence_overall"].items()))

    if args.export_coding_sample:
        path = export_coding_sample(args.export_coding_sample)
        print(f"wrote coding sample -> {path}")

    write_manifest("classify_frames", classifier="keyword-dictionary (headline only)",
                   frames=list(FRAMES), events=summary)


if __name__ == "__main__":
    main()
