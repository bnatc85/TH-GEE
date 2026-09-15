"""Event-level replication of the TH-EE profiles using Wikipedia pageviews.

The YouTube application anchors engagement to publication time, so a quarterly
value is the eventual engagement of a *publication cohort* rather than audience
activity during that quarter (paper Sections 4.2, 6.3). Wikipedia pageviews are
audience activity dated to the day it occurred, so recomputing the profiles on
pageview series for the same 37 topics tests whether the publication-cohort
profiles survive the move to event-level data.

Trace construction. Equation 1 with a single interaction type (a pageview,
w = 1) and t_tau set to the number of days in window tau, so

    E_tau = (pageviews in window tau) / (days in window tau)

is mean daily pageviews: a rate, independent of window length, on the same
grid for every topic. Windows are 90 days (the paper's primary specification),
anchored at 2015-07-01, the first day the pageviews API serves.

Comparability. Two indicators used in the paper do not transfer unchanged:

  * Active-window coverage is degenerate. A YouTube window is zero when nobody
    published; a Wikipedia window is essentially never zero, because a live
    article always draws some traffic. Coverage is therefore ~100% for almost
    every topic, and the q_h-based active fraction is ~75% by construction
    (q_h is the 25th percentile of nonzero windows). Both are reported, and
    both are uninformative on this data -- which is itself the finding.

  * The scale-free persistence indicator that does transfer is the ratio of the
    Geometric Persistence Score to the arithmetic mean of the same trace,

        G/A = P_h / mean(E_tau)  in (0, 1],

    which is 1 exactly when the trace is flat and falls toward 0 as engagement
    concentrates into few windows. It is built only from quantities already in
    the paper (P_h and the arithmetic-mean baseline of Section 4.5), is
    invariant to rescaling the trace, and reproduces the RQ1 separation
    directly: for the construct traces of Table 3 it is 0.10 for the acute
    burst and 1.00 for the continuous trace. It is computed for both platforms.

Horizon matching. Wikipedia series start 2015-07-01; YouTube spans start at each
topic's oldest sampled video. The primary comparison therefore recomputes the
YouTube traces on the *same* window grid restricted to 2015-07-01..2026-06-30,
so both platforms have identical T and identical window boundaries. Full-span
YouTube profiles are reported alongside for reference.

Usage:
    python analysis/wikipedia_analysis.py
    python analysis/wikipedia_analysis.py --window-days 30
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from metrics import geometric_persistence, thgee, reemergence_index  # noqa: E402

WIKI_RAW = HERE / "wikipedia_raw.json"
YT_RAW = HERE / "youtube_raw.json"
OUT_PATH = HERE / "wikipedia_results.json"

# Paper's primary specification, reused verbatim.
GAMMA = 0.10
DORMANCY = 2
W_YT = {"publication": 1.0, "like": 0.3, "comment": 0.7}

PATTERN_ORDER = ["persistent", "acute", "reemergent", "recent", "debunked"]


# ------------------------------------------------------------------ profiling
def activity_threshold(trace: np.ndarray, rule: str = "quartile",
                       k: float = 2.0) -> float:
    """The activity threshold q_h separating active windows from dormant ones.

    ``quartile`` is the paper's rule: the 25th percentile of the topic's nonzero
    window scores. It suits sparse publication data, where a dormant window is
    literally empty and the quartile just trims the thin tail.

    ``elevation`` sets q_h to ``k`` times the topic's median window score, so
    "active" means *elevated above the topic's own baseline*. On dense attention
    data the quartile rule has a specific consequence worth testing rather than
    assuming: a live Wikipedia article always draws traffic, so an evergreen
    topic never falls below its own 25th percentile for long and never registers
    as dormant, whatever happens above the baseline. Under the framework's own
    definitions that is the correct answer -- such a topic is continuously
    active, and its spikes are bursts rather than returns -- but it makes R_h
    structurally unable to respond to spike structure on this kind of series, so
    the alternative rule is reported alongside.
    """
    if rule == "elevation":
        med = float(np.median(trace))
        return max(med * k, 1e-9)
    active = trace[trace > 0]
    return float(np.percentile(active, 25)) if active.size else 1.0


def profile(trace: np.ndarray, qh_rule: str = "quartile",
            qh_k: float = 2.0) -> dict:
    """Paper profile plus the two scale-free indicators used for comparison."""
    active = trace[trace > 0]
    qh = activity_threshold(trace, qh_rule, qh_k)
    Ph = geometric_persistence(trace)
    G = thgee(trace, GAMMA)
    R, reacts = reemergence_index(trace, max(qh, 1e-9), DORMANCY)
    mean = float(trace.mean())
    return {
        "Ph": Ph,
        "thgee": G,
        "recency_ratio": (G / Ph if Ph > 1e-9 else 0.0),
        "R": R,
        "n_reactivations": len(reacts),
        "reactivation_windows": list(reacts),
        "windows": int(trace.size),
        "active_windows": int((trace > 0).sum()),
        "coverage": float((trace > 0).sum() / trace.size),
        "coverage_qh": float((trace >= qh).sum() / trace.size),
        "qh": qh,
        "ga_ratio": (Ph / mean if mean > 1e-12 else 0.0),
        "arith_mean": mean,
        "median": float(np.median(trace)),
    }


def wiki_trace(views: list[int], window_days: int,
               span: str = "common") -> np.ndarray:
    """Mean daily pageviews per window; trailing partial window is dropped.

    ``span="common"`` keeps the whole 2015-07-01 horizon, so every topic sits on
    an identical axis. ``span="own"`` drops the leading all-zero windows first,
    which matters for articles created inside the horizon (ChatGPT, DeepSeek,
    Birds Aren't Real): on the common axis their years of pre-creation zeros
    depress every distributional measure, conflating "did not exist" with
    "concentrated in few windows". This is the same zero-padding artifact
    ``robustness.py`` identifies for the YouTube traces, and the own-span
    variant is the counterpart control.
    """
    v = np.asarray(views, dtype=float)
    T = v.size // window_days
    trace = v[: T * window_days].reshape(T, window_days).mean(axis=1)
    if span == "own":
        nz = np.flatnonzero(trace > 0)
        if nz.size:
            trace = trace[nz[0]:]
    return trace


def yt_trace(videos: list[dict], window_days: int, start: datetime,
             end: datetime, span: str = "common") -> np.ndarray | None:
    """YouTube publication-cohort trace on a fixed [start, end) window grid.

    Same construction as youtube_analysis.build_trace, except the time axis is
    supplied rather than taken from the sample's min/max, so the trace lands on
    the identical grid as the Wikipedia series. ``span`` behaves as in
    ``wiki_trace``.
    """
    window = timedelta(days=window_days)
    T = int((end - start) / window)
    num = np.zeros(T)
    den = np.zeros(T)
    kept = 0
    for v in videos:
        t = datetime.fromisoformat(v["publishedAt"])
        if not (start <= t < start + T * window):
            continue
        idx = int((t - start) / window)
        num[idx] += (W_YT["publication"] + W_YT["like"] * v["like"]
                     + W_YT["comment"] * v["comment"])
        den[idx] += 1
        kept += 1
    if kept == 0:
        return None
    trace = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    if span == "own":
        nz = np.flatnonzero(trace > 0)
        if nz.size:
            trace = trace[nz[0]:]
    return trace


def yt_trace_fullspan(videos: list[dict], window_days: int) -> np.ndarray:
    """The paper's own construction: axis spans the sample's own min..max."""
    window = timedelta(days=window_days)
    times = [datetime.fromisoformat(v["publishedAt"]) for v in videos]
    start, end = min(times), max(times)
    T = max(1, int(math.ceil((end - start) / window)))
    num = np.zeros(T)
    den = np.zeros(T)
    for v, t in zip(videos, times):
        idx = min(T - 1, int((t - start) / window))
        num[idx] += (W_YT["publication"] + W_YT["like"] * v["like"]
                     + W_YT["comment"] * v["comment"])
        den[idx] += 1
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


# ----------------------------------------------------------------- statistics
def _rank(values) -> np.ndarray:
    """Average ranks, so ties do not distort Spearman."""
    a = np.asarray(values, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(a.size, dtype=float)
    # average tied ranks
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(np.arange(i, j + 1, dtype=float))
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    a, b = _rank(x), _rank(y)
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a ** 2).sum()) * float((b ** 2).sum()))
    return float((a * b).sum() / denom) if denom else float("nan")


def perm_p(x, y, n_perm: int = 20000, seed: int = 20260731) -> float:
    """Two-sided permutation p-value for Spearman rho (n = 37 is small).

    Spearman on the raw values is Pearson on their ranks, and permuting y
    permutes its ranks, so the whole null distribution is one matrix product
    against pre-centred ranks rather than n_perm re-rankings.
    """
    a = _rank(x)
    b = _rank(y)
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a ** 2).sum()) * float((b ** 2).sum()))
    if denom == 0:
        return float("nan")
    obs = abs(float((a * b).sum()) / denom)

    rng = np.random.default_rng(seed)
    perms = np.tile(b, (n_perm, 1))
    rng.permuted(perms, axis=1, out=perms)
    null = np.abs(perms @ a) / denom
    return float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1))


# ------------------------------------------------------------------- analysis
def run(window_days: int, span: str = "common",
        qh_rule: str = "quartile", qh_k: float = 2.0) -> dict:
    wiki_raw = json.loads(WIKI_RAW.read_text())
    yt_raw = json.loads(YT_RAW.read_text())
    meta = wiki_raw["_meta"]
    start = datetime.fromisoformat(meta["start"]).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(meta["end"]).replace(tzinfo=timezone.utc)

    out: dict = {
        "params": {"window_days": window_days, "span": span, "gamma": GAMMA,
                   "dormancy": DORMANCY, "qh_rule": qh_rule, "qh_k": qh_k,
                   "horizon": [meta["start"], meta["end"]],
                   "wiki_normalization": "E_tau = mean daily pageviews in window"},
        "wikipedia": {}, "wikipedia_alt": {},
        "youtube_matched": {}, "youtube_fullspan": {},
    }

    for key, rec in wiki_raw.items():
        if key == "_meta":
            continue
        trace = wiki_trace(rec["views"], window_days, span)
        prof = profile(trace, qh_rule, qh_k)
        prof.update({"topic": rec["topic"], "article": rec["article"],
                     "pattern": rec["pattern"], "fidelity": rec["fidelity"],
                     "total_views": int(sum(rec["views"])),
                     "no_data_years": rec.get("no_data_years", [])})
        target = out["wikipedia_alt"] if rec["role"] == "alt" else out["wikipedia"]
        target[rec["topic"]] = prof

    for topic, rec in yt_raw.items():
        videos = rec["videos"]
        if not videos:
            continue
        matched = yt_trace(videos, window_days, start, end, span)
        if matched is not None:
            p = profile(matched, qh_rule, qh_k)
            in_horizon = sum(1 for v in videos
                             if start <= datetime.fromisoformat(v["publishedAt"]) < end)
            p.update({"pattern": rec["pattern"], "n_videos": len(videos),
                      "n_videos_in_horizon": in_horizon})
            out["youtube_matched"][topic] = p
        p_full = profile(yt_trace_fullspan(videos, window_days), qh_rule, qh_k)
        p_full["pattern"] = rec["pattern"]
        out["youtube_fullspan"][topic] = p_full

    # ------------------------------------------------------------- agreement
    topics = [t for t in out["wikipedia"] if t in out["youtube_matched"]]
    topics.sort()
    high = [t for t in topics if out["wikipedia"][t]["fidelity"] == "high"]

    metrics = ["ga_ratio", "recency_ratio", "R", "n_reactivations", "coverage"]
    agreement = {}
    for yt_key in ("youtube_matched", "youtube_fullspan"):
        agreement[yt_key] = {}
        for subset_name, subset in (("all", topics), ("high_fidelity", high)):
            block = {"n": len(subset), "topics": subset}
            for m in metrics:
                x = [out["wikipedia"][t][m] for t in subset]
                y = [out[yt_key][t][m] for t in subset]
                block[m] = {"rho": spearman(x, y), "p": perm_p(x, y)}
            agreement[yt_key][subset_name] = block
    out["agreement"] = agreement

    # ------------------------------------------- group structure on Wikipedia
    groups = {}
    for pat in PATTERN_ORDER:
        ts = [t for t in out["wikipedia"] if out["wikipedia"][t]["pattern"] == pat]
        groups[pat] = {"n": len(ts), "topics": sorted(ts)}
        for m in metrics + ["Ph"]:
            groups[pat][m] = float(np.median([out["wikipedia"][t][m] for t in ts]))
    out["wikipedia_group_medians"] = groups

    # RQ6 replication: do debunked cohorts occupy a distinct region?
    deb = [t for t in out["wikipedia"] if out["wikipedia"][t]["pattern"] == "debunked"]
    cmp_ = [t for t in out["wikipedia"] if out["wikipedia"][t]["pattern"] != "debunked"]
    rq6 = {}
    for m in ["ga_ratio", "recency_ratio", "R"]:
        d = np.array([out["wikipedia"][t][m] for t in deb])
        c = np.array([out["wikipedia"][t][m] for t in cmp_])
        rq6[m] = {
            "debunked_range": [float(d.min()), float(d.max())],
            "debunked_median": float(np.median(d)),
            "comparison_range": [float(c.min()), float(c.max())],
            "comparison_median": float(np.median(c)),
            "overlap_fraction": float(np.mean(
                (d >= c.min()) & (d <= c.max()))),
        }
    rq6["all_debunked_reemergent"] = bool(all(
        out["wikipedia"][t]["n_reactivations"] >= 2 for t in deb))
    rq6["debunked_R_range"] = [
        float(min(out["wikipedia"][t]["R"] for t in deb)),
        float(max(out["wikipedia"][t]["R"] for t in deb))]
    rq6["n_debunked_with_reactivation"] = int(sum(
        out["wikipedia"][t]["n_reactivations"] > 0 for t in deb))
    out["rq6_replication"] = rq6

    # mapping sensitivity: primary vs alternate article
    alt_rows = []
    for t, alt in out["wikipedia_alt"].items():
        prim = out["wikipedia"][t]
        alt_rows.append({"topic": t, "primary": prim["article"], "alt": alt["article"],
                         **{f"{m}_primary": prim[m] for m in metrics},
                         **{f"{m}_alt": alt[m] for m in metrics}})
    out["mapping_sensitivity"] = alt_rows
    return out


# ------------------------------------------------------------------ reporting
def report(out: dict) -> None:
    W = out["params"]["window_days"]
    print("=" * 96)
    print(f"WIKIPEDIA EVENT-LEVEL PROFILES  ({W}-day windows, "
          f"{out['params']['horizon'][0]}..{out['params']['horizon'][1]}, "
          f"gamma={GAMMA}, d={DORMANCY})")
    print("=" * 96)
    hdr = (f"{'topic':24} {'pattern':11} {'article':34} "
           f"{'P_h':>9} {'G/A':>5} {'recency':>8} {'R_h':>6} {'rct':>4} {'cov':>5}")
    print(hdr)
    for pat in PATTERN_ORDER:
        for t, p in sorted(out["wikipedia"].items(),
                           key=lambda kv: -kv[1]["ga_ratio"]):
            if p["pattern"] != pat:
                continue
            print(f"{t:24} {pat:11} {p['article'][:34]:34} "
                  f"{p['Ph']:9.1f} {p['ga_ratio']:5.2f} {p['recency_ratio']:8.2f} "
                  f"{p['R']:6.3f} {p['n_reactivations']:4d} {p['coverage']*100:4.0f}%")

    print()
    print("=" * 96)
    print("GROUP MEDIANS ON WIKIPEDIA (do the a priori patterns separate on event-level data?)")
    print("=" * 96)
    print(f"  {'pattern':12}{'n':>3}{'G/A':>8}{'recency':>10}{'R_h':>8}{'react':>7}{'cov':>8}")
    for pat, g in out["wikipedia_group_medians"].items():
        print(f"  {pat:12}{g['n']:3d}{g['ga_ratio']:8.2f}{g['recency_ratio']:10.2f}"
              f"{g['R']:8.3f}{g['n_reactivations']:7.1f}{g['coverage']*100:7.0f}%")

    print()
    print("=" * 96)
    print("AGREEMENT WITH THE YOUTUBE PUBLICATION-COHORT PROFILES (Spearman rho, "
          "permutation p)")
    print("=" * 96)
    for yt_key, label in (("youtube_matched", "matched horizon (2015-07..2026-06)"),
                          ("youtube_fullspan", "full span (paper's own axis)")):
        print(f"\n-- YouTube {label} --")
        for subset in ("all", "high_fidelity"):
            b = out["agreement"][yt_key][subset]
            cells = "  ".join(
                f"{m}={b[m]['rho']:+.2f}(p={b[m]['p']:.3f})"
                for m in ["ga_ratio", "recency_ratio", "R", "n_reactivations"])
            print(f"  {subset:14} n={b['n']:2d}  {cells}")

    print()
    print("=" * 96)
    print("RQ6 REPLICATION -- do debunked cohorts occupy a distinct region of profile space?")
    print("=" * 96)
    r = out["rq6_replication"]
    for m in ["ga_ratio", "recency_ratio", "R"]:
        v = r[m]
        print(f"  {m:14} debunked [{v['debunked_range'][0]:.3f}, {v['debunked_range'][1]:.3f}] "
              f"med {v['debunked_median']:.3f}   |   comparison "
              f"[{v['comparison_range'][0]:.3f}, {v['comparison_range'][1]:.3f}] "
              f"med {v['comparison_median']:.3f}   |   "
              f"{v['overlap_fraction']*100:.0f}% of debunked inside comparison range")
    print(f"  all 19 debunked topics show >=2 qualifying reactivations: "
          f"{r['all_debunked_reemergent']}  "
          f"({r['n_debunked_with_reactivation']}/19 have >=1); "
          f"R_h in [{r['debunked_R_range'][0]:.3f}, {r['debunked_R_range'][1]:.3f}]")

    if out["mapping_sensitivity"]:
        print()
        print("=" * 96)
        print("MAPPING SENSITIVITY -- primary article vs alternate")
        print("=" * 96)
        print(f"  {'topic':24}{'G/A p':>8}{'G/A a':>8}{'rec p':>9}{'rec a':>9}"
              f"{'R p':>7}{'R a':>7}")
        for row in out["mapping_sensitivity"]:
            print(f"  {row['topic']:24}{row['ga_ratio_primary']:8.2f}"
                  f"{row['ga_ratio_alt']:8.2f}{row['recency_ratio_primary']:9.2f}"
                  f"{row['recency_ratio_alt']:9.2f}{row['R_primary']:7.3f}"
                  f"{row['R_alt']:7.3f}")


def main() -> None:
    global OUT_PATH
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-days", type=int, default=90,
                    help="window length (default 90, the paper's primary spec)")
    ap.add_argument("--span", choices=("common", "own"), default="common",
                    help="'common' keeps the whole horizon; 'own' starts each "
                         "trace at its first active window")
    ap.add_argument("--qh-rule", choices=("quartile", "elevation"),
                    default="quartile",
                    help="activity threshold rule (default: the paper's quartile rule)")
    ap.add_argument("--qh-k", type=float, default=2.0,
                    help="multiplier of the median for --qh-rule elevation")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.out:
        OUT_PATH = args.out
    elif args.window_days != 90 or args.span != "common" or args.qh_rule != "quartile":
        OUT_PATH = HERE / (f"wikipedia_results_{args.window_days}d_{args.span}"
                           f"_{args.qh_rule}.json")

    out = run(args.window_days, args.span, args.qh_rule, args.qh_k)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    report(out)
    print("\nwrote", OUT_PATH)


if __name__ == "__main__":
    main()
