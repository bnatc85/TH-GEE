"""Sequence-oriented baselines on the event-level Wikipedia pageview series.

Section 5.4 of the paper defers Kleinberg burst detection and change-point
analysis on the grounds that they "require event-level interaction sequences
that the publication data used here do not provide". Daily pageviews are such
sequences, so both baselines are runnable here, and the open question can be
answered: does the Re-Emergence Index agree with methods that were designed for
event-level data?

Two baselines, both implemented from their source papers with no new
dependencies.

1. Kleinberg's two-state burst automaton (Kleinberg 2003, Section 4, the
   batched-stream formulation). Day t contributes r_t views of the article out
   of d_t views of the whole 37-article corpus. State i has expected share
   p_i = min(p_0 * s^i, 1-eps) where p_0 = sum(r)/sum(d) is the article's
   baseline share; the corpus denominator absorbs Wikipedia-wide weekly and
   seasonal traffic cycles, so a "burst" is a rise relative to the corpus, not
   to the calendar. Emission cost is the negative binomial log-likelihood (the
   binomial coefficient is constant across states and drops out of the Viterbi
   recursion), and moving up j-i states costs (j-i)*gamma*ln(n). Bursts are the
   maximal runs of state >= 1 in the Viterbi path.

2. PELT with an L2 (change-in-mean) cost on log1p(daily views), following
   Killick, Fearnhead and Eckley (2012), with the standard BIC penalty
   beta = pen_mult * sigma^2 * ln(n). Exact segmentation, with the PELT pruning
   rule; sigma^2 is estimated from the median absolute difference of successive
   values, which is robust to the very jumps being detected.

The comparison then asks three things:

  (a) Localisation. Do the qualifying reactivation windows found by R_h contain
      burst onsets / change points more often than windows chosen at random?
      Reported as a lift over the per-topic base rate with a permutation p-value.
  (b) Ranking. Across the 37 topics, does R_h (and the reactivation count) rank
      topics the same way burst count and change-point count do?
  (c) Coverage. What fraction of detected bursts falls inside a reactivation
      window -- i.e. how much of what the event-level detectors find does the
      quarterly re-emergence measure see at all?

Usage:
    python analysis/sequence_baselines.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from wikipedia_analysis import (
    profile, spearman, wiki_trace, perm_p, PATTERN_ORDER,
)

WIKI_RAW = HERE / "wikipedia_raw.json"
OUT_PATH = HERE / "sequence_baseline_results.json"

WINDOW_DAYS = 90
KLEINBERG_S = 2.0      # burst-state rate multiplier (Kleinberg's default)
KLEINBERG_GAMMA = 1.0  # state-transition cost multiplier (Kleinberg's default)
KLEINBERG_STATES = 3   # base + two burst levels
PELT_PEN_MULT = 4.0    # beta = PELT_PEN_MULT * sigma^2 * ln(n)
PELT_MIN_SEG = 14      # days; below this a shift is a burst, not a regime change


# ------------------------------------------------------- Kleinberg (2003) §4
def overdispersion(r: np.ndarray, d: np.ndarray) -> float:
    """Robust quasi-binomial dispersion phi of r_t given d_t at a constant share.

    Daily pageviews are far more variable than a binomial with d_t ~ 10^5 trials
    allows: weekday cycles, corpus composition drift and reader heterogeneity all
    inflate the variance. Left uncorrected, a routine 10% wobble on 30,000 views
    is astronomically unlikely under the model, so the burst automaton spends
    half the horizon in a burst state and "burst" stops meaning anything. phi is
    estimated from the *median* squared Pearson residual, rescaled by the median
    of a chi-square with one degree of freedom (0.4549), so that the genuine
    bursts being searched for do not inflate the very quantity used to discount
    them.
    """
    p0 = float(r.sum()) / float(d.sum())
    var = d * p0 * (1.0 - p0)
    ok = var > 0
    if not np.any(ok):
        return 1.0
    resid2 = (r[ok] - d[ok] * p0) ** 2 / var[ok]
    phi = float(np.median(resid2)) / 0.4549
    return max(phi, 1.0)


def kleinberg_bursts(r: np.ndarray, d: np.ndarray, *, s: float = KLEINBERG_S,
                     gamma: float = KLEINBERG_GAMMA,
                     k: int = KLEINBERG_STATES,
                     correct_overdispersion: bool = True
                     ) -> list[tuple[int, int, int]]:
    """Maximal (start, end, level) runs of burst state in the Viterbi path.

    With ``correct_overdispersion`` the counts are divided by the quasi-binomial
    dispersion before the Viterbi pass. This is the standard quasi-likelihood
    correction: it leaves the fitted shares p_i untouched and scales every
    emission log-likelihood by 1/phi, so the state-transition cost is weighed
    against a likelihood with the *effective* sample size the data support rather
    than the nominal one.
    """
    n = r.size
    total_r, total_d = float(r.sum()), float(d.sum())
    if total_r <= 0 or total_d <= 0:
        return []
    p0 = total_r / total_d
    if correct_overdispersion:
        phi = overdispersion(r, d)
        r = r / phi
        d = d / phi
    p = np.array([min(p0 * s ** i, 1.0 - 1e-12) for i in range(k)])

    # emission cost: -log Binomial(r_t; d_t, p_i), binomial coefficient dropped
    with np.errstate(divide="ignore", invalid="ignore"):
        cost = -(np.outer(r, np.log(p)) + np.outer(d - r, np.log1p(-p)))
    cost = np.nan_to_num(cost, nan=np.inf, posinf=np.inf)

    trans = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if j > i:
                trans[i, j] = (j - i) * gamma * math.log(max(n, 2))

    best = np.full(k, np.inf)
    best[0] = cost[0, 0]
    back = np.zeros((n, k), dtype=int)
    for t in range(1, n):
        prev = best[:, None] + trans          # (from, to)
        back[t] = np.argmin(prev, axis=0)
        best = prev[back[t], np.arange(k)] + cost[t]

    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmin(best))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]

    bursts: list[tuple[int, int, int]] = []
    t = 0
    while t < n:
        if path[t] >= 1:
            u = t
            while u + 1 < n and path[u + 1] >= 1:
                u += 1
            bursts.append((t, u, int(path[t:u + 1].max())))
            t = u + 1
        else:
            t += 1
    return bursts


# ----------------------------------------- PELT (Killick et al. 2012), L2 cost
def _long_run_variance(x: np.ndarray, batch: int) -> float:
    """Batch-means long-run variance, the scale a change-in-mean test needs.

    The usual successive-difference estimator assumes independent observations.
    Daily pageviews are strongly autocorrelated -- yesterday's readership is much
    the best predictor of today's -- so that estimator measures day-to-day
    jitter, not the variability of a *mean* over a stretch of days, and PELT
    responds by cutting the series into dozens of spurious regimes. Batch means
    over non-overlapping blocks of ``batch`` days capture the autocorrelation
    directly: the variance of the block means, scaled by the block length, is the
    variance of the quantity actually being compared across a candidate split.
    """
    n = x.size
    b = max(2, min(batch, n // 4))
    m = n // b
    if m < 2:
        return float(np.var(x)) or 1.0
    means = x[: m * b].reshape(m, b).mean(axis=1)
    # median-based dispersion of the block means, robust to the real shifts
    mad = float(np.median(np.abs(np.diff(means)))) / (math.sqrt(2.0) * 0.6745)
    sigma2 = b * mad ** 2
    return sigma2 if sigma2 > 0 else (float(np.var(x)) or 1.0)


def pelt_changepoints(x: np.ndarray, *, pen_mult: float = PELT_PEN_MULT,
                      min_size: int = PELT_MIN_SEG) -> list[int]:
    """Exact change-in-mean segmentation; returns interior change-point indices."""
    n = x.size
    if n < 2 * min_size:
        return []
    sigma2 = _long_run_variance(x, min_size)
    beta = pen_mult * sigma2 * math.log(n)

    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x ** 2)])

    F = np.full(n + 1, np.inf)
    F[0] = -beta
    prev = np.full(n + 1, -1, dtype=int)     # backpointer, reconstructed at the end
    candidates = np.array([0], dtype=int)
    for t in range(min_size, n + 1):
        # Only candidates at least min_size back can close a segment at t; the
        # rest are not yet eligible but must be carried forward, not discarded.
        eligible = candidates[t - candidates >= min_size]
        if eligible.size:
            m = (t - eligible).astype(float)
            total = cs[t] - cs[eligible]
            seg = (cs2[t] - cs2[eligible]) - total * total / m
            vals = F[eligible] + seg + beta
            j = int(np.argmin(vals))
            F[t] = vals[j]
            prev[t] = int(eligible[j])
            # PELT pruning (K = 0 for the L2 cost): drop s that can never again
            # be optimal. Ineligible candidates are kept regardless.
            keep = eligible[vals - beta <= F[t]]
            candidates = np.concatenate([
                candidates[t - candidates < min_size], keep, [t]])
        else:
            candidates = np.append(candidates, t)

    if not math.isfinite(F[n]):
        return []
    out: list[int] = []
    t = n
    while t > 0 and prev[t] > 0:
        t = int(prev[t])
        out.append(t)
    return out[::-1]


# ------------------------------------------------------------------- analysis
def run(window_days: int, qh_rule: str = "quartile", qh_k: float = 2.0) -> dict:
    raw = json.loads(WIKI_RAW.read_text())
    meta = raw["_meta"]
    series = {rec["topic"]: np.asarray(rec["views"], dtype=float)
              for k, rec in raw.items() if k != "_meta" and rec["role"] == "primary"}
    patterns = {rec["topic"]: rec["pattern"]
                for k, rec in raw.items() if k != "_meta" and rec["role"] == "primary"}

    n_days = min(v.size for v in series.values())
    n_days -= n_days % window_days              # whole windows only
    T = n_days // window_days
    corpus = np.sum([v[:n_days] for v in series.values()], axis=0)

    out: dict = {"params": {"window_days": window_days, "n_days": int(n_days),
                            "qh_rule": qh_rule, "qh_k": qh_k,
                            "n_windows": int(T), "horizon": [meta["start"], meta["end"]],
                            "kleinberg": {"s": KLEINBERG_S, "gamma": KLEINBERG_GAMMA,
                                          "states": KLEINBERG_STATES},
                            "pelt": {"cost": "L2 on log1p(daily views)",
                                     "penalty": f"{PELT_PEN_MULT} * sigma^2 * ln(n)",
                                     "min_segment_days": PELT_MIN_SEG}},
                 "topics": {}}

    for topic, v in series.items():
        r = v[:n_days]
        bursts = kleinberg_bursts(r, corpus)
        cps = pelt_changepoints(np.log1p(r))
        prof = profile(wiki_trace(list(r), window_days), qh_rule, qh_k)

        burst_windows = sorted({b[0] // window_days for b in bursts})
        cp_windows = sorted({c // window_days for c in cps})
        react = set(prof["reactivation_windows"])

        out["topics"][topic] = {
            "pattern": patterns[topic],
            "R": prof["R"], "n_reactivations": prof["n_reactivations"],
            "reactivation_windows": prof["reactivation_windows"],
            "recency_ratio": prof["recency_ratio"], "ga_ratio": prof["ga_ratio"],
            "n_bursts": len(bursts),
            "burst_days": int(sum(b[1] - b[0] + 1 for b in bursts)),
            "burst_onset_windows": burst_windows,
            "n_changepoints": len(cps),
            "changepoint_windows": cp_windows,
            "react_hits_burst": int(len(react & set(burst_windows))),
            "react_hits_cp": int(len(react & set(cp_windows))),
        }

    # ------------------------------------------------------- (a) localisation
    loc = {}
    for name, key in (("kleinberg_burst_onset", "burst_onset_windows"),
                      ("pelt_changepoint", "changepoint_windows")):
        hits = precision_num = 0
        base_rates, lifts = [], []
        recall_num = recall_den = 0
        for topic, rec in out["topics"].items():
            marks = set(rec[key])
            react = set(rec["reactivation_windows"])
            if not react or not marks:
                continue
            base = len(marks) / T                       # P(random window marked)
            hit = len(react & marks) / len(react)       # observed precision
            hits += len(react & marks)
            precision_num += len(react)
            recall_num += len(react & marks)
            recall_den += len(marks)
            base_rates.append(base)
            lifts.append(hit / base if base > 0 else float("nan"))
        loc[name] = {
            "reactivation_windows_total": precision_num,
            "reactivation_windows_marked": hits,
            "precision": hits / precision_num if precision_num else float("nan"),
            "base_rate_mean": float(np.mean(base_rates)) if base_rates else float("nan"),
            # Aggregate lift is the headline: the per-topic median is 0 whenever
            # most topics happen to score no hits, which says nothing about the
            # pooled effect the permutation test is actually evaluating.
            "lift_aggregate": ((hits / precision_num) / float(np.mean(base_rates))
                               if precision_num and base_rates
                               and np.mean(base_rates) > 0 else float("nan")),
            "lift_median": float(np.nanmedian(lifts)) if lifts else float("nan"),
            "recall_of_marks": recall_num / recall_den if recall_den else float("nan"),
        }

    # permutation test: reshuffle each topic's reactivation windows uniformly
    rng = np.random.default_rng(20260731)
    for name, key in (("kleinberg_burst_onset", "burst_onset_windows"),
                      ("pelt_changepoint", "changepoint_windows")):
        obs = loc[name]["reactivation_windows_marked"]
        n_perm = 10000
        null = np.zeros(n_perm, dtype=int)
        specs = [(len(r["reactivation_windows"]), r[key])
                 for r in out["topics"].values()
                 if r["reactivation_windows"] and r[key]]
        # Per topic, draw k windows without replacement n_perm times at once and
        # count how many land on a mark: argsort of uniforms is a vectorised
        # sample-without-replacement over the T windows.
        for k, marks in specs:
            flag = np.zeros(T, dtype=np.int8)
            flag[np.asarray(marks, dtype=int)] = 1
            draws = np.argsort(rng.random((n_perm, T)), axis=1)[:, :k]
            null += flag[draws].sum(axis=1)
        loc[name]["permutation_p"] = float((np.sum(null >= obs) + 1) / (null.size + 1))
        loc[name]["null_mean_marked"] = float(null.mean())
    out["localisation"] = loc

    # ---------------------------------------------------------- (b) ranking
    topics = sorted(out["topics"])
    rank = {}
    for a in ("R", "n_reactivations"):
        for b in ("n_bursts", "burst_days", "n_changepoints"):
            x = [out["topics"][t][a] for t in topics]
            y = [out["topics"][t][b] for t in topics]
            rank[f"{a}~{b}"] = {"rho": spearman(x, y), "p": perm_p(x, y)}
    out["rank_agreement"] = rank

    # group medians, to see whether the baselines recover the a priori patterns
    groups = {}
    for pat in PATTERN_ORDER:
        ts = [t for t in topics if out["topics"][t]["pattern"] == pat]
        if not ts:
            continue
        groups[pat] = {"n": len(ts)}
        for m in ("R", "n_reactivations", "n_bursts", "burst_days", "n_changepoints"):
            groups[pat][m] = float(np.median([out["topics"][t][m] for t in ts]))
    out["group_medians"] = groups
    return out


def report(out: dict) -> None:
    p = out["params"]
    print("=" * 100)
    print(f"SEQUENCE BASELINES ON DAILY PAGEVIEWS  ({p['n_days']} days, "
          f"{p['n_windows']} x {p['window_days']}-day windows)")
    print("=" * 100)
    print(f"{'topic':24} {'pattern':11} {'R_h':>6} {'rct':>4} {'bursts':>7} "
          f"{'burstd':>7} {'chgpts':>7} {'rct@burst':>10} {'rct@cp':>7}")
    for pat in PATTERN_ORDER:
        for t, r in sorted(out["topics"].items(), key=lambda kv: -kv[1]["R"]):
            if r["pattern"] != pat:
                continue
            print(f"{t:24} {pat:11} {r['R']:6.3f} {r['n_reactivations']:4d} "
                  f"{r['n_bursts']:7d} {r['burst_days']:7d} {r['n_changepoints']:7d} "
                  f"{r['react_hits_burst']:10d} {r['react_hits_cp']:7d}")

    print()
    print("=" * 100)
    print("(a) LOCALISATION -- do R_h reactivation windows land on independently detected events?")
    print("=" * 100)
    for name, L in out["localisation"].items():
        print(f"  {name}")
        print(f"     {L['reactivation_windows_marked']}/{L['reactivation_windows_total']} "
              f"reactivation windows contain a mark  (precision {L['precision']:.3f})")
        print(f"     random-window base rate {L['base_rate_mean']:.3f}  ->  "
              f"lift {L['lift_aggregate']:.2f}x (per-topic median {L['lift_median']:.2f}x)"
              f"   permutation p = {L['permutation_p']:.4f} "
              f"(null mean {L['null_mean_marked']:.1f} marked)")
        print(f"     recall of all marks by reactivation windows: {L['recall_of_marks']:.3f}")

    print()
    print("=" * 100)
    print("(b) RANK AGREEMENT ACROSS THE 37 TOPICS (Spearman rho, permutation p)")
    print("=" * 100)
    for k, v in out["rank_agreement"].items():
        print(f"  {k:32} rho = {v['rho']:+.3f}   p = {v['p']:.4f}")

    print()
    print("=" * 100)
    print("GROUP MEDIANS")
    print("=" * 100)
    print(f"  {'pattern':12}{'n':>3}{'R_h':>8}{'react':>8}{'bursts':>9}"
          f"{'burstdays':>11}{'chgpts':>9}")
    for pat, g in out["group_medians"].items():
        print(f"  {pat:12}{g['n']:3d}{g['R']:8.3f}{g['n_reactivations']:8.1f}"
              f"{g['n_bursts']:9.1f}{g['burst_days']:11.1f}{g['n_changepoints']:9.1f}")


def main() -> None:
    global OUT_PATH
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    ap.add_argument("--qh-rule", choices=("quartile", "elevation"),
                    default="quartile")
    ap.add_argument("--qh-k", type=float, default=2.0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.out:
        OUT_PATH = args.out
    out = run(args.window_days, args.qh_rule, args.qh_k)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    report(out)
    print("\nwrote", OUT_PATH)


if __name__ == "__main__":
    main()
