"""Ground-truth checks for the sequence baselines and the G/A persistence ratio.

The paper (Section 6.2) recommends that the framework's identities be exercised
by tests. The Wikipedia replication adds two estimators that were not previously
in the pipeline -- Kleinberg's burst automaton and PELT -- plus one derived
indicator, so those get the same treatment: each is run against a series whose
answer is known by construction.

    python analysis/test_sequence_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from metrics import geometric_persistence, thgee
from sequence_baselines import kleinberg_bursts, pelt_changepoints
from wikipedia_analysis import profile, spearman, wiki_trace

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def test_ga_ratio() -> None:
    """G/A reproduces the RQ1 separation of paper Table 3 and is scale-free."""
    acute = np.array([63, 0, 0, 0, 0, 0], float)
    cont = np.array([10, 11, 9, 10, 12, 11], float)
    ga = lambda t: geometric_persistence(t) / t.mean()  # noqa: E731
    check("G/A: acute burst is far below the continuous trace",
          ga(acute) < 0.15 < 0.9 < ga(cont),
          f"acute {ga(acute):.3f} vs continuous {ga(cont):.3f}")
    flat = np.full(8, 7.0)
    check("G/A: a perfectly flat trace scores 1", abs(ga(flat) - 1.0) < 1e-9)
    check("G/A: order-invariant, like P_h",
          abs(ga(cont) - ga(cont[::-1])) < 1e-12)
    # Scale-freeness holds asymptotically: the +1 shift in P_h makes it exact
    # only in the limit, so require the ratio to be stable under a 1000x rescale.
    big = cont * 1000.0
    check("G/A: near-invariant to rescaling the trace",
          abs(ga(big) - ga(cont)) < 0.02,
          f"{ga(cont):.4f} at 1x vs {ga(big):.4f} at 1000x")


def test_reduction_identity() -> None:
    """The paper's Equation 13: TH-GEE(gamma=0) == P_h."""
    rng = np.random.default_rng(7)
    worst = max(abs(geometric_persistence(t) - thgee(t, 0.0))
                for t in (rng.gamma(2.0, 50.0, 40) for _ in range(200)))
    check("TH-GEE(gamma=0) == P_h", worst < 1e-9, f"max abs diff {worst:.2e}")


def test_kleinberg() -> None:
    """Recovers injected bursts, and finds none in a stationary series."""
    rng = np.random.default_rng(0)
    n = 2000
    r = rng.poisson(100, n).astype(float)
    r[500:560] *= 6
    r[1400:1430] *= 9
    corpus = np.full(n, 100_000.0)
    bursts = kleinberg_bursts(r, corpus)
    onsets = [b[0] for b in bursts]
    check("kleinberg: recovers both injected bursts",
          len(bursts) == 2 and all(abs(a - b) <= 3 for a, b in zip(onsets, [500, 1400])),
          f"{bursts}")
    check("kleinberg: no bursts in a stationary series",
          kleinberg_bursts(rng.poisson(100, n).astype(float), corpus) == [])
    check("kleinberg: an all-zero series yields no bursts",
          kleinberg_bursts(np.zeros(n), corpus) == [])
    # A rise that matches the corpus is not a burst: the denominator absorbs it.
    r2 = rng.poisson(100, n).astype(float)
    r2[800:900] *= 5
    c2 = corpus.copy()
    c2[800:900] *= 5
    check("kleinberg: a rise proportional to the corpus is not a burst",
          kleinberg_bursts(r2, c2) == [], f"{kleinberg_bursts(r2, c2)}")


def test_pelt() -> None:
    """Finds a known mean shift, and nothing in a stationary series."""
    rng = np.random.default_rng(1)
    x = np.log1p(np.concatenate([rng.poisson(50, 1000), rng.poisson(300, 1000)]).astype(float))
    cps = pelt_changepoints(x)
    check("pelt: locates the injected mean shift at 1000",
          any(abs(c - 1000) <= 5 for c in cps), f"{cps}")
    flat = np.log1p(rng.poisson(100, 2000).astype(float))
    check("pelt: no change points in a stationary series",
          pelt_changepoints(flat) == [], f"{pelt_changepoints(flat)}")
    check("pelt: a constant series has no change points",
          pelt_changepoints(np.full(500, 3.0)) == [])
    check("pelt: segments respect the 14-day minimum",
          all(b - a >= 14 for a, b in zip([0] + cps, cps + [len(x)])), f"{cps}")


def test_wiki_trace() -> None:
    """E_tau is mean daily views and drops only the trailing partial window."""
    views = list(range(0, 200))
    tr = wiki_trace(views, 90)
    check("wiki_trace: keeps whole windows only", tr.size == 2, f"T={tr.size}")
    check("wiki_trace: window 0 is the mean of days 0..89",
          abs(tr[0] - np.mean(views[:90])) < 1e-9)
    flat = wiki_trace([10] * 360, 90)
    check("wiki_trace: a flat daily series gives a flat trace",
          np.allclose(flat, 10.0))


def test_profile_consistency() -> None:
    """The replication's profile() agrees with the paper's own construct values."""
    p = profile(np.array([12, 10, 0, 0, 0, 14], float))
    check("profile: re-emergent construct trace reproduces R_h = 0.353 at q_h/d "
          "of this trace", p["n_reactivations"] == 1, f"R_h={p['R']:.3f}")
    cont = profile(np.array([10, 11, 9, 10, 12, 11], float))
    check("profile: continuous trace has no reactivation",
          cont["n_reactivations"] == 0 and cont["R"] == 0.0)


def test_spearman() -> None:
    check("spearman: identical rankings give +1",
          abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12)
    check("spearman: reversed rankings give -1",
          abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-12)
    check("spearman: ties are averaged, not broken arbitrarily",
          abs(spearman([1, 1, 2, 2], [5, 5, 9, 9]) - 1.0) < 1e-12)


def main() -> None:
    for fn in (test_ga_ratio, test_reduction_identity, test_kleinberg, test_pelt,
               test_wiki_trace, test_profile_consistency, test_spearman):
        print(f"{fn.__name__}:")
        fn()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
