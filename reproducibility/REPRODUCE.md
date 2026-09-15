# Reproducibility bundle — TH-GEE

Self-contained materials to reproduce the empirical results in *TH-GEE:
Measuring Persistent and Re-Emergent Engagement in Online Influence Activity*:
the synthetic construct tests (RQ1–RQ4) and the known-pattern YouTube topic
analysis (Results, including the debunked-claim topics).

## Contents

| File | Role |
|------|------|
| `metrics.py` | Reference implementation of the three measures: Geometric Persistence Score `P_h`, `TH-GEE`, Re-Emergence Index `R_h`. |
| `youtube_analysis.py` | Runs the synthetic tests and the topic analysis; writes `youtube_results.json`. |
| `youtube_raw.json` | **Frozen data snapshot** — 37 topics × up to 50 videos (including the full 19-topic debunked-claim set), each with its publication timestamp and final like/comment counts. Public counts and timestamps only; no video content. |
| `make_figure.py` | Regenerates the Results figure `figures/youtube_exemplars.pdf`. |
| `collector.py`, `database.py` | *Optional* — only needed to re-collect from the YouTube Data API. |
| `requirements.txt` | Dependencies. |

## Reproduce (no API key required)

```bash
pip install numpy matplotlib
python youtube_analysis.py     # -> youtube_results.json
python make_figure.py          # -> figures/youtube_exemplars.pdf
```

`youtube_analysis.py` reads the frozen `youtube_raw.json`, so results are
deterministic and do not depend on live YouTube responses.

### Expected output (spot-check against the paper)

Synthetic construct tests:

- RQ1: matched-volume acute burst `P_h = 1.000` vs. continuous `P_h = 10.460`.
- RQ2: across all permutations `P_h` std `≈ 3.4e-16` (order-invariant);
  `|P_h − TH-GEE(γ=0)|` max `≈ 4.4e-15` (reduction identity).

YouTube topics (90-day windows), scale-free indicators:

| topic | coverage | recency (TH-GEE/P_h) | R_h | reactivations |
|-------|---------:|---------------------:|----:|--------------:|
| meditation music (persistent) | 55% | 0.78 | 0.145 | 4 |
| harlem shake (acute) | 22% | 1.26 | 0.315 | 5 |
| super bowl halftime (re-emergent) | 33% | 2.38 | 0.523 | 8 |
| chatgpt (recent) | 86% | 1.56 | 0.085 | 1 |
| flat earth (debunked → persistent) | 67% | 15.28 | 0.100 | 3 |
| pizzagate (debunked → re-emergent) | 44% | 3.14 | 0.323 | 5 |

## Re-collect from YouTube (optional)

Live search results change over time, so re-collection will **not** reproduce
the exact numbers — this is why the frozen snapshot is the citable artifact. To
refresh the snapshot with your own key:

```bash
pip install google-api-python-client sqlalchemy
export YOUTUBE_API_KEY=...            # a free YouTube Data API v3 key
python youtube_analysis.py --collect  # only fetches topics not already cached
```

## Data note

`youtube_raw.json` contains public engagement counts and timestamps retrieved via
the YouTube Data API v3. No video content is redistributed. The topics are listed
in `youtube_analysis.py` (`TOPICS`) and were chosen a priori to instantiate the
temporal patterns the framework separates.
## Event-level validation (Section 5.4 figure)

Reproduces Figure~\ref{fig:eventlevel} and the numbers quoted with it. Daily
Wikipedia pageview series are audience interactions dated to the day they
occurred, so the sequence-oriented baselines the paper positions as complements
can be run on the same topics.

```bash
python test_sequence_baselines.py                  # ground-truth checks, ~2 s
python wikipedia_analysis.py --window-days 7 --qh-rule elevation
python sequence_baselines.py  --window-days 7 --qh-rule elevation \
    --out sequence_baseline_results_7d_elevation.json
python make_wikipedia_figures.py                   # -> figures/wikipedia_bursts.pdf
```

| File | Role |
|------|------|
| `wikipedia_collect.py` | Topic-to-article mapping (with fidelity grades) and the collector. Only needed to refresh the snapshot; needs network, no API key. |
| `wikipedia_raw.json` | **Frozen snapshot** — daily pageviews for 37 topics + 8 alternate articles, 2015-07-01 to 2026-06-30. |
| `wikipedia_analysis.py` | Weekly trace construction and the temporal profile on this data. |
| `sequence_baselines.py` | Kleinberg burst automaton and PELT change-point estimation, implemented from their source papers; no new dependencies. |
| `test_sequence_baselines.py` | Runs both estimators against series whose answers are known by construction, and re-exercises the `TH-GEE(γ=0) = P_h` identity. |
| `make_wikipedia_figures.py` | Regenerates `figures/wikipedia_bursts.pdf`. |

### Expected output (spot-check against the paper)

- Kleinberg burst onsets: 63 of 351 reactivation windows contain one
  (lift 10.7×, permutation p = 0.0001).
- PELT change points: 68 of 351 (lift 6.6×, p = 0.0001).
- Reactivation count vs change-point count: Spearman ρ = +0.56.
- Recall: 20% of burst onsets, 13% of change points.

### Two notes on the specification

`q_h` here is twice each topic's median window score, not the 25th-percentile
rule used for the YouTube analysis. A continuously read Wikipedia article never
falls below a quantile of its own distribution, so it never registers as dormant
and `R_h` cannot respond to spike structure; the paper states this where the
figure is discussed.

The Wikimedia pageviews endpoint emits spurious 404s that its CDN then caches —
a URL 404s deterministically while both halves of the same range return data.
A naive collector silently records affected articles as having zero lifetime
pageviews. `wikipedia_collect.py` retries with a fresh cache key, bisects, and
uses the monthly endpoint to distinguish a poisoned cache entry from genuine
missing data. This matters only if you re-collect.
