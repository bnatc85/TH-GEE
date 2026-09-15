# TH-GEE empirical analysis

Reproduces the numbers and figure in the paper's **Results** section: the
synthetic construct tests (RQ1–RQ4) and the known-pattern YouTube topic analysis.

## Primary: YouTube topic analysis

`youtube_analysis.py` collects public YouTube metadata (video publish times +
final like/comment counts) for a curated set of topics chosen to instantiate each
temporal pattern, then builds video-publication-with-engagement traces and
computes each topic's profile.

```bash
# refresh the cached snapshot from the API (needs YOUTUBE_API_KEY):
python analysis/youtube_analysis.py --collect

# analyze the cached snapshot (no API key needed — reproduces the paper):
python analysis/youtube_analysis.py

# render the exemplar timelines figure:
python analysis/make_figure.py
```

`youtube_raw.json` is the frozen snapshot the paper numbers come from (public
counts and timestamps only — no video content). `youtube_results.json` holds the
computed profiles and synthetic tests. `figures/youtube_exemplars.pdf` is the
Results figure.

Parameters: interaction weights `like=0.3, comment=0.7` plus a unit `publication`
term; 90-day (quarterly) windows; recency `γ=0.10`; dormancy `d=2`; activity
threshold `q_h` = 25th percentile of each topic's active windows.

## Superseded: RedNote-Vibe

`rednote_analysis.py` implements an earlier proof-of-concept over the
RedNote-Vibe corpus. The manuscript now uses YouTube instead, but the script is
kept because it still runs and documents the same pipeline on a different corpus.
See its header for the data download.

## Event-level replication: Wikipedia pageviews (Online Resource 1)

Reproduces the supplementary event-level validation. The YouTube traces anchor
engagement to *publication* time; Wikipedia pageviews are audience activity dated
to the day it happened, so the same 37 topics can be re-measured on event-level
data — and the sequence-oriented baselines the paper defers in §5.4 become
runnable.

```bash
bash analysis/run_wikipedia_replication.sh   # end to end
```

or step by step:

```bash
python analysis/wikipedia_collect.py --alts    # -> wikipedia_raw.json (network, no API key)
python analysis/test_sequence_baselines.py     # ground-truth checks
python analysis/wikipedia_analysis.py          # -> wikipedia_results.json
python analysis/sequence_baselines.py          # -> sequence_baseline_results.json
python analysis/make_wikipedia_figures.py      # -> figures/wikipedia_*.pdf
python analysis/make_wikipedia_tables.py       # -> ../reproducibility/or1_tables.tex
```

| File | Role |
|------|------|
| `wikipedia_collect.py` | Topic→article mapping (with fidelity grades) and daily pageview collection, 2015-07-01→2026-06-30 |
| `wikipedia_analysis.py` | Event-level traces and profiles; recomputes the YouTube profiles on a matched horizon and reports rank agreement |
| `sequence_baselines.py` | Kleinberg burst automaton and PELT change-point detection, and their agreement with `R_h` |
| `test_sequence_baselines.py` | Checks both estimators against series with known answers |
| `make_wikipedia_figures.py`, `make_wikipedia_tables.py` | Figures and generated LaTeX tables |
| `wikipedia_raw.json` | **Frozen snapshot** — 37 primary + 8 alternate daily series |

### Collection caveat

The Wikimedia pageviews endpoint returns *spurious* 404s which its CDN then
caches: a URL 404s deterministically while both halves of the same range return
data, and the identical range with a throwaway query parameter succeeds. A naive
collector silently records affected articles as having zero lifetime pageviews.
`wikipedia_collect.py` retries with a fresh cache key, bisects, and uses the
monthly endpoint as an oracle to tell a poisoned cache entry from genuine
missing data (an article created inside the horizon).

### Two indicators do not transfer

Active-window coverage is degenerate on pageview data — a live article is never
inactive, so coverage is ~100% for nearly every topic. The scale-free persistence
quantity that does transfer is `ga_ratio` = `P_h` / arithmetic mean, which is 1
for a flat trace and →0 as engagement concentrates. It reproduces the paper's RQ1
separation exactly (0.095 acute vs 0.996 continuous on Table 3's traces).

The activity threshold `q_h` also needs restating: the paper's 25th-percentile
rule means an evergreen topic never registers as dormant, so `--qh-rule
elevation` (`q_h = k ×` median) is reported alongside. See Online Resource 1 §6.

### Headline results

- The group-level RQ6 conclusion replicates: at weekly resolution all 19
  debunked-claim cohorts show ≥2 reactivations and overlap the comparison cohorts.
- Topic-level rank agreement with the YouTube profiles is weak: persistence
  ρ=+0.43 (p=0.008) on the primary specification but +0.19 under an own-span
  control; recency +0.28…+0.43; re-emergence never significant.
- Under the elevation rule at weekly resolution, `R_h` reactivation windows
  contain a Kleinberg burst onset 10.7× and a PELT change point 6.6× more often
  than chance (both p=0.0001), ρ=+0.56 with change-point counts.
