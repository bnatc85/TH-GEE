# Reproducing *Curiosity, Panic, or Preparedness?*

Pipeline for the manuscript in `main(1).tex`, implementing the repository
contract in its Appendix C. Every number in `outputs/` is produced by these
scripts from the frozen configuration in `config/`.

Run date of the results checked in here: **15 September 2026**
(t₀ + 6 days for the primary event).

## Baseline placement (design point)

The Wikipedia baseline is 56 days long (`analysis.pre_event_days`). *Where it
ends* is set per event by `baseline_gap_days` in `config/events.yml`:

| Event | t₀ | `baseline_gap_days` | Baseline |
|---|---|---|---|
| Statement on AI Risk | 30 May 2023 | 0 | `[-56, -1]` = 4 Apr – 29 May 2023 |
| September 2026 warning | 9 Sep 2026 | 1 | `[-57, -2]` = 14 Jul – 7 Sep 2026 |

The 2026 attention event is 9 September (first major coverage), but the
warning began on 8 September. A baseline ending at −1 would contain the
announcement day, and because the counterfactual has AR(1) errors the forecast
for t₀ onward would be conditioned on that day's residual. The 2026 baseline
therefore ends on 7 September; 8 September is forecast out of sample like every
later day (it appears in the daily series and in Figure 1). The gap is applied
relative to whichever date is being treated as t₀, so the placebo dates and the
t₀ = 8 Sep sensitivity check use the same `[-57, -2]` design.

Two consequences for the collector: it fetches from t₀ − 57 rather than
t₀ − 56, and it now fetches enough history for the longest alternative
baseline (`alt_baselines`, 84 days), so that robustness check 2 actually
differs from the 56-day model. In the previous run only 56 days had been
collected, so the "84-day" estimate was silently identical to the 56-day one.
The extra history also lengthens the baselines available to the earliest
placebo dates. The as-of-event title-resolution rule is still scored on the
56-day baseline only.

## Install and run

```bash
pip install -r requirements.txt

export FMP_API_KEY=...            # market data; never written to disk

python src/collect_wikipedia.py                  # ~5 min
python src/collect_news.py                       # ~25 min (rate-limited)
python src/collect_markets.py                    # ~1 min
python src/classify_frames.py --export-coding-sample 400
python src/analyze_attention.py --robustness     # ~10 min
python src/analyze_markets.py
python src/build_outputs.py
```

`data/derived/run_manifest.json` records package versions, run timestamps,
random seeds, per-ticker data source, and SHA-256 hashes of each config file.

## What the configuration freezes

| File | Contents |
|---|---|
| `config/events.yml` | event dates, t₀ alternatives, per-event baseline gap, observation and estimation windows, placebo seed, kill-switch intervention dates |
| `config/wikipedia_pages.yml` | the six preregistered page families, orientation vector, negative controls, frame→family map |
| `config/portfolios.csv` | the three portfolios and three benchmarks, with selection rationale per security |

Page families and portfolio membership are read only from these files and are
never revised after outcome data are inspected.

## Data sources actually used

| Channel | Source | Notes |
|---|---|---|
| Wikipedia | Wikimedia REST pageviews | `en.wikipedia.org`, `all-access`, `agent=user` |
| News | Google News RSS | GDELT DOC 2.0, the paper's default, refused every request from this environment with its shared-IP rate limit; its static GKG mirror is ~5 MB per 15-minute slice (~30 GB per window) |
| Markets | Financial Modeling Prep | dividend- and split-adjusted closes; the paper's named source |

## Deviations from the manuscript, and why

1. **News source.** As above. Google News RSS also reaches May 2023, which the
   GDELT DOC rolling ~3-month archive cannot, so the replication event gets a
   media channel it otherwise would not have.

2. **News dating.** Google News honours `after:`/`before:` only coarsely:
   consecutive day-queries return ~92% the same ranked set, with publication
   dates spread across neighbouring days. Every article is therefore re-dated
   from its own `pubDate` and filtered to the event window
   (`classify_frames.article_date`).

3. **Frame-prevalence denominator.** The paper's narrow conceptual query cannot
   supply a daily series here: it matches a small candidate pool, so its ranked
   top-100 barely changes from day to day and, after URL deduplication, yields a
   median of **zero** stories per day. The daily series therefore uses the
   five-query union (conceptual query + the four Appendix B mechanism queries).
   The query set is identical on every day, which is what makes day-to-day
   movement in a frame's *share* interpretable; the absolute level is a property
   of the query design, and daily counts are a ranked sample, not a census.
   `n_base_corpus` is retained per day as a diagnostic.

4. **Frame 5 split.** The paper bundles governance with emergency shutdown and
   kill-switch control. H3 makes a directional claim about *control* framing
   specifically, so frame 5 is measured as two components (`governance`,
   `control`); their union reproduces the paper's frame 5.

5. **Structural time series.** `CausalImpact` is unavailable, so robustness
   check 6 fits the same idea directly with a local-level + weekly-seasonal
   unobserved-components model (`structural_counterfactual`).

6. **Robustness 11 (earnings exclusion).** No earnings-calendar source is
   available. Instead, a security is flagged when it shows both heavily elevated
   volume and a large abnormal return inside the event window, and every
   portfolio effect is reported with and without the flagged securities. This
   check matters: see "Known confounds" below.

7. **Frame-classifier validation is not complete.** The automated labels are
   keyword-dictionary labels over *headlines only*. The paper requires two human
   coders, Krippendorff's α, and a macro-F1 ≥ threshold before a frame is used.
   `data/derived/frame_coding_sample.csv` is the stratified 400-item sample for
   that step; until it is coded, the validation fields in the manifest stay
   `null`. **Frame results are provisional.**

## What could not be run, and why

| Item | Status |
|---|---|
| 2026 attention windows `[0,+7]`, `[0,+14]` | Pageviews published only through **14 Sep 2026** (t₀+5). Marked `"available": false` in the results rather than computed on a short window. |
| 2026 market windows `[-1,+1]`, `[-3,+3]`, `[0,+5]` | Require sessions after **14 Sep 2026**. Only the prespecified episode window `[8–14 Sep]` is complete. |
| 2023 event-named pages | *Statement on AI Risk*, *Pause Giant AI Experiments*, *Dario Amodei*, *Existential risk studies* had no pre-event baseline in May 2023 — an article created *by* the event cannot have a counterfactual. Dropped for 2023 under the preregistered "insufficient baseline observations" rule and logged in `data/derived/page_resolution_log_statement2023.json`. |
| Human-coded frame validation | Sample exported; coding not performed. |

Because family composition differs between events for that reason, the RQ5
cross-event comparison is reported at the one window both events support,
`[0,+3]`.

## Two things the checks caught

**Historical page titles.** Pageviews are recorded against the *requested*
title, and resolving a preregistered title to its *current* name is wrong for a
historical event: in May 2023 "Existential risk from artificial intelligence"
lived at "Existential risk from advanced artificial intelligence" and had zero
traffic under today's name. `collect_wikipedia.py` resolves each title as of the
event date, choosing among redirect aliases on **pre-event baseline coverage
only**, so no post-event outcome informs the choice.

**Baseline drift in 2023.** The negative-control pages show a cumulative
abnormal attention of −3.15 at `[0,+3]` and −24.67 at `[0,+14]` (p < 0.01) for
the 2023 event — they should sit at zero. This is genuine late-May seasonality
in the control set, not a coding error: raw views fall 0–11% post-event (Everest
climbing season peaks just before t₀; Photosynthesis and Roman Empire follow the
school year). The 2026 event shows no such drift (+1.86, p = 0.30). Every family
is therefore also reported net of this drift as
`control_adjusted_families`, and **the RQ5 comparison should use the
control-adjusted values.** Note that for 2026 the adjustment subtracts an
estimate that is not distinguishable from zero; it is applied for consistency
of rule, and the unadjusted values are reported alongside.

**Baseline placement matters for one family.** Robustness check 14
(`14_baseline_gap`) refits 2026 with the baseline run to t₀ − 1, i.e. with
8 September treated as a normal day. Actors/institutions falls from 24.1 to
11.2 SD-days — *Sam Altman*, *Geoffrey Hinton* and *OpenAI* had already
doubled on 8 September, so including that day both inflates σ and raises the
AR(1) forecast origin — while no other family moves by more than 2.2. This is
why the 2026 baseline is `[-57, -2]`.

**Placebo baselines.** Placebo dates are modelled with the same design as the
true event. In the earlier run the collector only had 56 days of history, so
the earliest placebo dates (5–6 Aug 2026) were fitted on 21–23-day baselines
and produced artefactual estimates (control −10.5, −9.2) that inflated the
placebo band. With full-length baselines the 2026 control band falls from
5.7 to 4.3 and the observed control effect (7.0) now clears it; with the
short-baseline placebos it would not have (band 9.8). The manuscript reports
the full-length version.

## Known confounds

**The 2023 market channel sits inside an earnings cluster.** The only
conventionally significant 2023 portfolio result — AI infrastructure
CAR `[-3,+3]` = **+4.84%**, randomisation p ≈ 0 — is driven by NVDA (+10.5%) and
AVGO (+9.4%), both of which reported earnings inside that window. Excluding the
flagged securities leaves **−0.30%**. The 2023 market result should not be read
as a response to the Statement on AI Risk.

## Outputs

```
figures/figure_1_attention.pdf      family attention time series (small multiples)
figures/figure_2_gap.pdf            threat-preparedness gap vs. placebo range
figures/figure_3_transmission.pdf   media framing -> attention, FDR-adjusted
figures/figure_4_market_car.pdf     portfolio CARs with 95% intervals
figures/figure_5_divergence.pdf     attention vs. equity repricing of resilience
outputs/tables/table_main_results.tex       family effects, both events
outputs/tables/table_orientation_vector.tex response-orientation vector R_t
outputs/tables/table_market_car.tex         portfolio CARs
```

`.png` copies sit beside each `.pdf` for quick inspection. `build_outputs.py`
also retains `figure_frames()`, a supplementary frame-prevalence-over-time view
that is not part of the manuscript figure set.

The manuscript (`main(1).tex`) reports only what these outputs support. It
states research questions rather than directional hypotheses, because several
prespecified windows could not be evaluated at this run date and the frame
classifier has not completed human validation. On Overleaf, rename
`references(1).bib` to `references.bib` to match `\bibliography{references}`.
