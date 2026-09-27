# Reproducing *Curiosity, Panic, or Preparedness?*

Pipeline for the manuscript in `main(1).tex`, implementing the repository
contract in its Appendix C. Every number in `outputs/` is produced by these
scripts from the frozen configuration in `config/`.

Run date of the results checked in here: **27 September 2026**
(t₀ + 18 days for the primary event).

This run supersedes the 15 September 2026 run. What changed, and why:

| Channel | Action | Effect |
|---|---|---|
| Wikipedia | Re-collected both events | Pageviews now published through **26 Sep 2026** (t₀+17), so the prespecified `[0,+7]` and `[0,+14]` attention windows are observable for the first time. 2023 series are byte-identical; the only change to the 2023 payload is one new redirect alias (`Jacob Coxon`) appearing in the candidate log, which does not change any chosen title. |
| Markets | Re-collected both events | Last session **25 Sep 2026**, so the prespecified `[-1,+1]`, `[-3,+3]` and `[0,+5]` windows are complete for 2026. Same source (FMP) as the previous run; 2023 CARs reproduce to within 0.01 pp. |
| News | Re-collected **warning2026 only** | Window extended from 14 Sep to 27 Sep. `statement2023` was deliberately *not* re-queried: its window closed in June 2023, and Google News RSS returns a fresh ranked sample on every request, so re-querying would inject sampling noise into the replication event for no gain. Its derived outputs reproduce exactly (754 stories, 385 outlets, identical frame prevalence), which also confirms the classifier is deterministic on unchanged input. |
| Attention / markets / outputs | Fully re-run | All `[0,+3]` estimates and the entire robustness battery reproduce **exactly**. |

The one substantive shift is RQ4. Because the 2026 news corpus is a fresh ranked
draw, shared-day frame shares moved by a mean absolute difference of .02–.04,
which moved four *secondary* frame→family pairs across the q<.05 threshold. The
two theoretically matched pairs (existential→threat, infrastructure→infra­structure)
are unchanged in direction and significance. This instability is now stated in the
manuscript rather than left implicit.

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
| ~~2026 attention windows `[0,+7]`, `[0,+14]`~~ | **Resolved in this run.** Pageviews are published through 26 Sep 2026 (t₀+17); both windows are now computed for both events. |
| ~~2026 market windows `[-1,+1]`, `[-3,+3]`, `[0,+5]`~~ | **Resolved in this run.** Sessions through 25 Sep 2026 are available; all three windows are complete. |
| 2023 event-named pages | *Statement on AI Risk*, *Pause Giant AI Experiments*, *Dario Amodei*, *Existential risk studies* had no pre-event baseline in May 2023 — an article created *by* the event cannot have a counterfactual. Dropped for 2023 under the preregistered "insufficient baseline observations" rule and logged in `data/derived/page_resolution_log_statement2023.json`. |
| Human-coded frame validation | Sample exported; coding not performed. |

Family composition still differs between events for that reason, so the RQ5
cross-event comparison remains headlined at `[0,+3]` — but it is no longer
*restricted* to it. `_rq5_comparison` now carries `by_window` with every
prespecified window both events support (`[0,+3]`, `[0,+7]`, `[0,+14]`);
`events` still holds the `[0,+3]` snapshot so existing consumers are unaffected.

`[0,+3]` stays primary because negative-control drift accumulates with window
length: in 2023 it runs −3.15 → −10.8 → −24.7 across the three windows, so the
drift adjustment does progressively more of the work at longer horizons.

## What the checks caught

**Historical page titles.** Pageviews are recorded against the *requested*
title, and resolving a preregistered title to its *current* name is wrong for a
historical event: in May 2023 "Existential risk from artificial intelligence"
lived at "Existential risk from artificial general intelligence" and had
essentially no traffic under today's name. `collect_wikipedia.py` resolves each
title as of the event date, choosing among redirect aliases on **pre-event
baseline information only** — coverage as a gate, baseline volume as the
selector — so no post-event outcome informs the choice.

> ### ✅ RESOLVED: the as-of-event tie-break now selects on baseline volume
>
> `baseline_coverage` scores an alias by the *share of baseline days with at
> least one view*. That measure saturates at 1.0 for any alias with steady
> traffic, so it could not rank the aliases that mattered most. For the 2023
> threat page, three aliases tied at coverage 1.0 with very different volumes:
>
> | Alias | Baseline coverage | Baseline mean views/day |
> |---|---|---|
> | Existential risk from advanced artificial intelligence | 1.00 | 7.5 ← previously chosen |
> | Orthogonality thesis | 1.00 | 11.2 |
> | Existential risk from artificial general intelligence | 1.00 | **1093.9** ← now chosen |
>
> The old winner was decided by `if c > best_cover` (strictly greater), so the
> first alias the MediaWiki API returned won every tie — API ordering, not a
> criterion. The pipeline was modelling a ~7 views/day redirect in place of the
> article that actually held the topic's traffic in May 2023.
>
> **Fix.** `alias_rank(cover, volume)` now ranks candidates as
> `(clears coverage gate, coverage if not gated, baseline volume)`. Coverage is
> a *gate* — was this title live through the baseline? — and volume is the
> *selector* among titles that clear it. Both read only pre-event days, so the
> preregistered "no post-event outcome informs the choice" constraint still
> holds. `page_resolution_log_*.json` now records `baseline_mean_views`
> alongside `coverage` for every candidate, so the choice is auditable.
>
> Two pages change title under the new rule:
>
> | Event | Requested title | Was | Now |
> |---|---|---|---|
> | statement2023 | Existential risk from artificial intelligence | Existential risk from advanced artificial intelligence (7.5/day) | **Existential risk from artificial general intelligence** (1094/day) |
> | warning2026 | Statement on AI Risk | Statement on AI risk of extinction | **Statement on AI Risk** (39/day) |
>
> No page counts change: 48 modelled for 2023 and 52 for 2026, with the same
> five and one documented drops respectively.
>
> **Effect, and it is narrowly contained.** Only the two affected families move;
> every other family shifts by less than 0.05 SD-days (numerical noise), and the
> negative-control drift, placebo design, and market channel are untouched.
>
> | Quantity | Was | Now |
> |---|---|---|
> | 2023 threat `[0,+3]` | 10.35 | **13.95** |
> | 2023 threat `[0,+7]` | 14.60 | **20.20** |
> | 2023 threat `[0,+14]` | 16.30 | **22.80** |
> | 2026 safety `[0,+3]` | 20.18 | **17.89** |
> | 2026 safety `[0,+7]` | 41.30 | **36.94** |
> | 2026 safety `[0,+14]` | 61.00 | **53.52** |
> | 2023 TPG/day `[0,+3]` | 2.07 | **2.97** |
> | 2026 TPG/day `[0,+3]` | 7.77 | **7.96** |
> | TPG ratio 2026:2023 | 3.8x | **2.7x** |
>
> The direction is the expected one. Modelling the real 2023 article raises the
> 2023 threat response, and a higher-volume *Statement on AI Risk* series lowers
> the 2026 safety response; both narrow the cross-event TPG difference. The
> paper's substantive ordering is unchanged: threat attention still dominates
> preparedness attention in both episodes, 2026 far more than 2023, and the
> 2026 threat-to-infrastructure ratio is untouched at 11.3, 12.5 and 10.4 across
> the three windows. Two knock-on refits: the 2023 placebo band for threat rises
> from 3.9 to 5.3 (13.95 still clears it), and `existential->threat` in the 2023
> distributed-lag model rises from 8.1 to 10.9.

**API key never reaches disk.** `_get` embedded the full request URL — including
`apikey=` — in its `FetchError`, and `collect_markets.main` stores that string in
`failed`, which is written to `data/raw/markets.json` and the run manifest, both
committed. A single FMP failure would therefore have published the key. `_get`
now routes the message through `_redact`. Verified: no key material in `data/`,
`outputs/` or `src/` after this run.

**Figure 1 compared unequal horizons.** Both events were drawn over a fixed
`[-21, +21]` day window, but the 2023 window is closed while the 2026 episode
was published only to t0+17. The 2023 series therefore ran 4 days further than
the 2026 series in every panel, which reads as "2026 stopped" rather than "2026
is not published yet". `common_horizon()` now cuts both series to the range both
events observe, so the panels compare equal spans.

**Figure 4 outgrew the page.** With all four windows available for both events,
the single-column CAR figure reached 24 rows and an aspect ratio of about 1:1.7,
taller than a journal page at `0.89\linewidth`. `figure_market` now draws one
panel per event on a shared x-axis, which halves the height and keeps the
cross-event magnitude difference readable as horizontal extent.

**Baseline drift in 2023.** The negative-control pages show a cumulative
abnormal attention of −3.15 at `[0,+3]` and −24.67 at `[0,+14]` (p < 0.01) for
the 2023 event — they should sit at zero. This is genuine late-May seasonality
in the control set, not a coding error: raw views fall 0–11% post-event (Everest
climbing season peaks just before t₀; Photosynthesis and Roman Empire follow the
school year). The 2026 event shows no significant drift at `[0,+3]` (+1.86, p = 0.30), though
the point estimate grows with window length (+5.47 at `[0,+7]`, +9.15 at
`[0,+14]`, neither significant); subtracting it leaves control-adjusted
infrastructure attention at only 0.6 SD-days over both longer windows, which is
why the manuscript reads infrastructure attention beyond `[0,+3]` as not
reliably separable from drift. Every family
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
states research questions rather than directional hypotheses. Every
prespecified window is now evaluated, but the frame classifier still has not
completed human validation, so RQ4 remains provisional.

### Manuscript wiring fixed in this run

* `figure_5_divergence.pdf` was generated by the pipeline and listed here as a
  manuscript figure, but `main(1).tex` never included it. It is now placed in
  the RQ5 section and cross-referenced.
* Figures 3 and 4 were included as raster `.png`; all five now use the vector
  `.pdf`.
* `main(1).tex` referenced figures bare (`{figure_1_attention.pdf}`) while the
  pipeline writes them to `figures/`. A `\graphicspath{{figures/}{./}}` is now
  in the preamble, so the project builds whether the figures sit beside
  `main.tex` or in `figures/`.

On Overleaf, rename `references(1).bib` to `references.bib` to match
`\bibliography{references}`.
