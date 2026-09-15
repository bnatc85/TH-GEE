"""Estimate abnormal attention, family scores, and the threat-preparedness gap.

Implements paper Sec. 6.1 (Eqs. 1-5) and the primary tests of Sec. 7.1, plus
the robustness and falsification battery of Sec. 7.2.

Counterfactual. For each page the pre-event interval alone identifies a model
with a page-specific level and linear trend, day-of-week effects, and AR(1)
errors. Fitting on pre-event data only is what makes the post-event prediction a
counterfactual rather than a smoother:

    Y_pt = log(1 + V_pt)
    Y_pt = a_p + b_p*t + dow_t'g_p + u_pt,   u_pt = rho_p*u_p,t-1 + e_pt

Standardising by the pre-event residual SD (Eq. 3) keeps a high-volume page such
as *Artificial intelligence* from mechanically dominating its family, and makes
scores comparable across pages and across the two events.

Baseline placement. The baseline is ``pre_event_days`` long and ends
``baseline_gap_days`` + 1 days before t0 (events.yml). For the 2026 event the
warning began on t0 - 1, so the gap is 1 and the baseline is [-57, -2]; the
announcement day is then forecast out of sample like every later day, and the
AR(1) forecast for t0 conditions on the last *uncontaminated* residual rather
than on the announcement day's. For 2023 the gap is 0 and the baseline is
[-56, -1].

Usage:
    python src/analyze_attention.py
    python src/analyze_attention.py --robustness
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (DERIVED, RAW, baseline_gap, baseline_label, load_events,
                    load_pages, save_json, write_manifest)

warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402

BOOTSTRAP = 5000
RNG_SEED = 20260909


# ------------------------------------------------------------------ loading
def load_event_data(event_id: str) -> dict:
    path = RAW / f"wikipedia_{event_id}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run src/collect_wikipedia.py first")
    return json.loads(path.read_text())


def to_frame(payload: dict) -> pd.DataFrame:
    """Wide frame of daily views: rows = date, columns = requested page title."""
    df = pd.DataFrame(payload["series"])
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ------------------------------------------------------- counterfactual model
def design_matrix(idx: pd.DatetimeIndex, t0: pd.Timestamp,
                  baseline_days: int, trend: bool = True) -> pd.DataFrame:
    """Constant, linear trend, and six day-of-week dummies (Monday omitted).

    The trend is expressed in baseline-lengths rather than days so that its
    column has unit scale; an unscaled day counter spanning [-56, +28] is badly
    conditioned against a constant and makes the MLE fail to converge.
    """
    X = pd.DataFrame(index=idx)
    X["const"] = 1.0
    if trend:
        X["trend"] = (idx - t0).days.astype(float) / float(baseline_days)
    dow = pd.get_dummies(idx.dayofweek, prefix="dow", drop_first=True).astype(float)
    dow.index = idx
    return pd.concat([X, dow], axis=1)


def abnormal_attention(series: pd.Series, t0: pd.Timestamp,
                       baseline_days: int, trend: bool = True,
                       gap: int = 0) -> tuple[pd.Series, dict]:
    """Standardised abnormal attention A_pt for one page (Eqs. 1-3).

    Returns the full-window A_pt series (NaN where the source has no
    observation) and diagnostics from the pre-event fit.

    ``gap`` is the number of days immediately before t0 excluded from the
    baseline. The fit uses ``[t0 - gap - baseline_days, t0 - gap)``; every
    day from ``t0 - gap`` onward is forecast out of sample.

    ``trend=False`` drops the page-specific linear trend. The prespecified model
    includes it, but a trend estimated on 56 days and extrapolated 14 days ahead
    drifts; the negative-control pages measure that drift directly, so both
    specifications are reported.
    """
    y = np.log1p(series.astype(float))
    pre_end = t0 - pd.Timedelta(days=gap)          # exclusive
    pre_start = pre_end - pd.Timedelta(days=baseline_days)
    pre = y[(y.index >= pre_start) & (y.index < pre_end)].dropna()
    if len(pre) < 21:
        return pd.Series(np.nan, index=series.index), {"status": "insufficient baseline"}

    X = design_matrix(y.index, t0, baseline_days, trend=trend)
    try:
        model = SARIMAX(pre.values, exog=X.loc[pre.index].values,
                        order=(1, 0, 0), trend="n",
                        enforce_stationarity=False)
        fit = model.fit(disp=False, maxiter=500, method="lbfgs")
        if not fit.mle_retvals.get("converged", True):
            fit = model.fit(disp=False, maxiter=2000, method="nm",
                            start_params=fit.params)
    except Exception as exc:                                # noqa: BLE001
        return pd.Series(np.nan, index=series.index), {"status": f"fit failed: {exc}"}
    converged = bool(fit.mle_retvals.get("converged", True))

    # In-sample counterfactual for the baseline days, out-of-sample forecast
    # for everything after the baseline: the gap days (if any), the event day,
    # and the post-event days.
    post_idx = y.index[y.index >= pre_end]
    pred_pre = pd.Series(fit.fittedvalues, index=pre.index)
    yhat = pd.Series(np.nan, index=y.index, dtype=float)
    yhat.loc[pre.index] = pred_pre.values
    if len(post_idx):
        fc = fit.get_forecast(steps=len(post_idx),
                              exog=X.loc[post_idx].values).predicted_mean
        yhat.loc[post_idx] = np.asarray(fc)

    resid = pre - pred_pre
    sigma = float(resid.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return pd.Series(np.nan, index=series.index), {"status": "degenerate sigma"}

    A = (y - yhat) / sigma
    A[y.isna()] = np.nan          # never impute an unobserved day
    return A, {
        "status": "ok", "sigma": sigma, "n_pre": len(pre),
        "rho": float(fit.params[-1]) if len(fit.params) else np.nan,
        "pre_mean_views": float(series.loc[pre.index].mean()),
        "converged": converged,
    }


# ------------------------------------------------------------- family scores
def family_scores(A: pd.DataFrame, families: dict[str, list[str]]) -> pd.DataFrame:
    """Equal-weighted family abnormal attention FAA_kt (Eq. 2)."""
    out = {}
    for fam, pages in families.items():
        cols = [p for p in pages if p in A.columns]
        out[fam] = A[cols].mean(axis=1, skipna=True) if cols else np.nan
    return pd.DataFrame(out)


def tpg_series(faa: pd.DataFrame, cfg: dict) -> pd.Series:
    """Threat-preparedness gap (Eq. 4)."""
    prep = [f for f in cfg["tpg"]["preparedness_families"] if f in faa.columns]
    return faa[cfg["tpg"]["threat_family"]] - faa[prep].mean(axis=1)


def orientation_vector(faa: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Response-orientation vector R_t = (C_t, L_t, P_t) (Eq. 5)."""
    o = cfg["orientation"]
    return pd.DataFrame({
        "curiosity": faa[[f for f in o["curiosity"] if f in faa.columns]].mean(axis=1),
        "alarm": faa[[f for f in o["alarm"] if f in faa.columns]].mean(axis=1),
        "preparedness": faa[[f for f in o["preparedness"] if f in faa.columns]].mean(axis=1),
    })


# ----------------------------------------------------------------- inference
def window_mask(idx: pd.DatetimeIndex, t0: pd.Timestamp, w: tuple[int, int]):
    lo, hi = t0 + pd.Timedelta(days=w[0]), t0 + pd.Timedelta(days=w[1])
    return (idx >= lo) & (idx <= hi)


def page_caa(A: pd.DataFrame, t0: pd.Timestamp, w: tuple[int, int]) -> pd.Series:
    """Cumulative abnormal attention per page over a window.

    Returns NaN for a page whose window is not fully observed, so a truncated
    window cannot masquerade as a smaller effect.
    """
    m = window_mask(A.index, t0, w)
    sub = A.loc[m]
    expected = (w[1] - w[0]) + 1
    if sub.shape[0] < expected:
        return pd.Series(np.nan, index=A.columns)
    return sub.sum(axis=0, skipna=False)


def cluster_bootstrap(values_by_page: pd.Series, rng: np.random.Generator,
                      b: int = BOOTSTRAP) -> dict:
    """Page-clustered bootstrap of a family mean (paper Sec. 7.1 item 3)."""
    v = values_by_page.dropna().values
    if len(v) == 0:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_two_sided": np.nan, "n_pages": 0}
    draws = rng.choice(v, size=(b, len(v)), replace=True).mean(axis=1)
    est = float(v.mean())
    # Two-sided bootstrap p-value for H0: mean = 0, by null-centring.
    centred = draws - est
    p = float((np.abs(centred) >= abs(est)).mean())
    return {"estimate": est,
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)),
            "p_two_sided": p, "n_pages": int(len(v))}


def contrast_bootstrap(a: pd.Series, b_: pd.Series, rng: np.random.Generator,
                       b: int = BOOTSTRAP) -> dict:
    """Bootstrap of mean(a) - mean(b), resampling pages within each group."""
    va, vb = a.dropna().values, b_.dropna().values
    if len(va) == 0 or len(vb) == 0:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_two_sided": np.nan}
    da = rng.choice(va, size=(b, len(va)), replace=True).mean(axis=1)
    db = rng.choice(vb, size=(b, len(vb)), replace=True).mean(axis=1)
    diff = da - db
    est = float(va.mean() - vb.mean())
    p = float((np.abs(diff - est) >= abs(est)).mean())
    return {"estimate": est, "ci_low": float(np.percentile(diff, 2.5)),
            "ci_high": float(np.percentile(diff, 97.5)), "p_two_sided": p,
            "n_pages_a": int(len(va)), "n_pages_b": int(len(vb))}


# --------------------------------------------------------------- driver
def analyse_event(event: dict, pages_cfg: dict, analysis: dict,
                  baseline_days: int | None = None,
                  t0_override: date | None = None,
                  drop_pages: set[str] | None = None,
                  page_filter=None, trend: bool = True,
                  caa_window: tuple[int, int] = (0, 3),
                  gap_override: int | None = None) -> dict:
    payload = load_event_data(event["id"])
    views = to_frame(payload)
    t0 = pd.Timestamp(t0_override or event["t0"])
    baseline_days = baseline_days or analysis["pre_event_days"]
    gap = baseline_gap(event) if gap_override is None else int(gap_override)
    drop_pages = drop_pages or set()

    families = {fam: [p for p in spec["pages"]
                      if p in views.columns and p not in drop_pages]
                for fam, spec in pages_cfg["families"].items()}
    if page_filter is not None:
        families = {k: [p for p in v if page_filter(k, p)] for k, v in families.items()}

    modelled, diagnostics = {}, {}
    wanted = sorted({p for ps in families.values() for p in ps}
                    | {p for p in pages_cfg["negative_controls"] if p in views.columns})
    for page in wanted:
        A, diag = abnormal_attention(views[page], t0, baseline_days,
                                     trend=trend, gap=gap)
        diagnostics[page] = diag
        if diag["status"] == "ok":
            modelled[page] = A
    A = pd.DataFrame(modelled)

    # A page that failed to fit is excluded from its family, and recorded.
    families = {k: [p for p in v if p in A.columns] for k, v in families.items()}
    faa = family_scores(A, families)
    tpg = tpg_series(faa, pages_cfg)
    rvec = orientation_vector(faa, pages_cfg)

    published_through = pd.Timestamp(payload["published_through"])
    rng = np.random.default_rng(RNG_SEED)

    results = {
        "event_id": event["id"], "label": event["label"], "role": event["role"],
        "t0": str(t0.date()), "baseline_days": baseline_days,
        "baseline_gap_days": gap,
        "baseline_window": baseline_label(baseline_days, gap),
        "baseline_dates": [str((t0 - pd.Timedelta(days=gap + baseline_days)).date()),
                           str((t0 - pd.Timedelta(days=gap + 1)).date())],
        "published_through": str(published_through.date()),
        "families": {k: v for k, v in families.items()},
        "n_pages_modelled": int(A.shape[1]),
        "pages_failed": {p: d["status"] for p, d in diagnostics.items()
                         if d["status"] != "ok"},
        "windows": {},
    }

    for w in analysis["effect_windows"]:
        w = tuple(w)
        hi_date = t0 + pd.Timedelta(days=w[1])
        available = hi_date <= published_through
        wkey = f"[{w[0]},+{w[1]}]"
        entry = {"available": bool(available),
                 "last_day_required": str(hi_date.date())}
        if not available:
            entry["reason"] = ("window extends beyond the last day published by "
                               "the pageviews API")
            results["windows"][wkey] = entry
            continue

        caa = page_caa(A, t0, w)
        entry["families"] = {}
        for fam, pages in families.items():
            entry["families"][fam] = cluster_bootstrap(caa[pages], rng)

        # H2 contrast: threat + capabilities vs. the three preparedness families.
        threat_cap = [p for f in ("threat", "capabilities") for p in families[f]]
        prep = [p for f in pages_cfg["tpg"]["preparedness_families"] for p in families[f]]
        entry["h2_threat_capability_vs_preparedness"] = contrast_bootstrap(
            caa[threat_cap], caa[prep], rng)

        # Narrow TPG contrast: threat family alone vs. preparedness pages.
        entry["tpg_contrast"] = contrast_bootstrap(
            caa[families["threat"]], caa[prep], rng)
        entry["tpg_mean_daily"] = float(tpg.loc[window_mask(tpg.index, t0, w)].mean())

        entry["orientation"] = {
            k: float(rvec[k].loc[window_mask(rvec.index, t0, w)].mean())
            for k in rvec.columns}

        controls = [p for p in pages_cfg["negative_controls"] if p in A.columns]
        entry["negative_controls"] = cluster_bootstrap(caa[controls], rng)

        # Negative controls should sit at zero. Any systematic departure is
        # counterfactual drift common to all pages, so each family is also
        # reported net of it (paper Sec. 7.2 item 8).
        drift = entry["negative_controls"]["estimate"]
        entry["control_adjusted_families"] = {
            fam: cluster_bootstrap(caa[pages] - drift, rng)
            for fam, pages in families.items()}
        entry["drift_estimate"] = drift
        results["windows"][wkey] = entry

    # Daily series exported for figures and the distributed-lag model.
    results["daily"] = {
        "dates": [d.date().isoformat() for d in faa.index],
        "faa": {k: [None if pd.isna(x) else float(x) for x in faa[k]] for k in faa.columns},
        "tpg": [None if pd.isna(x) else float(x) for x in tpg],
        "orientation": {k: [None if pd.isna(x) else float(x) for x in rvec[k]]
                        for k in rvec.columns},
    }
    results["page_level"] = {
        p: {"sigma": diagnostics[p].get("sigma"),
            "pre_mean_views": diagnostics[p].get("pre_mean_views"),
            "peak_A": (None if A[p].isna().all() else float(A[p].max())),
            "peak_day": (None if A[p].isna().all()
                         else str(A[p].idxmax().date()))}
        for p in A.columns}

    # Page-level CAA over the common window, kept so that membership-only
    # robustness checks (leave-one-out, page exclusions) can regroup these
    # scores instead of refitting every page's model.
    caa_common = page_caa(A, t0, caa_window)
    results["_caa_0_3"] = {p: _f(caa_common.get(p)) for p in A.columns}
    return results


def _f(v):
    return None if v is None or pd.isna(v) else float(v)


def family_means(caa: dict, families: dict[str, list[str]]) -> dict:
    """Family means from precomputed page-level CAA (no model refitting)."""
    out = {}
    for fam, pages in families.items():
        vals = [caa[p] for p in pages if caa.get(p) is not None]
        if vals:
            out[fam] = float(np.mean(vals))
    return out


# --------------------------------------------- media -> attention (Eq. 6, H3)
def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH step-up adjusted q-values (paper Sec. 7.1 item 4)."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if not ok.any():
        return list(q)
    idx = np.argsort(p[ok])
    ranked = p[ok][idx]
    m = len(ranked)
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[idx] = np.clip(adj, 0, 1)
    q[ok] = out
    return list(q)


def distributed_lag(event_id: str, faa_daily: dict, pages_cfg: dict) -> dict:
    """Frame-to-family distributed-lag associations with HAC standard errors.

    Estimates Eq. 6 for each prespecified (frame, family) pair. The reported
    quantity is the cumulative lag-0..3 association, which is the object H3
    makes a claim about; individual lag coefficients are also returned.
    """
    import statsmodels.api as sm

    news_path = DERIVED / f"news_daily_{event_id}.json"
    if not news_path.exists():
        return {"available": False, "reason": "no classified news for this event"}
    news = pd.DataFrame(json.loads(news_path.read_text()))
    if news.empty:
        return {"available": False, "reason": "empty news corpus"}
    news["date"] = pd.to_datetime(news["date"])
    news = news.set_index("date").sort_index()

    faa = pd.DataFrame(faa_daily["faa"])
    faa.index = pd.to_datetime(faa_daily["dates"])

    results, pairs = {}, []
    for frame, family in pages_cfg["frame_family_map"].items():
        if frame not in news.columns or family not in faa.columns:
            continue
        df = pd.concat([faa[family].rename("y"), news[frame].rename("m"),
                        news["n_stories"].rename("n")], axis=1).dropna()
        if len(df) < 30:
            continue
        X = pd.DataFrame(index=df.index)
        for lag in range(4):
            X[f"m_lag{lag}"] = df["m"].shift(lag)
        X["log_volume"] = np.log1p(df["n"])
        dow = pd.get_dummies(df.index.dayofweek, prefix="dow",
                             drop_first=True).astype(float)
        dow.index = df.index
        X = pd.concat([X, dow], axis=1).dropna()
        y = df["y"].loc[X.index]
        if len(X) < 30:
            continue

        model = sm.OLS(y.values, sm.add_constant(X.values, has_constant="add"))
        fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": 7})
        lag_names = [f"m_lag{i}" for i in range(4)]
        pos = {n: list(X.columns).index(n) + 1 for n in lag_names}
        contrast = np.zeros(len(fit.params))
        for n in lag_names:
            contrast[pos[n]] = 1.0
        cum = float(contrast @ fit.params)
        cum_se = float(np.sqrt(contrast @ fit.cov_params() @ contrast))
        from scipy import stats as _st
        cum_p = float(2 * (1 - _st.norm.cdf(abs(cum / cum_se)))) if cum_se > 0 else np.nan

        results[f"{frame}->{family}"] = {
            "frame": frame, "family": family, "n_obs": int(len(X)),
            "cumulative_lag0_3": cum, "se": cum_se, "p": cum_p,
            "ci95": [cum - 1.96 * cum_se, cum + 1.96 * cum_se],
            "lag_coefficients": {n: float(fit.params[pos[n]]) for n in lag_names},
        }
        pairs.append((f"{frame}->{family}", cum_p))

    qs = benjamini_hochberg([p for _, p in pairs])
    for (key, _), q in zip(pairs, qs):
        results[key]["q_bh"] = None if np.isnan(q) else float(q)
    return {"available": True, "n_pairs": len(results), "pairs": results}


# ------------------------------------------------------------- robustness
def pre_post_ratios(event: dict, pages_cfg: dict, analysis: dict) -> dict:
    """Robustness 1: raw median and mean pre/post view ratios, model-free."""
    payload = load_event_data(event["id"])
    views = to_frame(payload)
    t0 = pd.Timestamp(event["t0"])
    pre_end = t0 - pd.Timedelta(days=baseline_gap(event))   # exclusive
    pub = pd.Timestamp(payload["published_through"])
    hi = min(t0 + pd.Timedelta(days=3), pub)
    pre = views[(views.index >= pre_end - pd.Timedelta(days=28)) & (views.index < pre_end)]
    post = views[(views.index >= t0) & (views.index <= hi)]

    out = {}
    for fam, spec in pages_cfg["families"].items():
        cols = [p for p in spec["pages"] if p in views.columns]
        if not cols:
            continue
        ratios = (post[cols].mean() / pre[cols].mean()).replace([np.inf], np.nan).dropna()
        out[fam] = {"mean_ratio": float(ratios.mean()),
                    "median_ratio": float(ratios.median()),
                    "n_pages": int(len(ratios))}
    return {"window": [str(t0.date()), str(hi.date())],
            "pre_window": [str((pre_end - pd.Timedelta(days=28)).date()),
                           str((pre_end - pd.Timedelta(days=1)).date())],
            "families": out}


def placebo_dates(event: dict, analysis: dict, placebo_cfg: dict,
                  kill_dates: list[date]) -> list[date]:
    """Deterministic placebo dates inside the pre-event interval (robustness 7)."""
    t0 = event["t0"]
    lo = t0 - timedelta(days=analysis["pre_event_days"])
    # Leave room for a full [0,+3] window and keep a 7-day exclusion zone
    # around t0 and around the known kill-switch intervention dates.
    excl = [t0] + list(kill_dates) + list(event.get("t0_alternatives", []))
    cand = [d for d in pd.date_range(lo + timedelta(days=21),
                                     t0 - timedelta(days=8)).date
            if all(abs((d - e).days) > placebo_cfg["exclusion_days"] for e in excl)]
    if not cand:
        return []
    rng = np.random.default_rng(placebo_cfg["seed"])
    n = min(placebo_cfg["n"], len(cand))
    return [cand[i] for i in sorted(rng.choice(len(cand), n, replace=False))]


def run_robustness(event: dict, pages_cfg: dict, analysis: dict,
                   events_cfg: dict) -> dict:
    """The Sec. 7.2 battery. Each entry states what it varies and what it found."""
    out = {}
    base_w = (0, 3)     # the only window available for both events

    out["1_pre_post_ratios"] = pre_post_ratios(event, pages_cfg, analysis)

    out["2_alt_baselines"] = {}
    for b in analysis["alt_baselines"]:
        r = analyse_event(event, pages_cfg, analysis, baseline_days=b)
        w = r["windows"].get("[0,+3]", {})
        out["2_alt_baselines"][f"{b}d"] = {
            fam: s["estimate"] for fam, s in w.get("families", {}).items()}

    out["3_alt_event_dates"] = {}
    for alt in event.get("t0_alternatives", []):
        r = analyse_event(event, pages_cfg, analysis, t0_override=alt)
        w = r["windows"].get("[0,+3]", {})
        out["3_alt_event_dates"][str(alt)] = {
            fam: s["estimate"] for fam, s in w.get("families", {}).items()}

    base = analyse_event(event, pages_cfg, analysis)
    caa = base["_caa_0_3"]
    fams = base["families"]

    excl = set(pages_cfg["event_named_pages"])
    out["4_exclude_event_named_pages"] = {
        "dropped": sorted(excl),
        "families": family_means(caa, {f: [p for p in ps if p not in excl]
                                       for f, ps in fams.items()})}

    # 5: leave-one-page-out, reported as the range of family estimates.
    loo = {}
    full = family_means(caa, fams)
    for fam, pages in fams.items():
        if len(pages) < 2:
            continue
        vals = {}
        for drop in pages:
            m = family_means(caa, {fam: [p for p in pages if p != drop]})
            if fam in m:
                vals[drop] = m[fam]
        if vals and fam in full:
            loo[fam] = {"min": min(vals.values()), "max": max(vals.values()),
                        "full": full[fam],
                        "most_influential_page": max(
                            vals, key=lambda k: abs(vals[k] - full[fam]))}
    out["5_leave_one_page_out"] = loo

    out["6_structural_time_series"] = structural_counterfactual(event, pages_cfg,
                                                                analysis)

    kill_dates = [k["date"] if isinstance(k["date"], date)
                  else date.fromisoformat(k["date"])
                  for k in events_cfg["kill_switch_interventions"]]
    pl = placebo_dates(event, analysis, events_cfg["placebo"], kill_dates)
    pl_res = []
    for d in pl:
        rr = analyse_event(event, pages_cfg, analysis, t0_override=d)
        w = rr["windows"].get("[0,+3]", {}).get("families", {})
        if w:
            pl_res.append({"date": str(d),
                           **{fam: s["estimate"] for fam, s in w.items()}})
    out["7_placebo_dates"] = {
        "n": len(pl_res), "dates": [p["date"] for p in pl_res],
        "results": pl_res,
        "summary": {fam: {
            "mean": float(np.mean([p[fam] for p in pl_res])),
            "p95_abs": float(np.percentile([abs(p[fam]) for p in pl_res], 95)),
            "observed": full.get(fam),
            # A real effect should sit outside the spread of pseudo-events
            # drawn from the same pre-event interval.
            "observed_exceeds_placebo_p95": (
                None if full.get(fam) is None else
                bool(abs(full[fam]) > np.percentile(
                    [abs(p[fam]) for p in pl_res], 95))),
        } for fam in (pl_res[0].keys() - {"date"})} if pl_res else {}}

    out["8_negative_controls"] = base["windows"].get("[0,+3]", {}).get(
        "negative_controls", {})

    r = analyse_event(event, pages_cfg, analysis, caa_window=(1, 3))
    out["9_exclude_first_day"] = {
        "note": ("cumulated over [+1,+3] instead of [0,+3]; tests persistence "
                 "beyond the day of immediate news exposure"),
        "families": family_means(r["_caa_0_3"], r["families"])}

    out["10_11_market_specifications"] = {
        "note": ("Alternative market benchmarks are run in analyze_markets.py "
                 "(broad_spy and sector_adjusted specifications). No "
                 "earnings-calendar source is available in this environment, "
                 "so analyze_markets.py instead flags securities showing both "
                 "heavily elevated volume and a large abnormal return inside "
                 "the event window and reports portfolio effects with and "
                 "without them (see market_results.json).")}

    ks_page = pages_cfg["killswitch_page"]
    out["12_killswitch"] = {
        "control_with_kill_switch": full.get("control"),
        "control_without_kill_switch": family_means(
            caa, {"control": [p for p in fams.get("control", []) if p != ks_page]}
        ).get("control"),
        "kill_switch_page_caa": caa.get(ks_page),
        "intervention_dates_modelled": [str(d) for d in kill_dates],
        "kill_switch_page_alone": base["page_level"].get(ks_page),
    }

    out["13_traffic_normalised"] = traffic_normalised(event, pages_cfg, analysis)

    # 14: baseline placement. For an event whose baseline stops short of t0
    # (baseline_gap_days > 0), refit with the baseline run right up to t0 - 1
    # so the announcement day is treated as normal. The difference measures how
    # much the excluded day would have contaminated the counterfactual.
    gap = baseline_gap(event)
    if gap > 0:
        r = analyse_event(event, pages_cfg, analysis, gap_override=0)
        w = r["windows"].get("[0,+3]", {})
        out["14_baseline_gap"] = {
            "prespecified_gap_days": gap,
            "prespecified_baseline": baseline_label(analysis["pre_event_days"], gap),
            "alternative_baseline": baseline_label(analysis["pre_event_days"], 0),
            "families_with_prespecified_gap": full,
            "families_without_gap": {
                fam: s["estimate"] for fam, s in w.get("families", {}).items()},
            "negative_controls_without_gap": w.get("negative_controls", {}).get("estimate"),
        }
    else:
        out["14_baseline_gap"] = {"note": "baseline_gap_days is 0; no alternative"}
    return out


def structural_counterfactual(event: dict, pages_cfg: dict,
                              analysis: dict) -> dict:
    """Robustness 6: local-level structural time-series counterfactual.

    The paper names CausalImpact; that package is not available here, so the
    same idea is fitted directly with a local-level + seasonal unobserved-
    components model, trained on pre-event data and forecast forward.
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    payload = load_event_data(event["id"])
    views = to_frame(payload)
    t0 = pd.Timestamp(event["t0"])
    gap = baseline_gap(event)
    pre_end = t0 - pd.Timedelta(days=gap)                  # exclusive
    pub = pd.Timestamp(payload["published_through"])
    hi = min(t0 + pd.Timedelta(days=3), pub)
    if hi < t0 + pd.Timedelta(days=3):
        return {"available": False, "reason": "post window not fully published"}

    fam_scores = {}
    for fam, spec in pages_cfg["families"].items():
        vals = []
        for p in spec["pages"]:
            if p not in views.columns:
                continue
            y = np.log1p(views[p].astype(float))
            pre = y[(y.index >= pre_end - pd.Timedelta(days=analysis["pre_event_days"]))
                    & (y.index < pre_end)].dropna()
            if len(pre) < 30:
                continue
            try:
                m = UnobservedComponents(pre.values, level="local level",
                                         freq_seasonal=[{"period": 7,
                                                         "harmonics": 2}])
                f = m.fit(disp=False, maxiter=200)
                post = y[(y.index >= t0) & (y.index <= hi)].dropna()
                # Forecast from the end of the baseline; the first ``gap``
                # steps cover the excluded days and are discarded.
                pred = f.get_forecast(steps=gap + len(post)).predicted_mean[gap:]
                sigma = float(np.std(pre.values - f.fittedvalues, ddof=1))
                if sigma > 0:
                    vals.append(float(np.sum((post.values - pred) / sigma)))
            except Exception:                                # noqa: BLE001
                continue
        if vals:
            fam_scores[fam] = {"mean_cumulative_A": float(np.mean(vals)),
                               "n_pages": len(vals)}
    return {"available": True, "model": "local level + weekly seasonal",
            "window": "[0,+3]",
            "baseline_window": baseline_label(analysis["pre_event_days"], gap),
            "families": fam_scores}


def traffic_normalised(event: dict, pages_cfg: dict, analysis: dict) -> dict:
    """Normalise family views by total en.wikipedia user traffic (Sec. 7.2).

    Guards against reading a platform-wide traffic change as topic-specific
    attention.
    """
    payload = load_event_data(event["id"])
    agg = payload.get("aggregate") or {}
    if not agg:
        return {"available": False, "reason": "no aggregate traffic collected"}
    views = to_frame(payload)
    total = pd.Series({pd.Timestamp(k): v for k, v in agg.items()}).sort_index()
    t0 = pd.Timestamp(event["t0"])
    pre_end = t0 - pd.Timedelta(days=baseline_gap(event))   # exclusive
    pre_lo = pre_end - pd.Timedelta(days=28)
    pub = pd.Timestamp(payload["published_through"])
    hi = min(t0 + pd.Timedelta(days=3), pub)

    pre_t = total[(total.index >= pre_lo) & (total.index < pre_end)].mean()
    post_t = total[(total.index >= t0) & (total.index <= hi)].mean()
    out = {"aggregate_pre_mean": float(pre_t), "aggregate_post_mean": float(post_t),
           "aggregate_ratio": float(post_t / pre_t),
           "pre_window": [str(pre_lo.date()), str((pre_end - pd.Timedelta(days=1)).date())],
           "families": {}}
    for fam, spec in pages_cfg["families"].items():
        cols = [p for p in spec["pages"] if p in views.columns]
        if not cols:
            continue
        share_pre = (views.loc[(views.index >= pre_lo)
                               & (views.index < pre_end), cols].sum(axis=1)
                     / total.reindex(views.index)[(views.index >= pre_lo)
                                                  & (views.index < pre_end)]).mean()
        share_post = (views.loc[(views.index >= t0) & (views.index <= hi), cols].sum(axis=1)
                      / total.reindex(views.index)[(views.index >= t0)
                                                   & (views.index <= hi)]).mean()
        out["families"][fam] = {"share_ratio": float(share_post / share_pre)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robustness", action="store_true",
                    help="also run the Sec. 7.2 robustness battery")
    ap.add_argument("--media-only", action="store_true",
                    help="recompute only the distributed-lag media model and "
                         "merge it into existing results (cheap; use after the "
                         "news corpus is refreshed)")
    args = ap.parse_args()

    events_cfg, pages_cfg = load_events(), load_pages()
    analysis = events_cfg["analysis"]

    if args.media_only:
        path = DERIVED / "attention_results.json"
        if not path.exists():
            raise SystemExit("no existing results to merge into")
        primary = json.loads(path.read_text())
        for event in events_cfg["events"]:
            eid = event["id"]
            if eid not in primary:
                continue
            primary[eid]["distributed_lag"] = distributed_lag(
                eid, primary[eid]["daily"], pages_cfg)
            print(f"[{eid}] media pairs: "
                  f"{primary[eid]['distributed_lag'].get('n_pairs', 0)}")
        save_json(path, primary)
        write_manifest("analyze_attention_media", events=[e["id"] for e in events_cfg["events"]])
        print(f"updated {path}")
        return

    primary = {}
    for event in events_cfg["events"]:
        print(f"[{event['id']}] modelling pages ...", flush=True)
        res = analyse_event(event, pages_cfg, analysis)
        res["distributed_lag"] = distributed_lag(event["id"], res["daily"], pages_cfg)
        primary[event["id"]] = res
        print(f"  modelled {res['n_pages_modelled']} pages; "
              f"failed: {len(res['pages_failed'])}; "
              f"media pairs: {res['distributed_lag'].get('n_pairs', 0)}", flush=True)

    # RQ5: standardised response-orientation vectors compared across events at
    # the one window both events can support.
    comparison = {}
    common = "[0,+3]"
    for eid, res in primary.items():
        w = res["windows"].get(common, {})
        if w.get("available"):
            comparison[eid] = {
                "orientation": w["orientation"],
                "tpg_mean_daily": w["tpg_mean_daily"],
                "families": {f: s["estimate"] for f, s in w["families"].items()},
                "control_adjusted": {f: s["estimate"] for f, s
                                     in w["control_adjusted_families"].items()},
                "drift": w["drift_estimate"],
            }
    primary["_rq5_comparison"] = {"window": common, "events": comparison}

    if args.robustness:
        for event in events_cfg["events"]:
            print(f"[{event['id']}] robustness battery ...", flush=True)
            primary[event["id"]]["robustness"] = run_robustness(
                event, pages_cfg, analysis, events_cfg)

    save_json(DERIVED / "attention_results.json", primary)
    write_manifest("analyze_attention", seed=RNG_SEED, bootstrap=BOOTSTRAP,
                   robustness=bool(args.robustness),
                   events=[e["id"] for e in events_cfg["events"]])
    print(f"wrote {DERIVED / 'attention_results.json'}")


if __name__ == "__main__":
    main()
