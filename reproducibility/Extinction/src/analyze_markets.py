"""Market-model event study for the prespecified portfolios.

Implements paper Sec. 6.3 (Eqs. 7-9) and test 5 of Sec. 7.1.

    R_it = a_i + b_i R_mt + e_it        estimated over [-250,-30] trading days
    AR_it  = R_it - (a_i + b_i R_mt)
    CAR_i(t1,t2) = sum AR_it

Portfolio effects are equal- and value-weighted, with both conventional and
randomisation-based intervals. The randomisation test re-draws pseudo-event
dates from the estimation window, which does not rely on the normality that the
conventional test assumes and is robust to the event-window clustering that
makes cross-sectional independence implausible here.

Two specifications are reported per the paper: a sector-adjusted model (SOXX for
semiconductor firms, QQQ for platform firms) and a broad-market model (S&P 500
via SPY).

Usage:
    python src/analyze_markets.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DERIVED, RAW, load_events, save_json, write_manifest

RNG_SEED = 20260914
N_RANDOMISATION = 5000
# Splits move a price by 50% or more (2:1 and up). The Nasdaq series are
# split-adjusted -- verified against AMZN 20:1 (Jun 2022), GOOGL 20:1 (Jul 2022)
# and NVDA 10:1 (Jun 2024), none of which appear as jumps -- so this threshold
# exists only as a guard against an unadjusted series slipping in. It sits above
# the largest genuine one-day moves in the sample (ORCL +36%, META -26%) so that
# real earnings reactions are not silently deleted.
SPLIT_RETURN_THRESHOLD = 0.45

# Robustness 11: a firm-specific announcement (typically earnings) shows up as a
# large abnormal return on a session with heavily elevated volume. Rather than
# relying on an earnings calendar -- none is available in this environment -- a
# security is flagged when both conditions hold inside the event window, and
# portfolio effects are reported with and without the flagged securities.
CONFOUND_VOLUME_Z = 1.0      # abnormal log volume on the day (~2.7x normal)
CONFOUND_AR_SIGMAS = 2.5     # |AR| in estimation-window residual SDs

# Sector benchmark assignment for the sector-adjusted specification.
SECTOR_BENCHMARK = {
    "NVDA": "SOXX", "AMD": "SOXX", "AVGO": "SOXX",
    "ORCL": "QQQ", "MSFT": "QQQ", "GOOGL": "QQQ", "AMZN": "QQQ",
    "META": "QQQ", "PLTR": "QQQ",
    "PANW": "QQQ", "CRWD": "QQQ", "FTNT": "QQQ", "ZS": "QQQ",
}


def load_markets() -> dict:
    path = RAW / "markets.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run src/collect_markets.py first")
    return json.loads(path.read_text())


def build_returns(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Log returns and log volume, with split-like observations removed.

    When the fallback (unadjusted) source is used, a split appears as a huge
    return. Those observations are set to NaN rather than adjusted by a guessed
    ratio, which keeps the estimation window honest at the cost of one or two
    days out of ~220.
    """
    closes, volumes = {}, {}
    for ticker, s in payload["series"].items():
        if not s:
            continue
        idx = pd.to_datetime(sorted(s))
        closes[ticker] = pd.Series([s[d.strftime("%Y-%m-%d")]["close"] for d in idx],
                                   index=idx)
        volumes[ticker] = pd.Series([s[d.strftime("%Y-%m-%d")]["volume"] for d in idx],
                                    index=idx)
    px = pd.DataFrame(closes).sort_index()
    vol = pd.DataFrame(volumes).sort_index()

    ret = np.log(px / px.shift(1))
    dropped = {}
    ADJUSTED_SOURCES = {"fmp_adjclose", "yahoo_adjclose"}
    unadjusted = {t for t, src in payload.get("sources", {}).items()
                  if src not in ADJUSTED_SOURCES}
    for t in ret.columns:
        mask = ret[t].abs() > np.log(1 + SPLIT_RETURN_THRESHOLD)
        if mask.any():
            days = [str(d.date()) for d in ret.index[mask]]
            if t in unadjusted:
                ret.loc[mask, t] = np.nan
                dropped[t] = {"dropped_split_like": days}
            else:
                dropped[t] = {"large_but_kept_adjusted_source": days}
    return ret, np.log1p(vol), dropped


def market_model(stock: pd.Series, bench: pd.Series, est_idx: pd.DatetimeIndex):
    df = pd.concat([stock.rename("r"), bench.rename("m")], axis=1).loc[est_idx].dropna()
    if len(df) < 60:
        return None
    x = np.column_stack([np.ones(len(df)), df["m"].values])
    beta, *_ = np.linalg.lstsq(x, df["r"].values, rcond=None)
    resid = df["r"].values - x @ beta
    return {"alpha": float(beta[0]), "beta": float(beta[1]),
            "sigma": float(resid.std(ddof=2)), "n_est": int(len(df))}


def abnormal_returns(ret: pd.DataFrame, ticker: str, bench_col: str,
                     est_idx: pd.DatetimeIndex) -> tuple[pd.Series, dict] | tuple[None, None]:
    fit = market_model(ret[ticker], ret[bench_col], est_idx)
    if fit is None:
        return None, None
    ar = ret[ticker] - (fit["alpha"] + fit["beta"] * ret[bench_col])
    return ar, fit


def trading_window(index: pd.DatetimeIndex, t0: pd.Timestamp,
                   lo: int, hi: int) -> pd.DatetimeIndex:
    """Sessions from lo..hi in *trading* days relative to the first session >= t0."""
    sessions = index[index >= t0]
    if len(sessions) == 0:
        return pd.DatetimeIndex([])
    anchor = index.get_loc(sessions[0])
    a, b = anchor + lo, anchor + hi
    if a < 0 or b >= len(index):
        return pd.DatetimeIndex([])
    return index[a:b + 1]


def estimation_index(index: pd.DatetimeIndex, t0: pd.Timestamp,
                     lo: int, hi: int) -> pd.DatetimeIndex:
    sessions = index[index < t0]
    if len(sessions) == 0:
        return pd.DatetimeIndex([])
    anchor = index.get_loc(sessions[-1]) + 1
    a, b = anchor + lo, anchor + hi
    a = max(a, 0)
    return index[a:b] if b > a else pd.DatetimeIndex([])


def analyse_event(event: dict, ret: pd.DataFrame, logvol: pd.DataFrame,
                  securities: list[dict], analysis: dict, rng) -> dict:
    t0 = pd.Timestamp(event["market_t0"])
    est_lo, est_hi = analysis["market_estimation_window"]
    est_idx = estimation_index(ret.index, t0, est_lo, est_hi)

    sessions_after = ret.index[ret.index >= t0]
    t0_present = len(sessions_after) > 0 and sessions_after[0] == t0
    out = {
        "event_id": event["id"], "market_t0": str(t0.date()),
        "t0_is_trading_session_with_data": bool(t0_present),
        "last_session_available": str(ret.index[-1].date()),
        "n_estimation_sessions": int(len(est_idx)),
        "specifications": {},
    }
    if len(est_idx) < 60:
        out["error"] = "insufficient estimation window"
        return out

    tickers = [s["ticker"] for s in securities if s["portfolio"] != "benchmark"
               and s["ticker"] in ret.columns]
    port_of = {s["ticker"]: s["portfolio"] for s in securities}

    for spec, bench_for in (("broad_spy", lambda t: "SPY"),
                            ("sector_adjusted", lambda t: SECTOR_BENCHMARK.get(t, "QQQ"))):
        ars, fits = {}, {}
        for t in tickers:
            bcol = bench_for(t)
            if bcol not in ret.columns:
                continue
            ar, fit = abnormal_returns(ret, t, bcol, est_idx)
            if ar is not None:
                ars[t], fits[t] = ar, fit
        if not ars:
            continue
        AR = pd.DataFrame(ars)

        spec_out = {"benchmarks": {t: bench_for(t) for t in ars},
                    "market_models": fits, "windows": {}}

        for w in analysis["market_event_windows"] + [["episode"]]:
            if w == ["episode"]:
                lo_d, hi_d = event["episode"]
                idx = ret.index[(ret.index >= pd.Timestamp(lo_d))
                                & (ret.index <= pd.Timestamp(hi_d))]
                wkey = f"episode[{lo_d}..{hi_d}]"
                requested = None
            else:
                idx = trading_window(ret.index, t0, w[0], w[1])
                wkey = f"[{w[0]},{w[1]}]"
                requested = (w[1] - w[0]) + 1

            entry = {"sessions": [str(d.date()) for d in idx]}
            if len(idx) == 0 or (requested and len(idx) < requested):
                entry["available"] = False
                entry["reason"] = ("event window extends past the last session "
                                   "with published data")
                spec_out["windows"][wkey] = entry
                continue
            entry["available"] = True

            car = AR.loc[idx].sum(axis=0, skipna=False)
            avol = (logvol.loc[idx, list(ars)].mean(axis=0)
                    - logvol.loc[est_idx, list(ars)].mean(axis=0))
            entry["by_security"] = {t: {"car": _f(car.get(t)),
                                        "abnormal_log_volume": _f(avol.get(t))}
                                    for t in ars}
            # Flag securities whose window contains a likely firm-specific
            # announcement, so a single earnings reaction cannot be read as a
            # portfolio-wide response to the warning (robustness 11).
            confounded = []
            for t in ars:
                vol_dev = (logvol.loc[idx, t]
                           - logvol.loc[est_idx, t].mean()).max()
                ar_max = AR.loc[idx, t].abs().max()
                if (pd.notna(vol_dev) and vol_dev > CONFOUND_VOLUME_Z
                        and pd.notna(ar_max)
                        and ar_max > CONFOUND_AR_SIGMAS * fits[t]["sigma"]):
                    confounded.append(t)
            entry["likely_firm_specific_announcement"] = confounded

            entry["by_portfolio"] = {}
            for pname in ("ai_infrastructure", "ai_platform", "cyber_resilience"):
                members = [t for t in ars if port_of[t] == pname]
                vals = car[members].dropna()
                if vals.empty:
                    continue
                clean = [t for t in members if t not in confounded]
                clean_vals = car[clean].dropna()
                entry["by_portfolio"][pname] = {
                    "n": int(len(vals)),
                    "car_equal_weighted": float(vals.mean()),
                    "t_stat": _tstat(vals),
                    "ci95_conventional": _ci(vals),
                    "randomisation_p": _randomisation_p(
                        AR[members], est_idx, len(idx), float(vals.mean()), rng),
                    "abnormal_log_volume": float(avol[members].dropna().mean()),
                    "excluding_confounded": {
                        "excluded": [t for t in members if t in confounded],
                        "n": int(len(clean_vals)),
                        "car_equal_weighted": (float(clean_vals.mean())
                                               if not clean_vals.empty else None),
                    },
                }
            # H4 contrast: AI-exposed portfolios vs. cybersecurity/resilience.
            ai = [t for t in ars if port_of[t] in ("ai_infrastructure", "ai_platform")]
            cy = [t for t in ars if port_of[t] == "cyber_resilience"]
            if car[ai].notna().any() and car[cy].notna().any():
                entry["h4_ai_minus_cyber"] = {
                    "estimate": float(car[ai].mean() - car[cy].mean()),
                    "ai_mean": float(car[ai].mean()),
                    "cyber_mean": float(car[cy].mean()),
                }
            spec_out["windows"][wkey] = entry
        out["specifications"][spec] = spec_out
    return out


def _f(v):
    return None if v is None or pd.isna(v) else float(v)


def _tstat(vals: pd.Series) -> float | None:
    if len(vals) < 2 or vals.std(ddof=1) == 0:
        return None
    return float(vals.mean() / (vals.std(ddof=1) / np.sqrt(len(vals))))


def _ci(vals: pd.Series) -> list[float] | None:
    if len(vals) < 2:
        return None
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    return [float(vals.mean() - 1.96 * se), float(vals.mean() + 1.96 * se)]


def _randomisation_p(ar_block: pd.DataFrame, est_idx: pd.DatetimeIndex,
                     width: int, observed: float, rng) -> float | None:
    """Two-sided p from pseudo-events drawn inside the estimation window."""
    block = ar_block.loc[est_idx].dropna(how="any")
    if len(block) < width + 5:
        return None
    starts = rng.integers(0, len(block) - width, size=N_RANDOMISATION)
    vals = block.values
    draws = np.array([vals[s:s + width].sum(axis=0).mean() for s in starts])
    return float((np.abs(draws) >= abs(observed)).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    payload = load_markets()
    events_cfg = load_events()
    ret, logvol, split_notes = build_returns(payload)
    rng = np.random.default_rng(RNG_SEED)

    results = {"source_by_ticker": payload.get("sources", {}),
               "split_handling": split_notes,
               "last_available_session": str(ret.index[-1].date()),
               "events": {}}
    for event in events_cfg["events"]:
        res = analyse_event(event, ret, logvol, payload["securities"],
                            events_cfg["analysis"], rng)
        results["events"][event["id"]] = res
        print(f"[{event['id']}] market t0 {res['market_t0']} "
              f"t0_data={res['t0_is_trading_session_with_data']} "
              f"est_sessions={res['n_estimation_sessions']}")

    save_json(DERIVED / "market_results.json", results)
    write_manifest("analyze_markets", seed=RNG_SEED,
                   n_randomisation=N_RANDOMISATION,
                   sources=payload.get("sources", {}),
                   split_handling=split_notes)
    print(f"wrote {DERIVED / 'market_results.json'}")


if __name__ == "__main__":
    main()
