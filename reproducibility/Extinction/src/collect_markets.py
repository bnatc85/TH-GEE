"""Collect daily prices and volume for the prespecified event-study portfolios.

Implements paper Sec. 5.3. Source order, best first:

1. **Financial Modeling Prep** (the paper's named source), dividend-adjusted
   daily closes. Requires ``FMP_API_KEY`` in the environment -- the key is never
   written to the repository or to any output file.
2. Yahoo Finance chart endpoint (adjusted closes, no key).
3. Nasdaq (split-adjusted but *not* dividend-adjusted) as a last resort.

Adjusted closes matter here: AMZN and GOOGL both split 20:1 inside the 2023
estimation window, and an unadjusted series would put those splits into the
return series as enormous one-day moves.

Whichever source answers is recorded per ticker in the run manifest, so the
adjustment basis of any published estimate is auditable.

Portfolio membership comes from config/portfolios.csv and is never revised after
event-window returns are inspected.

Usage:
    python src/collect_markets.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CONFIG, RAW, FetchError, load_events, save_json, write_manifest

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
# The endpoint rejects non-browser agents, and rate-limits bursts.
BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/121.0 Safari/537.36")
POLITE_DELAY = 4.0
HISTORY_START = date(2022, 1, 1)   # covers [-250,-30] sessions for both events


def load_portfolios() -> list[dict]:
    with open(CONFIG / "portfolios.csv", newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["included"] == "1"]


FMP = "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"
NASDAQ = "https://api.nasdaq.com/api/quote/{symbol}/historical"
# Yahoo answers 429 to bursts and stays blocked for minutes, so back off in
# minutes rather than seconds.
YAHOO_BACKOFF = [0, 30, 90, 180, 300]


def _get(url: str, backoff: list[int] | None = None) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": BROWSER_UA, "Accept": "application/json"})
    last = None
    for wait in (backoff if backoff is not None else YAHOO_BACKOFF):
        if wait:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:                          # noqa: BLE001
            last = exc
    raise FetchError(f"{url}: {last}")


def fetch_symbol_fmp(ticker: str, end: date) -> dict[str, dict]:
    """Dividend- and split-adjusted daily closes from Financial Modeling Prep."""
    key = os.environ.get("FMP_API_KEY", "").strip()
    if not key:
        raise FetchError("FMP_API_KEY not set in environment")
    params = {"symbol": ticker, "from": HISTORY_START.isoformat(),
              "to": end.isoformat(), "apikey": key}
    data = _get(f"{FMP}?{urllib.parse.urlencode(params)}", backoff=[0, 5, 15])
    if isinstance(data, dict):
        # The API reports plan/auth problems as a JSON object, not a list.
        raise FetchError(f"FMP error for {ticker}: "
                         f"{str(data.get('Error Message', data))[:120]}")
    out = {}
    for r in data or []:
        try:
            out[r["date"]] = {"close": float(r["adjClose"]),
                              "volume": float(r.get("volume") or 0.0)}
        except (KeyError, TypeError, ValueError):
            continue
    if not out:
        raise FetchError(f"FMP returned no rows for {ticker}")
    return out


def fetch_symbol_nasdaq(ticker: str, end: date) -> dict[str, dict]:
    """Fallback source. Closes are NOT split/dividend adjusted.

    Used only when Yahoo is unavailable; callers must record the source, since
    an unadjusted series puts a split inside the estimation window straight into
    the return series.
    """
    # Benchmarks are ETFs, which the endpoint serves under a different
    # assetclass; try both rather than assuming.
    rows = []
    for assetclass in ("stocks", "etf"):
        params = {"assetclass": assetclass, "fromdate": HISTORY_START.isoformat(),
                  "todate": end.isoformat(), "limit": "9999"}
        data = _get(f"{NASDAQ.format(symbol=ticker)}?{urllib.parse.urlencode(params)}",
                    backoff=[0, 10, 30])
        rows = (((data.get("data") or {}).get("tradesTable") or {}).get("rows")) or []
        if rows:
            break
    out = {}
    for r in rows:
        try:
            mm, dd, yyyy = r["date"].split("/")
            close = float(r["close"].replace("$", "").replace(",", ""))
            vol = float((r.get("volume") or "0").replace(",", ""))
        except (KeyError, ValueError):
            continue
        out[f"{yyyy}-{mm}-{dd}"] = {"close": close, "volume": vol}
    if not out:
        raise FetchError(f"nasdaq returned no rows for {ticker}")
    return out


def detect_split_like(series: dict[str, dict], threshold: float = 0.35) -> list[str]:
    """Sessions whose absolute simple return exceeds ``threshold``.

    On an adjusted series these are real moves and should be rare; on an
    unadjusted series they flag split dates that would otherwise enter the
    market model as enormous returns.
    """
    days = sorted(series)
    flagged = []
    for prev, cur in zip(days, days[1:]):
        p0, p1 = series[prev]["close"], series[cur]["close"]
        if p0 > 0 and abs(p1 / p0 - 1.0) > threshold:
            flagged.append(cur)
    return flagged


def fetch_symbol(ticker: str, end: date) -> dict[str, dict]:
    """Daily adjusted close and volume keyed by ISO session date."""
    params = {
        "period1": int(datetime.combine(HISTORY_START, datetime.min.time(),
                                        tzinfo=timezone.utc).timestamp()),
        "period2": int(datetime.combine(end + timedelta(days=2), datetime.min.time(),
                                        tzinfo=timezone.utc).timestamp()),
        "interval": "1d", "events": "div,split",
    }
    data = _get(f"{CHART.format(symbol=ticker)}?{urllib.parse.urlencode(params)}")
    result = (data.get("chart") or {}).get("result") or []
    if not result:
        raise FetchError(f"no data for {ticker}: {str(data)[:160]}")
    r = result[0]
    stamps = r.get("timestamp") or []
    quote = (r.get("indicators", {}).get("quote") or [{}])[0]
    adj = (r.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    closes = adj if adj else quote.get("close")
    volumes = quote.get("volume") or [None] * len(stamps)
    # Stamps are session opens in exchange local time; shift by the reported
    # offset so a session is dated to its own trading day, not a UTC boundary.
    offset = r.get("meta", {}).get("gmtoffset", 0) or 0

    out = {}
    for ts, c, v in zip(stamps, closes or [], volumes):
        if c is None:
            continue
        day = datetime.fromtimestamp(ts + offset, tz=timezone.utc).date().isoformat()
        out[day] = {"close": float(c), "volume": float(v or 0.0)}
    if not out:
        raise FetchError(f"empty series for {ticker}")
    return out


def main() -> None:
    events_cfg = load_events()
    rows = load_portfolios()

    today = date.today()

    # Probe sources once, best first. Yahoo blocks by IP for minutes at a time,
    # so retrying it per symbol would cost hours for no new information; one
    # probe decides the source for the whole run.
    candidates = [
        ("fmp_adjclose", fetch_symbol_fmp, 0.2,
         "Financial Modeling Prep (dividend- and split-adjusted)"),
        ("yahoo_adjclose", fetch_symbol, POLITE_DELAY,
         "Yahoo (split/dividend adjusted closes)"),
        ("nasdaq_unadjusted", fetch_symbol_nasdaq, POLITE_DELAY,
         "Nasdaq (split-adjusted, NOT dividend-adjusted)"),
    ]
    chosen = None
    for name, fn, delay, label in candidates:
        try:
            fn("SPY", today)
            chosen = (name, fn, delay)
            print(f"source: {label}")
            break
        except FetchError as exc:
            print(f"  {name} unavailable: {str(exc)[-70:]}")
    if chosen is None:
        raise SystemExit("no market data source available")
    src_name, fetch_fn, delay = chosen

    series, failed, sources, split_flags = {}, {}, {}, {}
    for row in rows:
        t = row["ticker"]
        try:
            series[t] = fetch_fn(t, today)
            sources[t] = src_name
        except FetchError as exc:
            failed[t] = str(exc)
            print(f"  {t:6s} FAILED: {str(exc)[-70:]}")
            time.sleep(delay)
            continue
        days = sorted(series[t])
        flags = detect_split_like(series[t])
        if flags:
            split_flags[t] = flags
        print(f"  {t:6s} {row['portfolio']:18s} {len(days):5d} days "
              f"{days[0]}..{days[-1]} [{sources[t]}]"
              f"{' SPLIT-LIKE:' + ','.join(flags) if flags else ''}")
        time.sleep(delay)

    last_day = max((max(s) for s in series.values() if s), default=None)
    coverage = {}
    for event in events_cfg["events"]:
        mt0 = event["market_t0"]
        hi = mt0 + timedelta(days=events_cfg["analysis"]["market_event_windows"][-1][1] + 4)
        coverage[event["id"]] = {
            "market_t0": mt0.isoformat(),
            "t0_present": any(mt0.isoformat() in s for s in series.values()),
            "last_available": last_day,
            "post_window_complete": bool(last_day and last_day >= hi.isoformat()),
        }

    payload = {"securities": [dict(r) for r in rows], "series": series,
               "failed": failed, "last_available": last_day,
               "sources": sources, "split_like_returns": split_flags,
               "event_coverage": coverage}
    path = save_json(RAW / "markets.json", payload)
    write_manifest("collect_markets", n_symbols=len(series), failed=failed,
                   sources=sources, split_like_returns=split_flags,
                   last_available=last_day, event_coverage=coverage)
    print(f"wrote {path}; last available session {last_day}")
    for eid, c in coverage.items():
        print(f"  {eid}: market t0 {c['market_t0']} present={c['t0_present']} "
              f"post-window complete={c['post_window_complete']}")


if __name__ == "__main__":
    main()
