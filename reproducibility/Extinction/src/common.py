"""Shared configuration loading, HTTP helpers, and run-manifest support.

Every collection and analysis script in this pipeline imports from here so that
the frozen configuration is read the same way everywhere and so that the run
manifest (paper Appendix C: "package versions, query dates, event configuration
hashes, and random seeds") is written from a single implementation.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
OUTPUTS = ROOT / "outputs"

# Wikimedia asks for a descriptive agent with contact details.
USER_AGENT = "TH-GEE-extinction-study/1.0 (https://github.com/USERNAME/REPOSITORY; bnatc85@gmail.com)"

# Sentinel for "the source has no observation for this day", kept distinct from
# a true zero count (paper Sec. 6.3: "No missing pageview day is silently
# converted to zero").
MISSING = None


# ----------------------------------------------------------------- config I/O
def load_yaml(name: str) -> dict:
    with open(CONFIG / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def config_hash(name: str) -> str:
    """SHA-256 of a frozen config file, recorded in the run manifest."""
    return hashlib.sha256((CONFIG / name).read_bytes()).hexdigest()


def load_events() -> dict:
    cfg = load_yaml("events.yml")
    for ev in cfg["events"]:
        for key in ("t0", "market_t0"):
            if isinstance(ev.get(key), str):
                ev[key] = date.fromisoformat(ev[key])
        ev["episode"] = [d if isinstance(d, date) else date.fromisoformat(d)
                         for d in ev["episode"]]
        ev["t0_alternatives"] = [d if isinstance(d, date) else date.fromisoformat(d)
                                 for d in ev.get("t0_alternatives", [])]
    return cfg


def load_pages() -> dict:
    return load_yaml("wikipedia_pages.yml")


# -------------------------------------------------------------------- dates
def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def event_window(t0: date, pre: int, post: int, gap: int = 0) -> tuple[date, date]:
    """Observation window ``[t0 - pre - gap, t0 + post]``.

    ``gap`` is the event's ``baseline_gap_days``: the number of days
    immediately before t0 that are excluded from the baseline, so the window
    must start that many days earlier to keep the baseline ``pre`` days long.
    """
    return t0 - timedelta(days=pre + gap), t0 + timedelta(days=post)


def baseline_gap(event: dict) -> int:
    """Days between the end of the Wikipedia baseline and t0 (events.yml).

    A gap of 0 gives the baseline ``[-pre, -1]``; a gap of 1 gives
    ``[-(pre+1), -2]``, used when the warning began the day before t0.
    """
    return int(event.get("baseline_gap_days", 0) or 0)


def baseline_label(pre: int, gap: int) -> str:
    return f"[-{pre + gap},-{gap + 1}]"


# --------------------------------------------------------------------- HTTP
class FetchError(RuntimeError):
    """Raised when a source cannot be reached after the configured retries.

    Callers must let this propagate rather than substituting zeros, so that an
    API failure is never recorded as an observed count.
    """


def fetch_json(url: str, *, retries: int = 5, backoff: float = 2.0,
               timeout: int = 45, ok_404: bool = False):
    """GET a JSON endpoint with exponential backoff.

    Returns ``None`` only for a 404 when ``ok_404`` is set (used for Wikimedia
    articles with no recorded traffic in a window, which is a meaningful
    "no data" answer rather than a transport failure).
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 404 and ok_404:
                return None
            # 429/5xx are transient; 4xx other than 429 will not improve.
            if exc.code not in (429, 500, 502, 503, 504):
                raise FetchError(f"HTTP {exc.code} for {url}") from exc
        except Exception as exc:                      # noqa: BLE001 - transport
            last = exc
        time.sleep(backoff * (2 ** attempt))
    raise FetchError(f"failed after {retries} attempts: {url} ({last})")


def fetch_text(url: str, *, retries: int = 5, backoff: float = 2.0,
               timeout: int = 45) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:                      # noqa: BLE001 - transport
            last = exc
            time.sleep(backoff * (2 ** attempt))
    raise FetchError(f"failed after {retries} attempts: {url} ({last})")


# ------------------------------------------------------------------ manifest
def write_manifest(step: str, **extra) -> Path:
    """Append a step record to the machine-readable run manifest."""
    DERIVED.mkdir(parents=True, exist_ok=True)
    path = DERIVED / "run_manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"steps": []}
    manifest["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    # Recomputed on every step so the manifest always describes the config
    # that produced the most recent outputs, not the one first seen.
    manifest["config_hashes"] = {
        name: config_hash(name)
        for name in ("events.yml", "wikipedia_pages.yml", "portfolios.csv")
    }
    manifest["steps"] = [s for s in manifest["steps"] if s["step"] != step]
    manifest["steps"].append({
        "step": step,
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    })
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def _package_versions() -> dict:
    out = {}
    for mod in ("numpy", "pandas", "statsmodels", "scipy", "matplotlib",
                "yaml", "vaderSentiment"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            out[mod] = "not installed"
    return out


def save_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path
