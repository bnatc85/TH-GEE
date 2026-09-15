"""Collect daily English Wikipedia pageviews for the preregistered families.

Implements paper Sec. 5.2. Three properties matter for the analysis downstream:

1. **Redirect and page-move resolution.** Wikimedia records a pageview against
   the *requested* title, so a preregistered title that is now a redirect has
   its own (usually small) series while the real audience sits on the target.
   Each title is resolved through the MediaWiki API before collection and both
   the requested and resolved titles are written to the resolution log, so the
   frozen page list stays auditable.

2. **Missing is not zero.** The pageviews API omits days with no recorded
   traffic *and* days that have not been published yet. These are distinguished
   using the aggregate-traffic endpoint: the last day present there is the last
   published day. Any gap at or before that day is a true zero; anything after
   it is recorded as ``null`` (paper Sec. 6.3).

3. **Aggregate traffic is collected alongside.** Family effects are normalised
   by total en.wikipedia user traffic in robustness check (paper Sec. 6.2) so a
   platform-wide change cannot be misread as topic-specific attention.

Usage:
    python src/collect_wikipedia.py
    python src/collect_wikipedia.py --event warning2026
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (DERIVED, RAW, FetchError, baseline_gap, baseline_label,
                    daterange, event_window, fetch_json, load_events,
                    load_pages, save_json, write_manifest, ymd)

PAGEVIEW_API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews"
                "/per-article/{project}/{access}/{agent}/{title}/daily/{start}/{end}")
AGGREGATE_API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews"
                 "/aggregate/en.wikipedia/{access}/{agent}/daily/{start}/{end}")
MW_API = "https://en.wikipedia.org/w/api.php"

POLITE_DELAY = 0.12   # seconds between article requests


# ------------------------------------------------------------------ resolution
MIN_BASELINE_COVERAGE = 0.80   # share of baseline days with a recorded view


def redirect_aliases(target: str) -> list[str]:
    """Mainspace titles that currently redirect to ``target``.

    A page move leaves the old title behind as a redirect, so this set contains
    the historical titles under which the article accumulated pageviews. Draft
    and user-space titles are excluded: attention to a draft is not attention to
    an article.
    """
    params = {"action": "query", "format": "json", "formatversion": "2",
              "prop": "redirects", "rdlimit": "500", "rdnamespace": "0",
              "titles": target}
    data = fetch_json(f"{MW_API}?{urllib.parse.urlencode(params)}")
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return []
    return [r["title"] for r in pages[0].get("redirects", [])
            if ":" not in r["title"].split(" ")[0] or not r["title"].startswith(
                ("Draft:", "User:", "Talk:", "Wikipedia:"))]


def baseline_coverage(series: dict, baseline_start: date,
                      baseline_end: date) -> float:
    """Share of baseline days on which the API recorded a view.

    Scored over the primary baseline only, so the extra history fetched for
    the alternative-baseline check cannot change which title is chosen.
    """
    pre = [v for k, v in series.items()
           if baseline_start <= date.fromisoformat(k) <= baseline_end]
    if not pre:
        return 0.0
    return sum(1 for v in pre if v) / len(pre)


def resolve_titles(titles: list[str]) -> dict[str, dict]:
    """Resolve redirects, normalisation, and missing pages via the MediaWiki API.

    Returns one record per requested title with the resolved target, the
    resolution kind, and whether the target is a disambiguation page (which the
    preregistration excludes).
    """
    out: dict[str, dict] = {}
    for chunk_start in range(0, len(titles), 40):
        chunk = titles[chunk_start:chunk_start + 40]
        params = {
            "action": "query", "format": "json", "formatversion": "2",
            "redirects": "1", "prop": "pageprops", "ppprop": "disambiguation",
            "titles": "|".join(chunk),
        }
        data = fetch_json(f"{MW_API}?{urllib.parse.urlencode(params)}")
        q = data.get("query", {})
        # Chain: requested -> normalized -> redirected.
        norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
        redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
        pages = {p["title"]: p for p in q.get("pages", [])}

        for requested in chunk:
            step1 = norm.get(requested, requested)
            step2 = redir.get(step1, step1)
            page = pages.get(step2, {})
            kind = "direct"
            if step1 != requested:
                kind = "normalized"
            if step2 != step1:
                kind = "redirect"
            out[requested] = {
                "requested": requested,
                "resolved": step2,
                "resolution": kind,
                "missing": bool(page.get("missing", False)),
                "disambiguation": "disambiguation" in page.get("pageprops", {}),
            }
    return out


# ------------------------------------------------------------------ collection
def last_published_day(access: str, agent: str, probe_start: date,
                       probe_end: date) -> date:
    """Last day for which the pageviews API has published data."""
    url = AGGREGATE_API.format(access=access, agent=agent,
                               start=ymd(probe_start), end=ymd(probe_end))
    items = fetch_json(url)["items"]
    return date.fromisoformat(items[-1]["timestamp"][:8][:4] + "-"
                              + items[-1]["timestamp"][4:6] + "-"
                              + items[-1]["timestamp"][6:8])


def aggregate_series(access: str, agent: str, start: date, end: date) -> dict:
    url = AGGREGATE_API.format(access=access, agent=agent,
                               start=ymd(start), end=ymd(end))
    items = fetch_json(url)["items"]
    return {f"{i['timestamp'][:4]}-{i['timestamp'][4:6]}-{i['timestamp'][6:8]}":
            i["views"] for i in items}


def article_series(title: str, project: str, access: str, agent: str,
                   start: date, end: date, published_through: date) -> dict:
    """Daily views keyed by ISO date; ``None`` where the API has not published.

    A 404 means the article has no recorded traffic in the window at all, which
    for a live article is a true run of zeros, not a transport failure.
    """
    url = PAGEVIEW_API.format(project=project, access=access, agent=agent,
                              title=urllib.parse.quote(title.replace(" ", "_"), safe=""),
                              start=ymd(start), end=ymd(end))
    data = fetch_json(url, ok_404=True)
    observed = {}
    if data is not None:
        for item in data["items"]:
            ts = item["timestamp"]
            observed[f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"] = item["views"]

    series = {}
    for d in daterange(start, end):
        key = d.isoformat()
        if key in observed:
            series[key] = observed[key]
        elif d <= published_through:
            series[key] = 0          # published day with no recorded traffic
        else:
            series[key] = None       # not yet published by the API
    return series


def collect_event(event: dict, pages_cfg: dict, analysis: dict) -> dict:
    project = pages_cfg["project"]
    access = pages_cfg["access"]
    agent = pages_cfg["agent"]
    t0 = event["t0"]
    gap = baseline_gap(event)
    start, end = event_window(t0, analysis["pre_event_days"],
                              analysis["post_event_days"], gap)
    # Last baseline day. With gap = 1 the announcement day (t0 - 1) is kept in
    # the window but not counted as baseline.
    baseline_end = t0 - timedelta(days=gap + 1)
    # Collect enough history for the longest alternative baseline (robustness
    # check 2) and for placebo dates early in the pre-event interval; the
    # primary model still uses only [start, baseline_end].
    history = max([analysis["pre_event_days"], *analysis.get("alt_baselines", [])])
    fetch_start = t0 - timedelta(days=history + gap)

    published_through = last_published_day(access, agent,
                                           t0 - timedelta(days=10),
                                           end)
    truncated = published_through < end
    print(f"[{event['id']}] window {start}..{end} (fetched from {fetch_start}) "
          f"| baseline {baseline_label(analysis['pre_event_days'], gap)} = "
          f"{start}..{baseline_end} | API published through "
          f"{published_through}" + (" (TRUNCATED)" if truncated else ""))

    # Assemble every title we need, including negative controls.
    family_titles = {fam: list(spec["pages"])
                     for fam, spec in pages_cfg["families"].items()}
    controls = list(pages_cfg["negative_controls"])
    all_titles = sorted({t for ts in family_titles.values() for t in ts} | set(controls))

    resolution = resolve_titles(all_titles)
    dropped = {}
    for title, rec in resolution.items():
        if rec["missing"]:
            dropped[title] = "nonexistent page"
        elif rec["disambiguation"]:
            dropped[title] = "disambiguation page"

    series: dict[str, dict] = {}
    for i, title in enumerate(all_titles, 1):
        if title in dropped:
            print(f"  ({i}/{len(all_titles)}) SKIP {title}: {dropped[title]}")
            continue
        target = resolution[title]["resolved"]
        try:
            fetched = article_series(target, project, access, agent,
                                     fetch_start, end, published_through)
        except FetchError as exc:
            # Re-queried by rerunning the script; never zero-filled.
            print(f"  ({i}/{len(all_titles)}) FAIL {title}: {exc}")
            dropped[title] = f"api failure: {exc}"
            continue

        # Titles resolve to the article's *current* name. For a historical
        # event the article may have sat at a different title, whose series
        # carries the traffic. Fall back to the alias set, scored only on
        # pre-event baseline days so no post-event outcome informs the choice.
        cover = baseline_coverage(fetched, start, baseline_end)
        if cover < MIN_BASELINE_COVERAGE:
            best, best_cover, tried = target, cover, {target: round(cover, 3)}
            for alias in redirect_aliases(target):
                time.sleep(POLITE_DELAY)
                try:
                    cand = article_series(alias, project, access, agent,
                                          fetch_start, end, published_through)
                except FetchError:
                    continue
                c = baseline_coverage(cand, start, baseline_end)
                tried[alias] = round(c, 3)
                if c > best_cover:
                    best, best_cover, fetched = alias, c, cand
            resolution[title].update({
                "as_of_resolved": best, "as_of_baseline_coverage": round(best_cover, 3),
                "as_of_candidates": tried,
            })
            if best_cover < MIN_BASELINE_COVERAGE:
                # Documented removal rule: insufficient baseline observations
                # (article did not exist, or was a draft, at this event date).
                dropped[title] = (f"insufficient baseline observations "
                                  f"(best coverage {best_cover:.2f} at {best!r})")
                print(f"  ({i}/{len(all_titles)}) DROP {title}: {dropped[title]}")
                continue
            target = best
            resolution[title]["resolution"] = "as-of-event title"
            print(f"  ({i}/{len(all_titles)}) {title} -> {target} "
                  f"[as-of-event, baseline coverage {best_cover:.2f}]")
        else:
            print(f"  ({i}/{len(all_titles)}) {title} -> {target} "
                  f"[{resolution[title]['resolution']}]")

        resolution[title]["series_title"] = target
        series[title] = fetched
        time.sleep(POLITE_DELAY)

    return {
        "event_id": event["id"],
        "t0": t0.isoformat(),
        "window": [start.isoformat(), end.isoformat()],
        "fetched_from": fetch_start.isoformat(),
        "baseline_gap_days": gap,
        "baseline": [start.isoformat(), baseline_end.isoformat()],
        "baseline_days_relative": baseline_label(analysis["pre_event_days"], gap),
        "published_through": published_through.isoformat(),
        "truncated": truncated,
        "project": project, "access": access, "agent": agent,
        "families": family_titles,
        "negative_controls": controls,
        "resolution": resolution,
        "dropped": dropped,
        "series": series,
        "aggregate": aggregate_series(access, agent, fetch_start,
                                      min(end, published_through)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", help="collect a single event id")
    args = ap.parse_args()

    events_cfg = load_events()
    pages_cfg = load_pages()
    analysis = events_cfg["analysis"]

    targets = [e for e in events_cfg["events"]
               if args.event is None or e["id"] == args.event]
    if not targets:
        raise SystemExit(f"no event matching {args.event!r}")

    summary = {}
    for event in targets:
        payload = collect_event(event, pages_cfg, analysis)
        path = save_json(RAW / f"wikipedia_{event['id']}.json", payload)
        save_json(DERIVED / f"page_resolution_log_{event['id']}.json", {
            "resolution": payload["resolution"], "dropped": payload["dropped"],
        })
        summary[event["id"]] = {
            "pages_collected": len(payload["series"]),
            "pages_dropped": len(payload["dropped"]),
            "published_through": payload["published_through"],
            "truncated": payload["truncated"],
            "path": str(path.relative_to(path.parents[2])),
        }
        print(f"[{event['id']}] wrote {path}")

    write_manifest("collect_wikipedia", events=summary)


if __name__ == "__main__":
    main()
