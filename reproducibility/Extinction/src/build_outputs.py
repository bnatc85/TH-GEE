"""Render the manuscript's figures and tables from the derived results.

Writes to the paths the Overleaf project expects (paper Appendix C):

    figures/figure_1_attention.pdf      family attention time series
    figures/figure_2_gap.pdf            threat-preparedness gap
    figures/figure_3_transmission.pdf   media framing -> attention
    figures/figure_4_market_car.pdf     portfolio CARs
    figures/figure_5_divergence.pdf     attention vs. repricing
    outputs/tables/*.tex

Figures are written beside main.tex rather than under outputs/, so the Overleaf
project references them as figures/<name>.pdf.

Figures are print targets for a LaTeX manuscript, so they commit to a single
light rendering rather than carrying a dark variant. Colours follow a validated
categorical order; every panel is direct-labelled by title, which supplies the
relief the lighter hues require against a light surface.

Usage:
    python src/build_outputs.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DERIVED, OUTPUTS, ROOT, load_events, load_pages, write_manifest  # noqa: E402

# Validated categorical order (see references/palette.md; checked with
# scripts/validate_palette.js on the light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
# Neutrals are kept near-black: the figures are read in a compiled PDF where
# mid-greys lose contrast, so secondary lines, intervals, and the placebo band
# must survive print and screen at reduced size.
INK = "#000000"
INK_2 = "#1c1c1a"      # labels, ticks, zero lines
MUTED = "#2e2e2b"      # comparison series, interval bars, notes
SURFACE = "#fcfcfb"
GRID = "#cfcfca"       # gridlines only
BAND = "#9a9a94"       # placebo range fill, drawn translucent

FAMILY_ORDER = ["threat", "capabilities", "actors", "safety", "infrastructure", "control"]
FAMILY_LABEL = {
    "threat": "Existential threat", "capabilities": "Capabilities/history",
    "actors": "Actors/institutions", "safety": "Safety/governance",
    "infrastructure": "Infrastructure/resilience", "control": "Shutdown/control",
}
PORT_LABEL = {"ai_infrastructure": "AI infrastructure",
              "ai_platform": "AI platforms/exposure",
              "cyber_resilience": "Cybersecurity/resilience"}

# Reader-facing names. Internal identifiers such as
# "warning2026 episode[2026-09-08..2026-09-14]" are fine in a results file and
# unreadable in a printed figure.
EVENT_LABEL = {"statement2023": "May 2023 statement",
               "warning2026": "September 2026 episode"}


def window_label(event_id: str, wkey: str) -> str:
    """Human-readable window name for axis labels and captions."""
    if wkey.startswith("episode"):
        span = wkey[wkey.find("[") + 1:-1].split("..")
        if event_id == "statement2023":
            return "Event day (30 May)"
        lo, hi = (pd.Timestamp(s) for s in span)
        return f"Episode, {lo.day}\u2013{hi.day} {hi.strftime('%b')}"
    return f"Trading days {wkey}"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


def load(name: str):
    p = DERIVED / name
    return json.loads(p.read_text()) if p.exists() else None



def common_horizon(att: dict, event_ids: list[str], lo: int = -21,
                   hi: int = 21) -> tuple[int, int]:
    """Day offsets observed for *every* event, clipped to ``[lo, hi]``.

    The events have different publication cutoffs: in September 2026 the
    pageviews API had published to t0+17, while the closed 2023 window runs the
    full t0+28. Drawing each series to its own last observed day makes the
    shorter one look like it stopped rather than that it has not been published,
    and invites a cross-event comparison over unequal spans. Both series are
    therefore cut to the horizon both events actually cover.
    """
    starts, ends = [], []
    for eid in event_ids:
        d = att[eid]["daily"]
        t0 = pd.Timestamp(att[eid]["t0"])
        days = (pd.to_datetime(d["dates"]) - t0).days
        seen = [day for k, series in d["faa"].items()
                for day, v in zip(days, series) if v is not None]
        if seen:
            starts.append(min(seen))
            ends.append(max(seen))
    if not starts:
        return lo, hi
    return max(lo, max(starts)), min(hi, min(ends))


# ------------------------------------------------------------------ figure 1
def figure_attention(att: dict, events_cfg: dict) -> Path:
    """Family-level standardised abnormal attention, one panel per family.

    Small multiples rather than six overlaid lines: with two events drawn per
    panel, overlaying twelve series would make identity unreadable and would put
    the whole categorical palette into all-pairs comparison.
    """
    evs = [e for e in events_cfg["events"] if e["id"] in att]
    lo, hi = common_horizon(att, [e["id"] for e in evs])
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), sharex=True, sharey=True)

    for i, fam in enumerate(FAMILY_ORDER):
        ax = axes.flat[i]
        _style(ax)
        # Six small panels at 0.98\linewidth print small; type is scaled up
        # so panel titles, axes, and the legend remain readable at page size.
        ax.tick_params(labelsize=10)
        for j, ev in enumerate(evs):
            d = att[ev["id"]]["daily"]
            if fam not in d["faa"]:
                continue
            t0 = pd.Timestamp(att[ev["id"]]["t0"])
            days = (pd.to_datetime(d["dates"]) - t0).days
            y = pd.Series(d["faa"][fam], dtype=float)
            m = (days >= lo) & (days <= hi)
            ax.plot(np.asarray(days)[m], y[m], lw=1.6 if j == 0 else 1.4,
                    color=SERIES[i] if j == 1 else MUTED,
                    ls="-" if j == 1 else (0, (3, 1.6)),
                    label=("2026 episode" if j == 1 else "2023 statement"),
                    zorder=3 if j == 1 else 2)
        ax.axhline(0, color=INK_2, lw=0.7, zorder=1)
        ax.axvline(0, color=MUTED, lw=0.7, ls=":", zorder=1)
        ax.set_title(FAMILY_LABEL[fam], fontsize=11, pad=5, loc="left")
        if i % 3 == 0:
            ax.set_ylabel("Standardised\nabnormal attention", fontsize=10)
        if i >= 3:
            ax.set_xlabel("Days from $t_0$", fontsize=10)

    # Colour identifies the family (one per panel), so the legend must carry
    # line style only -- a coloured handle would imply the 2026 series is blue
    # in every panel.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=MUTED, lw=1.4, ls=(0, (3, 1.6)),
                      label="2023 statement"),
               Line2D([], [], color=INK_2, lw=1.6, ls="-",
                      label="2026 episode (panel colour)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=10)
    fig.suptitle("Wikipedia attention by information family around AI extinction warnings",
                 fontsize=12.5, x=0.012, ha="left", y=1.0, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.965))
    return _save(fig, "figure_1_attention")


# ------------------------------------------------------------------ figure 2
def figure_gap(att: dict) -> Path:
    """The threat--preparedness gap: family effects against placebo noise.

    One panel per event, families on a shared axis, ordered threat-oriented
    first. The grey band is the 95th percentile of |effect| across placebo
    dates drawn from the same pre-event interval, so a bar inside the band is
    indistinguishable from background variation.
    """
    events = [e for e in ("statement2023", "warning2026") if e in att]
    # Panels stacked vertically on a shared x-axis: the cross-event magnitude
    # difference is itself a result, and separate scales would make 2023 and
    # 2026 look alike. Stacking puts the two effect scales directly above one
    # another so the reader compares horizontal extents, not panel widths.
    fig, axes = plt.subplots(len(events), 1, figsize=(6.6, 6.4),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)

    ypos = {fam: -i for i, fam in enumerate(FAMILY_ORDER)}
    for ax, eid in zip(axes, events):
        _style(ax)
        w = att[eid]["windows"]["[0,+3]"]
        placebo = att[eid].get("robustness", {}).get(
            "7_placebo_dates", {}).get("summary", {})

        for fam in FAMILY_ORDER:
            s = w["families"].get(fam)
            if not s:
                continue
            y = ypos[fam]
            band = placebo.get(fam, {}).get("p95_abs")
            if band:
                # Light, translucent fill with a crisp outline: the band is
                # context, and the estimate labels must stay legible over it.
                ax.barh(y, 2 * band, left=-band, height=0.74,
                        facecolor=BAND, alpha=0.22, edgecolor=INK_2,
                        linewidth=0.6, zorder=1)
            prep = fam in ("safety", "infrastructure", "control")
            colour = SERIES[2] if prep else SERIES[1]
            ax.plot([s["ci_low"], s["ci_high"]], [y, y], color=MUTED,
                    lw=1.5, zorder=3, solid_capstyle="butt")
            ax.plot([s["estimate"]], [y], "o", ms=7, color=colour,
                    mec=SURFACE, mew=1.3, zorder=4)
            inside = band is not None and abs(s["estimate"]) <= band
            # Long "n.s." labels run rightward into open space instead of
            # being centred, which would push them past the left axis.
            ax.annotate(f"{s['estimate']:.1f}" + ("  n.s. vs placebo" if inside else ""),
                        (s["estimate"], y), textcoords="offset points",
                        xytext=(-10, 9) if inside else (0, 9),
                        ha="left" if inside else "center", fontsize=7.4,
                        color=INK_2 if not inside else MUTED, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE,
                                  edgecolor="none", alpha=0.9))

        ax.axvline(0, color=INK_2, lw=0.8, zorder=2)
        ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels([FAMILY_LABEL[f] for f in FAMILY_ORDER], fontsize=8.4)
        ax.set_title(EVENT_LABEL[eid], fontsize=9.5, loc="left", pad=14)
        ax.set_ylim(-len(FAMILY_ORDER) + 0.4, 0.9)
    # Shared x-axis, so the label belongs to the bottom panel only.
    axes[-1].set_xlabel("Cumulative abnormal attention, $[0,+3]$ (SD-days)",
                        fontsize=8)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], marker="o", ls="none", ms=7, color=SERIES[1],
               label="Threat-oriented family"),
        Line2D([], [], marker="o", ls="none", ms=7, color=SERIES[2],
               label="Preparedness-oriented family"),
        Patch(facecolor=BAND, alpha=0.22, edgecolor=INK_2, linewidth=0.6,
              label="Placebo range (95th pct. of |effect|)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, "figure_2_gap")


# ------------------------------------------------------------------ figure 5
def figure_divergence(att: dict, mkt: dict) -> Path | None:
    """Public attention and market repricing of resilience, side by side.

    The two panels are deliberately not on one axis: they are different
    quantities (standardised attention vs. percentage abnormal return). The
    comparison the figure supports is directional, not numerical -- resilience
    is the smallest family response in public information seeking (though above
    its placebo threshold) while resilience providers are repriced the most.
    """
    if not mkt or "warning2026" not in att:
        return None
    res = mkt["events"].get("warning2026", {}).get(
        "specifications", {}).get("sector_adjusted")
    if not res:
        return None
    ep = next((w for k, w in res["windows"].items()
               if k.startswith("episode") and w.get("available")), None)
    if not ep:
        return None

    w = att["warning2026"]["windows"]["[0,+3]"]
    fams = ["threat", "safety", "infrastructure", "control"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax in (ax0, ax1):
        _style(ax)

    ys = [-i for i in range(len(fams))]
    for y, fam in zip(ys, fams):
        s = w["families"][fam]
        colour = SERIES[1] if fam == "threat" else SERIES[2]
        ax0.plot([s["ci_low"], s["ci_high"]], [y, y], color=MUTED, lw=1.5)
        ax0.plot([s["estimate"]], [y], "o", ms=7, color=colour,
                 mec=SURFACE, mew=1.3)
        ax0.annotate(f"{s['estimate']:.1f}", (s["estimate"], y),
                     textcoords="offset points", xytext=(0, 9), ha="center",
                     fontsize=7.4, color=INK_2)
    ax0.axvline(0, color=INK_2, lw=0.8)
    ax0.set_yticks(ys)
    ax0.set_yticklabels([FAMILY_LABEL[f] for f in fams], fontsize=8.4)
    ax0.set_xlabel("Cumulative abnormal attention, $[0,+3]$ (SD-days)", fontsize=8)
    ax0.set_title("Public information seeking", fontsize=9.5, loc="left", pad=14)
    ax0.set_ylim(min(ys) - 0.6, max(ys) + 0.75)

    ports = ["ai_infrastructure", "ai_platform", "cyber_resilience"]
    ys2 = [-i for i in range(len(ports))]
    for y, p in zip(ys2, ports):
        v = ep["by_portfolio"].get(p)
        if not v:
            continue
        car = 100 * v["car_equal_weighted"]
        ci = v.get("ci95_conventional") or [v["car_equal_weighted"]] * 2
        colour = SERIES[2] if p == "cyber_resilience" else SERIES[1]
        ax1.plot([100 * ci[0], 100 * ci[1]], [y, y], color=MUTED, lw=1.5)
        ax1.plot([car], [y], "o", ms=7, color=colour, mec=SURFACE, mew=1.3)
        ax1.annotate(f"{car:+.1f}%", (car, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=7.4, color=INK_2)
    ax1.axvline(0, color=INK_2, lw=0.8)
    ax1.set_yticks(ys2)
    ax1.set_yticklabels([PORT_LABEL[p] for p in ports], fontsize=8.4)
    ax1.set_xlabel("Cumulative abnormal return, 8\u201314 Sep (%)", fontsize=8)
    ax1.set_title("Equity repricing", fontsize=9.5, loc="left", pad=14)
    ax1.set_ylim(min(ys2) - 0.6, max(ys2) + 0.75)

    fig.suptitle("Markets repriced resilience most; the public sought it least",
                 fontsize=10.5, x=0.012, ha="left", y=1.0, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "figure_5_divergence")


# ------------------------------------------------------------------ figure 4
def figure_transmission(att: dict) -> Path | None:
    """Is a media frame associated with attention to the matching topic?

    Cumulative lag-0..3 associations from Eq. 6 with 95% intervals. Filled
    markers survive Benjamini--Hochberg control at q < 0.05.
    """
    events = [e for e in ("statement2023", "warning2026")
              if att.get(e, {}).get("distributed_lag", {}).get("available")]
    if not events:
        return None

    order = ["existential->threat", "bio->threat", "capability->capabilities",
             "economic->capabilities", "skeptic->actors", "governance->safety",
             "infrastructure->infrastructure", "control->control"]
    # Short "frame -> family" labels; the column header above the axis says
    # what the two sides are, so "framing" need not repeat on every row.
    nice = {"existential->threat": "Existential $\\rightarrow$ threat",
            "bio->threat": "Bio/chem $\\rightarrow$ threat",
            "capability->capabilities": "Capability $\\rightarrow$ capabilities",
            "economic->capabilities": "Economic $\\rightarrow$ capabilities",
            "skeptic->actors": "Skeptic $\\rightarrow$ actors",
            "governance->safety": "Governance $\\rightarrow$ safety",
            "infrastructure->infrastructure": "Infrastructure $\\rightarrow$ infrastructure",
            "control->control": "Control $\\rightarrow$ shutdown/control"}

    fig, axes = plt.subplots(1, len(events), figsize=(10.6, 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, eid in zip(axes, events):
        _style(ax)
        ax.tick_params(labelsize=9.6, labelcolor=INK, color=INK)
        pairs = att[eid]["distributed_lag"]["pairs"]
        for i, key in enumerate(order):
            v = pairs.get(key)
            if not v:
                continue
            y = -i
            sig = v.get("q_bh") is not None and v["q_bh"] < 0.05
            mech = key.startswith(("infrastructure", "control"))
            colour = SERIES[2] if mech else SERIES[1]
            ax.plot(v["ci95"], [y, y], color=MUTED, lw=1.5)
            ax.plot([v["cumulative_lag0_3"]], [y], "o", ms=7,
                    color=colour if sig else SURFACE,
                    mec=colour, mew=1.6)
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.set_yticks([-i for i in range(len(order))])
        ax.set_yticklabels([nice[k] for k in order], fontsize=9.6, color=INK)
        ax.set_title(EVENT_LABEL[eid], fontsize=11.5, loc="left", pad=8,
                     color=INK)
        ax.set_xlabel("Cumulative lag-0..3 association", fontsize=10,
                      color=INK)
    # Header over the shared tick-label column, level with the panel titles.
    axes[0].text(-0.02, 1.0, "Framing $\\rightarrow$ attention family",
                 transform=axes[0].transAxes, ha="right", va="bottom",
                 fontsize=10, color=INK, style="italic")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="none", ms=7, color=SERIES[1],
               label="Threat/capability pathway"),
        Line2D([], [], marker="o", ls="none", ms=7, color=SERIES[2],
               label="Infrastructure/control pathway"),
        Line2D([], [], marker="o", ls="none", ms=7, color=SURFACE,
               mec=INK_2, mew=1.6, label="Hollow: not significant at $q<.05$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.05), fontsize=9.6,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, "figure_3_transmission")


# ------------------------------------------------------------------ figure 3
def figure_frames(att: dict, pages_cfg: dict, event_id: str) -> Path | None:
    """Daily frame prevalence above matched family attention, shared x-axis.

    Two measures on different scales share a time axis but never an axis: the
    panels are stacked instead of twinned.
    """
    news = load(f"news_daily_{event_id}.json")
    if not news or event_id not in att:
        return None
    nd = pd.DataFrame(news)
    nd["date"] = pd.to_datetime(nd["date"])
    nd = nd.sort_values("date")

    d = att[event_id]["daily"]
    t0 = pd.Timestamp(att[event_id]["t0"])
    faa = pd.DataFrame(d["faa"])
    faa.index = pd.to_datetime(d["dates"])

    pairs = [("existential", "threat"), ("infrastructure", "infrastructure"),
             ("control", "control"), ("governance", "safety")]
    pairs = [(f, k) for f, k in pairs if f in nd.columns and k in faa.columns]

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})
    ax0, ax1 = axes
    for ax in axes:
        _style(ax)

    for i, (frame, fam) in enumerate(pairs):
        days = (nd["date"] - t0).dt.days
        m = (days >= -21) & (days <= 21)
        ax0.plot(days[m], nd[frame][m], lw=1.8, color=SERIES[i],
                 label=f"{frame} frame")
        fd = (faa.index - t0).days
        m2 = (fd >= -21) & (fd <= 21)
        ax1.plot(np.asarray(fd)[m2], faa[fam].values[m2], lw=1.8,
                 color=SERIES[i], label=FAMILY_LABEL[fam])

    ax0.text(0.995, -0.16,
             "Shares are from a ranked news sample, not a census of coverage",
             transform=ax0.transAxes, ha="right", va="top",
             fontsize=7, color=MUTED)

    ax0.set_ylabel("Frame prevalence\n(share of stories)", fontsize=8)
    ax1.set_ylabel("Standardised\nabnormal attention", fontsize=8)
    ax1.set_xlabel("Days from $t_0$", fontsize=8)
    for ax in axes:
        ax.axvline(0, color=MUTED, lw=0.7, ls=":")
    ax1.axhline(0, color=INK_2, lw=0.7)
    ax0.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    ax1.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    fig.suptitle("Media frames and matched information-seeking families "
                 f"({event_id})", fontsize=10.5, x=0.012, ha="left", y=1.0,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, "figure_3_frames_attention")


# ------------------------------------------------------------------ figure 3
def figure_market(mkt: dict) -> Path | None:
    """Portfolio CARs with 95% intervals, one panel per event.

    One panel per event rather than a single stacked column: once both events
    carry all four windows the stacked version runs to twenty-four rows and
    taller than a journal page. Side-by-side panels halve the height, and the
    shared x-axis keeps the cross-event magnitude difference --- which is the
    result --- readable as horizontal extent rather than panel width.
    """
    if not mkt:
        return None
    rows = []
    for eid, res in mkt["events"].items():
        spec = res.get("specifications", {}).get("sector_adjusted")
        if not spec:
            continue
        for wkey, w in spec["windows"].items():
            if not w.get("available"):
                continue
            for pname, p in w.get("by_portfolio", {}).items():
                rows.append({"event": eid, "window": wkey, "portfolio": pname,
                             "car": p["car_equal_weighted"],
                             "ci": p.get("ci95_conventional"),
                             "p": p.get("randomisation_p"),
                             "confounded": bool(
                                 p.get("excluding_confounded", {}).get("excluded"))})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    events = [e for e in ("statement2023", "warning2026")
              if e in set(df["event"])]

    def layout(sub: pd.DataFrame):
        """Row positions, one header per window group followed by its members."""
        out, y = [], 0.0
        for wkey, g in sub.groupby("window", sort=False):
            out.append({"kind": "header", "y": y,
                        "label": window_label(g["event"].iloc[0], wkey)})
            y -= 1
            for _, r in g.iterrows():
                note = ""
                ci = r["ci"] or [r["car"], r["car"]]
                note = f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]"
                if r["p"] is not None and not pd.isna(r["p"]):
                    pv = ("< .001" if r["p"] < 0.001
                          else f"= {r['p']:.3f}".replace("= 0.", "= ."))
                    note += f", $p_{{\\mathrm{{rand}}}}$ {pv}"
                out.append({"kind": "point", "y": y,
                            "label": f"    {PORT_LABEL[r['portfolio']]}",
                            "x": 100 * r["car"],
                            "lo": 100 * (r["car"] - ci[0]),
                            "hi": 100 * (ci[1] - r["car"]),
                            "colour": SERIES[["ai_infrastructure", "ai_platform",
                                              "cyber_resilience"].index(r["portfolio"])],
                            "flag": bool(r["confounded"]), "note": note})
                y -= 1
            y -= 0.6
        return out

    panels = {eid: layout(df[df["event"] == eid]) for eid in events}
    n_rows = max(len(v) for v in panels.values())
    # Rows advance by 1 and each window group adds a 0.6 gap, so the vertical
    # extent is larger than the row count; clipping to -n_rows would cut the
    # final group off the bottom of the panel.
    y_min = min(it["y"] for v in panels.values() for it in v)
    h = max(3.2, 0.34 * (abs(y_min) + 1) + 1.5)
    fig, axes = plt.subplots(1, len(events), figsize=(11.0, h), sharex=True)
    axes = np.atleast_1d(axes)

    any_flag = False
    for ax, eid in zip(axes, events):
        _style(ax)
        ax.axvline(0, color=INK_2, lw=0.8, zorder=1)
        items = panels[eid]
        pts = [it for it in items if it["kind"] == "point"]
        ax.errorbar([it["x"] for it in pts], [it["y"] for it in pts],
                    xerr=[[it["lo"] for it in pts], [it["hi"] for it in pts]],
                    fmt="none", elinewidth=1.4, ecolor=MUTED, zorder=3)
        for it in pts:
            any_flag = any_flag or it["flag"]
            ax.plot([it["x"]], [it["y"]], "o", ms=6.5, color=it["colour"],
                    mec=SURFACE, mew=1.3, zorder=4)
            ax.annotate(f"{it['x']:+.2f}%" + ("\u2020" if it["flag"] else "")
                        + "  " + it["note"],
                        (it["x"], it["y"]), textcoords="offset points",
                        xytext=(0, 7.5), ha="center", fontsize=6.6,
                        color=INK_2, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE,
                                  edgecolor="none", alpha=0.85))
        ax.set_yticks([it["y"] for it in items])
        ax.set_yticklabels([it["label"] for it in items], fontsize=7.6)
        for t, it in zip(ax.get_yticklabels(), items):
            if it["kind"] == "header":
                t.set_color(INK); t.set_fontweight("bold"); t.set_fontsize(7.9)
        ax.set_ylim(y_min - 0.6, 0.9)
        ax.set_title(EVENT_LABEL[eid], fontsize=10, loc="left", pad=8,
                     color=INK)
        ax.set_xlabel("Cumulative abnormal return (%),\nsector-adjusted market model",
                      fontsize=8)
    # Labels are centred on the point, so the extremes need breathing room.
    lo, hi = axes[0].get_xlim()
    axes[0].set_xlim(lo - 0.04 * (hi - lo), hi + 0.10 * (hi - lo))

    foot = ("Labels: CAR [95% conventional interval], "
            "$p_{\\mathrm{rand}}$ = randomisation $p$ from placebo windows")
    if any_flag:
        foot += ("\n\u2020 window contains a firm-specific announcement "
                 "(earnings); see robustness")
    fig.text(0.0, -0.005, foot, fontsize=7, color=MUTED, va="top",
             linespacing=1.5)
    fig.suptitle("Portfolio cumulative abnormal returns", fontsize=10.5,
                 x=0.012, ha="left", y=1.0, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return _save(fig, "figure_4_market_car")


def _save(fig, stem: str) -> Path:
    # Figures sit next to main.tex so the Overleaf project can reference them
    # as figures/<name>.pdf without recreating a nested output tree.
    d = ROOT / "figures"
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return d / f"{stem}.pdf"


# ------------------------------------------------------------------- tables
def _fmt(v, nd=2):
    return "---" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def table_main(att: dict) -> Path:
    """Table 3 in the manuscript: primary behavioural-response estimates."""
    comp = att.get("_rq5_comparison", {})
    evs = comp.get("events", {})
    a, b = evs.get("statement2023"), evs.get("warning2026")

    lines = [
        r"% Auto-generated by src/build_outputs.py -- do not edit by hand.",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Outcome & 2023 effect & 2026 effect & Difference \\", r"\midrule",
    ]
    for fam in FAMILY_ORDER:
        x = a["families"].get(fam) if a else None
        y = b["families"].get(fam) if b else None
        diff = (y - x) if (x is not None and y is not None) else None
        lines.append(f"{FAMILY_LABEL[fam]} attention & {_fmt(x)} & {_fmt(y)} "
                     f"& {_fmt(diff)} \\\\")
    xa = a["tpg_mean_daily"] if a else None
    yb = b["tpg_mean_daily"] if b else None
    lines.append(f"Threat--preparedness gap & {_fmt(xa)} & {_fmt(yb)} & "
                 f"{_fmt((yb - xa) if (xa is not None and yb is not None) else None)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    p = OUTPUTS / "tables" / "table_main_results.tex"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    return p


def table_orientation(att: dict) -> Path:
    comp = att.get("_rq5_comparison", {}).get("events", {})
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Event & Curiosity $C$ & Alarm $L$ & Preparedness $P$ & TPG \\",
             r"\midrule"]
    for eid, d in comp.items():
        o = d["orientation"]
        lines.append(f"{eid} & {_fmt(o['curiosity'])} & {_fmt(o['alarm'])} & "
                     f"{_fmt(o['preparedness'])} & {_fmt(d['tpg_mean_daily'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    p = OUTPUTS / "tables" / "table_orientation_vector.tex"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    return p


def table_market(mkt: dict) -> Path | None:
    if not mkt:
        return None
    lines = [r"\begin{tabular}{lllrrr}", r"\toprule",
             r"Event & Window & Portfolio & $N$ & CAR (\%) & Rand. $p$ \\",
             r"\midrule"]
    any_row = False
    for eid, res in mkt["events"].items():
        spec = res.get("specifications", {}).get("sector_adjusted")
        if not spec:
            continue
        for wkey, w in spec["windows"].items():
            if not w.get("available"):
                continue
            for pname, p in w.get("by_portfolio", {}).items():
                any_row = True
                lines.append(
                    f"{eid} & {wkey} & {PORT_LABEL[pname]} & {p['n']} & "
                    f"{_fmt(100 * p['car_equal_weighted'])} & "
                    f"{_fmt(p.get('randomisation_p'), 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if not any_row:
        return None
    p = OUTPUTS / "tables" / "table_market_car.tex"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    return p


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    att = load("attention_results.json")
    mkt = load("market_results.json")
    if att is None:
        raise SystemExit("no attention results; run src/analyze_attention.py first")

    events_cfg, pages_cfg = load_events(), load_pages()
    written = []
    written.append(figure_attention(att, events_cfg))
    written.append(figure_gap(att))
    # figure_frames() is retained in the module as a supplementary view; the
    # transmission plot below reports the same relationship as estimates.
    for fn in (lambda: figure_transmission(att),
               lambda: figure_market(mkt),
               lambda: figure_divergence(att, mkt)):
        p = fn()
        if p:
            written.append(p)
    written.append(table_main(att))
    written.append(table_orientation(att))
    t3 = table_market(mkt)
    if t3:
        written.append(t3)

    for p in written:
        print(f"  wrote {p.relative_to(ROOT)}")
    write_manifest("build_outputs", files=[str(p.name) for p in written])


if __name__ == "__main__":
    main()
