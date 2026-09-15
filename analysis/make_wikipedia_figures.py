"""Figures for the Wikipedia event-level replication (Online Resource 1).

Produces, into analysis/figures/:

  wikipedia_exemplars     quarterly mean-daily-pageview traces for the five
                          featured topics of paper Table 4, with the qualifying
                          reactivation windows under BOTH q_h rules, which is
                          what makes the threshold finding legible
  wikipedia_profile_space debunked vs comparison cohorts in (persistence,
                          recency) space, the counterpart of paper Figure 2
  wikipedia_vs_youtube    rank-rank agreement per indicator
  wikipedia_bursts        daily series with the three detectors' events shown as
                          separate lanes rather than overplotted rules

Palette: categorical slots 1-3 of the reference palette (blue / orange / aqua),
validated for all-pairs CVD separation. Identity is never carried by colour
alone -- every series is also legended and, in the scatter, given its own marker
shape.

Usage:
    python analysis/make_wikipedia_figures.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from wikipedia_analysis import wiki_trace, _rank  # noqa: E402

FIGDIR = HERE / "figures"
WINDOW_DAYS = 90
START = date(2015, 7, 1)

# --- palette (reference categorical slots 1-3; validated --pairs all) --------
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # blue, orange, aqua
INK, MUTED, FAINT, GRID = "#1a1a19", "#5c5c58", "#8e8e88", "#e6e6e2"
SURFACE = "#ffffff"

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": GRID, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.linestyle": "-", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})

FEATURED = ["meditation music", "harlem shake", "solar eclipse",
            "super bowl halftime", "chatgpt"]
FEATURED_LABEL = {
    "meditation music": "meditation music — persistent",
    "harlem shake": "harlem shake — acute / former",
    "solar eclipse": "solar eclipse — re-emergent (multi-year)",
    "super bowl halftime": "super bowl halftime — re-emergent (annual)",
    "chatgpt": "chatgpt — recent",
}

# Only these get a label in the scatter; the rest would collide into noise.
# Chosen as the topics Online Resource 1 discusses by name.
ANNOTATED = {
    "chatgpt": (9, 0), "threads app": (12, 11), "deepseek ai": (-8, 7),
    "phantom time hypothesis": (24, 5), "birds aren't real": (34, -1),
    "haarp weather control": (20, 7), "5g coronavirus": (16, -8),
    "fluoride water poisoning": (8, 3), "pizzagate": (7, -10),
    "mandela effect": (8, 2), "super bowl halftime": (-12, -12),
    "world cup final": (-24, -12), "solar eclipse": (-30, -12),
    "meditation music": (-10, -12), "bigfoot sighting": (-34, -12),
    "flat earth": (6, 3), "minecraft tutorial": (-60, 2),
    "fidget spinner": (8, 2), "mannequin challenge": (8, -3),
}


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", FIGDIR / f"{name}.pdf")


def _load(name: str):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------- exemplars
def fig_exemplars(raw: dict, quart: dict, elev: dict | None) -> None:
    """Traces with reactivation windows under both q_h rules.

    Showing only the quartile rule leaves panels with no shading at all --
    solar eclipse most conspicuously, whose two eclipse spikes are the clearest
    returns in the dataset. That absence is the finding, but it reads as a
    missing layer unless the alternative rule is drawn beside it.
    """
    n = len(FEATURED)
    fig, axes = plt.subplots(n, 1, figsize=(6.6, 7.6), sharex=True)
    for ax, topic in zip(axes, FEATURED):
        trace = wiki_trace(raw[topic]["views"], WINDOW_DAYS)
        dates = [START + timedelta(days=i * WINDOW_DAYS) for i in range(trace.size)]
        q = quart["wikipedia"][topic]
        e = elev["wikipedia"][topic] if elev else None

        # bands first, so the series line stays readable on top of them
        for w in q["reactivation_windows"]:
            ax.axvspan(dates[w], dates[w] + timedelta(days=WINDOW_DAYS),
                       color=C1, alpha=0.20, lw=0, zorder=0)
        if e:
            for w in e["reactivation_windows"]:
                ax.axvspan(dates[w], dates[w] + timedelta(days=WINDOW_DAYS),
                           color=C2, alpha=0.20, lw=0, zorder=0)

        ax.fill_between(dates, trace, color=INK, alpha=0.06, step="post", zorder=1)
        ax.step(dates, trace, where="post", color=INK, lw=0.9, zorder=3)
        ax.axhline(q["qh"], color=C1, lw=0.7, zorder=2)
        if e:
            ax.axhline(e["qh"], color=C2, lw=0.7, zorder=2)

        ax.set_ylabel("views / day")
        ax.margins(x=0.01)
        note = (f"quartile $q_h$: {q['n_reactivations']} react."
                + (f"   ·   elevation $q_h$: {e['n_reactivations']} react."
                   if e else ""))
        ax.set_title(f"{FEATURED_LABEL[topic]}      {note}", loc="left",
                     color=INK, pad=4)
        ax.tick_params(length=2)

    axes[-1].set_xlabel("90-day window")
    handles = [
        Patch(facecolor=C1, alpha=0.20,
              label="reactivation window, quartile $q_h$ (paper's rule)"),
        Patch(facecolor=C2, alpha=0.20,
              label="reactivation window, elevation $q_h$ ($2\\times$ median)"),
        Line2D([], [], color=C1, lw=0.9, label="quartile $q_h$"),
        Line2D([], [], color=C2, lw=0.9, label="elevation $q_h$"),
        Line2D([], [], color=INK, lw=0.9, label="mean daily pageviews"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper left",
               bbox_to_anchor=(0.085, 1.005), handlelength=1.6,
               labelspacing=0.35, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    _save(fig, "wikipedia_exemplars")


# ----------------------------------------------------------- profile space
def fig_profile_space(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    groups = (
        ("comparison cohorts (18)", C1, "o",
         [(t, p) for t, p in res["wikipedia"].items() if p["pattern"] != "debunked"]),
        ("debunked-claim cohorts (19)", C2, "^",
         [(t, p) for t, p in res["wikipedia"].items() if p["pattern"] == "debunked"]),
    )
    for label, colour, marker, sel in groups:
        ax.scatter([p["ga_ratio"] for _, p in sel],
                   [p["recency_ratio"] for _, p in sel],
                   s=32, c=colour, marker=marker, alpha=0.9,
                   edgecolors=SURFACE, linewidths=0.8, label=label, zorder=3)

    for topic, (dx, dy) in ANNOTATED.items():
        p = res["wikipedia"].get(topic)
        if p is None:
            continue
        # leader lines: in the dense low-persistence cluster, a bare offset
        # label cannot be attributed to its marker
        ax.annotate(topic, (p["ga_ratio"], p["recency_ratio"]), fontsize=6,
                    color=MUTED, xytext=(dx, dy), textcoords="offset points",
                    zorder=4, ha="right" if dx < 0 else "left",
                    va="center",
                    arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.4,
                                    shrinkA=0, shrinkB=3,
                                    connectionstyle="arc3,rad=0"))

    ax.axhline(1.0, color=FAINT, lw=0.6, zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("persistence   $P_h$ / arithmetic mean      "
                  "(1 = engagement evenly spread across windows)")
    ax.set_ylabel("recency   TH-GEE / $P_h$")
    ax.set_xlim(-0.14, 1.08)
    # inside the plot: the upper-middle region is empty, and an outside legend
    # collides with the title
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.30, 0.99),
              ncol=1, handletextpad=0.4, labelspacing=0.5)
    ax.set_title("Event-level temporal-profile space   ·   Wikipedia pageviews, "
                 "90-day windows", loc="left", color=INK, pad=8)
    fig.tight_layout()
    _save(fig, "wikipedia_profile_space")


# ------------------------------------------------------------- rank-rank
def fig_rank_rank(res: dict) -> None:
    metrics = [("ga_ratio", "persistence   $P_h$/mean"),
               ("recency_ratio", "recency   TH-GEE/$P_h$"),
               ("R", "re-emergence   $R_h$"),
               ("n_reactivations", "reactivation count")]
    topics = sorted(t for t in res["wikipedia"] if t in res["youtube_matched"])
    n = len(topics)
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.9), sharey=True)
    for ax, (m, label) in zip(axes, metrics):
        x = _rank([res["youtube_matched"][t][m] for t in topics])
        y = _rank([res["wikipedia"][t][m] for t in topics])
        deb = np.array([res["wikipedia"][t]["pattern"] == "debunked" for t in topics])
        ax.plot([-1, n], [-1, n], color=FAINT, lw=0.7, zorder=1)
        for mask, colour, marker in ((~deb, C1, "o"), (deb, C2, "^")):
            ax.scatter(x[mask], y[mask], s=22, c=colour, marker=marker,
                       alpha=0.9, edgecolors=SURFACE, linewidths=0.7, zorder=3)
        a = res["agreement"]["youtube_matched"]["all"][m]
        sig = "" if a["p"] >= 0.05 else "  ★"
        ax.set_title(f"{label}\n$\\rho$ = {a['rho']:+.2f}   p = {a['p']:.3f}{sig}",
                     loc="left", color=INK, pad=5)
        ax.set_xlabel("YouTube rank")
        ax.set_xlim(-1.5, n + 0.5)
        ax.set_ylim(-1.5, n + 0.5)
        ax.set_xticks([0, 12, 24, 36])
        ax.set_yticks([0, 12, 24, 36])
        ax.tick_params(length=2)
    axes[0].set_ylabel("Wikipedia rank")
    handles = [
        Line2D([], [], color=C1, marker="o", ls="", markersize=4.5,
               markeredgecolor=SURFACE, label="comparison cohort"),
        Line2D([], [], color=C2, marker="^", ls="", markersize=4.5,
               markeredgecolor=SURFACE, label="debunked-claim cohort"),
        Line2D([], [], color=FAINT, lw=0.7, label="exact rank agreement"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper left",
               bbox_to_anchor=(0.065, 1.10), handletextpad=0.4)
    fig.suptitle("Rank agreement: event-level Wikipedia vs publication-cohort "
                 "YouTube   ·   matched horizon, n = 37   ·   ★ p < 0.05",
                 fontsize=8.5, x=0.065, ha="left", y=1.20, color=INK)
    fig.tight_layout()
    _save(fig, "wikipedia_vs_youtube")


# --------------------------------------------------------------- baselines
def fig_bursts(raw: dict, seq: dict) -> None:
    """Series above, detector events as separate lanes below.

    Drawing three detectors as full-height rules over 4,000 daily points buries
    the series in its own annotations and makes co-location impossible to judge.
    Giving each detector its own lane under the series keeps the vertical
    position meaningful (which detector) and leaves the horizontal one free to
    be read for what matters (do the marks line up).
    """
    topics = ["solar eclipse", "super bowl halftime", "flat earth"]
    n_days = seq["params"]["n_days"]
    wd = seq["params"]["window_days"]

    # Print legibility: 4,000 daily points drawn as a hairline reproduce as a
    # grey smear at column width. The series is shown as a filled weekly
    # envelope -- the weekly maximum, which is also the resolution the
    # reactivation windows are computed on -- so a one-day eclipse spike keeps
    # its full height instead of being averaged away. Detection still runs on
    # the daily series.
    n_weeks = n_days // 7
    week_days = np.array([START + timedelta(days=i * 7) for i in range(n_weeks)])

    # Three rows per topic -- series, lane, spacer -- so the lane sits tight
    # under the series it describes and the gap falls between topics. With a
    # single uniform hspace every lane floats midway between two panels and its
    # ownership is ambiguous.
    fig, all_axes = plt.subplots(len(topics) * 3, 1, figsize=(6.8, 5.8),
                                 sharex=True,
                                 gridspec_kw={
                                     "height_ratios": [3.0, 1.15, 0.7] * len(topics),
                                     "hspace": 0.10})
    for k in range(2, len(all_axes), 3):        # spacer rows carry nothing
        all_axes[k].set_visible(False)
    axes = [all_axes[3 * i + j] for i in range(len(topics)) for j in (0, 1)]
    lanes = [("Kleinberg burst onset", "burst_onset_windows", C1),
             ("PELT change point", "changepoint_windows", C2),
             ("$R_h$ reactivation", "reactivation_windows", C3)]

    for i, topic in enumerate(topics):
        ax, lane_ax = axes[2 * i], axes[2 * i + 1]
        v = np.asarray(raw[topic]["views"][:n_days], dtype=float)
        weekly = np.maximum(v[: n_weeks * 7].reshape(n_weeks, 7).max(axis=1), 1.0)
        rec = seq["topics"][topic]

        ax.fill_between(week_days, 1.0, weekly, color=INK, alpha=0.10, lw=0)
        ax.plot(week_days, weekly, color=INK, lw=0.7)
        ax.set_yscale("log")
        ax.set_ylim(weekly.min() * 0.75, weekly.max() * 3.2)
        ax.set_ylabel("views / day", labelpad=2)
        ax.margins(x=0.01)
        ax.tick_params(length=2, which="major")
        ax.tick_params(length=0, which="minor")
        ax.text(0.006, 0.94,
                f"{topic}   ·   {rec['n_bursts']} bursts, "
                f"{rec['n_changepoints']} change points, "
                f"{rec['n_reactivations']} reactivations "
                f"({rec['react_hits_burst']} on a burst onset)",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                color=INK, bbox=dict(facecolor=SURFACE, edgecolor="none",
                                     pad=1.5, alpha=0.9))

        for j, (_, key, colour) in enumerate(lanes):
            xs = [START + timedelta(days=int(w * wd + wd / 2)) for w in rec[key]]
            lane_ax.plot(xs, np.full(len(xs), 2 - j), marker="|", ls="",
                         color=colour, markersize=8, markeredgewidth=1.5)
        lane_ax.set_ylim(-0.8, 2.8)
        lane_ax.set_yticks([2, 1, 0])
        lane_ax.set_yticklabels(["K", "P", "$R_h$"], fontsize=7, color=MUTED)
        lane_ax.grid(False)
        lane_ax.tick_params(length=0)
        for side in ("left", "bottom"):
            lane_ax.spines[side].set_visible(False)
        lane_ax.margins(x=0.01)

    # sharex hands the tick labels to the bottom-most axes, which is a hidden
    # spacer -- put them back on the last lane
    axes[-1].tick_params(axis="x", labelbottom=True, length=2, color=MUTED)
    axes[-1].set_xlabel("day", labelpad=3)
    handles = [Line2D([], [], color=c, marker="|", ls="", markersize=8,
                      markeredgewidth=1.5, label=lbl) for lbl, _, c in lanes]
    # Title and legend live inside the canvas, just above the first panel;
    # anchoring them past y = 1 with bbox_inches="tight" opens a band of
    # whitespace between the heading and the content.
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.suptitle("Detected events on the daily pageview series   ·   "
                 "weekly windows, elevation $q_h$",
                 fontsize=8.5, x=0.062, ha="left", y=0.995, va="top", color=INK)
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper left",
               bbox_to_anchor=(0.062, 0.955), handletextpad=0.5,
               columnspacing=2.2)
    _save(fig, "wikipedia_bursts")


def main() -> None:
    raw = {rec["topic"]: rec for k, rec in
           json.loads((HERE / "wikipedia_raw.json").read_text()).items()
           if k != "_meta" and rec["role"] == "primary"}
    # Each figure is emitted only if its inputs are present. The reproducibility
    # bundle ships just the files behind the manuscript figure, so a reviewer
    # running this there should get that figure rather than a traceback.
    quart = _load("wikipedia_results.json")
    elev = _load("wikipedia_results_90d_common_elevation.json")
    if quart is not None:
        if elev is None:
            print("note: run wikipedia_analysis.py --qh-rule elevation for the "
                  "second reactivation layer in the exemplars figure")
        fig_exemplars(raw, quart, elev)
        fig_profile_space(quart)
        fig_rank_rank(quart)
    else:
        print("skipping exemplars/profile-space/rank figures: "
              "run wikipedia_analysis.py to produce wikipedia_results.json")

    seq = (_load("sequence_baseline_results_7d_elevation.json")
           or _load("sequence_baseline_results.json"))
    if seq is not None:
        fig_bursts(raw, seq)
    else:
        print("skipping the burst figure: run sequence_baselines.py first")


if __name__ == "__main__":
    main()
