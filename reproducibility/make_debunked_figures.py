"""Export topic profiles to CSV and render the debunked-claim figures.

Reads youtube_results.json (profiles) and youtube_raw.json (traces) and writes:

  topic_profiles.csv        all 37 topics, one row per topic
  debunked_profiles.csv     the 19 debunked-claim topics only
  debunked_profile_space.pdf/.png   coverage x recency ratio, debunked vs known-pattern
  debunked_reemergence.pdf/.png     R_h ranked, with reactivation counts
  debunked_traces.pdf/.png          quarterly traces for four exemplars

Labels use the same thresholds stated in the caption of Table 6: coverage 55%
separates persistent from intermittent, recency ratio 10 separates
recency-weighted from strongly recency-weighted.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
W = {"publication": 1.0, "like": 0.3, "comment": 0.7}
WINDOW = timedelta(days=90)

COVERAGE_CUT = 0.55
RECENCY_CUT = 10.0

# Validated categorical pair (scripts/validate_palette.js, light mode: all pass).
# Marker shape carries the same distinction, so identity is never colour-alone.
C_DEBUNKED = "#2b6cb0"
C_KNOWN = "#c05621"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"

# Display names as they appear in Table 6.
DISPLAY = {
    "flat earth": "flat earth", "pizzagate": "pizzagate",
    "5g coronavirus": "5G–coronavirus", "moon landing hoax": "moon-landing hoax",
    "chemtrails": "chemtrails", "bigfoot sighting": "bigfoot",
    "loch ness monster": "Loch Ness monster", "area 51 aliens": "Area 51 aliens",
    "climate change hoax": "climate change (hoax)", "mandela effect": "Mandela effect",
    "birds aren't real": "Birds aren't real", "hollow earth": "Hollow Earth",
    "phantom time hypothesis": "phantom time hypothesis",
    "project blue beam": "Project Blue Beam", "haarp weather control": "HAARP weather control",
    "paul is dead conspiracy": "Paul McCartney (“Paul is dead”)",
    "fluoride water poisoning": "fluoride water poisoning", "mothman": "Mothman",
    "cern portals": "CERN portals",
}

# Exemplars for the trace figure: one per argument made in Section 6.6.
TRACE_EXEMPLARS = [
    ("flat earth", "Persistent, strongly recency-weighted"),
    # Metrically the lowest ratio in the set, though 61% of its engagement falls in
    # the final quarter of its windows -- see the note in the caption.
    ("5g coronavirus", "Recent origin, lowest recency ratio (1.4)"),
    ("bigfoot sighting", "Long horizon, most recency-weighted"),
    ("fluoride water poisoning", "Strongest re-emergence in the set"),
]

# Points worth naming on the scatter; the rest stay unlabelled to avoid clutter.
# (dx, dy) in points, plus a leader line where the offset is large enough that the
# label would otherwise be ambiguous among neighbouring marks.
SCATTER_LABELS = {
    "bigfoot sighting": (16, 14, True), "cern portals": (-50, -24, True),
    "5g coronavirus": (8, 3, False), "birds aren't real": (10, 5, False),
    "flat earth": (9, 5, False), "mothman": (9, 9, False),
    "project blue beam": (9, 3, False), "fluoride water poisoning": (-30, 20, True),
    "harlem shake": (9, -4, False), "meditation music": (12, -9, True),
    "chatgpt": (-12, 9, False), "summer olympics": (11, 6, False),
}


def label_for(cov: float, recency: float) -> str:
    a = "persistent" if cov >= COVERAGE_CUT else "intermittent"
    b = "strongly recency-weighted" if recency >= RECENCY_CUT else "recency-weighted"
    return f"{a}, {b}"


def load():
    res = json.loads((HERE / "youtube_results.json").read_text())
    raw = json.loads((HERE / "youtube_raw.json").read_text())
    rows = []
    for topic, p in res["topics"].items():
        cov = p["active_windows"] / p["windows"]
        rows.append({
            "topic": DISPLAY.get(topic, topic),
            "query": topic,
            "group": "debunked" if p["pattern"] == "debunked" else "known-pattern",
            "expected_pattern": "" if p["pattern"] == "debunked" else p["pattern"],
            "n_videos": p["n_videos"],
            "span_start": p["span"][0],
            "span_end": p["span"][1],
            "span_years": p["span_years"],
            "windows": p["windows"],
            "active_windows": p["active_windows"],
            # Raw values: figures and the CSV must not disagree with Table 6, so
            # round once, at the point of display, never twice.
            "coverage": cov,
            "Ph": p["Ph"],
            "thgee": p["thgee"],
            "recency_ratio": p["recency_ratio"],
            "R_h": p["R"],
            "n_reactivations": p["n_reactivations"],
            "profile": label_for(cov, p["recency_ratio"]) if p["pattern"] == "debunked" else "",
        })
    return res, raw, rows


# Enough precision that re-rounding a CSV value to the paper's display precision
# always reproduces the printed digit; 4 dp did not (Area 51's R_h 0.2625 -> 0.263
# against the table's 0.262).
CSV_PRECISION = {"coverage": 6, "Ph": 6, "thgee": 6, "recency_ratio": 6, "R_h": 6}


def export_csv(rows):
    fields = list(rows[0].keys())
    for name, subset in [("topic_profiles.csv", rows),
                         ("debunked_profiles.csv", [r for r in rows if r["group"] == "debunked"])]:
        with (HERE / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows({k: round(v, CSV_PRECISION[k]) if k in CSV_PRECISION else v
                         for k, v in r.items()} for r in subset)
        print(f"wrote {name} ({len(subset)} rows)")


def save(fig, stem):
    for target in (HERE, HERE.parent / "analysis" / "figures"):
        target.mkdir(parents=True, exist_ok=True)
        fig.savefig(target / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(target / f"{stem}.png", dpi=200, bbox_inches="tight")
    print(f"wrote {stem}.pdf/.png")
    plt.close(fig)


def fig_profile_space(rows):
    """Coverage against recency ratio: the two axes that separate this set."""
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)

    ax.axvline(COVERAGE_CUT * 100, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.axhline(RECENCY_CUT, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.text(COVERAGE_CUT * 100 + 1.0, 44, "coverage 55%", fontsize=7, color=MUTED)
    ax.text(97.5, RECENCY_CUT * 1.14, "recency ratio 10", fontsize=7, color=MUTED, ha="right")

    for group, colour, marker, size in [("debunked", C_DEBUNKED, "o", 52),
                                        ("known-pattern", C_KNOWN, "^", 46)]:
        sel = [r for r in rows if r["group"] == group]
        ax.scatter([r["coverage"] * 100 for r in sel], [r["recency_ratio"] for r in sel],
                   s=size, marker=marker, facecolor=colour, edgecolor="white",
                   linewidth=0.8, alpha=0.9, zorder=3,
                   label=f"{group} topics (n={len(sel)})")

    for r in rows:
        spec = SCATTER_LABELS.get(r["query"])
        if spec:
            dx, dy, leader = spec
            ax.annotate(r["topic"], (r["coverage"] * 100, r["recency_ratio"]),
                        textcoords="offset points", xytext=(dx, dy), fontsize=7, color=INK,
                        ha="right" if dx < 0 else "left", va="center", zorder=4,
                        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6,
                                        shrinkA=1, shrinkB=4) if leader else None)

    ax.set_yscale("log")
    ax.set_xlabel("Active-window coverage (%)", fontsize=9)
    ax.set_ylabel("Recency ratio  TH-GEE / $P_h$  (log scale)", fontsize=9)
    ax.set_title("Debunked claims occupy the same profile space as benign topics",
                 fontsize=10.5, loc="left", color=INK)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    leg = ax.legend(fontsize=8, frameon=False, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    save(fig, "debunked_profile_space")


def fig_reemergence(rows):
    """R_h for all nineteen: shared property, widely differing magnitude."""
    sel = sorted((r for r in rows if r["group"] == "debunked"), key=lambda r: r["R_h"])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    y = np.arange(len(sel))
    ax.barh(y, [r["R_h"] for r in sel], height=0.62, color=C_DEBUNKED, edgecolor="none", zorder=3)
    ax.set_yticks(y, [r["topic"] for r in sel], fontsize=8)
    for i, r in enumerate(sel):
        ax.text(r["R_h"] + 0.006, i, f"{r['R_h']:.3f}  ({r['n_reactivations']})",
                va="center", fontsize=7.5, color=MUTED)

    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_xlim(0, max(r["R_h"] for r in sel) * 1.28)
    ax.set_xlabel("Re-Emergence Index $R_h$   (reactivation count in parentheses)", fontsize=9)
    ax.set_title("Every debunked topic re-emerges; the magnitude varies more than fourfold",
                 fontsize=10.5, loc="left", color=INK)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    fig.tight_layout()
    save(fig, "debunked_reemergence")


def trace(videos):
    times = [datetime.fromisoformat(v["publishedAt"]) for v in videos]
    start, end = min(times), max(times)
    T = max(1, int(math.ceil((end - start) / WINDOW)))
    num, counts = np.zeros(T), np.zeros(T)
    for v, t in zip(videos, times):
        idx = min(T - 1, int((t - start) / WINDOW))
        num[idx] += W["publication"] + W["like"] * v["like"] + W["comment"] * v["comment"]
        counts[idx] += 1
    edges = [start + i * WINDOW for i in range(T)]
    return edges, np.divide(num, counts, out=np.zeros_like(num), where=counts > 0)


def fig_traces(raw, rows):
    by_query = {r["query"]: r for r in rows}
    fig, axes = plt.subplots(len(TRACE_EXEMPLARS), 1, figsize=(7.2, 7.0))
    for ax, (query, note) in zip(axes, TRACE_EXEMPLARS):
        edges, tr = trace(raw[query]["videos"])
        r = by_query[query]
        ax.bar(edges, tr, width=80, align="edge", color=C_DEBUNKED, edgecolor="none", zorder=3)
        ax.set_title(f"“{r['topic']}” — {note}", fontsize=9.5, loc="left", color=INK)
        ax.annotate(f"coverage {r['coverage'] * 100:.0f}%   recency {r['recency_ratio']:.1f}   "
                    f"$R_h$ {r['R_h']:.3f} ({r['n_reactivations']} reactivations)",
                    xy=(0.995, 0.86), xycoords="axes fraction", ha="right",
                    fontsize=7.5, color=MUTED)
        ax.set_ylabel("E$_\\tau$", fontsize=9)
        ax.margins(x=0.01)
        ax.set_ylim(0, tr.max() * 1.32)
        ax.tick_params(labelsize=8)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
    axes[-1].set_xlabel("Time (video publication date, 90-day windows)", fontsize=9)
    fig.suptitle("Quarterly engagement traces for four debunked-claim exemplars",
                 fontsize=11, y=0.997, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "debunked_traces")


def main():
    _, raw, rows = load()
    export_csv(rows)
    fig_profile_space(rows)
    fig_reemergence(rows)
    fig_traces(raw, rows)


if __name__ == "__main__":
    main()
