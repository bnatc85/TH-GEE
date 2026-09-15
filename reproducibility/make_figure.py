"""Render the YouTube exemplar timelines figure for the paper.

Reads youtube_raw.json and writes figures/youtube_exemplars.pdf: one panel per
featured topic showing the quarterly weighted-engagement trace, annotated with
its temporal pattern.
"""
from __future__ import annotations

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

FEATURED = [
    ("meditation music", "Persistent (evergreen)"),
    ("harlem shake", "Acute / formerly active"),
    ("solar eclipse", "Re-emergent (multi-year)"),
    ("super bowl halftime", "Re-emergent (annual)"),
    ("chatgpt", "Recent breakout"),
]


def trace(videos):
    times = [datetime.fromisoformat(v["publishedAt"]) for v in videos]
    start, end = min(times), max(times)
    T = max(1, int(math.ceil((end - start) / WINDOW)))
    num = np.zeros(T)
    edges = [start + i * WINDOW for i in range(T)]
    for v, t in zip(videos, times):
        idx = min(T - 1, int((t - start) / WINDOW))
        num[idx] += W["publication"] + W["like"] * v["like"] + W["comment"] * v["comment"]
    counts = np.zeros(T)
    for t in times:
        counts[min(T - 1, int((t - start) / WINDOW))] += 1
    trace = np.divide(num, counts, out=np.zeros_like(num), where=counts > 0)
    return edges, trace


def main():
    raw = json.loads((HERE / "youtube_raw.json").read_text())
    fig, axes = plt.subplots(len(FEATURED), 1, figsize=(7.2, 8.4))
    for ax, (topic, label) in zip(axes, FEATURED):
        edges, tr = trace(raw[topic]["videos"])
        ax.bar(edges, tr, width=80, align="edge", color="#2b6cb0", edgecolor="none")
        ax.set_title(f"“{topic}” — {label}", fontsize=10, loc="left")
        ax.set_ylabel("E$_\\tau$", fontsize=9)
        ax.margins(x=0.01)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (video publication date, 90-day windows)", fontsize=9)
    fig.suptitle("Quarterly engagement traces for topics with known temporal patterns",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    (HERE / "figures").mkdir(exist_ok=True)
    out = HERE / "figures" / "youtube_exemplars.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
