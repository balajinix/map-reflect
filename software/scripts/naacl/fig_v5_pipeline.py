#!/usr/bin/env python3
"""Figure 1 of the v5 paper: the Map-Reflect v5 pipeline. Writes paper/figures/fig_v5_pipeline.pdf.

Usage: .venv/bin/python scripts/naacl/fig_v5_pipeline.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "paper/figures"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 6.6})
BLUE, ORANGE, GREEN, PURPLE, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#52514e"
FILL = {BLUE: "#eaf2fc", ORANGE: "#fdeee7", GREEN: "#e6f6ef", PURPLE: "#eeecf8", GREY: "#f3f2ef"}

fig = plt.figure(figsize=(3.15, 3.7))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 114.5)
ax.axis("off")


def box(x, y, w, h, col, title, lines, tsize=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.8", fc=FILL[col], ec=col, lw=1.0))
    ax.text(x + 1.5, y + h - 1.6, title, ha="left", va="top", fontsize=tsize, fontweight="bold", color=col)
    ax.text(x + 1.5, y + h - 7.0, "\n".join(lines), ha="left", va="top", fontsize=7.4, color="#0b0b0b",
            linespacing=1.25)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.9, color=GREY))


# first pass
box(2, 92, 96, 21, BLUE, "MAP  over a covering list",
    ["the target type in retrieval order (up to 60) + NONE",
     r"Alert.severity $\rightarrow$ ?    1 priority 0.62",
     "2 criticalityLevel 0.58    3 NONE 0.10    4 status 0.05"])
arrow(50, 91.5, 50, 87.5)
# signals
box(2, 66, 46, 21, ORANGE, "Selection uncertainty",
    [r"margin $\gamma_1-\gamma_2$, top score:", "is the top answer", "clearly ahead?"], tsize=7.8)
box(52, 66, 46, 21, ORANGE, "Coverage uncertainty",
    ["NONE score and rank:", "is the answer in the", "list at all?"], tsize=7.8)
arrow(25, 65.5, 45, 60.5)
arrow(75, 65.5, 55, 60.5)
# router
box(2, 44, 96, 16, ORANGE, "ROUTE  frozen per-model router",
    ["+ structural features (declined, sizes, position)", r"P(wrong) $\geq$ threshold, or declined: second pass"],
    tsize=8.0)
arrow(20, 43.5, 20, 39.5)
arrow(72, 43.5, 72, 39.5)
box(2, 17, 36, 22, GREEN, "Keep", ["the first answer", "stands, no extra", "call"])
box(42, 17, 56, 22, PURPLE, "REFLECT + CHECK",
    ["neutral prompt, previous", "answer, new evidence;", "an old-vs-new check", "confirms any change"])
arrow(20, 16.5, 38, 12.3)
arrow(70, 16.5, 60, 12.3)
box(12, 1.5, 76, 10.5, GREY, "Mapping", [r"Alert.severity $\rightarrow$ Incident.priority"], tsize=8.0)

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "fig_v5_pipeline.pdf")
fig.savefig(OUT / "fig_v5_pipeline.png", dpi=220)
