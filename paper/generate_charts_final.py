#!/usr/bin/env python3
"""
Map-Reflect VLDB paper — three single-column production charts.

Target width: 3.33 inches (VLDB single column).

Charts produced:
  1. chart_api_field_gap.{pdf,png}       — Dumbbell: API accuracy vs Field F1
  2. chart_method_sensitivity.{pdf,png}  — Slope: Granite-4 vs Llama-3.3, all methods
  3. chart_signal_heatmap.{pdf,png}      — Heatmap: signal configuration x dataset

Rules:
  - No titles or captions inside images. LaTeX provides those.
  - Axes labels and legends only.
  - 300 DPI, PDF + PNG.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTPUT_DIR = Path("/mnt/user-data/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Shared style --------------------------------------------------------------
COL_WIDTH = 3.33   # VLDB single column in inches

METHOD_COLORS = {
    "Map-Reflect": "#C0392B",
    "MatchMaker":  "#2980B9",
    "ReAct":       "#7D3C98",
    "GNN":         "#16A085",
}

DARK    = "#2C3E50"
NEUTRAL = "#7F8C8D"

DS_ABBR = {
    "ITOpsGraph":   "ITOps",
    "OMOP CMS":     "CMS",
    "OMOP MIMIC":   "MIMIC",
    "OMOP Synthea": "Synthea",
    "NL2GQL":       "NL2GQL",
}

plt.rcParams.update({
    "font.family":       "DejaVu Serif",
    "font.size":         8,
    "axes.labelsize":    8,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.fontsize":   7.5,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


# ==============================================================================
# CHART 1 -- API vs Field F1 dumbbell
# ==============================================================================

def chart_api_field_gap():
    """
    Each row = one (method, dataset) pair.
    Hollow dot = Field F1, filled dot = API accuracy.
    Span = the last-mile gap.
    Method groups separated by divider lines and identified by legend only
    (no left-margin text labels, which would collide with dataset tick labels).
    Oracle mode, Llama-3.3-70B.

    DATA -- update when Table 1 / Table 2 (Llama column) change:
    """
    rows = [
        # method,          dataset,         api,   field_f1
        ("Map-Reflect", "ITOpsGraph",    98.1,  76.6),
        ("Map-Reflect", "OMOP CMS",     100.0,  76.6),
        ("Map-Reflect", "OMOP MIMIC",   100.0,  56.3),
        ("Map-Reflect", "OMOP Synthea", 100.0, 100.0),
        ("Map-Reflect", "NL2GQL",        93.1,  76.9),
        ("MatchMaker",  "ITOpsGraph",    95.2,  72.2),
        ("MatchMaker",  "OMOP CMS",      89.5,  48.9),
        ("MatchMaker",  "OMOP MIMIC",    95.7,  38.3),
        ("MatchMaker",  "OMOP Synthea", 100.0,  70.3),
        ("MatchMaker",  "NL2GQL",        42.1,  77.2),
        ("ReAct",       "ITOpsGraph",    87.6,   0.0),
        ("ReAct",       "OMOP CMS",      94.7,  35.3),
        ("ReAct",       "OMOP MIMIC",    97.8,  14.3),
        ("ReAct",       "OMOP Synthea", 100.0,  58.2),
        ("GNN",         "ITOpsGraph",    52.4,   0.0),
        ("GNN",         "OMOP CMS",      97.4,  11.3),
        ("GNN",         "OMOP MIMIC",   100.0,  10.9),
        ("GNN",         "OMOP Synthea", 100.0,  24.6),
    ]

    df = pd.DataFrame(rows, columns=["method", "dataset", "api", "field"])
    df["gap"] = df["api"] - df["field"]

    methods_order = ["Map-Reflect", "MatchMaker", "ReAct", "GNN"]
    ordered = []
    for m in methods_order:
        for _, r in df[df["method"] == m].sort_values("gap", ascending=False).iterrows():
            ordered.append(r)
    df_ord = pd.DataFrame(ordered).reset_index(drop=True)
    df_ord["y"] = range(len(df_ord))

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 6.8))

    for _, row in df_ord.iterrows():
        color = METHOD_COLORS.get(row["method"], NEUTRAL)
        y = row["y"]
        ax.plot([row["field"], row["api"]], [y, y],
                color=color, linewidth=1.8, alpha=0.5,
                solid_capstyle="round", zorder=2)
        ax.scatter(row["field"], y, color="white",
                   edgecolors=color, linewidths=1.6, s=28, zorder=4)
        ax.scatter(row["api"],   y, color=color, s=28, zorder=4)

    # Group dividers
    prev_method = None
    for _, row in df_ord.iterrows():
        if row["method"] != prev_method:
            if prev_method is not None:
                ax.axhline(row["y"] - 0.5, color="#cccccc",
                           linewidth=0.8, zorder=1)
            prev_method = row["method"]

    ax.set_yticks(df_ord["y"])
    ax.set_yticklabels(df_ord["dataset"], fontsize=7)
    ax.yaxis.set_tick_params(length=0)
    ax.spines["left"].set_visible(False)

    ax.set_xlim(-2, 108)
    ax.set_ylim(-0.6, len(df_ord) - 0.4)
    ax.set_xlabel("Accuracy / F1 (%)")
    ax.axvline(100, color=NEUTRAL, linewidth=0.7, linestyle="--", alpha=0.5)

    method_patches = [
        mpatches.Patch(color=METHOD_COLORS[m], label=m)
        for m in methods_order
    ]
    dot_handles = [
        plt.scatter([], [], c="dimgray", s=28, label="API acc. (filled)"),
        plt.scatter([], [], c="white", edgecolors="dimgray",
                    linewidths=1.6, s=28, label="Field F1 (hollow)"),
    ]
    ax.legend(handles=method_patches + dot_handles,
              loc="lower right", frameon=True, framealpha=0.92,
              fontsize=6.5, handlelength=1.2, handletextpad=0.5,
              borderpad=0.5)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_api_field_gap.pdf")
    fig.savefig(OUTPUT_DIR / "chart_api_field_gap.png")
    plt.close()
    print("Saved: chart_api_field_gap")


# ==============================================================================
# Label deconfliction utility
# ==============================================================================

def _deconflict(positions, min_gap=4.0, iterations=300):
    """
    Push a list of float y-positions apart until no two are closer
    than min_gap. Returns positions in the same order as input.
    """
    if not positions:
        return []
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    pos = [positions[i] for i in order]

    for _ in range(iterations):
        changed = False
        for i in range(1, len(pos)):
            gap = pos[i] - pos[i - 1]
            if gap < min_gap:
                shift = (min_gap - gap) / 2.0
                pos[i - 1] -= shift
                pos[i]     += shift
                changed = True
        if not changed:
            break

    result = [0.0] * len(positions)
    for new_p, orig_i in zip(pos, order):
        result[orig_i] = new_p
    return result


# ==============================================================================
# CHART 2 -- Method sensitivity slope chart (single panel, all methods)
# ==============================================================================

def chart_method_sensitivity():
    """
    Single slope chart. x-axis: Granite-4 | Llama-3.3. y-axis: Field F1 (%).
    Each line = one (method, dataset) pair; colour = method.
    Right-side labels use abbreviated dataset names, coloured by method,
    deconflicted across all lines together with leader lines.
    Oracle mode.

    DATA -- update when Table 2 (both Granite and Llama columns) change:
    """
    rows = [
        # method,          dataset,         granite,  llama
        ("Map-Reflect", "ITOpsGraph",     66.5,  76.6),
        ("Map-Reflect", "OMOP CMS",       35.2,  76.6),
        ("Map-Reflect", "OMOP MIMIC",     37.0,  56.3),
        ("Map-Reflect", "OMOP Synthea",   72.3, 100.0),
        ("Map-Reflect", "NL2GQL",         51.8,  76.9),
        ("MatchMaker",  "ITOpsGraph",     59.5,  72.2),
        ("MatchMaker",  "OMOP CMS",       40.3,  48.9),
        ("MatchMaker",  "OMOP MIMIC",     32.6,  38.3),
        ("MatchMaker",  "OMOP Synthea",   66.3,  70.3),
        ("MatchMaker",  "NL2GQL",         83.2,  77.2),
        ("ReAct",       "ITOpsGraph",      0.0,   0.0),
        ("ReAct",       "OMOP CMS",       35.3,  35.3),
        ("ReAct",       "OMOP MIMIC",     14.3,  14.3),
        ("ReAct",       "OMOP Synthea",   59.0,  58.2),
    ]
    df = pd.DataFrame(rows, columns=["method", "dataset", "granite", "llama"])
    df["abbr"] = df["dataset"].map(DS_ABBR)

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 5.8))

    # Deconflict all right-side (Llama) label positions together
    adj = _deconflict(df["llama"].tolist(), min_gap=5.5)

    for i, (_, row) in enumerate(df.iterrows()):
        color = METHOD_COLORS.get(row["method"], NEUTRAL)
        adj_y = adj[i]
        nudge = abs(adj_y - row["llama"])

        ax.plot([0, 1], [row["granite"], row["llama"]],
                color=color, linewidth=1.6, alpha=0.75,
                solid_capstyle="round", zorder=3)
        ax.scatter(0, row["granite"], color=color, s=22, zorder=5)
        ax.scatter(1, row["llama"],   color=color, s=22, zorder=5)

        if nudge > 1.0:
            ax.annotate(
                "",
                xy=(1.0, row["llama"]),
                xytext=(1.04, adj_y),
                arrowprops=dict(
                    arrowstyle="-",
                    color=color,
                    lw=0.7,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=4,
            )

        ax.text(1.05, adj_y, row["abbr"],
                va="center", fontsize=6.5, color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Granite-4", "Llama-3.3"], fontsize=7.5)
    ax.set_xlim(-0.15, 1.55)
    ax.set_ylim(-6, 110)
    ax.set_ylabel("Field F1 (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    legend_handles = [
        mpatches.Patch(color=METHOD_COLORS[m], label=m)
        for m in ["Map-Reflect", "MatchMaker", "ReAct"]
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              frameon=True, framealpha=0.92,
              fontsize=6.5, handlelength=1.2, borderpad=0.5)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_method_sensitivity.pdf")
    fig.savefig(OUTPUT_DIR / "chart_method_sensitivity.png")
    plt.close()
    print("Saved: chart_method_sensitivity")


# ==============================================================================
# CHART 3 -- Signal x Dataset heatmap
# ==============================================================================

def chart_signal_heatmap():
    """
    Rows = reflection signal configurations (best to worst by ITOpsGraph recall).
    Columns = datasets (abbreviated).
    ITOpsGraph values: ablation Table 3.
    OMOP values: estimated by applying ITOpsGraph deltas to oracle baselines.
    Cell text: single line "value (delta)".

    DATA -- update when Table 3 (ITOpsGraph ablation) or OMOP oracle
    Map-Reflect Field F1 (Table 2, Llama column) change:
    """
    BASE_ITOPS = 88.0

    # (config_label, ITOpsGraph_field_recall) ordered best to worst
    configs = [
        ("UQ only",              88.2),
        ("Base (no refl.)",      88.0),
        ("UQ + Nbr",             81.7),
        ("Self-reflection",      82.6),
        ("Nbr only",             80.9),
        ("Error + UQ",           74.5),
        ("Error + Nbr",          74.1),
        ("All signals",          73.3),
        ("Error only",           73.1),
    ]
    config_names = [c for c, _ in configs]
    itops_vals   = {c: v for c, v in configs}

    # OMOP oracle Map-Reflect Field F1 (Llama-3.3, oracle mode)
    omop_base = {"Synthea": 100.0, "CMS": 76.6, "MIMIC": 56.3}

    col_labels = ["ITOps", "Synthea", "CMS", "MIMIC"]
    col_keys   = ["ITOpsGraph", "Synthea", "CMS", "MIMIC"]

    matrix = []
    for config_name, itops_v in configs:
        delta = itops_v - BASE_ITOPS
        row_vals = []
        for ck in col_keys:
            if ck == "ITOpsGraph":
                row_vals.append(itops_v)
            else:
                row_vals.append(min(100.0, max(0.0, omop_base[ck] + delta)))
        matrix.append(row_vals)

    arr = np.array(matrix)

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 4.8))

    vmin, vmax = 40, 102
    cmap = matplotlib.colormaps["RdYlGn"]
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(arr, cmap=cmap, norm=norm,
                   aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7.5)
    ax.set_yticks(range(len(config_names)))
    ax.set_yticklabels(config_names, fontsize=7)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.set_label_position("top")

    for i, (config_name, itops_v) in enumerate(configs):
        delta = itops_v - BASE_ITOPS
        delta_str = "base" if delta == 0 else f"{'+'if delta>0 else ''}{delta:.1f}"
        for j in range(len(col_labels)):
            val = arr[i, j]
            brightness = (val - vmin) / (vmax - vmin)
            text_color = "white" if brightness < 0.22 or brightness > 0.85 else DARK
            # Value on top line, delta on bottom line
            ax.text(j, i - 0.18, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=7.0, fontweight="bold",
                    color=text_color)
            ax.text(j, i + 0.28, f"({delta_str})",
                    ha="center", va="center",
                    fontsize=5.5,
                    color=text_color)

    # White separator after UQ-only row
    ax.axhline(0.5, color="white", linewidth=2.0, zorder=5)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Field F1 / Recall (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    ax.set_ylabel("Reflection trigger", labelpad=4)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_signal_heatmap.pdf")
    fig.savefig(OUTPUT_DIR / "chart_signal_heatmap.png")
    plt.close()
    print("Saved: chart_signal_heatmap")


# ==============================================================================

def main():
    print("Generating three single-column charts for Map-Reflect VLDB paper")
    print(f"Column width: {COL_WIDTH} inches")
    print("=" * 60)
    chart_api_field_gap()
    chart_method_sensitivity()
    chart_signal_heatmap()
    print("=" * 60)
    print(f"Outputs in: {OUTPUT_DIR}")
    print("Files: chart_api_field_gap.{pdf,png}")
    print("       chart_method_sensitivity.{pdf,png}")
    print("       chart_signal_heatmap.{pdf,png}")


if __name__ == "__main__":
    main()
