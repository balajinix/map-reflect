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

OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Shared style --------------------------------------------------------------
COL_WIDTH = 3.33   # VLDB single column in inches

METHOD_COLORS = {
    "Map-Reflect": "#C0392B",
    "MatchMaker":  "#2980B9",
    "AnyTool":     "#8E44AD",
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
    "ToolBench":    "TBench",
    "Spider":       "Spider",
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
        # method,          dataset,         api(type F1),  field_f1
        # Audited numbers from tables_2_3_final_audit.md (Apr 2026)
        ("Map-Reflect", "ITOpsGraph",    94.4,  83.1),
        ("Map-Reflect", "OMOP CMS",     100.0,  76.6),
        ("Map-Reflect", "OMOP MIMIC",   100.0,  59.7),
        ("Map-Reflect", "OMOP Synthea", 100.0, 100.0),
        ("Map-Reflect", "NL2GQL",        70.5,  63.2),
        ("Map-Reflect", "ToolBench",     96.3,  94.4),
        ("Map-Reflect", "Spider",        95.0,  75.2),
        ("MatchMaker",  "ITOpsGraph",    86.1,  36.3),
        ("MatchMaker",  "OMOP CMS",      98.7,  70.4),
        ("MatchMaker",  "OMOP MIMIC",    95.5,  56.5),
        ("MatchMaker",  "OMOP Synthea", 100.0,  71.0),
        ("MatchMaker",  "NL2GQL",        61.1,  53.0),
        ("MatchMaker",  "ToolBench",     89.4,  87.7),
        ("MatchMaker",  "Spider",        83.9,  70.5),
        ("AnyTool",     "ITOpsGraph",    43.3,  48.4),
        ("AnyTool",     "OMOP CMS",      91.9,  60.5),
        ("AnyTool",     "OMOP MIMIC",    47.7,  49.5),
        ("AnyTool",     "OMOP Synthea", 100.0,  72.3),
        ("AnyTool",     "NL2GQL",        60.0,  32.9),
        ("AnyTool",     "ToolBench",     69.9,  77.1),
        ("AnyTool",     "Spider",        86.6,  86.7),
    ]

    df = pd.DataFrame(rows, columns=["method", "dataset", "api", "field"])
    df["gap"] = df["api"] - df["field"]

    methods_order = ["Map-Reflect", "MatchMaker", "AnyTool"]
    ordered = []
    for m in methods_order:
        for _, r in df[df["method"] == m].sort_values("gap", ascending=False).iterrows():
            ordered.append(r)
    df_ord = pd.DataFrame(ordered).reset_index(drop=True)
    df_ord["y"] = range(len(df_ord))

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 5.5))

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
    ax.set_xlabel("F1 (%)")
    ax.axvline(100, color=NEUTRAL, linewidth=0.7, linestyle="--", alpha=0.5)

    method_patches = [
        mpatches.Patch(color=METHOD_COLORS[m], label=m)
        for m in methods_order
    ]
    dot_handles = [
        plt.scatter([], [], c="dimgray", s=28, label="Type F1 (filled)"),
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
    Slope chart with three models. x-axis: Granite-4 | Llama-3.3 | Mistral.
    y-axis: Field F1 (%).
    Each line = one (method, dataset) pair; colour = method.
    Right-side labels use abbreviated dataset names, coloured by method,
    deconflicted across all lines together with leader lines.
    Oracle mode.

    DATA -- update when Table 3 (all three model columns) change:
    """
    rows = [
        # method,          dataset,         granite,  llama,  mistral
        # Audited numbers from tables_2_3_final_audit.md (Apr 2026)
        ("Map-Reflect", "ITOpsGraph",     54.6,  83.1,  85.4),
        ("Map-Reflect", "OMOP CMS",       59.5,  76.6,  71.0),
        ("Map-Reflect", "OMOP MIMIC",     36.6,  59.7,  51.5),
        ("Map-Reflect", "OMOP Synthea",   58.4, 100.0,  58.4),
        ("Map-Reflect", "NL2GQL",         55.4,  63.2,  63.2),
        ("Map-Reflect", "ToolBench",      69.6,  94.4,  92.8),
        ("Map-Reflect", "Spider",         53.6,  75.2,  79.9),
        ("MatchMaker",  "ITOpsGraph",     39.8,  36.3,  37.9),
        ("MatchMaker",  "OMOP CMS",       34.9,  70.4,  19.0),
        ("MatchMaker",  "OMOP MIMIC",     42.8,  56.5,  10.8),
        ("MatchMaker",  "OMOP Synthea",   56.8,  71.0,  19.6),
        ("MatchMaker",  "NL2GQL",         53.6,  53.0,   0.0),
        ("MatchMaker",  "ToolBench",      61.0,  87.7,  53.4),
        ("MatchMaker",  "Spider",         74.1,  70.5,  60.5),
    ]
    df = pd.DataFrame(rows, columns=["method", "dataset", "granite", "llama", "mistral"])
    df["abbr"] = df["dataset"].map(DS_ABBR)

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 5.2))

    # Deconflict all right-side (Llama) label positions together
    adj = _deconflict(df["llama"].tolist(), min_gap=5.0)

    for i, (_, row) in enumerate(df.iterrows()):
        color = METHOD_COLORS.get(row["method"], NEUTRAL)
        adj_y = adj[i]
        nudge = abs(adj_y - row["llama"])

        ax.plot([0, 1, 2], [row["granite"], row["mistral"], row["llama"]],
                color=color, linewidth=1.4, alpha=0.75,
                solid_capstyle="round", zorder=3)
        ax.scatter(0, row["granite"], color=color, s=20, zorder=5)
        ax.scatter(1, row["mistral"], color=color, s=20, zorder=5)
        ax.scatter(2, row["llama"],   color=color, s=20, zorder=5)

        if nudge > 1.0:
            ax.annotate(
                "",
                xy=(2.0, row["llama"]),
                xytext=(2.04, adj_y),
                arrowprops=dict(
                    arrowstyle="-",
                    color=color,
                    lw=0.7,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=4,
            )

        ax.text(2.06, adj_y, row["abbr"],
                va="center", fontsize=6.0, color=color)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Granite-4", "Mistral", "Llama-3.3"], fontsize=7.5)
    ax.set_xlim(-0.15, 2.65)
    ax.set_ylim(-6, 110)
    ax.set_ylabel("Field F1 (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    legend_handles = [
        mpatches.Patch(color=METHOD_COLORS[m], label=m)
        for m in ["Map-Reflect", "MatchMaker"]
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
    BASE_ITOPS = 81.2

    # (config_label, ITOpsGraph_field_recall) ordered best to worst
    # Audited no-cache ablation (Apr 2026, 1429 fields)
    configs = [
        ("UQ only",              81.9),
        ("Nbr only",             81.5),
        ("Base (no refl.)",      81.2),
        ("Self-reflection",      79.7),
        ("Error + UQ",           72.4),
        ("Error only",           72.7),
    ]
    config_names = [c for c, _ in configs]
    itops_vals   = {c: v for c, v in configs}

    # OMOP oracle Map-Reflect Field F1 (Llama-3.3, oracle mode)
    omop_base = {"Synthea": 100.0, "CMS": 76.6, "MIMIC": 59.7}

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
