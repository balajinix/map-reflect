#!/usr/bin/env python3
"""C14 (docs/naacl/v6/PLAN_PHASE2.md): Map-Reflect's gain per model and dataset cell, run 1 against run 2 (R6).

Reads paper/figures/data6/c14_r6_gain.csv (written by scripts/naacl/v6_calls/r6_analysis.py) and draws
paper/figures/fig6_c14.pdf with the palette, markers and style of scripts/naacl/v6_charts.py (imported, so the
panel matches the other chart pages). Each point is one cell, with its 95% split-group interval in both runs; the
diagonal is perfect agreement. Pooled values per model are marked with a larger hollow symbol.
It can move into v6_charts.py as c14().

Usage: .venv/bin/python scripts/naacl/v6_calls/c14_plot.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "naacl"))
import v6_charts as V  # noqa: E402

plt = V.plt


def main():
    R = V.rd("c14_r6_gain")
    fig, ax = plt.subplots(figsize=(V.COLW, 2.6))
    vals = []
    for r in R:
        if r["model"] == "pooled":
            continue
        m = r["model"]
        g1, g2 = float(r["gain1"]), float(r["gain2"])
        vals += [float(r["gain1_lo"]), float(r["gain1_hi"]), float(r["gain2_lo"]), float(r["gain2_hi"])]
        if r["dataset"] == "all":
            ax.plot([g1], [g2], marker=V.MMRK[m], ms=7, mfc="none", mec=V.MCOL[m], mew=1.2, ls="none", zorder=4,
                    label=f"{V.MNAME[m]}, pooled")
            continue
        ax.plot([float(r["gain1_lo"]), float(r["gain1_hi"])], [g2, g2], color=V.MCOL[m], lw=0.5, alpha=0.5)
        ax.plot([g1, g1], [float(r["gain2_lo"]), float(r["gain2_hi"])], color=V.MCOL[m], lw=0.5, alpha=0.5)
        ax.plot([g1], [g2], marker=V.MMRK[m], ms=3.2, color=V.MCOL[m], ls="none", zorder=3,
                label=f"{V.MNAME[m]}, per dataset" if not any(
                    h.get_label() == f"{V.MNAME[m]}, per dataset" for h in ax.lines) else None)
    lo, hi = min(vals + [0]) - 1, max(vals + [0]) + 1
    ax.plot([lo, hi], [lo, hi], color=V.GREY, lw=0.6, ls=":", zorder=1)
    ax.axhline(0, color=V.LIGHT, lw=0.6, zorder=0)
    ax.axvline(0, color=V.LIGHT, lw=0.6, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Gain in run 1 (points)")
    ax.set_ylabel("Gain in run 2 (points)")
    V.style(ax, None)
    ax.legend(frameon=False, loc="upper left", fontsize=5.4)
    V.save(fig, "c14")
    print("wrote paper/figures/fig6_c14.pdf")


if __name__ == "__main__":
    main()
