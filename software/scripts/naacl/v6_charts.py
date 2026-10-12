#!/usr/bin/env python3
"""The chart pages of the v6 paper (docs/naacl/v6/PLAN_PHASE2.md, C1 to C13 and C15), plus Figure 2 (fig6_pareto) and
the calibration figure (fig6_calibration), drawn only from the per-panel CSVs in paper/figures/data6 that
scripts/naacl/v6_numbers.py (via v6_phase2.py) writes. No analysis happens here, so every plotted number is in a CSV
next to its figure.

Figure 2 and the calibration figure are first drawn by v6_numbers.fig2/fig3. When v6_numbers calls main() at its end,
main() writes their in-memory inputs (FIG2, X4) to data6/pareto.csv, data6/calibration.csv and
data6/calibration_models.csv, and redraws both in this palette; a standalone run redraws them from those CSVs.

One palette for every panel (R10): Okabe-Ito, the colour-blind-safe set of Okabe and Ito (2008).
- the margin, and Map-Reflect wherever it is a mark of its own, are vermilion (never black ink);
- models: gpt-oss-120b blue, Llama-4 orange, Granite-4 bluish green, Qwen3.5 reddish purple, each with its own marker;
- signals: each keeps one colour in every chart, with a line style as the second encoding;
- routers: struct bluish green, every field blue; random routing is grey dashed, the oracle grey dotted;
- bars and areas are filled.

Sizes (paper/STYLE.md section 8): each figure is drawn at the width it is printed at in paper/main_v6.tex (WIDTH),
so one matplotlib point is one printed point, and every text is at least 7 pt. save() measures every text of every
figure at print scale and lists any below 7 pt.

C14 (run 1 against run 2) is drawn here as c14(), from data6/c14_r6_gain.csv (scripts/naacl/v6_calls/c14_plot.py
first drew it with this module's palette). C15 is drawn only when the h1 human-label CSVs exist.

Usage: .venv/bin/python scripts/naacl/v6_charts.py   (after v6_numbers.py)
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/figures/data6"
OUT = ROOT / "paper/figures"
PREVIEW = Path(os.environ.get("V6_PREVIEW_DIR", "/tmp"))

# Okabe-Ito. Yellow (#F0E442) is too pale for a line on white, so no series uses it.
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "vermilion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}
MARGIN = OI["vermilion"]  # the margin signal and Map-Reflect
SLOT = [OI["blue"], OI["orange"], OI["green"], OI["sky"], OI["purple"], OI["vermilion"], OI["black"]]
INK, GREY, LIGHT = "#1a1a1a", "#8a8a8a", "#e6e6e6"  # INK is text and axis ink only, never a data series
MODELS = ["gptoss", "llama4", "granite", "qwen35"]
MNAME = {"gptoss": "gpt-oss-120b", "llama4": "Llama-4", "granite": "Granite-4", "qwen35": "Qwen3.5"}
MCOL = {"gptoss": OI["blue"], "llama4": OI["orange"], "granite": OI["green"], "qwen35": OI["purple"]}
MMRK = {"gptoss": "o", "llama4": "s", "granite": "^", "qwen35": "D"}
SIG = ["margin", "s1", "ratio", "entropy", "lp_answer", "verbalised", "p_true", "sem_entropy", "combo_paid"]
SNAME = {"margin": "Margin", "s1": "Top score", "ratio": "Ratio", "entropy": "Score entropy",
         "lp_answer": "Token log-prob.", "verbalised": "Stated conf.", "p_true": "P(True)",
         "sem_entropy": "Answer entropy", "combo_paid": "Logistic (paid)"}
SCOL = {"margin": MARGIN, "s1": OI["blue"], "ratio": OI["sky"], "entropy": OI["green"], "lp_answer": OI["orange"],
        "verbalised": OI["purple"], "p_true": OI["black"], "sem_entropy": OI["orange"], "combo_paid": OI["blue"]}
SLS = {"margin": "-", "s1": "-", "ratio": "-", "entropy": "-", "lp_answer": "-", "verbalised": "--",
       "p_true": "--", "sem_entropy": "--", "combo_paid": "--"}  # free signals solid, paid signals dashed
RCOL = {"Map-Reflect": MARGIN, "struct": OI["green"], "every": OI["blue"], "random": GREY, "oracle": GREY,
        "keep": OI["black"]}
FREE, PAID = ["margin", "s1", "ratio", "entropy", "lp_answer"], ["margin", "verbalised", "p_true", "sem_entropy", "combo_paid"]
# Printed widths, inches: ACL \textwidth is 455.24 pt and \columnwidth 219.09 pt (72.27 pt to the inch).
TEXTW, COLW = 455.24 / 72.27, 219.09 / 72.27
FULLW = 0.84 * TEXTW  # the wide chart-page panels
WIDTH = {"pareto": COLW, "calibration": 0.85 * TEXTW, "c14": 0.36 * TEXTW,
         "c1_budget": FULLW, "c2_interventions": FULLW, "c9_repairs_breaks": FULLW, "c3_accuracy_calls": FULLW,
         "c6_margin_hist": FULLW, "c11_thresholds": FULLW, "c7_roc": 0.49 * TEXTW, "c8_risk_coverage": 0.49 * TEXTW,
         "c5_reliability": 0.37 * TEXTW, "c12_detection_vs_routing": 0.34 * TEXTW, "c4_family": 0.31 * TEXTW,
         "c10_gold_location": 0.31 * TEXTW, "c13_heldout": 0.27 * TEXTW, "c15_human_labels": COLW}
FS = 7  # every text, in printed points
MIN_PT = 7.0
FILL = 0.18  # alpha of filled areas

plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                     "legend.fontsize": FS, "axes.titlesize": FS, "axes.linewidth": 0.6, "lines.linewidth": 1.4, "pdf.fonttype": 42,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": INK, "text.color": INK, "hatch.linewidth": 0.6})


def rd(name):
    with open(DATA / f"{name}.csv") as f:
        return list(csv.DictReader(f))


def style(ax, grid="y"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=LIGHT, lw=0.5)
    ax.set_axisbelow(True)


def fig_for(name, height, **kw):
    """A figure at its printed width (WIDTH[name]) and the given height, in inches."""
    return plt.subplots(figsize=(WIDTH[name], height), **kw)


SMALL = []  # (figure, text, printed size) for every text under MIN_PT


def check_sizes(fig, name):
    fig.canvas.draw()
    scale = WIDTH.get(name, fig.get_figwidth()) / fig.get_figwidth()
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() * scale < MIN_PT - 1e-6:
            SMALL.append((name, t.get_text(), round(t.get_fontsize() * scale, 2)))
    return scale


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout(pad=0.3)
    SCALES[name] = check_sizes(fig, name)
    md = {"CreationDate": None, "ModDate": None}
    if name in ("c10_gold_location", "c14") or name.endswith("_mr"):  # appendix "Ablation: Margin-Reflect" (DECISIONS row 41)
        md["Subject"] = "group: ablation-mr"
    fig.savefig(OUT / f"fig6_{name}.pdf", metadata=md)
    fig.savefig(PREVIEW / f"fig6_{name}.png", dpi=200)
    plt.close(fig)
    DONE.append(name)


DONE = []
SCALES = {}


# The chart pages, in main_v6.tex's reading order: (figure label, chart, its panels). Each panel gets the next letter
# of its page; a row of per-model small multiples (C3, C6) is one panel. main() writes the map to
# data6/panel_letters.csv so the text can cite a panel by its letter.
PAGES = [
    # main-text figures, each lettered on its own
    ("fig:selrate", "c1_budget", ["Free confidence estimates", "Paid confidence estimates"]),
    # with the v5 strategies (1b) C9 is one panel, so it carries no letter and is not listed (fig:repairs); its
    # acceptance panel is Margin-Reflect's, kept in fig6_c9_repairs_breaks_mr
] + ([] if (Path(__file__).resolve().parents[2] / "paper/figures/data6/v5_c9_strategies.csv").exists() else [
    ("fig:repairs", "c9_repairs_breaks", ["Reflection strategies, all fields", "Acceptance"])]) + [
    ("fig:detect", "c12_detection_vs_routing", ["Detection against selection"]),
    ("fig:detect", "c13_heldout", ["Held-out datasets"]),
    ("fig:detect", "c5_reliability", ["Top score", "Margin, isotonic"]),
    # appendix "Ablation: Margin-Reflect", Figure fig:mrcharts: the caption letters these two (a) and (b); the charts carry
    # their titles only, no letter (the paper trims the title band)
    ("fig:mrcharts", "c14", ["Run 1 against run 2"]),
    ("fig:mrcharts", "c10_gold_location", ["Gold location"]),
    # the chart pages at the end of the appendix
    ("fig:chartsA", "c2_interventions", ["Loses accuracy", "Widens the list", "Small repairs"]),
    ("fig:chartsA", "c3_accuracy_calls", ["Accuracy against calls, per model"]),
    ("fig:chartsA", "c6_margin_hist", ["Margins of correct and wrong first answers, per model"]),
    ("fig:chartsB", "c11_thresholds", ["Development accuracy", "Refit on less data"]),
    ("fig:chartsB", "c7_roc", ["ROC, core test", "ROC, held-out"]),
    ("fig:chartsB", "c8_risk_coverage", ["Risk against coverage, core test", "Risk against coverage, held-out"]),
    ("fig:chartsB", "c4_family", ["Change by family"]),
]


def _letters():
    out, n = {}, {}
    for page, chart, panels in PAGES:
        for i, title in enumerate(panels):
            k = n.get(page, 0)
            out[(chart, i)] = (page, "abcdefghijklmnopqrstuvwxyz"[k], title)
            n[page] = k + 1
    return out


LETTERS = _letters()


def L(chart, i=0):
    """The printed letter of panel i of a chart-page chart, as '(x)' (the Margin-Reflect _mr versions reuse the
    letters; a panel the v5 version no longer has gets the next letter)."""
    if (chart, i) not in LETTERS:  # an unlisted chart (C9's _mr version) letters its panels from (a)
        if (chart, i - 1) not in LETTERS:
            return f"({'abcdefghijklmnopqrstuvwxyz'[i]})"
        return f"({'abcdefghijklmnopqrstuvwxyz'['abcdefghijklmnopqrstuvwxyz'.index(LETTERS[(chart, i - 1)][1]) + 1]})"
    return f"({LETTERS[(chart, i)][1]})"


def write_panel_letters():
    with open(DATA / "panel_letters.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "letter", "chart", "file", "panel"])
        for (chart, i), (page, let, title) in LETTERS.items():
            w.writerow([page, let, chart, f"figures/fig6_{chart}.pdf", title])


def letter(ax, s):
    ax.text(0.0, 1.02, s, transform=ax.transAxes, fontsize=FS, fontweight="bold", va="bottom")


def random_line(ax, x, y, label="Random selection", lw=1.0):
    ax.plot(x, y, color=GREY, ls="--", lw=lw, label=label, zorder=2)


def oracle_line(ax, x, y, label="Oracle", lw=1.0):
    ax.plot(x, y, color=GREY, ls=":", lw=lw, label=label, zorder=2)


# ---------------------------------------------------------------------------------------------------- Figure 2
def export_main_figs(g):
    """Write v6_numbers' in-memory inputs of Figure 2 and the calibration figure to data6 (called from main())."""
    rnd, sweep, pts = g["FIG2"][:3]
    bsw = g["FIG2"][3] if len(g["FIG2"]) > 3 else []
    with open(DATA / "pareto.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "label", "calls_per_field", "accuracy"])
        for x, y in rnd:
            w.writerow(["random", "", repr(float(x)), repr(float(y))])
        for x, y in sweep:
            w.writerow(["margin sweep", "", repr(float(x)), repr(float(y))])
        for x, y in bsw:  # method v5: the benefit router's threshold swept
            w.writerow(["benefit sweep", "", repr(float(x)), repr(float(y))])
        for lab, x, y in pts:
            w.writerow(["point", lab, repr(float(x)), repr(float(y))])
    X4, MS = g["X4"], g["MSHORT"]
    with open(DATA / "calibration.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "panel", "bin", "fields", "x", "accuracy"])
        for m in MODELS:
            for k, (nb, cf, ac) in enumerate(X4[m]["bins"]):
                w.writerow([m, "top_score", k, int(nb), repr(float(cf)), repr(float(ac))])
            for k, d in enumerate(X4[m]["decs"]):
                w.writerow([m, "margin_decile", k + 1, int(d[0]), k + 1, repr(float(d[3]))])
    with open(DATA / "calibration_models.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "name", "share_top_at_1", "margin_auroc"])
        for m in MODELS:
            w.writerow([m, MS[m], repr(float(X4[m]["at1"])), repr(float(X4[m]["auc"]))])


def pareto():
    if not (DATA / "pareto.csv").exists():
        return
    R = rd("pareto")
    get = lambda s: [(float(r["calls_per_field"]), float(r["accuracy"])) for r in R if r["series"] == s]  # noqa: E731
    rnd, sw, bs = get("random"), sorted(get("margin sweep")), sorted(get("benefit sweep"))
    pts = {r["label"]: (float(r["calls_per_field"]), float(r["accuracy"])) for r in R if r["series"] == "point"}
    fig, ax = fig_for("pareto", 2.6)
    rx, ry = zip(*rnd)
    sx, sy = zip(*sw)
    fx, fy = zip(*bs) if bs else (sx, sy)  # method v5: the benefit sweep is Map-Reflect's curve
    ax.fill_between(fx, np.interp(fx, rx, ry), fy, color=MARGIN, alpha=FILL, lw=0, zorder=1,
                    label="Gain over random selection")
    random_line(ax, rx, ry, "Random selection (rate swept)", lw=1.3)
    if bs:
        ax.plot(sx, sy, color=OI["sky"], lw=1.2, ls="--", marker="o", ms=2.0, label="Margin threshold swept", zorder=3)
        ax.plot(fx, fy, color=MARGIN, lw=1.7, marker="o", ms=2.4, label="Benefit, threshold swept", zorder=4)
    else:
        ax.plot(sx, sy, color=MARGIN, lw=1.7, marker="o", ms=2.4, label="Margin threshold swept", zorder=3)
    style_pt = {"No reflection": ("o", RCOL["keep"], 4.5), "Schema size": ("P", RCOL["struct"], 6.5),
                "Reflect on all": ("X", RCOL["every"], 6.5), "Oracle": ("s", GREY, 4.5), "Map-Reflect": ("*", MARGIN, 14),
                "Margin": ("D", OI["sky"], 4.5)}
    # Labels of the points that crowd the top of the curve sit above it, in a band of their own, each joined to its
    # marker by a thin leader line, so no label covers a marker, a line or another label at print size.
    ytop = max(y for _, y in pts.values())
    above = {"Schema size": (0.03, ytop + 0.75, "left"), "Map-Reflect": (0.47, ytop + 1.35, "center"),
             "Margin": (0.78, ytop + 0.75, "left"), "Reflect on all": (1.15, ytop + 1.35, "right")}
    for lab, (x, y) in pts.items():
        mk, col, ms = style_pt[lab]
        ax.plot([x], [y], marker=mk, ms=ms, color=col, markeredgecolor=INK if lab == "Map-Reflect" else "white",
                markeredgewidth=0.6, zorder=5, ls="none")
        kw = dict(fontsize=FS, color=MARGIN if lab == "Map-Reflect" else INK,
                  fontweight="bold" if lab == "Map-Reflect" else "normal")
        if lab in above:
            tx, ty, ha = above[lab]
            ax.annotate(lab, (x, y), xytext=(tx, ty), textcoords="data", ha=ha, va="bottom", zorder=6,
                        arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5, shrinkA=1, shrinkB=4), **kw)
        else:
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -3), va="top", ha="left", **kw)
    ax.set_ylim(min(y for _, y in pts.values()) - 0.9, ytop + 2.0)
    ax.set_xlabel("Second-pass calls per field")
    ax.set_ylabel("Test accuracy (%)")
    style(ax, "both")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.08), handlelength=1.8)  # clear of the No reflection label
    save(fig, "pareto")


def calibration():
    if not (DATA / "calibration.csv").exists():
        return
    R = rd("calibration")
    S = {r["model"]: r for r in rd("calibration_models")}
    fig, axes = fig_for("calibration", 3.0, ncols=2)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls=":", label="Perfect calibration")
    for m in MODELS:
        rs = [r for r in R if r["model"] == m and r["panel"] == "top_score" and int(r["fields"]) >= 30]
        ax.plot([float(r["x"]) for r in rs], [float(r["accuracy"]) for r in rs], color=MCOL[m], marker=MMRK[m], ms=3.5,
                lw=1.3, label=f"{S[m]['name']} ({float(S[m]['share_top_at_1']):.0f}% at 1.0; "
                              f"AUROC {float(S[m]['margin_auroc']):.2f})")
    ax.set_xlabel("Top score (bin mean)")
    ax.set_ylabel("First-pass accuracy")
    letter(ax, "(a) Top score as a probability")
    ax = axes[1]
    for m in MODELS:
        rs = [r for r in R if r["model"] == m and r["panel"] == "margin_decile" and int(r["fields"]) > 0]
        ax.plot([int(r["bin"]) for r in rs], [100 * float(r["accuracy"]) for r in rs], color=MCOL[m], marker=MMRK[m],
                ms=3.5, lw=1.3)
    ax.set_xlabel("Margin decile (1 = smallest margin)")
    ax.set_ylabel("First-pass accuracy (%)")
    ax.set_xticks(range(1, 11))
    letter(ax, "(b) Accuracy by margin decile")
    for ax in axes:
        style(ax, "both")
    h, lb = axes[0].get_legend_handles_labels()
    h, lb = h[1:] + h[:1], lb[1:] + lb[:1]  # the models first, then the diagonal
    fig.legend(h, lb, frameon=False, loc="lower center", ncol=2, columnspacing=1.2, handlelength=1.8)
    fig.tight_layout(pad=0.3, rect=(0, 0.19, 1, 1))
    save(fig, "calibration", tight=False)


# ---------------------------------------------------------------------------------------------------- C1
C1_LEFT_OUT = []


V5C1, V5C12 = DATA / "v5_c1_curves.csv", DATA / "v5_c12_detect_vs_select.csv"


def c1():
    """1a (DECISIONS row 51): on Map-Reflect v5's second pass once v5_c1_curves.csv exists (core rows); the
    Margin-Reflect version stays generable as fig6_c1_budget_mr."""
    if V5C1.exists():
        v5 = [dict(signal=r["estimate"], share=r["rate"], accuracy=r["accuracy"], random_accuracy=r["random_accuracy"],
                   oracle_accuracy=r["oracle_accuracy"], models=r["models"]) for r in rd("v5_c1_curves") if r["set"] == "core"]
        _c1(rd("c1_budget_curves"), "c1_budget_mr")
        _c1(v5, "c1_budget")
    else:
        _c1(rd("c1_budget_curves"), "c1_budget")


def _c1(R, out):
    by = defaultdict(list)
    for r in R:
        by[r["signal"]].append(r)
    fig, axes = fig_for("c1_budget", 1.6, ncols=2, sharey=True)
    pop = by["margin"][0]["models"]
    # every curve on the margin's population: a signal measured on fewer models is left out until it covers them all
    C1_LEFT_OUT[:] = [s for s in SIG if s in by and by[s][0]["models"] != pop]
    for ax, group, lab in ((axes[0], FREE, L("c1_budget", 0) + " Free confidence estimates"),
                           (axes[1], PAID, L("c1_budget", 1) + " Paid confidence estimates")):
        m = by["margin"]
        x = [100 * float(r["share"]) for r in m]
        b0 = float(m[0]["accuracy"])
        ry = [float(r["random_accuracy"]) - b0 for r in m]
        my = [float(r["accuracy"]) - b0 for r in m]
        ax.fill_between(x, ry, my, color=MARGIN, alpha=FILL, lw=0, zorder=1)
        random_line(ax, x, ry)
        oracle_line(ax, x, [float(r["oracle_accuracy"]) - b0 for r in m])
        for s in group:
            if s in C1_LEFT_OUT:
                continue
            rs = by[s]
            k0 = float(rs[0]["accuracy"])
            ax.plot([100 * float(r["share"]) for r in rs], [float(r["accuracy"]) - k0 for r in rs], color=SCOL[s], ls=SLS[s],
                    lw=2.0 if s == "margin" else 1.2, label=SNAME[s], zorder=4 if s == "margin" else 3)
        ax.set_xlabel("Reflection rate (%)")
        letter(ax, lab)
        style(ax)
    axes[0].set_ylabel("Change (points)")
    hl = {}
    for ax in axes:
        for h, lb in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(lb, h)
    fig.legend(list(hl.values()), [k.replace(" models)", ")") for k in hl], frameon=False, loc="center right",
               ncol=1, handlelength=1.8, labelspacing=0.25, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, rect=(0, 0, 0.71, 1))
    save(fig, out, tight=False)


# ---------------------------------------------------------------------------------------------------- C2
IVCOL = {"Doubt": OI["purple"], "SelfRefine": OI["orange"], "NeutralEvid": OI["sky"],
         "WidenReflect": OI["blue"], "RoutedAction": MARGIN, "Escalate": OI["green"],
         "Neutral": OI["green"], "Reflect": OI["blue"],
         # the v5 strategies (1b): distinct within each regime of C2
         "NeutralCheck": OI["sky"], "Evidence": OI["sky"], "NeutralEvidenceCheck": OI["blue"], "Widen": OI["blue"],
         "Routed": MARGIN}


V5C2, V5C9 = DATA / "v5_c2_strategies.csv", DATA / "v5_c9_strategies.csv"


def c2():
    """1b (DECISIONS row 53): on Map-Reflect v5's test fields once v5_c2_strategies.csv exists, fields in u(x) order
    against random order; the Margin-Reflect version (margin order) stays generable as fig6_c2_interventions_mr."""
    if V5C2.exists():
        v5 = [dict(intervention=r["intervention"], regime=r["regime"], share=r["share"], change_margin=r["change_u"],
                   change_random=r["change_random"]) for r in rd("v5_c2_strategies")]
        _c2(rd("c2_interventions"), "c2_interventions_mr", "margin order")
        _c2(v5, "c2_interventions", "$u(x)$ order")
    else:
        _c2(rd("c2_interventions"), "c2_interventions", "margin order")


def _c2(R, out, order):
    LAB = {"Doubt": "Doubt", "SelfRefine": "Self-Refine", "NeutralEvid": "Neutral + evidence", "Neutral": "Neutral",
           "Reflect": "Neutral + evidence + check", "WidenReflect": "Widen + reflect", "RoutedAction": "Reflect or widen + reflect",
           "Escalate": "Escalate", "NeutralCheck": "Neutral + check", "Evidence": "Neutral + evidence",
           "NeutralEvidenceCheck": "Neutral + evidence + check",
           "Widen": "Widen + reflect", "Routed": "Reflect or widen + reflect"}
    # on Map-Reflect's inputs the neutral arms repair a little; the Margin-Reflect chart keeps its own titles
    titles = (("Loses accuracy", "Widens the list", "Small repairs") if out == "c2_interventions"
              else ("Harmful", "Repairing", "Nothing to repair"))
    regs = [(reg, f"{L('c2_interventions', i)} {t}") for i, (reg, t) in
            enumerate(zip(("harmful", "repairing", "nothing to repair"), titles))]
    fig, axes = fig_for("c2_interventions", 2.05, ncols=3)
    for ax, (reg, title) in zip(axes, regs):
        ivs = list(dict.fromkeys(r["intervention"] for r in R if r["regime"] == reg))
        ivs.sort(key=lambda v: v not in ("RoutedAction", "Routed"))  # the method's own second pass is filled first
        filled = []
        for iv in ivs:
            rs = [r for r in R if r["intervention"] == iv]
            x = [100 * float(r["share"]) for r in rs]
            ym, yr = [float(r["change_margin"]) for r in rs], [float(r["change_random"]) for r in rs]
            c = IVCOL[iv]
            # one fill per region: a pair of curves that coincides with one already filled is drawn as lines only
            tol = 0.05 * max(abs(v) for v in ym + yr)  # 5% of the curve's own scale
            if not any(np.allclose(ym, a, atol=tol) and np.allclose(yr, b, atol=tol) for a, b in filled):
                ax.fill_between(x, yr, ym, color=c, alpha=FILL, lw=0)
                filled.append((ym, yr))
            ax.plot(x, ym, color=c, lw=1.6, label=LAB[iv], zorder=4 if iv in ("RoutedAction", "Routed") else 3)
            ax.plot(x, yr, color=c, lw=0.9, ls="--", zorder=2)
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_xlabel("Reflection rate (%)")
        letter(ax, title)
        style(ax)
        # left-aligned under each panel, so the longest strategy name stays inside the page
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.08, -0.36), ncol=1, handlelength=1.6,
                  labelspacing=0.2, borderaxespad=0)
    axes[0].set_ylabel("Change (points)")
    key = [Line2D([], [], color=GREY, lw=1.4, label=f"solid: {order}"),
           Line2D([], [], color=GREY, lw=0.9, ls="--", label="dashed: random order")]
    fig.legend(handles=key, frameon=False, loc="lower center", ncol=2, handlelength=1.6, borderaxespad=0.1)
    # fixed margins: tight_layout would shrink the axes to make room for the legends under them
    fig.subplots_adjust(left=0.1, right=0.99, top=0.9, bottom=0.42, wspace=0.32)
    # a legend whose longest name runs past the page edge moves left by the overrun (nothing clipped)
    # and each legend then clears its right-hand neighbour (right to left)
    fig.canvas.draw()
    limit = fig.bbox.width - 4
    for ax in list(axes)[::-1]:
        lg = ax.get_legend()
        bb = lg.get_window_extent()
        if bb.x1 > limit:
            x0, y0 = lg.get_bbox_to_anchor().transformed(ax.transAxes.inverted()).p0
            lg.set_bbox_to_anchor((x0 - (bb.x1 - limit) / ax.bbox.width, y0), transform=ax.transAxes)
            fig.canvas.draw()
        limit = lg.get_window_extent().x0 - 8
    save(fig, out, tight=False)


# ---------------------------------------------------------------------------------------------------- C3
def c3():
    R = rd("c3_accuracy_calls")
    fig, axes = fig_for("c3_accuracy_calls", 1.75, ncols=4)
    for ax, m in zip(axes, MODELS):
        rs = [r for r in R if r["model"] == m]
        get = lambda s: [(float(r["calls_per_field"]), float(r["accuracy"])) for r in rs if r["series"] == s]  # noqa: E731
        rnd = sorted(get("random"))
        sw = sorted(get("margin sweep"))
        bs = sorted(get("benefit sweep"))
        rx, ry = zip(*rnd)
        sx, sy = zip(*sw)
        fx, fy = zip(*bs) if bs else (sx, sy)  # method v5: the benefit sweep is Map-Reflect's curve
        ax.fill_between(fx, np.interp(fx, rx, ry), fy, color=MARGIN, alpha=FILL, lw=0, zorder=1)
        random_line(ax, rx, ry)
        if bs:
            ax.plot(sx, sy, color=OI["sky"], lw=1.1, ls="--", marker=MMRK[m], ms=1.8, label="Margin threshold swept", zorder=3)
            ax.plot(fx, fy, color=MARGIN, lw=1.4, marker=MMRK[m], ms=2.2, label="Benefit (classifier), threshold swept", zorder=4)
        else:
            ax.plot(sx, sy, color=MARGIN, lw=1.4, marker=MMRK[m], ms=2.2, label="Margin threshold swept", zorder=3)
        for s, mk, ms_, col, ec in (("struct", "P", 5.5, RCOL["struct"], "white"), ("every", "X", 5.5, RCOL["every"], "white"),
                                    ("Map-Reflect", "*", 10, MARGIN, INK)):
            (x, y), = get(s)
            ax.plot([x], [y], marker=mk, ms=ms_, color=col, markeredgecolor=ec, markeredgewidth=0.6, ls="none",
                    label={"struct": "Schema size", "every": "Reflect on all", "Map-Reflect": "Map-Reflect"}[s], zorder=5)
        ax.set_xlabel("Calls per field")
        letter(ax, (L("c3_accuracy_calls") + " " if m == MODELS[0] else "") + MNAME[m])
        style(ax, "both")
    axes[0].set_ylabel("Test accuracy (%)")
    h, lb = axes[0].get_legend_handles_labels()
    nc = 3 if len(h) > 5 else 5  # two legend rows when the benefit sweep is drawn, so nothing is clipped
    fig.legend(h, lb, frameon=False, loc="lower center", ncol=nc, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(pad=0.3, rect=(0, 0.17 if nc == 3 else 0.1, 1, 1))
    save(fig, "c3_accuracy_calls", tight=False)


# ---------------------------------------------------------------------------------------------------- C4
def c4():
    R = rd("c4_family_gain")
    fams = list(dict.fromkeys(r["family"] for r in R))
    rts = ["Map-Reflect", "random", "struct", "every", "oracle"]
    lab = {"Map-Reflect": "Map-Reflect", "random": "Random", "struct": "Schema size", "every": "Reflect on all",
           "oracle": "Oracle"}
    fig, ax = fig_for("c4_family", 2.6)
    w = 0.16
    for j, rt in enumerate(rts):
        xs = np.arange(len(fams)) + (j - 2) * w
        rows = [next(x for x in R if x["family"] == f and x["router"] == rt) for f in fams]
        e = np.array([float(r["change"]) for r in rows])
        lo = np.array([float(r["lo"]) for r in rows])
        hi = np.array([float(r["hi"]) for r in rows])
        if rt == "oracle":
            ax.bar(xs, e, w, color="white", edgecolor=GREY, linestyle=":", linewidth=1.0, label=lab[rt], zorder=2)
        elif rt == "random":
            ax.bar(xs, e, w, color=LIGHT, edgecolor=GREY, linestyle="--", linewidth=0.8, label=lab[rt], zorder=2)
        else:
            ax.bar(xs, e, w, color=RCOL[rt], edgecolor="white", linewidth=0.4, label=lab[rt], zorder=2)
        ax.errorbar(xs, e, yerr=[e - lo, hi - e], fmt="none", ecolor=INK, elinewidth=0.6, capsize=0, zorder=3)
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Change (points)")
    style(ax)
    letter(ax, L("c4_family") + " By family")
    h, lb = ax.get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, ncol=2, loc="lower center", columnspacing=0.6, handlelength=1.0, handletextpad=0.4)
    fig.tight_layout(pad=0.3, rect=(0, 0.22, 1, 1))
    save(fig, "c4_family", tight=False)


# ---------------------------------------------------------------------------------------------------- C5
def c5():
    R = rd("c5_reliability")
    fig, axes = fig_for("c5_reliability", 2.25, ncols=2, sharey=True)
    for ax, sc, title in ((axes[0], "TopRaw", L("c5_reliability", 0) + " Top score"),
                          (axes[1], "MarginIso", L("c5_reliability", 1) + " Margin, iso.")):
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls=":")
        for m in MODELS:
            rs = [r for r in R if r["model"] == m and r["score"] == sc and int(r["fields"]) >= 30]
            ax.plot([float(r["mean_predicted"]) for r in rs], [float(r["accuracy"]) for r in rs], color=MCOL[m],
                    marker=MMRK[m], ms=3, lw=1.2, label=MNAME[m])
        ax.set_xlabel("Pred. P(correct)")
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        letter(ax, title)
        style(ax, "both")
    axes[0].set_ylabel("First-pass accuracy")
    h, lb = axes[1].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, loc="lower center", ncol=2, columnspacing=0.8, handlelength=1.6)
    fig.tight_layout(pad=0.3, rect=(0, 0.16, 1, 1))
    save(fig, "c5_reliability", tight=False)


# ---------------------------------------------------------------------------------------------------- C6
def c6():
    R = rd("c6_margin_hist")
    fig, axes = fig_for("c6_margin_hist", 1.45, ncols=4, sharey=True)
    for ax, m in zip(axes, MODELS):
        for lab, c in (("correct", OI["blue"]), ("wrong", OI["orange"])):
            rs = [r for r in R if r["model"] == m and r["first_answer"] == lab]
            x = [float(r["bin_low"]) for r in rs] + [1.0]
            y = [100 * float(r["share"]) for r in rs]
            y = y + [y[-1]]
            ax.fill_between(x, y, step="post", color=c, alpha=0.35, lw=0)
            ax.step(x, y, where="post", color=c, lw=1.1, label=lab.capitalize())
        ax.set_xlabel("Margin")
        letter(ax, (L("c6_margin_hist") + " " if m == MODELS[0] else "") + MNAME[m])
        style(ax)
    axes[0].set_ylabel("Share of fields (%)")
    axes[0].legend(frameon=False, loc="upper right", handlelength=1.4)
    save(fig, "c6_margin_hist")


# ---------------------------------------------------------------------------------------------------- C7 / C8
def curves(name, xk, yk, xl, yl, diag, out, loc):
    R = rd(name)
    fig, axes = fig_for(out, 2.3, ncols=2, sharey=True)
    for ax, split, title in ((axes[0], "core", L(out, 0) + " Core test"), (axes[1], "heldout", L(out, 1) + " Held-out")):
        if diag:
            ax.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls=":")
        for s in SIG:
            rs = [r for r in R if r["signal"] == s and r["split"] == split and float(r[xk]) >= (0.02 if not diag else 0)]
            if rs:
                xs, ys = [float(r[xk]) for r in rs], [float(r[yk]) for r in rs]
                if s == "margin" and diag:
                    ax.fill_between(xs, xs, ys, color=MARGIN, alpha=FILL, lw=0, zorder=1)
                ax.plot(xs, ys, color=SCOL[s], ls=SLS[s], lw=2.0 if s == "margin" else 1.0, label=SNAME[s],
                        zorder=4 if s == "margin" else 2)
        ax.set_xlabel(xl)
        letter(ax, title)
        style(ax, "both")
    axes[0].set_ylabel(yl)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, loc="lower center", ncol=3, columnspacing=0.8, handlelength=1.8,
               labelspacing=0.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, rect=(0, 0.25, 1, 1))
    save(fig, out, tight=False)


# ---------------------------------------------------------------------------------------------------- C9
def c9():
    """1b (DECISIONS row 53): the strategies on every v5 test field once v5_c9_strategies.csv exists (one panel); the
    Margin-Reflect version, with its acceptance panel, stays generable as fig6_c9_repairs_breaks_mr."""
    if V5C9.exists():
        _c9(rd("c9_repairs_breaks"), "c9_repairs_breaks_mr")
        R = rd("v5_c9_strategies")
        fig, ax = fig_for("c9_repairs_breaks", 1.75)
        y = np.arange(len(R))[::-1]
        rep, brk = np.array([int(r["repairs"]) for r in R]), np.array([int(r["breaks"]) for r in R])
        ax.barh(y, rep, color=OI["green"], height=0.62, label="Repairs", edgecolor="white", linewidth=0.8)
        ax.barh(y, -brk, color=OI["orange"], height=0.62, label="Breaks", edgecolor="white", linewidth=0.8)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([r["label"] for r in R])
        ax.set_xlabel("Fields (breaks to the left)")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(int(v)):,}"))
        letter(ax, "Reflection strategies, all fields")  # one panel: no letter
        style(ax, "x")
        ax.legend(frameon=False, loc="center left", handlelength=1.4)  # the near-zero rows leave that corner empty
        save(fig, "c9_repairs_breaks")
    else:
        _c9(rd("c9_repairs_breaks"), "c9_repairs_breaks")


def _c9(R, out):
    fig, axes = fig_for("c9_repairs_breaks", 1.75, ncols=2, gridspec_kw={"width_ratios": [10, 7]})
    for ax, grp, title in ((axes[0], "intervention", L("c9_repairs_breaks", 0) + " Reflection strategies, all fields"),
                           (axes[1], "acceptance", L("c9_repairs_breaks", 1) + " Acceptance")):
        rs = [r for r in R if r["group"] == grp]
        y = np.arange(len(rs))[::-1]
        rep = np.array([int(r["repairs"]) for r in rs])
        brk = np.array([int(r["breaks"]) for r in rs])
        ax.barh(y, rep, color=OI["green"], height=0.62, label="Repairs", edgecolor="white", linewidth=0.8)
        ax.barh(y, -brk, color=OI["orange"], height=0.62, label="Breaks", edgecolor="white", linewidth=0.8)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([r["label"].replace(" (neutral + evidence + check)", "").replace(" (another model's first answer)", "")
                            .replace(" (accept every proposal)", "").replace("Fresh mapping over the widened pool", "Fresh")
                            .replace("Map-Reflect second pass (as routed)", "Reflect or widen + reflect")
                            .replace("Reflect or widen + reflect (by top score)", "Reflect or widen + reflect")
                            .replace("Widen + reflect without the check", "Widen + reflect, no check")
                            .replace("reflection's own margin grows", "own margin grows")
                            .replace("pairwise check (Margin-Reflect)", "Pairwise check") for r in rs])
        ax.set_xlabel("Fields (breaks to the left)" if grp == "intervention" else "Fields")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(int(v)):,}"))
        letter(ax, title)
        style(ax, "x")
    axes[0].legend(frameon=False, loc="lower left", handlelength=1.4)
    save(fig, out)


# ---------------------------------------------------------------------------------------------------- C10
def c10():
    R = rd("c10_gold_location")
    strata = ["Declined", "Wide", "Shown"]
    golds = [("InShown", "Gold among shown", OI["blue"]), ("Added", "Gold added by widening", OI["green"]),
             ("Absent", "Gold still absent", OI["orange"])]
    fig, ax = fig_for("c10_gold_location", 2.6)
    left = np.zeros(len(strata))
    for g, lab, col in golds:
        v = np.array([sum(int(r["routed_errors"]) for r in R if r["stratum"] == s and r["gold"] == g) for s in strata])
        ax.barh(range(len(strata)), v, left=left, color=col, height=0.6, label=lab, edgecolor="white", linewidth=1.0)
        left += v
    ax.set_yticks(range(len(strata)))
    ax.set_yticklabels(strata)
    ax.set_ylabel("Stratum")
    ax.invert_yaxis()
    ax.set_xlabel("Selected first-pass errors")
    style(ax, "x")
    letter(ax, "Gold location")  # no letter: fig:mrcharts letters it in its caption
    fig.legend(frameon=False, loc="lower center", ncol=1, handlelength=1.4)
    fig.tight_layout(pad=0.3, rect=(0, 0.25, 1, 1))
    save(fig, "c10_gold_location", tight=False)


# ---------------------------------------------------------------------------------------------------- C11
HEAT = LinearSegmentedColormap.from_list("oi_blue", ["#ffffff", OI["sky"], OI["blue"]])


def c11():
    H = rd("c11_threshold_heatmap")
    S = rd("c11_devsize")
    fig, axes = fig_for("c11_thresholds", 1.55, ncols=2, gridspec_kw={"width_ratios": [3, 1.5]})
    ax = axes[0]
    tms = sorted({float(r["t_m"]) for r in H})
    M = np.array([[next(float(r["dev_accuracy"]) for r in H if r["model"] == m and float(r["t_m"]) == t) for t in tms] for m in MODELS])
    rel = M - M.max(axis=1, keepdims=True)
    im = ax.imshow(rel, aspect="auto", cmap=HEAT, vmin=-3, vmax=0)
    for i, m in enumerate(MODELS):
        for j, t in enumerate(tms):
            fz = next(float(r["frozen"]) for r in H if r["model"] == m and float(r["t_m"]) == t)
            if fz:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=MARGIN, lw=1.6))
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MNAME[m] for m in MODELS])
    ax.set_xticks(range(0, len(tms), 2))
    ax.set_xticklabels([f"{tms[j]:.1f}" for j in range(0, len(tms), 2)])
    ax.set_xlabel("Margin threshold (box: frozen value)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("vs best (pts)", fontsize=FS, labelpad=2)
    cb.ax.tick_params(labelsize=FS)
    letter(ax, L("c11_thresholds", 0) + " Development accuracy")
    ax = axes[1]
    x = [100 * float(r["dev_fraction"]) for r in S]
    y = [float(r["mean_test_change"]) for r in S]
    ax.fill_between(x, [float(r["p2_5"]) for r in S], [float(r["p97_5"]) for r in S], color=MARGIN, alpha=0.22, lw=0)
    ax.plot(x, y, color=MARGIN, marker="o", ms=3)
    ax.set_xlabel("Dev. data used (%)")
    ax.set_ylabel("Test change (points)")
    letter(ax, L("c11_thresholds", 1) + " Refit, less data")
    style(ax, "both")
    fig.tight_layout(pad=0.3, w_pad=1.6)
    save(fig, "c11_thresholds", tight=False)


# ---------------------------------------------------------------------------------------------------- C12
def mark(s):
    """A signal's marker: filled for a free signal, hollow for a paid one (the markers' version of solid and dashed)."""
    if s in PAID and s != "margin":
        return dict(color=SCOL[s], markerfacecolor="white", markeredgecolor=SCOL[s], markeredgewidth=1.0)
    return dict(color=SCOL[s], markeredgecolor="white", markeredgewidth=0.4)


def c12():
    """1a (DECISIONS row 51): on Map-Reflect v5's second pass once v5_c12_detect_vs_select.csv exists (core rows); the
    Margin-Reflect version stays generable as fig6_c12_detection_vs_routing_mr."""
    if V5C12.exists():
        v5 = [dict(signal=r["estimate"], model=r["model"], auroc=r["auroc"], margin_auroc=r["margin_auroc"],
                   routing_delta=r["selection_delta"], lo=r["lo"], hi=r["hi"]) for r in rd("v5_c12_detect_vs_select")
              if r["set"] == "core"]
        _c12(rd("c12_auroc_vs_routing"), "c12_detection_vs_routing_mr")
        _c12(v5, "c12_detection_vs_routing")
    else:
        _c12(rd("c12_auroc_vs_routing"), "c12_detection_vs_routing")


def _c12(R, out):
    fig, ax = fig_for("c12_detection_vs_routing", 2.25)
    ax.axhline(0, color=MARGIN, lw=0.8, ls="-", alpha=0.7)
    ax.axvline(0, color=MARGIN, lw=0.8, ls="-", alpha=0.7)
    for r in R:
        x, y = float(r["auroc"]) - float(r["margin_auroc"]), float(r["routing_delta"])
        ax.plot([x, x], [float(r["lo"]), float(r["hi"])], color=SCOL[r["signal"]], lw=0.7, alpha=0.7)
        ax.plot([x], [y], marker=MMRK[r["model"]], ms=4.2, ls="none", **mark(r["signal"]))
    h = [Line2D([], [], marker="o", ls="none", ms=4, label=SNAME[s], **mark(s)) for s in SIG if s != "margin" and any(r["signal"] == s for r in R)]
    h += [Line2D([], [], color=GREY, marker=MMRK[m], ls="none", ms=4, label=MNAME[m]) for m in MODELS]
    h += [Line2D([], [], color=MARGIN, lw=0.8, label="Margin (zero)")]
    fig.legend(handles=h, frameon=False, loc="lower center", ncol=2, columnspacing=0.6, handlelength=1.0,
               handletextpad=0.3, labelspacing=0.15, borderaxespad=0.05)
    ax.set_xlabel("AUROC $\\Delta$ vs margin")
    ax.set_ylabel("Change from\nselection (pts)")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))
    style(ax, "both")
    letter(ax, L("c12_detection_vs_routing") + " Detection vs selection")
    fig.tight_layout(pad=0.3, rect=(0, 0.36, 1, 1))
    save(fig, out, tight=False)


# ---------------------------------------------------------------------------------------------------- C13
def c13():
    R = rd("c13_heldout")
    fig, ax = fig_for("c13_heldout", 2.25)
    full = {"All": "Pooled", "Mimic": "MIMIC-OMOP", "BioML": "Bio-ML", "OAEI": "OAEI Conf."}
    names = [full.get(r["set"], r["set"]) for r in R]
    y = np.arange(len(R))[::-1]
    for k, off, col, lab in (("change", 0.19, MARGIN, "Minus first pass"),
                             ("vs_random", -0.19, OI["blue"], "Minus random")):
        lo_k, hi_k = ("lo", "hi") if k == "change" else ("vs_random_lo", "vs_random_hi")
        e = np.array([float(r[k]) for r in R])
        lo = np.array([float(r[lo_k]) for r in R])
        hi = np.array([float(r[hi_k]) for r in R])
        ax.barh(y + off, e, height=0.36, color=col, edgecolor="white", linewidth=0.4, label=lab, zorder=2)
        ax.errorbar(e, y + off, xerr=[e - lo, hi - e], fmt="none", ecolor=INK, elinewidth=0.6, capsize=0, zorder=3)
    ax.axvline(0, color=GREY, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Gain (points)")
    style(ax, "x")
    letter(ax, L("c13_heldout") + " Held-out")
    fig.legend(frameon=False, loc="lower center", ncol=1, handlelength=1.4)
    fig.tight_layout(pad=0.3, rect=(0, 0.16, 1, 1))
    save(fig, "c13_heldout", tight=False)


# ---------------------------------------------------------------------------------------------------- C14
def c14():
    """Map-Reflect's gain per model and dataset cell, run 1 against run 2 (R6), with each cell's 95% split-group
    interval in both runs; the diagonal is perfect agreement and the pooled value per model is a larger hollow mark."""
    if not (DATA / "c14_r6_gain.csv").exists():
        return
    R = rd("c14_r6_gain")
    fig, ax = fig_for("c14", 3.0)
    vals, seen = [], set()
    for r in R:
        if r["model"] == "pooled":
            continue
        m = r["model"]
        g1, g2 = float(r["gain1"]), float(r["gain2"])
        vals += [float(r["gain1_lo"]), float(r["gain1_hi"]), float(r["gain2_lo"]), float(r["gain2_hi"])]
        if r["dataset"] == "all":
            ax.plot([g1], [g2], marker=MMRK[m], ms=7, mfc="none", mec=MCOL[m], mew=1.3, ls="none", zorder=4,
                    label=f"{MNAME[m]}, pooled")
            continue
        ax.plot([float(r["gain1_lo"]), float(r["gain1_hi"])], [g2, g2], color=MCOL[m], lw=0.5, alpha=0.5)
        ax.plot([g1, g1], [float(r["gain2_lo"]), float(r["gain2_hi"])], color=MCOL[m], lw=0.5, alpha=0.5)
        ax.plot([g1], [g2], marker=MMRK[m], ms=3.6, color=MCOL[m], markeredgecolor="white", markeredgewidth=0.4,
                ls="none", zorder=3, label=None if m in seen else f"{MNAME[m]}, per dataset")
        seen.add(m)
    lo, hi = min(vals + [0]) - 1, max(vals + [0]) + 1
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.6, ls=":", zorder=1)
    ax.axhline(0, color=LIGHT, lw=0.6, zorder=0)
    ax.axvline(0, color=LIGHT, lw=0.6, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Gain in run 1 (points)")
    ax.set_ylabel("Gain in run 2 (points)")
    style(ax, None)
    letter(ax, "Run 1 vs run 2")  # no letter: fig:mrcharts letters it in its caption
    h, lb = ax.get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, loc="lower center", ncol=1, handlelength=1.2, labelspacing=0.2)
    fig.tight_layout(pad=0.3, rect=(0, 0.25, 1, 0.97))
    save(fig, "c14", tight=False)


# ---------------------------------------------------------------------------------------------------- C15
def c15():
    if not (DATA / "c15_summary.csv").exists():
        return
    R = rd("c15_summary")
    if not R:
        return
    keys = list(R[0].keys())
    lab, val = keys[0], next((k for k in keys[1:] if all(_isnum(r[k]) for r in R)), None)
    if val is None:
        return
    fig, ax = fig_for("c15_human_labels", 2.0)
    ax.barh(range(len(R)), [float(r[val]) for r in R], color=OI["blue"], height=0.6)
    ax.set_yticks(range(len(R)))
    ax.set_yticklabels([r[lab] for r in R])
    ax.invert_yaxis()
    ax.set_xlabel(val.replace("_", " "))
    style(ax, "x")
    save(fig, "c15_human_labels")


def _isnum(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def main():
    caller = sys.modules.get("__main__")
    g = vars(caller) if caller is not None else {}
    if g.get("FIG2") and g.get("X4") and g.get("MSHORT"):  # called at the end of v6_numbers.py: keep its figure inputs
        export_main_figs(g)
    pareto()
    calibration()
    c1()
    c2()
    c3()
    c4()
    c5()
    c6()
    curves("c7_roc", "false_positive_rate", "true_positive_rate", "False positive rate",
           "TPR (errors flagged)", True, "c7_roc", "lower right")
    curves("c8_risk_coverage", "coverage", "risk", "Coverage", "Error rate, kept fields",
           False, "c8_risk_coverage", "upper left")
    c9()
    c10()
    c11()
    c12()
    c13()
    c14()  # once R6 has landed (data6/c14_r6_gain.csv)
    c15()
    write_panel_letters()
    if C1_LEFT_OUT:
        print("C1 leaves out signals measured on fewer models than the margin: " + ", ".join(C1_LEFT_OUT))
    print(f"wrote {len(DONE)} figures: " + ", ".join(f"fig6_{d}.pdf" for d in DONE))
    print("print scale (printed width / drawn width): " + ", ".join(f"{k} {v:.2f}" for k, v in SCALES.items()))
    if SMALL:
        print(f"{len(SMALL)} texts below {MIN_PT:g} pt at print size:")
        for name, txt, pt in SMALL:
            print(f"  fig6_{name}: {pt} pt {txt!r}")
    else:
        print(f"every text is at least {MIN_PT:g} pt at print size")


if __name__ == "__main__":
    main()
