#!/usr/bin/env python3
"""Held-out datasets (docs/naacl/plan/newdata.md): method v4 and its gates and controls on test, measured outcomes only.

Hygiene (docs/naacl/REDO_PLAN_V4.md): the v4 values are frozen (docs/naacl/results/v4_frozen.json, METHOD_V4.md) and
nothing is fitted on these sets.
  - scorer: eval.calc_accuracy.target_correct (full target identity) against the dataset file's gold
    (gold_targets), so every acceptable target counts
  - routing: scripts/naacl/router.py with the frozen per-model t_m and t_s; the v4 action of a field is the near-tie
    action (NT) when it is valid with s1 >= t_s, else the model's widen action (WT20 or WT50)
  - outcomes: results/naacl/v4/test/given rows, where NT and the widen action ran on every test field, so every gate
    is scored on measured outcomes; the stored v4 answer is checked against the router
  - intervals: the split-group bootstrap of scripts/naacl/group_stats.py (schema units, stratified by dataset, one draw
    shared by every model, policy and contrast; B = 2000), with two-sided bootstrap p-values and Holm's correction
    over the primary contrasts (v4 minus keep-all, random gate, reflect-all with the v4 action, declined-only retry,
    full candidate list)

Policies (top-1 field accuracy):
  keep all                         the first pass
  reflect all, NT                  the near-tie action (v2 Map-Reflect) on every field
  reflect all, v4 action           each field's v4 action on every field
  random gate                      v4's action on a random share of fields equal to v4's routed share in that model and
                                   dataset (exact expectation)
  oracle gate                      at the same share, only fields the v4 action fixes (the fraction that fits if capped)
  declined-only retry              the v4 action on fields whose first pass is invalid, nothing else
  v4                               the frozen method
Controls (runner outputs; a field a control did not rerun keeps its first pass, which the control would reproduce):
  re-ask declined                  the first-pass prompt again on declined fields (results/naacl/v4/test/reask_runs)
  hybrid top-20 shortlist          the first pass over the named type's hybrid top 20 (hybrid_runs)
  fresh mapping                    a first-pass prompt over the widen action's pool, no prior answer (fresh_*_runs)
  full candidate list              every field of the named type shown (results/naacl/v3/test/fulllist)
Second-pass calls per field: 1 + (a change proposed, so the check runs) on each routed field.
Gate swap: on answered fields with both paid signals, at each model's v4 routed share there, route by the margin,
P(True), stated confidence, or a logistic of margin, s1, P(True) and stated confidence fitted per model on core
development fields only (uq_ablation.gate_weights; no held-out field enters the fit); the second pass is the v4
action, measured.

Writes docs/naacl/results/newdata.md, paper/numbers_heldout.tex and paper/tab_heldout.tex.
Usage: newdata_analysis.py [--out ...] [--paper paper] [--B 2000] [datasets...]
"""
import argparse
import os
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
from eval.calc_accuracy import load_gold_file, target_correct  # noqa: E402
import router  # noqa: E402
import group_stats as GS  # noqa: E402


def _quiet_import(name):
    """Modules that parse their own arguments at import are imported with a bare argv."""
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        return __import__(name)
    finally:
        sys.argv = argv


R2 = _quiet_import("reflect_v2")
UQA = _quiet_import("uq_ablation")
gate_weights = UQA.gate_weights

MODELS = ["gptoss", "llama4", "granite", "qwen35"]
CORE9 = ["itopsgraph", "toolbench", "omop_cms", "omop_mimic", "omop_synthea",
         "valentine_chembl", "valentine_opendata", "valentine_tpcdi", "valentine_wikidata"]
NAMES = {"mimic_omop_rematch": "MIMIC-OMOP (ReMatch)", "semtab_m2kg_r1": "SemTab M2KG R1",
         "oaei_conference": "OAEI Conference", "bioml_omim_ordo": "Bio-ML OMIM-ORDO",
         "schemanet_cprd": "SchemaNet CPRD", "valentine_magellan": "Valentine Magellan (sanity)"}
SANITY = "valentine_magellan"
NOTES = {"semtab_m2kg_r1": "The gold is among the 20 shown candidates for only 58 of 141 columns (the 2,881-property "
                           "glossary is ordered by the runner's retriever, and v4 widens inside it), so most of this set "
                           "tests the widen path; see the band split.",
         "valentine_magellan": "Source and target names are identical for all 41 gold pairs; a sanity row, never pooled.",
         "mimic_omop_rematch": "112 null (NA) gold rows are not scored; 3 gold rows whose target is not a column of the "
                               "released OMOP schema are dropped.",
         "bioml_omim_ordo": "500 sources sampled with a fixed seed from the 2,605 test rows; candidates are the track's "
                            "101 per source, and widening stays inside them."}
POLICIES = ["keep all", "reflect all, NT", "reflect all, v4 action", "random gate", "oracle gate",
            "declined-only retry", "re-ask declined", "hybrid top-20 shortlist", "fresh mapping",
            "full candidate list", "v4"]
PRIMARY = [("v4", "keep all"), ("v4", "random gate"), ("v4", "reflect all, v4 action"),
           ("v4", "declined-only retry"), ("v4", "full candidate list")]
GATES = ["margin", "P(True)", "stated confidence", "logistic (core development fit)"]


route = router.route  # (band, action) of a field under a frozen threshold record; the shared rule, re-exported


def read(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []


def changed(r, k):
    return float(bool(r.get(k)) and r.get(k) != r.get("base") and bool(r.get("base")))


def control_keys(ds, m, kind):
    """The "<sample_id>||<field>" keys a control was asked to rerun (results/naacl/v4/keys, written by v4_keys.py)."""
    p = ROOT / f"results/naacl/v4/keys/test__{ds}__{m}.{kind}"
    if not p.exists():
        return set()
    if kind == "declined":
        return {l.strip() for l in open(p) if l.strip()}
    return set(json.load(open(p)))


def control_answers(pattern, ids, keyset):
    """key -> first answer of a runner control output on the fields it was asked to rerun (protocol v2 records, ids
    resolved by the runner); an invalid answer is None (wrong)."""
    out = {}
    for slot in glob.glob(pattern):
        if not glob.glob(f"{slot}/mapreflect_*.json"):
            continue
        for rec in R2.records(R2.load_slot(slot), ids):
            if f"{rec['sample_id']}||{rec['field']}" in keyset:
                out[rec["key"]] = rec["base"] if rec["valid"] else None
    return out


class Cell:
    """One (dataset, model) cell: per-field vectors, with .ds, .n, .sid for group_stats.GroupBootstrap."""


def valid_uq(r):
    """With V6_UQ_VALID_ONLY=1, a stated confidence whose answer did not resolve to a candidate is missing, not 1 - c
    (docs/naacl/v6/reviews/opendata_uq_audit/REPORT.md, item 4). Off by default, so v4 reproduces exactly."""
    if os.environ.get("V6_UQ_VALID_ONLY") == "1" and r.get("verbalised") is not None and not r.get("verbalised_answer"):
        r = dict(r, verbalised=None)
    return r


def cell(ds, m, ids, gold, t, notes):
    rows = read(ROOT / f"results/naacl/v4/test/given/{ds}__{m}/fields.jsonl")
    if not rows or not (ROOT / f"results/naacl/v4/test/given/{ds}__{m}/fields.jsonl.done").exists():
        return None
    rows = [r for r in rows if str(r["sample_id"]) in ids]
    fl = {r["key"]: r for r in read(ROOT / f"results/naacl/v3/test/fulllist/{ds}__{m}/fields.jsonl")}
    uq = {r["key"]: valid_uq(r) for r in read(ROOT / f"results/naacl/v3/uq/{ds}__{m}.jsonl")}
    ctl = {nm: control_answers(str(ROOT / f"results/naacl/v4/test/{pat}/{ds}__{m}"), ids, control_keys(ds, m, kind))
           for nm, pat, kind in (("re-ask declined", "reask_runs", "declined"),
                                 ("hybrid top-20 shortlist", "hybrid_runs", "hybrid.json"),
                                 ("fresh mapping", f"fresh_{t['widen']}_runs", "fresh.json"))}

    def ok(r, pred):
        return target_correct(r.get(pred) if isinstance(pred, str) and pred in r else pred,
                              gold[str(r["sample_id"])][0][r["key"].rsplit("||", 1)[1]])

    c = Cell()
    c.ds, c.model, c.n = ds, m, len(rows)
    c.sid = np.array([str(r["sample_id"]) for r in rows])
    c.valid = np.array([bool(r["valid"]) for r in rows])
    c.margin = router.canonical_margin(np.array([float(r["margin"]) for r in rows]))  # identity unless v4.1
    c.s1 = np.array([float(r.get("s1", 0) or 0) for r in rows])
    c.band = router.bands(c.margin, c.valid, c.s1, t["t_m"], t["t_s"])
    c.act = np.array([router.action(v, s, t["t_s"], t["near"], t["widen"]) for v, s in zip(c.valid, c.s1)])
    c.routed = (c.band != router.KEEP).astype(float)
    base = np.array([ok(r, "base") for r in rows], float)
    act_ok = np.array([ok(r, a) if a in r else np.nan for r, a in zip(rows, c.act)], float)
    notes["no measured v4-action outcome"] += int(np.isnan(act_ok).sum())
    act_ok = np.where(np.isnan(act_ok), base, act_ok)
    nt_ok = np.array([ok(r, "NT") if "NT" in r else ok(r, "base") for r in rows], float)
    v4 = np.where(c.routed == 1, act_ok, base)
    stored = np.array([ok(r, "v4") if "v4" in r else np.nan for r in rows], float)
    notes["stored v4 answer disagrees with the router"] += int(np.nansum(stored != v4))
    s = c.routed.mean()
    fix = np.where(base == 0, act_ok, 0.0)
    cap = min(1.0, s * c.n / fix.sum()) if fix.sum() else 1.0
    c.vec = {"keep all": base, "reflect all, NT": nt_ok, "reflect all, v4 action": act_ok,
             "random gate": base + s * (act_ok - base), "oracle gate": base + cap * fix,
             "declined-only retry": np.where(c.valid, base, act_ok), "v4": v4}
    for nm, ans in ctl.items():
        if ans:
            c.vec[nm] = np.array([ok(r, ans[r["key"]]) if r["key"] in ans else b for r, b in zip(rows, base)], float)
    if fl:
        c.vec["full candidate list"] = np.array([ok(fl[r["key"]], "base") if r["key"] in fl else np.nan for r in rows])
    c.calls = {"v4": c.routed * (1 + np.array([changed(r, f"{a}_proposed") for r, a in zip(rows, c.act)])),
               "reflect all, NT": c.valid * (1 + np.array([changed(r, "NT_proposed") for r in rows])),
               "reflect all, v4 action": 1 + np.array([changed(r, f"{a}_proposed") for r, a in zip(rows, c.act)])}
    c.calls["random gate"] = s * c.calls["reflect all, v4 action"]
    c.calls["declined-only retry"] = (~c.valid) * c.calls["reflect all, v4 action"]
    c.rerun = {nm: (len(ans) / c.n) for nm, ans in ctl.items()}
    c.verb = np.array([float(uq[r["key"]]["verbalised"]) if r["key"] in uq and uq[r["key"]].get("verbalised") is not None
                       else np.nan for r in rows])
    c.ptrue = np.array([float(uq[r["key"]]["p_true"]) if r["key"] in uq and uq[r["key"]].get("p_true") is not None
                        else np.nan for r in rows])
    return c


def fit_combo():
    """Per model: P(first answer correct) from a logistic on margin, s1, P(True) and stated confidence, fitted on the
    core development fields only: the core samples of the earlier hash development split (manifest old_dev_excluded),
    the only core fields outside the reported test that carry P(True) and stated confidence. Full-identity scorer.
    Never fitted on a held-out set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, Y = defaultdict(list), defaultdict(list)
    for ds in CORE9:
        man = json.load(open(ROOT / f"data/splits/{ds}.json"))
        dev = set(man.get("old_dev_excluded") or [])
        fname = {"itopsgraph": "data/v2/itopsgraph_v2_cross_product.jsonl",
                 "toolbench": "data/toolbench_eval_dataset.jsonl"}.get(ds, f"data/{ds}_dataset.jsonl")
        try:
            gold = load_gold_file(ROOT / fname)
        except Exception:
            gold = {}
        for m in MODELS:
            uq = {r["key"]: valid_uq(r) for r in read(ROOT / f"results/naacl/v3/uq/{ds}__{m}.jsonl")}
            for r in read(ROOT / f"results/naacl/v3/test/given/{ds}__{m}/fields.jsonl"):
                u = uq.get(r["key"], {})
                if str(r["sample_id"]) not in dev or not r.get("valid", True) or u.get("p_true") is None \
                        or u.get("verbalised") is None:
                    continue
                tg = (gold.get(str(r["sample_id"])) or ({},))[0].get(r["key"].rsplit("||", 1)[1])
                X[m].append([float(r["margin"]), float(r.get("s1", 0) or 0), float(u["p_true"]), float(u["verbalised"])])
                Y[m].append(target_correct(r.get("base"), tg) if tg else int(r["base_ok"]))
    return {m: (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(np.array(X[m]), Y[m]),
                len(Y[m]), float(np.mean(Y[m]))) for m in MODELS if len(set(Y[m])) == 2}


def swap_cell(c, combo):
    """A sub-cell over answered fields with both paid signals, with one accuracy vector per gate."""
    pop = c.valid & ~np.isnan(c.verb) & ~np.isnan(c.ptrue)
    if not pop.any():
        return None
    k = float(c.routed[pop].sum())
    base, act = c.vec["keep all"][pop], c.vec["reflect all, v4 action"][pop]
    sig = {"margin": c.margin[pop], "P(True)": c.ptrue[pop], "stated confidence": c.verb[pop]}
    if combo is not None:
        sig[GATES[3]] = combo.predict_proba(np.column_stack([c.margin[pop], c.s1[pop], c.ptrue[pop], c.verb[pop]]))[:, 1]
    s = Cell()
    s.ds, s.model, s.n, s.sid = c.ds, c.model, int(pop.sum()), c.sid[pop]
    s.vec = {g: base + gate_weights(np.asarray(v, float), k) * (act - base) for g, v in sig.items()}
    return s


def estimate(GB, cells, datasets, a, b=None):
    """(point, lo, hi, draws) of policy a, or of a minus b, pooled over datasets (field-weighted)."""
    use = [c for c in cells if a in c.vec and (b is None or b in c.vec)]
    if not use:
        return None

    def fn(c):
        v = c.vec[a] - (c.vec[b] if b else 0)
        return np.nan_to_num(v)
    keep = [c for c in use if not np.isnan(c.vec[a]).any() and (b is None or not np.isnan(c.vec[b]).any())]
    if not keep:
        return None
    pt, dr = GB.pooled(keep, fn, datasets)
    lo, hi = GS.interval(dr)
    return pt, lo, hi, dr


def auroc(score, correct):
    score, correct = np.asarray(score, float), np.asarray(correct, int)
    pos, neg = score[correct == 1], score[correct == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


TAG = {"mimic_omop_rematch": "Mimic", "semtab_m2kg_r1": "SemTab", "oaei_conference": "OAEI",
       "bioml_omim_ordo": "BioML", "schemanet_cprd": "SchemaNet", "valentine_magellan": "Magellan"}
LABEL = {"mimic_omop_rematch": "MIMIC to OMOP", "semtab_m2kg_r1": "SemTab Metadata2KG",
         "oaei_conference": "OAEI Conference", "bioml_omim_ordo": "Bio-ML OMIM--ORDO", "schemanet_cprd": "SchemaNet CPRD"}
SWAPMAC = {"PTrue": "P(True)", "Stated": "stated confidence", "Logit": GATES[3]}
CTLMAC = {"Decl": "declined-only retry", "Reask": "re-ask declined", "Hybrid": "hybrid top-20 shortlist",
          "Fresh": "fresh mapping", "Full": "full candidate list", "ReflectNT": "reflect all, NT",
          "ReflectAct": "reflect all, v4 action", "Oracle": "oracle gate"}


def write_paper(paper, R, sets):
    sg = lambda x: f"{x:+.2f}"
    M = []

    def put(name, val):
        M.append(f"\\newcommand{{\\HO{name}}}{{{val}}}")

    def tri(prefix, e):
        if e:
            put(prefix, sg(e[0]))
            put(prefix + "Lo", sg(e[1]))
            put(prefix + "Hi", sg(e[2]))

    put("NSets", {5: "five", 4: "four", 6: "six"}.get(len(sets), str(len(sets))))
    for key, d in [(TAG[d], d) for d in sets] + [("All", "ALL")]:
        r = R[d]
        put(f"{key}Fields", f"{r['fields_per_model']:,}")
        put(f"{key}First", f"{r['acc']['keep all'][0]:.1f}")
        put(f"{key}VFour", f"{r['acc']['v4'][0]:.1f}")
        tri(f"{key}VThree", r["diff"].get(("v4", "keep all")))
        put(f"{key}Random", sg(r["diff"][("random gate", "keep all")][0]))
        tri(f"{key}VMinusRandom", r["diff"].get(("v4", "random gate")))
        tri(f"{key}VMinusReflectAct", r["diff"].get(("v4", "reflect all, v4 action")))
        tri(f"{key}VMinusDecl", r["diff"].get(("v4", "declined-only retry")))
        tri(f"{key}VMinusFull", r["diff"].get(("v4", "full candidate list")))
        for mac, pol in CTLMAC.items():
            e = r["diff"].get((pol, "keep all"))
            if e:
                put(f"{key}{mac}", sg(e[0]))
        put(f"{key}Routed", f"{100 * r['routed']:.0f}")
        put(f"{key}Calls", f"{r['calls']['v4']:.2f}")
        put(f"{key}ReflectAllCalls", f"{r['calls']['reflect all, NT']:.2f}")
        put(f"{key}ReflectActCalls", f"{r['calls']['reflect all, v4 action']:.2f}")
        for mac, gname in SWAPMAC.items():
            tri(f"{key}Swap{mac}", r["swap"].get(gname))
    if "ALL" in R:
        put("AllMacroVThree", sg(R["ALL"]["macro_v4"][0]))
        put("AllMacroVThreeLo", sg(R["ALL"]["macro_v4"][1]))
        put("AllMacroVThreeHi", sg(R["ALL"]["macro_v4"][2]))
    if SANITY in R:
        r = R[SANITY]
        put("MagellanFields", f"{r['fields_per_model']:,}")
        put("MagellanFirst", f"{r['acc']['keep all'][0]:.1f}")
        put("MagellanVThree", sg(r["diff"][("v4", "keep all")][0]))
    (paper / "numbers_heldout.tex").write_text(
        "% Generated by scripts/naacl/newdata_analysis.py from the method v4 held-out runs (docs/naacl/results/newdata.md);"
        " do not edit by hand.\n% Frozen v4 values (docs/naacl/results/v4_frozen.json); test only; four core models; "
        "full-identity scorer; measured outcomes only. Fields: test fields per model. First, VFour: accuracy (%). VThree "
        "(name kept for the prose): v4 minus keep-all (points). Random: the random gate with v4's action at v4's routed "
        "share minus keep-all. VMinus*: v4 minus a gate or control. Decl, Reask, Hybrid, Fresh, Full, ReflectNT, "
        "ReflectAct, Oracle: that policy minus keep-all. Routed: % of fields. Calls: second-pass calls per field. Swap*: "
        "gate swap minus the margin. All: pooled without Magellan (field-weighted); AllMacro: mean over the sets. "
        "Intervals: 95% split-group bootstrap (scripts/naacl/group_stats.py).\n" + "\n".join(M) + "\n")
    rows = [f"{LABEL[d]} & \\HO{TAG[d]}Fields & \\HO{TAG[d]}First & $\\HO{TAG[d]}VThree$ & $\\HO{TAG[d]}Random$ & "
            f"\\HO{TAG[d]}Routed \\\\" for d in sets]
    tab = ["% Generated by scripts/naacl/newdata_analysis.py (docs/naacl/results/newdata.md); macros in numbers_heldout.tex.",
           "% Method v4, frozen values (docs/naacl/results/v4_frozen.json); nothing is fitted on these sets.",
           "\\begin{table}[t]", "\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\columnwidth}{!}{%",
           "\\begin{tabular}{lrrrrr}", "\\toprule", " & & & \\multicolumn{2}{c}{Change} & \\\\", "\\cmidrule(lr){4-5}",
           "Dataset & Fields & First & Map-Reflect & Random & Routed \\\\", "\\midrule", *rows, "\\midrule",
           "All held-out & \\HOAllFields & \\HOAllFirst & $\\HOAllVThree$ & $\\HOAllRandom$ & \\HOAllRouted \\\\"]
    if SANITY in R:
        tab += ["\\midrule", "Valentine Magellan & \\HOMagellanFields & \\HOMagellanFirst & $\\HOMagellanVThree$ & & \\\\"]
    tab += ["\\bottomrule", "\\end{tabular}}",
            "\\caption{On harder data the frozen thresholds route most fields, and Map-Reflect gains on every set but "
            "SchemaNet, where the change is not distinguishable from zero. Held-out datasets, run once with the "
            "thresholds frozen on development. First: first-pass accuracy "
            "(\\%), four models. Change in points. Random: a random gate with the same second pass on the same share of "
            "fields. Magellan is a sanity check.}", "\\label{tab:heldout}", "\\end{table}"]
    (paper / "tab_heldout.tex").write_text("\n".join(tab) + "\n")
    return len(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(NAMES))
    ap.add_argument("--out", default=str(ROOT / "docs/naacl/results/newdata.md"))
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--paper", default=str(ROOT / "paper"), help="where to write the paper files (empty: none)")
    ap.add_argument("--variant", default="v4", choices=["v4", "v4_1"],
                    help="v4_1: canonical margins and docs/naacl/results/v4_1_frozen.json; outputs under docs/naacl/results/"
                         "v4_1 unless --out/--paper are given")
    a = ap.parse_args()
    if a.variant == "v4_1":  # method v4.1, scored as the v6 analysis scores it: unresolved stated answers are missing
        os.environ["MAPREFLECT_CANONICAL_MARGIN"] = "1"
        os.environ["V6_UQ_VALID_ONLY"] = "1"
        router.CANONICAL = True
        if a.out == str(ROOT / "docs/naacl/results/newdata.md"):
            a.out = str(ROOT / "docs/naacl/results/newdata_v4_1.md")
        if a.paper == str(ROOT / "paper"):
            a.paper = str(ROOT / "docs/naacl/results/v4_1")
    fz = json.loads((ROOT / "docs/naacl/results" / ("v4_frozen.json" if a.variant == "v4" else "v4_1_frozen.json"))
                    .read_text())["thresholds"]
    notes = defaultdict(int)
    cells, missing = [], []
    for ds in a.datasets:
        man = json.load(open(ROOT / f"data/splits/{ds}.json"))
        gold = load_gold_file(ROOT / f"data/{ds}_dataset.jsonl")
        for m in MODELS:
            c = cell(ds, m, set(man["test"]), gold, fz[m], notes)
            (cells.append(c) if c else missing.append(f"{ds}__{m}"))
    present = sorted({c.ds for c in cells}, key=a.datasets.index)
    groups = GS.split_groups(present)
    GB = GS.GroupBootstrap(cells, groups, a.B)
    combos = fit_combo()
    swaps = [s for s in (swap_cell(c, (combos.get(c.model) or (None,))[0]) for c in cells) if s]
    GBs = GS.GroupBootstrap(swaps, groups, a.B) if swaps else None
    sets = [d for d in present if d != SANITY]
    views = [(d, [d]) for d in present] + ([("ALL", sets)] if sets else [])
    R = {}
    for name, dss in views:
        cs = [c for c in cells if c.ds in dss]
        n = sum(c.n for c in cs)
        r = dict(n=n, models=len({c.model for c in cs}), fields_per_model=n // max(1, len({c.model for c in cs})),
                 acc={p: estimate(GB, cs, dss, p) for p in POLICIES},
                 diff={}, routed=sum(c.routed.sum() for c in cs) / n,
                 calls={k: sum(np.sum(c.calls[k]) for c in cs) / n for k in cs[0].calls},
                 bands={b: int(sum((c.band == b).sum() for c in cs)) for b in router.BANDS},
                 rerun={k: np.mean([c.rerun.get(k, 0) for c in cs]) for k in ("re-ask declined", "hybrid top-20 shortlist",
                                                                                "fresh mapping")})
        for p in POLICIES:
            if p != "keep all":
                r["diff"][(p, "keep all")] = estimate(GB, cs, dss, p, "keep all")
        for x, y in PRIMARY:
            r["diff"][(x, y)] = estimate(GB, cs, dss, x, y)
        prim = [(k, r["diff"][k]) for k in PRIMARY if r["diff"].get(k)]
        if prim:
            ps = [GS.boot_p(e[3]) for _, e in prim]
            r["p"] = {k: (p_, h) for (k, _), p_, h in zip(prim, ps, GS.holm(ps))}
        if name == "ALL":
            pt, dr = GB.macro([c for c in cs], lambda c: np.nan_to_num(c.vec["v4"] - c.vec["keep all"]), dss)
            r["macro_v4"] = (pt, *GS.interval(dr))
        sw = [s for s in swaps if s.ds in dss]
        r["swap"], r["swap_acc"] = {}, {}
        if sw and GBs:
            for g in GATES:
                r["swap_acc"][g] = estimate(GBs, sw, dss, g)
                if g != "margin":
                    r["swap"][g] = estimate(GBs, sw, dss, g, "margin")
            r["swap_n"] = sum(s.n for s in sw)
        r["auroc"] = {c.model: (int(m_.sum()), 100 * (1 - c.vec["keep all"][m_].mean()),
                                auroc(c.margin[m_], c.vec["keep all"][m_]), auroc(c.verb[m_], c.vec["keep all"][m_]),
                                auroc(c.ptrue[m_], c.vec["keep all"][m_]))
                      for c in cs for m_ in [c.valid & ~np.isnan(c.verb) & ~np.isnan(c.ptrue)] if m_.sum()} \
            if name != "ALL" else {}
        R[name] = r

    f = lambda e, sign=False: "n/a" if not e else (f"{e[0]:+.2f} [{e[1]:+.2f}, {e[2]:+.2f}]" if sign else
                                                    f"{e[0]:.2f} [{e[1]:.2f}, {e[2]:.2f}]")
    nm = lambda d: "Pooled, held-out sets without Magellan" if d == "ALL" else NAMES.get(d, d)
    L = ["# Held-out datasets: method v4 on test\n",
         "Generated by `scripts/naacl/newdata_analysis.py`. Frozen v4 values (`docs/naacl/results/v4_frozen.json`, "
         "`docs/naacl/METHOD_V4.md`); nothing fitted on these sets. Routing by `scripts/naacl/router.py`; scored with the "
         "full-identity scorer (`eval.calc_accuracy.target_correct`) against every acceptable target; measured outcomes "
         "only (NT and the widen action ran on every test field). Accuracy (%) and changes (points) pooled over the four "
         f"core models with the split-group bootstrap of `scripts/naacl/group_stats.py` (B = {a.B}, 95% intervals); "
         "p-values two-sided, Holm-adjusted over the five primary contrasts within each view.\n",
         "Frozen values: " + "; ".join(f"{m} t_m {t['t_m']}, t_s {t['t_s']}, {t['near']}/{t['widen']}, {t['accept']}"
                                       for m, t in fz.items()) + ".\n",
         f"Cells: {len(cells)}" + (f"; missing: {', '.join(missing)}" if missing else "") + ". Checks: " +
         ", ".join(f"{k} {v}" for k, v in sorted(notes.items())) + ".\n",
         "## Summary\n",
         "| Dataset | Test fields per model | Keep all | v4 | v4 minus keep all | v4 minus random gate | v4 minus reflect all (v4 action) | v4 minus declined-only retry | Routed | v4 calls / field | Reflect-all calls / field |",
         "|" + "---|" * 11]
    for d, r in R.items():
        L.append(f"| {nm(d)} | {r['fields_per_model']:,} | {r['acc']['keep all'][0]:.2f} | {r['acc']['v4'][0]:.2f} | "
                 f"{f(r['diff'][('v4', 'keep all')], True)} | {f(r['diff'][('v4', 'random gate')], True)} | "
                 f"{f(r['diff'].get(('v4', 'reflect all, v4 action')), True)} | {f(r['diff'].get(('v4', 'declined-only retry')), True)} | "
                 f"{100 * r['routed']:.1f}% | {r['calls']['v4']:.2f} | {r['calls']['reflect all, NT']:.2f} |")
    if "ALL" in R:
        L.append(f"\nMacro average over the {len(sets)} held-out sets, v4 minus keep all: "
                 f"{R['ALL']['macro_v4'][0]:+.2f} [{R['ALL']['macro_v4'][1]:+.2f}, {R['ALL']['macro_v4'][2]:+.2f}].\n")
    L += ["\n## Gate swap: which signal routes the v4 action best\n",
          "Answered fields with both paid signals; at each model's v4 routed share there, each gate routes its least "
          "confident fields to the measured v4 action and keeps the first answer elsewhere (ties share the budget, "
          "`uq_ablation.gate_weights`). The logistic is fitted per model on core development fields only (" +
          ", ".join(f"{m} n = {v[1]:,}" for m, v in sorted(combos.items())) + "). Accuracy (%) and differences against "
          "the margin (points).\n",
          "| Dataset | fields | margin | P(True) | stated confidence | logistic | P(True) minus margin | stated minus margin | logistic minus margin |",
          "|" + "---|" * 9]
    for d, r in R.items():
        if not r["swap_acc"]:
            continue
        sa = lambda g: f"{r['swap_acc'][g][0]:.2f}" if r["swap_acc"].get(g) else "n/a"
        L.append(f"| {nm(d)} | {r.get('swap_n', 0):,} | {sa('margin')} | {sa('P(True)')} | {sa('stated confidence')} | "
                 f"{sa(GATES[3])} | {f(r['swap'].get('P(True)'), True)} | {f(r['swap'].get('stated confidence'), True)} | "
                 f"{f(r['swap'].get(GATES[3]), True)} |")
    for d, r in R.items():
        L += [f"\n## {nm(d)}\n"] + ([NOTES[d] + "\n"] if d in NOTES else []) + [
            f"{r['n']:,} test field-model pairs. Bands: keep {r['bands']['keep']:,}, near tie {r['bands']['near_tie']:,}, "
            f"widen {r['bands']['widen']:,}. Controls rerun on " + ", ".join(
                f"{k} {100 * v:.0f}%" for k, v in r["rerun"].items()) + " of fields (the rest keep the first pass).\n",
            "| Policy | accuracy (%) [95% CI] | change from keep all (points) | second-pass calls per field |",
            "|---|--:|--:|--:|"]
        for p in POLICIES:
            e = r["acc"].get(p)
            if not e:
                continue
            ch = r["diff"].get((p, "keep all"))
            L.append(f"| {p} | {f(e)} | {f(ch, True) if ch else '+0.00'} | "
                     + (f"{r['calls'][p]:.2f}" if p in r["calls"] else ("0.00" if p == "keep all" else "n/a")) + " |")
        L += ["\n| Primary contrast | points [95% CI] | p | Holm p |", "|---|--:|--:|--:|"]
        for k in PRIMARY:
            e = r["diff"].get(k)
            if e:
                p_, h = r.get("p", {}).get(k, (float("nan"), float("nan")))
                L.append(f"| {k[0]} minus {k[1]} | {f(e, True)} | {p_:.4f} | {h:.4f} |")
        if r["auroc"]:
            L += ["\n| Model | answered fields with both signals | first-pass error % | AUROC margin | stated | P(True) |",
                  "|---|--:|--:|--:|--:|--:|"]
            L += [f"| {m} | {x[0]} | {x[1]:.1f} | {x[2]:.3f} | {x[3]:.3f} | {x[4]:.3f} |" for m, x in r["auroc"].items()]
    Path(a.out).write_text("\n".join(L) + "\n")
    if a.paper and "ALL" in R:
        k = write_paper(Path(a.paper), R, sets)
        print(f"wrote {a.paper}/numbers_heldout.tex ({k} macros) and tab_heldout.tex")
    print("\n".join(L[4:7 + len(R)]))


if __name__ == "__main__":
    main()
