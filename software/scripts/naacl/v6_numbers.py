#!/usr/bin/env python3
"""Every number of the v6 paper (docs/naacl/v6/EXPERIMENTS.md), from the method-v4 frozen outputs only. No model call.

One entry point. It
  1. reruns the v4 generators in memory, with every file write intercepted (nothing shared is touched):
     scripts/naacl/v4_analysis.py (main table, contrasts, strata, bands, sweep), scripts/naacl/uq_ablation.py in v4
     mode (scripts/naacl/uq_ablation_v4.py: signals) and scripts/naacl/newdata_analysis.py (held-out sets), and
     checks that each regenerated file is byte-identical to the committed one;
  2. takes the v4 facts from those runs (descriptive, or confirmatory for the five Holm contrasts);
  3. adds the offline analyses made after the v4 freeze (exploratory): X1 intervention x router, X6 structure-
     conditioned repairability, X4 discrimination and calibration, X2 earlier trigger families, X3 escalation;
  4. writes docs/naacl/v6/FACTS.md, paper/numbers_v6.tex, paper/tab6_*.tex and paper/figures/fig6_*.pdf.

Reused, not rewritten: v4_analysis (load, POL, mask_cells, STRUCT, the bootstrap object), group_stats (split-group
bootstrap, B = 2000), router, uq_ablation_v4, eval.calc_accuracy.target_correct (full identity; ITOpsGraph v2.1).

Usage: ITOPS_V21=<v2.1 label file> .venv/bin/python scripts/naacl/v6_numbers.py
"""
import glob
import json
import math
import os
import pathlib
import re
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import time  # noqa: E402
_START_TIME = time.time()
# Method variant every v6 fact is computed from. v4_1 (the default since phase 4) is method v4 with canonical margins
# (router.py, MAPREFLECT_CANONICAL_MARGIN=1) and docs/naacl/results/v4_1_frozen.json, with unresolved stated answers
# treated as missing (V6_UQ_VALID_ONLY=1). V6_VARIANT=v4 V6_CHECK_ONLY=1 reruns the v4 path for the reproduction check.
VARIANT = os.environ.get("V6_VARIANT", "v4_1")
CHECK_ONLY = os.environ.get("V6_CHECK_ONLY") == "1"
if VARIANT == "v4_1":
    os.environ["MAPREFLECT_CANONICAL_MARGIN"] = "1"
    os.environ["V6_UQ_VALID_ONLY"] = "1"
FROZEN_REL = "docs/naacl/results/v4_frozen.json" if VARIANT == "v4" else "docs/naacl/results/v4_1_frozen.json"
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
LABELS = os.environ.get("ITOPS_V21")
if not LABELS or not Path(LABELS).exists():
    sys.exit("set ITOPS_V21 to the ITOpsGraph v2.1 label file (docs/naacl/plan/stats.md)")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

# ------------------------------------------------------------------------------------------------ 1. capture runs
_WRITE, _SAVE = pathlib.Path.write_text, Figure.savefig
CAP = {}


CAP9 = {}  # writes of the U9 rerun (kept apart, so the reproduction check sees only the v4 rerun)
_CAPTO = [CAP]


def _cap_write(self, data, *a, **k):
    _CAPTO[0][str(Path(self).resolve())] = data
    return len(data)


def _cap_save(self, fname, *a, **k):
    CAP[str(Path(fname).resolve())] = None


pathlib.Path.write_text, Figure.savefig = _cap_write, _cap_save
GOT = {}
UQCAP = {}  # uq_ablation_v4.run's locals at return (population, signals, bootstrap draws)
UQCAP9 = {}  # the same for the U9 rerun
try:
    argv = sys.argv
    sys.argv = ["v4_analysis.py"] + (["--variant", "v4_1"] if VARIANT == "v4_1" else [])
    import v4_analysis as V  # noqa: E402  parses its own argv (ITOPS_V21) at import
    _load, _slices, _extras = V.load, V.slices_tex, V.review_extras

    def load_w():
        GOT["load"] = _load()
        return GOT["load"]

    def slices_w(cells, POL, GB, SUBSET):
        GOT.update(POL=POL, GB=GB, SUBSET=SUBSET)
        return _slices(cells, POL, GB, SUBSET)

    def extras_w(cells, GB, POL, num, n, rows, oracle, num_s=lambda *a: None):
        GOT.update(rows=rows, oracle=oracle)
        return _extras(cells, GB, POL, num, n, rows, oracle, num_s)
    V.load, V.slices_tex, V.review_extras = load_w, slices_w, extras_w
    V.main()
    sys.argv = ["uq_ablation.py", "--labels-v21", LABELS] + (["--v4-frozen", FROZEN_REL] if VARIANT == "v4_1" else [])
    import uq_ablation  # noqa: E402,F401  cached, so newdata_analysis's own import reuses these labels
    import uq_ablation_v4 as UQ4  # noqa: E402
    # phase 2: keep the UQ run's own field population (its locals at return) and two extra score-list variants per
    # field, without changing anything it writes (the reproduction check below compares every file)
    def _score_x(score):
        def f(r):
            v = score(r)
            sc = r["scores"]
            pr = np.array([max(x, 0.0) for x in sc], float)
            tot = pr.sum()
            v["norm_margin"] = float((pr[0] - (pr[1] if len(pr) > 1 else 0.0)) / tot) if tot > 0 else 0.0
            v["gap13"] = float((sc[0] if sc else 0.0) - (sc[2] if len(sc) > 2 else 0.0))
            v["_key"] = r["key"]
            return v
        return f

    def _prof(frame, event, arg):
        if frame.f_code is UQ4.run.__code__:
            if event == "call":
                U_ = frame.f_locals["U"]
                UQCAP["U"] = U_
                U_.score_signals = _score_x(U_.score_signals)
            elif event == "return":
                UQCAP.update(frame.f_locals)
    sys.setprofile(_prof)
    try:
        runpy.run_path(str(ROOT / "scripts/naacl/uq_ablation.py"), run_name="__main__")
    finally:
        sys.setprofile(None)
    sys.argv = ["newdata_analysis.py"] + (["--variant", "v4_1"] if VARIANT == "v4_1" else [])
    runpy.run_path(str(ROOT / "scripts/naacl/newdata_analysis.py"), run_name="__main__")
    sys.argv = argv
    # U9 (phase 2): the same UQ run again, in memory, with the U9 rows (answer entropy for Granite-4 and Llama-4,
    # token log-probabilities for Llama-4; results/naacl/v6/u9) merged into the paid signals exactly as
    # scripts/naacl/v6_calls/u9_score.py does; a key that already has a value keeps it. Its writes go to CAP9.
    U9DIR = ROOT / "results/naacl/v6/u9"
    if U9DIR.exists() and not CHECK_ONLY:
        U_ = UQCAP["U"]
        _rows0 = U_.uq_rows

        def _uq_rows9(ds, m):
            out = _rows0(ds, m)
            for sig, names in (("samples", ("sem_entropy", "n_distinct")), ("logprob", ("lp_answer", "lp_mean", "lp_entropy"))):
                f = U9DIR / sig / f"{ds}__{m}.jsonl"
                if not f.exists():
                    continue
                for r in U_.read(f):
                    d = out.setdefault(r["key"], {"_lp_top": None, "_pt_src": None})
                    for s_ in names:
                        if r.get(s_) is None or s_ in d:
                            continue
                        d[s_] = -float(r[s_]) if s_ in ("sem_entropy", "n_distinct", "lp_entropy") else float(r[s_])
                    if sig == "logprob" and d.get("_lp_top") is None:
                        d["_lp_top"] = r.get("lp_top")
            return out

        def _prof9(frame, event, arg):
            if frame.f_code is UQ4.run.__code__ and event == "return":
                UQCAP9.update(frame.f_locals)
        U_.uq_rows = _uq_rows9
        _CAPTO[0] = CAP9
        sys.setprofile(_prof9)
        try:
            UQ4.run(U_)
        finally:
            sys.setprofile(None)
            _CAPTO[0] = CAP
            U_.uq_rows = _rows0
finally:
    pathlib.Path.write_text, Figure.savefig = _WRITE, _SAVE

# Table hold (V6_TABLE_HOLD=<dir>): while another agent edits paper/main_v6.tex, every paper/tab*.tex this run would
# write goes to <dir> instead, so FACTS.md and numbers_v6.tex update and no table the paper inputs changes. Off by default.
if os.environ.get("V6_TABLE_HOLD"):
    _HOLD = Path(os.environ["V6_TABLE_HOLD"])
    _HOLD.mkdir(parents=True, exist_ok=True)
    _PAPER = (ROOT / "paper").resolve()

    def _held_write(self, data, *a, **k):
        q = Path(self).resolve()
        if q.parent == _PAPER and q.name.startswith("tab") and q.suffix == ".tex":
            return _WRITE(_HOLD / q.name, data, *a, **k)
        return _WRITE(self, data, *a, **k)
    pathlib.Path.write_text = _held_write

# Every pre-v6 paper file now lives in the archive (paths below paper/ preserved). The v4 generators above still name
# their old paper/ targets, but every write was intercepted into CAP, so none of them reaches paper/; the reproduction
# check compares each regenerated paper file with its archived copy (other files, e.g. docs/naacl/results, in place).
ARCHIVE = ROOT / "paper_archive/pre_v6_2026-10-10"


def committed(p):
    p = Path(p).resolve()
    try:
        return ARCHIVE / p.relative_to((ROOT / "paper").resolve())
    except ValueError:
        return p


# guard: no intercepted v4 output may exist on disk under paper/ with a modification time from this run
_T0 = _START_TIME
for _p in CAP:
    _q = Path(_p)
    assert not (_q.exists() and str(_q.resolve()).startswith(str((ROOT / "paper").resolve())) and _q.stat().st_mtime >= _T0), _p

REPRO = []  # (file, identical to the committed copy)
for p, txt in sorted(CAP.items()):
    if txt is None:
        continue
    c = committed(p)
    if VARIANT == "v4_1":  # the second check: the v4.1 outputs of v4_analysis against the committed v4.1 replay
        rel_ = os.path.relpath(Path(p).resolve(), ROOT)
        if not (rel_.startswith("docs/naacl/results/v4_1/") or rel_ in ("docs/naacl/results/v4_1_test.md",
                                                                          "docs/naacl/results/newdata_v4_1.md")):
            continue
    REPRO.append((os.path.relpath(c, ROOT), c.exists() and c.read_text() == txt))


def OUTP(rel):
    """Where a v4 generator wrote rel in this variant (v4_analysis.OUT for v4.1; newdata_analysis --variant v4_1 writes
    its paper files under docs/naacl/results/v4_1)."""
    if VARIANT == "v4_1" and rel in ("paper/numbers_heldout.tex", "paper/tab_heldout.tex"):
        return ROOT / "docs/naacl/results/v4_1" / Path(rel).name
    if VARIANT == "v4_1" and rel == "docs/naacl/results/newdata.md":
        return ROOT / "docs/naacl/results/newdata_v4_1.md"
    return V.OUT(rel) if hasattr(V, "OUT") and rel != "paper/numbers_uq.tex" else ROOT / rel


def macros(rel):
    txt = CAP[str(OUTP(rel).resolve())]
    return {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", txt, re.M)}


MV4 = {**macros("paper/numbers_v4.tex"), **macros("paper/numbers_v4_struct.tex")}
MUQ = macros("paper/numbers_uq.tex")
# U9 rerun (exploratory): its macros and report, cross-checked against the run agent's own scoring (u9_facts.json)
_k9 = str((ROOT / "paper/numbers_uq.tex").resolve())
MUQ9 = ({m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", CAP9[_k9], re.M)}
        if _k9 in CAP9 else {})
UQMD9 = CAP9.get(str((ROOT / "docs/naacl/results/uq_ablation_v4.md").resolve()), "")
U9SIG = ("lp_answer", "sem_entropy") if MUQ9 else ()
if MUQ9 and VARIANT == "v4" and (ROOT / "docs/naacl/results/u9_facts.json").exists():
    for _k, _v in json.loads((ROOT / "docs/naacl/results/u9_facts.json").read_text()).items():
        _m = "UQAbl" + _k[len("F-U9"):]
        assert MUQ9.get(_m) == _v, ("U9 rerun differs from u9_facts.json", _k, MUQ9.get(_m), _v)
MHO = macros("paper/numbers_heldout.tex")
UQMD = CAP[str((ROOT / "docs/naacl/results/uq_ablation_v4.md").resolve())]
V4MD = CAP[str(OUTP("docs/naacl/results/v4_test.md").resolve())]

import group_stats as GS  # noqa: E402
import router  # noqa: E402
import reflect_v2 as R2  # noqa: E402
from eval.calc_accuracy import target_correct  # noqa: E402

cells, _missing, _mism = GOT["load"]
POL, GB, SUBSET, ROWS = GOT["POL"], GOT["GB"], GOT["SUBSET"], GOT["rows"]
TH, MODELS, DATASETS, FAMILY = V.TH, V.MODELS, V.DATASETS, V.FAMILY
N = sum(c.n for c in cells)
FIRST = 100 * sum(c.keep.sum() for c in cells) / N
MT = {"gptoss": "Gptoss", "llama4": "Llama", "granite": "Granite", "qwen35": "Qwen"}
MSHORT = V.SHORT
DT = {"itopsgraph": "ITOps", "toolbench": "ToolBench", "nl2gql": "NLGQL", "omop_cms": "OMOPCMS",
      "omop_mimic": "OMOPMIMIC", "omop_synthea": "OMOPSynthea", "valentine_chembl": "ValChEMBL",
      "valentine_opendata": "ValOpenData", "valentine_tpcdi": "ValTPCDI", "valentine_wikidata": "ValWikiData"}
FT = {"ITOpsGraph": "ITOps", "ToolBench": "ToolBench", "NL2GQL": "NLGQL", "OMOP": "OMOP", "Valentine": "Valentine"}
SHARE_M = {m: sum(c.routed.sum() for c in cells if c.model == m) / sum(c.n for c in cells if c.model == m)
           for m in MODELS}

# ------------------------------------------------------------------------------------------------ fact registry
FACTS = []
SRC_V4 = (("docs/naacl/results/v4_test.md" if VARIANT == "v4" else "docs/naacl/results/v4_1_test.md"), "v4_analysis.main")
SRC_V4S = (("paper_archive/pre_v6_2026-10-10/numbers_v4_struct.tex" if VARIANT == "v4" else "docs/naacl/results/v4_1/numbers_v4_struct.tex"),
           "v4_analysis.main")
SRC_UQ = (("docs/naacl/results/uq_ablation_v4.md" if VARIANT == "v4" else "uq_ablation_v4.run under v4.1 (in memory)"),
          "uq_ablation_v4.run")
SRC_HO = (("docs/naacl/results/newdata.md" if VARIANT == "v4" else "docs/naacl/results/newdata_v4_1.md (in memory)"),
          "newdata_analysis.main")
SRC_V6 = ("results/naacl/v4/test/given", "v6_numbers")


def sg(x, nd=2):
    return f"{x:+.{nd}f}"


def fact(fid, claim, value, src, status, lo=None, hi=None, func=None):
    assert re.fullmatch(r"[A-Za-z]+", fid), fid
    FACTS.append(dict(id=fid, claim=claim, value=str(value), lo=None if lo is None else str(lo),
                      hi=None if hi is None else str(hi), src=src[0], fn=func or src[1], status=status))


def fact_ci(fid, claim, est, draws, src, status, nd=2, func=None):
    lo, hi = GS.interval(draws)
    fact(fid, claim, sg(est, nd), src, status, sg(lo, nd), sg(hi, nd), func)
    return lo, hi


def pooled(rs, f):
    return GB.pooled(rs, f)


def fmt_int(x):
    return f"{int(round(x)):,}"


def note(fid, claim, text, status="note"):
    FACTS.append(dict(id=fid, claim=claim, value=text, lo=None, hi=None, src="", fn="", status=status))


# =============================================================================================== 2. v4 facts (X5)
def x5():
    D, C, E = "descriptive", "confirmatory", "exploratory"
    # data
    fact("DataDatasets", "core datasets", len(DATASETS), SRC_V4, D)
    fact("DataFamilies", "dataset families", len(FAMILY), SRC_V4, D)
    fact("DataModels", "models", len(MODELS), SRC_V4, D)
    fact("DataPairs", "test field-model pairs", fmt_int(N), SRC_V4, D)
    fact("DataCells", "model-dataset cells", len(cells), SRC_V4, D)
    fact("DataGroups", "split groups (bootstrap clusters) in the test", MV4["VFourGroupsTotal"], SRC_V4S, D)
    fact("DataFieldsPerModel", "test fields per model", fmt_int(N / len(MODELS)), SRC_V4, D)
    fact("DataAnswered", "answered (valid) first passes, field-model pairs", MV4["VFourAnsweredFields"].replace("{,}", ","), SRC_V4, D)
    fact("DataDeclined", "declined first passes, field-model pairs", fmt_int(sum((~c.valid).sum() for c in cells)), SRC_V4, D)
    for ds in DATASETS:
        ns = {c.model: c.n for c in cells if c.ds == ds}
        assert len(set(ns.values())) == 1, (ds, ns)
        fact(f"DataFields{DT[ds]}", f"{V.DNAME[ds]}: scored test fields per model", fmt_int(next(iter(ns.values()))), SRC_V4, D)
        fact(f"DataGroups{DT[ds]}", f"{V.DNAME[ds]}: split groups in the test", GB.n_groups(ds), SRC_V4, D)
    agr = json.load(open(Path(LABELS).parent / "review/agreement.json"))
    conf = {tuple(k.split(" | ")): v for k, v in agr["confusion"].items()}
    nk = sum(conf.values())
    po = sum(v for (x, y), v in conf.items() if x == y) / nk
    pa, pb = agr["pass_a_decisions"], agr["pass_b_decisions"]
    pe = sum(pa.get(x, 0) * pb.get(x, 0) for x in set(pa) | set(pb)) / nk ** 2
    kap = (po - pe) / (1 - pe)
    ks = range(nk)
    assert abs(kap - agr["decision_kappa"]) < 1e-9, (kap, agr["decision_kappa"])
    fact("ITOpsKappa", "Cohen's kappa of the two blind LLM labelling passes of ITOpsGraph v2.1, four-way decision (correct, corrected, set, "
         f"remove), over {len(ks)} gold mappings in 234 review units; recomputed from the stored confusion counts)", f"{kap:.3f}",
         ("schema-mapping-datasets@feb8a9f data/itopsgraph_v2_1/review/agreement.json", "build_itopsgraph_v2_1.py (kappa); v6_numbers.x5 recomputes"), D)
    fact("ITOpsKappaN", "mappings the ITOpsGraph v2.1 kappa is computed on (both passes)", f"{agr['mappings_reviewed']:,}",
         ("schema-mapping-datasets@feb8a9f data/itopsgraph_v2_1/review/agreement.json", "build_itopsgraph_v2_1.py"), D)
    assert agr["mappings_reviewed"] == nk
    hos = ["mimic_omop_rematch", "semtab_m2kg_r1", "oaei_conference", "bioml_omim_ordo", "schemanet_cprd"]
    hog = {}
    for ds in hos:
        man = json.loads((ROOT / f"data/splits/{ds}.json").read_text())
        hog[ds] = sum(1 for v in (man.get("groups") or {}).values() if v == "test")
    hgu = GS.split_groups(hos)
    for ds in hos:
        man = json.loads((ROOT / f"data/splits/{ds}.json").read_text())
        ids = set(man["test"]) - set(man.get("old_dev_excluded") or [])
        rows_ = {str(json.loads(l)["sample_id"]) for l in open(ROOT / f"results/naacl/v4/test/given/{ds}__gptoss/fields.jsonl")}
        used = {hgu[(ds, x)] for x in ids & rows_}
        assert len(used) == hog[ds], (ds, len(used), hog[ds])  # the manifest's test groups are the bootstrap units used
    HOT = {"mimic_omop_rematch": "Mimic", "semtab_m2kg_r1": "SemTab", "oaei_conference": "OAEI", "bioml_omim_ordo": "BioML",
           "schemanet_cprd": "SchemaNet"}
    for ds in hos:
        fact(f"HOGroups{HOT[ds]}", f"held-out {ds}: test split groups (bootstrap units)", fmt_int(hog[ds]), (f"data/splits/{ds}.json", "groups"), D)
    fact("HOGroupsTotal", "held-out sets: test split groups in all five", fmt_int(sum(hog.values())), ("data/splits/<held-out>.json", "groups"), D)
    fact("DataHOSets", "held-out datasets (Magellan sanity row excluded)", MHO["HONSets"], SRC_HO, D)
    fact("DataHOFields", "held-out test fields per model, pooled", MHO["HOAllFields"], SRC_HO, D)
    # frozen values; thresholds print with two decimals (Table 7's uniform decimals), never losing a digit
    def t2(x):
        out = f"{float(x):.2f}"
        assert abs(float(out) - float(x)) < 1e-12, (x, out)
        return out
    for m in MODELS:
        fact(f"FrozenTm{MT[m]}", f"{MSHORT[m]}: frozen margin threshold t_m", t2(TH[m]["t_m"]), (FROZEN_REL, "v4_fit"), D)
        fact(f"FrozenK{MT[m]}", f"{MSHORT[m]}: frozen widening size K", TH[m]["widen"].lstrip("WTX"), (FROZEN_REL, "v4_fit"), D)
    for m in MODELS:
        fact(f"FrozenTs{MT[m]}", f"{MSHORT[m]}: frozen top-score threshold t_s", t2(TH[m]["t_s"]), (FROZEN_REL, "v4_fit"), D)
    ts = {m: TH[m]["t_s"] for m in MODELS}
    if len(set(ts.values())) == 1:
        fact("FrozenTs", "frozen top-score threshold t_s (the same for every model)", t2(next(iter(ts.values()))), (FROZEN_REL, "v4_fit"), D)
    else:  # differs by model: a note generated from the file, no single number
        groups = defaultdict(list)
        for m, v in ts.items():
            groups[v].append(MSHORT[m])
        common = max(groups, key=lambda v: len(groups[v]))
        txt = ", ".join(f"{v} for {', '.join(groups[v])}" for v in groups if v != common) + f", {common} for the others"
        note("FrozenTs", f"frozen top-score threshold t_s ({FROZEN_REL})", txt)
    for key in ("near", "accept"):  # other frozen values: one fact only when every model shares it
        vals = {TH[m].get(key) for m in MODELS}
        assert len(vals) == 1, (key, vals)  # a per-model difference would need per-model facts, as t_s above
    fact("FrozenWidenInType", "every model's fitted widen action stays inside the named target type (WT, not WX)",
         "yes" if all(TH[m]["widen"].startswith("WT") for m in MODELS) else "no", (FROZEN_REL, "v4_fit"), D)
    fact("FrozenDevAcc", "development accuracy at the frozen values, %", MV4["VFourDevAcc"], (FROZEN_REL, "v4_fit"), D)
    fact("FrozenDevKeep", "development keep-all accuracy, %", MV4["VFourDevKeep"], (FROZEN_REL, "v4_fit"), D)
    note("RouterInputs", "the frozen router reads only validity, the margin and the top score, never the target type's "
         "width (scripts/naacl/router.py, band())", "method fact: band(margin, valid, s1, t_m, t_s)")
    # main table rows
    TAG = {"Keep": "Keep", "Method": "Main", "Random": "Rand", "RandomDS": "RandDS", "RandomStrat": "RandStrat",
           "ReflectAllAction": "Every", "ReflectAll": "ReflectNT", "DeclRetry": "Decl", "ReAsk": "ReAsk",
           "FullList": "FullList", "Hybrid": "Hybrid", "Fresh": "Fresh", "Oracle": "Oracle", "Structural": "Struct"}
    LAB = dict(V.__dict__.get("LABEL", {}))
    for k, tg in TAG.items():
        a, e, lo, hi, en, em, cl, tk = ROWS[k]
        src = SRC_V4S if k in ("Structural", "RandomStrat") else SRC_V4
        nm = {"Structural": "structural router", "RandomStrat": "random gate per structural stratum"}.get(k, k)
        fact(f"{tg}Acc", f"{nm}: test accuracy, %", f"{a:.2f}", src, D)
        if k != "Keep":
            fact(f"{tg}Gain", f"{nm}: change over the first pass, points", sg(e), src, D, sg(lo), sg(hi))
            fact(f"{tg}NoOD", f"{nm}: change without Valentine OpenData", sg(en), src, D)
            fact(f"{tg}Macro", f"{nm}: unweighted mean change over datasets", sg(em), src, D)
            if cl == cl:
                fact(f"{tg}Calls", f"{nm}: second-pass calls per field", f"{cl:.2f}", src, D)
                fact(f"{tg}Tokens", f"{nm}: second-pass tokens per field", fmt_int(tk), src, D)
        if k in SUBSET and len(SUBSET[k]) != len(cells):
            fact(f"{tg}Pairs", f"{nm}: field-model pairs it is measured on", fmt_int(sum(c.n for c in SUBSET[k])), src, D)
    fact("Keep", "first-pass accuracy, %", f"{FIRST:.2f}", SRC_V4, D)
    fact("MainRouted", "share of fields Map-Reflect routes to a second pass, %", MV4["VFourRouted"], SRC_V4, D)
    fact("EveryCallShare", "Map-Reflect calls as a share of the every-field second pass, %", MV4["VFourCallShare"], SRC_V4, D)
    fact("EveryTokenShare", "Map-Reflect tokens as a share of the every-field second pass, %", MV4["VFourTokenShare"], SRC_V4, D)
    fact("EveryNILower", "Map-Reflect minus every field: one-sided 95% lower bound, points", MV4["VFourNILower"], SRC_V4, D)
    fact("EveryNIMargin", "pre-stated non-inferiority margin, points", MV4["VFourNIMargin"], SRC_V4, D)
    fact("EveryNIResult", "non-inferiority verdict against every field", MV4["VFourNIResult"], SRC_V4, D)
    fact("OracleOverMain", "oracle minus Map-Reflect, points", MV4["VFourOracleOverMethod"], SRC_V4, D)
    # primary contrasts, recomputed exactly as v4_analysis (same POL, GB, SUBSET), with Holm
    PRIM = [("Keep", "Keep"), ("Random", "Rand"), ("ReflectAllAction", "Every"), ("DeclRetry", "Decl"), ("FullList", "FullList")]
    for fam in ("pooled", "macro"):
        res = []
        for k, tg in PRIM:
            rs = SUBSET.get(k, cells)
            e, d = getattr(GB, fam)(rs, (lambda kk: lambda c: POL["Method"](c) - POL[kk](c))(k))
            res.append((k, tg, e, d, GS.boot_p(d)))
        adj = GS.holm([x[4] for x in res])
        for (k, tg, e, d, p), pa in zip(res, adj):
            suf = "" if fam == "pooled" else "Macro"
            st = C if fam == "pooled" else D
            fact_ci(f"MainVs{tg}{suf}", f"Map-Reflect minus {k} ({'field-pooled, primary family' if fam == 'pooled' else 'macro over datasets, secondary family'})",
                    e, d, SRC_V4, st)
            fact(f"MainVs{tg}{suf}Holm", f"Holm-adjusted p, Map-Reflect minus {k} ({fam})", f"{pa:.4f}", SRC_V4, st)
    fact("Holm", "Holm-adjusted p of each of the five primary pooled contrasts (bootstrap floor 5/B)",
         f"{max(GS.holm([GS.boot_p(GB.pooled(SUBSET.get(k, cells), (lambda kk: lambda c: POL['Method'](c) - POL[kk](c))(k))[1]) for k, _ in PRIM])):.4f} (largest of the five)",
         SRC_V4, C)
    for k, tg in (("RandomDS", "RandDS"), ("Hybrid", "Hybrid"), ("Fresh", "Fresh"), ("ReAsk", "ReAsk"),
                  ("Oracle", "Oracle"), ("ReflectAll", "ReflectNT")):
        rs = SUBSET.get(k, cells)
        e, d = pooled(rs, (lambda kk: lambda c: POL["Method"](c) - POL[kk](c))(k))
        fact_ci(f"MainVs{tg}", f"Map-Reflect minus {k} (outside the Holm family)", e, d, SRC_V4, D)
    fact("MainVsRandStrat", "Map-Reflect minus the random gate matched per structural stratum and cell", MV4["VFourVsRandomStrat"],
         SRC_V4S, E, MV4["VFourVsRandomStratLo"], MV4["VFourVsRandomStratHi"])
    fact("MainVsStruct", "Map-Reflect minus the structural router", MV4["VFourVsStructural"], SRC_V4, D,
         MV4["VFourVsStructuralLo"], MV4["VFourVsStructuralHi"])
    fact("UnionGain", "structure-or-margin (union) routing: change over the first pass", MV4["VFourUnionRouter"], SRC_V4, D)
    fact("UnionCalls", "union routing: second-pass calls per field", MV4["VFourUnionRouterCalls"], SRC_V4, D)
    fact("UnionRouted", "union routing: share of fields routed, %", MV4["VFourUnionRouterRouted"], SRC_V4, D)
    fact("UnionOverMain", "union routing minus Map-Reflect, points", MV4["VFourUnionOverMethod"], SRC_V4S, D,
         MV4["VFourUnionOverMethodLo"], MV4["VFourUnionOverMethodHi"])
    fact("StructRouted", "structural router: share of fields routed, %", MV4["VFourStructuralRouterRouted"], SRC_V4, D)
    # without OpenData / ITOpsGraph / Valentine
    e, d = pooled([c for c in cells if c.ds != "itopsgraph"], POL["Method"])
    fact_ci("MainNoITOps", "Map-Reflect change without ITOpsGraph (field-pooled)", e, d, SRC_V6, D, func="v6_numbers.x5")
    e, d = pooled([c for c in cells if c.ds != "valentine_opendata"], POL["Method"])
    fact_ci("MainNoODCI", "Map-Reflect change without Valentine OpenData, with interval", e, d, SRC_V6, D, func="v6_numbers.x5")
    fact("MainFamMacro", "Map-Reflect change, unweighted mean over the five families", MV4["VFourMethodFamMacro"], SRC_V4, D,
         MV4["VFourMethodFamMacroLo"], MV4["VFourMethodFamMacroHi"])
    fact("MainFamNoVal", "Map-Reflect change, unweighted mean over the four families without Valentine",
         MV4["VFourMethodFamMacroNoVal"], SRC_V4, D, MV4["VFourMethodFamMacroNoValLo"], MV4["VFourMethodFamMacroNoValHi"])
    # per family and per model
    def block(prefix, name, rs):
        n_ = sum(c.n for c in rs)
        fact(f"{prefix}First", f"{name}: first-pass accuracy, %", f"{100 * sum(c.keep.sum() for c in rs) / n_:.2f}", SRC_V4, D)
        fact(f"{prefix}Fields", f"{name}: field-model pairs", fmt_int(n_), SRC_V4, D)
        for k, tg in (("Method", "Gain"), ("Random", "Rand"), ("ReflectAllAction", "Every"), ("Oracle", "Oracle"),
                      ("DeclRetry", "Decl"), ("RandomDS", "RandDS")):
            e, d = pooled([c for c in rs if c in SUBSET.get(k, rs)] or rs, POL[k])
            if k == "Oracle":
                fact(f"{prefix}{tg}", f"{name}: {k} change", sg(e), SRC_V4, D)
            else:
                fact_ci(f"{prefix}{tg}", f"{name}: {k} change", e, d, SRC_V4, D)
        e, d = pooled(rs, lambda c: POL["Method"](c) - POL["Random"](c))
        fact_ci(f"{prefix}VsRand", f"{name}: Map-Reflect minus the random gate", e, d, SRC_V6, D, func="v6_numbers.x5")
        fact(f"{prefix}Routed", f"{name}: share routed, %", f"{100 * sum(c.routed.sum() for c in rs) / n_:.1f}", SRC_V4, D)
    gm = {}
    for m in MODELS:
        rs = [c for c in cells if c.model == m]
        block(f"Model{MT[m]}", MSHORT[m], rs)
        gm[m] = pooled(rs, POL["Method"])[0]
    fact("ModelMin", "smallest per-model Map-Reflect change", sg(min(gm.values())), SRC_V4, D)
    fact("ModelMax", "largest per-model Map-Reflect change", sg(max(gm.values())), SRC_V4, D)
    for name, dss in FAMILY:
        block(f"Fam{FT[name]}", name, [c for c in cells if c.ds in dss])
        tg = FT[name]
        fact(f"Fam{tg}StructGain", f"{name}: structural router change", MV4[f"VFourStructFam{tg}"], SRC_V4S, D)
        fact(f"Fam{tg}StructCalls", f"{name}: structural router calls per field", MV4[f"VFourStructFam{tg}Calls"], SRC_V4S, D)
        fact(f"Fam{tg}MainCalls", f"{name}: Map-Reflect calls per field", MV4[f"VFourMethodFam{tg}Calls"], SRC_V4S, D)
        fact(f"Fam{tg}VsStruct", f"{name}: Map-Reflect minus the structural router", MV4[f"VFourVsStructFam{tg}"], SRC_V4S, D,
             MV4[f"VFourVsStructFam{tg}Lo"], MV4[f"VFourVsStructFam{tg}Hi"])
    for ds in DATASETS:
        rs = [c for c in cells if c.ds == ds]
        e, d = pooled(rs, POL["Method"])
        fact(f"DSFirst{DT[ds]}", f"{V.DNAME[ds]}: first-pass accuracy, %", f"{100 * sum(c.keep.sum() for c in rs) / sum(c.n for c in rs):.2f}", SRC_V6, D, func="v6_numbers.x5")
        fact_ci(f"DSGain{DT[ds]}", f"{V.DNAME[ds]}: Map-Reflect change", e, d, SRC_V6, D, func="v6_numbers.x5")
    # bands
    for tg, b in (("Keep", router.KEEP), ("Reflect", router.NEAR_TIE), ("Widen", router.WIDEN)):
        mk = [(c, c.band == b) for c in cells]
        tot = sum(k.sum() for _, k in mk)
        fx = int(sum(((c.keep == 0) & (c.method == 1) & k).sum() for c, k in mk))
        br = int(sum(((c.keep == 1) & (c.method == 0) & k).sum() for c, k in mk))
        fact(f"BandShare{tg}", f"share of fields in the {b} band, %", f"{100 * tot / N:.1f}", SRC_V4, D)
        fact(f"BandFirst{tg}", f"first-pass accuracy in the {b} band, %", f"{100 * sum(c.keep[k].sum() for c, k in mk) / tot:.1f}", SRC_V4, D)
        fact(f"BandRepairs{tg}", f"repairs (wrong to right) in the {b} band", fmt_int(fx), SRC_V4, D)
        fact(f"BandBreaks{tg}", f"breaks (right to wrong) in the {b} band", fmt_int(br), SRC_V4, D)
    for tg in ("Declined", "Wide", "Rest"):
        fact(f"Net{tg}", f"net repairs of Map-Reflect from {tg.lower()} fields", MV4[f"VFourNet{tg}"].replace("{,}", ","), SRC_V4, D)
        fact(f"Net{tg}Share", f"share of Map-Reflect's net repairs from {tg.lower()} fields, %", MV4[f"VFourNet{tg}Share"], SRC_V4, D)
    # strata
    for tg in ("Declined", "Wide", "Shown"):
        fact(f"Strat{tg}Share", f"share of test pairs in the {tg.lower()} stratum, %", MV4[f"VFourStrat{tg}Share"], SRC_V4S, D)
        fact(f"Strat{tg}Routed", f"share of the {tg.lower()} stratum Map-Reflect routes, %", MV4[f"VFourStrat{tg}Routed"], SRC_V4S, D)
        fact(f"Strat{tg}Gain", f"Map-Reflect change inside the {tg.lower()} stratum", MV4[f"VFourStrat{tg}Gain"], SRC_V4S, D)
        fact(f"Strat{tg}", f"{tg.lower()} stratum: Map-Reflect minus random at Map-Reflect's share of the stratum in each "
             "model-dataset cell (the chosen, per-cell stratum-matched estimand)", MV4[f"VFourStrat{tg}VsRandom"], SRC_V4S, E,
             MV4[f"VFourStrat{tg}VsRandomLo"], MV4[f"VFourStrat{tg}VsRandomHi"])
    note("StratShownEstimand", "shown-stratum estimand", "F-StratShown uses the per-cell stratum-matched random gate "
         f"({MV4['VFourStratShownVsRandom']}). v4_test.md also reports {MV4['VFourShownVsRandom']} "
         "[" + MV4['VFourShownVsRandomLo'] + ", " + MV4['VFourShownVsRandomHi'] + "] with random matched per model "
         "(F-StratShownPerModel); the paper uses the per-cell value.")
    fact("StratShownPerModel", "shown stratum: Map-Reflect minus random matched per model (not the chosen estimand)",
         MV4["VFourShownVsRandom"], SRC_V4, E, MV4["VFourShownVsRandomLo"], MV4["VFourShownVsRandomHi"])
    # errors at the margin gate (uq population)
    fact("ErrRouted", "share of answered first-pass errors the margin gate routes at v4's share, % (uq population)", MUQ["UQAblErrRouted"], SRC_UQ, D)
    fact("ErrInShown", "of those routed errors, share whose gold is among the candidates the first pass RANKED, % "
         "(uq_ablation_v4 'shown' = gold in the first-pass ranking, not in the 20 shown candidates; see X6)", MUQ["UQAblErrInShown"], SRC_UQ, D)
    fact("ErrRepaired", "of those routed errors, share the v4 action repairs, %", MUQ["UQAblErrRepaired"], SRC_UQ, D)
    # held-out
    for k, tg, cl in (("First", "First", "first-pass accuracy, %"), ("VFour", "Acc", "Map-Reflect accuracy, %"),
                      ("Routed", "Routed", "share routed, %"), ("Calls", "Calls", "Map-Reflect calls per field"),
                      ("ReflectActCalls", "EveryCalls", "every-field second pass (v4 action) calls per field"),
                      ("ReflectAllCalls", "ReflectNTCalls", "reflect-over-shown on every field, calls per field"),
                      ("ReflectAct", "Every", "every-field second pass change"), ("Oracle", "Oracle", "oracle change"),
                      ("Random", "Rand", "random gate change"), ("Decl", "Decl", "declined retry change")):
        fact(f"HO{tg}", f"held-out pooled: {cl}", MHO[f"HOAll{k}"], SRC_HO, D)
    for k, tg, cl in (("VThree", "Gain", "Map-Reflect change"), ("VMinusRandom", "VsRand", "Map-Reflect minus random"),
                      ("VMinusDecl", "VsDecl", "Map-Reflect minus declined retry"),
                      ("VMinusReflectAct", "VsEvery", "Map-Reflect minus every-field second pass"),
                      ("MacroVThree", "GainMacro", "Map-Reflect change, mean over sets")):
        fact(f"HO{tg}", f"held-out pooled: {cl}", MHO[f"HOAll{k}"], SRC_HO, D, MHO[f"HOAll{k}Lo"], MHO[f"HOAll{k}Hi"])
    sets = sorted({m.group(1) for k in MHO for m in [re.fullmatch(r"HO(\w+?)Fields", k)] if m and m.group(1) not in ("All", "Magellan")})
    for s in sets:
        for k, cl in (("Fields", "test fields per model"), ("First", "first pass, %"), ("VFour", "Map-Reflect, %"),
                      ("Routed", "share routed, %"), ("Calls", "calls per field")):
            fact(f"HO{s}{k}", f"held-out {s}: {cl}", MHO[f"HO{s}{k}"], SRC_HO, D)
        for k, cl in (("VThree", "Map-Reflect change"), ("VMinusRandom", "Map-Reflect minus random"),
                      ("VMinusReflectAct", "Map-Reflect minus every field"), ("VMinusDecl", "Map-Reflect minus declined retry")):
            fact(f"HO{s}{k.replace('VThree', 'Gain')}", f"held-out {s}: {cl}", MHO[f"HO{s}{k}"], SRC_HO, D,
                 MHO[f"HO{s}{k}Lo"], MHO[f"HO{s}{k}Hi"])
    # ablations: sensitivity, K, other mappers
    for m in MODELS:
        fact(f"Sens{MT[m]}", f"{MSHORT[m]}: development accuracy range within 0.1 of the frozen t_m, points", MV4[f"VFourSens{ {'gptoss': 'Gptoss', 'llama4': 'Llamafour', 'granite': 'Granite', 'qwen35': 'Qwen'}[m] }"], ("paper_archive/pre_v6_2026-10-10/tab_v4_sensitivity.tex", "v4_analysis.review_extras"), D)
    fit = (ROOT / "docs/naacl/results/v4_dev_fit.md").read_text()
    for m in MODELS:
        row = re.search(rf"^\| {m} \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$", fit, re.M)
        if row:
            fact(f"KDevTwenty{MT[m]}", f"{MSHORT[m]}: development accuracy with K = 20, %", row.group(1), ("docs/naacl/results/v4_dev_fit.md", "v4_fit"), D)
            fact(f"KDevFifty{MT[m]}", f"{MSHORT[m]}: development accuracy with K = 50, %", row.group(2), ("docs/naacl/results/v4_dev_fit.md", "v4_fit"), D)
            fact(f"KDevBeyond{MT[m]}", f"{MSHORT[m]}: development accuracy widening beyond the type (WTX50), %", row.group(4), ("docs/naacl/results/v4_dev_fit.md", "v4_fit"), D)
    tb = (ARCHIVE / "tab_reflect_baselines.tex").read_text()
    allrow = re.search(r"^All & ([\d.]+) & ([\d.]+) & \$-\$([\d.]+) & ([+\-\d.]+) & ([+\-\d.]+) & ([+\-\d.]+)", tb, re.M)
    src = ("paper_archive/pre_v6_2026-10-10/tab_reflect_baselines.tex", "v2_baselines.py (pre-v4 reflect arm, allowed by the spec)")
    fact("OtherMappersFirst", "other mappers (Magneto, MatchMaker), pooled first-pass accuracy, %", allrow.group(1), src, D)
    fact("OtherMappersAUC", "other mappers: AUROC of their own margin for their errors", allrow.group(2), src, D)
    fact("OtherMappersDoubt", "other mappers: doubt prompt on every field, change", "-" + allrow.group(3), src, D)
    fact("OtherMappersEvery", "other mappers: Reflect on every field, change", allrow.group(4), src, D)
    fact("OtherMappersHalf", "other mappers: Reflect on the small-margin half, change", allrow.group(5), src, D)
    fact("OtherMappersRandHalf", "other mappers: Reflect on a random half, change", allrow.group(6), src, D)
    signals()


# ------------------------------------------------------------------------------------------------ signals (Table 3)
SIGS = [("margin", "Margin", "free", "BB"), ("s1", "TopScore", "free", "BB"), ("ratio", "Ratio", "free", "BB"),
        ("entropy", "ScoreEnt", "free", "BB"), ("lp_answer", "AnsLogp", "free", "WB"),
        ("verbalised", "Stated", "paid", "BB"), ("p_true", "PTrue", "paid", "WB (BB for Llama-4)"),
        ("sem_entropy", "SemEnt", "paid", "BB"), ("combo_paid", "ComboPaid", "paid", "WB")]
SIGTAB = []
SIGX = {}  # phase 2 (v6_phase2.r2_signals): held-out AUROC and share of the oracle gain filled for every signal


def uq_section(model_label, md=None):
    m = re.search(rf"### {re.escape(model_label)} \([\d,]+ fields\)\n(.*?)\n\n", (md or UQMD) + "\n\n", re.S)
    return m.group(1) if m else ""


def uq_row(section, label):
    for line in section.splitlines():
        cols = [x.strip() for x in line.strip("|").split("|")]
        if cols and cols[0] == label:
            return cols
    return None


def signals():
    D = "descriptive"
    ho = dict(margin=MUQ["UQAblHOMarginMeanAUC"], verbalised=MUQ["UQAblHOStatedMeanAUC"], p_true=MUQ["UQAblHOPTrueMeanAUC"],
              ratio=MUQ["UQAblHORatioMeanAUC"])
    allsec = uq_section("All four models")
    allsec9 = uq_section("All four models", UQMD9) if UQMD9 else ""
    for key, tg, fp, bw in SIGS:
        lab, calls = UQ4.SIG[key]
        if key in U9SIG:  # pooled over all four models with the U9 rows; the earlier subset values stay as Sub facts
            M_, D_, sec, SRC_ = MUQ9, "exploratory", allsec9, ("docs/naacl/results/u9_uq_ablation_v6.md", "uq_ablation_v4.run + U9 rows (v6_numbers)")
            sub = [x for x in ("AUC", "Delta", "Acc") if f"UQAbl{tg}{x}" in MUQ]
            for x in sub:
                fact(f"Sig{tg}Sub{x}", f"{lab}: {x} before U9, on the {MUQ.get(f'UQAbl{tg}Models', 'subset of')} models that had it then",
                     MUQ[f"UQAbl{tg}{x}"], SRC_UQ, "exploratory", MUQ.get(f"UQAbl{tg}Lo") if x == "Delta" else None,
                     MUQ.get(f"UQAbl{tg}Hi") if x == "Delta" else None)
        else:
            M_, D_, sec, SRC_ = MUQ, D, allsec, SRC_UQ
        auc = M_.get(f"UQAbl{tg}AUC")
        fact(f"Sig{tg}AUC", f"{lab}: pooled core-test error AUROC (within-model ranks)" + (", all four models (U9)" if key in U9SIG else ""),
             auc, SRC_, D_)
        fact(f"Sig{tg}Calls", f"{lab}: extra calls per field", calls, SRC_UQ, D)
        fact(f"Sig{tg}Kind", f"{lab}: free or paid, black-box or white-box", f"{fp}, {bw}", SRC_UQ, D)
        if key != "margin":
            fact(f"Sig{tg}Delta", f"{lab}: routing at v4's share minus the margin, points", M_[f"UQAbl{tg}Delta"], SRC_, D_,
                 M_[f"UQAbl{tg}Lo"], M_[f"UQAbl{tg}Hi"])
        fact(f"Sig{tg}Acc", f"{lab}: accuracy routing at v4's share, %", M_.get(f"UQAbl{tg}Acc", M_["UQAblMarginAcc"]), SRC_, D_)
        if key in ho:
            fact(f"Sig{tg}HOAUC", f"{lab}: mean held-out within-cell error AUROC (20 cells)", ho[key], SRC_UQ, D)
        r = uq_row(sec, lab)
        share = r[3] if r else None
        if share:
            fact(f"Sig{tg}OracleShare", f"{lab}: share of the oracle gain at v4's share, pooled over models where available"
                 + (" (all four, U9)" if key in U9SIG else ""), share.rstrip("%"), SRC_, D_)
        SIGTAB.append((key, tg, lab, fp, bw, calls, auc, ho.get(key), M_.get(f"UQAbl{tg}Delta"), M_.get(f"UQAbl{tg}Lo"),
                       M_.get(f"UQAbl{tg}Hi"), share))
    if VARIANT == "v4_1":
        prefix_facts()
    fact("SigStatedHOWins", "held-out cells where stated confidence has the higher error AUROC than the margin",
         f"{MUQ['UQAblHOStatedWins']} of {MUQ['UQAblHOCells']}", SRC_UQ, D)
    fact("SigPTrueHOWins", "held-out cells where P(True) beats the margin's error AUROC", f"{MUQ['UQAblHOPTrueWins']} of {MUQ['UQAblHOCells']}", SRC_UQ, D)
    fact("SigStatedCoreWins", "core-test cells where stated confidence beats the margin's error AUROC", f"{MUQ['UQAblCoreStatedWins']} of {MUQ['UQAblCoreCells']}", SRC_UQ, D)
    fact("SigCoreMarginMeanAUC", "mean core-test within-cell error AUROC, margin", MUQ["UQAblCoreMarginMeanAUC"], SRC_UQ, D)
    fact("SigCoreStatedMeanAUC", "mean core-test within-cell error AUROC, stated confidence", MUQ["UQAblCoreStatedMeanAUC"], SRC_UQ, D)
    fact("SigHOMeanAUC", "mean held-out within-cell error AUROC: stated vs margin",
         f"{MUQ['UQAblHOStatedMeanAUC']} vs {MUQ['UQAblHOMarginMeanAUC']}", SRC_UQ, D)
    st = uq_row(uq_section("gpt-oss-120b"), "stated confidence")
    m_ = re.match(r"([+\-][\d.]+) \[([+\-][\d.]+), ([+\-][\d.]+)\]", st[4])
    fact("SigStatedGptoss", "gpt-oss-120b: stated confidence routing minus the margin (the one significant exception)",
         m_.group(1), SRC_UQ, D, m_.group(2), m_.group(3))
    for lab, tg in (("gpt-oss-120b", "Gptoss"), ("Granite-4", "Granite"), ("Llama-4-Maverick", "Llama"), ("Qwen3.5-397B", "Qwen"),
                    ("All four models", "All")):
        r = uq_row(uq_section(lab), "top-2 margin (ours)")
        fact(f"ShareOfOracle{tg}", f"{lab}: share of the oracle gain the margin gate captures at v4's share, %", r[3].rstrip("%"), SRC_UQ, D)
    mreg = re.search(r"regenerated answer equals the scored one for gpt-oss-120b ([\d.]+)%, Granite-4 ([\d.]+)%, Qwen3.5-397B ([\d.]+)%", UQMD)
    # DECISIONS row 63: the stored share counts a field with no recorded regenerated answer as a mismatch. The
    # corrected share is over fields whose regenerated answer is recorded; the stored one stays as ...IncludingMissing.
    # Counted on the UQ population of the U9 rerun (the same fields as the v4 run, with Llama-4's log-probabilities).
    popu = (UQCAP9 or UQCAP).get("pop") or {}
    old = dict(zip(("Gptoss", "Granite", "Qwen"), mreg.groups()))
    for m, tg in (("gptoss", "Gptoss"), ("llama4", "Llama"), ("granite", "Granite"), ("qwen35", "Qwen")):
        fs = [f for f in popu.get(m, []) if "lp_answer" in f["sig"]]
        if not fs or "lp_rec" not in fs[0]:
            continue
        rec = sum(bool(f["lp_rec"]) for f in fs)
        agree = sum(bool(f["lp_agree"]) and bool(f["lp_rec"]) for f in fs)
        incl = 100 * agree / len(fs)
        if tg in old:
            assert f"{incl:.1f}" == old[tg], (tg, incl, old[tg])  # the stored share, reproduced
        fact(f"SigTokRegenMatch{tg}", f"{tg}: token log-probability regeneration, share of fields with a recorded "
             "regenerated answer whose regenerated answer equals the scored one, %", f"{100 * agree / rec:.1f}", SRC_UQ, D)
        fact(f"SigTokRegenMatch{tg}IncludingMissing", f"{tg}: the same share over every field with a token "
             "log-probability, a field with no recorded regenerated answer counted as a mismatch (the stored rule), %",
             f"{incl:.1f}", SRC_UQ, D)
        fact(f"SigTokRegenMissing{tg}", f"{tg}: fields with a token log-probability but no recorded regenerated answer",
             f"{len(fs) - rec:,}", SRC_UQ, D)
    row = next(l for l in UQMD.splitlines() if l.startswith("| same answer, gpt-oss-120b, Granite-4, Qwen3.5-397B | answer-token log-prob. |"))
    cols = [x.strip() for x in row.strip("|").split("|")]
    mm = re.match(r"([+\-][\d.]+) \[([+\-][\d.]+), ([+\-][\d.]+)\]", cols[4])
    fact("SigTokMatchedN", "answered pairs whose regenerated answer is the scored one (gpt-oss, Granite-4, Qwen3.5)", cols[2], SRC_UQ, D)
    fact("SigTokMatchedAUC", "answer-token log-probability: error AUROC on those matched pairs", cols[3], SRC_UQ, D)
    fact("SigTokMatchedVsMargin", "matched pairs: log-probability AUROC minus margin AUROC (signal minus margin, as stored)",
         mm.group(1), SRC_UQ, D, mm.group(2), mm.group(3))
    fact("SigFields", "answered fields in the signal comparison", MUQ["UQAblFields"].replace("{,}", ","), SRC_UQ, D)
    for k, tg, lab in (("SwapPTrue", "PTrue", "P(True)"), ("SwapStated", "Stated", "stated confidence"),
                       ("SwapLogit", "Logit", "logistic (core-development fit)")):
        fact(f"SigHOSwap{tg}", f"held-out pooled: routing by {lab} minus the margin at v4's share", MHO[f"HOAll{k}"], SRC_HO, D,
             MHO[f"HOAll{k}Lo"], MHO[f"HOAll{k}Hi"], func="newdata_analysis.swap_cell")
    note("SigLogisticFit", "how the logistic combination is fitted",
         "core test: cross-fitted on the test labels over split groups (2 folds x 5 repeats, uq_ablation_v4.crossfit_groups); "
         "exploratory cross-validation, not a dev-frozen router. Held-out: fitted per model on core development fields only "
         "(newdata_analysis.fit_combo), no held-out label enters.")


def prefix_facts():
    """Phase 4 item C: the stated-confidence and logistic values before the fixes (method v4, unresolved stated answers
    scored as 1 - c), from the committed v4 files, kept as exploratory *PreFix facts; plus the invalid-row counts."""
    E = "exploratory"
    mac = lambda f: {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", f.read_text(), re.M)}  # noqa: E731
    U0 = mac(ARCHIVE / "numbers_uq.tex")
    H0 = mac(ARCHIVE / "numbers_heldout.tex")
    md0 = (ROOT / "docs/naacl/results/uq_ablation_v4.md").read_text()
    sec0 = uq_section("All four models", md0)
    SRC0 = ("paper_archive/pre_v6_2026-10-10/numbers_uq.tex", "uq_ablation_v4.run (v4)")
    for tg, lab in (("Stated", "stated confidence"), ("ComboPaid", "logistic: margin, top score, P(True), stated")):
        fact(f"Sig{tg}PreFixAUC", f"{lab}: pooled core-test error AUROC before the fixes (v4; unresolved answers as 1 - c)",
             U0[f"UQAbl{tg}AUC"], SRC0, E)
        fact(f"Sig{tg}PreFixDelta", f"{lab}: routing at v4's share minus the margin, before the fixes", U0[f"UQAbl{tg}Delta"], SRC0, E,
             U0[f"UQAbl{tg}Lo"], U0[f"UQAbl{tg}Hi"])
        r = uq_row(sec0, lab)
        if r:
            fact(f"Sig{tg}PreFixOracleShare", f"{lab}: share of the oracle gain before the fixes, %", r[3].rstrip("%"),
                 ("docs/naacl/results/uq_ablation_v4.md", "uq_ablation_v4.run (v4)"), E)
    fact("SigStatedPreFixHOAUC", "stated confidence: mean held-out within-cell error AUROC before the fixes", U0["UQAblHOStatedMeanAUC"], SRC0, E)
    fact("SigStatedPreFixHOWins", "held-out cells where stated confidence beat the margin, before the fixes",
         f"{U0['UQAblHOStatedWins']} of {U0['UQAblHOCells']}", SRC0, E)
    for k, tg in (("SwapStated", "Stated"), ("SwapLogit", "Logit"), ("SwapPTrue", "PTrue")):
        fact(f"SigHOSwap{tg}PreFix", f"held-out pooled: routing by {tg} minus the margin before the fixes", H0[f"HOAll{k}"],
             ("paper_archive/pre_v6_2026-10-10/numbers_heldout.tex", "newdata_analysis.main (v4)"), E, H0[f"HOAll{k}Lo"], H0[f"HOAll{k}Hi"])
    # invalid stated rows: unresolved answer with a stored confidence, test split, per dataset and model
    cnt, tot = defaultdict(int), 0
    for f in sorted(glob.glob(str(ROOT / "results/naacl/v3/uq/*.jsonl"))):
        ds, m = Path(f).name[:-6].rsplit("__", 1)
        sp = ROOT / f"data/splits/{ds}.json"
        if not sp.exists():
            continue
        test = set(json.loads(sp.read_text())["test"])
        for r in read(f):
            if r.get("verbalised") is not None and not r.get("verbalised_answer") and str(r["sample_id"]) in test:
                cnt[(ds, m)] += 1
                tot += 1
    fact("UQInvalidStatedRows", "test rows with an unresolved stated answer but a stored confidence (now missing, not 1 - c)",
         f"{tot:,}", ("results/naacl/v3/uq", "v6_numbers.prefix_facts"), "descriptive")
    for m in MODELS:
        fact(f"UQInvalidStated{MT[m]}", f"{MSHORT[m]}: test rows with an unresolved stated answer", f"{sum(v for (d, mm), v in cnt.items() if mm == m):,}",
             ("results/naacl/v3/uq", "v6_numbers.prefix_facts"), "descriptive")
    note("UQInvalidStatedByCell", "unresolved stated answers per dataset and model (test)",
         "; ".join(f"{d} {MSHORT.get(m, m)} {v}" for (d, m), v in sorted(cnt.items())) + ". These rows are treated as "
         "missing: a field without a valid stated confidence leaves the paid-signal population of its model, as any missing "
         "signal does, and the logistic gate's development fit drops them too.")


# ================================================================================================ per-field extras
def read(f):
    return [json.loads(l) for l in open(f) if l.strip()]


POOLS = {ds: {k.split("||", 1)[0] + "||" + k.split("||", 1)[1].lower(): v
              for k, v in json.load(open(ROOT / f"data/v4_pools/{ds}.json")).items()} for ds in DATASETS}


def rows_of(c):
    """The v4 rows behind cell c, in the cell's order (same filters as v4_analysis.load), with the gold list."""
    if hasattr(c, "_rows"):
        return c._rows
    man = json.loads((ROOT / f"data/splits/{c.ds}.json").read_text())
    test_ids = set(man["test"]) - set(man.get("old_dev_excluded") or [])
    out = []
    for r in read(ROOT / f"results/naacl/v4/test/given/{c.ds}__{c.model}/fields.jsonl"):
        sid = str(r["sample_id"])
        if sid not in test_ids:
            continue
        fld = r["key"].rsplit("||", 1)[1]
        tg = V.gold_of(c.ds, sid, fld, r["gts"])
        if tg is None:
            continue
        r["_gold"], r["_ck"] = tg, f"{sid}||{fld}"
        out.append(r)
    assert len(out) == c.n and all(str(r["sample_id"]) == s for r, s in zip(out, c.sid))
    assert all(float(target_correct(r["base"], r["_gold"])) == k for r, k in zip(out, c.keep))
    c._rows = out
    return out


def low_k(s, k):
    """Exact-expectation weights of the k smallest values of s (ties at the boundary share the remainder)."""
    s = np.asarray(s, float)
    n = len(s)
    if k <= 0 or n == 0:
        return np.zeros(n)
    k = min(k, n)
    t = np.sort(s)[min(max(int(math.ceil(k)) - 1, 0), n - 1)]
    w = (s < t).astype(float)
    tied = s == t
    if tied.any():
        w[tied] = min(max((k - w.sum()) / tied.sum(), 0.0), 1.0)
    return w


def per_model_weights(rs, score, share):
    """Weights routing the share[model] lowest-score fields within each model's fields (all its cells together)."""
    out = {}
    for m in MODELS:
        cs = [c for c in rs if c.model == m]
        if not cs:
            continue
        s = np.concatenate([score(c) for c in cs])
        w = low_k(s, share[m] * len(s) if isinstance(share, dict) else share * len(s))
        i = 0
        for c in cs:
            out[id(c)] = w[i:i + c.n]
            i += c.n
    return out


def masked(rs, fn):
    """v4_analysis.mask_cells, also masking the per-field arrays held in dicts (c.x1, c.x1chg) and tuples (c.x1cost)."""
    out = []
    for c in rs:
        mk = fn(c)
        if not mk.any():
            continue
        x = V.mask_cells([c], lambda _c: mk)[0]
        for k, v in c.__dict__.items():
            if isinstance(v, dict):
                setattr(x, k, {a: (tuple(np.broadcast_to(z, (c.n,))[mk] for z in b) if isinstance(b, tuple)
                                   else (b[mk] if isinstance(b, np.ndarray) and b.shape[:1] == (c.n,) else b))
                               for a, b in v.items()})
        out.append(x)
    return out


def call_tok_v4(c):
    """(calls, tokens) per field of the widen action and the routed action on every field, as v4_analysis.cost()."""
    if not hasattr(call_tok_v4, "T"):
        V4D = ROOT / "results/naacl/v4"
        call_tok_v4.T = dict(
            v4=V.call_means(glob.glob(str(V4D / "tokens/reflect__test__*.jsonl")),
                            {f"{p}:{a}" for p in ("reflect", "verify") for a in ("WT20", "WT50")}),
            v2=V.call_means(glob.glob(str(ROOT / "results/naacl/v2/tokens/reflect__given__*.jsonl")), {"reflect:B", "verify:BC"}),
            fresh={k: V.call_means(glob.glob(str(V4D / f"tokens/{k}__test__*.jsonl")), {f"rank:{k}"}) for k in ("fresh_WT20", "fresh_WT50")},
            first=V.call_means(glob.glob(str(ROOT / "results/naacl/v2/tokens/first__given__*.jsonl")), {"rank"}))
    T = call_tok_v4.T

    def tk(tab, st):
        v = tab.get((c.ds, c.model, st), tab.get((c.model, st), tab.get((st,))))
        return float(v) if v is not None else 0.0
    W = TH[c.model]["widen"]
    nt_r, nt_v = tk(T["v2"], "reflect:B"), tk(T["v2"], "verify:BC")
    w_r, w_v = tk(T["v4"], f"reflect:{W}"), tk(T["v4"], f"verify:{W}")
    isnt = c.act == "NT"
    return dict(
        widen=(1 + c.ver_w, w_r + c.ver_w * w_v),
        action=(1 + c.ver_act, np.where(isnt, nt_r + c.ver_act * nt_v, w_r + c.ver_act * w_v)),
        nt=(1 + c.ver_nt, nt_r + c.ver_nt * nt_v),
        fresh=(np.ones(c.n), np.full(c.n, tk(T["fresh"][f"fresh_{W}"], f"rank:fresh_{W}"))),
        first=(np.ones(c.n), np.full(c.n, tk(T["first"], "rank"))))


def _wr(c):
    """Tokens of the widen action's reflect call alone (no check call) for cell c."""
    T = call_tok_v4.T
    W = TH[c.model]["widen"]
    v = T["v4"].get((c.ds, c.model, f"reflect:{W}"), T["v4"].get((c.model, f"reflect:{W}"), T["v4"].get((f"reflect:{W}",))))
    return float(v or 0.0)


# ===================================================================================== X1 intervention x router
V2ARMS = [("old", "naive", "Doubt", "Doubt"), ("SR", "self_refine", "Self-Refine", "SelfRefine"), ("A", "A", "Neutral", "Neutral"),
          ("AC", "AC", "Neutral + check", "NeutralCheck"), ("B", "B", "Neutral + evidence", "NeutralEvid"),
          ("BC", "BC", "Reflect (neutral + evidence + check)", "Reflect")]
X1ROWS = []


def v2_arm_rows(ds, m):
    f = ROOT / f"results/naacl/v2/reflect/given/{ds}__{m}/fields.jsonl"
    if not f.exists() or not Path(str(f) + ".done").exists():
        return {}
    return {r["key"]: r for r in read(f)}


def v2_arm_costs(ds, m):
    """Per-field calls and tokens of each v2 arm from its token log: calls and tokens of the arm's stages divided by
    the number of fields the cell reflected (one reflect:A call per field)."""
    f = ROOT / f"results/naacl/v2/tokens/reflect__given__{ds}__{m}.jsonl"
    cnt, tok = defaultdict(float), defaultdict(float)
    if f.exists():
        for r in read(f):
            st = r.get("stage")
            cnt[st] += 1
            tok[st] += (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
    nf = cnt.get("reflect:A") or cnt.get("reflect:B") or 0
    stages = {"old": ["reflect:naive"], "SR": ["reflect:self_refine"], "A": ["reflect:A"], "AC": ["reflect:A", "verify:AC"],
              "B": ["reflect:B"], "BC": ["reflect:B", "verify:BC"]}
    return {a: ((sum(cnt[s] for s in ss) / nf, sum(tok[s] for s in ss) / nf) if nf else (np.nan, np.nan)) for a, ss in stages.items()}


def x1():
    E = "exploratory"
    cov = {}
    # attach every intervention's per-field outcome (NaN = missing), joined by full field identity
    for c in cells:
        rs = rows_of(c)
        v2 = v2_arm_rows(c.ds, c.model)
        costs = v2_arm_costs(c.ds, c.model)
        c.x1 = {}
        c.x1chg = {}
        c.x1cost = {}
        for arm, raw, lab, tg in V2ARMS:
            ok = np.full(c.n, np.nan)
            chg = np.zeros(c.n)
            nm = mism = 0
            for i, r in enumerate(rs):
                q = v2.get(r["key"])
                if q is None or raw not in q:
                    nm += 1
                    continue
                if (q.get("base") or "") != (r["base"] or "") or abs(float(q.get("margin", r["margin"])) - float(r["margin"])) > 1e-9:
                    mism += 1
                    continue
                ans = q[raw] or ""
                ok[i] = float(target_correct(ans, r["_gold"]))
                chg[i] = float(ans != (r["base"] or ""))
            c.x1[tg], c.x1chg[tg] = ok, chg
            c.x1cost[tg] = (np.full(c.n, costs[arm][0]), np.full(c.n, costs[arm][1]))
            cov.setdefault(tg, [0, 0, 0])
            cov[tg][0] += nm
            cov[tg][1] += mism
            cov[tg][2] += c.n
        # sanity: v4's NT is the v2 reflect arm
        nt_same = np.nansum(np.where(np.isnan(c.x1["Reflect"]), np.nan, c.x1["Reflect"] == c.nt))
        c.x1_ntagree = (nt_same, (~np.isnan(c.x1["Reflect"])).sum())
        ct = call_tok_v4(c)
        W = TH[c.model]["widen"]
        c.x1["WidenReflect"] = c.widen.copy()
        c.x1chg["WidenReflect"] = np.array([float((r[W] or "") != (r["base"] or "")) for r in rs])
        c.x1cost["WidenReflect"] = ct["widen"]
        prop = [(r.get(f"{W}_proposed") or r["base"] or "") for r in rs]
        c.x1["WidenNoCheck"] = np.array([float(target_correct(a, r["_gold"])) for a, r in zip(prop, rs)])
        c.x1chg["WidenNoCheck"] = np.array([float(a != (r["base"] or "")) for a, r in zip(prop, rs)])
        c.x1cost["WidenNoCheck"] = (np.ones(c.n), np.full(c.n, _wr(c)))
        c.x1["RoutedAction"] = c.action.copy()
        c.x1chg["RoutedAction"] = np.array([float((r[c.act[i]] or "") != (r["base"] or "")) for i, r in enumerate(rs)])
        c.x1cost["RoutedAction"] = ct["action"]
        fr = np.where(c.fresh_call > 0, c.fresh, np.nan)  # fresh ran only where the widened pool adds candidates
        c.x1["Fresh"] = fr
        fra = V.control_answers(f"fresh_{W}", c.ds, c.model)
        c.x1chg["Fresh"] = np.array([float((fra.get(r["_ck"], ("",))[0] or "") != (r["base"] or "")) if c.fresh_call[i] else 0.0
                                     for i, r in enumerate(rs)])
        c.x1cost["Fresh"] = ct["fresh"]
        for tg in ("WidenReflect", "WidenNoCheck", "RoutedAction", "Fresh"):
            cov.setdefault(tg, [0, 0, 0])
            cov[tg][0] += int(np.isnan(c.x1[tg]).sum())
            cov[tg][2] += c.n
    agree = sum(c.x1_ntagree[0] for c in cells) / sum(c.x1_ntagree[1] for c in cells)
    sub = masked(cells, lambda c: ~np.isnan(c.x1["Fresh"]))
    e, d = pooled(sub, lambda c: c.x1["WidenReflect"] - c.x1["Fresh"])
    fact_ci("IntWidenMinusFresh", "on the pairs where fresh mapping ran: widen + reflect minus fresh mapping, every field",
            e, d, SRC_V6, E, func="v6_numbers.x1")
    fact("IntFreshFirst", "first-pass accuracy on the pairs where fresh mapping ran, %",
         f"{100 * sum(c.keep.sum() for c in sub) / sum(c.n for c in sub):.2f}", SRC_V6, E, func="v6_numbers.x1")
    fact("IntNTMatchesReflect", "share of fields where v4's stored NT outcome equals the rejoined v2 reflect-arm outcome, %",
         f"{100 * agree:.2f}", SRC_V6, E, func="v6_numbers.x1")
    LAB = {tg: lab for _, _, lab, tg in V2ARMS}
    LAB.update(WidenReflect="Widen + reflect", WidenNoCheck="Widen + reflect without the check", RoutedAction="Reflect or widen + reflect (by top score)",
               Fresh="Fresh mapping over the widened pool")
    for tg in [t for *_, t in V2ARMS] + ["WidenReflect", "WidenNoCheck", "RoutedAction", "Fresh"]:
        rs = masked(cells, lambda c, tg=tg: ~np.isnan(c.x1[tg]))
        n_ = sum(c.n for c in rs)
        f0 = 100 * sum(c.keep.sum() for c in rs) / n_
        d_ = lambda c, tg=tg: c.x1[tg] - c.keep  # noqa: E731
        e_all, dr_all = pooled(rs, d_)
        w30 = per_model_weights(rs, lambda c: c.margin, 0.3)
        e_m30, _ = pooled(rs, lambda c: w30[id(c)] * d_(c))
        e_r30, _ = pooled(rs, lambda c: 0.3 * d_(c))
        e_c30, _ = pooled(rs, lambda c: low_k(c.margin, 0.3 * c.n) * d_(c))
        e_dm, dd = pooled(rs, lambda c: w30[id(c)] * d_(c) - 0.3 * d_(c))
        shr = {m: (sum(c.routed.sum() for c in rs if c.model == m) / max(1, sum(c.n for c in rs if c.model == m))) for m in MODELS}
        e_mr, dmr = pooled(rs, lambda c: c.routed * d_(c))
        e_rmr, _ = pooled(rs, lambda c: shr[c.model] * d_(c))
        e_dmr, ddmr = pooled(rs, lambda c: c.routed * d_(c) - shr[c.model] * d_(c))
        chg = int(sum(c.x1chg[tg].sum() for c in rs))
        rep = int(sum(((c.keep == 0) & (c.x1[tg] == 1)).sum() for c in rs))
        brk = int(sum(((c.keep == 1) & (c.x1[tg] == 0)).sum() for c in rs))
        calls = sum(float(np.sum(c.x1cost[tg][0])) for c in rs) / n_
        toks = sum(float(np.sum(c.x1cost[tg][1])) for c in rs) / n_
        mrcalls = sum(float(np.sum(c.routed * c.x1cost[tg][0])) for c in rs) / n_
        m30calls = sum(float(np.sum(w30[id(c)] * c.x1cost[tg][0])) for c in rs) / n_
        name = LAB[tg]
        fact(f"Int{tg}Pairs", f"{name}: field-model pairs with a measured outcome", fmt_int(n_), SRC_V6, E, func="v6_numbers.x1")
        fact(f"Int{tg}Missing", f"{name}: pairs without a stored outcome (not filled)", fmt_int(cov[tg][0]), SRC_V6, E, func="v6_numbers.x1")
        if cov[tg][1]:
            fact(f"Int{tg}Mismatch", f"{name}: pairs dropped because the stored first pass differs from v4's", fmt_int(cov[tg][1]), SRC_V6, E, func="v6_numbers.x1")
        fact_ci(f"Int{tg}Every", f"{name}: change when applied to every field", e_all, dr_all, SRC_V6, E, func="v6_numbers.x1")
        fact(f"Int{tg}MarginThirty", f"{name}: change at the 30% smallest-margin fields of each model", sg(e_m30), SRC_V6, E, func="v6_numbers.x1")
        fact(f"Int{tg}MarginThirtyCell", f"{name}: change at the 30% smallest-margin fields of each model-dataset cell "
             "(the per-cell estimand of tab_ablation.tex, for reproduction)", sg(e_c30), SRC_V6, E, func="v6_numbers.x1")
        fact(f"Int{tg}RandThirty", f"{name}: change at a random 30% (exact expectation, same population)", sg(e_r30), SRC_V6, E, func="v6_numbers.x1")
        lo, hi = fact_ci(f"Int{tg}MarginVsRand", f"{name}: margin at 30% minus random at 30%", e_dm, dd, SRC_V6, E, func="v6_numbers.x1")
        fact_ci(f"Int{tg}Router", f"{name}: change on the fields Map-Reflect's router routes", e_mr, dmr, SRC_V6, E, func="v6_numbers.x1")
        fact(f"Int{tg}RouterRand", f"{name}: change at a random share equal to Map-Reflect's per model", sg(e_rmr), SRC_V6, E, func="v6_numbers.x1")
        lo2, hi2 = fact_ci(f"Int{tg}RouterVsRand", f"{name}: Map-Reflect's router minus random at its share", e_dmr, ddmr, SRC_V6, E, func="v6_numbers.x1")
        fact(f"Int{tg}Changed", f"{name}: answers changed with every field", fmt_int(chg), SRC_V6, E, func="v6_numbers.x1")
        fact(f"Int{tg}Repaired", f"{name}: answers repaired with every field", fmt_int(rep), SRC_V6, E, func="v6_numbers.x1")
        fact(f"Int{tg}Broken", f"{name}: answers broken with every field", fmt_int(brk), SRC_V6, E, func="v6_numbers.x1")
        fact(f"Int{tg}Calls", f"{name}: calls per field when applied to every field (token logs)", f"{calls:.2f}", SRC_V6, E, func="v6_numbers.x1")
        fact(f"Int{tg}Tokens", f"{name}: tokens per field when applied to every field (token logs)", fmt_int(toks), SRC_V6, E, func="v6_numbers.x1")
        fact(f"Int{tg}RouterCalls", f"{name}: calls per field under Map-Reflect's router", f"{mrcalls:.2f}", SRC_V6, E, func="v6_numbers.x1")
        if e_all > 0:
            fact(f"Int{tg}RouterShareOfEvery", f"{name}: Map-Reflect's router gain as a share of the every-field gain, %",
                 f"{100 * e_mr / e_all:.0f}", SRC_V6, E, func="v6_numbers.x1")
        X1ROWS.append(dict(tg=tg, name=name, n=n_, f0=f0, every=e_all, m30=e_m30, r30=e_r30, dm=(e_dm, lo, hi), mr=e_mr,
                           rmr=e_rmr, dmr=(e_dmr, lo2, hi2), chg=chg, rep=rep, brk=brk, calls=calls, toks=toks,
                           mrcalls=mrcalls, m30calls=m30calls))
    note("IntDefinitions", "X1 definitions",
         "Outcomes rejoined by full field identity (the v4 key) to results/naacl/v2/reflect/given (doubt = 'naive', "
         "Self-Refine, neutral A, neutral+check AC, neutral+evidence B, reflect BC) and to the v4 rows (widen action, routed "
         "action) and fresh_<W>_runs; a row is dropped, not filled, when the stored first pass differs. Fresh mapping is "
         "measured only where the widened pool adds candidates (its runner skipped the rest), so its row covers those pairs. "
         "Scorer: target_correct, ITOpsGraph v2.1. The 30% is chosen within each model over that population, ties sharing the "
         "boundary; random is the exact expectation on the same population. Calls and tokens: v2 arm token logs "
         "(results/naacl/v2/tokens) per reflected field, check calls included; v4 actions as in v4_analysis.cost().")


def check_facts():
    """F-Check*: the pairwise check on and off, from the X1 rows (every field)."""
    E = "exploratory"
    R = {r["tg"]: r for r in X1ROWS}
    for tg, on, off, what in (("Widen", "WidenReflect", "WidenNoCheck", "widen + reflect"),
                              ("Evidence", "Reflect", "NeutralEvid", "neutral + evidence over the shown candidates"),
                              ("Neutral", "NeutralCheck", "Neutral", "neutral over the shown candidates")):
        for suf, r in (("With", R[on]), ("Without", R[off])):
            fact(f"Check{tg}{suf}Gain", f"{what}, check {'on' if suf == 'With' else 'off'}: change on every field", sg(r["every"]), SRC_V6, E, func="v6_numbers.x1")
            fact(f"Check{tg}{suf}Broken", f"{what}, check {'on' if suf == 'With' else 'off'}: answers broken on every field", fmt_int(r["brk"]), SRC_V6, E, func="v6_numbers.x1")
            fact(f"Check{tg}{suf}Repaired", f"{what}, check {'on' if suf == 'With' else 'off'}: answers repaired on every field", fmt_int(r["rep"]), SRC_V6, E, func="v6_numbers.x1")


# ======================================================================== X6 structure-conditioned repairability
X6TAB = []


def x6():
    E = "exploratory"
    for c in cells:
        rs = rows_of(c)
        K = int(TH[c.model]["widen"].lstrip("WTX"))
        shown, added, flag_agree = np.zeros(c.n), np.zeros(c.n), 0
        for i, r in enumerate(rs):
            p = POOLS[c.ds][r["_ck"]]
            sh = any(target_correct(x, r["_gold"]) for x in p["shown"])
            ad = (not sh) and any(target_correct(x, r["_gold"]) for x in p["type_rest"][:K])
            shown[i], added[i] = sh, ad
            flag_agree += int(bool(r.get(f"{TH[c.model]['widen']}_gold_in_pool")) == bool(sh or ad))
        c.gold_loc = np.where(shown > 0, 0, np.where(added > 0, 1, 2))
        c._flag_agree = flag_agree
    agree = sum(c._flag_agree for c in cells) / N
    fact("RepairPoolFlagAgree", "share of fields where the recomputed 'gold in shown + K' equals the stored X_gold_in_pool flag, %",
         f"{100 * agree:.1f}", SRC_V6, E, func="v6_numbers.x6")
    STR = {"Declined": lambda c: ~c.valid, "Wide": lambda c: c.valid & (c.wide > 0), "Shown": lambda c: c.valid & (c.wide == 0)}
    GRP = {"InShown": 0, "Added": 1, "Absent": 2}
    errs_routed = sum(((c.keep == 0) & (c.routed > 0)).sum() for c in cells)
    for gname, g in GRP.items():
        k = sum(((c.keep == 0) & (c.routed > 0) & (c.gold_loc == g)).sum() for c in cells)
        rep = sum(((c.keep == 0) & (c.routed > 0) & (c.gold_loc == g) & (c.method == 1)).sum() for c in cells)
        fact(f"Repair{gname}Share", f"routed first-pass errors whose gold is {gname} (of all routed errors), %",
             f"{100 * k / errs_routed:.1f}", SRC_V6, E, func="v6_numbers.x6")
        fact(f"Repair{gname}Rate", f"routed first-pass errors with gold {gname}: share repaired, %",
             f"{100 * rep / max(k, 1):.1f}", SRC_V6, E, func="v6_numbers.x6")
    fact("RepairErrRouted", "routed first-pass errors, all fields (declined included)", fmt_int(errs_routed), SRC_V6, E, func="v6_numbers.x6")
    for sname, sf in STR.items():
        for gname, g in GRP.items():
            sub = masked(cells, lambda c, sf=sf, g=g: sf(c) & (c.gold_loc == g))
            if not sub:
                continue
            nf = sum(c.n for c in sub)
            err = int(sum((c.keep == 0).sum() for c in sub))
            rerr = int(sum(((c.keep == 0) & (c.routed > 0)).sum() for c in sub))
            rep = int(sum(((c.keep == 0) & (c.method == 1)).sum() for c in sub))
            brk = int(sum(((c.keep == 1) & (c.method == 0)).sum() for c in sub))
            rt = 100 * sum(c.routed.sum() for c in sub) / nf
            e_m, _ = pooled(sub, POL["Method"])
            e_d, dd = pooled(sub, lambda c: POL["Method"](c) - c.routed.mean() * (c.action - c.keep))
            lo, hi = GS.interval(dd)
            tg = f"Repair{sname}{gname}"
            fact(f"{tg}Fields", f"{sname.lower()} stratum, gold {gname}: field-model pairs", fmt_int(nf), SRC_V6, E, func="v6_numbers.x6")
            fact(f"{tg}Errors", f"{sname.lower()} stratum, gold {gname}: first-pass errors (routed)", f"{err:,} ({rerr:,})", SRC_V6, E, func="v6_numbers.x6")
            fact(f"{tg}Repairs", f"{sname.lower()} stratum, gold {gname}: repairs / breaks under v4", f"{rep:,} / {brk:,}", SRC_V6, E, func="v6_numbers.x6")
            fact(f"{tg}VsRand", f"{sname.lower()} stratum, gold {gname}: Map-Reflect minus random at its per-cell share there",
                 sg(e_d), SRC_V6, E, sg(lo), sg(hi), func="v6_numbers.x6")
            X6TAB.append((sname, gname, nf, err, rerr, rep, brk, rt, e_m, e_d, lo, hi))
    # per model: routed errors and repairs per gold location
    for m in MODELS:
        for gname, g in GRP.items():
            rs = [c for c in cells if c.model == m]
            k = sum(((c.keep == 0) & (c.routed > 0) & (c.gold_loc == g)).sum() for c in rs)
            rep = sum(((c.keep == 0) & (c.routed > 0) & (c.gold_loc == g) & (c.method == 1)).sum() for c in rs)
            fact(f"Repair{MT[m]}{gname}", f"{MSHORT[m]}: routed errors with gold {gname} / repaired", f"{k:,} / {rep:,}", SRC_V6, E, func="v6_numbers.x6")
    # does the margin read coverage? answered first-pass errors, label = gold not among the 20 shown
    for tg, sf in (("", lambda c: c.valid), ("Wide", lambda c: c.valid & (c.wide > 0))):
        sub = masked(cells, lambda c, sf=sf: sf(c) & (c.keep == 0))
        score = np.concatenate([-c.margin for c in sub])
        lab = np.concatenate([(c.gold_loc != 0).astype(float) for c in sub])
        auc = wauc(score, lab, np.ones(len(lab)))
        W = boot_weights(sub)
        draws = np.array([wauc(score, lab, W[b]) for b in range(W.shape[0])])
        lo, hi = GS.interval(draws)
        fact(f"MarginCoverageAUC{tg}", "AUROC of a low margin for 'gold not among the shown candidates', answered first-pass errors"
             + (", wide-type fields only" if tg else ", pooled over models (raw margins)"), f"{auc:.3f}", SRC_V6, E, f"{lo:.3f}", f"{hi:.3f}",
             func="v6_numbers.x6")
        fact(f"MarginCoverageN{tg}", "fields in that AUROC (absent-from-shown share, %)",
             f"{len(lab):,} ({100 * lab.mean():.1f}%)", SRC_V6, E, func="v6_numbers.x6")
    note("RepairDefinitions", "X6 definitions",
         "Gold location per field: InShown = a gold target (target_correct, ITOpsGraph v2.1) is among the 20 candidates the "
         "first pass showed (data/v4_pools 'shown'); Added = not shown but among the model's widened pool (shown + the type's "
         "next K fields in hybrid (dense + BM25) order, K frozen per model, as reflect_v4.pool_of); Absent = in neither. Strata as v4_analysis: declined, "
         "wide (answered, named type has more than 20 fields), shown (answered, type fully shown). Random is matched to "
         "Map-Reflect's share inside each stratum-group population of each model-dataset cell. Note that F-ErrInShown (uq "
         "population) counts gold among the first pass's ranked candidates, a narrower set than the 20 shown.")


def boot_weights(sub):
    """B x fields matrix of split-group bootstrap multiplicities (GB's shared draws) for the fields of sub, in order."""
    cols = []
    for c in sub:
        ix = GB.index[c.ds]
        ci = np.array([ix[GB.unit(c.ds, x)] for x in c.sid])
        cols.append(GB.counts[c.ds][:, ci])
    return np.concatenate(cols, axis=1)


def wauc(score, lab, w):
    """Weighted AUROC (ties count one half) of score for lab == 1."""
    o = np.argsort(score, kind="mergesort")
    s, y, w = score[o], lab[o], w[o]
    wp, wn = w * y, w * (1 - y)
    if wp.sum() == 0 or wn.sum() == 0:
        return np.nan
    _, start = np.unique(s, return_index=True)
    grp_p = np.add.reduceat(wp, start)
    grp_n = np.add.reduceat(wn, start)
    below_n = np.concatenate([[0], np.cumsum(grp_n)[:-1]])
    return float((grp_p * (below_n + 0.5 * grp_n)).sum() / (wp.sum() * wn.sum()))


# ============================================================================ X4 discrimination and calibration
X4 = {}


def x4():
    E = "exploratory"
    for m in MODELS:
        rs = masked([c for c in cells if c.model == m], lambda c: c.valid)
        s1 = np.concatenate([c.s1 for c in rs])
        mg = np.concatenate([c.margin for c in rs])
        ok = np.concatenate([c.keep for c in rs])
        at1, at9 = 100 * (s1 >= 1.0).mean(), 100 * (s1 >= 0.9).mean()
        edges = np.linspace(0, 1, 11)
        b = np.minimum((s1 * 10).astype(int), 9)
        b = np.where(s1 >= 1.0, 9, b)
        bins = [(int((b == k).sum()), float(s1[b == k].mean()) if (b == k).any() else np.nan,
                 float(ok[b == k].mean()) if (b == k).any() else np.nan) for k in range(10)]
        ece = sum(nb / len(s1) * abs(cf - ac) for nb, cf, ac in bins if nb)
        brier = float(np.mean((s1 - ok) ** 2))
        # margin deciles preserving ties: a tied value goes wholly to the decile of its first position
        o = np.sort(mg)
        first_pos = np.searchsorted(o, mg, side="left")
        dec = np.minimum((10 * first_pos) // len(mg), 9)
        decs = [(int((dec == k).sum()), float(mg[dec == k].min()) if (dec == k).any() else np.nan,
                 float(mg[dec == k].max()) if (dec == k).any() else np.nan,
                 float(ok[dec == k].mean()) if (dec == k).any() else np.nan) for k in range(10)]
        auc = wauc(-mg, 1 - ok, np.ones(len(ok)))
        W = boot_weights(rs)
        draws = np.array([wauc(-mg, 1 - ok, W[k]) for k in range(W.shape[0])])
        lo, hi = GS.interval(draws)
        X4[m] = dict(at1=at1, at9=at9, bins=bins, ece=ece, brier=brier, decs=decs, auc=auc, lo=lo, hi=hi, n=len(s1),
                     acc=100 * ok.mean(), conf=100 * s1.mean())
        t = MT[m]
        fact(f"CalTopMax{t}", f"{MSHORT[m]}: answered fields whose top score is 1.0, %", f"{at1:.1f}", SRC_V6, E, func="v6_numbers.x4")
        fact(f"CalTopNine{t}", f"{MSHORT[m]}: answered fields whose top score is at least 0.9, %", f"{at9:.1f}", SRC_V6, E, func="v6_numbers.x4")
        fact(f"CalECE{t}", f"{MSHORT[m]}: ECE of the top score as a probability of correctness (10 equal-width bins)", f"{ece:.3f}", SRC_V6, E, func="v6_numbers.x4")
        fact(f"CalBrier{t}", f"{MSHORT[m]}: Brier score of the top score", f"{brier:.3f}", SRC_V6, E, func="v6_numbers.x4")
        fact(f"CalMeanConf{t}", f"{MSHORT[m]}: mean top score vs first-pass accuracy on answered fields, %", f"{100 * s1.mean():.1f} vs {100 * ok.mean():.1f}", SRC_V6, E, func="v6_numbers.x4")
        fact(f"CalMarginAUC{t}", f"{MSHORT[m]}: margin's error AUROC on all answered test fields", f"{auc:.3f}", SRC_V6, E, f"{lo:.3f}", f"{hi:.3f}", func="v6_numbers.x4")
        fact(f"CalDecileLow{t}", f"{MSHORT[m]}: first-pass accuracy in the lowest margin decile, %", f"{100 * decs[0][3]:.1f}", SRC_V6, E, func="v6_numbers.x4")
        fact(f"CalDecileHigh{t}", f"{MSHORT[m]}: first-pass accuracy in the highest non-empty margin decile, %",
             f"{100 * [d for d in decs if d[0]][-1][3]:.1f}", SRC_V6, E, func="v6_numbers.x4")
    allrs = masked(cells, lambda c: c.valid)
    s1 = np.concatenate([c.s1 for c in allrs])
    fact("CalTopMaxAll", "all models: answered fields whose top score is 1.0, %", f"{100 * (s1 >= 1).mean():.1f}", SRC_V6, E, func="v6_numbers.x4")
    note("CalDefinitions", "X4 definitions", "Answered (valid) first passes, all test fields per model. Reliability: 10 "
         "equal-width top-score bins, 1.0 in the last; ECE = sum over bins of (bin share) x |accuracy - mean top score|; Brier "
         "= mean (top score - correct)^2. Margin deciles keep ties together (a tied value goes to the decile of its first "
         "sorted position), so deciles can be empty or uneven. AUROC: errors positive, low margin = error, split-group "
         "bootstrap interval. These margin AUROCs are on every answered field, not the uq population of Table 3.")


# ======================================================================== X2 earlier trigger families
X2ROWS = []


def first_pass_preds(ds, m):
    """key -> (stored detected_errors, the runner-order initial ranking) from the v2 first pass (results/naacl/v2/first)."""
    d = R2.load_slot(str(ROOT / f"results/naacl/v2/first/given/{ds}__{m}"))
    man = json.loads((ROOT / f"data/splits/{ds}.json").read_text())
    ids = set(man["test"])
    res = {str(r.get("sample_id")): r for r in d["results"]}
    out = {}
    for rec in R2.records(d, ids):
        p = (res[rec["sample_id"]].get("predictions") or {}).get(rec["field"]) or {}
        out[rec["key"]] = (p.get("detected_errors"), p.get("initial_ranking") or [], rec["base"] or "")
    return out


def binary_router(cs_by_model, trig):
    """Exact-expectation weights at Map-Reflect's per-model share: triggered fields first (all of them if they fit, else
    an equal fraction of each), the remaining budget spread evenly over the untriggered fields."""
    out = {}
    for m, cs in cs_by_model.items():
        n = sum(c.n for c in cs)
        k = SHARE_M[m] * n
        T = sum(trig(c).sum() for c in cs)
        for c in cs:
            t = trig(c).astype(float)
            if T >= k:
                out[id(c)] = t * (k / T)
            else:
                out[id(c)] = t + (1 - t) * ((k - T) / max(n - T, 1))
    return out


def x2():
    E = "exploratory"
    miss = mism = 0
    for c in cells:
        fp = first_pass_preds(c.ds, c.model)
        err, nbr = np.zeros(c.n), np.zeros(c.n)
        c.x2_ok = np.ones(c.n, bool)
        for i, r in enumerate(rows_of(c)):
            q = fp.get(r["key"])
            if q is None or q[0] is None:
                miss += 1
                c.x2_ok[i] = False
                continue
            if q[2] != (r["base"] or ""):
                mism += 1
                c.x2_ok[i] = False
                continue
            err[i] = float(len(q[0]) > 0)
            ir = [x for x in q[1] if isinstance(x, dict)]
            if len(ir) >= 2:
                def num(x, d):
                    try:
                        return float(x.get("score", d))
                    except (TypeError, ValueError):
                        return d
                nbr[i] = float(num(ir[0], 1) - num(ir[1], 0) < 0.2)
        c.trig_err, c.trig_nbr = err, nbr
    fact("RouterErrMissing", "pairs without a stored first-pass trigger record or with a different first pass", f"{miss + mism:,}",
         SRC_V6, E, func="v6_numbers.x2")
    ok_cells = masked(cells, lambda c: c.x2_ok)
    by_m = {m: [c for c in ok_cells if c.model == m] for m in MODELS}
    nn = sum(c.n for c in ok_cells)
    f0 = 100 * sum(c.keep.sum() for c in ok_cells) / nn
    e_main, _ = pooled(ok_cells, POL["Method"])
    for tg, name, trig in (("Err", "error-heuristic trigger (stored detected_errors non-empty)", lambda c: c.trig_err > 0),
                           ("Nbr", "'neighbourhood' trigger as coded (top-two gap of the runner-order ranking < 0.2)", lambda c: c.trig_nbr > 0)):
        own = 100 * sum(trig(c).sum() for c in ok_cells) / nn
        w = binary_router(by_m, trig)
        e, d = pooled(ok_cells, lambda c: w[id(c)] * (c.action - c.keep))
        e_d, dd = pooled(ok_cells, lambda c: POL["Method"](c) - w[id(c)] * (c.action - c.keep))
        e_own, d_own = pooled(ok_cells, lambda c: trig(c) * (c.action - c.keep))
        calls = sum(float(np.sum(w[id(c)] * (1 + c.ver_act))) for c in ok_cells) / nn
        ocalls = sum(float(np.sum(trig(c) * (1 + c.ver_act))) for c in ok_cells) / nn
        fact(f"Router{tg}Own", f"{name}: share of fields it fires on, %", f"{own:.1f}", SRC_V6, E, func="v6_numbers.x2")
        fact_ci(f"Router{tg}OwnGain", f"{name}: change when it routes exactly the fields it fires on (v4 action)", e_own, d_own, SRC_V6, E, func="v6_numbers.x2")
        fact(f"Router{tg}OwnCalls", f"{name}: calls per field at its own share", f"{ocalls:.2f}", SRC_V6, E, func="v6_numbers.x2")
        glo, ghi = fact_ci(f"Router{tg}Gain", f"{name}: change at Map-Reflect's per-model share (v4 action)", e, d, SRC_V6, E, func="v6_numbers.x2")
        toks = sum(float(np.sum(w[id(c)] * call_tok_v4(c)["action"][1])) for c in ok_cells) / nn
        fact(f"Router{tg}Calls", f"{name}: second-pass calls per field at Map-Reflect's per-model share", f"{calls:.2f}", SRC_V6, E, func="v6_numbers.x2")
        fact(f"Router{tg}Tokens", f"{name}: second-pass tokens per field at Map-Reflect's per-model share (v4 action token logs)",
             fmt_int(toks), SRC_V6, E, func="v6_numbers.x2")
        fact_ci(f"MainVsRouter{tg}", f"Map-Reflect minus the {name} at the same share", e_d, dd, SRC_V6, E, func="v6_numbers.x2")
        X2ROWS.append(dict(tg=tg, name=name, own=own, e_own=e_own, ocalls=ocalls, e=e, lo=glo, hi=ghi, calls=calls,
                           toks=toks, e_d=e_d, dci=GS.interval(dd)))
    fact("RouterPopFirst", "first-pass accuracy on the X2 population, %", f"{f0:.2f}", SRC_V6, E, func="v6_numbers.x2")
    fact("RouterPopMain", "Map-Reflect change on the X2 population", sg(e_main), SRC_V6, E, func="v6_numbers.x2")
    note("RouterErrDefinition", "X2 error trigger (map_reflect/map_reflect_runner.py, detect_mapping_errors and the trigger "
         "loop)", "A field fires when the runner's detect_mapping_errors returned any message. The checks: the ranking cannot "
         "be parsed (declined); the top prediction is prefixed by the source product; a source field ending in _id names "
         "another table the prediction lacks; high name overlap with a reason citing name similarity; type keywords "
         "(string, int, date, ...) disagree; top score below 0.6; top two scores within 0.1. The runner stored this list "
         "with every v2 first pass (predictions[field].detected_errors), so the trigger is read, not re-derived. It is "
         "binary: at Map-Reflect's per-model share, fired fields are routed first (an equal fraction of each when they "
         "exceed the budget) and any remaining budget is spread evenly over the others (exact expectation).")
    note("RouterNbrDefinition", "X2 neighbourhood trigger", "The runner's 'neighborhood' trigger is not sibling "
         "disagreement: it fires when neighbor_k > 0 and the top two scores of the runner-order initial ranking differ by "
         "less than 0.2 (a TODO placeholder in the code). It is therefore a fixed-threshold margin, reported only for "
         "completeness. Sibling disagreement on the target type or table is vacuous here, because every candidate a field "
         "sees belongs to the one named target type, so all siblings agree on type and table by construction; that "
         "router is dropped.")


# ================================================================================================= X3 escalation
X3ROWS = []


def x3():
    E = "exploratory"
    dev_acc = {}
    for ds in DATASETS:
        man = json.loads((ROOT / f"data/splits/{ds}.json").read_text())
        dev_ids = set(man["dev"])
        for m in MODELS:
            ok = []
            for r in read(ROOT / f"results/naacl/v4/dev/given/{ds}__{m}/fields.jsonl"):
                sid = str(r["sample_id"])
                if sid not in dev_ids:
                    continue
                tg = V.gold_of(ds, sid, r["key"].rsplit("||", 1)[1], r["gts"])
                if tg is None:
                    continue
                ok.append(float(target_correct(r["base"], tg)))
            dev_acc[(ds, m)] = (sum(ok), len(ok))
    pooled_dev = {m: sum(dev_acc[(ds, m)][0] for ds in DATASETS) / sum(dev_acc[(ds, m)][1] for ds in DATASETS) for m in MODELS}
    dest = {}
    for ds in DATASETS:
        for m in MODELS:
            cand = [x for x in MODELS if x != m]
            dest[(ds, m)] = max(cand, key=lambda x: (dev_acc[(ds, x)][0] / max(dev_acc[(ds, x)][1], 1), pooled_dev[x]))
    rowmap = {(c.ds, c.model): {r["key"]: r for r in rows_of(c)} for c in cells}
    for c in cells:
        other = rowmap[(c.ds, dest[(c.ds, c.model)])]
        rs = rows_of(c)
        c.esc = np.array([float(target_correct(other[r["key"]]["base"], r["_gold"])) if r["key"] in other else np.nan for r in rs])
    assert not any(np.isnan(c.esc).any() for c in cells)
    d_ = lambda c: c.esc - c.keep  # noqa: E731
    e_all, dr = pooled(cells, d_)
    w30 = per_model_weights(cells, lambda c: c.margin, 0.3)
    e_m30, _ = pooled(cells, lambda c: w30[id(c)] * d_(c))
    e_r30, _ = pooled(cells, lambda c: 0.3 * d_(c))
    e_dm, dd = pooled(cells, lambda c: w30[id(c)] * d_(c) - 0.3 * d_(c))
    e_mr, dmr = pooled(cells, lambda c: c.routed * d_(c))
    e_dmr, ddmr = pooled(cells, lambda c: c.routed * d_(c) - SHARE_M[c.model] * d_(c))
    e_rmr, _ = pooled(cells, lambda c: SHARE_M[c.model] * d_(c))
    rep = int(sum(((c.keep == 0) & (c.esc == 1)).sum() for c in cells))
    brk = int(sum(((c.keep == 1) & (c.esc == 0)).sum() for c in cells))
    fact("EscKeysAligned", "test field keys are identical across the four models in every core dataset (and on dev)", "yes", SRC_V6, E, func="v6_numbers.x3")
    fact_ci("IntEscalateEvery", "escalation (another model's first answer, chosen on dev) on every field", e_all, dr, SRC_V6, E, func="v6_numbers.x3")
    fact("IntEscalateMarginThirty", "escalation at the 30% smallest-margin fields of each model", sg(e_m30), SRC_V6, E, func="v6_numbers.x3")
    fact("IntEscalateRandThirty", "escalation at a random 30%", sg(e_r30), SRC_V6, E, func="v6_numbers.x3")
    lo, hi = fact_ci("IntEscalateMarginVsRand", "escalation: margin at 30% minus random at 30%", e_dm, dd, SRC_V6, E, func="v6_numbers.x3")
    fact_ci("IntEscalateRouter", "escalation on the fields Map-Reflect routes", e_mr, dmr, SRC_V6, E, func="v6_numbers.x3")
    lo2, hi2 = fact_ci("IntEscalateRouterVsRand", "escalation: Map-Reflect's router minus random at its share", e_dmr, ddmr, SRC_V6, E, func="v6_numbers.x3")
    fact("IntEscalateRepaired", "escalation on every field: repaired", fmt_int(rep), SRC_V6, E, func="v6_numbers.x3")
    fact("IntEscalateBroken", "escalation on every field: broken", fmt_int(brk), SRC_V6, E, func="v6_numbers.x3")
    cnt = defaultdict(int)
    for (ds, m), x in dest.items():
        cnt[x] += 1
    fact("IntEscalateDest", "destination model chosen on dev, number of (dataset, source model) cells",
         ", ".join(f"{MSHORT[k]} {v}" for k, v in sorted(cnt.items(), key=lambda t: -t[1])), SRC_V6, E, func="v6_numbers.x3")
    chg = int(sum(np.array([float(rowmap[(c.ds, dest[(c.ds, c.model)])][r["key"]]["base"] != r["base"]) for r in rows_of(c)]).sum() for c in cells))
    calls = 1.0
    toks = sum(float(np.sum(call_tok_v4(c)["first"][1])) for c in cells) / N
    fact("IntEscalateChanged", "escalation on every field: answers changed", fmt_int(chg), SRC_V6, E, func="v6_numbers.x3")
    fact("IntEscalateCalls", "escalation on every field: calls per field (one destination first pass)", f"{calls:.2f}", SRC_V6, E, func="v6_numbers.x3")
    fact("IntEscalateTokens", "escalation on every field: tokens per field (destination first-pass tokens, v2 first-pass logs)", fmt_int(toks), SRC_V6, E, func="v6_numbers.x3")
    X3ROWS.append(dict(tg="Escalate", name="Escalate (another model's first answer)", n=N, f0=FIRST, every=e_all, m30=e_m30,
                       r30=e_r30, dm=(e_dm, lo, hi), mr=e_mr, rmr=e_rmr, dmr=(e_dmr, lo2, hi2), chg=chg, rep=rep, brk=brk,
                       calls=calls, toks=toks, mrcalls=sum(c.routed.sum() for c in cells) / N, m30calls=0.3))
    note("EscDefinition", "X3 escalation", "For each dataset and source model, the destination is the other model with the "
         "highest development first-pass accuracy on that dataset (results/naacl/v4/dev/given, full-identity scorer, "
         "ITOpsGraph v2.1), ties to the higher pooled development accuracy; the source model is excluded. A routed field "
         "takes the destination model's stored test first answer for the same key. Cost: one first-pass call of the "
         "destination model per routed field (tokens from the v2 first-pass logs). Development sets are small for "
         "OMOP-Synthea (26 fields) and WikiData (5), so those choices are noisy.")


# ============================================================== R2: calls and tokens of union, oracle (phase 2)
R2COST = {}


def r2_costs():
    """Calls and tokens per field for struct-or-margin (union) and the oracle, from the same token logs as
    v4_analysis.cost(): a routed field costs its action's reflect call plus a check call when it proposes a change.
    The oracle's cost is that of the fields it routes (its weights). Also the union's interval."""
    E = "exploratory"
    st = V.STRUCT
    cm = ct = 0.0
    for c in cells:
        a = call_tok_v4(c)["action"]
        cm += float(np.sum(c.routed * a[1]))
    assert abs(cm / N - ROWS["Method"][7]) < 0.5, (cm / N, ROWS["Method"][7])  # same costing as Table 1's row
    uc = ut = oc = ot = 0.0
    for c in cells:
        T = call_tok_v4(c)
        a, w = T["action"], T["widen"]
        rt = c.routed > 0
        s_ = st(c) & ~rt
        uc += float(np.sum(np.where(rt, a[0], np.where(s_, w[0], 0.0))))
        ut += float(np.sum(np.where(rt, a[1], np.where(s_, w[1], 0.0))))
        fix = np.where(c.keep == 0, c.action, 0.0)
        tot = sum(np.where(x.keep == 0, x.action, 0.0).sum() for x in cells if x.model == c.model)
        k = SHARE_M[c.model] * sum(x.n for x in cells if x.model == c.model)
        ow = fix * min(1.0, k / tot) if tot else fix
        assert np.allclose(ow, POL["Oracle"](c))  # the oracle's gain is its routing weight (it routes only repairs)
        oc += float(np.sum(ow * a[0]))
        ot += float(np.sum(ow * a[1]))
    assert abs(uc / N - float(MV4["VFourUnionRouterCalls"])) < 0.006, (uc / N, MV4["VFourUnionRouterCalls"])
    uf = lambda c: np.where(c.routed > 0, c.action, np.where(st(c), c.widen, c.keep)) - c.keep  # noqa: E731
    e, d = pooled(cells, uf)
    lo, hi = GS.interval(d)
    assert f"{e:+.2f}" == MV4["VFourUnionRouter"], (e, MV4["VFourUnionRouter"])
    for f in FACTS:
        if f["id"] == "UnionGain":
            f["lo"], f["hi"] = sg(lo), sg(hi)
    fact("UnionTokens", "union routing: second-pass tokens per field", fmt_int(ut / N), SRC_V6, E, func="v6_numbers.r2_costs")
    fact("OracleCalls", "oracle: second-pass calls per field (the calls of the fields it routes)", f"{oc / N:.2f}", SRC_V6, E,
         func="v6_numbers.r2_costs")
    fact("OracleTokens", "oracle: second-pass tokens per field (the fields it routes)", fmt_int(ot / N), SRC_V6, E,
         func="v6_numbers.r2_costs")
    R2COST.update(union=(uc / N, ut / N, e, lo, hi), oracle=(oc / N, ot / N))


# ======================================================================================== figures
def fig2():
    import matplotlib.pyplot as plt
    import v6_phase2 as P2_
    if P2_.V5:  # method v5: every curve and point with the v5 second pass
        core = P2_.V5["core"]
        mc = ROOT / "paper/figures/data6/v5_matched_curves.csv"
        if mc.exists():  # the matched-budget curves of the v5 cost check (v5_matched_facts.json), test split
            import csv as _csv
            R = [r for r in _csv.DictReader(open(mc)) if r["split"] == "test"]
            cur = lambda k: sorted((float(r["calls_per_field"]), float(r["accuracy"])) for r in R if r["router"] == k)  # noqa: E731
            one = lambda k: next((float(r["calls_per_field"]), float(r["accuracy"])) for r in R if r["router"] == k)  # noqa: E731
            rnd5, msw5, bsw5 = cur("random"), cur("margin"), cur("benefit")
            pts5 = [("No reflection", 0.0, rnd5[0][1]), ("Schema size", *one("struct")), ("Margin", *one("margin_own")),
                    ("Map-Reflect", *one("benefit_own"))]
            acc, cl, _ = P2_.v5_policy(core, "every")
            pts5.insert(2, ("Reflect on all", cl, acc))
        else:
            rnd5, msw5, bsw5 = P2_.v5_curves(core)
            pts5 = []
            for lab, k in (("No reflection", "keep"), ("Schema size", "struct"), ("Reflect on all", "every"), ("Oracle", "oracle"),
                           ("Map-Reflect", "v5")):
                acc, cl, _ = P2_.v5_policy(core, k)
                pts5.append((lab, cl, acc))
        FIG2[:] = [rnd5, msw5, pts5, bsw5]
        return
    th = TH
    n = N
    ever_calls = sum(float(np.sum(1 + c.ver_act)) for c in cells) / n
    rnd = [(s * ever_calls, FIRST + 100 * s * sum(float(np.sum(c.action - c.keep)) for c in cells) / n) for s in np.linspace(0, 1, 21)]
    sweep = []
    for dlt in np.round(np.arange(-0.5, 0.55, 0.05), 2):
        a = cl = 0.0
        for c in cells:
            t = th[c.model]
            rt = router.routed(c.margin, c.valid, c.s1, max(0.0, t["t_m"] + dlt), t["t_s"])
            a += float(np.sum(np.where(rt, c.action, c.keep)))
            cl += float(np.sum(rt * (1 + c.ver_act)))
        sweep.append((cl / n, 100 * a / n))
    st_calls = float(MV4["VFourStructuralRouterCalls"])
    pts = [("No reflection", 0.0, FIRST), ("Schema size", st_calls, ROWS["Structural"][0]),
           ("Reflect on all", ever_calls, ROWS["ReflectAllAction"][0]),
           ("Oracle", ROWS["Method"][6], ROWS["Oracle"][0]), ("Map-Reflect", ROWS["Method"][6], ROWS["Method"][0])]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.plot([x for x, _ in rnd], [y for _, y in rnd], color=VC.GREY, ls="--", lw=1.4, label="Random selection (rate swept)")
    ax.plot([x for x, _ in sweep], [y for _, y in sweep], color=VC.INK, lw=1.6, marker="o", ms=2.5,
            label="Margin threshold swept")
    off = {"No reflection": (5, 2), "Schema size": (-5, 5), "Reflect on all": (-4, -6), "Oracle": (-5, 5), "Map-Reflect": (6, -6)}
    for lab, x, y in pts:
        star = lab == "Map-Reflect"
        ax.plot([x], [y], marker="*" if star else "s", ms=13 if star else 4.5, color="#1A1A1A" if star else "#6E6E6E",
                markeredgecolor="white", markeredgewidth=0.8, zorder=5, linestyle="none")
        o = off[lab]
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=o, fontsize=6, va="top" if o[1] < 0 else "baseline",
                    ha="right" if o[0] < 0 else "left", color="#1A1A1A")
    ax.set_xlabel("Second-pass calls per field", fontsize=7)
    ax.set_ylabel("Test accuracy (%)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(color="#E6E6E6", lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "paper/figures/fig6_pareto.pdf")
    fig.savefig(PREVIEW / "fig6_pareto.png", dpi=200)
    plt.close(fig)
    FIG2[:] = [rnd, sweep, pts]


FIG2 = []
PREVIEW = Path(os.environ.get("V6_PREVIEW_DIR", "/tmp"))  # PNG previews, outside both repositories
import v6_charts as VC  # noqa: E402  the one palette of every v6 figure
COL, MRK = VC.MCOL, VC.MMRK


def fig3():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8, ls="--")
    for m in MODELS:
        bins = [(nb, cf, ac) for nb, cf, ac in X4[m]["bins"] if nb >= 30]
        ax.plot([cf for _, cf, _ in bins], [ac for *_, ac in bins], color=COL[m], marker=MRK[m], ms=3.5, lw=1.2,
                label=f"{MSHORT[m]} ({X4[m]['at1']:.0f}% at 1.0)")
    ax.set_xlabel("Top score (bin mean)", fontsize=7)
    ax.set_ylabel("First-pass accuracy", fontsize=7)
    ax.set_title("(a) Top score as a probability", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    ax = axes[1]
    for m in MODELS:
        ds = [(k, d) for k, d in enumerate(X4[m]["decs"]) if d[0]]
        ax.plot([k + 1 for k, _ in ds], [100 * d[3] for _, d in ds], color=COL[m], marker=MRK[m], ms=3.5, lw=1.2,
                label=f"{MSHORT[m]} (AUROC {X4[m]['auc']:.2f})")
    ax.set_xlabel("Margin decile (1 = smallest margin)", fontsize=7)
    ax.set_ylabel("First-pass accuracy (%)", fontsize=7)
    ax.set_title("(b) Accuracy by margin decile", fontsize=7)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    for ax in axes:
        ax.tick_params(labelsize=6)
        ax.grid(color="#E6E6E6", lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "paper/figures/fig6_calibration.pdf")
    fig.savefig(PREVIEW / "fig6_calibration.png", dpi=200)
    plt.close(fig)


# ========================================================================================= tables
def tex_s(x, nd=2):
    """A signed number in math mode, so a negative value prints a true minus sign."""
    return f"${float(x):+.{nd}f}$"


def tci(lo, hi, nd=2):
    """A 95% interval in small type with true minus signs."""
    return "[" + tex_s(lo, nd) + ", " + tex_s(hi, nd) + "]"


def no_dash(path):
    """R1: no placeholder dash and no text-mode hyphen before a digit in any generated table cell."""
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("%") or "&" not in ln:
            continue
        cells_ = [x.strip() for x in ln.rstrip("\\ ").split("&")]
        for x in cells_:
            assert x not in ("--", "-", "---"), (path, ln)
            bare = re.sub(r"\$[^$]*\$", "", x)
            assert not re.search(r"(?<![A-Za-z])-\d", bare), (path, ln)


def tnum(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


# name -> table*, from a scratch ACL two-column compile of every table at \small (overfull boxes at column width)
TABLE_ENV = {n: "table*" for n in ("main", "main_v41", "featabl", "slices", "interventions", "interventions_counts", "signals", "heldout", "repair",
                                   "calibration", "budget", "detect", "marginvar", "routerparts", "coverage", "accept",
                                   "logprob", "margincal", "perdataset", "perdataset_models")}
SSHORT = {"margin": "Margin", "s1": "Top score", "ratio": "Ratio", "entropy": "Score entropy", "lp_answer": "Token log-prob.",
          "verbalised": "Stated conf.", "p_true": "P(True)", "sem_entropy": "Answer entropy", "combo_paid": "Logistic (paid)"}
INTSHORT = {"Doubt": "Doubt", "SelfRefine": "Self-Refine", "Neutral": "Neutral", "NeutralCheck": "+Check",
            "NeutralEvid": "+Evidence", "Reflect": "Reflect", "WidenReflect": "Widen + reflect",
            "WidenNoCheck": "Widen + reflect, no check", "RoutedAction": "Reflect or widen + reflect", "Fresh": r"Fresh$^{\dagger}$",
            "Escalate": "Escalate"}


# Appendix "Ablation: Margin-Reflect" (author decision, DECISIONS row 41): the one place Margin-Reflect is named. Its
# tables and charts keep their Margin-Reflect labels, carry a group tag in their header comment and are listed, in
# order, in paper/tab6_ablation_mr_index.tex. No main-text table or chart carries a Margin-Reflect label.
# In the section's order in main_v6.tex (f12e48b), with the main table's ablation copy first (DECISIONS row 57).
ABLATION_MR = [("tab", "main_mr"), ("tab", "sensitivity"), ("tab", "holm"), ("tab", "perdataset_models"), ("tab", "slices"),
               ("tab", "heldout"), ("tab", "routerparts"), ("tab", "coverage"), ("tab", "accept"), ("tab", "marginvar"),
               ("tab", "benefit"), ("tab", "samplecount"), ("tab", "repair"), ("fig", "c14"), ("fig", "c10_gold_location")]
MR_TAG = '% group: ablation-mr (appendix "Ablation: Margin-Reflect"; listed in tab6_ablation_mr_index.tex)'
TABLE_SEP = {}  # name -> \tabcolsep in pt where 4 pt does not fit (never below 3, STYLE.md section 8)
PENDING_V5 = {"interventions": "R-X1", "interventions_counts": "R-X1", "signals": "R-UQ", "detect": "R-UQ",
              "budget": "R-UQ", "marginvar": "R-UQ", "logprob": "R-UQ", "samplecount": "R-UQ", "accept": "R-X1",
              "coverage": "R-X1", "routerparts": "R-X1", "repair": "R-X1"}


V5SIG_JSON = ROOT / "docs/naacl/results/v5_signals_facts.json"
MR_ALT = {"signals", "budget", "detect"}  # their Margin-Reflect versions become <name>_mr once the v5 versions exist
V5STRAT_JSON = ROOT / "docs/naacl/results/v5_strategies_facts.json"
MR_ALT_STRAT = {"interventions", "interventions_counts"}  # the same for the strategy tables (DECISIONS row 53)


def write_tab(name, colspec, header, body, env="table", note_="", label=None, v5=False):
    # STYLE.md section 8: never scaled, one font size (\small); a table too wide for a column is set as table*
    env = TABLE_ENV.get(name, env)
    import v6_phase2 as P2_
    if not v5 and ((name in MR_ALT and V5SIG_JSON.exists()) or (name in MR_ALT_STRAT and V5STRAT_JSON.exists())):
        # DECISIONS row 51: the main name holds the v5 version (v6_phase2.v5_signals); this Margin-Reflect version stays
        # generable as <name>_mr for the appendix ablation group, at the writer's choice
        env, label, name = TABLE_ENV.get(name, env), (label or name) + "_mr", name + "_mr"
        note_ = (note_ + " " if note_ else "") + "Margin-Reflect version (appendix ablation group, optional; DECISIONS rows 51 and 53)."
    elif P2_.V5 and name in PENDING_V5 and not v5:
        # these stay on the ablation configuration's own second pass by design (DECISIONS row 61), not pending a rerun
        note_ = (note_ + " " if note_ else "") + ("By design, the ablation configuration's own second pass (Margin-Reflect), "
                                                  "as in appendix app:ablmr; the v5 reruns do not replace it.")
    t = [f"% Generated by scripts/naacl/v6_numbers.py; do not edit by hand.{(' ' + note_) if note_ else ''}"]
    t += [MR_TAG] if ("tab", name) in ABLATION_MR or name.endswith("_mr") or name == "main_v41" else []
    t += [rf"\begin{{{env}}}[t]", rf"\centering\small\setlength{{\tabcolsep}}{{{TABLE_SEP.get(name, 4)}pt}}",
         rf"\begin{{tabular}}{{{colspec}}}",
         r"\toprule", header, r"\midrule"]
    t += body + [r"\bottomrule", r"\end{tabular}", r"\gencaption", rf"\label{{tab:{label or name}}}", rf"\end{{{env}}}"]
    (ROOT / f"paper/tab6_{name}.tex").write_text("\n".join(t) + "\n")


def tables():
    # Table 1: routers
    def row(lab, a, e, lo, hi, cl, tk, bold=False):
        nm = rf"\textbf{{{lab}}}" if bold else lab
        cs = f"{cl:.2f}"
        ts = tnum(tk)
        ci = "" if lo is None else " " + tci(lo, hi, 1)
        return f"{nm} & {a:.1f} & {tex_s(e)}{ci} & {cs} & {ts} \\\\"
    R = ROWS
    body = [f"No reflection & {R['Keep'][0]:.1f} & ref. & 0.00 & 0 \\\\"]
    for lab, k in (("Declined only", "DeclRetry"), ("Random selection", "Random"), ("Random selection, per dataset", "RandomDS"),
                   ("Random selection, per stratum", "RandomStrat"), ("Schema-size baseline", "Structural")):
        a, e, lo, hi, en, em, cl, tk = R[k]
        body.append(row(lab, a, e, lo, hi, cl, tk))
    for x in X2ROWS:
        if x["tg"] == "Err":
            body.append(row("Error trigger", FIRST + x["e"], x["e"], x["lo"], x["hi"], x["calls"], x["toks"]))
    a, e, lo, hi, en, em, cl, tk = R["Method"]
    body.append(row("Margin-Reflect", a, e, lo, hi, cl, tk, bold=True))
    uc, ut, u, ulo, uhi = R2COST["union"]
    body.append(row("Schema size or margin", FIRST + u, u, ulo, uhi, float(MV4["VFourUnionRouterCalls"]), ut))
    for lab, k in (("Reflect on all", "ReflectAllAction"), ("Oracle", "Oracle")):
        a, e, lo, hi, en, em, cl, tk = R[k]
        if k == "Oracle":
            cl, tk = R2COST["oracle"]
        body.append(row(lab, a, e, lo, hi, cl, tk))
    import v6_phase2 as P2_
    write_tab("main_v41" if P2_.V5 else "main", "lrrrr", r"Selection & Acc. & Change [95\% CI] & Calls & Tokens \\", body,
              env="table", note_="Method v4.1 (the reference when method v5 is scored). Source: docs/naacl/results/"
              "v4_1_test.md (rows), X2 for the error row.", label="main_v41" if P2_.V5 else None)
    if P2_.V5:
        P2_.v5_router_table(own=R["Method"])
    # per family / model slices
    body = []
    for gi, groups in enumerate(([(MSHORT[m], [c for c in cells if c.model == m]) for m in MODELS],
                                 [(nm, [c for c in cells if c.ds in dss]) for nm, dss in FAMILY])):
        for nm, rs in groups:
            f0 = 100 * sum(c.keep.sum() for c in rs) / sum(c.n for c in rs)
            v = [pooled([c for c in rs if c in SUBSET.get(k, rs)] or rs, POL[k])[0] for k in ("Method", "Random", "RandomDS", "ReflectAllAction", "Oracle")]
            rt = 100 * sum(c.routed.sum() for c in rs) / sum(c.n for c in rs)
            body.append(f"{nm} & {f0:.1f} & " + " & ".join(tex_s(x) for x in v) + f" & {rt:.0f} \\\\")
    write_tab("slices", "lrrrrrrr", r" & First & Margin-Reflect & Random & Random per dataset & Reflect on all & Oracle & Reflection rate (\%) \\", body)
    # Table 2: interventions, split in two so each fits the text width at \small (STYLE.md section 8)
    body, body2 = [], []
    for r in X1ROWS + X3ROWS:
        d, lo, hi = r["dm"]
        nm = INTSHORT[r["tg"]]
        body.append(f"{nm} & {tex_s(r['every'])} & {tex_s(r['m30'])} & {tex_s(r['r30'])} & {tex_s(d)} {tci(lo, hi)} & "
                    f"{tex_s(r['mr'])} \\\\")
        body2.append(f"{nm} & {tnum(r['n'])} & {r['f0']:.1f} & {tnum(r['chg'])} & {tnum(r['rep'])} & {tnum(r['brk'])} & "
                     f"{r['calls']:.2f} & {tnum(r['toks'])} \\\\")
    write_tab("interventions", "lrrrrr", r"Strategy & Reflect on all & Margin, 30\% & Random, 30\% & Margin$-$Random [95\% CI] & Margin-Reflect \\",
              body, note_="X1 (exploratory). Every: the intervention on every field; Routed: on the fields Map-Reflect routes. "
              "Fresh (dagger) is measured on the pairs where widening adds candidates.")
    write_tab("interventions_counts", "lrrrrrrr", r"Strategy & Pairs & First & Changed & Repaired & Broken & Calls & Tokens \\",
              body2, note_="X1 (exploratory). The intervention on every field: answers changed, repaired and broken, calls and "
              "tokens per field.")
    # Table 3: signals
    body = []
    for key, tg, lab, fp, bw, calls, auc, hoauc, dl, lo, hi, share in SIGTAB:
        lab = SSHORT[key]
        bw = bw.replace("WB (BB for Llama-4)", "WB/BB")
        nm = rf"\textbf{{{lab}}}" if key == "margin" else lab
        x = SIGX.get(key, {})
        dd = "ref." if dl is None else f"{tex_s(dl)} {tci(lo, hi)}"
        share = x.get("share", share)
        body.append(f"{nm} & {fp} & {bw} & {calls} & {auc} & {hoauc or x.get('hoauc', 'not run')} & {dd} & {share} \\\\")
    write_tab("signals", "lllrrrrr", r"UQ method & Cost & Access & Calls & Test AUROC & Held-out & $\Delta$ vs margin [95\% CI] & Oracle share \\",
              [b.replace("%", r"\%") for b in body], env="table*", note_="Source: docs/naacl/results/uq_ablation_v4.md; answer-token log-probability and answer entropy over all four models with the U9 rows (the same UQ run, in memory); held-out AUROC of top score, score entropy and the logistic from v6_phase2.r2_signals.")
    # held-out (appendix)
    sets = sorted({m.group(1) for k in MHO for m in [re.fullmatch(r"HO(\w+?)Fields", k)] if m and m.group(1) not in ("All", "Magellan")})
    body = []
    for s in sets + ["All"]:
        g = lambda k: MHO[f"HO{s}{k}"]  # noqa: E731
        if s == "All":
            body.append(r"\midrule")
        body.append(f"{'Pooled' if s == 'All' else s} & {g('Fields')} & {g('First')} & ${g('VThree')}$ {tci(g('VThreeLo'), g('VThreeHi'))} & "
                    f"${g('VMinusRandom')}$ & ${g('VMinusDecl')}$ & ${g('VMinusReflectAct')}$ & {g('Routed')} & {g('Calls')} & {g('ReflectActCalls')} \\\\")
    write_tab("heldout", "lrrrrrrrrr", r"Set & Fields & First & Change [95\% CI] & $-$Random & $-$Declined & $-$All & Rate (\%) & Calls & All: calls \\",
              body, env="table*", note_="Source: docs/naacl/results/newdata.md.")
    # X6 repairability (appendix)
    body = []
    for sname, gname, nf, err, rerr, rep, brk, rt, e_m, e_d, lo, hi in X6TAB:
        body.append(f"{sname} & {gname} & {tnum(nf)} & {tnum(err)} & {tnum(rerr)} & {tnum(rep)} & {tnum(brk)} & {rt:.0f} & "
                    f"{tex_s(e_m)} & {tex_s(e_d)} {tci(lo, hi)} \\\\")
    write_tab("repair", "llrrrrrrrr", r"Stratum & Gold & Pairs & Errors & Selected err. & Repairs & Breaks & Rate (\%) & Change & MR$-$Random [95\% CI] \\",
              body, env="table*", note_="X6 (exploratory).")
    # X4 calibration (appendix)
    body = []
    for m in MODELS:
        x = X4[m]
        body.append(f"{MSHORT[m]} & {tnum(x['n'])} & {x['at1']:.1f} & {x['at9']:.1f} & {x['conf']:.1f} & {x['acc']:.1f} & {x['ece']:.3f} & "
                    f"{x['brier']:.3f} & {x['auc']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}] \\\\")
    write_tab("calibration", "lrrrrrrrr", r"Model & Answered & Top=1 (\%) & Top$\geq$0.9 (\%) & Mean top & Accuracy & ECE & Brier & Margin AUROC [95\% CI] \\",
              body, note_="X4 (exploratory).")
    # sensitivity (appendix): development accuracy against t_m on the full 0.05 grid, read from the regenerated
    # v4_test.md; bold marks exactly each model's frozen t_m (v4_frozen.json), which the t_m column also states
    sec = V4MD[V4MD.index("Development accuracy against t_m"):]
    hdr = next(l for l in sec.splitlines() if l.startswith("| model | dev fields |"))
    grid = [float(x) for x in hdr.strip("|").split("|")[2:]]
    cols_, fz_ = {}, {}
    for m in MODELS:
        line = next(l for l in sec.splitlines() if l.startswith(f"| {MSHORT[m]} |"))
        cols_[m] = [float(x) for x in line.strip("|").split("|")[2:]]
        fz_[m] = float(TH[m]["t_m"])
        assert sum(abs(g - fz_[m]) < 1e-9 for g in grid) == 1, (m, fz_[m])
    body = []  # transposed (one row per t_m) so the table fits a column at \small
    for i, g in enumerate(grid):
        body.append(f"{g:.2f} & " + " & ".join(rf"\textbf{{{cols_[m][i]:.1f}}}" if abs(g - fz_[m]) < 1e-9 else f"{cols_[m][i]:.1f}"
                                            for m in MODELS) + r" \\")
    write_tab("sensitivity", "l" + "r" * len(MODELS), r"$t_m$ & " + " & ".join(MSHORT[m] for m in MODELS) + r" \\",
              body, note_="Development accuracy (%) against t_m, frozen t_s, widen action and check; source docs/naacl/results/"
              "v4_1_test.md (v4_analysis.review_extras, the Margin-Reflect configuration). Bold: each model's frozen t_m "
              "(v4_1_frozen.json).", label="sensitivity6")


# ===================================================================================== mismatch check, writers
def check():
    """Compare v6 facts with the committed v4 macros and result files wherever the same quantity exists."""
    disk4 = {}
    files = ([ARCHIVE / f for f in ("numbers.tex", "numbers_v4.tex", "numbers_v4_struct.tex", "numbers_uq.tex", "numbers_heldout.tex")]
             if VARIANT == "v4" else [ROOT / "docs/naacl/results/v4_1" / f for f in ("numbers_v4.tex", "numbers_v4_struct.tex")])
    for f in files:
        disk4.update({m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", f.read_text(), re.M)})
    F = {f["id"]: f for f in FACTS}
    pairs = [("Keep", "VFourKeepAcc"), ("MainAcc", "VFourMethodAcc"), ("MainGain", "VFourMethodGain"),
             ("MainGain", "VFourMethodGainLo", "lo"), ("MainGain", "VFourMethodGainHi", "hi"), ("MainCalls", "VFourMethodCalls"),
             ("MainTokens", "VFourMethodTokens"), ("MainRouted", "VFourRouted"), ("RandGain", "VFourRandomGain"),
             ("MainVsRand", "VFourVsRandom"), ("MainVsRand", "VFourVsRandomLo", "lo"), ("MainVsEvery", "VFourVsReflectAllAction"),
             ("MainVsEveryHolm", "VFourVsReflectAllActionHolm"), ("MainVsDecl", "VFourVsDeclRetry"), ("MainVsFullList", "VFourVsFullList"),
             ("MainVsEveryMacro", "VFourMacroVsReflectAllAction"), ("EveryGain", "VFourReflectAllActionGain"),
             ("EveryCalls", "VFourReflectAllActionCalls"), ("EveryCallShare", "VFourCallShare"), ("EveryTokenShare", "VFourTokenShare"),
             ("EveryNILower", "VFourNILower"), ("OracleGain", "VFourOracleGain"), ("StructGain", "VFourStructGain"),
             ("StructCalls", "VFourStructCalls"), ("RandStratGain", "VFourRandomStratGain"), ("MainVsRandStrat", "VFourVsRandomStrat"),
             ("StratWide", "VFourStratWideVsRandom"), ("StratShown", "VFourStratShownVsRandom"), ("MainVsRandDS", "VFourVsRandomDS"),
             ("ModelMin", "VFourGainMin"), ("ModelMax", "VFourGainMax"), ("FamValentineGain", "VFourFamValentine"),
             ("FamITOpsGain", "VFourFamITOps"), ("FamITOpsGain", "VFourFamITOpsLo", "lo"), ("NetDeclinedShare", "VFourNetDeclinedShare"),
             ("NetWideShare", "VFourNetWideShare"), ("BandShareWiden", "VFourBandShareWiden"), ("BandRepairsWiden", "VFourFixWiden"),
             ("BandBreaksWiden", "VFourBreakWiden"), ("MainFamNoVal", "VFourMethodFamMacroNoVal"), ("DataGroups", "VFourGroupsTotal"),
             ("FrozenTmQwen", "VFourTmQwen"), ("FrozenKLlama", "VFourKLlamafour"),
             ("SigMarginAUC", "UQAblMarginAUC"), ("SigStatedDelta", "UQAblStatedDelta"), ("SigPTrueDelta", "UQAblPTrueDelta"),
             ("SigComboPaidDelta", "UQAblComboPaidDelta"), ("ITOpsKappa", "ITOpsKappa"), ("ErrRouted", "UQAblErrRouted"), ("ErrInShown", "UQAblErrInShown"),
             ("HOFirst", "HOAllFirst"), ("HOGain", "HOAllVThree"), ("HOVsRand", "HOAllVMinusRandom"), ("HOVsEvery", "HOAllVMinusReflectAct"),
             ("HOEveryCalls", "HOAllReflectActCalls"), ("SigHOSwapStated", "HOAllSwapStated"),
             ("IntReflectEvery", None, None, "+0.24", "paper_archive/pre_v6_2026-10-10/tab_ablation.tex Reflect all"),
             ("IntDoubtEvery", None, None, "-23.36", "paper_archive/pre_v6_2026-10-10/tab_ablation.tex Doubt all"),
             ("IntNeutralEvidBroken", None, None, "347", "paper_archive/pre_v6_2026-10-10/tab_ablation.tex Neutral + evidence broken"),
             ("IntSelfRefineEvery", None, None, "-3.10", "paper_archive/pre_v6_2026-10-10/tab_selfrefine.tex Self-Refine all"),
             ("IntReflectMarginThirtyCell", None, None, "+0.21", "paper_archive/pre_v6_2026-10-10/tab_ablation.tex Reflect unc. 30"),
             ("IntDoubtMarginThirtyCell", None, None, "-4.64", "paper_archive/pre_v6_2026-10-10/tab_ablation.tex Doubt unc. 30"),
             ("IntRoutedActionEvery", "VFourReflectAllActionGain"), ("IntWidenReflectEvery", None, None, None, None)]
    out = []
    for p in pairs:
        fid, mac = p[0], p[1]
        part = p[2] if len(p) > 2 else None
        if fid not in F:
            continue
        f = F[fid]
        mine = {"lo": f["lo"], "hi": f["hi"]}.get(part, f["value"])
        if mac is None:
            if len(p) < 5 or p[3] is None or VARIANT != "v4":  # literal v4 table values: the v4 check only
                continue
            ref, where = p[3], p[4]
        else:
            if mac not in disk4:  # e.g. a uq or held-out macro: no committed v4.1 copy
                continue
            ref, where = disk4.get(mac), mac
        def num(s):
            s = str(s).replace("{,}", "").replace(",", "").replace("$", "")
            m = re.search(r"[+\-]?\d+(\.\d+)?", s)
            return (float(m.group(0)), len(m.group(1)) - 1 if m.group(1) else 0) if m else (None, 0)
        a, na = num(mine)
        b, nb = num(ref)
        same = a is not None and b is not None and abs(round(a, nb) - b) < 10 ** (-nb) / 2 + 1e-9
        out.append((fid, mine, where, ref, same))
    return out


def write_facts(chk, v4c=None):
    L = ["# v6 FACTS ledger", "",
         ("**Method v4.1.** Every fact is computed under method v4.1: method v4 with canonical margins (scripts/naacl/router.py, "
          "MAPREFLECT_CANONICAL_MARGIN=1, margins rounded to 12 decimals and threshold ties routed) and the thresholds frozen on "
          "development in docs/naacl/results/v4_1_frozen.json, with unresolved stated-confidence answers treated as missing "
          "(V6_UQ_VALID_ONLY=1). Both fixes come from docs/naacl/v6/reviews/opendata_uq_audit/. Values that changed are kept "
          "as exploratory *PreFix facts where the plan asks for them. " if VARIANT == "v4_1" else "")
         + "Generated by `scripts/naacl/v6_numbers.py`; do not edit by hand. Every number comes from the method-v4 frozen "
         "outputs or an offline analysis of them (no model call). Each fact `F-<ID>` is the macro `\\F<ID>` in "
         "`paper/numbers_v6.tex` (with `\\F<ID>Lo` and `\\F<ID>Hi` when it has an interval). Status: confirmatory = one of "
         "v4's five pre-declared Holm contrasts; descriptive = v4, outside the Holm family; exploratory = computed after the "
         "v4 freeze (X1 to X6, and the stratified gates added after the freeze); note = a definition or method fact, no macro.",
         "", "Intervals: 95% split-group bootstrap (scripts/naacl/group_stats.py, B = 2000, one shared draw). Signed values are "
         "points of accuracy unless the claim says otherwise.", "",
         "## Reproduction of the v4 generators", "",
         "The script reruns `v4_analysis.py`, `uq_ablation.py` (v4 mode) and `newdata_analysis.py` in memory with every write "
         "intercepted (none reaches paper/), and compares each regenerated file with its committed copy; pre-v6 paper files "
         "are compared with their archived copies in paper_archive/pre_v6_2026-10-10/:", ""]
    if VARIANT == "v4_1":
        L[-1] = ("Two checks. (1) The v4 path, unchanged: the script reruns `v4_analysis.py`, `uq_ablation.py` (v4 mode) and "
                 "`newdata_analysis.py` without canonical margins in a separate process, every write intercepted, and compares "
                 "each regenerated file with its committed copy (pre-v6 paper files with paper_archive/pre_v6_2026-10-10/):")
        L.append("")
        L += [f"- `{p}`: {'identical' if s else 'DIFFERS'}" for p, s in (v4c or {}).get("repro", [])]
        n4 = (v4c or {}).get("chk", [])
        L += ["", f"v4 facts against the committed v4 numbers: {sum(1 for x in n4 if x[4])} of {len(n4)} agree.", "",
              "(2) The v4.1 path: `v4_analysis.py --variant v4_1` rerun in memory against the committed v4.1 replay "
              "(docs/naacl/results/v4_1_test.md, docs/naacl/results/v4_1/):", ""]
    L += [f"- `{p}`: {'identical' if s else 'DIFFERS'}" for p, s in REPRO]
    L += ["", "## Facts", "", "| ID | claim in plain words | value | interval | source file | script/function | status |",
          "|---|---|---|---|---|---|---|"]
    for f in FACTS:
        iv = f"[{f['lo']}, {f['hi']}]" if f["lo"] is not None else ""
        L.append(f"| F-{f['id']} | {f['claim']} | {f['value']} | {iv} | {f['src']} | {f['fn']} | {f['status']} |")
    L += ["", "## Mismatch check against the committed " + ("v4" if VARIANT == "v4" else "v4.1") + " numbers", "",
          f"{sum(1 for x in chk if x[4])} of {len(chk)} sampled facts agree with the committed macro or table at its printed precision.", "",
          "| fact | v6 value | committed source | committed value | agrees |", "|---|---|---|---|---|"]
    L += [f"| F-{a} | {b} | {c} | {d} | {'yes' if e else 'NO'} |" for a, b, c, d, e in chk]
    L += [""]
    (ROOT / "docs/naacl/v6/FACTS.md").write_text("\n".join(L) + "\n")


def write_macros():
    out = ["% Generated by scripts/naacl/v6_numbers.py; do not edit by hand. Facts: docs/naacl/v6/FACTS.md."]
    seen = set()

    def tex(v):
        v = str(v)
        v = re.sub(r"(?<=\d),(?=\d{3})", "{,}", v)
        return v.replace("%", r"\%")
    for f in FACTS:
        if f["status"] == "note":
            continue
        for suf, v in (("", f["value"]), ("Lo", f["lo"]), ("Hi", f["hi"])):
            if v is None:
                continue
            name = "F" + f["id"] + suf
            assert name not in seen, name
            seen.add(name)
            out.append(f"\\newcommand{{\\{name}}}{{{tex(v)}}}")
    (ROOT / "paper/numbers_v6.tex").write_text("\n".join(out) + "\n")


def v4_check():
    """The reproduction check of the v4 path, unchanged: this script again with V6_VARIANT=v4 V6_CHECK_ONLY=1 in a
    separate process (the canonical-margin switch is read at import), returning its file and fact comparisons."""
    import subprocess
    env = dict(os.environ, V6_VARIANT="v4", V6_CHECK_ONLY="1")
    env.pop("MAPREFLECT_CANONICAL_MARGIN", None)
    env.pop("V6_UQ_VALID_ONLY", None)
    out = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("V4CHECK "))
    return json.loads(line[len("V4CHECK "):])


def write_mr_index():
    """paper/tab6_ablation_mr_index.tex: the appendix "Ablation: Margin-Reflect" group in order, for the writer. Each
    table is a \gentable line with the caption main_v6.tex gives it at this run (read only; the writer owns captions);
    each chart an \includegraphics line."""
    tex = (ROOT / "paper/main_v6.tex").read_text() if (ROOT / "paper/main_v6.tex").exists() else ""
    out = ["% Generated by scripts/naacl/v6_numbers.py; do not edit by hand.",
           '% Appendix "Ablation: Margin-Reflect": every Margin-Reflect-only table and chart, in order (DECISIONS row 41).',
           MR_TAG,
           "% Captions are copied from main_v6.tex as of this run; an empty caption means main_v6.tex has none yet."]
    for kind, name in ABLATION_MR:
        if kind == "tab":
            m = re.search(r"^\\gentable\{tab6_" + re.escape(name) + r"\}\{.*\}\s*$", tex, re.M)
            out.append(m.group(0).strip() if m else rf"\gentable{{tab6_{name}}}{{}}")
        else:
            out.append(rf"\includegraphics[width=\columnwidth]{{figures/fig6_{name}.pdf}}")
    if V5SIG_JSON.exists():  # DECISIONS row 51: the Margin-Reflect confidence-estimate versions, if the writer wants them
        out.append("% Optional, at the writer's choice: the Margin-Reflect versions of the confidence-estimate tables and charts.")
        for n in ("signals_compact", "signals", "signals_ho", "budget", "detect") + (
                ("interventions", "interventions_counts") if V5STRAT_JSON.exists() else ()):
            out.append(rf"% \gentable{{tab6_{n}_mr}}{{}}")
        for n in ("c1_budget_mr", "c12_detection_vs_routing_mr") + (
                ("c2_interventions_mr", "c9_repairs_breaks_mr") if V5STRAT_JSON.exists() else ()):
            out.append(rf"% \includegraphics[width=\columnwidth]{{figures/fig6_{n}.pdf}}")
    (ROOT / "paper/tab6_ablation_mr_index.tex").write_text("\n".join(out) + "\n")


def main():
    x5()
    x1()
    if CHECK_ONLY:
        chk = check()
        print("V4CHECK " + json.dumps(dict(repro=REPRO, chk=[list(map(str, x[:4])) + [bool(x[4])] for x in chk])))
        return
    check_facts()
    x6()
    x4()
    x2()
    x3()
    r2_costs()
    import v6_phase2 as P2
    P2.run_r2(globals())
    fig2()
    fig3()
    tables()
    P2.run_rest(globals())
    VC.main()  # the chart pages, from the CSVs just written
    chk = check()
    v4c = v4_check() if VARIANT == "v4_1" else None
    write_facts(chk, v4c)
    write_macros()
    write_mr_index()
    for f in sorted(glob.glob(str(ROOT / "paper/tab6_*.tex"))):
        no_dash(f)
    bad = [p for p, s in REPRO if not s]
    print(f"{len(FACTS)} facts ({VARIANT}); reproduction: {len(REPRO) - len(bad)} of {len(REPRO)} files identical {bad or ''}; "
          f"check: {sum(1 for x in chk if x[4])}/{len(chk)} agree")
    if v4c:
        b4 = [p for p, s in v4c["repro"] if not s]
        print(f"v4 path: reproduction {len(v4c['repro']) - len(b4)} of {len(v4c['repro'])} files identical {b4 or ''}; "
              f"check {sum(1 for x in v4c['chk'] if x[4])}/{len(v4c['chk'])} agree")
    for x in chk:
        if not x[4]:
            print("MISMATCH", x)


if __name__ == "__main__":
    main()
