#!/usr/bin/env python3
"""Method v4 on the test split: every gate and control from measured outcomes (docs/naacl/METHOD_V4.md, REDO_PLAN_V4.md).

Inputs: results/naacl/v4/test/given/<ds>__<model>/fields.jsonl (first pass, NT and the frozen widen action on every
field), the runner controls under results/naacl/v4/test/{hybrid,fresh_<W>,reask}_runs with their key files in
results/naacl/v4/keys, the full-list first pass of results/naacl/v3/test/fulllist, docs/naacl/results/v4_frozen.json.
Routing is scripts/naacl/router.py; every correctness flag is recomputed with eval.calc_accuracy.target_correct
(ITOpsGraph against its label file's acceptable targets, removed fields dropped). Statistics: scripts/naacl/group_stats.py
(split-group bootstrap within dataset, Holm over the pre-declared primary contrasts in docs/naacl/plan/stats.md).

Policies (per field, all measured):
  Keep              the first answer
  DeclRetry         the frozen widen action on declined (invalid) first passes only
  ReAsk             the first-pass prompt again on declined fields only
  FullList          the first pass over every field of the named type (no NL2GQL run; scored on the other datasets)
  Hybrid            the first pass over the named type's hybrid top 20
  ReflectAll        NT (reflection over the shown candidates) on every field
  ReflectAllAction  the field's routed action (NT when valid with s1 >= t_s, else the widen action) on every field
  Fresh             a first-pass prompt over the widen action's pool, no prior answer, evidence or check
  Random            the routed action on a random share of each model's fields equal to v4's routed share
                    (exact expectation); RandomDS matches the share within each model-dataset cell
  Oracle            at v4's share, exactly the fields the routed action fixes
  Method            v4: keep, near_tie (NT) or widen by the router

  Structural        (paper v4 only, not a Holm contrast) the widen action on declined fields and on fields whose named
                    target type has more than 20 fields, which the first pass cannot show in full; no uncertainty

Usage: .venv/bin/python scripts/naacl/v4_analysis.py [--B 2000]
Writes docs/naacl/results/v4_test.md, paper/numbers_v4.tex, paper/tab_v4.tex, and for the v4 paper
paper/numbers_v4_struct.tex and paper/tab_main_v4.tex (tab_v4.tex plus the structural-router row).
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--v4-root", default=str(ROOT / "results/naacl/v4"))
ap.add_argument("--labels-v21", default=__import__("os").environ.get("ITOPS_V21") or str(
    ROOT.parent / "schema-mapping-datasets/data/itopsgraph_v2_1/itopsgraph_v2.1_cross_product.jsonl"))
ap.add_argument("--variant", default="v4", choices=["v4", "v4_1"],
                help="v4_1: canonical margins (router.py) with docs/naacl/results/v4_1_frozen.json; outputs go to "
                     "docs/naacl/results/v4_1_test.md and docs/naacl/results/v4_1/, never over the v4 files")
ARGS = ap.parse_args()
if ARGS.variant == "v4_1":
    __import__("os").environ["MAPREFLECT_CANONICAL_MARGIN"] = "1"
argv, sys.argv = sys.argv, sys.argv[:1]
import reflect_v2 as R2  # noqa: E402
import router  # noqa: E402
import group_stats as GS  # noqa: E402
from eval.calc_accuracy import load_gold_file, target_correct  # noqa: E402
from stale_guard import require_no_stale  # noqa: E402
sys.argv = argv

V4 = Path(ARGS.v4_root)
FROZEN_FILE = "v4_frozen.json" if ARGS.variant == "v4" else "v4_1_frozen.json"
FROZEN = json.loads((ROOT / "docs/naacl/results" / FROZEN_FILE).read_text())


def OUT(rel):
    """Where an output goes: its v4 path, or for v4.1 a separate file that never overwrites a v4 output."""
    if ARGS.variant == "v4":
        return ROOT / rel
    if rel == "docs/naacl/results/v4_test.md":
        return ROOT / "docs/naacl/results/v4_1_test.md"
    d = ROOT / "docs/naacl/results/v4_1"
    d.mkdir(parents=True, exist_ok=True)
    return d / Path(rel).name
TH = FROZEN["thresholds"]
MODELS = ["gptoss", "llama4", "granite", "qwen35"]
DATASETS = FROZEN["datasets"]
SHORT = {"gptoss": "gpt-oss-120b", "llama4": "Llama-4", "granite": "Granite-4", "qwen35": "Qwen3.5"}
DNAME = {"itopsgraph": "ITOpsGraph", "toolbench": "ToolBench", "nl2gql": "NL2GQL", "omop_cms": "OMOP-CMS",
         "omop_mimic": "OMOP-MIMIC", "omop_synthea": "OMOP-Synthea", "valentine_chembl": "Val. ChEMBL",
         "valentine_opendata": "Val. OpenData", "valentine_tpcdi": "Val. TPC-DI", "valentine_wikidata": "Val. WikiData"}
FAMILY = [("ITOpsGraph", ["itopsgraph"]), ("ToolBench", ["toolbench"]), ("NL2GQL", ["nl2gql"]),
          ("OMOP", ["omop_cms", "omop_mimic", "omop_synthea"]),
          ("Valentine", ["valentine_chembl", "valentine_opendata", "valentine_tpcdi", "valentine_wikidata"])]
GOLD = load_gold_file(ARGS.labels_v21)
ITOPS_DUP = {sid: md.get("duplicate_group") for sid, (_, _, md) in GOLD.items()}


def read(f):
    return [json.loads(l) for l in open(f) if l.strip()]


def gold_of(ds, sid, field_lower, gts):
    """(targets or None when the field is not scored)."""
    if ds != "itopsgraph":
        return gts
    tg, rm, _ = GOLD[sid]
    if field_lower in rm:
        return None
    return tg[field_lower]


def control_answers(kind, ds, m, phase="test"):
    """'sample_id||field lower' -> answer of a runner control (hybrid, fresh_<W>, reask) on the fields it ran on."""
    out = {}
    for f in glob.glob(str(V4 / phase / f"{kind}_runs" / f"{ds}__{m}" / "mapreflect_*.json")):
        if "checkpoint" in f:
            continue
        d = json.load(open(f))
        ids = {str(r.get("sample_id")) for r in d["results"]}
        for rec in R2.records(d, ids):
            out[f"{rec['sample_id']}||{rec['field'].lower()}"] = (rec["base"] or "", rec["valid"])
    return out


def control_keys(ds, m, ext, phase="test"):
    f = V4 / "keys" / f"{phase}__{ds}__{m}.{ext}"
    if not f.exists():
        return set()
    if ext == "declined":
        return {l.strip().split("||", 1)[0] + "||" + l.strip().split("||", 1)[1].lower() for l in open(f) if l.strip()}
    return {k.split("||", 1)[0] + "||" + k.split("||", 1)[1].lower() for k in json.load(open(f))}


def call_means(files, stages):
    acc = defaultdict(lambda: np.zeros(2))
    for f in files:
        ds, m = Path(f).stem.split("__")[-2:]
        for r in read(f):
            st = r.get("stage")
            if st in stages:
                v = np.array([1, (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)], float)
                acc[(ds, m, st)] += v
                acc[(m, st)] += v
                acc[(st,)] += v
    return {k: v[1] / v[0] for k, v in acc.items() if v[0]}


class Cell:
    pass


SUBDESC = {"FullList": "no NL2GQL full-list run", "Fresh": "cells whose fresh-mapping run has finished",
           "Hybrid": "cells whose shortlist run has finished"}
FRESHCOV = [0.0]
SUBNOTE = {}  # policy -> (first-pass accuracy, pairs) when its row covers only part of the population
NOCTRL = defaultdict(int)
POOLS = {}  # dataset -> {"sample_id||field lower": the named type has fields the first pass did not show}  # (control, dataset, model) -> fields the control was meant to run on but has no outcome


def load():
    cells, missing, mism = [], [], 0
    full = {}
    for f in glob.glob(str(ROOT / "results/naacl/v3/test/fulllist/*/fields.jsonl")):
        ds, m = Path(f).parent.name.rsplit("__", 1)
        full[(ds, m)] = {r["key"]: r for r in read(f)}
    for ds in DATASETS:
        man = json.loads((ROOT / f"data/splits/{ds}.json").read_text())
        test_ids = set(man["test"]) - set(man.get("old_dev_excluded") or [])
        for m in MODELS:
            f = V4 / "test/given" / f"{ds}__{m}" / "fields.jsonl"
            if not f.exists() or not Path(str(f) + ".done").exists():
                missing.append(f"{ds}__{m}")
                continue
            t = TH[m]
            W = t["widen"]
            hyb, hyk = control_answers("hybrid", ds, m), control_keys(ds, m, "hybrid.json")
            frs, frk = control_answers(f"fresh_{W}", ds, m), control_keys(ds, m, "fresh.json")
            rsk = control_answers("reask", ds, m)
            fl = full.get((ds, m), {})
            pools = POOLS.setdefault(ds, {k.split("||", 1)[0] + "||" + k.split("||", 1)[1].lower(): bool(v.get("type_rest"))
                                          for k, v in json.load(open(ROOT / f"data/v4_pools/{ds}.json")).items()})
            rows = []
            for r in read(f):
                sid = str(r["sample_id"])
                if sid not in test_ids:
                    continue
                fld = r["key"].rsplit("||", 1)[1]  # some schema names contain "||"
                tg = gold_of(ds, sid, fld, r["gts"])
                if tg is None:
                    continue
                ok = lambda p: target_correct(p, tg)  # noqa: E731
                band, act = router.route(float(r["margin"]), bool(r["valid"]), float(r["s1"]), t, ("NT", W))
                if router.canon(r.get("band", band)) != band:
                    mism += 1
                ck = f"{sid}||{fld}"
                base = r["base"]
                act_ans = r[act]
                v4 = base if band == router.KEEP else act_ans
                ver = lambda a: float(bool(r.get(f"{a}_proposed")) and r.get(f"{a}_proposed") != base and bool(base))  # noqa
                if not r["valid"] and ck not in rsk:
                    NOCTRL[("reask", ds, m)] += 1
                rows.append(dict(
                    sid=sid, valid=bool(r["valid"]), band=band, act=act,
                    keep=ok(base), method=ok(v4), nt=ok(r["NT"]), action=ok(act_ans),
                    hybrid=ok(hyb[ck][0]) if ck in hyk and ck in hyb else ok(base),
                    fresh=ok(frs[ck][0]) if ck in frk and ck in frs else ok(base),
                    reask=ok(rsk[ck][0]) if (not r["valid"] and ck in rsk) else ok(base),
                    full=(ok(fl[r["key"]]["base"]) if r["key"] in fl else np.nan),
                    ver_nt=ver("NT"), ver_act=ver(act), widen=ok(r[W]), ver_w=ver(W),
                    wide=float(pools[ck]), margin=router.canonical_margin(float(r["margin"])), s1=float(r["s1"]),
                    hyb_call=float(ck in hyk), fresh_call=float(ck in frk)))
                for nm, ks, got in (("hybrid", hyk, hyb), ("fresh", frk, frs)):
                    if ck in ks and ck not in got:
                        NOCTRL[(nm, ds, m)] += 1
            c = Cell()
            c.ds, c.model, c.n = ds, m, len(rows)
            c.sid = np.array([x["sid"] for x in rows])
            for k in ("keep", "method", "nt", "action", "hybrid", "fresh", "reask", "full", "ver_nt", "ver_act",
                      "hyb_call", "fresh_call", "widen", "ver_w", "wide", "margin", "s1"):
                setattr(c, k, np.array([x[k] for x in rows], float))
            c.valid = np.array([x["valid"] for x in rows])
            c.band = np.array([x["band"] for x in rows])
            c.act = np.array([x["act"] for x in rows])
            c.routed = (c.band != router.KEEP).astype(float)
            c.fresh_done = NOCTRL.get(("fresh", ds, m), 0) == 0
            c.hyb_done = NOCTRL.get(("hybrid", ds, m), 0) == 0
            cells.append(c)
    return cells, missing, mism


def main():
    require_no_stale(ROOT / "results/naacl/v2", "the v4 test analysis")
    cells, missing, mism = load()
    groups = GS.split_groups(DATASETS, ITOPS_DUP)
    GB = GS.GroupBootstrap(cells, groups, B=ARGS.B)
    n = sum(c.n for c in cells)
    share_m = {m: sum(c.routed.sum() for c in cells if c.model == m) / sum(c.n for c in cells if c.model == m)
               for m in MODELS if any(c.model == m for c in cells)}

    def oracle(c, view):
        fix = np.where(c.keep == 0, c.action, 0.0)
        tot = sum(np.where(x.keep == 0, x.action, 0.0).sum() for x in view if x.model == c.model)
        k = share_m[c.model] * sum(x.n for x in view if x.model == c.model)
        return fix * min(1.0, k / tot) if tot else fix

    POL = {
        "Keep": lambda c: np.zeros(c.n),
        "DeclRetry": lambda c: (~c.valid) * (c.action - c.keep),
        "ReAsk": lambda c: c.reask - c.keep,
        "FullList": lambda c: np.nan_to_num(c.full - c.keep),
        "Hybrid": lambda c: c.hybrid - c.keep,
        "ReflectAll": lambda c: c.nt - c.keep,
        "ReflectAllAction": lambda c: c.action - c.keep,
        "Fresh": lambda c: c.fresh - c.keep,
        "Random": lambda c: share_m[c.model] * (c.action - c.keep),
        "RandomDS": lambda c: c.routed.mean() * (c.action - c.keep),
        "Oracle": lambda c: oracle(c, cells),
        "Method": lambda c: c.method - c.keep,
    }
    LABEL = {"Keep": "Keep every first answer", "DeclRetry": "Retry declined answers (widen)",
             "ReAsk": "Re-ask declined answers", "FullList": "Full candidate list", "Hybrid": "Hybrid top-20 shortlist",
             "ReflectAll": "Reflect on every field", "ReflectAllAction": "Second pass on every field",
             "Fresh": "Fresh mapping over the widened pool", "Random": "Random gate", "RandomDS": "Random gate, per dataset",
             "Oracle": "Oracle gate", "Method": "Map-Reflect (routed)"}
    # second-pass calls and tokens per field
    tv4 = call_means(glob.glob(str(V4 / "tokens/reflect__test__*.jsonl")),
                     {f"{p}:{a}" for p in ("reflect", "verify") for a in ("WT20", "WT50")})
    tv2 = call_means(glob.glob(str(ROOT / "results/naacl/v2/tokens/reflect__given__*.jsonl")), {"reflect:B", "verify:BC"})
    tct = {k: call_means(glob.glob(str(V4 / f"tokens/{k}__test__*.jsonl")), {f"rank:{k}"})
           for k in ("hybrid", "reask", "fresh_WT20", "fresh_WT50")}
    tfull = call_means(glob.glob(str(ROOT / "results/naacl/v3/tokens/full__test__*.jsonl")), {"rank:fulllist"})
    tfirst = call_means(glob.glob(str(ROOT / "results/naacl/v2/tokens/first__given__*.jsonl")), {"rank"})

    def tk(tab, ds, m, st):
        v = tab.get((ds, m, st), tab.get((m, st), tab.get((st,))))
        return float(v) if v is not None else 0.0

    def cost(c):
        W = TH[c.model]["widen"]
        nt_r, nt_v = tk(tv2, c.ds, c.model, "reflect:B"), tk(tv2, c.ds, c.model, "verify:BC")
        w_r, w_v = tk(tv4, c.ds, c.model, f"reflect:{W}"), tk(tv4, c.ds, c.model, f"verify:{W}")
        isnt = c.act == "NT"
        act_calls = 1 + c.ver_act
        act_tok = np.where(isnt, nt_r + c.ver_act * nt_v, w_r + c.ver_act * w_v)
        nt_calls, nt_tok = 1 + c.ver_nt, nt_r + c.ver_nt * nt_v
        fr = tk(tct[f"fresh_{W}"], c.ds, c.model, f"rank:fresh_{W}")
        return {
            "Keep": (0, 0), "DeclRetry": ((~c.valid) * act_calls, (~c.valid) * act_tok),
            "ReAsk": ((~c.valid) * 1.0, (~c.valid) * tk(tct["reask"], c.ds, c.model, "rank:reask")),
            "FullList": (np.full(c.n, np.nan), np.full(c.n, np.nan)),
            "Hybrid": (np.full(c.n, np.nan), np.full(c.n, np.nan)),
            "ReflectAll": (nt_calls, nt_tok), "ReflectAllAction": (act_calls, act_tok),
            "Fresh": (c.fresh_call, c.fresh_call * fr),
            "Random": (share_m[c.model] * act_calls, share_m[c.model] * act_tok),
            "RandomDS": (c.routed.mean() * act_calls, c.routed.mean() * act_tok),
            "Oracle": (np.full(c.n, np.nan), np.full(c.n, np.nan)),
            "Method": (c.routed * act_calls, c.routed * act_tok),
            "_first": (np.ones(c.n), np.full(c.n, tk(tfirst, c.ds, c.model, "rank"))),
            "_full": (np.ones(c.n), np.full(c.n, tk(tfull, c.ds, c.model, "rank:fulllist"))),
            "_hybrid": (c.hyb_call, c.hyb_call * tk(tct["hybrid"], c.ds, c.model, "rank:hybrid")),
        }
    COST = {}
    for c in cells:
        for k, (a, b) in cost(c).items():
            s = COST.setdefault(k, [0.0, 0.0])
            s[0] += float(np.sum(np.broadcast_to(a, (c.n,))))
            s[1] += float(np.sum(np.broadcast_to(b, (c.n,))))
    no_od = [c for c in cells if c.ds != "valentine_opendata"]
    fl_cells = [c for c in cells if not np.isnan(c.full).any()]
    fr_cells = [c for c in cells if c.fresh_done]
    SUBSET = {"FullList": fl_cells, "Fresh": fr_cells, "Hybrid": [c for c in cells if c.hyb_done]}
    first = 100 * sum(c.keep.sum() for c in cells) / n
    N = {}

    def num(k, v, meaning=""):
        N[k] = (v, meaning)

    def sgn(x):
        return f"{x:+.2f}"

    L = ["# Method v4 on the test split\n",
         "Generated by `scripts/naacl/v4_analysis.py` after the stale-row guard. Every gate and control is scored on "
         "measured outcomes; routing is `scripts/naacl/router.py`; correctness is `eval.calc_accuracy.target_correct` "
         "with full target identity; intervals are 95% split-group bootstrap intervals (B = "
         f"{ARGS.B}, `scripts/naacl/group_stats.py`).\n",
         f"Frozen values (docs/naacl/results/{FROZEN_FILE}): " + "; ".join(
             f"{SHORT[m]} t_m = {t['t_m']}, t_s = {t['t_s']}, widen {t['widen']}, accept {t['accept']}"
             for m, t in TH.items()) + ".\n",
         f"{n:,} test field-model pairs in {len(cells)} model-dataset cells; first pass {first:.2f}%."
         + (f" Cells without a finished v4 test run: {', '.join(missing)}." if missing else "")
         + (f" Stored bands that disagree with the router: {mism}." if mism else " Every stored band agrees with the router.")
         + " The full-list control has no NL2GQL run: its row and contrast use the "
         f"{sum(c.n for c in fl_cells):,} pairs of the other datasets."
         + (" Hybrid and fresh-mapping controls still running: " + ", ".join(f"{k[1]}__{k[2]} ({k[0]}, {v} fields)"
            for k, v in sorted(NOCTRL.items()) if v) + "; their rows use the other cells." if any(NOCTRL.values()) else "")
         + "\n",
         "## Main table\n",
         "| Policy | accuracy (%) | change [95% CI] | without OpenData | macro over datasets | second-pass calls per field | second-pass tokens per field |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    rows = {}
    SUBNOTE.clear()
    for key, f in POL.items():
        rs = SUBSET.get(key, cells)
        e, d = GB.pooled(rs, f)
        lo, hi = GS.interval(d)
        en, _ = GB.pooled([c for c in rs if c.ds != "valentine_opendata"], f)
        em, _ = GB.macro(rs, f)
        f0 = 100 * sum(c.keep.sum() for c in rs) / sum(c.n for c in rs)
        calls, toks = COST[key][0] / n, COST[key][1] / n
        rows[key] = (f0 + e, e, lo, hi, en, em, calls, toks)
        if len(rs) != len(cells):
            SUBNOTE[key] = (f0, sum(c.n for c in rs))
        cs = "--" if calls != calls else f"{calls:.2f}"
        ts = "--" if toks != toks else f"{toks:,.0f}"
        L.append(f"| {LABEL[key]} | {f0 + e:.2f} | {sgn(e)} [{sgn(lo)}, {sgn(hi)}] | {sgn(en)} | {sgn(em)} | {cs} | {ts} |")
        num(f"VFour{key}Acc", f"{f0 + e:.1f}", f"{LABEL[key]}: accuracy %")
        num(f"VFour{key}Gain", sgn(e), "change over the first pass, points")
        num(f"VFour{key}GainLo", sgn(lo), "split-group CI low")
        num(f"VFour{key}GainHi", sgn(hi), "split-group CI high")
        num(f"VFour{key}NoOD", sgn(en), "change without OpenData")
        num(f"VFour{key}Macro", sgn(em), "unweighted mean over datasets")
        num(f"VFour{key}Calls", cs, "second-pass calls per field")
        num(f"VFour{key}Tokens", ts.replace(",", "{,}"), "second-pass tokens per field")
    # structural router (no uncertainty): its main-table row, for the v4 paper only (paper/numbers_v4_struct.tex)
    NS = {}

    def num_s(k, v, meaning=""):
        NS[k] = (v, meaning)

    f_st = lambda c: STRUCT(c) * (c.widen - c.keep)  # noqa: E731
    e, d = GB.pooled(cells, f_st)
    lo, hi = GS.interval(d)
    en, _ = GB.pooled(no_od, f_st)
    em, _ = GB.macro(cells, f_st)
    st_calls = st_toks = 0.0
    for c in cells:
        W = TH[c.model]["widen"]
        w_r, w_v = tk(tv4, c.ds, c.model, f"reflect:{W}"), tk(tv4, c.ds, c.model, f"verify:{W}")
        st_calls += float(np.sum(STRUCT(c) * (1 + c.ver_w)))
        st_toks += float(np.sum(STRUCT(c) * (w_r + c.ver_w * w_v)))
    rows["Structural"] = (first + e, e, lo, hi, en, em, st_calls / n, st_toks / n)
    L.append(f"| Structural router (no uncertainty) | {first + e:.2f} | {sgn(e)} [{sgn(lo)}, {sgn(hi)}] | {sgn(en)} | "
             f"{sgn(em)} | {st_calls / n:.2f} | {st_toks / n:,.0f} |")
    num_s("VFourStructAcc", f"{first + e:.1f}", "structural router: accuracy %")
    num_s("VFourStructGain", sgn(e), "structural router: change over the first pass, points")
    num_s("VFourStructGainLo", sgn(lo), "CI low")
    num_s("VFourStructGainHi", sgn(hi), "CI high")
    num_s("VFourStructNoOD", sgn(en), "structural router: change without OpenData")
    num_s("VFourStructMacro", sgn(em), "structural router: unweighted mean over datasets")
    num_s("VFourStructCalls", f"{st_calls / n:.2f}", "structural router: second-pass calls per field")
    num_s("VFourStructTokens", f"{st_toks / n:,.0f}".replace(",", "{,}"), "structural router: second-pass tokens per field")

    # random gate matched within structural strata (declined, wide type, shown type) of each model-dataset cell, at
    # Map-Reflect's routed share in each stratum; exact expectation, stored outcomes only (exploratory)
    STRATA = {"declined": lambda c: ~c.valid, "wide": lambda c: c.valid & (c.wide > 0),
              "shown": lambda c: c.valid & (c.wide == 0)}

    def strat_w(c):
        w = np.zeros(c.n)
        for mf in STRATA.values():
            mk = mf(c)
            if mk.any():
                w[mk] = c.routed[mk].mean()
        return w

    f_rs = lambda c: strat_w(c) * (c.action - c.keep)  # noqa: E731
    e, d = GB.pooled(cells, f_rs)
    lo, hi = GS.interval(d)
    en, _ = GB.pooled(no_od, f_rs)
    em, _ = GB.macro(cells, f_rs)
    rs_calls = sum(float(np.sum(strat_w(c) * cost(c)["ReflectAllAction"][0])) for c in cells) / n
    rs_toks = sum(float(np.sum(strat_w(c) * cost(c)["ReflectAllAction"][1])) for c in cells) / n
    rows["RandomStrat"] = (first + e, e, lo, hi, en, em, rs_calls, rs_toks)
    L.append(f"| Random gate, per structural stratum | {first + e:.2f} | {sgn(e)} [{sgn(lo)}, {sgn(hi)}] | {sgn(en)} | "
             f"{sgn(em)} | {rs_calls:.2f} | {rs_toks:,.0f} |")
    num_s("VFourRandomStratAcc", f"{first + e:.1f}", "random gate per structural stratum: accuracy %")
    num_s("VFourRandomStratGain", sgn(e), "random gate per structural stratum: change, points")
    num_s("VFourRandomStratGainLo", sgn(lo), "CI low")
    num_s("VFourRandomStratGainHi", sgn(hi), "CI high")
    e, d = GB.pooled(cells, lambda c: POL["Method"](c) - f_rs(c))
    lo, hi = GS.interval(d)
    num_s("VFourVsRandomStrat", sgn(e), "Map-Reflect minus the stratified random gate, points")
    num_s("VFourVsRandomStratLo", sgn(lo), "CI low")
    num_s("VFourVsRandomStratHi", sgn(hi), "CI high")
    strat_lines = [f"Map-Reflect minus the random gate matched per structural stratum (exploratory), all fields: "
                   f"{sgn(e)} [{sgn(lo)}, {sgn(hi)}].", "",
                   "Inside each stratum (points of that stratum; random at Map-Reflect's share of the stratum in each "
                   "model-dataset cell):", "",
                   "| Stratum | pairs | share of test | first pass | routed | Map-Reflect | random | Map-Reflect minus random [95% CI] |",
                   "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for sn, mf in STRATA.items():
        sub = mask_cells(cells, mf)
        ns_ = sum(c.n for c in sub)
        em_, _ = GB.pooled(sub, POL["Method"])
        er_, _ = GB.pooled(sub, lambda c: c.routed.mean() * (c.action - c.keep))
        ed_, dd_ = GB.pooled(sub, lambda c: POL["Method"](c) - c.routed.mean() * (c.action - c.keep))
        l_, h_ = GS.interval(dd_)
        f0_ = 100 * sum(c.keep.sum() for c in sub) / ns_
        rt_ = 100 * sum(c.routed.sum() for c in sub) / ns_
        strat_lines.append(f"| {sn} | {ns_:,} | {100 * ns_ / n:.1f}% | {f0_:.2f} | {rt_:.1f}% | {sgn(em_)} | {sgn(er_)} | "
                           f"{sgn(ed_)} [{sgn(l_)}, {sgn(h_)}] |")
        tg = sn.capitalize()
        num_s(f"VFourStrat{tg}Share", f"{100 * ns_ / n:.0f}", f"% of test pairs in the {sn} stratum")
        num_s(f"VFourStrat{tg}Routed", f"{rt_:.0f}", f"% of the {sn} stratum routed by Map-Reflect")
        num_s(f"VFourStrat{tg}Gain", sgn(em_), f"Map-Reflect change inside the {sn} stratum")
        num_s(f"VFourStrat{tg}Random", sgn(er_), f"random change inside the {sn} stratum")
        num_s(f"VFourStrat{tg}VsRandom", sgn(ed_), f"Map-Reflect minus random inside the {sn} stratum")
        num_s(f"VFourStrat{tg}VsRandomLo", sgn(l_), "CI low")
        num_s(f"VFourStrat{tg}VsRandomHi", sgn(h_), "CI high")

    # band shares at one decimal, and the number of split groups per dataset
    for tag, fn in (("Keep", lambda c: c.routed == 0), ("Tie", lambda c: (c.routed > 0) & (c.act == "NT")),
                    ("Widen", lambda c: (c.routed > 0) & (c.act != "NT"))):
        num_s(f"VFourBandShare{tag}", f"{100 * sum(fn(c).sum() for c in cells) / n:.1f}", f"% of fields in the {tag} band")
    ng = {ds: GB.n_groups(ds) for ds in DATASETS}
    num_s("VFourGroupsTotal", f"{sum(ng.values())}", "split groups in the test, all datasets")
    num_s("VFourGroupsMin", f"{min(ng.values())}", "fewest split groups in one dataset")
    num_s("VFourGroupsMax", f"{max(ng.values())}", "most split groups in one dataset")
    strat_lines += ["", "Split groups per dataset (bootstrap clusters): " + ", ".join(f"{DNAME[k]} {v}" for k, v in ng.items())
                    + f"; {sum(ng.values())} in all."]

    L += ["", "Structural router (no uncertainty): the frozen widen action on declined fields and on fields whose named "
          "target type has more than 20 fields; for the v4 paper only, outside the Holm family. Random gate, per "
          "structural stratum: the routed action on a random share of each stratum (declined, wide type, shown type) of "
          "each model-dataset cell equal to Map-Reflect's share there (exact expectation)."]
    L += [""] + strat_lines
    L += [""] + [f"{LABEL[k]}: on {nn:,} pairs ({SUBDESC.get(k, 'a subset')}); accuracy is that subset's first pass "
                 f"({f0:.2f}%) plus the change." for k, (f0, nn) in SUBNOTE.items()]
    L += ["", "Full list and hybrid shortlist replace the first pass rather than add a second one; their first-pass "
          f"tokens per field are {COST['_full'][1] / n:,.0f} (full list) and the hybrid call runs on "
          f"{100 * COST['_hybrid'][0] / n:.1f}% of fields at {COST['_hybrid'][1] / max(COST['_hybrid'][0], 1):,.0f} "
          f"tokens per call, against {COST['_first'][1] / n:,.0f} first-pass tokens per field.\n"]

    # primary contrasts (pooled, Holm) and the macro family
    PRIM = [("Keep", "Keep"), ("Random", "Random"), ("ReflectAllAction", "ReflectAllAction"),
            ("DeclRetry", "DeclRetry"), ("FullList", "FullList")]
    for fam in ("pooled", "macro"):
        res = []
        for tag, k in PRIM:
            rs = SUBSET.get(k, cells)
            g = (lambda kk: lambda c: POL["Method"](c) - POL[kk](c))(k)
            e, d = getattr(GB, fam)(rs, g)
            res.append((tag, e, d, GS.boot_p(d)))
        adj = GS.holm([x[3] for x in res])
        L += [f"## Primary contrasts, {'field-pooled (primary family)' if fam == 'pooled' else 'macro over datasets (secondary family)'}\n",
              "| Map-Reflect minus | estimate | 95% CI | bootstrap p | Holm p |", "|---|--:|--:|--:|--:|"]
        for (tag, e, d, p), pa in zip(res, adj):
            lo, hi = GS.interval(d)
            L.append(f"| {LABEL[tag]} | {sgn(e)} | [{sgn(lo)}, {sgn(hi)}] | {p:.4f} | {pa:.4f} |")
            if fam == "macro" and tag == "ReflectAllAction":
                num("VFourMacroVsReflectAllAction", sgn(e), "macro over datasets: Map-Reflect minus the second pass on every field")
                num("VFourMacroVsReflectAllActionLo", sgn(lo), "CI low")
                num("VFourMacroVsReflectAllActionHi", sgn(hi), "CI high")
                num("VFourMacroVsReflectAllActionHolm", f"{pa:.3f}" if pa >= 0.001 else "<0.001", "Holm p, macro family")
            if fam == "pooled":
                hs = f"{pa:.3f}" if pa >= 0.001 else "<0.001"
                if tag != "Keep":
                    num(f"VFourVs{tag}", sgn(e), f"Map-Reflect minus {LABEL[tag]}")
                    num(f"VFourVs{tag}Lo", sgn(lo), "CI low")
                    num(f"VFourVs{tag}Hi", sgn(hi), "CI high")
                num(f"VFourVs{tag}Holm", hs, "Holm-adjusted p over the five primary contrasts")
        L.append("")

    # per model and per family
    def block(title, groups_):
        out = [f"## {title}\n", "| | fields | keep (%) | Map-Reflect | random gate | second pass on every field | "
               "declined retry | hybrid top-20 | fresh mapping | oracle | routed |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        vals = {}
        for name, rs in groups_:
            if not rs:
                continue
            f0 = 100 * sum(c.keep.sum() for c in rs) / sum(c.n for c in rs)
            es = {k: GB.pooled([c for c in rs if c in SUBSET.get(k, rs)] or rs, POL[k])
                  for k in ("Method", "Random", "ReflectAllAction", "DeclRetry", "Hybrid", "Fresh", "Oracle")}
            fm = lambda k: f"{sgn(es[k][0])} [{sgn(GS.interval(es[k][1])[0])}, {sgn(GS.interval(es[k][1])[1])}]"  # noqa
            rt = 100 * sum(c.routed.sum() for c in rs) / sum(c.n for c in rs)
            out.append(f"| {name} | {sum(c.n for c in rs):,} | {f0:.2f} | {fm('Method')} | {fm('Random')} | "
                       f"{fm('ReflectAllAction')} | {fm('DeclRetry')} | {fm('Hybrid')} | {fm('Fresh')} | "
                       f"{sgn(es['Oracle'][0])} | {rt:.1f}% |")
            vals[name] = (es["Method"][0],) + GS.interval(es["Method"][1])
        return out + [""], vals
    slices_tex(cells, POL, GB, SUBSET)
    T, vm = block("Per model", [(SHORT[m], [c for c in cells if c.model == m]) for m in MODELS])
    L += T
    T, vf = block("Per family", [(name, [c for c in cells if c.ds in dss]) for name, dss in FAMILY])
    L += T
    num("VFourGainMin", sgn(min(v[0] for v in vm.values())), "smallest per-model Map-Reflect change")
    num("VFourGainMax", sgn(max(v[0] for v in vm.values())), "largest per-model Map-Reflect change")
    for name, tag in (("Valentine", "Valentine"), ("ToolBench", "ToolBench"), ("OMOP", "OMOP"), ("NL2GQL", "NLGQL"),
                      ("ITOpsGraph", "ITOps")):
        num(f"VFourFam{tag}", sgn(vf[name][0]), f"Map-Reflect change, {name}")
        if tag == "ITOps":
            num("VFourFamITOpsLo", sgn(vf[name][1]), "CI low")
            num("VFourFamITOpsHi", sgn(vf[name][2]), "CI high")

    # Map-Reflect minus the per-dataset random gate (exploratory, outside the Holm family)
    e, d = GB.pooled(cells, lambda c: POL["Method"](c) - POL["RandomDS"](c))
    lo, hi = GS.interval(d)
    num("VFourVsRandomDS", sgn(e), "Map-Reflect minus the per-dataset random gate")
    num("VFourVsRandomDSLo", sgn(lo), "CI low")
    num("VFourVsRandomDSHi", sgn(hi), "CI high")
    L += [f"Map-Reflect minus the random gate matched per dataset (exploratory): {sgn(e)} [{sgn(lo)}, {sgn(hi)}].\n"]
    # answered fields (the population of the uncertainty-signal gate swap)
    na = int(sum(c.valid.sum() for c in cells))
    ka = 100 * sum(c.keep[c.valid].sum() for c in cells) / na
    num("VFourAnsweredFields", f"{na:,}".replace(",", "{,}"), "answered (valid first pass) test field-model pairs")
    num("VFourAnsweredKeep", f"{ka:.2f}", "first-pass accuracy on answered fields, %")
    # fresh-mapping coverage: fields whose widen-action pool adds candidates to the shown ones
    fcov = 100 * sum(c.fresh_call.sum() for c in cells) / n
    wid = 100 * sum((c.band == router.WIDEN).sum() for c in cells) / n
    num("VFourFreshShare", f"{fcov:.0f}", "% of fields where the widened pool adds candidates (the fresh call runs)")
    L += [f"Answered fields: {na:,} pairs, first pass {ka:.2f}%. Fresh mapping runs on the {fcov:.1f}% of fields whose "
          f"widened pool adds candidates to the shown ones (every field, not only the {wid:.1f}% widen band); elsewhere "
          "it equals the first pass.\n"]
    FRESHCOV[0] = fcov

    # bands
    L += ["## Routing bands\n", "| Band | fields | share | first pass | Map-Reflect | fixes | breaks |", "|---|--:|--:|--:|--:|--:|--:|"]
    for b, tag in ((router.KEEP, None), (router.NEAR_TIE, "Tie"), (router.WIDEN, "Widen")):
        mk = [(c, c.band == b) for c in cells]
        tot = sum(k.sum() for _, k in mk)
        fx = int(sum(((c.keep == 0) & (c.method == 1) & k).sum() for c, k in mk))
        br = int(sum(((c.keep == 1) & (c.method == 0) & k).sum() for c, k in mk))
        L.append(f"| {b} | {tot:,} | {100 * tot / n:.1f}% | {100 * sum(c.keep[k].sum() for c, k in mk) / max(tot, 1):.1f} | "
                 f"{100 * sum(c.method[k].sum() for c, k in mk) / max(tot, 1):.1f} | {fx:,} | {br:,} |")
        sh = 100 * tot / n
        num(f"VFour{tag or 'Keep'}First", f"{100 * sum(c.keep[k].sum() for c, k in mk) / max(tot, 1):.1f}",
            f"first-pass accuracy in the {b} band, %")
        num(f"VFourShare{tag or 'Keep'}", f"{sh:.0f}" if sh >= 10 else f"{sh:.1f}", f"% of fields in the {b} band")
        if tag:
            num(f"VFourFix{tag}", f"{fx:,}".replace(",", "{,}"), f"fixes, {b} band")
            num(f"VFourBreak{tag}", f"{br:,}".replace(",", "{,}"), f"breaks, {b} band")
    routed = 100 * sum(c.routed.sum() for c in cells) / n
    num("VFourRouted", f"{routed:.0f}", "% of fields routed to a second pass")
    num("VFourCallShare", f"{100 * COST['Method'][0] / COST['ReflectAllAction'][0]:.0f}",
        "Map-Reflect calls as % of the second pass on every field")
    num("VFourTokenShare", f"{100 * COST['Method'][1] / COST['ReflectAllAction'][1]:.0f}",
        "Map-Reflect tokens as % of the second pass on every field")
    num("VFourOracleOverMethod", sgn(rows["Oracle"][1] - rows["Method"][1]), "oracle minus Map-Reflect, points")
    for m in MODELS:
        tg = {"gptoss": "Gptoss", "llama4": "Llamafour", "granite": "Granite", "qwen35": "Qwen"}[m]
        num(f"VFourTm{tg}", f"{TH[m]['t_m']}", "frozen margin threshold")
        num(f"VFourK{tg}", TH[m]["widen"].lstrip("WTX"), "frozen widen K")
    num("VFourTs", f"{TH['gptoss']['t_s']}", "frozen top-score threshold (every model)")
    num("VFourDevAcc", f"{100 * FROZEN['dev_accuracy']:.1f}", "development accuracy at the frozen values")
    num("VFourDevKeep", f"{100 * FROZEN['dev_keep_accuracy']:.1f}", "development keep-all accuracy")
    L += ["", f"Routed {routed:.1f}% of fields. Second-pass calls per field: Map-Reflect {COST['Method'][0] / n:.3f}, "
          f"second pass on every field {COST['ReflectAllAction'][0] / n:.3f}.\n"]

    L += review_extras(cells, GB, POL, num, n, rows, oracle, num_s)
    OUT("docs/naacl/results/v4_test.md").write_text("\n".join(L) + "\n")
    nt = ["% Generated by scripts/naacl/v4_analysis.py; do not edit by hand. Method v4 on the test split."]
    nt += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, (v, _) in N.items()]
    OUT("paper/numbers_v4.tex").write_text("\n".join(nt) + "\n")
    idx = ["# numbers_v4.tex index", "", "| Macro | Value | Meaning |", "|---|--:|---|"]
    idx += [f"| `\\{k}` | {v} | {mm} |" for k, (v, mm) in N.items()]
    OUT("docs/naacl/plan/numbers_v4_index.md").write_text("\n".join(idx) + "\n")
    tab_v4(rows, LABEL)
    tab_main_v4(rows, LABEL)
    ns = ["% Generated by scripts/naacl/v4_analysis.py; do not edit by hand. Structural router (v4 paper only)."]
    ns += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, (v, _) in NS.items()]
    OUT("paper/numbers_v4_struct.tex").write_text("\n".join(ns) + "\n")
    print("\n".join(L[:40]))


NI_MARGIN = 0.5  # non-inferiority margin in points, pre-stated in docs/naacl/PRINCIPLED_REVISION_PLAN.md


def STRUCT(c):
    """Fields the structural router sends to the widen action: declined, or of a named type wider than 20 fields."""
    return (~c.valid) | (c.wide > 0)


def mask_cells(cells, fn):
    """Cells restricted to the fields where fn(cell) is True (every per-field array masked)."""
    out = []
    for c in cells:
        mk = fn(c)
        if not mk.any():
            continue
        x = Cell()
        for k, v in c.__dict__.items():
            setattr(x, k, v[mk] if isinstance(v, np.ndarray) and v.shape[:1] == (c.n,) else v)
        x.n = int(mk.sum())
        out.append(x)
    return out


def review_extras(cells, GB, POL, num, n, rows, oracle, num_s=lambda *a: None):
    """The final reviews' analyses, all on stored v4 test outcomes: a structural router, the union of structure and
    margin, the margin on answered fully shown types, Pareto sweeps, non-inferiority, macro over families, the
    decomposition of net repairs, and development threshold sensitivity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def sgn(x):
        return f"{x:+.2f}"

    def ci_(d):
        return GS.interval(d)

    struct = STRUCT  # routed without any uncertainty signal
    P = {
        "Structural": (lambda c: struct(c) * (c.widen - c.keep), lambda c: struct(c) * (1 + c.ver_w)),
        "Union": (lambda c: np.where(c.routed > 0, c.action, np.where(struct(c), c.widen, c.keep)) - c.keep,
                  lambda c: np.where(c.routed > 0, 1 + c.ver_act, np.where(struct(c), 1 + c.ver_w, 0.0))),
        "Method": (POL["Method"], lambda c: c.routed * (1 + c.ver_act)),
    }
    calls = {k: sum(float(np.sum(f[1](c))) for c in cells) / n for k, f in P.items()}
    first = 100 * sum(c.keep.sum() for c in cells) / n
    L = ["## Final-review analyses (stored v4 test outcomes, no model call)\n",
         "Structural router: send a field to the frozen widen action when its first pass is declined or its named "
         "target type has fields the first pass did not show (more than 20), keep it otherwise; no uncertainty signal. "
         "Union: Map-Reflect's routing, plus the structural fields it keeps (sent to the widen action).\n",
         "| Policy | accuracy (%) | change [95% CI] | calls per field | routed |", "|---|--:|--:|--:|--:|"]
    E = {}
    for k, (f, cf) in P.items():
        e, d = GB.pooled(cells, f)
        E[k] = (e, d)
        rt = {"Structural": sum(struct(c).sum() for c in cells),
              "Union": sum(((c.routed > 0) | struct(c)).sum() for c in cells),
              "Method": sum(c.routed.sum() for c in cells)}[k] / n
        L.append(f"| {k} | {first + e:.2f} | {sgn(e)} [{sgn(ci_(d)[0])}, {sgn(ci_(d)[1])}] | {calls[k]:.2f} | {100 * rt:.1f}% |")
        num(f"VFour{k}Router" if k != "Method" else "VFourMethodRouter", sgn(e), f"{k} routing: change, points")
        num(f"VFour{k}RouterCalls" if k != "Method" else "VFourMethodRouterCalls", f"{calls[k]:.2f}", "calls per field")
        num(f"VFour{k}RouterRouted" if k != "Method" else "VFourMethodRouterRouted", f"{100 * rt:.0f}", "% routed")
    for k in ("Structural", "Union"):
        e, d = GB.pooled(cells, lambda c, k=k: P["Method"][0](c) - P[k][0](c))
        lo, hi = ci_(d)
        L.append(f"\nMap-Reflect minus {k.lower()} routing: {sgn(e)} [{sgn(lo)}, {sgn(hi)}].")
        num(f"VFourVs{k}", sgn(e), f"Map-Reflect minus {k.lower()} routing")
        if k == "Union":
            num_s("VFourUnionOverMethod", sgn(-e), "union routing minus Map-Reflect, points")
            num_s("VFourUnionOverMethodLo", sgn(-hi), "CI low")
            num_s("VFourUnionOverMethodHi", sgn(-lo), "CI high")
        num(f"VFourVs{k}Lo", sgn(lo), "CI low")
        num(f"VFourVs{k}Hi", sgn(hi), "CI high")
    # the margin where structure has nothing to say: answered fields of fully shown types
    sub = mask_cells(cells, lambda c: c.valid & (c.wide == 0))
    ns = sum(c.n for c in sub)
    sh = {c.model: 0 for c in sub}
    for m in sh:
        sh[m] = sum(c.routed.sum() for c in sub if c.model == m) / sum(c.n for c in sub if c.model == m)
    e_m, d_m = GB.pooled(sub, POL["Method"])
    e_r, d_r = GB.pooled(sub, lambda c: sh[c.model] * (c.action - c.keep))
    e_d, d_d = GB.pooled(sub, lambda c: POL["Method"](c) - sh[c.model] * (c.action - c.keep))
    f0s = 100 * sum(c.keep.sum() for c in sub) / ns
    rts = 100 * sum(c.routed.sum() for c in sub) / ns
    L += ["", f"Margin gate on answered fields of fully shown types ({ns:,} pairs, {100 * ns / n:.1f}% of the test; "
          f"first pass {f0s:.2f}%; {rts:.1f}% routed): Map-Reflect {sgn(e_m)} [{sgn(ci_(d_m)[0])}, {sgn(ci_(d_m)[1])}], "
          f"random gate at the same share {sgn(e_r)}, difference {sgn(e_d)} [{sgn(ci_(d_d)[0])}, {sgn(ci_(d_d)[1])}] "
          "(points of that population).\n"]
    num("VFourShownFields", f"{ns:,}".replace(",", "{,}"), "answered fields of fully shown types (pairs)")
    num("VFourShownShare", f"{100 * ns / n:.0f}", "% of test pairs that are answered fields of fully shown types")
    num("VFourShownFirst", f"{f0s:.1f}", "first pass on that population, %")
    num("VFourShownRouted", f"{rts:.0f}", "% of that population routed by the margin")
    num("VFourShownGain", sgn(e_m), "Map-Reflect change on that population")
    num("VFourShownGainLo", sgn(ci_(d_m)[0]), "CI low")
    num("VFourShownGainHi", sgn(ci_(d_m)[1]), "CI high")
    num("VFourShownVsRandom", sgn(e_d), "Map-Reflect minus random at the same share, that population")
    num("VFourShownVsRandomLo", sgn(ci_(d_d)[0]), "CI low")
    num("VFourShownVsRandomHi", sgn(ci_(d_d)[1]), "CI high")
    # per family
    L += ["Per family (change, points; calls per field):", "",
          "| Family | Map-Reflect | structural | union | Map-Reflect minus structural |", "|---|--:|--:|--:|--:|"]
    for name, dss in FAMILY:
        rs = [c for c in cells if c.ds in dss]
        nf = sum(c.n for c in rs)
        vals = {k: GB.pooled(rs, P[k][0])[0] for k in P}
        cl = {k: sum(float(np.sum(P[k][1](c))) for c in rs) / nf for k in P}
        e, d = GB.pooled(rs, lambda c: P["Method"][0](c) - P["Structural"][0](c))
        L.append(f"| {name} | {sgn(vals['Method'])} ({cl['Method']:.2f}) | {sgn(vals['Structural'])} "
                 f"({cl['Structural']:.2f}) | {sgn(vals['Union'])} ({cl['Union']:.2f}) | {sgn(e)} "
                 f"[{sgn(ci_(d)[0])}, {sgn(ci_(d)[1])}] |")
        tg = {"ITOpsGraph": "ITOps", "ToolBench": "ToolBench", "NL2GQL": "NLGQL", "OMOP": "OMOP",
              "Valentine": "Valentine"}[name]
        num_s(f"VFourStructFam{tg}", sgn(vals["Structural"]), f"{name}: structural router change, points")
        num_s(f"VFourStructFam{tg}Calls", f"{cl['Structural']:.2f}", f"{name}: structural router calls per field")
        num_s(f"VFourMethodFam{tg}Calls", f"{cl['Method']:.2f}", f"{name}: Map-Reflect calls per field")
        num_s(f"VFourVsStructFam{tg}", sgn(e), f"{name}: Map-Reflect minus structural routing, points")
        num_s(f"VFourVsStructFam{tg}Lo", sgn(ci_(d)[0]), "CI low")
        num_s(f"VFourVsStructFam{tg}Hi", sgn(ci_(d)[1]), "CI high")
    L.append("")

    # non-inferiority against the second pass on every field
    e, d = GB.pooled(cells, lambda c: POL["Method"](c) - POL["ReflectAllAction"](c))
    lower = float(np.percentile(d[~np.isnan(d)], 5))
    ni = lower > -NI_MARGIN
    L += [f"Non-inferiority, Map-Reflect minus the second pass on every field: {sgn(e)}, one-sided 95% lower bound "
          f"{sgn(lower)} against a margin of -{NI_MARGIN} points: {'non-inferior' if ni else 'not shown non-inferior'}.\n"]
    num("VFourNIDiff", sgn(e), "Map-Reflect minus the second pass on every field")
    num("VFourNILower", sgn(lower), "one-sided 95% lower bound (split-group bootstrap)")
    num("VFourNIMargin", f"{NI_MARGIN}", "pre-stated non-inferiority margin, points")
    num("VFourNIResult", "non-inferior" if ni else "not shown non-inferior", "non-inferiority verdict")
    num("VFourHolmFloor", "$p < 0.003$ (bootstrap resolution)",
        f"Holm p at the bootstrap floor: 5 x 1/{GB.B} = {5 / GB.B:.4f}")

    # macro over families (all five, and the four without Valentine)
    fam_d = {name: GB.pooled([c for c in cells if c.ds in dss], POL["Method"]) for name, dss in FAMILY}
    for tag, names in (("FamMacro", [f for f, _ in FAMILY]), ("FamMacroNoVal", [f for f, _ in FAMILY if f != "Valentine"])):
        e = float(np.mean([fam_d[f][0] for f in names]))
        d = np.mean([fam_d[f][1] for f in names], axis=0)
        lo, hi = ci_(d)
        L.append(f"Map-Reflect, mean over {'the five families' if tag == 'FamMacro' else 'the four families without Valentine'}: "
                 f"{sgn(e)} [{sgn(lo)}, {sgn(hi)}].")
        num(f"VFourMethod{tag}", sgn(e), f"Map-Reflect change, unweighted mean over {len(names)} families")
        num(f"VFourMethod{tag}Lo", sgn(lo), "CI low")
        num(f"VFourMethod{tag}Hi", sgn(hi), "CI high")
    L.append("")

    # where the net repairs come from
    parts = {"declined": lambda c: ~c.valid, "wide": lambda c: c.valid & (c.wide > 0),
             "rest": lambda c: c.valid & (c.wide == 0)}
    net = {}
    for k, mf in parts.items():
        fx = sum(((c.keep == 0) & (c.method == 1) & mf(c)).sum() for c in cells)
        br = sum(((c.keep == 1) & (c.method == 0) & mf(c)).sum() for c in cells)
        net[k] = (int(fx), int(br), int(fx - br))
    tot = sum(v[2] for v in net.values())
    L += ["Net repairs of Map-Reflect (fixes minus breaks) by field type:", "", "| Fields | fixes | breaks | net | share of net |",
          "|---|--:|--:|--:|--:|"]
    for k, (fx, br, nt_) in net.items():
        L.append(f"| {k} | {fx:,} | {br:,} | {nt_:,} | {100 * nt_ / tot:.1f}% |")
        tg = {"declined": "Declined", "wide": "Wide", "rest": "Rest"}[k]
        num(f"VFourNet{tg}", f"{nt_:,}".replace(",", "{,}"), f"net repairs, {k} fields")
        num(f"VFourNet{tg}Share", f"{100 * nt_ / tot:.0f}", f"% of net repairs from {k} fields")
    L.append("")

    # Pareto: accuracy against second-pass calls per field
    th = FROZEN["thresholds"]
    pts = {"Keep every first answer": (0.0, first),
           "Retry declined answers": (sum(float(np.sum((~c.valid) * (1 + c.ver_act))) for c in cells) / n,
                                      first + rows["DeclRetry"][1]),
           "Structural router": (calls["Structural"], first + E["Structural"][0]),
           "Structure or margin": (calls["Union"], first + E["Union"][0]),
           "Second pass on every field": (sum(float(np.sum(1 + c.ver_act)) for c in cells) / n,
                                          first + rows["ReflectAllAction"][1]),
           "Oracle gate": (calls["Method"], first + rows["Oracle"][1]),
           "Map-Reflect": (calls["Method"], first + rows["Method"][1])}
    rnd = [(s_ * sum(float(np.sum(1 + c.ver_act)) for c in cells) / n,
            first + 100 * s_ * sum(float(np.sum(c.action - c.keep)) for c in cells) / n) for s_ in np.linspace(0, 1, 11)]
    sweep = []
    for dlt in np.round(np.arange(-0.5, 0.55, 0.05), 2):
        a = cl = 0.0
        for c in cells:
            t = th[c.model]
            rt = router.routed(c.margin, c.valid, c.s1, max(0.0, t["t_m"] + dlt), t["t_s"])
            a += float(np.sum(np.where(rt, c.action, c.keep)))
            cl += float(np.sum(rt * (1 + c.ver_act)))
        sweep.append((dlt, cl / n, 100 * a / n))
    L += ["Margin-threshold sweep (every model's frozen t_m shifted by the same amount, frozen t_s and actions):", "",
          "| shift | calls per field | accuracy (%) |", "|--:|--:|--:|"]
    L += [f"| {dl:+.2f} | {cl:.3f} | {a:.2f} |" for dl, cl, a in sweep] + [""]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot([x for x, _ in rnd], [y for _, y in rnd], color="#C55A11", lw=2, label="Random gate (share swept)")
    ax.plot([x for _, x, _ in sweep], [y for _, _, y in sweep], color="#009E73", lw=2, marker="o", ms=3,
            label="Margin threshold swept")
    for lab, (x, y) in pts.items():
        star = lab == "Map-Reflect"
        ax.plot([x], [y], marker="*" if star else "s", ms=11 if star else 5, color="#1A1A1A" if star else "#6E6E6E",
                markeredgecolor="white", markeredgewidth=0.8, zorder=5, linestyle="none")
        off = {"Keep every first answer": (6, 3), "Second pass on every field": (-4, -5), "Oracle gate": (-5, 4),
               "Map-Reflect": (6, -5), "Structural router": (-5, 4), "Structure or margin": (5, 5),
               "Retry declined answers": (6, -3)}.get(lab, (4, 2))
        ax.annotate(lab.replace("Second pass on every field", "Second pass\non every field"), (x, y),
                    textcoords="offset points", xytext=off, fontsize=6, va="top" if off[1] < 0 else "baseline",
                    ha="right" if off[0] < 0 else "left", color="#1A1A1A")
    ax.set_ylim(min(y for _, y in rnd) - 0.3, max(y for _, y in pts.values()) + 0.45)
    ax.set_xlabel("Second-pass calls per field", fontsize=7)
    ax.set_ylabel("Test accuracy (%)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(color="#E6E6E6", lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT("paper/figures/fig_pareto.pdf"))
    plt.close(fig)

    # development threshold sensitivity (dev rows, frozen t_s and widen action, check accept)
    grid = np.round(np.arange(0, 1.01, 0.05), 2)
    sens = {}
    for m in MODELS:
        t, W = th[m], th[m]["widen"]
        acc = np.zeros(len(grid))
        nn = 0
        for ds in DATASETS:
            f = V4 / "dev/given" / f"{ds}__{m}" / "fields.jsonl"
            if not f.exists():
                continue
            man = json.loads((ROOT / f"data/splits/{ds}.json").read_text())
            dev_ids = set(man["dev"])
            for r in read(f):
                sid = str(r["sample_id"])
                if sid not in dev_ids or W not in r:
                    continue
                tg = gold_of(ds, sid, r["key"].rsplit("||", 1)[1], r["gts"])
                if tg is None:
                    continue
                act = "NT" if (r["valid"] and float(r["s1"]) >= t["t_s"]) else W
                kp, ac = target_correct(r["base"], tg), target_correct(r[act], tg)
                rt = np.array([router.band(float(r["margin"]), bool(r["valid"]), float(r["s1"]), g, t["t_s"]) != router.KEEP
                               for g in grid])
                acc += np.where(rt, ac, kp)
                nn += 1
        sens[m] = (100 * acc / nn, nn)
    t_ = [r"\begin{table}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{2.5pt}",
          "% Generated by scripts/naacl/v4_analysis.py; do not edit by hand.", r"\resizebox{\columnwidth}{!}{%",
          r"\begin{tabular}{l" + "r" * 11 + "|rr}", r"\toprule",
          r"$t_m$ & " + " & ".join(f"{g:.1f}" for g in grid[::2]) + r" & Frozen & Acc. \\",
          r"\midrule"]
    L += ["Development accuracy against t_m (frozen t_s, widen action and check):", "",
          "| model | dev fields | " + " | ".join(f"{g:.2f}" for g in grid) + " |", "|---|--:|" + "--:|" * len(grid)]
    frozen_acc = {}
    for m in MODELS:
        a, nn = sens[m]
        L.append(f"| {SHORT[m]} | {nn:,} | " + " | ".join(f"{x:.2f}" for x in a) + " |")
        fz = th[m]["t_m"]
        cells_ = []
        for g, x in zip(grid[::2], a[::2]):
            cells_.append(f"{x:.1f}")
        fa = a[int(np.argmin(np.abs(grid - fz)))]  # the frozen t_m lies on the 0.05 grid
        assert abs(grid[int(np.argmin(np.abs(grid - fz)))] - fz) < 1e-9
        frozen_acc[m] = fa
        t_.append(f"{SHORT[m]} & " + " & ".join(cells_) + f" & {fz:g} & " + r"\textbf{" + f"{fa:.1f}" + r"} \\")
        near = [x for g, x in zip(grid, a) if abs(g - fz) <= 0.1 + 1e-9]
        num(f"VFourSens{ {'gptoss': 'Gptoss', 'llama4': 'Llamafour', 'granite': 'Granite', 'qwen35': 'Qwen'}[m] }",
            f"{max(near) - min(near):.2f}", "dev accuracy range within 0.1 of the frozen t_m, points")
    L.append("")
    best = all(abs(frozen_acc[m] - sens[m][0].max()) < 1e-9 for m in MODELS)
    ga = sens["gptoss"][0]
    gties = [g for g, x in zip(grid, ga) if round(x, 1) == round(frozen_acc["gptoss"], 1)]
    tie_txt = ", ".join(f"{g:.2f}" for g in gties[:-1]) + f" and {gties[-1]:.2f}" if len(gties) > 1 else f"{gties[0]:.2f}"
    t_ += [r"\bottomrule", r"\end{tabular}}",
           r"\caption{Accuracy changes little near the frozen thresholds. Development accuracy (\%) against the margin "
           r"threshold $t_m$ on a 0.1 grid, with each model's frozen $t_s$, widen action and check. Frozen: each model's "
           r"frozen $t_m$, and in bold its development accuracy there (from the 0.05 grid of the fit)."
           + (r" Every model's frozen value attains its best development accuracy on the 0.05 grid; at the printed "
              r"precision gpt-oss-120b reaches " + f"{frozen_acc['gptoss']:.1f}" + " at $t_m$ = " + tie_txt + "."
              if best else "") + "}",
           r"\label{tab:sensitivity}", r"\end{table}"]
    OUT("paper/tab_v4_sensitivity.tex").write_text("\n".join(t_) + "\n")
    return L


def slices_tex(cells, POL, GB, SUBSET):
    """paper/tab_v4_slices.tex: Table 2's main comparisons per model and per family (change over the first pass)."""
    cols = [("Method", "Map-Reflect"), ("Random", "Random"), ("RandomDS", "Random/DS"),
            ("ReflectAllAction", "All fields"), ("Fresh", "Fresh"), ("Hybrid", "Hybrid"), ("DeclRetry", "Declined")]
    t = [r"\begin{table}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{2.5pt}",
         "% Generated by scripts/naacl/v4_analysis.py; do not edit by hand.", r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{lr" + "r" * len(cols) + "}", r"\toprule",
         " & First & " + " & ".join(c for _, c in cols) + r" \\", r"\midrule"]
    groups = [(SHORT[m], [c for c in cells if c.model == m]) for m in MODELS]
    fams = [(name, [c for c in cells if c.ds in dss]) for name, dss in FAMILY]
    for gi, grp in enumerate((groups, fams)):
        if gi:
            t.append(r"\midrule")
        for name, rs in grp:
            if not rs:
                continue
            f0 = 100 * sum(c.keep.sum() for c in rs) / sum(c.n for c in rs)
            # each column on the same cells as its Table 2 row (a control without a finished run drops that cell)
            vals = [GB.pooled([c for c in rs if c in SUBSET.get(k, rs)] or rs, POL[k])[0] for k, _ in cols]
            best = max(vals)
            cells_ = [(r"\textbf{" if abs(v - best) < 1e-9 else "") + f"${v:+.2f}$" + ("}" if abs(v - best) < 1e-9 else "")
                      for v in vals]
            t.append(f"{name} & {f0:.1f} & " + " & ".join(cells_) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Map-Reflect beats both random gates for every model and every family. "
          r"Table~\ref{tab:main} per model and per dataset family: change in accuracy (points) over the first "
          r"pass (First, \%). Random/DS: random gate matched per dataset. All fields: the routed second pass on every "
          r"field. Fresh: fresh mapping over the widened pool. Declined: retry of declined answers only. Best per row in "
          r"bold.}", r"\label{tab:slices}", r"\end{table}"]
    OUT("paper/tab_v4_slices.tex").write_text("\n".join(t) + "\n")


def tab_v4(rows, LABEL):
    order = [("Keep", None), ("Hybrid", None), ("FullList", None), ("ReAsk", None), ("DeclRetry", None),
             ("ReflectAll", None), ("Fresh", None), ("ReflectAllAction", None), ("Random", None), ("RandomDS", None),
             ("Oracle", None), ("Method", None)]
    t = [r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{4pt}",
         "% Generated by scripts/naacl/v4_analysis.py; do not edit by hand.",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Policy & Accuracy & Change [95\% CI] & w/o OpenData & Macro & Calls & Tokens \\", r"\midrule"]
    for k, _ in order:
        a, e, lo, hi, en, em, c, tk = rows[k]
        if k in ("ReflectAll", "Random", "Method"):
            t.append(r"\midrule")
        cs = "--" if c != c else f"{c:.2f}"
        ts = "--" if tk != tk else f"{tk:,.0f}".replace(",", "{,}")
        name = r"\textbf{" + LABEL[k] + "}" if k == "Method" else LABEL[k]
        if k in SUBNOTE:
            name += r"$^{" + "*" * (list(SUBNOTE).index(k) + 1) + "}$"
        t.append(f"{name} & {a:.1f} & ${e:+.2f}$ \\tiny[{lo:+.1f}, {hi:+.1f}] & ${en:+.2f}$ & ${em:+.2f}$ & {cs} & {ts} \\\\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Map-Reflect beats every policy but the oracle and the second pass on every field, which is "
          r"slightly more accurate at about twice the calls. Test accuracy of every policy, four models and ten datasets "
          r"pooled. Change: points over the first "
          r"pass, with 95\% split-group bootstrap intervals; w/o OpenData and Macro (unweighted over datasets) are "
          r"changes too. Every second pass is measured on every field it could route. Calls and tokens: second-pass "
          r"calls and tokens per field (full list and shortlist replace the first pass). Fresh mapping runs on the "
          f"{FRESHCOV[0]:.0f}" r"\% of fields whose widened pool adds candidates to the shown ones; elsewhere it is the first "
          r"pass." +
          "".join(f" {'*' * (i + 1)}On {nn:,} pairs".replace(",", "{,}") + f" ({SUBDESC.get(k, 'a subset')}; first pass "
                  f"{f0:.1f}\\%)." for i, (k, (f0, nn)) in enumerate(SUBNOTE.items())) + "}",
          r"\label{tab:main}", r"\end{table*}"]
    OUT("paper/tab_v4.tex").write_text("\n".join(t) + "\n")



def tab_main_v4(rows, LABEL):
    """paper/tab_main_v4.tex: tab_v4.tex with the structural-router row, for the v4 paper."""
    order = ["Keep", "Hybrid", "FullList", "ReAsk", "DeclRetry", "ReflectAll", "Fresh", "ReflectAllAction", "Random",
             "RandomDS", "RandomStrat", "Oracle", "Structural", "Method"]
    lab = dict(LABEL, Structural="Structural router (no uncertainty)", RandomStrat="Random gate, per structural stratum")
    t = [r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{4pt}",
         "% Generated by scripts/naacl/v4_analysis.py (tab_main_v4); do not edit by hand. Source: docs/naacl/results/v4_test.md.",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Policy & Accuracy & Change [95\% CI] & w/o OpenData & Macro & Calls & Tokens \\", r"\midrule"]
    for k in order:
        a, e, lo, hi, en, em, c, tk = rows[k]
        if k in ("ReflectAll", "Random", "Structural"):
            t.append(r"\midrule")
        cs = "--" if c != c else f"{c:.2f}"
        ts = "--" if tk != tk else f"{tk:,.0f}".replace(",", "{,}")
        name = r"\textbf{" + lab[k] + "}" if k == "Method" else lab[k]
        if k in SUBNOTE:
            name += r"$^{" + "*" * (list(SUBNOTE).index(k) + 1) + "}$"
        t.append(f"{name} & {a:.1f} & ${e:+.2f}$ \\tiny[{lo:+.1f}, {hi:+.1f}] & ${en:+.2f}$ & ${em:+.2f}$ & {cs} & {ts} \\\\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Map-Reflect leads random and declined-only routing, the full list and every other first-pass control. "
          r"A structural router, which sends declined fields and types too large to show, comes within a tenth of a point "
          r"at fewer calls but more tokens. The second pass on every field is slightly more accurate at about twice the "
          r"calls. Test accuracy of every policy, four models and ten datasets pooled. Change: points over the first "
          r"pass, with 95\% split-group bootstrap intervals; w/o OpenData and Macro (unweighted over datasets) are "
          r"changes too. Every second pass is measured on every field it could route. Calls and tokens: second-pass "
          r"calls and tokens per field (full list and shortlist replace the first pass). Fresh mapping runs on the "
          f"{FRESHCOV[0]:.0f}" r"\% of fields whose widened pool adds candidates to the shown ones; elsewhere it is the first "
          r"pass. Structural router: the widen action on declined fields and on fields whose target type has more than "
          r"20 fields, with no uncertainty signal. Random gate per structural stratum: Map-Reflect's routed share within the "
          r"declined, wide-type and shown-type fields of each dataset and model." +
          "".join(f" {'*' * (i + 1)}On {nn:,} pairs".replace(",", "{,}") + f" ({SUBDESC.get(k, 'a subset')}; first pass "
                  f"{f0:.1f}\\%)." for i, (k, (f0, nn)) in enumerate(SUBNOTE.items())) + "}",
          r"\label{tab:main}", r"\end{table*}"]
    OUT("paper/tab_main_v4.tex").write_text("\n".join(t) + "\n")


if __name__ == "__main__":
    main()
