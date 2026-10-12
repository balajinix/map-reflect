#!/usr/bin/env python3
"""Task 1b: the reflection strategies (interventions table and chart) on Map-Reflect v5's test fields and selection.
No model calls; exploratory.

Strategies, each applied to a field means its answer replaces the first answer:
  Doubt, SelfRefine, Neutral, NeutralCheck, Evidence, NeutralEvidenceCheck   the six reflect_v2 intervention arms
      (naive, self_refine, A, AC, B, BC) over the first ranking's candidates
  Widen   v5's own second pass with widening (neutral reflection with symmetric evidence and the either-order check,
          over the widened pool, the W action), measured on every v5 test field
  Routed  v5's second pass with the action its router names (NT or W), every field
Inputs of the six arms (ff1_rerun_keys / the task 1b estimate: prompts rendered with v5's flags and lookups and
compared field by field):
  OMOP-CMS, OMOP-MIMIC, OMOP-Synthea  the R-X1 ff1 rows (results/naacl/v6/ff1rr/merged/x1): v5's prompts exactly
  ToolBench                           the stored rows: v5's prompts are byte-identical
  ITOpsGraph, NL2GQL, Valentine x4    the stored protocol v2 rows: without the ff1 type text (ITOpsGraph) or the v4.1
                                      source metadata and target name (NL2GQL, Valentine)
  every arm row                       read with the protocol v2 JSON parser (v5 reads robustly); no raw responses
A cell whose arm rows are missing for a field keeps the first answer there and is counted.

Selections (the first answer is kept elsewhere):
  Every  every field
  Rate   v5's own selection: the frozen benefit model u(x) above its frozen tau (docs/naacl/results/v5_frozen.json)
  TopU   the top 30% of each cell's fields by u(x)
  and, for Rate and TopU, a random selection of the same share in each cell (in expectation) and the paired difference
Gains are against the first pass on the v5 test fields (four models, ten core datasets, ITOpsGraph v2.1), pooled over
field-model pairs, with v5.py's split-group bootstrap (B = 2000, seed 20261009).

Task 1b rerun (scripts/naacl/v6_calls/run_strat5.sh, results/naacl/v6/strat5): once a cell is done its rerun rows
replace the arms there; the values before it are kept as F-VFiveStratPreVFive*, and F-VFiveStratCellsOn* name the
cells on v5's inputs and those still on stored arms.

Writes docs/naacl/results/v5_strategies_facts.json (F-VFiveStrat*).
Usage: ITOPS_V21=<v2.1 label file> .venv/bin/python scripts/naacl/v5_strategies.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.argv = [sys.argv[0], "--stage", "score"]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
import v5 as V  # noqa: E402  (its argument parser reads the argv above; nothing runs on import)

ARMS = [("naive", "Doubt"), ("self_refine", "SelfRefine"), ("A", "Neutral"), ("AC", "NeutralCheck"), ("B", "Evidence"),
        ("BC", "NeutralEvidenceCheck")]
OMOP = {"omop_cms", "omop_mimic", "omop_synthea"}
INPUT = {"omop_cms": "v5 prompts (R-X1 ff1 rerun)", "omop_mimic": "v5 prompts (R-X1 ff1 rerun)",
         "omop_synthea": "v5 prompts (R-X1 ff1 rerun)", "toolbench": "v5 prompts (stored rows, byte-identical)",
         "itopsgraph": "stored protocol v2 inputs (no ff1 type text)",
         "nl2gql": "stored protocol v2 inputs (no v4.1 source metadata or target name)"}
TOP = 0.30
facts = []


def F(fid, claim, value, lo=None, hi=None):
    f = dict(id=f"F-VFiveStrat{fid}", claim=f"test (exploratory): {claim}", value=value,
             src="docs/naacl/results/v5_strategies_facts.json", fn="v5_strategies", status="exploratory")
    if lo is not None:
        f.update(lo=lo, hi=hi)
    facts.append(f)


def strat5_done(ds, m):
    f = ROOT / f"results/naacl/v6/strat5/{ds}__{m}/fields.jsonl"
    return f if Path(str(f) + ".done").exists() else None


MODELS4 = ["gptoss", "granite", "llama4", "qwen35"]


def ffx1(ds, m):
    return ROOT / f"results/naacl/v6/ff1rr/merged/x1/{ds}__{m}/fields.jsonl"


def complete(ds):
    """A dataset uses rerun arms only when every model's rerun is in (DECISIONS: rerun and stored arms never mix
    within a dataset): the R-X1 ff1 rows on OMOP, the task 1b rerun on Valentine, NL2GQL and ITOpsGraph."""
    if ds in OMOP:
        return all(ffx1(ds, m).exists() for m in MODELS4)
    return all(strat5_done(ds, m) for m in MODELS4)


def arm_rows(ds, m, version):
    """(rows by key, source label). version "pre": the existing arms; "v5": the task 1b rerun on v5's inputs replaces
    the rerun fields (the others kept byte-identical prompts) of a dataset whose four models are all done; "cell":
    the same per cell, for the exploratory per-dataset facts."""
    use = complete(ds) if version == "v5" else version == "cell"
    f = ffx1(ds, m) if ds in OMOP and use else None
    src = "v5 prompts (R-X1 ff1 rerun)"
    if f is None or not f.exists():
        f = ROOT / f"results/naacl/v2/reflect/given/{ds}__{m}/fields.jsonl"
        src = INPUT.get(ds, "stored protocol v2 inputs (no v4.1 source metadata or target name)") if ds not in OMOP \
            else "stored protocol v2 inputs (the R-X1 ff1 rerun is not in for all four models)"
    rows = {r["key"]: r for r in V.read(f)} if f.exists() else {}
    s5 = strat5_done(ds, m) if use and ds not in OMOP else None
    if s5:
        new = {r["key"]: r for r in V.read(s5)}
        rows.update(new)
        src = (f"v5 inputs (task 1b rerun, {len(new)} fields; the rest byte-identical prompts)" if ds == "itopsgraph"
               else "v5 inputs (task 1b rerun)")
    return rows, src


def fill(c, version, missing, sources):
    rows, src = arm_rows(c.ds, c.model, version)
    sources[(c.ds, c.model)] = src
    gold = {}
    c.arm = {}
    for raw, tg in ARMS:
        ok = c.keep.copy()
        miss = 0
        for i, key in enumerate(c.keys):
            q = rows.get(key)
            if q is None or raw not in q:
                miss += 1
                continue
            if key not in gold:
                gold[key] = V.gold_of(c.ds, q["sample_id"], key.rsplit("||", 1)[1], q["gts"])
            g = gold[key]
            ok[i] = float(bool(q[raw]) and g is not None and V.target_correct(q[raw], g)) if q[raw] else 0.0
        c.arm[tg] = ok
        missing[(c.ds, c.model, tg)] = miss
    c.arm["Widen"] = np.where(c.W_has > 0, c.W_v5, c.keep)
    c.arm["Routed"] = c.act_v5


def score(cells, GB, pre, label):
    N = sum(c.n for c in cells)
    for nm, sel in (("Every", None), ("Rate", "sel_rate"), ("TopU", "sel_top")):
        for tg in [t for _, t in ARMS] + (["Widen", "Routed"] if not pre else []):
            def vec(c, tg=tg, sel=sel):
                a = c.arm[tg]
                return (a if sel is None else np.where(getattr(c, sel) > 0, a, c.keep)) - c.keep
            g, gd = GB.pooled(cells, vec)
            lo, hi = V.GS.interval(gd)
            acc = 100 * sum((vec(c) + c.keep).sum() for c in cells) / N
            F(f"{pre}{tg}{nm}Gain", f"{label}strategy {tg}, selection {nm}: change over the first pass, points", V.sg(g),
              V.sg(lo), V.sg(hi))
            F(f"{pre}{tg}{nm}Acc", f"{label}strategy {tg}, selection {nm}: field accuracy, %", f"{acc:.2f}")
            if sel is not None:  # a random selection of the same share in each cell, in expectation, and the paired gap
                def rnd(c, tg=tg, sel=sel):
                    return getattr(c, sel).mean() * (c.arm[tg] - c.keep)
                r, rd = GB.pooled(cells, rnd)
                rlo, rhi = V.GS.interval(rd)
                F(f"{pre}{tg}{nm}RandomGain", f"{label}strategy {tg}, a random selection of the {nm} share in each cell "
                  "(expected): change over the first pass, points", V.sg(r), V.sg(rlo), V.sg(rhi))
                d, dd = GB.pooled(cells, lambda c, tg=tg, sel=sel: vec(c) - rnd(c))
                dlo, dhi = V.GS.interval(dd)
                F(f"{pre}{tg}{nm}MinusRandom", f"{label}strategy {tg}: selection {nm} minus the random selection of its "
                  "share, points", V.sg(d), V.sg(dlo), V.sg(dhi))
            if sel is None:
                rep = sum(float(np.sum((c.keep == 0) * c.arm[tg])) for c in cells)
                brk = sum(float(np.sum((c.keep == 1) * (1 - c.arm[tg]))) for c in cells)
                F(f"{pre}{tg}{nm}Repairs", f"{label}strategy {tg}, every field: repairs / breaks", f"{rep:,.0f} / {brk:,.0f}")


def main():
    fz = json.loads(V.FROZEN_F.read_text())
    bm = fz["c_benefit"]
    mu, sd = np.array(bm["standardise"]["mu"]), np.array(bm["standardise"]["sd"])
    cells = V.load("test")
    for c in cells:
        M_, D_ = V.feats(c)
        u = V.u_of(np.hstack([(M_ - mu) / sd, D_]), bm["model"])
        c.sel_rate = (u > bm["tau"]).astype(float)
        k = int(round(TOP * c.n))
        order = np.argsort(-u, kind="mergesort")
        c.sel_top = np.zeros(c.n)
        c.sel_top[order[:k]] = 1.0
    groups = V.GS.split_groups(V.DATASETS, V.ITOPS_DUP)
    GB = V.GS.GroupBootstrap(cells, groups, B=V.A.B, seed=V.SEED)
    N = sum(c.n for c in cells)
    F("Fields", "v5 test field-model pairs (four models, ten core datasets)", f"{N:,}")
    F("RateShare", "fields selected by v5's frozen benefit model (u above tau), %",
      f"{100 * sum(c.sel_rate.sum() for c in cells) / N:.1f}")
    F("TopShare", "fields selected as each cell's top 30% by u(x), %", f"{100 * sum(c.sel_top.sum() for c in cells) / N:.1f}")
    srcs = {}
    for version, pre, label in (("pre", "PreVFive", "the existing arms (before the task 1b rerun): "), ("v5", "", "")):
        missing, sources = {}, {}
        for c in cells:
            fill(c, version, missing, sources)
        score(cells, GB, pre, label)
        F(f"{pre}ArmMissing", f"{label}arm outcomes missing for a v5 test field (first answer kept), over arms and cells",
          str(sum(missing.values())))
        srcs[version] = sources
    # exploratory: datasets whose rerun is in for some models only, rerun against stored arms on those models
    for ds in sorted(set(c.ds for c in cells)):
        if complete(ds) or ds == "toolbench":
            continue
        done = [m for m in MODELS4 if (ffx1(ds, m).exists() if ds in OMOP else strat5_done(ds, m))]
        if not done:
            continue
        sub = [c for c in cells if c.ds == ds and c.model in done]
        tag = "".join(x.capitalize() for x in ds.split("_"))
        res = {}
        for version in ("pre", "cell"):
            for c in sub:
                fill(c, version, {}, {})
            res[version] = {tg: 100 * sum((c.arm[tg] - c.keep).sum() for c in sub) / sum(c.n for c in sub)
                            for _, tg in ARMS}
        for _, tg in ARMS:
            F(f"Partial{tag}{tg}EveryGain", f"{ds}, models {', '.join(done)} only (not in the table: its rerun is not "
              f"in for all four models): strategy {tg} on every field, rerun on v5's inputs; stored arms in brackets, "
              "points", f"{res['cell'][tg]:+.2f} ({res['pre'][tg]:+.2f})")
    cov = srcs["v5"]
    on = sorted(f"{d}__{m}" for (d, m), s in cov.items() if s.startswith("v5"))
    off = sorted(f"{d}__{m}" for (d, m), s in cov.items() if not s.startswith("v5"))
    F("CellsOnVFiveInputs", "dataset-model cells whose arms are on v5's inputs", f"{len(on)} of {len(cov)}: " + ", ".join(on))
    F("CellsOnStoredArms", "dataset-model cells still on stored arms, with their input difference",
      "; ".join(f"{k}: {cov[tuple(k.split('__'))]}" for k in off) or "none")
    for c in cells:
        if c.ds == "itopsgraph" and strat5_done(c.ds, c.model):
            kf = ROOT / f"results/naacl/v6/strat5/keys/itopsgraph__{c.model}.txt"
            kk = {l.strip() for l in open(kf) if l.strip()}
            F("ITopsRerun" + {"gptoss": "GptOss", "granite": "Granite", "llama4": "LlamaFour", "qwen35": "QwenThreeFive"}[c.model], f"ITOpsGraph, {c.model}: scored fields rerun on v5's inputs (the rest keep "
              "stored rows, their prompts byte-identical)", f"{len(kk & set(c.keys))} of {c.n}")
    F("ArmParser", "JSON parser of the arm outcomes", "v5's robust parser on the task 1b rerun cells; the protocol v2 "
      "parser on every other cell (R-X1 and stored rows; no raw responses to re-read)")
    out = dict(generated_by="scripts/naacl/v5_strategies.py", status="exploratory", split="test",
               frozen="docs/naacl/results/v5_frozen.json", B=V.A.B, seed=V.SEED, top_share=TOP,
               arm_sources={f"{d}__{m}": s for (d, m), s in cov.items()},
               arm_sources_pre={f"{d}__{m}": s for (d, m), s in srcs["pre"].items()}, facts=facts)
    (ROOT / "docs/naacl/results/v5_strategies_facts.json").write_text(json.dumps(out, indent=1) + "\n")
    for f in facts:
        print(f["id"], f["value"], f.get("lo", ""), f.get("hi", ""))


if __name__ == "__main__":
    main()
