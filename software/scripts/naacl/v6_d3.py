#!/usr/bin/env python3
"""D3 of the v6 plan (Codex ideas 5 and 6): offline audit, development field selection and development scoring.
Development only and exploratory. No test field is ever selected or run; the audit reads stored test outcomes only to
describe the 37 rejected repairs the paper already reports.

  --stage audit   Idea 6, offline. On the routed fields of method v4.1 (frozen thresholds, canonical margins), every
                  proposal the pairwise check judged (a changed answer to a valid first answer), split into rejected
                  repairs, prevented breaks, accepted repairs and accepted breaks, with the signals that might tell them
                  apart: first margin, top score, revised margin, whether the proposal was newly added by widening, the
                  proposal's rank in the first ranking, the action, and the position the check showed the proposal in.
                  Test (the stored v4 outcomes, descriptive) and development. Writes docs/naacl/results/v6_d3_audit.md.
  --stage select  The development fields D3 runs (results/naacl/v6/d3/keys/<ds>__<model>.txt and
                  docs/naacl/results/v6_d3_selection.json): every development first-pass error whose gold is in the pool
                  of its routed action (router.action under the v4.1 thresholds), and a matched sample of correct
                  development fields that the router sends to a second pass (per model and dataset, 0.75 times the
                  number of selected errors, seeded), so breaks are measured too.
  --stage score   Scores the D3 development runs (results/naacl/v6/d3/dev/*.jsonl) and writes
                  docs/naacl/results/v6_d3_dev_facts.json (F-DThree*) and docs/naacl/results/v6_d3_dev.md.

Correctness: eval.calc_accuracy.target_correct, full identity; ITOpsGraph v2.1 labels (removed fields dropped).
Intervals: split-group bootstrap (scripts/naacl/group_stats.py, B = 2000, seed 20261009), paired across arms.

Usage: ITOPS_V21=<v2.1 label file> .venv/bin/python scripts/naacl/v6_d3.py --stage audit|select|score
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
os.environ["MAPREFLECT_CANONICAL_MARGIN"] = "1"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["audit", "select", "score"], required=True)
ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--runs", default="results/naacl/v6/d3/dev")
ap.add_argument("--frac", type=float, default=0.75)
ap.add_argument("--partial", action="store_true", help="score: preview the finished cells only; writes nothing")
A = ap.parse_args()
LABELS = os.environ.get("ITOPS_V21") or str(
    ROOT.parent / "schema-mapping-datasets/data/itopsgraph_v2_1/itopsgraph_v2.1_cross_product.jsonl")
sys.argv = sys.argv[:1]
import reflect_v2 as R2  # noqa: E402
import router  # noqa: E402
import group_stats as GS  # noqa: E402
from eval.calc_accuracy import load_gold_file, target_correct  # noqa: E402
from reflect_v4 import pool_of  # noqa: E402

FROZEN = json.load(open(ROOT / "docs/naacl/results/v4_1_frozen.json"))
TH = FROZEN["thresholds"]
MODELS = ["gptoss", "llama4", "granite", "qwen35"]
DATASETS = FROZEN["datasets"]
GOLD = load_gold_file(LABELS)
SEED = 20261009
KEYDIR = ROOT / "results/naacl/v6/d3/keys"


def gold_of(ds, sid, fld, gts):
    if ds != "itopsgraph":
        return gts
    tg, rm, _ = GOLD[sid]
    return None if fld in rm else tg[fld]


def ids_of(ds, split):
    man = json.load(open(ROOT / f"data/splits/{ds}.json"))
    return set(man["dev"]) if split == "dev" else set(man["test"]) - set(man.get("old_dev_excluded") or [])


def stored_rows(split):
    """Stored v4 rows (every action on every field) with the v4.1 band and action and the v2.1 gold."""
    for m in MODELS:
        t = TH[m]
        for ds in DATASETS:
            pools = {k.split("||", 1)[0] + "||" + k.split("||", 1)[1].lower(): v
                     for k, v in json.load(open(ROOT / f"data/v4_pools/{ds}.json")).items()}
            keep = ids_of(ds, split)
            for l in open(ROOT / f"results/naacl/v4/{split}/given/{ds}__{m}/fields.jsonl"):
                r = json.loads(l)
                if r["sample_id"] not in keep:
                    continue
                fld = r["key"].rsplit("||", 1)[1]
                g = gold_of(ds, r["sample_id"], fld, r["gts"])
                if g is None:
                    continue
                band, act = router.route(float(r["margin"]), bool(r["valid"]), float(r["s1"]), t, ("NT", t["widen"]))
                r.update(ds=ds, model=m, gold=g, band=band, act=act, fld=fld,
                         shown=set(pools.get(f"{r['sample_id']}||{fld}", {}).get("shown", [])))
                yield r


def audit():
    L = ["# D3 audit: what the pairwise check rejects (Codex idea 6, offline)", "",
         "Method v4.1 routing (frozen thresholds, canonical margins) on the stored v4 outcomes. A judged proposal is a "
         "changed answer to a valid first answer from the field's routed action; the check accepts or rejects it. "
         "Test rows are the stored outcomes the paper already reports (descriptive only); development rows are what "
         "D3 designs on. ITOpsGraph v2.1 labels.", ""]
    out = {}
    for split in ("test", "dev"):
        G = defaultdict(list)
        for r in stored_rows(split):
            if r["band"] == router.KEEP:
                continue
            a = r["act"]
            prop = r.get(f"{a}_proposed") or ""
            if not (prop and r["base"] and prop != r["base"]):
                continue
            ok = lambda p: target_correct(p, r["gold"])  # noqa: E731
            bo, po, acc = ok(r["base"]), ok(prop), bool(r.get(f"{a}_check"))
            kind = ("repair" if (po and not bo) else "break" if (bo and not po) else
                    "wrong-to-wrong" if not bo else "correct-to-correct")
            shown = r["shown"]
            # the position the check showed the proposal in: reflect_v2.verify orders (old, new) by md5(key) parity
            pos = 2 if int(hashlib.md5(r["key"].encode()).hexdigest(), 16) % 2 == 0 else 1
            rm = r.get(f"{a}_rmargin")
            G[(kind, acc)].append(dict(margin=router.canonical_margin(float(r["margin"])), s1=float(r["s1"]),
                                       rm=(float(rm) if rm is not None else np.nan), new=float(prop not in shown),
                                       nt=float(a == "NT"), pos2=float(pos == 2), ds=r["ds"], model=r["model"]))
        out[split] = G
        L += [f"## {split}", "", "| group | n | mean margin | mean top score | mean revised margin | proposal added by "
              "widening | NT action | proposal shown as option 2 |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for kind in ("repair", "break", "wrong-to-wrong", "correct-to-correct"):
            for acc in (False, True):
                v = G[(kind, acc)]
                if not v:
                    continue
                f = lambda k: np.nanmean([x[k] for x in v])  # noqa: E731
                L.append(f"| {kind}, {'accepted' if acc else 'rejected'} | {len(v)} | {f('margin'):.3f} | "
                         f"{f('s1'):.3f} | {f('rm'):.3f} | {100*f('new'):.0f}% | {100*f('nt'):.0f}% | "
                         f"{100*f('pos2'):.0f}% |")
        rr, pb = G[("repair", False)], G[("break", False)]
        L += ["", f"Rejected repairs by model: {dict(Counter(x['model'] for x in rr))}; by dataset: "
              f"{dict(Counter(x['ds'] for x in rr))}.",
              f"Prevented breaks by model: {dict(Counter(x['model'] for x in pb))}; by dataset: "
              f"{dict(Counter(x['ds'] for x in pb))}.", ""]
    (ROOT / "docs/naacl/results/v6_d3_audit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return out


def select():
    """Every development error whose gold is in its routed action's pool, and a matched sample of routed correct fields."""
    KEYDIR.mkdir(parents=True, exist_ok=True)
    sel, summary = {}, {}
    for m in MODELS:
        t = TH[m]
        for ds in DATASETS:
            d = R2.load_slot(str(ROOT / f"results/naacl/v2/first/given/{ds}__{m}"))
            pools = json.load(open(ROOT / f"data/v4_pools/{ds}.json"))
            P = R2.Prompts("x", {})
            err, cor = [], []
            for rec in R2.records(d, ids_of(ds, "dev")):
                fld = rec["key"].rsplit("||", 1)[1]
                g = gold_of(ds, rec["sample_id"], fld, rec["gts"])
                if g is None:
                    continue
                sc = sorted((R2._num(c.get("score")) for c in rec["ranking"]), reverse=True)
                s1 = sc[0] if sc and sc[0] != float("-inf") else 0.0
                band, act = router.route(rec["margin"], rec["valid"], s1, t, ("NT", t["widen"]))
                cl = pool_of(act, rec, pools)
                pool = sorted(dict.fromkeys(cl)) if cl is not None else P.cands(rec)
                ok = lambda p: target_correct(p, g)  # noqa: E731
                if not ok(rec["base"]):
                    if act == "NT" and not rec["valid"]:
                        continue
                    if any(ok(c) for c in pool):
                        err.append(rec["key"])
                elif band != router.KEEP:
                    cor.append(rec["key"])
            k = min(len(cor), int(round(A.frac * len(err))))
            rng = np.random.default_rng(int(hashlib.md5(f"{SEED}|{ds}|{m}".encode()).hexdigest()[:8], 16))
            pick = sorted(rng.choice(sorted(cor), size=k, replace=False).tolist()) if k else []
            sel[f"{ds}__{m}"] = dict(errors=sorted(err), correct=pick, correct_available=len(cor))
            (KEYDIR / f"{ds}__{m}.txt").write_text("\n".join(sorted(err) + pick) + ("\n" if err or pick else ""))
            summary[f"{ds}__{m}"] = (len(err), len(pick), len(cor))
    tot = np.array(list(summary.values())).sum(0)
    meta = dict(stage="select", date=str(__import__("datetime").date.today()), split="dev", method="v4.1",
                frac=A.frac, seed=SEED, labels="ITOpsGraph v2.1", totals=dict(errors=int(tot[0]), correct=int(tot[1]),
                                                                               correct_available=int(tot[2])),
                cells=sel)
    (ROOT / "docs/naacl/results/v6_d3_selection.json").write_text(json.dumps(meta, indent=1) + "\n")
    for k, v in summary.items():
        print(k, v)
    print("total errors / correct sample / routed correct available:", tot)


# ============================================================================================ scoring
ARMS = ["cur", "symS", "symN"]


def score():
    sel = json.load(open(ROOT / "docs/naacl/results/v6_d3_selection.json"))
    rows = []
    for m in MODELS:
        for ds in DATASETS:
            f = ROOT / A.runs / f"{ds}__{m}.jsonl"
            cell = sel["cells"][f"{ds}__{m}"]
            want = set(cell["errors"]) | set(cell["correct"])
            if not want:
                continue
            got = {}
            if f.exists():
                for l in open(f):
                    r = json.loads(l)
                    got[r["key"]] = r
            if set(got) != want:
                if A.partial:
                    continue
                sys.exit(f"{f}: {len(set(got) & want)} of {len(want)} selected fields scored")
            w_cor = cell["correct_available"] / max(len(cell["correct"]), 1)
            for k in sorted(want):
                r = got[k]
                fld = k.rsplit("||", 1)[1]
                g = gold_of(ds, r["sample_id"], fld, r["gts"])
                r.update(_gold=g, _err=k in set(cell["errors"]), _w=1.0 if k in set(cell["errors"]) else w_cor)
                rows.append(r)
    groups = GS.split_groups(DATASETS, {sid: md.get("duplicate_group") for sid, (_, _, md) in GOLD.items()})

    class Run:
        pass
    by = defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["model"])].append(r)
    runs = []
    for (ds, m), rs in by.items():
        x = Run()
        x.ds, x.model, x.n, x.rows = ds, m, len(rs), rs
        x.sid = np.array([r["sample_id"] for r in rs])
        ok = lambda p, r: float(target_correct(p, r["_gold"]))  # noqa: E731
        x.base = np.array([ok(r["base"], r) for r in rs])
        x.err = np.array([r["_err"] for r in rs])
        x.w = np.array([r["_w"] for r in rs])
        x.routed = np.array([r["band"] != router.KEEP for r in rs])
        for a in ARMS:
            setattr(x, a, np.array([ok(r.get(a) or r["base"], r) for r in rs]))
            setattr(x, a + "_p", np.array([ok((r.get(a + "_proposed") or r["base"]), r) for r in rs]))
            setattr(x, a + "_calls", np.array([0.0 if (r["action"] == "NT" and not r["valid"]) else
                                               1.0 + float(r.get(a + "_check") is not None and bool(r["base"]))
                                               for r in rs]))
        # blind verifier on cur's judged proposals
        fired = np.array([bool(r.get("cur_proposed")) and bool(r["base"]) and r.get("cur_proposed") != r["base"]
                          for r in rs])
        x.fired = fired
        pick = lambda r, c: r.get(c) == r.get("cur_proposed")  # noqa: E731
        c1 = np.array([pick(r, "blind_c1") for r in rs])
        c2 = np.array([pick(r, "blind_c2") for r in rs])
        coin = np.array([int(hashlib.md5((r["key"] + "|d3-order").encode()).hexdigest(), 16) % 2 for r in rs])
        hashpos = np.array([int(hashlib.md5(r["key"].encode()).hexdigest(), 16) % 2 == 0 for r in rs])  # cur: new is option 2
        acc_rules = {"check": np.array([bool(r.get("cur_check")) for r in rs]),
                     "blind_random": np.where(coin == 0, c1, c2),
                     "blind_same_order": np.where(hashpos, c1, c2),
                     "blind_both": c1 & c2, "blind_either": c1 | c2}
        prop_ok = x.cur_p
        x.acc = {}
        for nm, accv in acc_rules.items():
            fin = np.where(fired, np.where(accv, prop_ok, x.base), x.cur)
            x.acc[nm] = fin
        runs.append(x)
    GB = GS.GroupBootstrap(runs, groups, B=A.B, seed=SEED)
    N = sum(x.n for x in runs)
    facts = []

    def F(name, desc, val, lo=None, hi=None):
        assert name.isalpha(), name
        facts.append(dict(id=f"F-DThree{name}", claim=f"development (exploratory): {desc}", value=str(val),
                          lo=None if lo is None else str(lo), hi=None if hi is None else str(hi),
                          src="docs/naacl/results/v6_d3_dev_facts.json", fn="v6_d3.score", status="exploratory"))

    def cnt(fn, mask=None):
        """(estimate, lo, hi) of a per-field count summed over the population, from the bootstrap of its mean."""
        e, d = GB.pooled(runs, lambda x: fn(x) * (mask(x) if mask else 1.0))
        lo, hi = GS.interval(d)
        return round(e * N / 100, 1), round(lo * N / 100, 1), round(hi * N / 100, 1)

    L = ["# D3 development results (Codex ideas 5 and 6; exploratory, development only)", "",
         f"{N:,} development field-model pairs: {int(sum(x.err.sum() for x in runs)):,} first-pass errors whose gold is "
         f"in the routed action's pool, and {int(sum((~x.err).sum() for x in runs)):,} routed correct fields (a matched "
         "sample). Method v4.1 code. Counts summed over the population; intervals: split-group bootstrap, B = "
         f"{A.B}, paired across arms.", ""]
    # ---------------- idea 5: reflection prompts
    L += ["## Idea 5: symmetric-evidence reflection", "",
          "| arm | repairs (errors) | breaks (correct) | net | net minus cur | proposals that repair | proposals that break | calls |",
          "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for a in ARMS:
        rep = cnt(lambda x: getattr(x, a) * (1 - x.base))
        brk = cnt(lambda x: (1 - getattr(x, a)) * x.base)
        net = cnt(lambda x: getattr(x, a) - x.base)
        dv = cnt(lambda x: getattr(x, a) - x.cur)
        prep = cnt(lambda x: getattr(x, a + "_p") * (1 - x.base))
        pbrk = cnt(lambda x: (1 - getattr(x, a + "_p")) * x.base)
        calls = float(sum(getattr(x, a + "_calls").sum() for x in runs))
        tag = {"cur": "Cur", "symS": "SymScores", "symN": "SymNoScores"}[a]
        F(f"{tag}Repairs", f"D3 idea 5, arm {a}: repairs on development errors with gold in pool (after the check)", *rep)
        F(f"{tag}Breaks", f"D3 idea 5, arm {a}: breaks on the routed correct sample (after the check)", *brk)
        F(f"{tag}Net", f"D3 idea 5, arm {a}: net repairs (repairs minus breaks) on the D3 population", *net)
        if a != "cur":
            F(f"{tag}NetVsCur", f"D3 idea 5, arm {a} minus the current reflection: net repairs, paired", *dv)
        F(f"{tag}ProposedRepairs", f"D3 idea 5, arm {a}: proposals that would repair, before the check", *prep)
        F(f"{tag}ProposedBreaks", f"D3 idea 5, arm {a}: proposals that would break, before the check", *pbrk)
        F(f"{tag}Calls", f"D3 idea 5, arm {a}: reflection and check calls", calls)
        fmt = lambda v: f"{v[0]:g} [{v[1]:g}, {v[2]:g}]"  # noqa: E731
        L.append(f"| {a} | {fmt(rep)} | {fmt(brk)} | {fmt(net)} | {'ref.' if a == 'cur' else fmt(dv)} | {fmt(prep)} | "
                 f"{fmt(pbrk)} | {calls:,.0f} |")
    # per model
    L += ["", "Per model, net repairs (cur / symS / symN):", ""]
    for m in MODELS:
        rs = [x for x in runs if x.model == m]
        L.append(f"- {m}: " + " / ".join(f"{sum((getattr(x, a) - x.base).sum() for x in rs):+.0f}" for a in ARMS))
    # the OMOP cells, affected by reflect_v2.Prompts.desc truncating every description to 300 characters
    L += ["", "Split by the 300-character description truncation (reflect_v2.Prompts.desc): the OMOP cells (OMOP-CMS, "
          "OMOP-MIMIC, OMOP-Synthea) are affected; MIMIC-OMOP is a held-out set and not in D3. Net repairs "
          "(cur / symS / symN):", ""]
    for lab, tg, sel in (("OMOP (affected)", "Omop", lambda x: x.ds.startswith("omop_")),
                         ("other datasets", "Other", lambda x: not x.ds.startswith("omop_"))):
        rs = [x for x in runs if sel(x)]
        nf_ = sum(x.n for x in rs)
        vals = [sum((getattr(x, a) - x.base).sum() for x in rs) for a in ARMS]
        L.append(f"- {lab} ({nf_} fields): " + " / ".join(f"{v:+.0f}" for v in vals))
        for a, v in zip(ARMS, vals):
            F({"cur": "Cur", "symS": "SymScores", "symN": "SymNoScores"}[a] + "Net" + tg,
              f"D3 idea 5, arm {a}: net repairs on {lab}", f"{v:+.0f}")
        F("Fields" + tg, f"D3 population fields on {lab}", nf_)
    # routed-population view (weights the correct sample back to every routed correct field)
    L += ["", "Weighted to the routed development population (correct sample weighted by its inverse sampling "
          "fraction; errors restricted to routed fields):", ""]
    for a in ARMS:
        v = sum(((getattr(x, a) - x.base) * x.w * x.routed).sum() for x in runs)
        L.append(f"- {a}: net {v:+.1f}")
        F({"cur": "Cur", "symS": "SymScores", "symN": "SymNoScores"}[a] + "NetRoutedWeighted",
          f"D3 idea 5, arm {a}: net repairs weighted to all routed development fields", round(float(v), 1))
    # ---------------- idea 6: verifier
    nf = int(sum(x.fired.sum() for x in runs))
    L += ["", "## Idea 6: order-balanced verifier with symmetric evidence", "",
          f"{nf} judged proposals of the current reflection (cur). Each rule accepts or rejects the same proposals.", "",
          "| rule | repairs kept | rejected repairs | breaks let through | prevented breaks | net | net minus check |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for nm in ("check", "blind_random", "blind_same_order", "blind_both", "blind_either"):
        fm = lambda x: x.fired.astype(float)  # noqa: E731
        rk = cnt(lambda x: x.acc[nm] * (1 - x.base) * x.cur_p, fm)
        rr = cnt(lambda x: (1 - x.acc[nm]) * (1 - x.base) * x.cur_p, fm)
        bt = cnt(lambda x: (1 - x.acc[nm]) * x.base * (1 - x.cur_p), fm)
        pb = cnt(lambda x: x.acc[nm] * x.base * (1 - x.cur_p), fm)
        net = cnt(lambda x: x.acc[nm] - x.base, fm)
        dv = cnt(lambda x: x.acc[nm] - x.acc["check"], fm)
        tag = "".join(p.capitalize() for p in nm.split("_"))
        F(f"Ver{tag}RepairsKept", f"D3 idea 6, rule {nm}: repairs accepted among cur's judged proposals", *rk)
        F(f"Ver{tag}RepairsRejected", f"D3 idea 6, rule {nm}: repairs rejected", *rr)
        F(f"Ver{tag}BreaksLetThrough", f"D3 idea 6, rule {nm}: breaks accepted", *bt)
        F(f"Ver{tag}BreaksPrevented", f"D3 idea 6, rule {nm}: breaks rejected", *pb)
        F(f"Ver{tag}Net", f"D3 idea 6, rule {nm}: net repairs among judged proposals", *net)
        if nm != "check":
            F(f"Ver{tag}NetVsCheck", f"D3 idea 6, rule {nm} minus the current check, paired", *dv)
        fmt = lambda v: f"{v[0]:g} [{v[1]:g}, {v[2]:g}]"  # noqa: E731
        L.append(f"| {nm} | {fmt(rk)} | {fmt(rr)} | {fmt(bt)} | {fmt(pb)} | {fmt(net)} | "
                 f"{'ref.' if nm == 'check' else fmt(dv)} |")
    L += ["", "Verifier rules on the OMOP cells (affected by the truncation) and the rest, net minus check:", ""]
    for lab, tg, sel in (("OMOP (affected)", "Omop", lambda x: x.ds.startswith("omop_")),
                         ("other datasets", "Other", lambda x: not x.ds.startswith("omop_"))):
        rs = [x for x in runs if sel(x)]
        L.append(f"- {lab} ({int(sum(x.fired.sum() for x in rs))} judged): " + "; ".join(
            f"{nm} {sum(((x.acc[nm] - x.acc['check']) * x.fired).sum() for x in rs):+.0f}"
            for nm in ("blind_random", "blind_same_order", "blind_both", "blind_either")))
    blind_calls = 2 * nf
    F("VerBlindCalls", "D3 idea 6: blind verifier calls (two orders per judged proposal)", blind_calls)
    meta = dict(generated_by="scripts/naacl/v6_d3.py --stage score", date=str(__import__("datetime").date.today()),
                population=N, status="exploratory", split="dev", base_method="v4.1", runs=A.runs,
                selection="docs/naacl/results/v6_d3_selection.json", B=A.B, seed=SEED)
    if not A.partial:
        (ROOT / "docs/naacl/results/v6_d3_dev_facts.json").write_text(json.dumps(dict(meta, facts=facts), indent=1) + "\n")
        (ROOT / "docs/naacl/results/v6_d3_dev.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"audit": audit, "select": select, "score": score}[A.stage]()
