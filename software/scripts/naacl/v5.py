#!/usr/bin/env python3
"""Method v5 (docs/naacl/v6/PLAN_PHASE2.md, "Method v5"): development fit and freeze, then one test scoring.

v5 = method v4.1 + the description repair (ff1) + the improved reflection (symmetric evidence, no old scores, no
uniqueness hint) + the either-order verifier + the benefit router as the uncertainty component. Runs:
results/naacl/v6/v5/{dev,test}/<ds>__<model>.jsonl (scripts/naacl/v5_run.py through run_v5.sh) and, for the current
reflection re-measured under ff1, results/naacl/v6/v5/d3c (OMOP cells) with the D3 rows (results/naacl/v6/d3/dev)
elsewhere, whose prompts the repair does not change.

  --stage fit    development only; it never opens a test output. Writes docs/naacl/results/v5_frozen.json with
                 (a) the selected intervention: on the D3 development population (scripts/naacl/v6_d3.py --stage
                     select) the four combinations {current, improved reflection} x {current check, either-order
                     verifier}, each on the field's routed action, net repairs with paired split-group intervals; the
                     one with the most net repairs is selected (the plan's combination unless a component beats it);
                 (b) the development outcomes of the selected intervention on every field, both actions (a digest);
                 (c) the benefit router: u(x) = P(repair) - P(break) of the routed action under the selected
                     intervention, a multinomial L2 logistic model over the D2 features (v6_d2.py), C by 5-fold
                     split-group cross-validated log loss; its budget is a threshold tau (route when u > tau) chosen on
                     cross-fitted development u to maximise development accuracy, ties to fewer calls;
                 and the margin router as a comparison: v4.1's t_s and widen action kept (the actions test runs), t_m
                     refitted on the v5 development outcomes by the v4.1 grid.
  --stage score  refuses to run unless v5_frozen.json is committed and unchanged. Scores the test split once:
                 keep, v5 (benefit router), margin (refitted t_m), margin at the v4.1 thresholds, struct, union,
                 every field, random and oracle (both at v5's routed share per model), and method v4.1 on its stored
                 outcomes as the reference. Writes docs/naacl/results/v5_facts.json (F-VFive*) and v5_test.md.

Correctness: eval.calc_accuracy.target_correct, full identity, ITOpsGraph v2.1. Intervals: split-group bootstrap
(group_stats.py, B = 2000, seed 20261009), paired across policies.

Usage: ITOPS_V21=<v2.1 label file> .venv/bin/python scripts/naacl/v5.py --stage fit|score
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
os.environ["MAPREFLECT_CANONICAL_MARGIN"] = "1"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["fit", "score"], required=True)
ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--root", default="results/naacl/v6/v5")
ap.add_argument("--partial", action="store_true", help="fit: preview on finished cells; writes nothing")
ap.add_argument("--d1", action="store_true",
                help="D1 (strong first-pass control): the same fit and scoring on the v5 second pass over the hybrid "
                     "top-20 first pass (results/naacl/v6/d1/first, data/v6_d1_pools; runs results/naacl/v6/d1/v5, "
                     "Qwen3.5 and Llama-4). The intervention is v5's frozen one (no selection); the benefit router and "
                     "the margin t_m are refitted on D1 development; the reference is v5 on the stored-order first pass. "
                     "Writes v6_d1_frozen.json, v6_d1_dev_fit.md, v6_d1_facts.json (F-DOne*), v6_d1_test.md")
A = ap.parse_args()
LABELS = os.environ.get("ITOPS_V21") or str(
    ROOT.parent / "schema-mapping-datasets/data/itopsgraph_v2_1/itopsgraph_v2.1_cross_product.jsonl")
sys.argv = sys.argv[:1]
import reflect_v2 as R2  # noqa: E402
import router  # noqa: E402
import group_stats as GS  # noqa: E402
from eval.calc_accuracy import load_gold_file, target_correct  # noqa: E402

V41 = json.load(open(ROOT / "docs/naacl/results/v4_1_frozen.json"))
TH41 = V41["thresholds"]
MODELS = ["llama4", "qwen35"] if A.d1 else ["gptoss", "llama4", "granite", "qwen35"]
DATASETS = V41["datasets"]
# Supplementary material: ITOpsGraph labels may be absent; SUPP_DATASETS (comma separated) restricts loading and
# scoring to the datasets that are present. The frozen model keeps all ten dataset indicators.
GOLD = load_gold_file(LABELS) if Path(LABELS).exists() else {}
SUPP = [d for d in os.environ.get("SUPP_DATASETS", "").split(",") if d]
present = lambda names: [d for d in names if not SUPP or d in SUPP]  # noqa: E731
ITOPS_DUP = {sid: md.get("duplicate_group") for sid, (_, _, md) in GOLD.items()}
SEED = 20261009
RUNS = ROOT / ("results/naacl/v6/d1/v5" if A.d1 else A.root)
TOKENS = ROOT / "results/naacl/v6/d1/tokens" if A.d1 else ROOT / A.root / "tokens"
TOKPRE = "v5" if A.d1 else ""
TAGF = "v6_d1" if A.d1 else "v5"
PREFIX = "F-DOne" if A.d1 else "F-VFive"


def first_slot(ds, m):
    return str(ROOT / (f"results/naacl/v6/d1/first/{ds}__{m}" if A.d1 else f"results/naacl/v2/first/given/{ds}__{m}"))


def pools_of(ds):
    return ROOT / (f"data/v6_d1_pools/{ds}.json" if A.d1 else f"data/v4_pools/{ds}.json")
FROZEN_F = ROOT / f"docs/naacl/results/{TAGF}_frozen.json"
FACTS_F = ROOT / f"docs/naacl/results/{TAGF}_facts.json"
CONT = ["margin", "s1", "top1", "declined", "single", "lognrank", "nshown", "trunc", "logntype", "agree", "margin_trunc"]
VERIFIERS = {"check": "current pairwise check", "either": "either-order verifier"}


def read(f):
    return [json.loads(l) for l in open(f) if l.strip()]


def gold_of(ds, sid, fld, gts):
    if ds != "itopsgraph":
        return gts
    tg, rm, _ = GOLD[sid]
    return None if fld in rm else tg[fld]


HELDOUT = ["mimic_omop_rematch", "semtab_m2kg_r1", "oaei_conference", "bioml_omim_ordo", "schemanet_cprd"]


def ids_of(ds, split):
    man = json.load(open(ROOT / f"data/splits/{ds}.json"))
    if split == "test" and man.get("all_test"):
        return {str(r.get("sample_id")) for r in R2.load_slot(str(ROOT / f"results/naacl/v2/first/given/{ds}__gptoss"))["results"]}
    return set(man["dev"]) if split == "dev" else set(man["test"]) - set(man.get("old_dev_excluded") or [])


def token_means(split):
    acc = defaultdict(lambda: np.zeros(2))
    for f in TOKENS.glob(f"{TOKPRE}{split}__*.jsonl"):
        ds, m = f.stem.split("__")[-2:]
        for r in read(f):
            st = str(r.get("stage") or "")
            kind = "reflect" if st.startswith("reflect") else "verify" if st.startswith("verify") else None
            if kind is None:
                continue
            v = np.array([1, (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)], float)
            acc[(ds, m, kind)] += v
            acc[(m, kind)] += v
    return {k: v[1] / v[0] for k, v in acc.items() if v[0]}


class Cell:
    pass


def load(split, partial=False, datasets=None):
    """Cells per dataset and model: first-pass signals, D2 features and, per action X in {NT, W}, the outcome under the
    either-order verifier (X_v5) and, on development, under the current check (X_chk)."""
    tok = token_means(split)
    cells, problems = [], []
    for m in MODELS:
        for ds in present(datasets or DATASETS):
            f = RUNS / split / f"{ds}__{m}.jsonl"
            if not (f.exists() and Path(str(f) + ".done").exists()):
                if not partial:
                    problems.append(f"missing or unfinished {f}")
                continue
            ids = ids_of(ds, split)
            recs = {r["key"]: r for r in R2.records(R2.load_slot(first_slot(ds, m)), ids)}
            pools = json.load(open(pools_of(ds)))
            rows = {r["key"]: r for r in read(f) if r["sample_id"] in ids}
            if set(rows) != set(recs):
                problems.append(f"{f}: {len(rows)} rows for {len(recs)} fields")
                continue
            out = []
            for k, r in rows.items():
                rec = recs[k]
                if (rec["base"] or "") != r["base"] or abs(rec["margin"] - r["margin"]) > 1e-9:
                    problems.append(f"stale row {k} in {f}")
                    continue
                fld = k.rsplit("||", 1)[1]
                g = gold_of(ds, r["sample_id"], fld, r["gts"])
                if g is None:
                    continue
                ok = lambda p: float(bool(p) and target_correct(p, g))  # noqa: E731
                p = pools[f"{r['sample_id']}||{rec['field']}"]
                sc = [R2._num(c.get("score")) for c in rec["ranking"]]
                hyb = p.get("hybrid20") or []
                x = dict(sid=r["sample_id"], key=k, keep=ok(r["base"]),
                         margin=float(router.canonical_margin(float(r["margin"]))), s1=float(r["s1"]),
                         valid=float(bool(r["valid"])), nrank=float(sum(v != float("-inf") for v in sc)),
                         nshown=float(len(p["shown"])), ntype=float(len(p["shown"]) + len(p["type_rest"])),
                         trunc=float(bool(p["type_rest"])),
                         agree=float(bool(r["base"]) and bool(hyb) and r["base"].lower() == str(hyb[0]).lower()),
                         isnt=float(r["action"] == "NT"))
                for X in ("NT", "W"):
                    has = f"{X}_v5" in r
                    prop = r.get(f"{X}_proposed") if has else None
                    judged = float(bool(prop) and prop != r["base"] and bool(r["base"]))
                    x[f"{X}_has"] = float(has)
                    x[f"{X}_v5"] = ok(r.get(f"{X}_v5")) if has else np.nan
                    x[f"{X}_chk"] = ok(r.get(f"{X}_chk")) if has and r.get(f"{X}_chk") is not None else np.nan
                    x[f"{X}_prop"] = ok(prop or r["base"]) if has else np.nan
                    called = has and not (X == "NT" and not r["valid"]) and not r.get(f"{X}_same_as")
                    x[f"{X}_calls"] = float(called) * (1 + 2 * judged) if has else np.nan
                    x[f"{X}_tok"] = (float(called) * (tok.get((ds, m, "reflect"), tok.get((m, "reflect"), 0.0))
                                     + 2 * judged * tok.get((ds, m, "verify"), tok.get((m, "verify"), 0.0)))
                                     if has else np.nan)
                out.append(x)
            if not out:
                continue
            c = Cell()
            c.ds, c.model, c.n, c.split = ds, m, len(out), split
            c.sid = np.array([x["sid"] for x in out])
            c.keys = [x["key"] for x in out]
            for key in out[0]:
                if key not in ("sid", "key"):
                    setattr(c, key, np.array([x[key] for x in out], float))
            c.validb = c.valid > 0
            c.struct = ((~c.validb) | (c.trunc > 0)).astype(float)
            # the routed action's outcome and cost (NT where router.action says so, else W)
            for s in ("v5", "chk", "prop", "calls", "tok"):
                setattr(c, f"act_{s}", np.where(c.isnt > 0, getattr(c, f"NT_{s}"), getattr(c, f"W_{s}")))
            standalone_costs(c)
            cells.append(c)
    if problems:
        sys.exit(f"refusing v5 {split}:\n  " + "\n  ".join(problems[:30]))
    return cells


def standalone_costs(c):
    """The one place that turns a cell's collection costs into standalone strategy costs.

    Two accountings exist, and both are kept:
      collection  (X_calls, X_tok, act_calls, act_tok, set by load): what the collector spent. A widen action whose
                  candidate list equals NT's (W_same_as: a fully shown type) copies NT's outcome and costs nothing
                  extra, so it is charged zero. The frozen v5 fit (tau's tie-break) and the scored v5 test numbers use
                  this accounting; it stays as it is, so no frozen or scored result changes.
      standalone  (Wc, Wt, Ac, At, set here): what a deployed strategy that sends the field to the widen action pays.
                  A copied widen outcome is charged the NT action's realised calls and tokens, since that reflection
                  (and its verifier calls) must still be made. reuse marks the copied widen outcomes.
    Struct's fields never copy, so its cost is the same in both. Used by v5_matched.py (matched budgets) and
    v5_cost_diff.py (the offline difference between the two accountings)."""
    reuse = (c.W_has > 0) & (np.nan_to_num(c.W_calls) == 0)
    assert not np.any(reuse & ~(np.nan_to_num(c.NT_calls) > 0)), (c.ds, c.model, "a reused widen field without an NT call")
    c.reuse = reuse
    c.Wc = np.where(reuse, c.NT_calls, c.W_calls)
    c.Wt = np.where(reuse, c.NT_tok, c.W_tok)
    nt = c.isnt > 0
    c.Ac = np.where(nt, c.NT_calls, c.Wc)
    c.At = np.where(nt, c.NT_tok, c.Wt)
    # off the copied widen fields the two accountings agree
    assert np.allclose(np.nan_to_num(np.where(reuse & ~nt, 0.0, c.Ac)), np.nan_to_num(np.where(reuse & ~nt, 0.0, c.act_calls)))
    return c


def feats(c):
    single = (c.validb & (c.nrank == 1)).astype(float)
    X = dict(margin=c.margin, s1=c.s1, top1=(c.s1 >= 1.0).astype(float), declined=1.0 - c.valid, single=single,
             lognrank=np.log1p(c.nrank), nshown=c.nshown / 20.0, trunc=c.trunc, logntype=np.log1p(c.ntype),
             agree=c.agree, margin_trunc=c.margin * c.trunc)
    M = np.column_stack([X[k] for k in CONT])
    D = np.zeros((c.n, len(DATASETS)))
    if c.ds in DATASETS:  # a held-out set has no dataset indicator: the core-development model transfers
        D[:, DATASETS.index(c.ds)] = 1
    Mo = np.zeros((c.n, len(MODELS)))
    Mo[:, MODELS.index(c.model)] = 1
    return M, np.hstack([D, Mo])


def u_of(X, model):
    W, b = np.array(model["coef"]), np.array(model["intercept"])
    z = X @ W.T + b
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    cl = list(model["classes"])
    return p[:, cl.index(1)] - p[:, cl.index(-1)]


def label(c, s="v5"):
    o = getattr(c, f"act_{s}")
    return np.where((c.keep == 0) & (o == 1), 1, np.where((c.keep == 1) & (o == 0), -1, 0)).astype(int)


# ============================================================================================ (a) selection
def d3_population():
    """The D3 development fields and their current-reflection outcomes under ff1: D3 rows where the repair changes no
    prompt, the d3c rows (current reflection rerun under ff1) on the OMOP cells."""
    sel = json.load(open(ROOT / "docs/naacl/results/v6_d3_selection.json"))["cells"]
    cur = {}
    for m in MODELS:
        for ds in DATASETS:
            want = set(sel[f"{ds}__{m}"]["errors"]) | set(sel[f"{ds}__{m}"]["correct"])
            if not want:
                continue
            ff = (ROOT / f"data/{ds}_reflect_lookup_ff1.json").exists()
            f = (RUNS / "d3c" / f"{ds}__{m}.jsonl") if ff else (ROOT / f"results/naacl/v6/d3/dev/{ds}__{m}.jsonl")
            got = {r["key"]: r for r in read(f)} if f.exists() else {}
            for k in want:
                cur[(ds, m, k)] = got.get(k)
    return sel, cur


def select(cells):
    sel, cur = d3_population()
    pop = []
    for c in cells:
        cell = sel[f"{c.ds}__{c.model}"]
        want = set(cell["errors"]) | set(cell["correct"])
        mask = np.array([k in want for k in c.keys])
        if not mask.any():
            continue
        x = Cell()
        x.ds, x.model = c.ds, c.model
        x.sid = c.sid[mask]
        x.n = int(mask.sum())
        x.keep = c.keep[mask]
        x.new_chk, x.new_either = c.act_chk[mask], c.act_v5[mask]
        cc, ce, missing = [], [], 0
        for k, keep_ok in zip(np.array(c.keys)[mask], x.keep):
            r = cur.get((c.ds, c.model, k))
            if r is None:
                missing += 1
                cc.append(np.nan)
                ce.append(np.nan)
                continue
            fld = k.rsplit("||", 1)[1]
            g = gold_of(c.ds, r["sample_id"], fld, r["gts"])
            ok = lambda p: float(bool(p) and target_correct(p, g))  # noqa: E731
            cc.append(ok(r.get("cur") or r["base"]))
            prop = r.get("cur_proposed")
            if prop and r["base"] and prop != r["base"]:
                ce.append(ok(prop) if prop in (r.get("blind_c1"), r.get("blind_c2")) else ok(r["base"]))
            else:
                ce.append(ok(r.get("cur") or r["base"]))
        if missing:
            sys.exit(f"{c.ds}__{c.model}: {missing} D3 fields without a current-reflection row")
        x.cur_chk, x.cur_either = np.array(cc), np.array(ce)
        pop.append(x)
    groups = GS.split_groups(present(DATASETS), ITOPS_DUP)
    GB = GS.GroupBootstrap(pop, groups, B=A.B, seed=SEED)
    N = sum(x.n for x in pop)
    res = {}
    for nm in ("cur_chk", "cur_either", "new_chk", "new_either"):
        e, d = GB.pooled(pop, lambda x: getattr(x, nm) - x.keep)
        dv, dd = GB.pooled(pop, lambda x: getattr(x, nm) - x.cur_chk)
        dc, dcd = GB.pooled(pop, lambda x: x.new_either - getattr(x, nm))
        res[nm] = dict(net=round(e * N / 100, 1), net_ci=[round(v * N / 100, 1) for v in GS.interval(d)],
                       minus_current=round(dv * N / 100, 1), minus_current_ci=[round(v * N / 100, 1) for v in GS.interval(dd)],
                       combination_minus_this=round(dc * N / 100, 1),
                       combination_minus_this_ci=[round(v * N / 100, 1) for v in GS.interval(dcd)],
                       repairs=int(sum(((x.keep == 0) & (getattr(x, nm) == 1)).sum() for x in pop)),
                       breaks=int(sum(((x.keep == 1) & (getattr(x, nm) == 0)).sum() for x in pop)))
        omop = [x for x in pop if x.ds.startswith("omop_")]
        res[nm]["net_omop"] = float(sum((getattr(x, nm) - x.keep).sum() for x in omop))
        res[nm]["net_other"] = float(sum((getattr(x, nm) - x.keep).sum() for x in pop if not x.ds.startswith("omop_")))
    best = max(res, key=lambda k: (res[k]["net"], k == "new_either"))
    return dict(population=N, arms=res, selected=best), pop


# ============================================================================================ fit
def route_margin(c, t_m, t_s, s="v5"):
    band = router.bands(c.margin, c.validb, c.s1, t_m, t_s)
    r = (band != router.KEEP).astype(float)
    return np.where(r > 0, getattr(c, f"act_{s}"), c.keep), r


def benefit_fit(cells, s):
    """The benefit model of the fit: features standardised on development, C by 5-fold split-group CV log loss, the
    full-development model and the cross-fitted u that chooses tau. Returns (model, ucf, mu, sd, C, best, y)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    groups = GS.split_groups(present(DATASETS), ITOPS_DUP)
    gkey = [f"{c.ds}::{groups[(c.ds, x)]}" for c in cells for x in c.sid]
    gid = np.unique(gkey, return_inverse=True)[1]
    Ms, Ds = zip(*(feats(c) for c in cells))
    M, Dm = np.vstack(Ms), np.vstack(Ds)
    mu, sd = M.mean(0), M.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    X = np.hstack([(M - mu) / sd, Dm])
    y = np.concatenate([label(c, s) for c in cells])
    best = None
    folds = list(GroupKFold(n_splits=5).split(X, y, gid))
    for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        ll = 0.0
        for tr, te in folds:
            clf = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])
            ll -= np.log(np.clip(p[np.arange(len(te)), np.searchsorted(clf.classes_, y[te])], 1e-12, 1)).sum()
        ll /= len(y)
        if best is None or ll < best[0]:
            best = (ll, C)
    C = best[1]
    clf = LogisticRegression(C=C, max_iter=5000).fit(X, y)
    model = dict(C=C, cv_logloss=best[0], classes=[int(k) for k in clf.classes_], coef=clf.coef_.tolist(),
                 intercept=clf.intercept_.tolist(), n_repair=int((y == 1).sum()), n_break=int((y == -1).sum()), n=int(len(y)))
    # cross-fitted u for the budget: each fold's u from a model fitted without it
    ucf = np.zeros(len(y))
    for tr, te in folds:
        m_ = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr])
        ucf[te] = u_of(X[te], dict(coef=m_.coef_.tolist(), intercept=m_.intercept_.tolist(),
                                   classes=[int(k) for k in m_.classes_]))
    return model, ucf, mu, sd, C, best, y


def tau_grid(ucf, act, keep, calls):
    """(development accuracy, -calls, tau, routed share) for every candidate tau; the fit takes the max, so ties in
    accuracy go to fewer calls under the given cost accounting."""
    taus = np.unique(np.r_[np.quantile(ucf, np.linspace(0, 1, 101)), -1.0, 1.0])
    grid = []
    for tau in taus:
        r = ucf > tau
        grid.append((float(np.where(r, act, keep).mean()), -float(calls[r].sum()), float(tau), float(r.mean())))
    return grid


def fit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    cells = load("dev", partial=A.partial)
    a_sel, pop = select(cells) if not (A.partial or A.d1) else (None, None)
    s = "v5" if (a_sel is None or a_sel["selected"].endswith("either")) else "chk"
    model, ucf, mu, sd, C, best, y = benefit_fit(cells, s)
    keep = np.concatenate([c.keep for c in cells])
    act = np.concatenate([getattr(c, f"act_{s}") for c in cells])
    calls = np.concatenate([c.act_calls for c in cells])
    grid = tau_grid(ucf, act, keep, calls)
    gbest = max(grid)
    tau = gbest[2]
    # margin router (comparison): v4.1's t_s and widen action, t_m refitted on these outcomes by the v4.1 grid
    MR = {}
    for m in MODELS:
        cs = [c for c in cells if c.model == m]
        if not cs:
            continue
        mg = np.concatenate([c.margin for c in cs])
        tms = np.unique(np.r_[0.0, np.round(np.quantile(mg, np.linspace(0, 1, 21)), 3)])
        cand = []
        for t_m in tms:
            okv, cl = [], []
            for c in cs:
                o, r = route_margin(c, t_m, TH41[m]["t_s"], s)
                okv.append(o)
                cl.append(r * c.act_calls)
            cand.append((float(np.concatenate(okv).mean()), -float(np.concatenate(cl).sum()), float(t_m)))
        MR[m] = dict(t_m=max(cand)[2], t_s=TH41[m]["t_s"], widen=TH41[m]["widen"], dev_accuracy=max(cand)[0])
    per = {}
    for m in MODELS:
        cs = [c for c in cells if c.model == m]
        if not cs:
            continue
        ix = np.concatenate([[c.model == m] * c.n for c in cells])
        r = ucf[ix] > tau
        per[m] = dict(n=int(ix.sum()), keep=float(keep[ix].mean()), v5_crossfit=float(np.where(r, act[ix], keep[ix]).mean()),
                      routed_share=float(r.mean()), every_field=float(act[ix].mean()))
    n = int(len(y))
    out = dict(date=str(datetime.date.today()), stage="fit", method="v5", status="exploratory after the v4 freeze",
               spec="docs/naacl/v6/PLAN_PHASE2.md, Method v5",
               flags="MAPREFLECT_V41=1 MAPREFLECT_CANONICAL_MARGIN=1; MAPREFLECT_FIELD_FIRST=1 where an ff1 lookup exists",
               a_selection=a_sel, intervention=dict(reflection="v6_d3_run.D3Prompts.symmetric(scores=False)" if (
                   a_sel is None or a_sel["selected"].startswith("new")) else "current reflection",
                   verifier=VERIFIERS["either" if s == "v5" else "check"], outcome_field=s),
               b_dev_outcomes=dict(fields=n, digest=hashlib.sha256(json.dumps(
                   [[c.ds, c.model, c.n, float(c.keep.sum()), float(np.nansum(c.NT_v5)), float(np.nansum(c.W_v5))]
                    for c in cells]).encode()).hexdigest(), files="results/naacl/v6/v5/dev/<ds>__<model>.jsonl"),
               c_benefit=dict(features=CONT + [f"ds:{d}" for d in DATASETS] + [f"model:{m}" for m in MODELS],
                              standardise=dict(mu=mu.tolist(), sd=sd.tolist()), model=model, tau=tau,
                              tau_rule="route when u > tau; tau maximises cross-fitted development accuracy, ties to fewer calls",
                              dev_crossfit_accuracy=gbest[0], dev_routed_share=gbest[3]),
               margin_router=MR, thresholds_v41=TH41, per_model=per, models=MODELS, datasets=DATASETS,
               labels="ITOpsGraph v2.1", note="Fitted on development outcomes only; no test output was read.")
    L = ["# Method v5: development fit" + (" (partial preview)" if A.partial else ""), "", f"{n:,} development field-model pairs.", ""]
    if a_sel:
        L += ["## (a) Intervention on the D3 development population", "",
              f"{a_sel['population']:,} fields (each on its routed action). Net repairs, paired split-group intervals.", "",
              "| reflection + verifier | net | minus current | combination minus this | repairs | breaks | OMOP net | other net |",
              "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for k, v in a_sel["arms"].items():
            L.append(f"| {k} | {v['net']:+g} {v['net_ci']} | {v['minus_current']:+g} {v['minus_current_ci']} | "
                     f"{v['combination_minus_this']:+g} {v['combination_minus_this_ci']} | {v['repairs']} | {v['breaks']} | "
                     f"{v['net_omop']:+g} | {v['net_other']:+g} |")
        L += ["", f"Selected: {a_sel['selected']}.", ""]
    L += ["## (c) Benefit router", "", f"C = {C}; tau = {tau:.4f}; cross-fitted dev accuracy {100*gbest[0]:.2f}% at a routed "
          f"share of {100*gbest[3]:.1f}%.", "", "| model | fields | keep | v5 (cross-fitted) | every field | routed |",
          "|---|--:|--:|--:|--:|--:|"]
    for m, v in per.items():
        L.append(f"| {m} | {v['n']:,} | {100*v['keep']:.2f} | {100*v['v5_crossfit']:.2f} | {100*v['every_field']:.2f} | "
                 f"{100*v['routed_share']:.1f}% |")
    L += ["", "Margin router (comparison), t_m refitted, v4.1 t_s: " + "; ".join(f"{m} t_m={v['t_m']}" for m, v in MR.items())]
    print("\n".join(L))
    if not A.partial:
        FROZEN_F.write_text(json.dumps(out, indent=1) + "\n")
        (ROOT / f"docs/naacl/results/{TAGF}_dev_fit.md").write_text("\n".join(L) + "\n")


# ============================================================================================ score
FACTS = []


def fact(fid, claim, value, lo=None, hi=None):
    assert fid.isalpha(), fid
    FACTS.append(dict(id=PREFIX + fid, claim=claim, value=str(value), lo=None if lo is None else str(lo),
                      hi=None if hi is None else str(hi), src=f"docs/naacl/results/{TAGF}_facts.json", fn="v5.score",
                      status="exploratory"))


def sg(x, nd=2):
    return f"{x:+.{nd}f}"


def require_frozen_committed():
    rel = str(FROZEN_F.relative_to(ROOT))
    log = subprocess.run(["git", "log", "-1", "--format=%H %cI", "--", rel], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", rel], capture_output=True, text=True).stdout.strip()
    if not log or dirty:
        sys.exit(f"{rel} must be committed and unchanged before scoring (log: {log!r}, status: {dirty!r})")
    return log


def v41_reference(cells):
    """Method v4.1 on its stored test outcomes (results/naacl/v4/test/given, v4.1 routing), aligned to the cells."""
    for c in cells:
        rows = {json.loads(l)["key"]: json.loads(l) for l in open(ROOT / f"results/naacl/v4/test/given/{c.ds}__{c.model}/fields.jsonl")}
        t = TH41[c.model]
        out, orig = [], []
        for k, sid in zip(c.keys, c.sid):
            r = rows[k]
            g = gold_of(c.ds, sid, k.rsplit("||", 1)[1], r["gts"])
            b, a = router.route(float(r["margin"]), bool(r["valid"]), float(r["s1"]), t, ("NT", t["widen"]))
            out.append(float(target_correct(r["base"] if b == router.KEEP else r[a], g)))
            orig.append(float(target_correct(r["base"], g)))
        c.v41 = np.array(out)
        c.orig = np.array(orig)


def score():
    commit = os.environ.get("V5_FROZEN_COMMIT") or require_frozen_committed()
    fz = json.loads(FROZEN_F.read_text())
    s = fz["intervention"]["outcome_field"]
    cells = load("test")
    bm = fz["c_benefit"]
    mu, sd = np.array(bm["standardise"]["mu"]), np.array(bm["standardise"]["sd"])
    v41_reference(cells)
    P = {}
    for c in cells:
        M_, D_ = feats(c)
        u = u_of(np.hstack([(M_ - mu) / sd, D_]), bm["model"])
        r5 = (u > bm["tau"]).astype(float)
        act, cc, tk = getattr(c, f"act_{s}"), c.act_calls, c.act_tok
        wok, wc, wt = getattr(c, f"W_{s}"), c.W_calls, c.W_tok
        mr = fz["margin_router"][c.model]
        _, rm = route_margin(c, mr["t_m"], mr["t_s"], s)
        _, r41 = route_margin(c, TH41[c.model]["t_m"], TH41[c.model]["t_s"], s)
        st = c.struct
        c.r5 = r5
        P[id(c)] = {
            "keep": (c.keep, np.zeros(c.n), np.zeros(c.n), np.zeros(c.n)),
            "v5": (np.where(r5 > 0, act, c.keep), r5 * cc, r5 * tk, r5),
            "margin": (np.where(rm > 0, act, c.keep), rm * cc, rm * tk, rm),
            "margin41": (np.where(r41 > 0, act, c.keep), r41 * cc, r41 * tk, r41),
            "struct": (np.where(st > 0, wok, c.keep), st * wc, st * wt, st),
            "union": (np.where(r5 > 0, act, np.where(st > 0, wok, c.keep)), r5 * cc + (1 - r5) * st * wc,
                      r5 * tk + (1 - r5) * st * wt, np.maximum(r5, st)),
            "every": (act, cc, tk, np.ones(c.n)),
            "v41": (c.v41, np.full(c.n, np.nan), np.full(c.n, np.nan), np.full(c.n, np.nan)),
        }
    for m in MODELS:
        cs = [c for c in cells if c.model == m]
        share = sum(c.r5.sum() for c in cs) / sum(c.n for c in cs)
        tot = sum(np.where(c.keep == 0, getattr(c, f"act_{s}"), 0.0).sum() for c in cs)
        k = share * sum(c.n for c in cs)
        for c in cs:
            act = getattr(c, f"act_{s}")
            P[id(c)]["random"] = (c.keep + share * (act - c.keep), share * c.act_calls, share * c.act_tok, np.full(c.n, share))
            fix = np.where(c.keep == 0, act, 0.0) * (min(1.0, k / tot) if tot else 1.0)
            P[id(c)]["oracle"] = (c.keep + fix, np.full(c.n, np.nan), np.full(c.n, np.nan), np.full(c.n, np.nan))
    groups = GS.split_groups(present(DATASETS), ITOPS_DUP)
    GB = GS.GroupBootstrap(cells, groups, B=A.B, seed=SEED)
    N = sum(c.n for c in cells)
    POL = ["keep", "v5", "margin", "margin41", "struct", "union", "every", "random", "oracle", "v41"]
    NAME = {"keep": "Keep the first answer", "v5": "Map-Reflect v5 (benefit router)", "margin": "v5 intervention, margin router (t_m refitted)",
            "margin41": "v5 intervention, margin router at the v4.1 thresholds", "struct": "v5 intervention, struct router",
            "union": "v5 intervention, benefit router or struct", "every": "v5 intervention on every field",
            "random": "v5 intervention, random at v5's share", "oracle": "Oracle at v5's share",
            "v41": "Method v4.1 (stored outcomes, reference)"}
    TAG = {"keep": "Keep", "v5": "Main", "margin": "Margin", "margin41": "MarginOld", "struct": "Struct", "union": "Union",
           "every": "Every", "random": "Random", "oracle": "Oracle", "v41": "VFourOne"}
    ref = lambda c, k: P[id(c)][k][0]  # noqa: E731
    fact("FrozenCommit", f"commit and time of docs/naacl/results/{TAGF}_frozen.json (fitted on development only)", commit)
    fact("TestFields", ("D1 test field-model pairs (two models, Qwen3.5 and Llama-4; ten core datasets, ITOpsGraph v2.1)"
                        if A.d1 else "v5 test field-model pairs (four models, ten core datasets, ITOpsGraph v2.1)"), f"{N:,}")
    L = [("# D1 on test: v5 second pass after a hybrid top-20 first pass (scored once; exploratory)" if A.d1
          else "# Method v5 on test (scored once)"), "", f"{N:,} field-model pairs. Frozen at {commit}. Every v5 policy uses the "
         f"same measured second pass. Intervals: split-group bootstrap, B = {A.B}.", "",
         "| Policy | accuracy | gain over keep | minus v4.1 | minus struct | routed | calls per field | tokens per field | repairs | breaks |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k in POL:
        acc = 100 * sum(ref(c, k).sum() for c in cells) / N
        g, gd = GB.pooled(cells, lambda c: ref(c, k) - c.keep)
        glo, ghi = GS.interval(gd)
        v, vd = GB.pooled(cells, lambda c: ref(c, k) - c.v41)
        vlo, vhi = GS.interval(vd)
        w, wd = GB.pooled(cells, lambda c: ref(c, k) - ref(c, "struct"))
        wlo, whi = GS.interval(wd)
        calls = sum(np.nansum(P[id(c)][k][1]) for c in cells) / N
        toks = sum(np.nansum(P[id(c)][k][2]) for c in cells) / N
        routed = 100 * sum(np.nansum(P[id(c)][k][3]) for c in cells) / N
        rep = float(sum(np.sum((c.keep == 0) * ref(c, k)) for c in cells))
        brk = float(sum(np.sum((c.keep == 1) * (1 - ref(c, k))) for c in cells))
        T = TAG[k]
        fact(f"{T}Acc", f"test accuracy, {NAME[k]}, %", f"{acc:.2f}")
        if k != "keep":
            fact(f"{T}Gain", f"test, {NAME[k]}: change over the first pass, points", sg(g), sg(glo), sg(ghi))
            fact(f"{T}Repairs", f"test, {NAME[k]}: repairs / breaks", f"{rep:,.0f} / {brk:,.0f}")
        if k not in ("keep", "v41"):
            fact(f"{T}VsVFourOne", f"test, {NAME[k]} minus method v4.1, points (paired)", sg(v), sg(vlo), sg(vhi))
        if k not in ("keep", "oracle", "v41"):
            fact(f"{T}Calls", f"test, {NAME[k]}: second-pass calls per field", f"{calls:.2f}")
            fact(f"{T}Tokens", f"test, {NAME[k]}: second-pass tokens per field", f"{toks:,.0f}")
            fact(f"{T}Routed", f"test, {NAME[k]}: routed share, %", f"{routed:.1f}")
        if k not in ("keep", "struct"):
            fact(f"{T}VsStruct", f"test, {NAME[k]} minus struct, points (paired)", sg(w), sg(wlo), sg(whi))
        L.append(f"| {NAME[k]} | {acc:.2f} | {sg(g)} [{sg(glo)}, {sg(ghi)}] | "
                 f"{'' if k == 'v41' else f'{sg(v)} [{sg(vlo)}, {sg(vhi)}]'} | "
                 f"{'' if k == 'struct' else f'{sg(w)} [{sg(wlo)}, {sg(whi)}]'} | "
                 f"{'' if k in ('oracle', 'v41') else f'{routed:.1f}%'} | {'' if k in ('oracle', 'v41') else f'{calls:.2f}'} | "
                 f"{'' if k in ('oracle', 'v41') else f'{toks:,.0f}'} | {rep:,.0f} | {brk:,.0f} |")
    L += ["", "Per model and per dataset, gain over keep (v5 / v4.1 / struct / every field), points:", ""]
    MT = {"gptoss": "Gptoss", "llama4": "Llama", "granite": "Granite", "qwen35": "Qwen"}
    for grp, sel in [(m, lambda c, m=m: c.model == m) for m in MODELS] + [(d, lambda c, d=d: c.ds == d) for d in present(DATASETS)]:
        cs = [c for c in cells if sel(c)]
        n_ = sum(c.n for c in cs)
        vals = [100 * sum((ref(c, k) - c.keep).sum() for c in cs) / n_ for k in ("v5", "v41", "struct", "every")]
        L.append(f"- {grp} ({n_:,}): " + " / ".join(sg(v) for v in vals))
        if grp in MT:
            fact(f"MainGain{MT[grp]}", f"test, v5 gain over keep, {grp}, points", sg(vals[0]))
            fact(f"MainVsVFourOne{MT[grp]}", f"test, v5 minus v4.1, {grp}, points", sg(vals[0] - vals[1]))
    if A.d1:  # the stored-order first pass on the same fields, against the hybrid one
        e, d = GB.pooled(cells, lambda c: c.keep - c.orig)
        lo, hi = GS.interval(d)
        orig = 100 * sum(c.orig.sum() for c in cells) / N
        fact("StoredOrderAcc", "test accuracy of the stored-order first pass on the D1 fields, %", f"{orig:.2f}")
        fact("HybridMinusStored", "hybrid top-20 first pass minus the stored-order first pass, points", sg(e), sg(lo), sg(hi))
        L += ["", f"First pass: hybrid top-20 {100 * sum(c.keep.sum() for c in cells) / N:.2f}% against the stored-order "
              f"first pass {orig:.2f}% ({sg(e)} [{sg(lo)}, {sg(hi)}]). 'v4.1' is method v4.1 on the stored-order first pass."]
    if not A.d1:
        # held-out sets: the core-development configuration transferred (no held-out development), v4.1 as reference
        ho = [] if A.d1 else load("test", datasets=HELDOUT)
        v41_reference(ho)
        GH = GS.GroupBootstrap(ho, GS.split_groups(present(HELDOUT), None), B=A.B, seed=SEED)
        NH = sum(c.n for c in ho)
        L += ["", f"## Held-out sets ({NH:,} field-model pairs; the core-development configuration transferred)", "",
              "| Policy | accuracy | gain over keep | minus v4.1 | calls per field |", "|---|--:|--:|--:|--:|"]
        for c in ho:
            M_, D_ = feats(c)
            u = u_of(np.hstack([(M_ - mu) / sd, D_]), bm["model"])
            r5 = (u > bm["tau"]).astype(float)
            act = getattr(c, f"act_{s}")
            _, r41 = route_margin(c, TH41[c.model]["t_m"], TH41[c.model]["t_s"], s)
            c.pol = {"keep": (c.keep, np.zeros(c.n)), "v5": (np.where(r5 > 0, act, c.keep), r5 * c.act_calls),
                     "margin41": (np.where(r41 > 0, act, c.keep), r41 * c.act_calls),
                     "struct": (np.where(c.struct > 0, getattr(c, f"W_{s}"), c.keep), c.struct * c.W_calls),
                     "every": (act, c.act_calls), "v41": (c.v41, np.full(c.n, np.nan))}
        for k in ("keep", "v5", "margin41", "struct", "every", "v41"):
            acc = 100 * sum(c.pol[k][0].sum() for c in ho) / NH
            g, gd = GH.pooled(ho, lambda c: c.pol[k][0] - c.keep)
            glo, ghi = GS.interval(gd)
            v, vd = GH.pooled(ho, lambda c: c.pol[k][0] - c.v41)
            vlo, vhi = GS.interval(vd)
            calls = sum(np.nansum(c.pol[k][1]) for c in ho) / NH
            T = "HO" + TAG[k]
            fact(f"{T}Acc", f"held-out test accuracy, {NAME[k]}, %", f"{acc:.2f}")
            if k != "keep":
                fact(f"{T}Gain", f"held-out test, {NAME[k]}: change over the first pass, points", sg(g), sg(glo), sg(ghi))
            if k not in ("keep", "v41"):
                fact(f"{T}VsVFourOne", f"held-out test, {NAME[k]} minus method v4.1, points (paired)", sg(v), sg(vlo), sg(vhi))
                fact(f"{T}Calls", f"held-out test, {NAME[k]}: second-pass calls per field", f"{calls:.2f}")
            L.append(f"| {NAME[k]} | {acc:.2f} | {sg(g)} [{sg(glo)}, {sg(ghi)}] | "
                     f"{'' if k == 'v41' else f'{sg(v)} [{sg(vlo)}, {sg(vhi)}]'} | {'' if k == 'v41' else f'{calls:.2f}'} |")
        for d in present(HELDOUT):
            cs = [c for c in ho if c.ds == d]
            n_ = sum(c.n for c in cs)
            L.append(f"- {d} ({n_:,}): v5 {sg(100 * sum((c.pol['v5'][0] - c.keep).sum() for c in cs) / n_)}, "
                     f"v4.1 {sg(100 * sum((c.v41 - c.keep).sum() for c in cs) / n_)}")
    meta = dict(generated_by="scripts/naacl/v5.py --stage score", frozen=str(FROZEN_F.relative_to(ROOT)),
                frozen_commit=commit, status="exploratory", B=A.B, seed=SEED)
    if SUPP:  # a subset run never overwrites the stored facts
        out = ROOT / "supp_out"
        out.mkdir(exist_ok=True)
        (out / f"{TAGF}_facts_subset.json").write_text(json.dumps(dict(meta, facts=FACTS), indent=1) + "\n")
        (out / f"{TAGF}_test_subset.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return
    FACTS_F.write_text(json.dumps(dict(meta, facts=FACTS), indent=1) + "\n")
    if A.d1:
        L += ["", "Note: the v4.1 row uses the stored-order first pass, so its repairs and breaks are counted against the "
              "hybrid first pass and are not comparable with the other rows; its accuracy and paired differences are."]
    (ROOT / f"docs/naacl/results/{TAGF}_test.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"fit": fit, "score": score}[A.stage]()
