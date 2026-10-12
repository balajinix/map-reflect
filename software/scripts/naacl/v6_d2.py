#!/usr/bin/env python3
"""D2 of the v6 plan (docs/naacl/v6/PLAN_PHASE2.md, Phase 3; docs/naacl/codex_ideas.txt ideas 1 to 3): uncertainty
that predicts the benefit of the second pass, rather than first-pass error. Offline, from stored v4 outputs; no model
call. Everything here is exploratory.

Two stages, run in this order, with a commit in between:

  --stage fit    development outcomes only (results/naacl/v4/dev/given). Fits and writes every parameter to
                 docs/naacl/results/v6_d2_frozen.json. It never opens a test output.
  --stage table  rebuilds paper/tab6_benefit.tex from paper/figures/data6/d2_routers.csv, without rescoring.
  --stage score  refuses to run unless v6_d2_frozen.json is committed and unchanged. Reads the frozen parameters
                 (nothing is refitted), scores development and test once, and writes docs/naacl/results/v6_d2_facts.json,
                 paper/tab6_benefit.tex and paper/figures/data6/d2_*.csv.

Routers (all use the stored outcomes of the frozen actions of the base method, v4.1 by default; correctness is eval.calc_accuracy.target_correct with
full target identity, ITOpsGraph v2.1):
  struct     the structural router (v4_analysis.STRUCT): the model's frozen widen action on declined fields and on
             fields whose named type has more than 20 fields; no uncertainty.
  Map-Reflect the frozen router of the base method (keep, NT or widen; v4.1 = v4 refitted with canonical margins).
  margin     the frozen v4 action on the fields with the smallest margin (declined first), filled to the budget.
  2-D rule   Codex idea 2. Keep a field only when its margin AND its top score are high: keep iff margin > a_m and
             s1 > b_m for answered fields that ranked two or more candidates, keep iff s1 > c_m for answered fields that
             ranked exactly one (their margin is their top score, with no measured competition); declined fields are
             always routed. (a_m, b_m, c_m) are fitted per model on development data to maximise net repairs under the
             frozen v4 action, subject to the model's development second-pass calls not exceeding the structural
             router's. At a matched budget the rule is extended to an ordering by its priority
             p = min(F(margin) - F(a), G(s1) - G(b)) (single-ranked: H(s1) - H(c)), with F, G, H the model's development
             empirical CDFs; p <= 0 reproduces the rule.
  benefit    Codex idea 1. u(x) = P(repair | x) - P(break | x) of the frozen v4 action, from an L2-regularised
             multinomial logistic model over {repair, break, no change} fitted on development fields of all four
             models. Free first-pass features only: margin, top score, top score = 1, declined, single ranked candidate,
             number ranked, number shown, truncated type (more than 20 fields), type width, retriever agreement (the
             first answer equals the hybrid dense + BM25 top candidate of the named type), margin x truncated,
             dataset and model indicators. C is chosen by 5-fold split-group cross-validated log loss on development.
             Routes the fields with the highest u, filled to the budget.
  struct+benefit  Codex idea 3 (matched-budget exchange). Start from struct; drop the share d of structural fields with
             the lowest u_W (the same model fitted to the widen action's outcome); spend the released budget on the
             non-structural fields with the highest u, under the v4 action. d is fitted on development (one value for
             all models) at the structural router's development calls.

Budgets are matched per model on realised second-pass cost: a routed field costs one reflection call plus one check
call when its action proposed a change, and the tokens of those calls (mean tokens per call from the v4 and v2 token
logs, as v6_numbers.call_tok_v4). Filling is greedy in priority order; a tied group at the boundary is routed with the
fractional weight that spends exactly the remaining budget (expected inclusion).

Intervals: split-group bootstrap (scripts/naacl/group_stats.py, B = 2000, seed 20261009), differences from struct
paired within draws.

Usage: ITOPS_V21=<v2.1 label file> .venv/bin/python scripts/naacl/v6_d2.py --stage fit|score
"""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["fit", "score", "table", "costfit", "costscore"], required=True)
ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--variant", default="v4_1", choices=["v4", "v4_1"],
                help="v4_1 (default): canonical margins (router.py) and docs/naacl/results/v4_1_frozen.json as the base "
                     "method; v4: the original float margins and v4_frozen.json (the first D2 run, frozen in 482df38)")
A = ap.parse_args()
LABELS = os.environ.get("ITOPS_V21")
if not LABELS or not Path(LABELS).exists():
    sys.exit("set ITOPS_V21 to the ITOpsGraph v2.1 label file")
if A.variant == "v4_1":
    os.environ["MAPREFLECT_CANONICAL_MARGIN"] = "1"
sys.argv = [sys.argv[0], "--labels-v21", LABELS, "--variant", A.variant]
import v4_analysis as V  # noqa: E402  (parses its own argv at import; loads the v2.1 gold)
import reflect_v2 as R2  # noqa: E402
import router  # noqa: E402
import group_stats as GS  # noqa: E402
from eval.calc_accuracy import target_correct  # noqa: E402

FROZEN_F = ROOT / ("docs/naacl/results/v6_d2_frozen.json" if A.variant == "v4" else "docs/naacl/results/v6_d2_1_frozen.json")
METHOD = "v4" if A.variant == "v4" else "v4.1"
FROZEN_REL = str(FROZEN_F.relative_to(ROOT))
OLD_FACTS_COMMIT = "a7d9128"  # the first D2 scoring (v4, float margins), for the old-vs-new diff
FACTS_F = ROOT / "docs/naacl/results/v6_d2_facts.json"
TAB_F = ROOT / "paper/tab6_benefit.tex"
DATA = ROOT / "paper/figures/data6"
TH = V.TH
assert router.CANONICAL == (A.variant == "v4_1")
MODELS = V.MODELS
DATASETS = V.DATASETS
MT = {"gptoss": "Gptoss", "llama4": "Llama", "granite": "Granite", "qwen35": "Qwen"}
SEED = 20261009


# ============================================================================================ loading
def read(f):
    return [json.loads(l) for l in open(f) if l.strip()]


def lower_key(k):
    a, b = k.split("||", 1)
    return a + "||" + b.lower()


POOLS = {ds: {lower_key(k): v for k, v in json.load(open(ROOT / f"data/v4_pools/{ds}.json")).items()} for ds in DATASETS}


def token_tables(split):
    """Mean tokens per call: the widen actions from the v4 logs of this split, NT from the v2 logs."""
    v4 = V.call_means([str(p) for p in (ROOT / "results/naacl/v4/tokens").glob(f"reflect__{split}__*.jsonl")],
                      {f"{p}:{a}" for p in ("reflect", "verify") for a in ("WT20", "WT50")})
    v2 = V.call_means([str(p) for p in (ROOT / "results/naacl/v2/tokens").glob("reflect__given__*.jsonl")],
                      {"reflect:B", "verify:BC"})
    return v4, v2


def tk(tab, ds, m, st):
    v = tab.get((ds, m, st), tab.get((m, st), tab.get((st,))))
    return float(v) if v is not None else 0.0


class Cell:
    pass


FIELDS = ["keep", "act_ok", "wid_ok", "ver_act", "ver_w", "margin", "s1", "nrank", "nshown", "ntype", "trunc",
          "agree", "valid", "isnt"]


def load(split):
    """Cells (one per dataset and model) of the split, rows in file order with v4_analysis.load's filters."""
    v4t, v2t = token_tables(split)
    cells = []
    for ds in DATASETS:
        man = json.loads((ROOT / f"data/splits/{ds}.json").read_text())
        ids = set(man["dev"]) if split == "dev" else set(man["test"]) - set(man.get("old_dev_excluded") or [])
        for m in MODELS:
            t = TH[m]
            W = t["widen"]
            f = ROOT / f"results/naacl/v4/{split}/given/{ds}__{m}/fields.jsonl"
            assert f.exists() and Path(str(f) + ".done").exists(), f
            recs = {r["key"]: r for r in R2.records(R2.load_slot(str(ROOT / f"results/naacl/v2/first/given/{ds}__{m}")), ids)}
            rows = []
            for r in read(f):
                sid = str(r["sample_id"])
                if sid not in ids:
                    continue
                fld = r["key"].rsplit("||", 1)[1]
                gold = V.gold_of(ds, sid, fld, r["gts"])
                if gold is None:
                    continue
                rec = recs[r["key"]]
                assert (rec["base"] or "") == (r["base"] or "") and abs(rec["margin"] - r["margin"]) < 1e-9, r["key"]
                band, act = router.route(float(r["margin"]), bool(r["valid"]), float(r["s1"]), t, ("NT", W))
                base = r["base"]
                ver = lambda a: float(bool(r.get(f"{a}_proposed")) and r.get(f"{a}_proposed") != base and bool(base))  # noqa
                p = POOLS[ds][f"{sid}||{fld}"]
                sc = [R2._num(c.get("score")) for c in rec["ranking"]]
                hyb = p.get("hybrid20") or []
                rows.append(dict(
                    sid=sid, keep=float(target_correct(base, gold)), act_ok=float(target_correct(r[act], gold)),
                    wid_ok=float(target_correct(r[W], gold)), ver_act=ver(act), ver_w=ver(W),
                    margin=float(router.canonical_margin(float(r["margin"]))), s1=float(r["s1"]), nrank=float(sum(x != float("-inf") for x in sc)),
                    nshown=float(len(p["shown"])), ntype=float(len(p["shown"]) + len(p["type_rest"])),
                    trunc=float(bool(p["type_rest"])),
                    agree=float(bool(base) and bool(hyb) and base.lower() == str(hyb[0]).lower()),
                    valid=float(bool(r["valid"])), isnt=float(act == "NT"), band=band))
            c = Cell()
            c.ds, c.model, c.n, c.split = ds, m, len(rows), split
            c.sid = np.array([x["sid"] for x in rows])
            for k in FIELDS:
                setattr(c, k, np.array([x[k] for x in rows], float))
            c.validb = c.valid > 0
            c.routed_mr = np.array([x["band"] != router.KEEP for x in rows], float)
            c.struct = ((~c.validb) | (c.trunc > 0)).astype(float)
            nt_r, nt_v = tk(v2t, ds, m, "reflect:B"), tk(v2t, ds, m, "verify:BC")
            w_r, w_v = tk(v4t, ds, m, f"reflect:{W}"), tk(v4t, ds, m, f"verify:{W}")
            # realised cost per field of the v4 action and of the widen action, if routed
            c.cost_act = {"calls": 1 + c.ver_act, "tokens": np.where(c.isnt > 0, nt_r + c.ver_act * nt_v, w_r + c.ver_act * w_v)}
            c.cost_wid = {"calls": 1 + c.ver_w, "tokens": w_r + c.ver_w * w_v}
            cells.append(c)
    return cells


# ============================================================================================ features
CONT = ["margin", "s1", "top1", "declined", "single", "lognrank", "nshown", "trunc", "logntype", "agree", "margin_trunc"]


def feats(c):
    single = ((c.validb) & (c.nrank == 1)).astype(float)
    X = dict(margin=c.margin, s1=c.s1, top1=(c.s1 >= 1.0).astype(float), declined=1.0 - c.valid, single=single,
             lognrank=np.log1p(c.nrank), nshown=c.nshown / 20.0, trunc=c.trunc, logntype=np.log1p(c.ntype),
             agree=c.agree, margin_trunc=c.margin * c.trunc)
    M = np.column_stack([X[k] for k in CONT])
    D = np.zeros((c.n, len(DATASETS)))
    D[:, DATASETS.index(c.ds)] = 1
    Mo = np.zeros((c.n, len(MODELS)))
    Mo[:, MODELS.index(c.model)] = 1
    return M, np.hstack([D, Mo])


def design(cells, mu=None, sd=None):
    Ms, Ds = zip(*(feats(c) for c in cells))
    M, D = np.vstack(Ms), np.vstack(Ds)
    if mu is None:
        mu, sd = M.mean(0), M.std(0)
        sd = np.where(sd > 0, sd, 1.0)
    return np.hstack([(M - mu) / sd, D]), mu, sd


def labels(cells, kind):
    """+1 repair, -1 break, 0 no change of the given action (act = the frozen v4 action, wid = the widen action)."""
    out = []
    for c in cells:
        ok = c.act_ok if kind == "act" else c.wid_ok
        out.append(np.where((c.keep == 0) & (ok == 1), 1, np.where((c.keep == 1) & (ok == 0), -1, 0)))
    return np.concatenate(out).astype(int)


def u_of(X, model):
    """u = P(repair) - P(break) from frozen multinomial coefficients."""
    W = np.array(model["coef"])
    b = np.array(model["intercept"])
    z = X @ W.T + b
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    cl = list(model["classes"])
    return p[:, cl.index(1)] - p[:, cl.index(-1)]


# ============================================================================================ budgets
def fill(prio, cost, budget):
    """Weights in [0, 1] routing fields in descending priority until the realised cost equals budget; a tied group at
    the boundary shares the remainder (expected inclusion)."""
    prio, cost = np.asarray(prio, float), np.asarray(cost, float)
    w = np.zeros(len(prio))
    if budget <= 0:
        return w
    order = np.argsort(-prio, kind="mergesort")
    ps = prio[order]
    starts = np.r_[0, np.nonzero(np.diff(ps) != 0)[0] + 1, len(ps)]
    spent = 0.0
    for a, b in zip(starts[:-1], starts[1:]):
        ix = order[a:b]
        gc = cost[ix].sum()
        if spent + gc <= budget + 1e-9:
            w[ix] = 1.0
            spent += gc
        else:
            w[ix] = (budget - spent) / gc if gc > 0 else 0.0
            break
    return w


def per_model(cells, fn):
    """fn(list of cells of one model) -> list of per-cell arrays; returns {id(cell): array}."""
    out = {}
    for m in MODELS:
        cs = [c for c in cells if c.model == m]
        for c, a in zip(cs, fn(cs)):
            out[id(c)] = a
    return out


def split_back(cs, v):
    out, i = [], 0
    for c in cs:
        out.append(v[i:i + c.n])
        i += c.n
    return out


def budget_of(cs, w_wid, w_act, unit):
    return float(sum(np.sum(w_wid[i] * c.cost_wid[unit] + w_act[i] * c.cost_act[unit]) for i, c in enumerate(cs)))


def route_by(prio_fn, unit, budget_fn):
    """Policy: per model, fill the v4 action in descending prio up to budget_fn(cells of the model)."""
    def pol(cells):
        def one(cs):
            pr = np.concatenate([prio_fn(c) for c in cs])
            cost = np.concatenate([c.cost_act[unit] for c in cs])
            return split_back(cs, fill(pr, cost, budget_fn(cs)))
        wa = per_model(cells, one)
        return {id(c): (np.zeros(c.n), wa[id(c)]) for c in cells}
    return pol


def struct_pol(cells):
    return {id(c): (c.struct.copy(), np.zeros(c.n)) for c in cells}


def mr_pol(cells):
    return {id(c): (np.zeros(c.n), c.routed_mr.copy()) for c in cells}


def struct_budget(unit):
    return lambda cs: float(sum(np.sum(c.struct * c.cost_wid[unit]) for c in cs))


def mr_budget(unit):
    return lambda cs: float(sum(np.sum(c.routed_mr * c.cost_act[unit]) for c in cs))


def exch_pol(d, unit, uW, uA):
    """struct+benefit: drop the share d of structural fields with the lowest u_W, refill non-structural fields by u."""
    def pol(cells):
        res = {}
        for m in MODELS:
            cs = [c for c in cells if c.model == m]
            S = np.concatenate([c.struct for c in cs]) > 0
            uw = np.concatenate([uW[id(c)] for c in cs])
            ua = np.concatenate([uA[id(c)] for c in cs])
            cw = np.concatenate([c.cost_wid[unit] for c in cs])
            ca = np.concatenate([c.cost_act[unit] for c in cs])
            budget = float(np.sum(S * cw))
            keepS = np.zeros(len(S))
            ns = int(S.sum())
            # keep the (1 - d) share of structural fields with the highest u_W (count-based, ties share)
            keepS[S] = fill(uw[S], np.ones(ns), (1 - d) * ns)
            spent = float(np.sum(keepS * cw))
            wa = np.zeros(len(S))
            wa[~S] = fill(ua[~S], ca[~S], budget - spent)
            for c, a, b in zip(cs, split_back(cs, keepS), split_back(cs, wa)):
                res[id(c)] = (a, b)
        return res
    return pol


def twod_keep(c, p):
    a, b, cc = p["a"], p["b"], p["c"]
    multi = c.validb & (c.nrank >= 2)
    single = c.validb & (c.nrank == 1)
    return (multi & (c.margin > a) & (c.s1 > b)) | (single & (c.s1 > cc))


def cdf(knots, x):
    k = np.asarray(knots, float)
    return np.interp(x, k, np.linspace(0, 1, len(k)))


def twod_prio(c, p):
    """Higher = route first. -p of the doc's priority (p <= 0 routes)."""
    big = 10.0
    fm = (cdf(p["Fm"], c.margin) - cdf(p["Fm"], p["a"])) if p["a"] > -1e8 else np.full(c.n, big)
    fs = (cdf(p["Fs"], c.s1) - cdf(p["Fs"], p["b"])) if p["b"] > -1e8 else np.full(c.n, big)
    f1 = (cdf(p["F1"], c.s1) - cdf(p["F1"], p["c"])) if p["c"] > -1e8 else np.full(c.n, big)
    # strict rule boundaries: a field outside the keep region gets p <= 0 even where the CDF is flat
    k = twod_keep(c, p)
    pr = np.where(c.nrank >= 2, np.minimum(fm, fs), f1)
    pr = np.where(k, np.maximum(pr, 1e-6), np.minimum(pr, 0.0))
    pr = np.where(c.validb, pr, -big)
    return -pr


def twod_own(P2):
    def pol(cells):
        return {id(c): (np.zeros(c.n), (~twod_keep(c, P2[c.model])).astype(float)) for c in cells}
    return pol


# ============================================================================================ evaluation
def evaluate(cells, W, GB=None, ref=None):
    """Gain, calls, tokens, repairs, breaks of policy weights W ({id: (w_wid, w_act)}); with ref, the paired
    difference and its bootstrap draws."""
    N = sum(c.n for c in cells)

    def delta(c, w):
        ww, wa = w[id(c)]
        return ww * (c.wid_ok - c.keep) + wa * (c.act_ok - c.keep)
    out = dict(
        gain=100 * sum(delta(c, W).sum() for c in cells) / N,
        calls=sum(np.sum(W[id(c)][0] * c.cost_wid["calls"] + W[id(c)][1] * c.cost_act["calls"]) for c in cells) / N,
        tokens=sum(np.sum(W[id(c)][0] * c.cost_wid["tokens"] + W[id(c)][1] * c.cost_act["tokens"]) for c in cells) / N,
        routed=100 * sum(np.sum(W[id(c)][0] + W[id(c)][1]) for c in cells) / N,
        repairs=float(sum(np.sum(W[id(c)][0] * ((c.keep == 0) & (c.wid_ok == 1)) + W[id(c)][1] * ((c.keep == 0) & (c.act_ok == 1))) for c in cells)),
        breaks=float(sum(np.sum(W[id(c)][0] * ((c.keep == 1) & (c.wid_ok == 0)) + W[id(c)][1] * ((c.keep == 1) & (c.act_ok == 0))) for c in cells)))
    if GB is not None:
        e, dr = GB.pooled(cells, lambda c: delta(c, W))
        out["lo"], out["hi"] = GS.interval(dr)
        if ref is not None:
            e2, d2 = GB.pooled(cells, lambda c: delta(c, W) - delta(c, ref))
            out["diff"] = e2
            out["dlo"], out["dhi"] = GS.interval(d2)
            out["p"] = GS.boot_p(d2)
    return out


def exchange(cells, W, S):
    """Fields one policy routes and the other does not, with their net repairs, and the overlap's action difference."""
    r = defaultdict(float)
    for c in cells:
        ww, wa = W[id(c)]
        sw, sa = S[id(c)]
        rw, sw_ = ww + wa, sw + sa
        only_r = np.clip(rw - sw_, 0, None)
        only_s = np.clip(sw_ - rw, 0, None)
        both = np.minimum(rw, sw_)
        # outcome of each side's own action on a field (struct: widen; router: its mix)
        r_ok = np.where(rw > 0, (ww * c.wid_ok + wa * c.act_ok) / np.where(rw > 0, rw, 1), c.act_ok)
        s_ok = np.where(sw_ > 0, (sw * c.wid_ok + sa * c.act_ok) / np.where(sw_ > 0, sw_, 1), c.wid_ok)
        for tag, w, ok in (("OnlyR", only_r, r_ok), ("OnlyS", only_s, s_ok)):
            r[tag + "Fields"] += w.sum()
            r[tag + "Repairs"] += np.sum(w * ((c.keep == 0) & (ok == 1)))
            r[tag + "Breaks"] += np.sum(w * ((c.keep == 1) & (ok == 0)))
        r["BothFields"] += both.sum()
        r["BothNet"] += np.sum(both * (r_ok - s_ok))
    return dict(r)


# ============================================================================================ fit (development only)
def fit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    cells = load("dev")
    N = sum(c.n for c in cells)
    groups = GS.split_groups(DATASETS, V.ITOPS_DUP)
    gkey = [f"{c.ds}::{groups[(c.ds, s)]}" for c in cells for s in c.sid]
    gid = np.unique(gkey, return_inverse=True)[1]
    X, mu, sd = design(cells)
    models = {}
    for kind in ("act", "wid"):
        y = labels(cells, kind)
        best = None
        for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            ll = 0.0
            for tr, te in GroupKFold(n_splits=5).split(X, y, gid):
                clf = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr])
                p = clf.predict_proba(X[te])
                ll -= np.log(np.clip(p[np.arange(len(te)), np.searchsorted(clf.classes_, y[te])], 1e-12, 1)).sum()
            ll /= len(y)
            if best is None or ll < best[0]:
                best = (ll, C)
        clf = LogisticRegression(C=best[1], max_iter=5000).fit(X, y)
        models[kind] = dict(C=best[1], cv_logloss=best[0], classes=[int(k) for k in clf.classes_],
                            coef=clf.coef_.tolist(), intercept=clf.intercept_.tolist(),
                            n_repair=int((y == 1).sum()), n_break=int((y == -1).sum()), n=int(len(y)))
    # 2-D rule per model, under the structural router's development calls
    P2 = {}
    for m in MODELS:
        cs = [c for c in cells if c.model == m]
        cat = lambda k: np.concatenate([getattr(c, k) for c in cs])  # noqa: E731
        mg, s1, nr, va = cat("margin"), cat("s1"), cat("nrank"), cat("valid") > 0
        keep, ok, ca = cat("keep"), cat("act_ok"), np.concatenate([c.cost_act["calls"] for c in cs])
        bud = struct_budget("calls")(cs)
        multi, single = va & (nr >= 2), va & (nr == 1)
        q = np.linspace(0, 1, 21)
        A_ = np.unique(np.r_[-1e9, np.quantile(mg[multi], q)])
        B_ = np.unique(np.r_[-1e9, np.quantile(s1[multi], q)])
        C_ = np.unique(np.r_[-1e9, np.quantile(s1[single], q) if single.any() else [], 1e9])
        d = ok - keep
        best = None
        for c_ in C_:
            ks = single & (s1 > c_)
            for a in A_:
                for b in B_:
                    k = ks | (multi & (mg > a) & (s1 > b))
                    rt = ~k
                    calls = float(np.sum(rt * ca))
                    if calls > bud + 1e-9:
                        continue
                    cand = (float(np.sum(rt * d)), -calls, float(a), float(b), float(c_))
                    if best is None or cand[:2] > best[:2]:
                        best = cand
        kn = np.linspace(0, 1, 201)
        P2[m] = dict(a=best[2], b=best[3], c=best[4], dev_net=best[0], dev_calls=-best[1] / len(keep),
                     dev_struct_calls=bud / len(keep),
                     Fm=np.quantile(mg[multi], kn).tolist(), Fs=np.quantile(s1[multi], kn).tolist(),
                     F1=(np.quantile(s1[single], kn) if single.any() else np.zeros(len(kn))).tolist())
    # struct+benefit: one share d for all models, at the structural router's development calls
    Xs = split_back(cells, X)
    uA = {id(c): u_of(x, models["act"]) for c, x in zip(cells, Xs)}
    uW = {id(c): u_of(x, models["wid"]) for c, x in zip(cells, Xs)}
    grid = []
    for d in np.round(np.arange(0, 0.61, 0.05), 2):
        r = evaluate(cells, exch_pol(float(d), "calls", uW, uA)(cells))
        grid.append((float(d), r["gain"]))
    dbest = max(grid, key=lambda t: (round(t[1], 9), -t[0]))[0]
    fz = dict(
        date="2026-10-10", stage="fit", base_method=METHOD, canonical_margin=router.CANONICAL,
        base_frozen="docs/naacl/results/" + ("v4_frozen.json" if A.variant == "v4" else "v4_1_frozen.json"), note="Fitted on development outcomes only (results/naacl/v4/dev/given); "
        "no test output was read. Exploratory (D2, docs/naacl/v6/PLAN_PHASE2.md Phase 3).",
        dev_fields=N, features=CONT + [f"ds:{d}" for d in DATASETS] + [f"model:{m}" for m in MODELS],
        standardise=dict(mu=mu.tolist(), sd=sd.tolist()),
        benefit_act=models["act"], benefit_wid=models["wid"], twod=P2, exchange_d=dbest, exchange_grid=grid,
        dev_digest=hashlib.sha256(json.dumps([[c.ds, c.model, c.n, float(c.keep.sum()), float(c.act_ok.sum())]
                                              for c in cells]).encode()).hexdigest())
    FROZEN_F.write_text(json.dumps(fz, indent=1) + "\n")
    print("wrote", FROZEN_F)
    for k in ("act", "wid"):
        print(k, {x: models[k][x] for x in ("C", "cv_logloss", "n_repair", "n_break", "n")})
    for m in MODELS:
        print(m, {x: P2[m][x] for x in ("a", "b", "c", "dev_net", "dev_calls", "dev_struct_calls")})
    print("exchange d", dbest, grid)


# ============================================================================================ score
def require_frozen_committed(path=None):
    rel = str((path or FROZEN_F).relative_to(ROOT))
    log = subprocess.run(["git", "log", "-1", "--format=%H %cI", "--", rel], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", rel], capture_output=True, text=True).stdout.strip()
    if not log or dirty:
        sys.exit(f"{rel} must be committed and unchanged before scoring (log: {log!r}, status: {dirty!r})")
    return log


FACTS = []


def fact(fid, claim, value, lo=None, hi=None):
    assert fid.isalpha(), fid
    FACTS.append(dict(id="F-DTwo" + fid, claim=claim, value=str(value), lo=None if lo is None else str(lo),
                      hi=None if hi is None else str(hi), src="docs/naacl/results/v6_d2_facts.json",
                      fn="v6_d2.score", status="exploratory"))


def sg(x, nd=2):
    return f"{x:+.{nd}f}"


HEAD = ["BenefitCallsVs", "BenefitTokensVs", "ExchCallsVs", "ExchTokensVs", "MarginCallsVs", "MarginTokensVs",
        "TwoDCallsVs", "TwoDTokensVs", "TwoDOwnOwnVs", "MapReflectOwnVs", "BenefitMRVs", "MapReflectOwnGain",
        "BenefitCallsGain", "StructCallsGain", "StructMissBenefit", "StructMiss"]


def diff_old():
    """Headline contrasts of the first D2 run (v4, float margins; facts at OLD_FACTS_COMMIT) next to this run's."""
    if A.variant == "v4":
        return []
    try:
        old = json.loads(subprocess.run(["git", "show", f"{OLD_FACTS_COMMIT}:docs/naacl/results/v6_d2_facts.json"],
                                        capture_output=True, text=True, check=True).stdout)["facts"]
    except Exception:
        return []
    O = {f["id"]: f for f in old}
    N_ = {f["id"]: f for f in FACTS}
    out = []
    for sp in ("Dev", ""):
        for h in HEAD:
            k = f"F-DTwo{sp}{h}"
            if k in O and k in N_:
                fmt = lambda f: f["value"] + (f" [{f['lo']}, {f['hi']}]" if f.get("lo") else "")  # noqa: E731
                out.append(dict(id=k, claim=N_[k]["claim"], v4_float=fmt(O[k]), v4_1_canonical=fmt(N_[k])))
    return out


def score():
    commit = require_frozen_committed()
    fz = json.loads(FROZEN_F.read_text())
    mu, sd = np.array(fz["standardise"]["mu"]), np.array(fz["standardise"]["sd"])
    P2 = fz["twod"]
    d = fz["exchange_d"]
    groups = GS.split_groups(DATASETS, V.ITOPS_DUP)
    RES, EXCH, CURVES = {}, {}, []
    for split in ("dev", "test"):
        cells = load(split)
        if split == "test":  # the same fields, outcomes and routing as v4_analysis.load
            vc = V.load()[0]
            assert [(c.ds, c.model, c.n) for c in vc] == [(c.ds, c.model, c.n) for c in cells]
            for a, b in zip(vc, cells):
                for x, y in (("keep", "keep"), ("action", "act_ok"), ("widen", "wid_ok"), ("routed", "routed_mr"),
                             ("ver_act", "ver_act"), ("ver_w", "ver_w"), ("wide", "trunc")):
                    assert np.array_equal(getattr(a, x), getattr(b, y)), (a.ds, a.model, x)
        GB = GS.GroupBootstrap(cells, groups, B=A.B)
        X, _, _ = design(cells, mu, sd)
        Xs = split_back(cells, X)
        uA = {id(c): u_of(x, fz["benefit_act"]) for c, x in zip(cells, Xs)}
        uW = {id(c): u_of(x, fz["benefit_wid"]) for c, x in zip(cells, Xs)}
        mprio = lambda c: -np.where(c.validb, c.margin, -1.0)  # noqa: E731  smallest margin first, declined first
        S = struct_pol(cells)
        MR = mr_pol(cells)
        pols = {("Struct", "calls"): S, ("MapReflect", "own"): MR, ("TwoDOwn", "own"): twod_own(P2)(cells)}
        for unit in ("calls", "tokens"):
            bud = struct_budget(unit)
            pols[("Margin", unit)] = route_by(mprio, unit, bud)(cells)
            pols[("TwoD", unit)] = route_by(lambda c: twod_prio(c, P2[c.model]), unit, bud)(cells)
            pols[("Benefit", unit)] = route_by(lambda c: uA[id(c)], unit, bud)(cells)
            pols[("Exch", unit)] = exch_pol(d, unit, uW, uA)(cells)
        pols[("Benefit", "mr")] = route_by(lambda c: uA[id(c)], "calls", mr_budget("calls"))(cells)
        pols[("Margin", "mr")] = route_by(mprio, "calls", mr_budget("calls"))(cells)
        for k, W in pols.items():
            ref = MR if k[1] == "mr" else S
            RES[(split, k)] = evaluate(cells, W, GB, ref)
            if k[0] in ("Benefit", "TwoD", "Margin", "Exch") and k[1] in ("calls", "tokens"):
                EXCH[(split, k)] = exchange(cells, W, S)
        # the structural repairs Map-Reflect misses: structural, kept by Map-Reflect, repaired by widening
        miss = {id(c): (c.struct > 0) & (c.routed_mr == 0) & (c.keep == 0) & (c.wid_ok == 1) for c in cells}
        RES[(split, "miss")] = dict(n=int(sum(miss[id(c)].sum() for c in cells)), **{
            nm: float(sum(np.sum(pols[(nm, "calls")][id(c)][1] * miss[id(c)] * (c.act_ok == 1)
                                 + pols[(nm, "calls")][id(c)][0] * miss[id(c)] * (c.wid_ok == 1)) for c in cells))
            for nm in ("TwoD", "Benefit", "Margin", "Exch")})
        RES[(split, "miss")]["TwoDOwn"] = float(sum(np.sum(pols[("TwoDOwn", "own")][id(c)][1] * miss[id(c)] * (c.act_ok == 1))
                                                    for c in cells))
        # share of the benefit router's matched-calls budget spent on fields with u <= 0
        Wb = pols[("Benefit", "calls")]
        RES[(split, "uneg")] = 100 * sum(np.sum(Wb[id(c)][1] * c.cost_act["calls"] * (uA[id(c)] <= 0)) for c in cells) / \
            sum(np.sum(Wb[id(c)][1] * c.cost_act["calls"]) for c in cells)
        # gain-calls curves of each ordering (no CIs), for a chart
        for frac in np.linspace(0.05, 1.0, 20):
            for nm, pr in (("margin", mprio), ("twod", lambda c: twod_prio(c, P2[c.model])), ("benefit", lambda c: uA[id(c)])):
                W = route_by(pr, "calls", lambda cs, f=frac: f * sum(np.sum(c.cost_act["calls"]) for c in cs))(cells)
                r = evaluate(cells, W)
                CURVES.append((split, nm, round(float(frac), 3), r["calls"], r["tokens"], r["gain"]))
        RES[(split, "N")] = sum(c.n for c in cells)
        RES[(split, "first")] = 100 * sum(c.keep.sum() for c in cells) / RES[(split, "N")]
        print(split, "done")
    write_outputs(RES, EXCH, CURVES, fz, commit)


NAME = {"Struct": "struct", "MapReflect": "Map-Reflect", "TwoDOwn": "2-D rule, own budget", "Margin": "margin",
        "TwoD": "2-D rule", "Benefit": "benefit", "Exch": "struct + benefit"}
BUD = {"calls": "at the structural router's calls", "tokens": "at the structural router's tokens", "own": "at its own budget",
       "mr": "at Map-Reflect's calls"}
BTAG = {"calls": "Calls", "tokens": "Tokens", "own": "Own", "mr": "MR"}


def write_outputs(RES, EXCH, CURVES, fz, commit):
    fact("FrozenCommit", "commit and time of " + FROZEN_REL + " (fitted on development only; base method " + METHOD + ")", commit)
    for k, nm in (("benefit_act", "BenefitAct"), ("benefit_wid", "BenefitWid")):
        mdl = fz[k]
        what = f"{METHOD} action" if k == "benefit_act" else "widen action"
        fact(f"{nm}C", f"benefit model ({what}): L2 inverse strength C chosen by grouped CV on development", mdl["C"])
        fact(f"{nm}DevRepairs", f"benefit model ({what}): development repairs / breaks / fields", f"{mdl['n_repair']} / {mdl['n_break']} / {mdl['n']:,}")
    for m in MODELS:
        p = fz["twod"][m]
        fmt = lambda x: "none" if x <= -1e8 else ("all" if x >= 1e8 else f"{x:.3f}")  # noqa: E731
        fact(f"TwoDThr{MT[m]}", f"{m}: 2-D rule thresholds a (margin), b (top score), c (single-ranked top score)",
             f"a = {fmt(p['a'])}, b = {fmt(p['b'])}, c = {fmt(p['c'])}")
    fact("ExchShare", "struct + benefit: share of structural fields dropped (fitted on development)", f"{fz['exchange_d']:.2f}")
    for split in ("dev", "test"):
        sp = "Dev" if split == "dev" else ""
        lab = "development" if split == "dev" else "test"
        fact(f"{sp}Fields", f"{lab} field-model pairs", f"{RES[(split, 'N')]:,}")
        for (s2, k), r in RES.items():
            if s2 != split or not isinstance(k, tuple):
                continue
            rt, bu = k
            tg = f"{sp}{rt}{BTAG[bu]}"
            desc = f"{lab}, {NAME[rt]} {BUD[bu]}"
            fact(tg + "Gain", f"{desc}: change over the first pass, points", sg(r["gain"]), sg(r["lo"]), sg(r["hi"]))
            fact(tg + "CallsPer", f"{desc}: second-pass calls per field", f"{r['calls']:.3f}")
            fact(tg + "TokensPer", f"{desc}: second-pass tokens per field", f"{r['tokens']:,.0f}")
            fact(tg + "Routed", f"{desc}: share of fields routed, %", f"{r['routed']:.1f}")
            fact(tg + "Repairs", f"{desc}: repairs / breaks", f"{r['repairs']:.0f} / {r['breaks']:.0f}")
            if rt != "Struct" and "diff" in r:
                other = "Map-Reflect" if bu == "mr" else "struct"
                fact(tg + "Vs", f"{desc}: minus {other}, points", sg(r["diff"]), sg(r["dlo"]), sg(r["dhi"]))
        for (s2, k), e in EXCH.items():
            if s2 != split:
                continue
            tg = f"{sp}Ex{k[0]}{BTAG[k[1]]}"
            desc = f"{lab}, {NAME[k[0]]} {BUD[k[1]]} vs struct"
            fact(tg + "OnlyR", f"{desc}: fields only the router routes (weighted): fields / repairs / breaks",
                 f"{e['OnlyRFields']:.0f} / {e['OnlyRRepairs']:.0f} / {e['OnlyRBreaks']:.0f}")
            fact(tg + "OnlyS", f"{desc}: fields only struct routes (weighted): fields / repairs / breaks",
                 f"{e['OnlySFields']:.0f} / {e['OnlySRepairs']:.0f} / {e['OnlySBreaks']:.0f}")
            fact(tg + "Both", f"{desc}: fields both route / net correct difference there from the router's action",
                 f"{e['BothFields']:.0f} / {e['BothNet']:+.0f}")
        ms = RES[(split, "miss")]
        fact(f"{sp}StructMiss", f"{lab}: structural fields Map-Reflect keeps that widening repairs", ms["n"])
        for nm in ("TwoD", "Benefit", "Margin", "Exch"):
            fact(f"{sp}StructMiss{nm}", f"{lab}: of those, repaired by {NAME[nm]} at the structural router's calls", f"{ms[nm]:.1f}")
        fact(f"{sp}StructMissTwoDOwn", f"{lab}: of those, repaired by the 2-D rule at its own budget", f"{ms['TwoDOwn']:.1f}")
        fact(f"{sp}BenefitUneg", f"{lab}: share of the benefit router's matched-calls budget on fields with u <= 0, %",
             f"{RES[(split, 'uneg')]:.1f}")
    FACTS_F.write_text(json.dumps(dict(
        generated_by="scripts/naacl/v6_d2.py --stage score", frozen=FROZEN_REL, base_method=METHOD, canonical_margin=router.CANONICAL, diff_vs_first_run=diff_old(),
        frozen_commit=commit, status="exploratory", facts=FACTS), indent=1) + "\n")
    print("wrote", FACTS_F, len(FACTS), "facts")
    # chart data
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "d2_routers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "router", "budget", "gain", "lo", "hi", "calls", "tokens", "routed_pct", "repairs", "breaks",
                    "diff", "diff_lo", "diff_hi"])
        for (sp, k), r in RES.items():
            if isinstance(k, tuple):
                w.writerow([sp, k[0], k[1]] + [f"{r[x]:.6g}" if x in r else "" for x in
                                                ("gain", "lo", "hi", "calls", "tokens", "routed", "repairs", "breaks", "diff", "dlo", "dhi")])
    with open(DATA / "d2_exchange.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["OnlyRFields", "OnlyRRepairs", "OnlyRBreaks", "OnlySFields", "OnlySRepairs", "OnlySBreaks", "BothFields", "BothNet"]
        w.writerow(["split", "router", "budget"] + cols)
        for (sp, k), e in EXCH.items():
            w.writerow([sp, k[0], k[1]] + [f"{e[c]:.6g}" for c in cols])
    with open(DATA / "d2_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "ordering", "budget_frac", "calls", "tokens", "gain"])
        for r in CURVES:
            w.writerow(list(r[:3]) + [f"{x:.6g}" for x in r[3:]])
    tables(RES)


def tex_s(x, nd=2):
    s = f"{x:+.{nd}f}"
    return f"${s}$" if x < 0 else s


def tables(RES):
    def ci(r):
        return f"{tex_s(r['diff'])} [{tex_s(r['dlo'])}, {tex_s(r['dhi'])}]"
    L = ["% Generated by scripts/naacl/v6_d2.py; do not edit by hand. D2 (exploratory): selection rules fitted and frozen on "
         f"development ({FROZEN_REL}, committed before any test scoring), test scored once; base method {METHOD}"
         + (" with canonical margins. " if router.CANONICAL else ". ") +
         "Calls and tokens per field matched per model to the schema-size baseline's realised second-pass calls or "
         "tokens; calls and tokens are test values per field. Gain = change over the first pass, points; Delta = minus "
         "the schema-size baseline (Map-Reflect's own row: at its own cost), split-group bootstrap 95% CI, B = 2000.",
         '% group: ablation-mr (appendix "Ablation: Margin-Reflect"; listed in tab6_ablation_mr_index.tex)',
         r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrlrl}", r"\toprule",
         r" & & & \multicolumn{2}{c}{Development} & \multicolumn{2}{c}{Test} \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Selection & Calls & Tokens & Gain & $\Delta$ vs schema size [95\% CI] & Gain & $\Delta$ vs schema size [95\% CI] \\",
         r"\midrule"]

    def row(lab, key):
        r, rd = RES[("test", key)], RES[("dev", key)]
        if key[0] == "Struct":
            return f"{lab} & {r['calls']:.2f} & {r['tokens']:,.0f} & {tex_s(rd['gain'])} & reference & {tex_s(r['gain'])} & reference \\\\"
        return (f"{lab} & {r['calls']:.2f} & {r['tokens']:,.0f} & {tex_s(rd['gain'])} & {ci(rd)} & {tex_s(r['gain'])} & {ci(r)} \\\\")
    for unit, head in (("calls", "Matched to the schema-size baseline's calls"), ("tokens", "Matched to the schema-size baseline's tokens")):
        L.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{head}}}}} \\")
        for k, lab in (("Struct", "Schema-size baseline"), ("Margin", "Margin"), ("TwoD", "2-D rule"), ("Benefit", "Benefit (classifier)"),
                       ("Exch", r"Schema size $+$ Benefit")):
            L.append(row(lab, (k, "calls" if k == "Struct" else unit)))
    L.append(r"\multicolumn{7}{l}{\emph{Own cost}} \\")
    L.append(row("2-D rule", ("TwoDOwn", "own")))
    L.append(row("Margin-Reflect", ("MapReflect", "own")))
    L += [r"\bottomrule", r"\end{tabular}", r"\gencaption", r"\label{tab:benefit}", r"\end{table*}"]
    TAB_F.write_text("\n".join(L) + "\n")
    print("wrote", TAB_F)


def table_from_csv():
    """Rebuild the table from the scored results in d2_routers.csv (no rescoring)."""
    RES = {}
    for r in csv.DictReader(open(DATA / "d2_routers.csv")):
        RES[(r["split"], (r["router"], r["budget"]))] = {k: float(v) for k, v in (
            ("gain", r["gain"]), ("calls", r["calls"]), ("tokens", r["tokens"]), ("diff", r["diff"]),
            ("dlo", r["diff_lo"]), ("dhi", r["diff_hi"])) if v != ""}
    tables(RES)



# ============================================================================================ D2-cost (phase 5)
# The matched-budget fill above charges each field its realised verification cost, which a router cannot know before
# the second pass runs, and routes a boundary tie group fractionally. D2-cost is the operational check: a router only
# knows development-estimated action costs, routes whole fields in priority order (ties broken by a fixed-seed random
# order), and stops at the first field whose expected cost no longer fits the budget left after its observed spend.
COST_F = ROOT / "docs/naacl/results/v6_d2_cost_frozen.json"
COST_SEED = 20261011


def costfit():
    """Development-estimated costs per model and action: the rate at which the action proposes a change (so a check
    call follows) and the mean tokens per reflect and check call. Development only."""
    assert A.variant == "v4_1"
    cells = load("dev")
    v4t, v2t = token_tables("dev")
    est = {}
    for m in MODELS:
        W = TH[m]["widen"]
        cs = [c for c in cells if c.model == m]
        isnt = np.concatenate([c.isnt for c in cs]) > 0
        va = np.concatenate([c.ver_act for c in cs])
        vw = np.concatenate([c.ver_w for c in cs])
        allc = [c for c in cells]
        pooled_nt = float(np.mean(np.concatenate([c.ver_act[c.isnt > 0] for c in allc])))
        est[m] = dict(
            widen=W,
            p_check_widen=float(np.mean(np.r_[va[~isnt], vw[isnt]])) if len(va) else 0.0,
            p_check_nt=float(np.mean(va[isnt])) if isnt.sum() >= 20 else pooled_nt,
            nt_source="model" if isnt.sum() >= 20 else "pooled over models (fewer than 20 development NT fields)",
            n_nt=int(isnt.sum()), n=int(len(va)),
            tok_reflect_widen=tk(v4t, "", m, f"reflect:{W}"), tok_verify_widen=tk(v4t, "", m, f"verify:{W}"),
            tok_reflect_nt=tk(v2t, "", m, "reflect:B"), tok_verify_nt=tk(v2t, "", m, "verify:BC"))
    fz = dict(date="2026-10-11", stage="costfit", base_method=METHOD, canonical_margin=router.CANONICAL,
              d2_frozen=FROZEN_REL, seed=COST_SEED,
              note="Development-estimated second-pass costs for D2-cost: expected calls = 1 + P(check | model, action); "
                   "expected tokens = mean reflect tokens + P(check) x mean check tokens, per model and action, from the "
                   "development token logs (widen: results/naacl/v4/tokens/reflect__dev__*; NT: the v2 reflect logs). "
                   "The widen check rate pools every development field's widen proposal (routed action or not).",
              est=est)
    COST_F.write_text(json.dumps(fz, indent=1) + "\n")
    print(json.dumps(est, indent=1))


def exp_cost(c, E, unit, action="act"):
    e = E[c.model]
    if action == "wid":
        p, r, v = e["p_check_widen"], e["tok_reflect_widen"], e["tok_verify_widen"]
        return np.full(c.n, 1 + p) if unit == "calls" else np.full(c.n, r + p * v)
    nt = c.isnt > 0
    p = np.where(nt, e["p_check_nt"], e["p_check_widen"])
    if unit == "calls":
        return 1 + p
    return np.where(nt, e["tok_reflect_nt"] + p * e["tok_verify_nt"], e["tok_reflect_widen"] + p * e["tok_verify_widen"])


def seq_fill(prio, ecost, rcost, budget, spent0, rng):
    """Whole fields in descending priority (fixed-seed random order inside ties); route a field while observed spend
    plus its expected cost fits the budget, charging its realised cost; stop at the first that does not fit."""
    tie = rng.random(len(prio))
    order = np.lexsort((tie, -np.asarray(prio, float)))
    w = np.zeros(len(prio))
    spent = spent0
    for i in order:
        if spent + ecost[i] > budget + 1e-9:
            break
        w[i] = 1.0
        spent += rcost[i]
    return w, spent


def seq_route(cells, prio_fn, unit, budget_fn, E, MIS, tag):
    out = {}
    rng = np.random.default_rng(COST_SEED)
    for m in MODELS:
        cs = [c for c in cells if c.model == m]
        pr = np.concatenate([prio_fn(c) for c in cs])
        ec = np.concatenate([exp_cost(c, E, unit) for c in cs])
        rc = np.concatenate([c.cost_act[unit] for c in cs])
        b = budget_fn(cs)
        w, spent = seq_fill(pr, ec, rc, b, 0.0, rng)
        MIS[(tag, m)] = (spent, b)
        for c, a in zip(cs, split_back(cs, w)):
            out[id(c)] = (np.zeros(c.n), a)
    return out


def seq_exch(cells, d, unit, uW, uA, E, MIS, tag):
    out = {}
    rng = np.random.default_rng(COST_SEED)
    for m in MODELS:
        cs = [c for c in cells if c.model == m]
        S = np.concatenate([c.struct for c in cs]) > 0
        uw, ua = np.concatenate([uW[id(c)] for c in cs]), np.concatenate([uA[id(c)] for c in cs])
        cw = np.concatenate([c.cost_wid[unit] for c in cs])
        ec = np.concatenate([exp_cost(c, E, unit) for c in cs])
        rc = np.concatenate([c.cost_act[unit] for c in cs])
        b = float(np.sum(S * cw))
        ns = int(S.sum())
        keepS = np.zeros(len(S))
        tie = rng.random(ns)
        o = np.lexsort((tie, -uw[S]))
        idx = np.nonzero(S)[0][o[:int(round((1 - d) * ns))]]
        keepS[idx] = 1.0
        spent0 = float(np.sum(keepS * cw))
        wa = np.zeros(len(S))
        w_, spent = seq_fill(ua[~S], ec[~S], rc[~S], b, spent0, rng)
        wa[~S] = w_
        MIS[(tag, m)] = (spent, b)
        for c, a, bb in zip(cs, split_back(cs, keepS), split_back(cs, wa)):
            out[id(c)] = (a, bb)
    return out


def costscore():
    assert A.variant == "v4_1"
    commit = require_frozen_committed(COST_F)
    require_frozen_committed(FROZEN_F)
    E = json.loads(COST_F.read_text())["est"]
    fz = json.loads(FROZEN_F.read_text())
    mu, sd = np.array(fz["standardise"]["mu"]), np.array(fz["standardise"]["sd"])
    P2, d = fz["twod"], fz["exchange_d"]
    groups = GS.split_groups(DATASETS, V.ITOPS_DUP)
    NEW, ROWS = [], []

    def F(fid, claim, value, lo=None, hi=None):
        assert fid.isalpha(), fid
        NEW.append(dict(id="F-DTwoCost" + fid, claim=claim, value=str(value), lo=None if lo is None else str(lo),
                        hi=None if hi is None else str(hi), src="docs/naacl/results/v6_d2_facts.json",
                        fn="v6_d2.costscore", status="exploratory"))
    DEF = ("D2-cost definition: whole fields in priority order (fixed-seed random order inside ties); a field is routed "
           "while observed spend plus its development-estimated expected cost (calls: 1 + P(check | model, action); "
           "tokens: mean reflect tokens + P(check) x mean check tokens, development logs) fits the budget; observed "
           "spend charges the realised calls (one check call only when the action proposed a change) and mean-token "
           "estimates per call (token logs per dataset, model and stage); stops at the first field that does not fit. "
           "Budget per model: the structural router's spend (calls, or mean-token estimates). No fractional fields.")
    F("Commit", "commit and time of docs/naacl/results/v6_d2_cost_frozen.json (development-estimated costs)", commit)
    for m in MODELS:
        e = E[m]
        F(f"Est{MT[m]}", f"{m}: development-estimated P(check) for widen / NT, and expected calls per routed widen field",
          f"{e['p_check_widen']:.3f} / {e['p_check_nt']:.3f} / {1 + e['p_check_widen']:.3f}")
    F("Definition", "D2-cost routing and budget definition", DEF)
    for split in ("dev", "test"):
        sp = "Dev" if split == "dev" else ""
        lab = "development" if split == "dev" else "test"
        cells = load(split)
        GB = GS.GroupBootstrap(cells, groups, B=A.B)
        X, _, _ = design(cells, mu, sd)
        Xs = split_back(cells, X)
        uA = {id(c): u_of(x, fz["benefit_act"]) for c, x in zip(cells, Xs)}
        uW = {id(c): u_of(x, fz["benefit_wid"]) for c, x in zip(cells, Xs)}
        mprio = lambda c: -np.where(c.validb, c.margin, -1.0)  # noqa: E731
        S = struct_pol(cells)
        MR = mr_pol(cells)
        MIS = {}
        pols = {}
        for unit in ("calls", "tokens"):
            bud = struct_budget(unit)
            pols[("Margin", unit)] = seq_route(cells, mprio, unit, bud, E, MIS, ("Margin", unit))
            pols[("TwoD", unit)] = seq_route(cells, lambda c: twod_prio(c, P2[c.model]), unit, bud, E, MIS, ("TwoD", unit))
            pols[("Benefit", unit)] = seq_route(cells, lambda c: uA[id(c)], unit, bud, E, MIS, ("Benefit", unit))
            pols[("Exch", unit)] = seq_exch(cells, d, unit, uW, uA, E, MIS, ("Exch", unit))
        pols[("Benefit", "mr")] = seq_route(cells, lambda c: uA[id(c)], "calls", mr_budget("calls"), E, MIS, ("Benefit", "mr"))
        N = sum(c.n for c in cells)
        for k, W in pols.items():
            ref = MR if k[1] == "mr" else S
            r = evaluate(cells, W, GB, ref)
            spent = sum(MIS[(k, m)][0] for m in MODELS)
            b = sum(MIS[(k, m)][1] for m in MODELS)
            worst = max(abs(MIS[(k, m)][0] - MIS[(k, m)][1]) / MIS[(k, m)][1] for m in MODELS if MIS[(k, m)][1] > 0)
            tg = f"{sp}{k[0]}{BTAG[k[1]]}"
            desc = f"D2-cost, {lab}, {NAME[k[0]]} {BUD[k[1]]} (development-estimated costs, sequential stopping)"
            F(tg + "Gain", f"{desc}: change over the first pass, points", sg(r["gain"]), sg(r["lo"]), sg(r["hi"]))
            other = "Map-Reflect" if k[1] == "mr" else "struct"
            F(tg + "Vs", f"{desc}: minus {other}, points", sg(r["diff"]), sg(r["dlo"]), sg(r["dhi"]))
            F(tg + "CallsPer", f"{desc}: realised second-pass calls per field", f"{r['calls']:.3f}")
            F(tg + "TokensPer", f"{desc}: second-pass tokens per field (mean-token estimates)", f"{r['tokens']:,.0f}")
            unit = "calls" if k[1] in ("calls", "mr") else "tokens"
            F(tg + "Mismatch", f"{desc}: residual budget mismatch, observed spend minus budget, {unit} per field "
              f"(% of budget; largest per-model absolute %)",
              f"{(spent - b) / N:+.4f} ({100 * (spent - b) / b:+.2f}%; max {100 * worst:.2f}%)")
            F(tg + "Repairs", f"{desc}: repairs / breaks", f"{r['repairs']:.0f} / {r['breaks']:.0f}")
            ROWS.append([split, k[0], k[1], r["gain"], r["lo"], r["hi"], r["calls"], r["tokens"], r["diff"], r["dlo"],
                         r["dhi"], (spent - b) / N, 100 * (spent - b) / b, 100 * worst])
        print(split, "done")
    D = json.loads(FACTS_F.read_text())
    D["facts"] = [f for f in D["facts"] if not f["id"].startswith("F-DTwoCost")] + NEW
    D["cost_check"] = dict(stage="costscore", frozen="docs/naacl/results/v6_d2_cost_frozen.json", frozen_commit=commit,
                           definition=DEF)
    FACTS_F.write_text(json.dumps(D, indent=1) + "\n")
    with open(DATA / "d2_cost.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "router", "budget", "gain", "lo", "hi", "calls", "tokens", "diff", "diff_lo", "diff_hi",
                    "mismatch_per_field", "mismatch_pct", "max_model_mismatch_pct"])
        for r in ROWS:
            w.writerow(r[:3] + [f"{x:.6g}" for x in r[3:]])
    for r in ROWS:
        print(r[0], r[1], r[2], f"gain {r[3]:+.2f} calls {r[6]:.3f} tok {r[7]:.0f} vs {r[8]:+.2f} [{r[9]:+.2f},{r[10]:+.2f}] mis {r[12]:+.2f}% max {r[13]:.2f}%")
    print("wrote", len(NEW), "D2-cost facts")


if __name__ == "__main__":
    {"fit": fit, "score": score, "table": table_from_csv, "costfit": costfit, "costscore": costscore}[A.stage]()
