#!/usr/bin/env python3
"""Matched-budget comparison on the method v5 second pass (D2-cost on v5). Offline; no model call. Exploratory.

The v5 headline (v5 minus struct, docs/naacl/results/v5_facts.json) compares 0.44 calls per field against struct's
0.37, so its budgets differ. Here every router uses the same measured v5 second pass (results/naacl/v6/v5/test, the
either-order verifier, the field's routed action; struct runs the widen action, as in v5.py) and is held to a matched
budget with the D2-cost rule (scripts/naacl/v6_d2.py, F-DTwoCost):

  accounting   v5.py charges a widen action that reuses the NT answer (W_same_as, a fully shown type) zero calls;
               here it is charged the NT action's realised calls and tokens, since a router sending the field to the
               widen action must still make that reflection call. Struct's fields never reuse, so its cost is unchanged.
  cost model   development-estimated expected cost per model and action (NT or widen), the mean over the v5
               development fields whose routed action it is of the realised calls (one reflection call when the action
               runs, plus two verifier calls when it proposes a change; none for a widen action that reuses the NT
               answer) and of the mean-token estimates (development token logs, results/naacl/v6/v5/tokens). Frozen in
               docs/naacl/results/v5_matched_cost_frozen.json and committed before any test scoring.
  routing      whole fields in priority order (a fixed-seed random order inside ties); a field is routed while observed
               spend plus its expected cost fits the budget; observed spend charges its realised calls and mean-token
               estimates (test token logs, as v5.py); stop at the first field that does not fit. No fractional fields.
  budgets      per model, the structural router's realised spend (calls, or mean-token estimates), or the margin
               router's (frozen refitted t_m) realised calls.

Orderings, all from frozen quantities (docs/naacl/results/v5_frozen.json); nothing is fitted on test:
  benefit      descending u(x) of the frozen v5 benefit model (c_benefit). The frozen threshold tau sets v5's own
               budget only; at a matched budget the ordering by u is used (derived from the frozen u, not refitted).
  margin       declined fields first, then ascending canonical margin. The frozen margin router (t_m refitted on v5
               development, v4.1 t_s) routes exactly a prefix of this ordering (valid and margin <= t_m), so at its own
               budget it is that router.
  struct       its own fields (declined, or named type wider than 20 fields), the widen action.
  random       expected value at a routed share s of each model's fields (exact expectation, the routed action).

Stages:
  --stage costfit    development only: writes docs/naacl/results/v5_matched_cost_frozen.json. Checks that the local
                     development rows give the digest recorded in v5_frozen.json.
  --stage score      refuses unless the cost file is committed; scores development and test once. Writes
                     docs/naacl/results/v5_matched_facts.json (F-VFiveMatched*), paper/figures/data6/v5_matched_curves.csv.

Intervals: split-group bootstrap (group_stats.py, B = 2000, seed 20261009), paired within draws.
Usage: ITOPS_V21=<v2.1 label file> .venv/bin/python scripts/naacl/v5_matched.py --stage costfit|score
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["costfit", "score", "addfacts"], required=True)
ap.add_argument("--B", type=int, default=2000)
ARG = ap.parse_args()
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
sys.argv = ["v5.py", "--stage", "score", "--B", str(ARG.B)]
import v5 as V5  # noqa: E402  (its loader, features, frozen-model scoring; main() does not run on import)

GS = V5.GS
MODELS = V5.MODELS
FZ = json.loads((ROOT / "docs/naacl/results/v5_frozen.json").read_text())
COST_F = ROOT / "docs/naacl/results/v5_matched_cost_frozen.json"
FACTS_F = ROOT / "docs/naacl/results/v5_matched_facts.json"
CURVES_F = ROOT / "paper/figures/data6/v5_matched_curves.csv"
SEED = 20261011
S = FZ["intervention"]["outcome_field"]
MT = {"gptoss": "Gptoss", "llama4": "Llama", "granite": "Granite", "qwen35": "Qwen"}


def fix_costs(c):
    """Standalone strategy costs (a copied widen outcome is charged the NT action's calls and tokens): the shared
    implementation in v5.standalone_costs, which v5.load also applies to every cell."""
    return V5.standalone_costs(c)


def dev_digest(cells):
    return hashlib.sha256(json.dumps([[c.ds, c.model, c.n, float(c.keep.sum()), float(np.nansum(c.NT_v5)),
                                       float(np.nansum(c.W_v5))] for c in cells]).encode()).hexdigest()


def costfit():
    cells = V5.load("dev")
    dg = dev_digest(cells)
    assert dg == FZ["b_dev_outcomes"]["digest"], ("local v5 development rows differ from the frozen digest", dg)
    for c in cells:
        fix_costs(c)
    est = {}
    for m in MODELS:
        cs = [c for c in cells if c.model == m]
        isnt = np.concatenate([c.isnt for c in cs]) > 0
        ac, at = np.concatenate([c.Ac for c in cs]), np.concatenate([c.At for c in cs])
        wc, wt = np.concatenate([c.Wc for c in cs]), np.concatenate([c.Wt for c in cs])
        e = dict(n=int(len(ac)), n_nt=int(isnt.sum()),
                 calls_widen=float(np.nanmean(ac[~isnt])), tokens_widen=float(np.nanmean(at[~isnt])),
                 calls_widen_struct=float(np.nanmean(wc)), tokens_widen_struct=float(np.nanmean(wt)))
        if isnt.sum() >= 20:
            e.update(calls_nt=float(np.nanmean(ac[isnt])), tokens_nt=float(np.nanmean(at[isnt])), nt_source="model")
        est[m] = e
    pooled = [np.concatenate([c.Ac[c.isnt > 0] for c in cells]), np.concatenate([c.At[c.isnt > 0] for c in cells])]
    for m in MODELS:
        if "calls_nt" not in est[m]:
            est[m].update(calls_nt=float(np.nanmean(pooled[0])), tokens_nt=float(np.nanmean(pooled[1])),
                          nt_source="pooled over models (fewer than 20 development NT fields)")
    out = dict(date="2026-10-11", stage="costfit", base="docs/naacl/results/v5_frozen.json", dev_digest=dg, seed=SEED,
               note="Development-estimated expected second-pass cost per model and action on the v5 second pass: mean "
                    "realised calls and mean-token estimates over the development fields whose routed action it is. "
                    "Fitted on development only; no test output was read.", est=est)
    COST_F.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(est, indent=1))


# ============================================================================================ routing
def exp_cost(c, E, unit):
    e = E[c.model]
    nt = c.isnt > 0
    k = "calls" if unit == "calls" else "tokens"
    return np.where(nt, e[f"{k}_nt"], e[f"{k}_widen"])


def real_cost(c, unit):
    return np.nan_to_num(c.Ac if unit == "calls" else c.At)


def seq_fill(prio, ecost, rcost, budget, rng):
    tie = rng.random(len(prio))
    order = np.lexsort((tie, -np.asarray(prio, float)))
    w = np.zeros(len(prio))
    spent = 0.0
    for i in order:
        if spent + ecost[i] > budget + 1e-9:
            break
        w[i] = 1.0
        spent += rcost[i]
    return w, spent


def split_back(cs, v):
    out, i = [], 0
    for c in cs:
        out.append(v[i:i + c.n])
        i += c.n
    return out


def seq_route(cells, prio, unit, budgets, E):
    """{id: routed weights}; {model: (spent, budget)}."""
    W, MIS = {}, {}
    rng = np.random.default_rng(SEED)
    for m in MODELS:
        cs = [c for c in cells if c.model == m]
        w, spent = seq_fill(np.concatenate([prio[id(c)] for c in cs]), np.concatenate([exp_cost(c, E, unit) for c in cs]),
                            np.concatenate([real_cost(c, unit) for c in cs]), budgets[m], rng)
        MIS[m] = (spent, budgets[m])
        for c, a in zip(cs, split_back(cs, w)):
            W[id(c)] = a
    return W, MIS


def prepare(cells):
    bm = FZ["c_benefit"]
    mu, sd = np.array(bm["standardise"]["mu"]), np.array(bm["standardise"]["sd"])
    for c in cells:
        fix_costs(c)
        M_, D_ = V5.feats(c)
        c.u = V5.u_of(np.hstack([(M_ - mu) / sd, D_]), bm["model"])
        c.act = getattr(c, f"act_{S}")
        c.wok = getattr(c, f"W_{S}")
        mr = FZ["margin_router"][c.model]
        _, c.rm = V5.route_margin(c, mr["t_m"], mr["t_s"], S)
        c.r5 = (c.u > bm["tau"]).astype(float)
        c.mprio = -np.where(c.validb, c.margin, -1.0)


def outcome(c, w, kind="act"):
    """Final correctness of a policy routing w with the routed action (kind act) or the widen action (struct)."""
    return np.where(w > 0, c.wok if kind == "wid" else c.act, c.keep)


def cost_of(cells, W, kind="act"):
    N = sum(c.n for c in cells)
    if kind == "wid":
        return (sum(np.sum(W[id(c)] * np.nan_to_num(c.Wc)) for c in cells) / N,
                sum(np.sum(W[id(c)] * np.nan_to_num(c.Wt)) for c in cells) / N)
    return (sum(np.sum(W[id(c)] * real_cost(c, "calls")) for c in cells) / N,
            sum(np.sum(W[id(c)] * real_cost(c, "tokens")) for c in cells) / N)


# ============================================================================================ score
FACTS = []
DEF = ("Cost definition (v5 matched budgets, as F-DTwoCost): whole fields in priority order (fixed-seed random order "
       "inside ties); a field is routed while observed spend plus its development-estimated expected cost (per model "
       "and action, mean realised calls and mean-token estimates on v5 development) fits the budget; observed spend "
       "charges realised calls (one reflection call, plus two either-order verifier calls when the action proposes a "
       "change; a widen action that reuses the NT answer is charged the NT calls, which v5.py does not charge) and "
       "mean-token estimates (test token logs); stops at the first field that does not fit; no fractional "
       "fields. Budget per model: the comparator's realised spend.")


def fact(fid, claim, value, lo=None, hi=None):
    assert fid.isalpha(), fid
    FACTS.append(dict(id="F-VFiveMatched" + fid, claim=claim + ". " + DEF, value=str(value),
                      lo=None if lo is None else str(lo), hi=None if hi is None else str(hi),
                      src="docs/naacl/results/v5_matched_facts.json", fn="v5_matched.score", status="exploratory"))


def sg(x, nd=2):
    return f"{x:+.{nd}f}"


def committed(path):
    rel = str(path.relative_to(ROOT))
    log = subprocess.run(["git", "log", "-1", "--format=%H %cI", "--", rel], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", rel], capture_output=True, text=True).stdout.strip()
    if not log or dirty:
        sys.exit(f"{rel} must be committed and unchanged before scoring")
    return log


def score():
    commit = committed(COST_F)
    E = json.loads(COST_F.read_text())["est"]
    groups = GS.split_groups(V5.DATASETS, V5.ITOPS_DUP)
    CURVES = []
    fact("CostCommit", "commit and time of docs/naacl/results/v5_matched_cost_frozen.json (development-estimated costs)", commit)
    for m in MODELS:
        e = E[m]
        fact(f"Est{MT[m]}", f"{m}: development-estimated expected calls per routed field, widen / NT",
             f"{e['calls_widen']:.3f} / {e['calls_nt']:.3f}")
    for split in ("dev", "test"):
        sp = "Dev" if split == "dev" else ""
        lab = "development" if split == "dev" else "test"
        cells = V5.load(split)
        if split == "dev":
            assert dev_digest(cells) == FZ["b_dev_outcomes"]["digest"]
        prepare(cells)
        N = sum(c.n for c in cells)
        GB = GS.GroupBootstrap(cells, groups, B=ARG.B, seed=V5.SEED)
        ST = {id(c): c.struct.copy() for c in cells}
        bud = {}
        for m in MODELS:
            cs = [c for c in cells if c.model == m]
            bud[("struct", "calls", m)] = float(sum(np.sum(c.struct * np.nan_to_num(c.Wc)) for c in cs))
            bud[("struct", "tokens", m)] = float(sum(np.sum(c.struct * np.nan_to_num(c.Wt)) for c in cs))
            bud[("margin", "calls", m)] = float(sum(np.sum(c.rm * real_cost(c, "calls")) for c in cs))
        if split == "test":  # reproduce v5.py's headline: v5 minus struct at their own budgets
            e, d = GB.pooled(cells, lambda c: outcome(c, c.r5) - outcome(c, c.struct, "wid"))
            lo, hi = GS.interval(d)
            f5, f0 = cost_of(cells, {id(c): c.r5 for c in cells}), cost_of(cells, ST, "wid")
            old5 = sum(np.sum(c.r5 * np.nan_to_num(c.act_calls)) for c in cells) / N
            nre = int(sum(np.sum((c.r5 > 0) & c.reuse & (c.isnt == 0)) for c in cells))
            fact("OwnCallsAccounting", "test, v5's own budget: calls per field charged by v5.py (a widen action reusing "
                 "the NT answer costs nothing) / with that field charged its NT call and verifier calls (this analysis) / "
                 "v5-routed fields affected", f"{old5:.3f} / {f5[0]:.3f} / {nre:,}")
            fact("OwnVsStruct", "test, v5 (benefit router at its frozen tau) minus struct at their own, unmatched budgets "
                 f"(v5 {f5[0]:.2f} vs struct {f0[0]:.2f} calls per field), points", sg(e), sg(lo), sg(hi))
        pr_b = {id(c): c.u for c in cells}
        pr_m = {id(c): c.mprio for c in cells}
        runs = {}
        for unit in ("calls", "tokens"):
            b = {m: bud[("struct", unit, m)] for m in MODELS}
            runs[("Benefit", "Struct", unit)] = seq_route(cells, pr_b, unit, b, E)
            runs[("Margin", "Struct", unit)] = seq_route(cells, pr_m, unit, b, E)
        runs[("Benefit", "Margin", "calls")] = seq_route(cells, pr_b, "calls", {m: bud[("margin", "calls", m)] for m in MODELS}, E)
        RM = {id(c): c.rm for c in cells}
        for (rt, ref, unit), (W, MIS) in runs.items():
            ref_out = (lambda c: outcome(c, c.struct, "wid")) if ref == "Struct" else (lambda c: outcome(c, c.rm))
            g, gd = GB.pooled(cells, lambda c: outcome(c, W[id(c)]) - c.keep)
            glo, ghi = GS.interval(gd)
            dlt, dd = GB.pooled(cells, lambda c: outcome(c, W[id(c)]) - ref_out(c))
            dlo, dhi = GS.interval(dd)
            calls, toks = cost_of(cells, W)
            spent, b = sum(MIS[m][0] for m in MODELS), sum(MIS[m][1] for m in MODELS)
            worst = max(abs(MIS[m][0] - MIS[m][1]) / MIS[m][1] for m in MODELS if MIS[m][1] > 0)
            refname = "struct" if ref == "Struct" else "the margin router (frozen refitted t_m)"
            tg = f"{sp}{rt}At{ref}{unit.capitalize()}"
            desc = f"{lab}, {rt.lower()} router at {refname}'s {unit}"
            fact(tg + "Gain", f"{desc}: change over the first pass, points", sg(g), sg(glo), sg(ghi))
            fact(tg + "Vs", f"{desc}: minus {refname}, points (paired)", sg(dlt), sg(dlo), sg(dhi))
            fact(tg + "CallsPer", f"{desc}: realised second-pass calls per field", f"{calls:.3f}")
            fact(tg + "TokensPer", f"{desc}: second-pass tokens per field (mean-token estimates)", f"{toks:,.0f}")
            u_ = "calls" if unit == "calls" else "tokens"
            fact(tg + "Mismatch", f"{desc}: residual budget mismatch, observed spend minus budget, {u_} per field "
                 "(% of budget; largest per-model absolute %)",
                 f"{(spent - b) / N:+.4f} ({100 * (spent - b) / b:+.2f}%; max {100 * worst:.2f}%)")
            rep = float(sum(np.sum((c.keep == 0) & (outcome(c, W[id(c)]) == 1)) for c in cells))
            brk = float(sum(np.sum((c.keep == 1) & (outcome(c, W[id(c)]) == 0)) for c in cells))
            fact(tg + "Repairs", f"{desc}: repairs / breaks", f"{rep:.0f} / {brk:.0f}")
            print(split, tg, sg(g), f"calls {calls:.3f} tok {toks:.0f} vs {sg(dlt)} [{sg(dlo)}, {sg(dhi)}] mis {100*(spent-b)/b:+.2f}%")
        # reference rows at own budgets
        for nm, W, kind in (("Struct", ST, "wid"), ("MarginOwn", RM, "act"), ("BenefitOwn", {id(c): c.r5 for c in cells}, "act")):
            g, gd = GB.pooled(cells, lambda c: outcome(c, W[id(c)], kind) - c.keep)
            glo, ghi = GS.interval(gd)
            calls, toks = cost_of(cells, W, kind)
            what = {"Struct": "struct (widen action on its fields)", "MarginOwn": "the margin router at its frozen refitted t_m",
                    "BenefitOwn": "v5's benefit router at its frozen tau"}[nm]
            fact(f"{sp}{nm}Gain", f"{lab}, {what}, own budget: change over the first pass, points", sg(g), sg(glo), sg(ghi))
            fact(f"{sp}{nm}CallsPer", f"{lab}, {what}: realised second-pass calls per field", f"{calls:.3f}")
            fact(f"{sp}{nm}TokensPer", f"{lab}, {what}: second-pass tokens per field (mean-token estimates)", f"{toks:,.0f}")
        # accuracy-vs-calls curves
        tot = {m: float(sum(np.sum(real_cost(c, "calls")) for c in cells if c.model == m)) for m in MODELS}
        first = 100 * sum(c.keep.sum() for c in cells) / N
        for frac in np.round(np.linspace(0, 1, 41), 3):
            for nm, pr in (("benefit", pr_b), ("margin", pr_m)):
                W, _ = seq_route(cells, pr, "calls", {m: frac * tot[m] for m in MODELS}, E)
                acc = 100 * sum(outcome(c, W[id(c)]).sum() for c in cells) / N
                calls, toks = cost_of(cells, W)
                CURVES.append((split, nm, frac, calls, toks, acc, acc - first))
            sh = frac  # random at share = frac of each model's fields (exact expectation)
            acc = 100 * sum((c.keep + sh * (c.act - c.keep)).sum() for c in cells) / N
            calls = sum(sh * np.sum(real_cost(c, "calls")) for c in cells) / N
            toks = sum(sh * np.sum(real_cost(c, "tokens")) for c in cells) / N
            CURVES.append((split, "random", frac, calls, toks, acc, acc - first))
        for nm, W, kind in (("struct", ST, "wid"), ("margin_own", RM, "act"), ("benefit_own", {id(c): c.r5 for c in cells}, "act")):
            acc = 100 * sum(outcome(c, W[id(c)], kind).sum() for c in cells) / N
            calls, toks = cost_of(cells, W, kind)
            CURVES.append((split, nm, "", calls, toks, acc, acc - first))
        fact(f"{sp}Fields", f"{lab} field-model pairs (ten core datasets, four models)", f"{N:,}")
        fact(f"{sp}FirstAcc", f"{lab} first-pass accuracy, %", f"{first:.2f}")
    FACTS_F.write_text(json.dumps(dict(generated_by="scripts/naacl/v5_matched.py --stage score",
                                       cost_frozen=str(COST_F.relative_to(ROOT)), cost_frozen_commit=commit,
                                       base_frozen="docs/naacl/results/v5_frozen.json", status="exploratory",
                                       definition=DEF, facts=FACTS), indent=1) + "\n")
    with open(CURVES_F, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "router", "budget_frac", "calls_per_field", "tokens_per_field", "accuracy", "gain"])
        for r in CURVES:
            w.writerow(list(r[:3]) + [f"{x:.6g}" for x in r[3:]])
    print("wrote", FACTS_F, len(FACTS))


def addfacts():
    """Two summary facts read from the stored facts (no rescoring): the matched-call residual of benefit against the
    schema-size baseline, and the mean-token cost definition."""
    D = json.loads(FACTS_F.read_text())
    F_ = {f["id"]: f for f in D["facts"]}
    mis = F_["F-VFiveMatchedBenefitAtStructCallsMismatch"]["value"]
    pct = mis.split("(")[1].split("%")[0]
    new = [dict(id="F-VFiveMatchedResidualBenefitStructCalls",
                claim="test, benefit selection matched to the schema-size baseline's calls: residual cost mismatch, observed "
                      "calls minus the baseline's, % of its calls (the sequential rule admits a field on its expected cost and "
                      "then charges its observed cost, so the match is approximate, not a strict cap). " + DEF,
                value=pct + "%", lo=None, hi=None, src=str(FACTS_F.relative_to(ROOT)), fn="v5_matched.addfacts",
                status="exploratory"),
           dict(id="F-VFiveMatchedTokenCostDefinition", claim="token cost definition of the v5 matched comparisons",
                value="Tokens per field are mean-token estimates, not exact per-call totals: each realised call is charged "
                      "the mean prompt plus completion tokens of its stage (reflection or verifier) for that dataset and model "
                      "in the test token logs (results/naacl/v6/v5/tokens), falling back to the model's mean; expected token "
                      "costs for selection use the same means from the development logs.",
                lo=None, hi=None, src=str(FACTS_F.relative_to(ROOT)), fn="v5_matched.addfacts", status="exploratory")]
    D["facts"] = [f for f in D["facts"] if f["id"] not in {x["id"] for x in new}] + new
    FACTS_F.write_text(json.dumps(D, indent=1) + "\n")
    print(new[0]["value"])


if __name__ == "__main__":
    {"costfit": costfit, "score": score, "addfacts": addfacts}[ARG.stage]()
