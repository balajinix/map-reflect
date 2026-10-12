#!/usr/bin/env python3
"""Confidence-estimate analyses on Map-Reflect v5's own second pass and inputs (task 1a). Offline; no model call.
Exploratory throughout.

Second pass: the stored v5 test rows (results/naacl/v6/v5/test; both actions measured on every field), the field's
routed action (NT or widen by v4.1's t_s), the either-order verifier, with the cost accounting of
scripts/naacl/v5_matched.py (a widen action reusing the NT answer is charged the NT calls). Core: the ten core
datasets; held-out: the five held-out sets of v5 (v5.HELDOUT), the same second pass.

Estimates (higher = more confident), per answered field (declined fields are reflected on by every selection alike,
so they are left out, as in the earlier analysis):
  free    margin (canonical), top score, top-2 ratio, score entropy (negated): the v5 first pass (results/naacl/v2/first)
  paid    stated confidence and P(True): results/naacl/v3/uq, with the ff1 (full-description) R-UQ row replacing the
          stored one for every field whose prompt ff1 changed (docs/naacl/results/ff1_rerun_keys.json; rows in
          results/naacl/v6/ff1rr/uq, all four models), the merge of ff1_rerun_score.merge. An unresolved stated answer
          is missing, not 1 - c (the audit's defect; uq_ablation's V6_UQ_VALID_ONLY rule).
  token   answer-token log-probability: results/naacl/uq_ablation/logprob, then the U9 rows (results/naacl/v6/u9, core;
          results/naacl/v6/u9ho, held-out; a .fill file replaces a missing value) fill what is missing.
  samples answer entropy over five samples (negated): results/naacl/v3/uq, then the U9 sample rows fill what is missing.
  combo   logistic over margin, top score, P(True) and stated confidence, cross-fitted over split groups per model
          (uq_ablation_v4.crossfit_groups) on the population.
Population: per model and set, an estimate is available when it has a valid value on at least 90% of the answered
fields; the model's population is the answered fields where every available estimate is valid (the common valid
population). Pooled comparisons use the models where both estimates are available.

Analyses:
  selection   at Map-Reflect v5's calls (its benefit selection at the frozen tau, on the population fields, per model),
              the matched-cost rule of v5_matched (development-estimated costs, sequential stopping, whole fields,
              fixed-seed random order inside ties): change over the first pass, minus the margin and minus Map-Reflect
              (paired, split-group bootstrap B = 2000), oracle share (oracle = the fields the action repairs first).
  curves      accuracy against reflection rate, 0 to 100% of each model's population fields in steps of 5%, exact
              expectation inside ties (uq_ablation.gate_weights), with random and oracle (C1).
  detection   error AUROC on the common valid population, ranks within model when pooled, split-group CIs; AUPRC, AURC;
              the margin's AUROC on every answered field as well.
  C12         per estimate and model: AUROC, the margin's AUROC, and the selection change minus the margin.

Writes docs/naacl/results/v5_signals_facts.json (F-VFiveSig*), paper/figures/data6/v5_c1_curves.csv and
paper/figures/data6/v5_c12_detect_vs_select.csv.
Usage: ITOPS_V21=<v2.1 label file> .venv/bin/python scripts/naacl/v5_signals.py
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
B = 2000
sys.argv = ["v5_matched.py", "--stage", "score", "--B", str(B)]
import v5_matched as VM  # noqa: E402
os.environ["V6_UQ_VALID_ONLY"] = "1"
LABELS = os.environ["ITOPS_V21"]
sys.argv = ["uq_ablation.py", "--no-paper", "--labels-v21", LABELS]
import uq_ablation as U  # noqa: E402
import uq_ablation_v4 as UQ4  # noqa: E402

V5, GS, MODELS = VM.V5, VM.GS, VM.MODELS
FACTS_F = ROOT / "docs/naacl/results/v5_signals_facts.json"
DATA = ROOT / "paper/figures/data6"
SIGS = ["margin", "s1", "ratio", "entropy", "lp_answer", "verbalised", "p_true", "sem_entropy", "combo_paid"]
LABEL = {"margin": "margin", "s1": "top score", "ratio": "top-2 ratio", "entropy": "score entropy",
         "lp_answer": "answer-token log-probability", "verbalised": "stated confidence", "p_true": "P(True)",
         "sem_entropy": "answer entropy (five samples)", "combo_paid": "logistic (margin, top score, P(True), stated)"}
TAG = UQ4.MAC
SHARES = [round(0.05 * i, 2) for i in range(21)]
KEYS = json.load(open(ROOT / "docs/naacl/results/ff1_rerun_keys.json"))["uq"]
FF = ROOT / "results/naacl/v6/ff1rr/uq"
COUNT = defaultdict(int)


def read(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []


def paid_rows(ds, m):
    """key -> {verbalised, p_true}: the stored rows with the ff1 rerun rows merged where ff1 changed the prompt."""
    stored = read(ROOT / f"results/naacl/v3/uq/{ds}__{m}.jsonl")
    info = KEYS.get(f"{ds}__{m}")
    if info:
        keys = {l.strip() for l in open(ROOT / info["keys"]) if l.strip()}
        if keys:
            assert (FF / f"{ds}__{m}.jsonl.done").exists(), f"ff1 rerun unfinished: {ds}__{m}"
        new = {r["key"]: r for r in read(FF / f"{ds}__{m}.jsonl")}
        miss = [k for k in keys if k not in new]
        assert not miss, (ds, m, len(miss))
        stored = [new[r["key"]] if r["key"] in keys else r for r in stored]
        COUNT["ff1_rows"] += sum(r["key"] in keys for r in stored)
    out = {}
    for r in stored:
        d = {}
        if r.get("p_true") is not None:
            d["p_true"] = float(r["p_true"])
        if r.get("verbalised") is not None:
            if r.get("verbalised_answer"):
                d["verbalised"] = float(r["verbalised"])
            else:
                COUNT["stated_unresolved"] += 1
        out[r["key"]] = d
    return out


def other_rows(ds, m, heldout):
    """key -> {lp_answer, sem_entropy}: uq_ablation's sources, then U9 (core) or U9 held-out rows fill what is missing."""
    base = U.uq_rows(ds, m)
    out = {k: {s: v[s] for s in ("lp_answer", "sem_entropy") if s in v} for k, v in base.items()}
    d = ROOT / ("results/naacl/v6/u9ho" if heldout else "results/naacl/v6/u9")
    for sig, s_, neg, suf in (("logprob", "lp_answer", False, ""), ("logprob", "lp_answer", False, ".fill"),
                              ("samples", "sem_entropy", True, "")):
        for r in read(d / sig / f"{ds}__{m}{suf}.jsonl"):
            if r.get(s_) is None:
                continue
            x = out.setdefault(r["key"], {})
            if s_ not in x:
                x[s_] = -float(r[s_]) if neg else float(r[s_])
    return out


def mask_cell(c, mk):
    x = V5.Cell()
    for k, v in c.__dict__.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (c.n,):
            setattr(x, k, v[mk])
        elif isinstance(v, dict) and all(isinstance(a, np.ndarray) and a.shape[:1] == (c.n,) for a in v.values()):
            setattr(x, k, {a: b[mk] for a, b in v.items()})
        elif isinstance(v, list) and len(v) == c.n:
            setattr(x, k, [y for y, t in zip(v, mk) if t])
        else:
            setattr(x, k, v)
    x.n = int(mk.sum())
    return x


def build(heldout):
    cells = V5.load("test", datasets=V5.HELDOUT if heldout else None)
    VM.prepare(cells)
    out = []
    for c in cells:
        fp = U.first_pass(c.ds, c.model, V5.ids_of(c.ds, "test"))
        pr, ot = paid_rows(c.ds, c.model), other_rows(c.ds, c.model, heldout)
        S = {s: np.full(c.n, np.nan) for s in SIGS if s != "combo_paid"}
        for i, k in enumerate(c.keys):
            rec = fp.get(k)
            if rec is not None and c.validb[i]:
                v = U.score_signals(rec)
                S["margin"][i] = c.margin[i]
                for s in ("s1", "ratio", "entropy"):
                    S[s][i] = v[s]
            for s, val in {**pr.get(k, {}), **ot.get(k, {})}.items():
                if s in S:
                    S[s][i] = val
        c.S = S
        x = mask_cell(c, c.validb.copy())
        x.S = {s: v[c.validb] for s, v in S.items()}
        out.append(x)
    return out


def population(cells):
    """Per model: available estimates and the common valid population (cells masked)."""
    avail, pop = {}, {}
    for m in MODELS:
        cs = [c for c in cells if c.model == m]
        n = sum(c.n for c in cs)
        avail[m] = [s for s in SIGS if s != "combo_paid" and sum(np.isfinite(c.S[s]).sum() for c in cs) >= 0.9 * n]
        pm = []
        for c in cs:
            mk = np.all([np.isfinite(c.S[s]) for s in avail[m]], axis=0)
            if mk.any():
                x = mask_cell(c, mk)
                x.S = {s: v[mk] for s, v in c.S.items()}
                pm.append(x)
        pop[m] = pm
    return avail, pop


def add_combo(avail, pop, groups):
    feats = ["margin", "s1", "p_true", "verbalised"]
    for m in MODELS:
        if not all(f in avail[m] for f in feats):
            continue
        fields = [dict(ds=c.ds, unit=groups[(c.ds, str(s))], base=c.keep[i], sig={f: c.S[f][i] for f in feats})
                  for c in pop[m] for i, s in enumerate(c.sid)]
        UQ4.crossfit_groups(fields, feats, "combo_paid")
        i = 0
        for c in pop[m]:
            c.S["combo_paid"] = np.array([fields[i + j]["sig"]["combo_paid"] for j in range(c.n)])
            i += c.n
        avail[m].append("combo_paid")


FACTS = []
PROV = ("Provenance: v5 test rows (results/naacl/v6/v5/test), routed action under the either-order verifier; free "
        "estimates from the v5 first pass; stated confidence and P(True) from results/naacl/v3/uq with the ff1 R-UQ "
        "rows merged where ff1 changed the prompt (all four models; an unresolved stated answer counts as missing); token "
        "log-probability and answer entropy from uq_ablation and the U9 rows; answered fields, common valid "
        "population; matched cost = Map-Reflect v5's calls per model, development-estimated costs with sequential "
        "stopping (v5_matched); split-group bootstrap, B = 2000")


def fact(fid, claim, value, lo=None, hi=None):
    assert fid.isalpha(), fid
    FACTS.append(dict(id="F-VFiveSig" + fid, claim=claim + ". " + PROV, value=str(value),
                      lo=None if lo is None else str(lo), hi=None if hi is None else str(hi),
                      src="docs/naacl/results/v5_signals_facts.json", fn="v5_signals", status="exploratory"))


def sg(x, nd=2):
    return f"{x:+.{nd}f}"


def seq_route(cells, prio, unit, budgets, E):
    """v5_matched.seq_route over the models present in cells only."""
    W = {}
    rng = np.random.default_rng(VM.SEED)
    for m in MODELS:
        cs = [c for c in cells if c.model == m]
        if not cs:
            continue
        w, _ = VM.seq_fill(np.concatenate([prio[id(c)] for c in cs]), np.concatenate([VM.exp_cost(c, E, unit) for c in cs]),
                           np.concatenate([VM.real_cost(c, unit) for c in cs]), budgets[m], rng)
        for c, a in zip(cs, VM.split_back(cs, w)):
            W[id(c)] = a
    return W


def bw(GB, cells):
    return np.concatenate([GB.counts[c.ds][:, [GB.index[c.ds][GB.unit(c.ds, s)] for s in c.sid]] for c in cells], axis=1)


def analyse(tagset, heldout, E, C1, C12):
    lab = "held-out" if heldout else "core test"
    sp = "HO" if heldout else ""
    cells = build(heldout)
    groups = GS.split_groups(V5.HELDOUT, None) if heldout else GS.split_groups(V5.DATASETS, V5.ITOPS_DUP)
    avail, pop = population(cells)
    add_combo(avail, pop, groups)
    allpop = [c for m in MODELS for c in pop[m]]
    GB = GS.GroupBootstrap(allpop, groups, B=B, seed=V5.SEED)
    nall = sum(c.n for c in cells)
    fact(f"{sp}Answered", f"{lab}: answered field-model pairs", f"{nall:,}")
    fact(f"{sp}Pop", f"{lab}: common valid population (answered fields with every available estimate valid)",
         f"{sum(c.n for c in allpop):,}")
    for m in MODELS:
        fact(f"{sp}Avail{VM.MT[m]}", f"{lab}, {m}: estimates available (valid on at least 90% of answered fields)",
             ", ".join(LABEL[s] for s in avail[m]))
    # margin AUROC on every answered field (pooled ranks within model)
    err_all = np.concatenate([1 - c.keep for m in MODELS for c in cells if c.model == m])
    cm_all = np.concatenate([U.pct_rank(np.concatenate([c.margin for c in cells if c.model == m])) for m in MODELS])
    GBa = GS.GroupBootstrap(cells, groups, B=B, seed=V5.SEED)
    order_cells = [c for m in MODELS for c in cells if c.model == m]
    lo, hi = GS.interval(U.auroc_draws(cm_all, err_all, bw(GBa, order_cells)))
    fact(f"{sp}MarginFullPopAUC", f"{lab}: the margin's error AUROC on every answered field (ranks within model)",
         f"{U.auroc_w(cm_all, err_all):.3f}", f"{lo:.3f}", f"{hi:.3f}")

    def models_for(s):
        return [m for m in MODELS if s in avail[m] and pop[m]]

    def sel(ms, s):
        """Weights per cell routing estimate s (or 'oracle', 'random', 'main') at Map-Reflect's calls per model."""
        cs = [c for m in ms for c in pop[m]]
        bud = {m: float(sum(np.sum(c.r5 * VM.real_cost(c, "calls")) for c in pop[m])) for m in ms}
        if s == "main":
            return cs, {id(c): c.r5 for c in cs}
        if s == "random":
            tot = {m: float(sum(np.sum(VM.real_cost(c, "calls")) for c in pop[m])) for m in ms}
            return cs, {id(c): np.full(c.n, bud[c.model] / tot[c.model]) for c in cs}
        prio = {id(c): (c.act - c.keep) if s == "oracle" else -c.S[s] for c in cs}
        return cs, seq_route(cs, prio, "calls", bud, E)

    def acc_vec(c, W):
        return c.keep + W[id(c)] * (c.act - c.keep)
    res = {}
    for s in SIGS:
        ms = models_for(s)
        if not ms:
            continue
        cs, W = sel(ms, s)
        _, Wm = sel(ms, "margin")
        _, Wo = sel(ms, "oracle")
        _, W5 = sel(ms, "main")
        N = sum(c.n for c in cs)
        g, gd = GB.pooled(cs, lambda c: acc_vec(c, W) - c.keep)
        go, _ = GB.pooled(cs, lambda c: acc_vec(c, Wo) - c.keep)
        dm, dmd = GB.pooled(cs, lambda c: acc_vec(c, W) - acc_vec(c, Wm))
        d5, d5d = GB.pooled(cs, lambda c: acc_vec(c, W) - acc_vec(c, W5))
        calls = sum(np.sum(W[id(c)] * VM.real_cost(c, "calls")) for c in cs) / N
        rate = 100 * sum(np.sum(W[id(c)]) for c in cs) / N
        t = f"{sp}{TAG[s]}"
        mods = ", ".join(ms)
        fact(t + "SelGain", f"{lab}: selection by {LABEL[s]} at Map-Reflect v5's calls, change over the first pass, points ({mods})",
             sg(g), *map(sg, GS.interval(gd)))
        if s != "margin":
            fact(t + "SelVsMargin", f"{lab}: selection by {LABEL[s]} minus selection by the margin, same calls and fields, points ({mods})",
                 sg(dm), *map(sg, GS.interval(dmd)))
        fact(t + "SelVsMain", f"{lab}: selection by {LABEL[s]} minus Map-Reflect v5 (benefit classifier at its tau), points ({mods})",
             sg(d5), *map(sg, GS.interval(d5d)))
        fact(t + "SelOracleShare", f"{lab}: selection by {LABEL[s]}: share of the oracle's change at the same calls, % ({mods})",
             f"{100 * g / go:.0f}" if go > 0 else "n/a")
        fact(t + "SelRate", f"{lab}: selection by {LABEL[s]}: reflection rate at Map-Reflect's calls, % ({mods})", f"{rate:.1f}")
        fact(t + "SelCalls", f"{lab}: selection by {LABEL[s]}: second-pass calls per field ({mods})", f"{calls:.3f}")
        res[s] = (g, dm)
        # detection on the common valid population, pooled ranks within model
        fs_cells = [c for m in ms for c in pop[m]]
        err = np.concatenate([1 - c.keep for c in fs_cells])
        cv = np.concatenate([U.pct_rank(np.concatenate([c.S[s] for c in pop[m]])) for m in ms])
        cmg = np.concatenate([U.pct_rank(np.concatenate([c.S["margin"] for c in pop[m]])) for m in ms])
        Wb = bw(GB, fs_cells)
        a, amg = U.auroc_w(cv, err), U.auroc_w(cmg, err)
        dr = U.auroc_draws(cv, err, Wb)
        lo, hi = GS.interval(dr)
        fact(t + "AUC", f"{lab}: error AUROC of {LABEL[s]} on the common valid population (ranks within model; {mods})",
             f"{a:.3f}", f"{lo:.3f}", f"{hi:.3f}")
        if s != "margin":
            lo2, hi2 = GS.interval(dr - U.auroc_draws(cmg, err, Wb))
            fact(t + "AUCVsMargin", f"{lab}: AUROC of {LABEL[s]} minus the margin's, same fields ({mods})", f"{a - amg:+.3f}",
                 f"{lo2:+.3f}", f"{hi2:+.3f}")
        fact(t + "AUPRC", f"{lab}: AUPRC (errors positive) of {LABEL[s]} ({mods})", f"{U.auprc(cv, err):.3f}")
        fact(t + "AURC", f"{lab}: area under risk-coverage of {LABEL[s]} ({mods})", f"{U.aurc(cv, err):.4f}")
        # C1 curves: accuracy against reflection rate (exact expectation), with random and oracle
        cur = {}
        for kind in (s, "margin", "random", "oracle"):
            accs = []
            for b in SHARES:
                tot = n = 0.0
                for m in ms:
                    base = np.concatenate([c.keep for c in pop[m]])
                    o = np.concatenate([c.act for c in pop[m]])
                    k = b * len(base)
                    if kind == "random":
                        w = np.full(len(base), b)
                    elif kind == "oracle":
                        w = U.gate_weights(-(o - base), k)
                    else:
                        w = U.gate_weights(np.concatenate([c.S[kind] for c in pop[m]]), k)
                    tot += float(np.sum(base + w * (o - base)))
                    n += len(base)
                accs.append(100 * tot / n)
            cur[kind] = accs
        for b, a_, am_, ar_, ao_ in zip(SHARES, cur[s], cur["margin"], cur["random"], cur["oracle"]):
            C1.append((tagset, s, "+".join(ms), b, a_, am_, ar_, ao_))
        k0, every = cur[s][0], cur[s][-1]
        gain = np.array(cur[s]) - k0
        reach = next((b for b, x in zip(SHARES, gain) if x >= 0.95 * (every - k0)), 1.0)
        for b in (0.1, 0.2, 0.3, 0.5):
            fact(f"{t}RateAt{ {0.1: 'Ten', 0.2: 'Twenty', 0.3: 'Thirty', 0.5: 'Fifty'}[b]}",
                 f"{lab}: change when {LABEL[s]} selects {int(100 * b)}% of fields (reflection rate), points ({mods})",
                 sg(float(gain[SHARES.index(b)])))
        fact(t + "RateMean", f"{lab}: mean change over reflection rates 0 to 100% in steps of 5, {LABEL[s]} ({mods})",
             sg(float(gain.mean())))
        fact(t + "RateReach", f"{lab}: smallest reflection rate reaching 95% of the all-field change, {LABEL[s]}, % ({mods})",
             f"{100 * reach:.0f}")
        # C12: per model
        if s != "margin":
            for m in ms:
                cm_ = pop[m]
                errm = np.concatenate([1 - c.keep for c in cm_])
                am_ = U.auroc_w(np.concatenate([c.S[s] for c in cm_]), errm)
                amm = U.auroc_w(np.concatenate([c.S["margin"] for c in cm_]), errm)
                _, Wsm = sel([m], s)
                _, Wmm = sel([m], "margin")
                dd, ddd = GB.pooled(cm_, lambda c: acc_vec(c, Wsm) - acc_vec(c, Wmm))
                lo3, hi3 = GS.interval(ddd)
                C12.append((tagset, s, m, am_, amm, dd, lo3, hi3))
    # correlation of AUROC advantage and selection change over (estimate, model)
    rows = [r for r in C12 if r[0] == tagset]
    if len(rows) > 2:
        x = np.array([r[3] - r[4] for r in rows])
        y = np.array([r[5] for r in rows])
        fact(f"{sp}DetectSelectCorr", f"{lab}: correlation over (estimate, model) of the AUROC advantage over the margin and "
             "the selection change minus the margin", f"{np.corrcoef(x, y)[0, 1]:.2f}")
        fact(f"{sp}DetectSelectPairs", f"{lab}: (estimate, model) pairs in that correlation", f"{len(rows)}")
    every = sum(np.sum(c.act - c.keep) for c in allpop) / sum(c.n for c in allpop) * 100
    fact(f"{sp}EveryField", f"{lab}: change when reflecting on every population field, points (all models)", sg(every))


def main():
    E = json.loads(VM.COST_F.read_text())["est"]
    C1, C12 = [], []
    for tagset, ho in (("core", False), ("heldout", True)):
        analyse(tagset, ho, E, C1, C12)
        print(tagset, "done")
    fact("FFRows", "stored stated-confidence/P(True) rows replaced by their ff1 rerun row (core and held-out)", f"{COUNT['ff1_rows']:,}")
    fact("StatedUnresolved", "stated-confidence rows with an unresolved answer, counted as missing", f"{COUNT['stated_unresolved']:,}")
    FACTS_F.write_text(json.dumps(dict(generated_by="scripts/naacl/v5_signals.py", status="exploratory",
                                       provenance=PROV, facts=FACTS), indent=1) + "\n")
    with open(DATA / "v5_c1_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "estimate", "models", "rate", "accuracy", "margin_accuracy_same_models", "random_accuracy", "oracle_accuracy"])
        for r in C1:
            w.writerow(list(r[:4]) + [f"{x:.6g}" for x in r[4:]])
    with open(DATA / "v5_c12_detect_vs_select.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "estimate", "model", "auroc", "margin_auroc", "selection_delta", "lo", "hi"])
        for r in C12:
            w.writerow(list(r[:3]) + [f"{x:.6g}" for x in r[3:]])
    print("wrote", FACTS_F, len(FACTS))
    for f in FACTS:
        if any(k in f["id"] for k in ("SelVsMargin", "SelVsMain", "AUC", "Pop", "Corr", "FFRows", "Unresolved", "Avail")):
            print(f["id"], f["value"], f["lo"] or "", f["hi"] or "")


if __name__ == "__main__":
    main()
