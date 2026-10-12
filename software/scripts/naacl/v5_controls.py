#!/usr/bin/env python3
"""Three free selection controls on the method v5 second pass (Codex review 3926722, task C). Offline; no model call.
Exploratory.

Every control selects fields for the same measured v5 second pass (results/naacl/v6/v5, the field's routed action:
NT or widen by v4.1's t_s) and is scored two ways:
  matched   at Map-Reflect v5's per-model cost (its benefit selection at the frozen tau), with the cost model and the
            sequential stopping rule of the v5 matched comparison (scripts/naacl/v5_matched.py,
            docs/naacl/results/v5_matched_cost_frozen.json): whole fields in priority order (fixed-seed random order
            inside ties), routed while observed spend plus the development-estimated expected cost fits.
  own       at the control's own threshold, fitted on v5 development outcomes (frozen before test scoring).

Controls, each a free estimate from the stored first pass (results/naacl/v2/first/given):
  lexical   an earlier lexical-overlap correctness classifier (scripts/uq_classifier.py): a logistic model of
            first-pass error over margin, top score, number of ranked candidates, length of the top reason, an
            "exact match / same name" marker in that reason, Jaccard overlap of source and predicted-target column
            tokens, and MiniLM (all-MiniLM-L6-v2) cosine of the source field name and the predicted target's
            description. Same optimiser as the original (gradient descent, 3,000 steps, rate 0.1, L2 0.01,
            standardised features). Fitted on v5 development first passes, all four models pooled. Selects the
            highest predicted error first.
  composite the legacy composite u(f) = 0.3 (1 - top score) + 0.3 (1 - min(1.5 x score range, 1)) + 0.2 x [no
            reasons] + 0.06 of the earlier method (map_reflect_runner.estimate_uncertainty_from_response), read from
            the stored first pass (predictions[field].uncertainty, method "heuristic"). Selects the highest u(f).
            Its deployed threshold was 0.25; the own threshold here is refitted on v5 development.
  trigger   the error trigger: the runner's detect_mapping_errors returned any message (stored
            predictions[field].detected_errors). Binary; at a matched cost the fields it fires on go first (random
            order inside), then the rest (random order). Its own setting is the fields it fires on; nothing is fitted.
Own thresholds (lexical, composite): on development, the threshold over a quantile grid that maximises development
accuracy under the v5 second pass, ties to fewer calls; the lexical classifier's development scores are cross-fitted
(5-fold split-group), as v5's tau was.

Stages: --stage fit (development only; writes docs/naacl/results/v5_controls_frozen.json) and --stage score (refuses
unless that file is committed; writes docs/naacl/results/v5_controls_facts.json, F-VFiveCtl*, and
paper/tab6_v5_controls.tex).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["fit", "score", "table"], required=True)
ap.add_argument("--B", type=int, default=2000)
ARG = ap.parse_args()
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
sys.argv = ["v5_matched.py", "--stage", "score", "--B", str(ARG.B)]
import v5_matched as VM  # noqa: E402  (its cost model, sequential fill and v5 loader; main() does not run on import)

V5, GS, MODELS = VM.V5, VM.GS, VM.MODELS
R2 = V5.R2
FROZEN_F = ROOT / "docs/naacl/results/v5_controls_frozen.json"
FACTS_F = ROOT / "docs/naacl/results/v5_controls_facts.json"
TAB_F = ROOT / "paper/tab6_v5_controls.tex"
SEED = 20261011
LEX = ["margin", "g1", "n_cand", "reason_len", "reason_exact", "name_overlap", "emb_cos"]
_EXACT = re.compile(r"exact match|same name|identical|same table and column", re.I)
_TOK = re.compile(r"[a-z0-9]+")


def col_of(fid):
    if "__" in fid:
        return fid.rsplit("__", 1)[1]
    if "." in fid:
        return fid.rsplit(".", 1)[1]
    return fid


def toks(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(s))
    return set(_TOK.findall(s.lower()))


_LK = {}


def target_text(ds, tid):
    if ds not in _LK:
        p = ROOT / f"data/{ds}_api_lookup.json"
        _LK[ds] = json.load(open(p)) if p.exists() else {}
    e = _LK[ds].get(tid)
    return (e.get("description") or e.get("name") or tid) if isinstance(e, dict) else (tid or "")


_ST = []


def embed(texts):
    if not _ST:
        from sentence_transformers import SentenceTransformer
        _ST.append(SentenceTransformer("all-MiniLM-L6-v2"))
    return _ST[0].encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=256)


def attach(cells, split):
    """Per field: the lexical features, the stored composite u(f) and the stored error trigger."""
    for c in cells:
        ids = V5.ids_of(c.ds, split)
        d = R2.load_slot(V5.first_slot(c.ds, c.model))
        res = {str(r.get("sample_id")): r for r in d["results"]}
        recs = {r["key"]: r for r in R2.records(d, ids)}
        F = {k: [] for k in LEX + ["comp", "trig", "src", "tgt"]}
        for k in c.keys:
            rec = recs[k]
            p = (res[rec["sample_id"]].get("predictions") or {}).get(rec["field"]) or {}
            rk = rec["ranking"]
            pred = rec["base"] or ""
            reason = str(rk[0].get("reason", "")) if rk else ""
            sc = [R2._num(x.get("score")) for x in rk]
            sc = [x for x in sc if x != float("-inf")]
            st, tt = toks(col_of(rec["field"])), toks(col_of(pred))
            F["margin"].append(float(V5.router.canonical_margin(float(rec["margin"]))))
            F["g1"].append(sc[0] if sc else 0.0)
            F["n_cand"].append(float(len(rk)))
            F["reason_len"].append(float(len(reason)))
            F["reason_exact"].append(float(bool(_EXACT.search(reason))))
            F["name_overlap"].append(len(st & tt) / len(st | tt) if (st | tt) else 0.0)
            F["src"].append(col_of(rec["field"]).replace("_", " "))
            F["tgt"].append(target_text(c.ds, pred) if pred else "")
            u = p.get("uncertainty")
            assert u is not None and (p.get("uncertainty_info") or {}).get("method") == "heuristic", (c.ds, c.model, k)
            F["comp"].append(float(u))
            de = p.get("detected_errors")
            assert de is not None, (c.ds, c.model, k)
            F["trig"].append(float(len(de) > 0))
        a, b = embed(F["src"]), embed(F["tgt"])
        F["emb_cos"] = list((a * b).sum(1))
        for kk in LEX + ["comp", "trig"]:
            setattr(c, "x_" + kk, np.array(F[kk], float))
        c.lexX = np.column_stack([getattr(c, "x_" + f) for f in LEX])


def fit_logistic(X, y, iters=3000, lr=0.1, l2=1e-2):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w)
        b -= lr * g.mean()
    return w, b


def own_threshold(score, act, keep, calls):
    """Route score > t; t over a quantile grid (and below every score) maximising accuracy, ties to fewer calls."""
    grid = np.unique(np.r_[np.quantile(score, np.linspace(0, 1, 101)), score.min() - 1, score.max() + 1])
    best = None
    for t in grid:
        r = score > t
        cand = (float(np.where(r, act, keep).mean()), -float(calls[r].sum()), float(t), float(r.mean()))
        if best is None or cand[:2] > best[:2]:
            best = cand
    return best


def fit():
    from sklearn.model_selection import GroupKFold
    cells = V5.load("dev")
    assert VM.dev_digest(cells) == VM.FZ["b_dev_outcomes"]["digest"]
    VM.prepare(cells)
    attach(cells, "dev")
    X = np.vstack([c.lexX for c in cells])
    y = 1.0 - np.concatenate([c.keep for c in cells])
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    w, b = fit_logistic((X - mu) / sd, y)
    groups = GS.split_groups(V5.DATASETS, V5.ITOPS_DUP)
    gid = np.unique([f"{c.ds}::{groups[(c.ds, s)]}" for c in cells for s in c.sid], return_inverse=True)[1]
    cf = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, gid):
        m_, s_ = X[tr].mean(0), X[tr].std(0)
        s_ = np.where(s_ > 0, s_, 1.0)
        w_, b_ = fit_logistic((X[tr] - m_) / s_, y[tr])
        cf[te] = 1 / (1 + np.exp(-(((X[te] - m_) / s_) @ w_ + b_)))
    act = np.concatenate([c.act for c in cells])
    keep = np.concatenate([c.keep for c in cells])
    calls = np.concatenate([VM.real_cost(c, "calls") for c in cells])
    comp = np.concatenate([c.x_comp for c in cells])
    trig = np.concatenate([c.x_trig for c in cells])
    lx = own_threshold(cf, act, keep, calls)
    cp = own_threshold(comp, act, keep, calls)
    out = dict(date="2026-10-11", stage="fit", note="Fitted on v5 development outcomes only; no test output was read.",
               dev_digest=VM.FZ["b_dev_outcomes"]["digest"], cost_frozen="docs/naacl/results/v5_matched_cost_frozen.json",
               lexical=dict(features=LEX, mu=mu.tolist(), sd=sd.tolist(), w=w.tolist(), b=float(b),
                            optimiser="gradient descent, 3000 steps, rate 0.1, L2 0.01 (scripts/uq_classifier.py)",
                            embedder="sentence-transformers all-MiniLM-L6-v2", target="first-pass error",
                            tau=lx[2], tau_rule="route when cross-fitted P(error) > tau; tau maximises development accuracy, ties to fewer calls",
                            dev_accuracy=lx[0], dev_rate=lx[3]),
               composite=dict(definition="0.3(1-top score) + 0.3(1-min(1.5 range,1)) + 0.2[no reasons] + 0.06 (stored)",
                              deployed_tau=0.25, tau=cp[2], dev_accuracy=cp[0], dev_rate=cp[3]),
               trigger=dict(definition="stored detected_errors non-empty", dev_rate=float(trig.mean()),
                            dev_accuracy=float(np.where(trig > 0, act, keep).mean())))
    FROZEN_F.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: (v if k not in ("lexical",) else {x: v[x] for x in ("w", "b", "tau", "dev_accuracy", "dev_rate")})
                      for k, v in out.items() if k in ("lexical", "composite", "trigger")}, indent=1))


# ============================================================================================ score
FACTS = []
DEF = ("Cost definition: the same v5 second pass for every selection (the field's NT or widen action, either-order "
       "verifier); 'matched' = Map-Reflect v5's realised calls per model, with whole fields in priority order (fixed-"
       "seed random order inside ties) selected while observed spend plus the development-estimated expected cost "
       "(docs/naacl/results/v5_matched_cost_frozen.json) fits; observed spend = realised calls (one reflection call plus "
       "two verifier calls when the action proposes a change); tokens are mean-token estimates per call (test token "
       "logs). 'own' = the control's own threshold, fitted on v5 development outcomes")


def fact(fid, claim, value, lo=None, hi=None):
    assert fid.isalpha(), fid
    FACTS.append(dict(id="F-VFiveCtl" + fid, claim=claim + ". " + DEF, value=str(value), lo=None if lo is None else str(lo),
                      hi=None if hi is None else str(hi), src="docs/naacl/results/v5_controls_facts.json",
                      fn="v5_controls.score", status="exploratory"))


def sg(x, nd=2):
    return f"{x:+.{nd}f}"


def score():
    commit = VM.committed(FROZEN_F)
    VM.committed(VM.COST_F)
    fz = json.loads(FROZEN_F.read_text())
    E = json.loads(VM.COST_F.read_text())["est"]
    L = fz["lexical"]
    groups = GS.split_groups(V5.DATASETS, V5.ITOPS_DUP)
    fact("FrozenCommit", "commit and time of docs/naacl/results/v5_controls_frozen.json (fitted on v5 development only)", commit)
    fact("LexTau", "lexical correctness classifier: own threshold on P(error), fitted on development", f"{L['tau']:.4f}")
    fact("CompTau", "legacy composite u(f): own threshold fitted on development (the earlier deployed value was 0.25)",
         f"{fz['composite']['tau']:.4f}")
    TAB = {}
    for split in ("dev", "test"):
        sp = "Dev" if split == "dev" else ""
        lab = "development" if split == "dev" else "test"
        cells = V5.load(split)
        VM.prepare(cells)
        attach(cells, split)
        N = sum(c.n for c in cells)
        GB = GS.GroupBootstrap(cells, groups, B=ARG.B, seed=V5.SEED)
        mu, sd = np.array(L["mu"]), np.array(L["sd"])
        rng = np.random.default_rng(SEED)
        for c in cells:
            c.p_lex = 1 / (1 + np.exp(-(((c.lexX - mu) / sd) @ np.array(L["w"]) + L["b"])))
            c.rnd = rng.random(c.n)
        R5 = {id(c): c.r5 for c in cells}
        bud = {m: float(sum(np.sum(c.r5 * VM.real_cost(c, "calls")) for c in cells if c.model == m)) for m in MODELS}
        prio = {"Lex": {id(c): c.p_lex for c in cells}, "Comp": {id(c): c.x_comp for c in cells},
                "Trig": {id(c): c.x_trig + 0.5 * c.rnd for c in cells}, "Margin": {id(c): c.mprio for c in cells}}
        own = {"Lex": {id(c): (c.p_lex > L["tau"]).astype(float) for c in cells},
               "Comp": {id(c): (c.x_comp > fz["composite"]["tau"]).astype(float) for c in cells},
               "Trig": {id(c): c.x_trig.copy() for c in cells}, "Margin": {id(c): c.rm for c in cells},
               "Main": R5, "Struct": {id(c): c.struct.copy() for c in cells}}
        NAME = {"Lex": "lexical correctness classifier", "Comp": "legacy composite u(f)", "Trig": "error trigger",
                "Margin": "margin (frozen refitted t_m)", "Main": "Map-Reflect v5 (benefit classifier, frozen tau)",
                "Struct": "schema-size baseline", "Random": "random selection"}

        def row(tag, W, kind, mode, mis=None):
            g, gd = GB.pooled(cells, lambda c: VM.outcome(c, W[id(c)], kind) - c.keep)
            glo, ghi = GS.interval(gd)
            dl, dd = GB.pooled(cells, lambda c: VM.outcome(c, W[id(c)], kind) - VM.outcome(c, c.r5))
            dlo, dhi = GS.interval(dd)
            calls, toks = VM.cost_of(cells, W, kind)
            rate = 100 * sum(np.sum(W[id(c)]) for c in cells) / N
            desc = f"{lab}, {NAME[tag]}, " + ("at Map-Reflect v5's calls (matched)" if mode == "matched" else "at its own threshold")
            t = f"{sp}{tag}{'Matched' if mode == 'matched' else 'Own'}"
            fact(t + "Gain", f"{desc}: change over the first pass, points", sg(g), sg(glo), sg(ghi))
            if tag != "Main":
                fact(t + "VsMain", f"{desc}: minus Map-Reflect v5 at its own threshold, points (paired)", sg(dl), sg(dlo), sg(dhi))
            fact(t + "Rate", f"{desc}: reflection rate (fields reflected on), %", f"{rate:.1f}")
            fact(t + "CallsPer", f"{desc}: second-pass calls per field", f"{calls:.3f}")
            fact(t + "TokensPer", f"{desc}: second-pass tokens per field (mean-token estimates)", f"{toks:,.0f}")
            if mis is not None:
                spent, b = sum(mis[m][0] for m in MODELS), sum(mis[m][1] for m in MODELS)
                fact(t + "Mismatch", f"{desc}: residual cost mismatch, observed calls minus Map-Reflect's, % of its calls",
                     f"{100 * (spent - b) / b:+.2f}")
            TAB[(split, tag, mode)] = dict(g=g, rate=rate, calls=calls, dl=dl, dlo=dlo, dhi=dhi)
        row("Main", R5, "act", "own")
        for tag in ("Lex", "Comp", "Trig", "Margin"):
            W, MIS = VM.seq_route(cells, prio[tag], "calls", bud, E)
            row(tag, W, "act", "matched", MIS)
        # random selection at Map-Reflect's calls: exact expectation, share per model = its calls / all fields' calls
        share = {m: bud[m] / float(sum(np.sum(VM.real_cost(c, "calls")) for c in cells if c.model == m)) for m in MODELS}
        WR = {id(c): np.full(c.n, share[c.model]) for c in cells}
        g, gd = GB.pooled(cells, lambda c: WR[id(c)] * (c.act - c.keep))
        dl, dd = GB.pooled(cells, lambda c: WR[id(c)] * (c.act - c.keep) - (VM.outcome(c, c.r5) - c.keep))
        glo, ghi = GS.interval(gd)
        dlo, dhi = GS.interval(dd)
        calls, toks = VM.cost_of(cells, WR)
        rate = 100 * sum(np.sum(WR[id(c)]) for c in cells) / N
        fact(f"{sp}RandomMatchedGain", f"{lab}, random selection at Map-Reflect v5's calls (exact expectation): change over "
             "the first pass, points", sg(g), sg(glo), sg(ghi))
        fact(f"{sp}RandomMatchedVsMain", f"{lab}, random selection at Map-Reflect v5's calls minus Map-Reflect v5, points",
             sg(dl), sg(dlo), sg(dhi))
        TAB[(split, "Random", "matched")] = dict(g=g, rate=rate, calls=calls, dl=dl, dlo=dlo, dhi=dhi)
        for tag in ("Lex", "Comp", "Trig", "Margin", "Struct"):
            row(tag, own[tag], "wid" if tag == "Struct" else "act", "own")
        print(split, "done")
    FACTS_F.write_text(json.dumps(dict(generated_by="scripts/naacl/v5_controls.py --stage score",
                                       frozen=str(FROZEN_F.relative_to(ROOT)), frozen_commit=commit,
                                       status="exploratory", definition=DEF, facts=FACTS), indent=1) + "\n")
    table(TAB)
    for k, v in TAB.items():
        print(k, {a: round(b, 3) for a, b in v.items()})


def tex_s(x):
    s = f"{x:+.2f}"
    return f"${s}$" if x < 0 else s


def table(T):
    L = ["% Generated by scripts/naacl/v5_controls.py; do not edit by hand. Exploratory: free selection controls on the "
         "same v5 second pass, matched to Map-Reflect's calls per model (development-estimated costs, sequential "
         "stopping) or at their own development-fitted thresholds. Gain = change over the first pass, points; Delta = "
         "minus Map-Reflect, split-group bootstrap 95% CI, B = 2000.",
         r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrrl}", r"\toprule",
         r"Selection & Rate (\%) & Calls & Dev & Test & $\Delta$ vs Map-Reflect \\", r"\midrule"]
    NM = {"Main": "Map-Reflect", "Margin": "Margin", "Lex": "Lexical classifier", "Comp": "Fixed composite",
          "Trig": "Error trigger", "Random": "Random selection", "Struct": "Schema size"}

    def r(tag, mode):
        t, d = T[("test", tag, mode)], T[("dev", tag, mode)]
        delta = "reference" if tag == "Main" else f"{tex_s(t['dl'])} [{tex_s(t['dlo'])}, {tex_s(t['dhi'])}]"
        return f"{NM[tag]} & {t['rate']:.1f} & {t['calls']:.2f} & {tex_s(d['g'])} & {tex_s(t['g'])} & {delta} \\\\"
    L.append(r"\multicolumn{6}{l}{\emph{Matched to Map-Reflect's calls}} \\")
    L.append(r("Main", "own"))
    for tag in ("Margin", "Lex", "Comp", "Trig", "Random"):
        L.append(r(tag, "matched"))
    L.append(r"\multicolumn{6}{l}{\emph{Own threshold}} \\")
    for tag in ("Margin", "Lex", "Comp", "Trig", "Struct"):
        L.append(r(tag, "own"))
    L += [r"\bottomrule", r"\end{tabular}", r"\gencaption", r"\label{tab:v5controls}", r"\end{table*}"]
    TAB_F.write_text("\n".join(L) + "\n")
    print("wrote", TAB_F)


def table_from_facts():
    """Rebuild the table from the stored facts, with no rescoring. The random row's reflection rate and calls are cost
    quantities (Map-Reflect's calls over all fields' calls per model); they are recomputed from the stored costs only
    (no correctness is read) and written as facts if missing."""
    D = json.loads(FACTS_F.read_text())
    F_ = {f["id"]: f for f in D["facts"]}
    T = {}
    for split in ("dev", "test"):
        sp = "Dev" if split == "dev" else ""
        for tag, mode in [("Main", "own")] + [(t, "matched") for t in ("Lex", "Comp", "Trig", "Margin")] + \
                [(t, "own") for t in ("Lex", "Comp", "Trig", "Margin", "Struct")]:
            b = f"F-VFiveCtl{sp}{tag}{'Matched' if mode == 'matched' else 'Own'}"
            v = F_.get(b + "VsMain")
            T[(split, tag, mode)] = dict(g=float(F_[b + "Gain"]["value"]), rate=float(F_[b + "Rate"]["value"]),
                                         calls=float(F_[b + "CallsPer"]["value"]),
                                         dl=float(v["value"]) if v else 0.0, dlo=float(v["lo"]) if v else 0.0,
                                         dhi=float(v["hi"]) if v else 0.0)
        b = f"F-VFiveCtl{sp}RandomMatched"
        if b + "Rate" not in F_:
            cells = V5.load(split)
            VM.prepare(cells)
            N = sum(c.n for c in cells)
            bud = {m: float(sum(np.sum(c.r5 * VM.real_cost(c, "calls")) for c in cells if c.model == m)) for m in MODELS}
            tot = {m: float(sum(np.sum(VM.real_cost(c, "calls")) for c in cells if c.model == m)) for m in MODELS}
            rate = 100 * sum(bud[m] / tot[m] * sum(c.n for c in cells if c.model == m) for m in MODELS) / N
            calls = sum(bud.values()) / N
            lab = "development" if split == "dev" else "test"
            for k, cl, val in (("Rate", "reflection rate (fields reflected on, expected), %", f"{rate:.1f}"),
                               ("CallsPer", "second-pass calls per field", f"{calls:.3f}")):
                FACTS.append(dict(id=b + k, claim=f"{lab}, random selection at Map-Reflect v5's calls (exact expectation): "
                                  f"{cl}; a cost quantity, recomputed from stored costs. " + DEF, value=val, lo=None, hi=None,
                                  src="docs/naacl/results/v5_controls_facts.json", fn="v5_controls.table_from_facts",
                                  status="exploratory"))
                F_[b + k] = FACTS[-1]
        v = F_[b + "VsMain"]
        T[(split, "Random", "matched")] = dict(g=float(F_[b + "Gain"]["value"]), rate=float(F_[b + "Rate"]["value"]),
                                               calls=float(F_[b + "CallsPer"]["value"]), dl=float(v["value"]),
                                               dlo=float(v["lo"]), dhi=float(v["hi"]))
    if FACTS:
        D["facts"] += FACTS
        FACTS_F.write_text(json.dumps(D, indent=1) + "\n")
    table(T)


if __name__ == "__main__":
    {"fit": fit, "score": score, "table": table_from_facts}[ARG.stage]()
