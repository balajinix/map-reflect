#!/usr/bin/env python3
"""Support, missingness and table consistency of the confidence-estimate comparison (review 2, M5, Q7, Q8). Offline;
no model call. Exploratory.

Reads the same per-field estimates as scripts/naacl/v5_signals.py (its build() and population(), imported, nothing
rewritten) on Map-Reflect's test rows.
  missing      per estimate, the share of answered core test pairs without a valid value, among first-pass correct
               and among first-pass wrong answers, and the dataset with the highest missing share
  support      per estimate, its error AUROC and the margin's on (a) the common valid population used by the paper and
               (b) the pairwise population where only the margin and that estimate must be valid; pooled with ranks
               within model, as v5_signals; (a) must reproduce F-VFiveSig*AUC
  regenerated  for the answer-token log-probability, the share of fields whose regenerated answer (lp_top of the row the
               estimate was read from) equals the scored first answer, for all four models (Q7 asks for Llama-4)
  P(True)      distinct values and the largest tie share per model: Llama-4's P(True) is a binary stated verdict
  tables       the held-out AUROC of every estimate in the three current tables that report it (tab6_signals_compact,
               tab6_signals, tab6_detect) against F-VFiveSig*HOAUC, and the superseded values the review cites
               (F-Sig*HOAUC, mean of 20 within-cell AUROCs; tab6_detect_mr, pooled), to explain Q8
Writes docs/naacl/results/rev2_uq_support_facts.json (F-RevTwoUQ*).
Usage: .venv/bin/python scripts/naacl/rev2_uq_support.py
"""
import json
import re

import numpy as np
from scipy.stats import rankdata

import rev2_common as RC
import v5_signals as VS

V5, MODELS = RC.V5, RC.MODELS
U = VS.U
F = RC.Facts("uq_support", "Per-field estimates as read by v5_signals.build (stored UQ rows, ff1 merge, U9 fills); "
             "answered core test pairs of the v5 test rows.")
EST = ["s1", "ratio", "entropy", "lp_answer", "verbalised", "p_true", "sem_entropy"]
ET = {"s1": "TopScore", "ratio": "Ratio", "entropy": "ScoreEnt", "lp_answer": "AnsLogp", "verbalised": "Stated",
      "p_true": "PTrue", "sem_entropy": "SemEnt", "margin": "Margin"}
MT = {"gptoss": "Gptoss", "llama4": "Llama", "granite": "Granite", "qwen35": "Qwen"}


def pooled_auc(cells, sig, mask_fn):
    conf, err = [], []
    for m in MODELS:
        cs = [c for c in cells if c.model == m]
        v = np.concatenate([c.S[sig] for c in cs]) if cs else np.array([])
        e = np.concatenate([1 - c.keep for c in cs]) if cs else np.array([])
        mk = np.concatenate([mask_fn(c) for c in cs]) if cs else np.array([], bool)
        if mk.sum() == 0:
            continue
        conf.append(rankdata(v[mk]) / mk.sum())
        err.append(e[mk])
    return U.auroc_w(np.concatenate(conf), np.concatenate(err)), int(sum(len(x) for x in err))


def table_col(path, col):
    out = {}
    for line in open(path):
        if line.startswith("%") or "&" not in line:
            continue
        parts = [p.strip() for p in line.rstrip("\\\n ").split("&")]
        name = re.sub(r"\\textbf\{(.*?)\}", r"\1", parts[0])
        try:
            out[name] = float(parts[col].split()[0])
        except (ValueError, IndexError):
            pass
    return out


def main():
    test = RC.load("test")
    H = RC.check_headline(test)
    cells = VS.build(False)
    avail, pop = VS.population(cells)
    # ------------------------------------------------------------------ missingness
    for s in EST:
        cs = [c for c in cells if s in avail[c.model]]
        if not cs:
            continue
        v = np.concatenate([c.S[s] for c in cs])
        k = np.concatenate([c.keep for c in cs])
        miss = ~np.isfinite(v)
        byds = {}
        for c in cs:
            byds.setdefault(c.ds, [0, 0])
            byds[c.ds][0] += int((~np.isfinite(c.S[s])).sum())
            byds[c.ds][1] += c.n
        worst = max(byds, key=lambda d: byds[d][0] / byds[d][1])
        F.add(f"UQMissing{ET[s]}", f"core test answered pairs of the models where it is available ({len({c.model for c in cs})}): "
              f"{s} missing among first-pass correct / among first-pass wrong, %; dataset with the highest missing share",
              f"{100 * miss[k == 1].mean():.1f} / {100 * miss[k == 0].mean():.1f}; {worst} {100 * byds[worst][0] / byds[worst][1]:.1f}")
    # ------------------------------------------------------------------ common vs pairwise support
    sig = json.load(open(RC.ROOT / "docs/naacl/results/v5_signals_facts.json"))
    SF = {f["id"]: f["value"] for f in sig["facts"]}
    allpop = [c for m in MODELS for c in pop[m]]
    a_m, n_common = pooled_auc(allpop, "margin", lambda c: np.ones(c.n, bool))
    assert f"{a_m:.3f}" == SF["F-VFiveSigMarginAUC"], (a_m, SF["F-VFiveSigMarginAUC"])
    F.add("UQCommonPop", "core test common valid population (answered pairs with every available estimate valid)", f"{n_common:,}")
    for s in EST:
        cs_c = [c for c in allpop if s in avail[c.model]]
        a_c, n_c = pooled_auc(cs_c, s, lambda c: np.ones(c.n, bool))
        am_c, _ = pooled_auc(cs_c, "margin", lambda c: np.ones(c.n, bool))
        fid = f"F-VFiveSig{ET[s]}AUC"
        if fid in SF:
            assert abs(a_c - float(SF[fid])) < 0.0015, (s, a_c, SF[fid])
        cs_p = [c for c in cells if s in avail[c.model]]
        both = lambda c: np.isfinite(c.S[s]) & np.isfinite(c.S["margin"])  # noqa: E731
        a_p, n_p = pooled_auc(cs_p, s, both)
        am_p, _ = pooled_auc(cs_p, "margin", both)
        F.add(f"UQSupport{ET[s]}", f"core test, {s}: AUROC minus the margin's on the common population (n) / on the "
              "pairwise population where only the margin and it must be valid (n)",
              f"{a_c - am_c:+.3f} ({n_c:,}) / {a_p - am_p:+.3f} ({n_p:,})")
    # ------------------------------------------------------------------ regenerated-answer agreement
    for m in MODELS:
        agree = rec = unrec = 0
        for c in test:
            if c.model != m:
                continue
            rows = {r["key"]: r for r in V5.read(V5.RUNS / "test" / f"{c.ds}__{c.model}.jsonl")}
            base = U.uq_rows(c.ds, m)
            u9 = {}
            for suf in ("", ".fill"):
                for r in VS.read(RC.ROOT / f"results/naacl/v6/u9/logprob/{c.ds}__{m}{suf}.jsonl"):
                    u9.setdefault(r["key"], r)
            for i, k in enumerate(c.keys):
                if not c.validb[i]:
                    continue
                b = base.get(k, {})
                if b.get("lp_answer") is not None:  # the source v5_signals reads first
                    top = b.get("_lp_top")
                elif k in u9 and u9[k].get("lp_answer") is not None:
                    top = u9[k].get("lp_top")
                else:
                    continue
                if top is None:
                    unrec += 1
                    continue
                rec += 1
                agree += int(top == rows[k]["base"])
        F.add(f"UQRegenMatch{MT[m]}", f"core test, {m}: answered fields with a token log-probability: regenerated answer "
              "equals the scored first answer, % of fields whose regenerated answer is recorded", f"{100 * agree / max(rec, 1):.1f}")
        F.add(f"UQRegenRecorded{MT[m]}", f"core test, {m}: answered fields with a token log-probability and a recorded "
              "regenerated answer", f"{rec:,}")
        F.add(f"UQRegenUnrecorded{MT[m]}", f"core test, {m}: answered fields with a token log-probability but no recorded "
              "regenerated answer", f"{unrec:,}")
        if unrec:
            F.add(f"UQRegenMatchAll{MT[m]}", f"core test, {m}: the same share counting unrecorded regenerated answers as "
                  "mismatches (the rule behind F-SigTokRegenMatch*), %", f"{100 * agree / (rec + unrec):.1f}")
    # ------------------------------------------------------------------ P(True) resolution
    for m in MODELS:
        v = np.concatenate([c.S["p_true"] for c in cells if c.model == m])
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        vals, cnt = np.unique(np.round(v, 6), return_counts=True)
        F.add(f"UQPTrueTies{MT[m]}", f"core test, {m}: P(True) distinct values / largest tie share, %", f"{len(vals):,} / {100 * cnt.max() / len(v):.1f}")
        F.add(f"UQPTrueDistinct{MT[m]}", f"core test, {m}: distinct P(True) values", f"{len(vals):,}")
        cs = [c for c in pop[m]]
        a, _ = pooled_auc(cs, "p_true", lambda c: np.ones(c.n, bool))
        F.add(f"UQPTrueAUC{MT[m]}", f"core test common population, {m}: error AUROC of P(True)", f"{a:.3f}")
    # ------------------------------------------------------------------ table consistency (Q8)
    P = RC.ROOT / "paper"
    NAME = {"Margin": "Margin", "Top score": "TopScore", "Ratio": "Ratio", "Score entropy": "ScoreEnt",
            "Token log-prob.": "AnsLogp", "Stated conf.": "Stated", "P(True)": "PTrue", "Answer entropy": "SemEnt",
            "Logistic (paid)": "ComboPaid"}
    cols = {"tab6_signals_compact": table_col(P / "tab6_signals_compact.tex", 3),
            "tab6_signals": table_col(P / "tab6_signals.tex", 4), "tab6_detect": table_col(P / "tab6_detect.tex", 3)}
    bad = []
    for nm, tag in NAME.items():
        ref = float(SF[f"F-VFiveSigHO{tag}AUC"])
        for t, col in cols.items():
            if nm in col and abs(col[nm] - ref) > 0.0051:
                bad.append(f"{t}:{nm} {col[nm]} vs {ref}")
    F.add("UQTablesConsistent", "held-out AUROC of every estimate in tab6_signals_compact, tab6_signals and tab6_detect "
          "equals F-VFiveSig*HOAUC to rounding (one population, pooled with ranks within model)",
          "yes" if not bad else "no: " + "; ".join(bad))
    try:
        old = {f["id"]: f["value"] for f in json.load(open(RC.ROOT / "docs/naacl/results/v5_signals_facts.json"))["facts"]}
        F.add("UQTablesNow", "current held-out AUROC (F-VFiveSigHO*AUC): score entropy / stated confidence / token "
              "log-probability", f"{old['F-VFiveSigHOScoreEntAUC']} / {old['F-VFiveSigHOStatedAUC']} / {old['F-VFiveSigHOAnsLogpAUC']}")
    except KeyError:
        pass
    detect_mr = table_col(P / "tab6_detect_mr.tex", 3)
    F.add("UQTablesSuperseded", "values the review cites, from the superseded ablation-configuration tables: score entropy "
          "0.61, stated 0.81, token log-prob. 0.74 are F-Sig*HOAUC, the mean of 20 within-cell AUROCs (5 held-out sets x 4 "
          "models, each estimate on its own valid fields); 0.549 / 0.832 / 0.706 are tab6_detect_mr, AUROC pooled with "
          "ranks within model on the common valid population", f"tab6_detect_mr: score entropy {detect_mr.get('Score entropy')}, "
          f"stated {detect_mr.get('Stated conf.')}, token log-prob. {detect_mr.get('Token log-prob.')}")
    F.write(dict(headline=H))


if __name__ == "__main__":
    main()
