"""UQ ablation on method v4 (docs/naacl/REDO_PLAN_V4.md, item G). Called by scripts/naacl/uq_ablation.py (default mode).

What changed from the v3 analysis, answering Reviewer-2 W4 and the Codex audit (finding 2):
- The second pass is the frozen v4 action (results/naacl/v4/test/given, measured on every field): the near-tie action
  NT or the model's widen action, chosen per field by scripts/naacl/router.py, scored with the full-identity scorer
  (eval.calc_accuracy.target_correct; ITOpsGraph v2.1 labels).
- The budget is v4's actual routed share per model on each model's population (router.routed with the frozen
  t_m and t_s), so the margin gate is v4's own gate up to ties.
- Random gate: exact expectation. Intervals: the split-group bootstrap of scripts/naacl/group_stats.py (schema units,
  ITOpsGraph duplicate groups joined, stratified by dataset, one draw shared by every model and signal).
- Token log-probabilities are charged no extra call (in deployment they come with the scored first pass). They were
  measured by a greedy regeneration, so a same-answer sensitivity (fields whose regenerated answer is the scored one)
  is reported beside them.
- P(True): log-probability based for gpt-oss, Granite-4 (its ENDPOINT_A copy) and Qwen3.5; binary stated True or False for
  Llama-4. AUROCs are given with and without Llama-4.
- Semantic entropy here clusters the five sampled answers by identity, so it is answer entropy over a closed candidate
  set and orders fields almost exactly as the distinct-answer count does; the two are reported as one signal.
- The logistic combination is cross-fitted over split groups (no group is scored by a model fitted on it).
"""
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CORE10 = ["itopsgraph", "toolbench", "nl2gql", "omop_cms", "omop_mimic", "omop_synthea", "valentine_chembl",
          "valentine_opendata", "valentine_tpcdi", "valentine_wikidata"]
SIG = {  # name -> (label, extra calls per field in deployment)
    "margin": ("top-2 margin (ours)", 0), "s1": ("top score", 0), "ratio": ("top-2 ratio", 0),
    "entropy": ("score entropy", 0), "lp_answer": ("answer-token log-prob.", 0),
    "verbalised": ("stated confidence", 1), "p_true": ("P(True)", 1),
    "sem_entropy": ("answer entropy, 5 samples (= distinct answers)", 5),
    "combo_paid": ("logistic: margin, top score, P(True), stated", 2),
}
MAC = {"margin": "Margin", "s1": "TopScore", "ratio": "Ratio", "entropy": "ScoreEnt", "lp_answer": "AnsLogp",
       "verbalised": "Stated", "p_true": "PTrue", "sem_entropy": "SemEnt", "combo_paid": "ComboPaid"}


def run(U):
    import router  # noqa: E402  the one routing rule
    import group_stats as GS  # noqa: E402
    fz = json.load(open(U.ARGS.v4_frozen))
    th = fz["thresholds"]
    dup = {sid: md.get("duplicate_group") for sid, (_, _, md) in U.GOLD21.items()} if U.GOLD21 else None
    groups = GS.split_groups(CORE10, dup)

    # ------------------------------------------------------------------ load
    F = defaultdict(list)
    notes = defaultdict(int)
    missing = []
    for ds in CORE10:
        man = json.load(open(f"data/splits/{ds}.json"))
        test_ids = set(man["test"]) - set(man.get("old_dev_excluded") or [])
        for m in U.MODELS:
            f = f"results/naacl/v4/test/given/{ds}__{m}/fields.jsonl"
            if not (os.path.exists(f) and os.path.exists(f + ".done")):
                missing.append(f"{ds}__{m}")
                continue
            t = th[m]
            fp = U.first_pass(ds, m, test_ids)
            uq = U.uq_rows(ds, m)
            for r in U.read(f):
                if str(r["sample_id"]) not in test_ids:
                    continue
                notes["fields"] += 1
                rec = fp.get(r["key"])
                if rec is None or (r["base"] or "") != (rec["base"] or "") or abs(r["margin"] - rec["margin"]) > 1e-9:
                    notes["stale"] += 1
                    continue
                g = U.gold(ds, r["sample_id"], r["key"])
                if g is None:
                    notes["unscored (ITOpsGraph v2.1)"] += 1
                    continue
                gl = r["gts"] if g == "keep" else g
                act = router.action(bool(r["valid"]), float(r["s1"]), t["t_s"], t["near"], t["widen"])
                if f"{act}_ok" not in r and act not in r:
                    notes[f"no {act} outcome"] += 1
                    continue
                routed = router.band(float(r["margin"]), bool(r["valid"]), float(r["s1"]), t["t_m"], t["t_s"]) \
                    != router.KEEP
                if not r["valid"]:
                    notes["declined"] += 1
                    continue
                v = U.score_signals(rec)
                u = uq.get(r["key"], {})
                v.update({k: x for k, x in u.items() if not k.startswith("_")})
                F[m].append(dict(ds=ds, sid=str(r["sample_id"]), unit=groups[(ds, str(r["sample_id"]))],
                                 base=U.target_correct(r["base"], gl), out=U.target_correct(r.get(act), gl),
                                 routed=routed, sig=v, ptsrc=u.get("_pt_src"),
                                 lp_agree=u.get("_lp_top") == r["base"],
                                 lp_rec=u.get("_lp_top") is not None,  # a regenerated answer is recorded (v6 only reads it)
                                 shown=any(U.target_correct(c.get("id"), gl) for c in rec["ranking"])))

    # population per model: answered fields with every signal the model has on 90% of its answered fields
    avail, pop = {}, {}
    for m in U.MODELS:
        fs = F[m]
        avail[m] = [s for s in SIG if s != "combo_paid" and sum(s in f["sig"] for f in fs) >= 0.9 * len(fs)]
        pop[m] = [f for f in fs if all(s in f["sig"] for s in avail[m])]
        if all(x in avail[m] for x in ("margin", "s1", "p_true", "verbalised")):
            crossfit_groups(pop[m], ["margin", "s1", "p_true", "verbalised"], "combo_paid")
            avail[m].append("combo_paid")
    budget = {m: np.mean([f["routed"] for f in pop[m]]) if pop[m] else 0.0 for m in U.MODELS}

    # group bootstrap helpers (one draw shared by everything)
    B = U.ARGS.B
    rng = np.random.default_rng(20261009)
    index = defaultdict(dict)
    for m in U.MODELS:
        for f in pop[m]:
            index[f["ds"]].setdefault(f["unit"], len(index[f["ds"]]))
    counts = {ds: rng.multinomial(len(ix), np.full(len(ix), 1.0 / len(ix)), size=B).astype(float)
              for ds, ix in sorted(index.items())}

    def boot_mean(fields, v):
        num = np.zeros(B)
        den = np.zeros(B)
        by = defaultdict(lambda: (np.zeros(len(index[ds_])), np.zeros(len(index[ds_]))) if False else None)
        acc = {}
        for f, x in zip(fields, v):
            ds_ = f["ds"]
            S, N = acc.setdefault(ds_, (np.zeros(len(index[ds_])), np.zeros(len(index[ds_]))))
            i = index[ds_][f["unit"]]
            S[i] += x
            N[i] += 1
        for ds_, (S, N) in acc.items():
            num += counts[ds_] @ S
            den += counts[ds_] @ N
        del by
        return 100 * num / np.where(den > 0, den, np.nan)

    def field_w(fields):
        return np.stack([counts[f["ds"]][:, index[f["ds"]][f["unit"]]] for f in fields], axis=1)

    def ci(d):
        d = d[~np.isnan(d)]
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    def arr(fs, k):
        return np.array([f[k] for f in fs], float)

    def conf(fs, s):
        return np.array([f["sig"][s] for f in fs], float)

    def weights(m, s):
        fs = pop[m]
        k = budget[m] * len(fs)
        if s == "keep":
            return np.zeros(len(fs))
        if s == "v4":
            return arr(fs, "routed")
        if s == "random":
            return np.full(len(fs), budget[m])
        if s == "oracle":
            return U.gate_weights(-(arr(fs, "out") - arr(fs, "base")), k)
        return U.gate_weights(conf(fs, s), k)

    def after(m, s):
        fs = pop[m]
        b, o = arr(fs, "base"), arr(fs, "out")
        return b + weights(m, s) * (o - b)

    def group(models, sigs):
        fields = [f for m in models for f in pop[m]]
        acc = {s: np.concatenate([after(m, s) for m in models]) for s in sigs}
        res = {}
        for s in sigs:
            d = acc[s] - acc["margin"]
            res[s] = (100 * acc[s].mean(), 100 * d.mean(), ci(boot_mean(fields, d)))
        return res, len(fields)

    L = []
    P = L.append
    P("# UQ ablation on method v4: routing the second pass by other uncertainty signals (core test)\n")
    P("Generated by `scripts/naacl/uq_ablation.py` (v4 mode, `scripts/naacl/uq_ablation_v4.py`) from result files; it "
      "also writes `paper/numbers_uq.tex` and `paper/tab_uqsignals.tex`. The superseded v3 analysis is "
      "`docs/naacl/results/uq_ablation.md` (`--v3`).\n")
    P("**Design.** The second pass is the frozen v4 action (`" + U.ARGS.v4_frozen + "`, dated " + str(fz.get("date"))
      + "), measured on every test field (`results/naacl/v4/test/given`): each routed field runs the action "
      "`scripts/naacl/router.py` names for it (near-tie action NT, or the model's widen action). Every gate routes "
      "v4's own share of each model's fields, its least confident ones (ties share the boundary); the margin gate is "
      "v4's gate up to ties. Correctness uses the full-identity scorer, ITOpsGraph on the v2.1 labels. Random gate: "
      "exact expectation. Oracle: fields the action fixes first, fields it breaks last. Intervals: split-group "
      f"bootstrap (`scripts/naacl/group_stats.py`, schema units, stratified by dataset), B = {B}, shared by all "
      "models and signals. No threshold is fitted for any signal.\n")
    P(f"Population: grouped test minus old development, ten core datasets, answered fields. Fields read "
      f"{notes['fields']:,}; declined {notes['declined']:,} (v4 widens every declined field and every gate would "
      f"route them alike, so they are left out); stale rows {notes['stale']:,}; unscored under ITOpsGraph v2.1 "
      f"{notes['unscored (ITOpsGraph v2.1)']:,}. Cells without a finished v4 test run: "
      f"{', '.join(missing) or 'none'}. Per model, the population is the answered fields where every signal the "
      "model has exists.\n")
    P("| Model | fields | first pass right | v4 routed share | signals |")
    P("|---|--:|--:|--:|---|")
    for m in U.MODELS:
        if pop[m]:
            P(f"| {U.MNAME[m]} | {len(pop[m]):,} | {100 * arr(pop[m], 'base').mean():.1f}% | {100 * budget[m]:.1f}% | "
              + ", ".join(SIG[s][0] for s in avail[m] if s != "margin") + " |")
    P("")
    lpa = {m: (sum(f["lp_agree"] for f in pop[m]), len(pop[m])) for m in U.MODELS if "lp_answer" in avail[m]}
    P("Signal notes. Token log-probability: measured by one greedy regeneration of the first-pass prompt on ENDPOINT_A and "
      "charged no call (in deployment it comes with the scored first pass); the regenerated answer equals the scored "
      "one for " + ", ".join(f"{U.MNAME[m]} {100 * a / n:.1f}%" for m, (a, n) in lpa.items())
      + ". Llama-4 returns no log-probabilities. P(True): from the True/False log-probabilities for gpt-oss, "
      "Qwen3.5 and Granite-4 (its ENDPOINT_A copy), a binary stated True or False for Llama-4. Answer entropy: the "
      "five sampled answers are clustered by candidate identity, which over a closed candidate set makes it the "
      "entropy of the answer distribution; it ranks fields almost exactly as the number of distinct answers does, so "
      "the two are one signal here (gpt-oss and Qwen3.5 only). The logistic combination is cross-fitted over split "
      "groups.\n")

    # ------------------------------------------------------------------ gate swap
    P("## 1. Gate swap at v4's routed share (final accuracy after the v4 second pass)\n")
    P("Delta = gate minus the margin gate, points, 95% split-group interval. Share of oracle gain = (gate - keep) / "
      "(oracle - keep).\n")
    rows_all = ["keep", "random", "margin", "s1", "ratio", "entropy", "lp_answer", "verbalised", "p_true",
                "sem_entropy", "combo_paid", "v4", "oracle"]
    LBL = dict({s: SIG[s][0] for s in SIG}, keep="keep first answers", random="random gate (expectation)",
               v4="v4 gate (router)", oracle="oracle gate")
    head = {}
    for m in U.MODELS + ["pooled"]:
        models = [x for x in U.MODELS if pop[x]] if m == "pooled" else [m]
        if not all(pop[x] for x in models):
            continue
        sigs = [s for s in rows_all if s in ("keep", "random", "v4", "oracle", "margin")
                or all(s in avail[x] for x in models)]
        res, n = group(models, sigs)
        head[m] = res
        P(f"### {'All four models' if m == 'pooled' else U.MNAME[m]} ({n:,} fields)\n")
        P("| Gate | extra calls | accuracy | share of oracle gain | delta vs margin [95% CI] |")
        P("|---|--:|--:|--:|--:|")
        k0, o0 = res["keep"][0], res["oracle"][0]
        for s in sigs:
            a, d, (lo, hi) = res[s]
            sh = f"{100 * (a - k0) / (o0 - k0):.0f}%" if o0 > k0 else ""
            P(f"| {LBL[s]} | {SIG[s][1] if s in SIG else 0} | {a:.2f} | {sh} | "
              + ("" if s == "margin" else f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]") + " |")
        P("")
    sub = {}
    P("### Signals on some models only, pooled over those models\n")
    P("| Signal | models | fields | accuracy | margin accuracy | delta vs margin [95% CI] |")
    P("|---|---|--:|--:|--:|--:|")
    for s in SIG:
        models = [x for x in U.MODELS if s in avail[x]]
        if not models or len(models) == len([x for x in U.MODELS if pop[x]]):
            continue
        res, n = group(models, ["margin", s])
        sub[s] = (models, res)
        a, d, (lo, hi) = res[s]
        P(f"| {SIG[s][0]} | {', '.join(U.MNAME[x] for x in models)} | {n:,} | {a:.2f} | {res['margin'][0]:.2f} | "
          f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    P("")
    # routing bottleneck at v4's share, margin gate
    errs = [(w, f["shown"], f["out"]) for m in U.MODELS if pop[m]
            for f, w in zip(pop[m], weights(m, "margin")) if not f["base"]]
    E = np.array(errs, float)
    err_stats = dict(UQAblErrRouted=f"{100 * E[:, 0].sum() / len(E):.0f}",
                     UQAblErrInShown=f"{100 * (E[:, 0] * E[:, 1]).sum() / E[:, 0].sum():.0f}",
                     UQAblErrRepaired=f"{100 * (E[:, 0] * E[:, 2]).sum() / E[:, 0].sum():.0f}")
    P(f"At v4's share the margin gate routes {E[:, 0].sum():,.0f} of {len(E):,} first-pass errors "
      f"({err_stats['UQAblErrRouted']}%); the gold target is among the shown candidates for "
      f"{err_stats['UQAblErrInShown']}% of them, and the v4 action repairs {err_stats['UQAblErrRepaired']}%.\n")

    # ------------------------------------------------------------------ discrimination
    P("## 2. Discrimination of first-pass errors (AUROC, errors positive)\n")
    P("Pooled values rank each signal within its model first. Difference = signal minus margin on the same fields, "
      "split-group bootstrap. Same-answer rows restrict token log-probabilities to fields whose regenerated answer is "
      "the scored one.\n")
    P("| Population | Signal | fields | AUROC | minus margin [95% CI] | AUPRC | AURC |")
    P("|---|---|--:|--:|--:|--:|--:|")
    disc = {}

    def disc_row(name, models, s, filt=None, key=None):
        fs = [f for x in models for f in pop[x] if filt is None or filt(f)]
        if not fs:
            return
        err = 1 - arr(fs, "base")
        if len(models) > 1:
            cs = np.concatenate([U.pct_rank(conf([f for f in pop[x] if filt is None or filt(f)], s)) for x in models])
            cm = np.concatenate([U.pct_rank(conf([f for f in pop[x] if filt is None or filt(f)], "margin"))
                                 for x in models])
        else:
            cs, cm = conf(fs, s), conf(fs, "margin")
        a, am = U.auroc_w(cs, err), U.auroc_w(cm, err)
        lo = hi = 0.0
        if s == "margin":
            dd = ""
        else:
            W = field_w(fs)
            lo, hi = ci(U.auroc_draws(cs, err, W) - U.auroc_draws(cm, err, W))
            dd = f"{a - am:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        disc[key or (name, s)] = (a, am, lo, hi)
        P(f"| {name} | {SIG[s][0]} | {len(fs):,} | {a:.3f} | {dd} | {U.auprc(cs, err):.3f} | {U.aurc(cs, err):.4f} |")

    for m in U.MODELS:
        for s in avail[m]:
            disc_row(U.MNAME[m], [m], s)
    allm = [x for x in U.MODELS if pop[x]]
    for s in SIG:
        models = [x for x in allm if s in avail[x]]
        if models:
            nm = "All four" if len(models) == len(allm) else ", ".join(U.MNAME[x] for x in models)
            disc_row(nm, models, s, key=("pooled", s))
    nolm = [x for x in allm if x != "llama4" and "p_true" in avail[x]]
    disc_row("without Llama-4 (P(True) from log-probabilities)", nolm, "p_true", key=("pooled_nollama", "p_true"))
    lpm = [x for x in allm if "lp_answer" in avail[x]]
    disc_row("same answer, " + ", ".join(U.MNAME[x] for x in lpm), lpm, "lp_answer", filt=lambda f: f["lp_agree"],
             key=("same_answer", "lp_answer"))
    # semantic entropy against the distinct-answer count
    sem = [f for x in allm if "sem_entropy" in avail[x] for f in pop[x] if "n_distinct" in f["sig"]]
    if sem:
        a1 = U.auroc_w(np.array([f["sig"]["sem_entropy"] for f in sem]), 1 - arr(sem, "base"))
        a2 = U.auroc_w(np.array([f["sig"]["n_distinct"] for f in sem]), 1 - arr(sem, "base"))
        same = np.mean([(f["sig"]["sem_entropy"] == 0) == (f["sig"]["n_distinct"] == -1) for f in sem])
        P(f"\nAnswer entropy against the distinct-answer count on the same {len(sem):,} fields (raw, pooled): AUROC "
          f"{a1:.3f} against {a2:.3f}; they agree on which fields have a unanimous sample for {100 * same:.1f}% of "
          "fields, so they are reported as one signal.\n")
    P("")

    # ------------------------------------------------------------------ per-dataset AUROCs (core) and held-out
    four = ["margin", "verbalised", "p_true", "ratio"]
    P("## 2b. Error AUROC within each dataset (core test)\n")
    P("AUROC computed inside each model and dataset cell (no between-dataset separation), for the margin, stated "
      "confidence, P(True) and the top-2 ratio; cells with fewer than 5 errors or 5 correct answers are left out. "
      "The summary averages the cells without weighting.\n")
    P("| Dataset | Model | fields | errors | margin | stated | P(True) | ratio |")
    P("|---|---|--:|--:|--:|--:|--:|--:|")
    core_cells = []
    for ds in CORE10:
        for m in U.MODELS:
            fs = [f for f in F[m] if f["ds"] == ds and all(x in f["sig"] for x in four)]
            e = sum(1 - f["base"] for f in fs)
            if e < 5 or len(fs) - e < 5:
                continue
            err = 1 - arr(fs, "base")
            a = {x: U.auroc_w(conf(fs, x), err) for x in four}
            core_cells.append(a)
            P(f"| {ds} | {U.MNAME[m]} | {len(fs):,} | {int(e)} | " + " | ".join(f"{a[x]:.3f}" for x in four) + " |")
    P("")
    ho_cells, ho_rows = heldout_aurocs(U, four)
    summ = {}
    for name, cells in (("core", core_cells), ("heldout", ho_cells),
                        ("heldout_nomag", [c for c, r in zip(ho_cells, ho_rows) if r[0] != "valentine_magellan"])):
        if not cells:
            continue
        mean = {x: float(np.mean([c[x] for c in cells])) for x in four}
        wins = {x: sum(c[x] > c["margin"] for c in cells) for x in four if x != "margin"}
        summ[name] = (len(cells), mean, wins)
        P(f"**{ {'core': 'Core test', 'heldout': 'Held-out sets', 'heldout_nomag': 'Held-out sets without Magellan'}[name]}** "
          f"({len(cells)} cells): mean AUROC margin {mean['margin']:.3f}, stated {mean['verbalised']:.3f}, P(True) "
          f"{mean['p_true']:.3f}, ratio {mean['ratio']:.3f}; cells where the signal beats the margin: stated "
          f"{wins['verbalised']}, P(True) {wins['p_true']}, ratio {wins['ratio']} of {len(cells)}.\n")
    P("### Held-out sets, error AUROC per model and dataset\n")
    P("| Dataset | Model | fields | errors | margin | stated | P(True) | ratio |")
    P("|---|---|--:|--:|--:|--:|--:|--:|")
    for ds, m, n_, e_, a in ho_rows:
        P(f"| {ds} | {U.MNAME[m]} | {n_:,} | {e_} | " + " | ".join(f"{a[x]:.3f}" for x in four) + " |")
    P("")

    # ------------------------------------------------------------------ cost
    cost = U.cost_table()
    P("## 3. Cost per field\n")
    P("| Model | signal | calls | input tokens | output tokens | seconds |")
    P("|---|---|--:|--:|--:|--:|")
    lab = {"verbalised": "stated confidence", "p_true": "P(True)", "samples": "5 samples (answer entropy)",
           "logprob": "token log-probabilities (free with the first pass; measured by regeneration)"}
    for m in U.MODELS:
        P(f"| {U.MNAME[m]} | margin, top score, ratio, score entropy | 0 | 0 | 0 | 0 |")
        for s in ("verbalised", "p_true", "samples", "logprob"):
            if (m, s) in cost:
                k, i, o, sec = cost[(m, s)]
                if s == "logprob":
                    P(f"| {U.MNAME[m]} | {lab[s]} | 0 | 0 | 0 | 0 |")
                else:
                    P(f"| {U.MNAME[m]} | {lab[s]} | {k} | {i:,.0f} | {o:,.0f} | {sec:.2f} |")
    P("")

    # ------------------------------------------------------------------ verdict and macros
    pooled = head.get("pooled", {})
    P("## 4. Verdict\n")
    better, worse, tie = [], [], []
    items = [(s, pooled[s]) for s in SIG if s in pooled and s != "margin"] + \
        [(s, r[s]) for s, (ms, r) in sub.items()]
    for s, (a, d, (lo, hi)) in items:
        (better if lo > 0 else worse if hi < 0 else tie).append(f"{SIG[s][0]} {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    P(f"At v4's routed share the margin gate reaches {pooled['margin'][0]:.2f}% pooled (keep {pooled['keep'][0]:.2f}%, "
      f"random {pooled['random'][0]:.2f}%, oracle {pooled['oracle'][0]:.2f}%). Signals minus the margin: better "
      f"{'; '.join(better) or 'none'}; not distinguishable {'; '.join(tie) or 'none'}; worse "
      f"{'; '.join(worse) or 'none'}.\n")
    P("Discrimination (pooled AUROC, within-model ranks): " + ", ".join(
        f"{SIG[s][0]} {disc[('pooled', s)][0]:.3f}" for s in sorted([s for s in SIG if ('pooled', s) in disc],
                                                                     key=lambda s: -disc[('pooled', s)][0])) + ".\n")
    mac = {}
    for s in SIG:
        if ("pooled", s) not in disc:
            continue
        mac[f"UQAbl{MAC[s]}AUC"] = f"{disc[('pooled', s)][0]:.3f}"
        r = pooled.get(s) if s in pooled else (sub[s][1][s] if s in sub else None)
        if r is not None and s != "margin":
            a, d, (lo, hi) = r
            mac[f"UQAbl{MAC[s]}Delta"] = f"{d:+.2f}"
            mac[f"UQAbl{MAC[s]}Lo"] = f"{lo:+.2f}"
            mac[f"UQAbl{MAC[s]}Hi"] = f"{hi:+.2f}"
        if r is not None:
            mac[f"UQAbl{MAC[s]}Acc"] = f"{r[0]:.2f}"
    for s, nm in (("sem_entropy", "Distinct"),):  # one signal: the table's distinct-answer macros mirror it
        for suf in ("AUC", "Delta", "Lo", "Hi", "Acc"):
            if f"UQAblSemEnt{suf}" in mac:
                mac[f"UQAbl{nm}{suf}"] = mac[f"UQAblSemEnt{suf}"]
    words = {1: "one", 2: "two", 3: "three", 4: "four"}
    for s, nm in (("lp_answer", "AnsLogp"), ("sem_entropy", "SemEnt"), ("p_true", "PTrue")):
        mac[f"UQAbl{nm}Models"] = words[len([x for x in allm if s in avail[x]])]
    if ("pooled_nollama", "p_true") in disc:
        mac["UQAblPTrueLogpAUC"] = f"{disc[('pooled_nollama', 'p_true')][0]:.3f}"
    if ("same_answer", "lp_answer") in disc:
        mac["UQAblAnsLogpSameAUC"] = f"{disc[('same_answer', 'lp_answer')][0]:.3f}"
    close = [SIG[x][0] for x in ("ratio", "verbalised", "p_true", "s1", "entropy", "lp_answer", "sem_entropy")
             if ("pooled", x) in disc and disc[("pooled", x)][3] >= 0]
    mac["UQAblAUCNotSig"] = " or ".join(close) if close else ""
    mac.update(err_stats)
    mac["UQAblKeepAcc"] = f"{pooled['keep'][0]:.2f}"
    mac["UQAblRandomAcc"] = f"{pooled['random'][0]:.2f}"
    mac["UQAblOracleAcc"] = f"{pooled['oracle'][0]:.2f}"
    mac["UQAblMarginAcc"] = f"{pooled['margin'][0]:.2f}"
    n = sum(len(pop[x]) for x in allm)
    mac["UQAblBudget"] = f"{100 * sum(budget[x] * len(pop[x]) for x in allm) / n:.1f}"
    mac["UQAblFields"] = f"{n:,}".replace(",", "{,}")
    answered = sum(len(F[x]) for x in U.MODELS)
    mac["UQAblFieldsAnswered"] = f"{answered:,}".replace(",", "{,}")
    for name, key in (("core", "Core"), ("heldout", "HO"), ("heldout_nomag", "HONoMag")):
        if name in summ:
            k_, mean, wins = summ[name]
            mac[f"UQAbl{key}Cells"] = str(k_)
            for x, nm in (("margin", "Margin"), ("verbalised", "Stated"), ("p_true", "PTrue"), ("ratio", "Ratio")):
                mac[f"UQAbl{key}{nm}MeanAUC"] = f"{mean[x]:.2f}"
                if x != "margin":
                    mac[f"UQAbl{key}{nm}Wins"] = str(wins[x])
    P("## 5. Macros (written to `paper/numbers_uq.tex`)\n")
    P("```latex")
    for k, v in mac.items():
        P(f"\\newcommand{{\\{k}}}{{{v}}}")
    P("```\n")
    Path(U.ARGS.v4_out).write_text("\n".join(L) + "\n")
    print(f"wrote {U.ARGS.v4_out}")
    if U.ARGS.paper:
        write_paper(U.REPO, mac)
        write_heldout_table(U.REPO, ho_rows, U.MNAME)


def crossfit_groups(fields, feats, name, folds=2, repeats=5):
    """Logistic regression on the named signals, cross-fitted over split groups (each group is scored by models
    fitted on other groups only), averaged over five random splits."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.array([[f["sig"][x] for x in feats] for f in fields], float)
    y = np.array([f["base"] for f in fields])
    units = sorted({(f["ds"], f["unit"]) for f in fields})
    pos = {u: i for i, u in enumerate(units)}
    uidx = np.array([pos[(f["ds"], f["unit"])] for f in fields])
    rng = np.random.default_rng(7)
    out = np.zeros(len(fields))
    for _ in range(repeats):
        fold = rng.integers(0, folds, len(units))[uidx]
        for k in range(folds):
            tr, te = fold != k, fold == k
            if te.sum() == 0 or len(set(y[tr])) < 2:
                out[te] += 0.5 / repeats
                continue
            mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            out[te] += mdl.predict_proba(X[te])[:, 1] / repeats
    for f, v in zip(fields, out):
        f["sig"][name] = float(v)


def write_paper(repo, mac):
    paper = repo / "paper"
    need = ["MarginAUC", "TopScoreAUC", "RatioAUC", "ScoreEntAUC", "AnsLogpAUC", "StatedAUC", "PTrueAUC", "SemEntAUC",
            "ComboPaidAUC", "TopScoreDelta", "RatioDelta", "ScoreEntDelta", "AnsLogpDelta", "StatedDelta",
            "PTrueDelta", "SemEntDelta", "ComboPaidDelta", "SemEntModels", "AnsLogpModels", "PTrueLogpAUC",
            "ErrRouted", "ErrInShown", "ErrRepaired"]
    miss = [x for x in need if f"UQAbl{x}" not in mac]
    if miss:
        sys.exit(f"cannot write the paper files, macros missing: {miss}")
    (paper / "numbers_uq.tex").write_text(
        "% Generated by scripts/naacl/uq_ablation.py (v4 mode) from result files (docs/naacl/results/uq_ablation_v4.md);"
        " do not edit by hand.\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    tab = r"""% Generated by scripts/naacl/uq_ablation.py (v4 mode); every cell is a macro from numbers_uq.tex. Source:
% docs/naacl/results/uq_ablation_v4.md. Routing: the v4 second pass at v4's routed share, pooled over models.
\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcrr}
\toprule
 & Extra & Error & Routing \\
Signal & calls & AUROC & vs.\ margin \\
\midrule
\textbf{Margin} $\gamma_1-\gamma_2$ & 0 & \textbf{\UQAblMarginAUC} & 0.00 \\
Top score $\gamma_1$ & 0 & \UQAblTopScoreAUC & $\UQAblTopScoreDelta$ \\
Ratio $\gamma_1/\gamma_2$ & 0 & \UQAblRatioAUC & $\UQAblRatioDelta$ \\
Score entropy & 0 & \UQAblScoreEntAUC & $\UQAblScoreEntDelta$ \\
Token log-probability$^\dagger$ & 0 & \UQAblAnsLogpAUC & $\UQAblAnsLogpDelta$ \\
\midrule
Stated confidence & 1 & \UQAblStatedAUC & $\UQAblStatedDelta$ \\
P(True)$^\ddagger$ & 1 & \UQAblPTrueAUC & $\UQAblPTrueDelta$ \\
Answer entropy$^\ast$ & 5 & \UQAblSemEntAUC & $\UQAblSemEntDelta$ \\
\midrule
Logistic: margin, top score, & 2 & \UQAblComboPaidAUC & $\UQAblComboPaidDelta$ \\
\quad P(True), stated & & & \\
\bottomrule
\end{tabular}
\caption{CAPTION_LEAD Error AUROC: how well the signal ranks wrong first answers above right ones (core test, \UQAblFields{} of \UQAblFieldsAnswered{} answered fields: those with every signal of their model). Routing: change in accuracy (points) when the signal, instead of the margin, picks the same share of fields as Map-Reflect. $^\dagger$Answer-token log-probability, free with the first pass, \UQAblAnsLogpModels{} models (Llama-4 returns none). $^\ddagger$Binary for Llama-4; \UQAblPTrueLogpAUC{} on the three models with log-probabilities. $^\ast$Identity-clustered entropy of five samples (equivalent to the distinct-answer count), \UQAblSemEntModels{} models; these rows compare with the margin on the same models.}
\label{tab:uqsignals}
\end{table}
"""
    single = [k for k in ("TopScore", "Ratio", "ScoreEnt", "AnsLogp", "Stated", "PTrue", "SemEnt")
              if float(mac.get(f"UQAbl{k}Lo", "-1")) > 0]
    combo_better = float(mac["UQAblComboPaidLo"]) > 0
    top = all(float(mac[f"UQAbl{k}AUC"]) <= float(mac["UQAblMarginAUC"]) for k in
              ("TopScore", "Ratio", "ScoreEnt", "AnsLogp", "Stated", "PTrue", "SemEnt"))
    lead = ("On the core test the free margin has the highest error AUROC of any single signal" if top
            else "Error detection by signal")
    if top and mac.get("UQAblAUCNotSig"):
        lead += ", though not significantly above the " + mac["UQAblAUCNotSig"]
    lead += (", and no single signal routes the second pass significantly better." if not single
             else "; some single signals route the second pass significantly better.")
    if combo_better:
        lead += (" A logistic combination with two paid calls detects errors better and routes slightly better "
                 "($\\UQAblComboPaidDelta$ points).")
    tab = tab.replace("CAPTION_LEAD", lead.replace("\\\\", "\\"))
    (paper / "tab_uqsignals.tex").write_text(tab)
    print(f"wrote {paper / 'numbers_uq.tex'} ({len(mac)} macros) and {paper / 'tab_uqsignals.tex'}")


HELDOUT = ["mimic_omop_rematch", "semtab_m2kg_r1", "schemanet_cprd", "bioml_omim_ordo", "oaei_conference",
           "valentine_magellan"]
HO_NAME = {"mimic_omop_rematch": "MIMIC-OMOP", "semtab_m2kg_r1": "SemTab", "schemanet_cprd": "SchemaNet",
           "bioml_omim_ordo": "Bio-ML", "oaei_conference": "OAEI Conference", "valentine_magellan": "Magellan"}


def heldout_aurocs(U, four):
    """Error AUROC per held-out model and dataset cell: answered fields of the v4 test rows (full-identity scorer),
    margin, s1 and s2 = s1 - margin from the stored first pass, stated confidence and P(True) from results/naacl/v3/uq."""
    cells, rows = [], []
    for ds in HELDOUT:
        for m in U.MODELS:
            f = f"results/naacl/v4/test/given/{ds}__{m}/fields.jsonl"
            if not os.path.exists(f + ".done"):
                continue
            uq = U.uq_rows(ds, m)
            fs = []
            for r in U.read(f):
                if not r["valid"]:
                    continue
                u = uq.get(r["key"], {})
                if "verbalised" not in u or "p_true" not in u:
                    continue
                import router
                s1, mg = float(r["s1"]), router.canonical_margin(float(r["margin"]))  # identity unless v4.1
                fs.append(dict(base=U.target_correct(r["base"], r["gts"]), margin=mg,
                               ratio=math.log((s1 + 1e-3) / (s1 - mg + 1e-3)), verbalised=u["verbalised"],
                               p_true=u["p_true"]))
            e = sum(1 - x["base"] for x in fs)
            if e < 5 or len(fs) - e < 5:
                continue
            err = np.array([1 - x["base"] for x in fs], float)
            a = {x: U.auroc_w(np.array([g[x] for g in fs], float), err) for x in four}
            cells.append(a)
            rows.append((ds, m, len(fs), int(e), a))
    return cells, rows


def write_heldout_table(repo, rows, mname):
    short = {"gptoss": "gpt-oss", "granite": "Granite-4", "llama4": "Llama-4", "qwen35": "Qwen3.5"}
    body = []
    for ds, m, n, e, a in rows:
        cells = [a["margin"], a["verbalised"], a["p_true"], a["ratio"]]
        best = max(cells)
        fmt = [f"\\textbf{{{c:.2f}}}" if c == best else f"{c:.2f}" for c in cells]
        body.append(f"{HO_NAME[ds]} & {short[m]} & {n} & {e} & " + " & ".join(fmt) + " \\\\")
    tab = (
        "% Generated by scripts/naacl/uq_ablation.py (v4 mode) from result files; source "
        "docs/naacl/results/uq_ablation_v4.md, section 2b.\n"
        "\\begin{table}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{llrrrrrr}\n\\toprule\n"
        "Dataset & Model & Fields & Errors & Margin & Stated & P(True) & Ratio \\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\midrule\n"
        "\\multicolumn{4}{l}{Mean over \\UQAblHOCells{} cells} & \\UQAblHOMarginMeanAUC & \\UQAblHOStatedMeanAUC & "
        "\\UQAblHOPTrueMeanAUC & \\UQAblHORatioMeanAUC \\\\\n"
        "\\multicolumn{4}{l}{Cells above the margin} & & \\UQAblHOStatedWins & \\UQAblHOPTrueWins & "
        "\\UQAblHORatioWins \\\\\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Error AUROC on the held-out sets, per model and dataset (answered fields with stated confidence and "
        "P(True); cells with fewer than five errors or five correct answers left out). Bold: the highest in the row. "
        "On these sets stated confidence detects errors better than the margin in \\UQAblHOStatedWins{} of "
        "\\UQAblHOCells{} cells.}\n\\label{tab:uq_auroc_heldout}\n\\end{table}\n")
    (repo / "paper" / "tab_uq_auroc_heldout.tex").write_text(tab)
    print(f"wrote {repo / 'paper' / 'tab_uq_auroc_heldout.tex'}")
