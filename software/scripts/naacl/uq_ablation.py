#!/usr/bin/env python3
"""UQ ablation: route the reflective second pass by other uncertainty signals instead of the top-2 margin.

The action is held fixed: Map-Reflect over the shown candidates (BC), which protocol v2 ran on every test field
(results/naacl/v2/reflect/given), so any gate's outcome is known without new calls. A gate only decides which fields
get the second pass. Plan and checklist: docs/naacl/plan/uq_ablation.md.

Population: protocol v2 grouped test minus old_dev_excluded, the nine datasets, the four core models, answered fields
(declined fields are routed by every gate alike). Signals, all oriented so that higher means surer:
  free     margin m = s1 - s2, top score s1, top-2 ratio s1 / s2, score entropy (negated), all from the first pass
  1 call   stated confidence, P(True) (results/naacl/v3/uq; ToolBench rerun in full on the final first passes)
  1 call   token log-probabilities of a greedy regeneration (results/naacl/uq_ablation/logprob): answer-token
           log-probability, mean token log-probability, mean token entropy (negated)
  5 calls  semantic entropy and distinct-answer count (negated), gpt-oss and Qwen3.5 only
Outputs docs/naacl/results/uq_ablation.md (tables, verdict, proposed table and macros for the writer).

Usage: .venv/bin/python scripts/naacl/uq_ablation.py [--B 2000] [--out docs/naacl/results/uq_ablation.md]
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "scripts" / "naacl")]
ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--out", default="docs/naacl/results/uq_ablation.md")
ap.add_argument("--frozen", default="docs/naacl/results/v3_frozen.json")
ap.add_argument("--v3", action="store_true",
                help="the superseded v3 analysis (BC action, v3 thresholds) into --out; writes no paper files")
ap.add_argument("--v4-out", default="docs/naacl/results/uq_ablation_v4.md")
ap.add_argument("--v4-frozen", default="docs/naacl/results/v4_frozen.json")
ap.add_argument("--no-paper", dest="paper", action="store_false",
                help="v4 mode: do not write paper/numbers_uq.tex and paper/tab_uqsignals.tex")
ap.add_argument("--labels", default="v21", choices=["v2", "v21"],
                help="ITOpsGraph gold: v21 (default, the paper's labels; removed fields unscored) or v2")
ap.add_argument("--labels-v21", default=str(Path(__file__).resolve().parents[3] / "schema-mapping-datasets/data/"
                                            "itopsgraph_v2_1/itopsgraph_v2.1_cross_product.jsonl"))
ARGS = ap.parse_args()
argv, sys.argv = sys.argv, sys.argv[:1]
import reflect_v2 as RV  # noqa: E402  records, margin and load_slot of protocol v2
import router  # noqa: E402  the one routing rule
from eval.calc_accuracy import fields_match, load_gold_file, target_correct  # noqa: E402
sys.argv = argv
GOLD21 = load_gold_file(ARGS.labels_v21) if ARGS.labels == "v21" else {}
if ARGS.labels == "v21" and not GOLD21:
    sys.exit(f"ITOpsGraph v2.1 labels not found at {ARGS.labels_v21}; pass --labels-v21 or --labels v2")


def gold(ds, sid, key):
    """Acceptable targets of a field under the active labels, None when the field is unscored (v2.1 removed)."""
    if ds != "itopsgraph" or ARGS.labels != "v21":
        return "keep"
    tg, rm, _ = GOLD21[str(sid)]
    fld = key.split("||", 1)[1]
    if fld in rm:
        return None
    if fld not in tg:
        raise KeyError(f"{key}: field neither scored nor removed in the v2.1 labels")
    return tg[fld]

DATASETS = ["itopsgraph", "toolbench", "omop_cms", "omop_mimic", "omop_synthea",
            "valentine_chembl", "valentine_opendata", "valentine_tpcdi", "valentine_wikidata"]
MODELS = ["gptoss", "granite", "llama4", "qwen35"]
MNAME = {"gptoss": "gpt-oss-120b", "granite": "Granite-4", "llama4": "Llama-4-Maverick", "qwen35": "Qwen3.5-397B"}
BUDGETS = [round(0.1 * i, 1) for i in range(1, 11)]
RNG = np.random.default_rng(20261009)
N_RANDOM = 20
ERRSTATS = {}  # margin gate at the matched budget: % of first-pass errors routed, gold shown, repaired
REPAIR_AGREE = defaultdict(list)  # (model, signal) -> [(repair value, rerun value)]

# name -> (label, cost class, extra calls per field)
SIGNALS = {
    "margin": ("top-2 margin (ours)", "free", 0),
    "s1": ("top score", "free", 0),
    "ratio": ("top-2 ratio", "free", 0),
    "entropy": ("score entropy", "free", 0),
    "lp_answer": ("answer-token log-prob.", "1 call", 1),
    "lp_mean": ("mean token log-prob.", "1 call", 1),
    "lp_entropy": ("mean token entropy", "1 call", 1),
    "p_true": ("P(True)", "1 call", 1),
    "verbalised": ("stated confidence", "1 call", 1),
    "sem_entropy": ("semantic entropy", "5 calls", 5),
    "n_distinct": ("distinct answers", "5 calls", 5),
    "combo_free": ("margin + top score (logistic)", "free", 0),
    "combo_paid": ("margin + top score + P(True) + stated (logistic)", "2 calls", 2),
}
COMBOS = {"combo_free": ["margin", "s1"], "combo_paid": ["margin", "s1", "p_true", "verbalised"]}
ALT = [s for s in SIGNALS if s != "margin"]


def read(f):
    return [json.loads(l) for l in open(f) if l.strip()]


# ------------------------------------------------------------------------------------------------------ loading
def first_pass(ds, m, ids):
    d = RV.load_slot(f"results/naacl/v2/first/given/{ds}__{m}")
    out = {}
    for rec in RV.records(d, ids):
        sc = sorted((RV._num(c.get("score")) for c in rec["ranking"]), reverse=True)
        sc = [x for x in sc if x != float("-inf")]
        out[rec["key"]] = dict(rec, scores=sc,
                               base_ok=int(bool(rec["base"]) and any(fields_match(g, rec["base"]) for g in rec["gts"])))
    return out


def score_signals(r):
    sc = r["scores"]
    s1 = sc[0] if sc else 0.0
    s2 = sc[1] if len(sc) > 1 else 0.0
    pr = np.array([max(x, 0.0) for x in sc], float)
    pr = pr / pr.sum() if pr.sum() > 0 else pr
    pr = pr[pr > 0]
    ent = float(-(pr * np.log(pr)).sum() / math.log(len(pr))) if len(pr) > 1 else 0.0
    return dict(margin=router.canonical_margin(float(r["margin"])), s1=s1, ratio=math.log((s1 + 1e-3) / (s2 + 1e-3)), entropy=-ent)


def uq_rows(ds, m):
    """key -> paid-signal values. results/naacl/v3/uq is the canonical source (the ToolBench cells were rerun in full
    on the final first passes); the repair draw in results/naacl/uq_ablation/repair is only compared with it."""
    rows = {}
    f = f"results/naacl/v3/uq/{ds}__{m}.jsonl"
    if os.path.exists(f):
        rows = {r["key"]: r for r in read(f)}
    if os.environ.get("V6_UQ_VALID_ONLY") == "1":  # an unresolved stated answer is missing, not 1 - c (off by default)
        rows = {k: (dict(r, verbalised=None) if r.get("verbalised") is not None and not r.get("verbalised_answer") else r)
                for k, r in rows.items()}
    for kind in ("missing", "stale"):
        g = f"results/naacl/uq_ablation/repair/{ds}__{m}.{kind}.jsonl"
        if os.path.exists(g):
            for r in read(g):
                o = rows.get(r["key"])
                for sname in ("verbalised", "p_true"):
                    if o and r.get(sname) is not None and o.get(sname) is not None:
                        REPAIR_AGREE[(m, sname)].append((float(r[sname]), float(o[sname])))
    for f in (f"results/naacl/uq_ablation/logprob/{ds}__{m}.jsonl", f"results/naacl/uq_ablation/logprob/{ds}__{m}.fill.jsonl"):
        if not os.path.exists(f):
            continue
        for r in read(f):
            rows.setdefault(r["key"], {}).update({k: v for k, v in r.items() if k.startswith("lp_")})
    out = {}
    for k, r in rows.items():
        d = {}
        for s in ("verbalised", "p_true", "lp_answer", "lp_mean"):
            if r.get(s) is not None:
                d[s] = float(r[s])
        for s in ("sem_entropy", "n_distinct", "lp_entropy"):
            if r.get(s) is not None:
                d[s] = -float(r[s])
        d["_lp_top"] = r.get("lp_top")
        d["_pt_src"] = r.get("p_true_source")
        out[k] = d
    return out


class Data:
    pass


def load():
    F = defaultdict(list)
    notes = defaultdict(int)
    lp_agree = defaultdict(lambda: [0, 0])
    for ds in DATASETS:
        man = json.load(open(f"data/splits/{ds}.json"))
        test_ids = set(man["test"]) - set(man.get("old_dev_excluded") or [])
        for m in MODELS:
            fp = first_pass(ds, m, test_ids)
            refl = {r["key"]: r for r in read(f"results/naacl/v2/reflect/given/{ds}__{m}/fields.jsonl")
                    if str(r.get("sample_id")) in test_ids}
            uq = uq_rows(ds, m)
            for k, r in fp.items():
                notes["fields"] += 1
                if not r["valid"]:
                    notes["declined"] += 1
                    continue
                x = refl.get(k)
                if x is None or "BC_ok" not in x:
                    notes["no reflection row"] += 1
                    continue
                if x.get("base") != r["base"]:
                    notes["reflection row on another first answer"] += 1
                    continue
                g = gold(ds, r["sample_id"], k)
                if g is None:
                    notes["unscored under the ITOpsGraph v2.1 labels"] += 1
                    continue
                if g != "keep":
                    r = dict(r, base_ok=target_correct(r["base"], g))
                    x = dict(x, BC_ok=target_correct(x.get("BC"), g))
                v = score_signals(r)
                u = uq.get(k, {})
                v.update({s: val for s, val in u.items() if not s.startswith("_")})
                if u.get("_lp_top") is not None or "lp_mean" in u:
                    lp_agree[m][0] += int(u.get("_lp_top") == r["base"])
                    lp_agree[m][1] += 1
                gl = r["gts"] if g == "keep" else g
                shown = any(target_correct(c.get("id"), gl) for c in r["ranking"])
                F[m].append(dict(ds=ds, sample=f"{ds}|{r['sample_id']}", key=k, base=r["base_ok"], shown=shown,
                                 bc=int(x["BC_ok"]), sig=v, pt_src=u.get("_pt_src"),
                                 lp_agree=u.get("_lp_top") == r["base"]))
    return F, notes, lp_agree


# ------------------------------------------------------------------------------------------------------ gates
def gate_weights(conf, k):
    """Reflect the k least confident fields; tied fields at the boundary share the remaining budget."""
    n = len(conf)
    w = np.zeros(n)
    if k <= 0:
        return w
    if k >= n:
        return np.ones(n)
    t = np.sort(conf)[int(math.ceil(k)) - 1]
    w[conf < t] = 1.0
    tied = conf == t
    w[tied] = min(max((k - w.sum()) / tied.sum(), 0.0), 1.0)
    return w


def oracle_weights(base, bc, k):
    gain = bc - base
    return gate_weights(-gain, k)  # fixable first, harmful last


def random_weights(n, k, seeds=N_RANDOM):
    w = np.zeros(n)
    kk = int(round(k))
    for _ in range(seeds):
        w[RNG.choice(n, kk, replace=False)] += 1.0 / seeds
    return w


def crossfit_logistic(fields, feats, name, folds=2, repeats=5):
    """P(first answer correct) from a logistic regression on the named signals, cross-fitted over samples: samples are
    split at random into two folds, each fold is scored by a model fitted on the other, and the scores are averaged
    over five random splits. No field is scored by a model that saw it. There are no development values for P(True)
    or stated confidence, so this is the development-free way to fit the combination."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.array([[f["sig"][x] for x in feats] for f in fields], float)
    y = np.array([f["base"] for f in fields])
    samples = sorted({f["sample"] for f in fields})
    pos = {c: i for i, c in enumerate(samples)}
    sidx = np.array([pos[f["sample"]] for f in fields])
    rng = np.random.default_rng(7)
    out = np.zeros(len(fields))
    for _ in range(repeats):
        fold = rng.integers(0, folds, len(samples))[sidx]
        for k in range(folds):
            tr, te = fold != k, fold == k
            mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            out[te] += mdl.predict_proba(X[te])[:, 1] / repeats
    for f, v in zip(fields, out):
        f["sig"][name] = float(v)


# ------------------------------------------------------------------------------------------------- bootstrap
class Boot:
    """Paired cluster bootstrap over samples within dataset, shared across models and signals."""

    def __init__(self, all_fields, B):
        clusters = sorted({f["sample"] for fs in all_fields for f in fs})
        self.idx = {c: i for i, c in enumerate(clusters)}
        by_ds = defaultdict(list)
        for c in clusters:
            by_ds[c.split("|")[0]].append(self.idx[c])
        self.counts = np.zeros((B, len(clusters)))
        for ds, ii in by_ds.items():
            draws = RNG.multinomial(len(ii), [1.0 / len(ii)] * len(ii), size=B)
            self.counts[:, ii] = draws

    def cl(self, fields):
        return np.array([self.idx[f["sample"]] for f in fields])

    def mean(self, fields, v):
        c = self.cl(fields)
        s = np.bincount(c, weights=v, minlength=self.counts.shape[1])
        n = np.bincount(c, minlength=self.counts.shape[1]).astype(float)
        return (self.counts @ s) / np.maximum(self.counts @ n, 1)

    def field_w(self, fields):
        return self.counts[:, self.cl(fields)]


def ci(draws):
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ---------------------------------------------------------------------------------------------- discrimination
def auroc_w(conf, err, w=None):
    """AUROC of -conf for errors (ties count half); w are field multiplicities (bootstrap)."""
    if w is None:
        w = np.ones(len(conf))
    u, gid = np.unique(-conf, return_inverse=True)
    pos = np.bincount(gid, weights=w * err, minlength=len(u))
    neg = np.bincount(gid, weights=w * (1 - err), minlength=len(u))
    below = np.cumsum(neg) - neg
    tp, tn = pos.sum(), neg.sum()
    if tp == 0 or tn == 0:
        return float("nan")
    return float((pos * (below + 0.5 * neg)).sum() / (tp * tn))


def auroc_draws(conf, err, W):
    u, gid = np.unique(-conf, return_inverse=True)
    out = np.empty(W.shape[0])
    for b in range(W.shape[0]):
        w = W[b]
        pos = np.bincount(gid, weights=w * err, minlength=len(u))
        neg = np.bincount(gid, weights=w * (1 - err), minlength=len(u))
        below = np.cumsum(neg) - neg
        out[b] = (pos * (below + 0.5 * neg)).sum() / max(pos.sum() * neg.sum(), 1e-12)
    return out


def auprc(conf, err):
    """Average precision for errors, ranking by -conf; tied fields enter together."""
    u, gid = np.unique(-conf, return_inverse=True)
    pos = np.bincount(gid, weights=err, minlength=len(u))[::-1]
    tot = np.bincount(gid, minlength=len(u))[::-1].astype(float)
    tp, n = np.cumsum(pos), np.cumsum(tot)
    if tp[-1] == 0:
        return float("nan")
    return float(((tp / n) * pos).sum() / tp[-1])


def aurc(conf, err):
    """Area under risk-coverage, most confident first, expected value over tie orders."""
    u, gid = np.unique(conf, return_inverse=True)
    e = np.bincount(gid, weights=err, minlength=len(u))[::-1]
    c = np.bincount(gid, minlength=len(u))[::-1].astype(float)
    risks, done_e, done_n = [], 0.0, 0
    for ge, gc in zip(e, c):
        j = np.arange(1, int(gc) + 1)
        risks.append((done_e + ge * j / gc) / (done_n + j))
        done_e += ge
        done_n += int(gc)
    return float(np.concatenate(risks).mean())


def pct_rank(x):
    """Within-model percentile rank (average for ties), so pooled discrimination is scale free."""
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt) - cnt
    return ((cum + (cnt + 1) / 2.0) / len(x))[inv]


# ---------------------------------------------------------------------------------------------------- cost
def cost_table():
    """(model, signal) -> (calls per field, input tokens, output tokens, seconds per field): per-call means from the
    token logs times the calls the signal needs (1, or 5 samples); for context the first pass (rank) and one
    Map-Reflect pass (reflect:B plus verify:BC when the answer changes, per reflected field)."""
    stages = {"uq:verbalised": ("verbalised", 1), "uq:p_true": ("p_true", 1), "uq:sample": ("samples", 5),
              "uq:logprob": ("logprob", 1)}
    acc = defaultdict(lambda: [0, 0, 0, 0.0])
    out = {}
    for m in MODELS:
        files = [f for f in glob.glob("results/naacl/v3/tokens/uq*__*.jsonl")
                 if f"__{m}__" in f or f"__{m}." in f or f.endswith(f"__{m}.jsonl")]
        for f in files:
            if not any(f"__{ds}__" in f for ds in DATASETS):
                continue
            for r in read(f):
                if r.get("stage") in stages:
                    a = acc[(m, stages[r["stage"]][0])]
                    a[0] += 1
                    a[1] += r.get("prompt_tokens") or 0
                    a[2] += r.get("completion_tokens") or 0
                    a[3] += r.get("seconds") or 0.0
        for st, (s, k) in stages.items():
            a = acc.get((m, s))
            if a and a[0]:
                out[(m, s)] = (k, k * a[1] / a[0], k * a[2] / a[0], k * a[3] / a[0])
        for f in glob.glob(f"results/naacl/v2/tokens/first__given__*__{m}.jsonl"):
            if any(f"__{ds}__" in f for ds in DATASETS):
                for r in read(f):
                    if r.get("stage") == "rank":
                        a = acc[(m, "first pass")]
                        a[0] += 1
                        a[1] += r.get("prompt_tokens") or 0
                        a[2] += r.get("completion_tokens") or 0
                        a[3] += r.get("seconds") or 0.0
        for f in glob.glob(f"results/naacl/v2/tokens/reflect__given__*__{m}.jsonl"):
            if any(f"__{ds}__" in f for ds in DATASETS):
                for r in read(f):
                    if r.get("stage") in ("reflect:B", "verify:BC"):
                        a = acc[(m, "Map-Reflect")]
                        a[0] += r["stage"] == "reflect:B"
                        a[1] += r.get("prompt_tokens") or 0
                        a[2] += r.get("completion_tokens") or 0
                        a[3] += r.get("seconds") or 0.0
        for s in ("first pass", "Map-Reflect"):
            a = acc.get((m, s))
            if a and a[0]:
                out[(m, s)] = (None, a[1] / a[0], a[2] / a[0], a[3] / a[0])
    return out


# ------------------------------------------------------------------------------------------------------- main
def main():
    F, notes, lp_agree = load()
    frozen = json.load(open(ARGS.frozen))
    fz_md5 = hashlib.md5(open(ARGS.frozen, "rb").read()).hexdigest()[:8]
    th = frozen["thresholds"]

    # per model: signals available on at least 90% of answered fields; population = fields with all of them
    avail, pop, dropped = {}, {}, {}
    for m in MODELS:
        fs = F[m]
        avail[m] = [s for s in SIGNALS if sum(s in f["sig"] for f in fs) >= 0.9 * len(fs)]
        avail[m] = [s for s in avail[m] if s not in COMBOS]
        pop[m] = [f for f in fs if all(s in f["sig"] for s in avail[m])]
        dropped[m] = len(fs) - len(pop[m])
        for c, feats in COMBOS.items():
            if all(x in avail[m] for x in feats):
                crossfit_logistic(pop[m], feats, c)
                avail[m].append(c)
    boot = Boot([pop[m] for m in MODELS], ARGS.B)

    def arr(fs, key):
        return np.array([f[key] for f in fs], float)

    def conf(fs, s):
        return np.array([f["sig"][s] for f in fs], float)

    # matched budget: the share the frozen v3 gate routes on each model's population
    budget, v3w = {}, {}
    for m in MODELS:
        t = th[m]
        # the population is answered fields, so every first pass here is valid
        w = np.array([float(router.band(f["sig"]["margin"], True, f["sig"]["s1"], t["t_m"], t["t_s"]) != router.KEEP)
                      for f in pop[m]])
        v3w[m], budget[m] = w, w.mean()

    def weights(m, s, b):
        fs = pop[m]
        k = b * len(fs)
        if s == "random":
            return random_weights(len(fs), k)
        if s == "oracle":
            return oracle_weights(arr(fs, "base"), arr(fs, "bc"), k)
        if s == "keep":
            return np.zeros(len(fs))
        if s == "v3":
            return v3w[m]
        return gate_weights(conf(fs, s), k)

    def after(m, w):
        fs = pop[m]
        b, r = arr(fs, "base"), arr(fs, "bc")
        return b + w * (r - b)

    L = []
    P = L.append
    P("# UQ ablation: routing the second pass by other uncertainty signals (protocol v2 test split)\n")
    P("Generated by `scripts/naacl/uq_ablation.py` from result files; plan and checklist in "
      "`docs/naacl/plan/uq_ablation.md`.\n")
    P("**Question.** Map-Reflect v3 sends a field to the reflective second pass when the first pass's own candidate "
      "scores say it is unsure (top-2 margin m = s1 - s2, with the top score s1). Would another uncertainty signal "
      "pick better fields? **Design.** The second pass is held fixed: Map-Reflect over the shown candidates (neutral "
      "+ evidence + check), which protocol v2 ran on every test field, so the outcome of any gate is known without new "
      "reflection calls. Each signal reflects the same number of fields: its least confident ones, at the share of "
      "fields the frozen v3 gate routes for that model (matched budget), and over a sweep from 10% to 100%. "
      "Bounds: random gate (mean of 20 seeds) and oracle gate (fields reflection fixes first, fields it breaks last). "
      "Intervals: paired cluster bootstrap over samples within dataset, shared across models and signals, "
      f"B = {ARGS.B}. No threshold is fitted for the matched-budget comparison.\n")
    P(f"Population: grouped test minus old development, nine datasets, answered fields with a Map-Reflect row on "
      f"the same first answer. Fields read: {notes['fields']:,}; declined by the first pass {notes['declined']:,} "
      f"(routed alike by every gate, left out); without a reflection row {notes['no reflection row']:,}; reflection "
      f"row on another first answer {notes['reflection row on another first answer']:,}; ITOpsGraph labels "
      f"{ARGS.labels} (fields unscored under v2.1: {notes['unscored under the ITOpsGraph v2.1 labels']:,}). Per "
      "model, the population is the answered fields where every signal available for that model exists.\n")
    P(f"Matched budget from `{ARGS.frozen}` (dated {frozen.get('date')}, md5 {fz_md5}): the share of the population "
      "the frozen v3 gate routes (scripts/naacl/router.py: kept when m > t_m, so routed when m <= t_m).\n")
    P("| Model | fields | dropped (a signal missing) | first pass right | matched budget | signals |")
    P("|---|--:|--:|--:|--:|---|")
    for m in MODELS:
        P(f"| {MNAME[m]} | {len(pop[m]):,} | {dropped[m]:,} | {100 * arr(pop[m], 'base').mean():.1f}% | "
          f"{100 * budget[m]:.1f}% | {', '.join(SIGNALS[s][0] for s in avail[m] if s != 'margin')} |")
    P("")
    lpa = ", ".join(f"{MNAME[m]} {100 * a / n:.1f}% of {n:,}" for m, (a, n) in lp_agree.items() if n)
    if lpa:
        P(f"Token log-probabilities come from one greedy regeneration of the first-pass prompt on ENDPOINT_A (Granite-4 "
          f"through the same weights on ENDPOINT_A). Its first ranked candidate equals the stored first answer for: {lpa}. "
          "Llama-4 has no log-probabilities (endpoint_b returns none; its ENDPOINT_A endpoint takes minutes per call). "
          "Llama-4's P(True) is its stated True or False (no log-probabilities), so it has two values only.\n")

    if REPAIR_AGREE:
        P("Side check, two independent draws of the same calls: the repair rows (results/naacl/uq_ablation/repair) "
          "against the canonical full ToolBench reruns in results/naacl/v3/uq, which are the ones used: "
          + "; ".join(f"{MNAME[m]} {sname}: {len(v)} fields, " + (
              f"{100 * np.mean([abs(a - b) < 0.05 for a, b in v]):.0f}% within 0.05, mean absolute difference "
              f"{np.mean([abs(a - b) for a, b in v]):.3f}") for (m, sname), v in sorted(REPAIR_AGREE.items())) + ".\n")

    # ---------------------------------------------------------------- matched budget
    def group_rows(models, sigs, b_of):
        """{signal: (accuracy, delta vs margin, (lo, hi))} pooled over models, each at its own budget."""
        fields = [f for m in models for f in pop[m]]
        acc_v = {}
        for s in sigs:
            acc_v[s] = np.concatenate([after(m, weights(m, s, b_of(m))) for m in models])
        out = {}
        for s in sigs:
            d = acc_v[s] - acc_v["margin"]
            draws = boot.mean(fields, d)
            out[s] = (100 * acc_v[s].mean(), 100 * d.mean(), tuple(100 * x for x in ci(draws)))
        return out, len(fields)

    ROWS = ["keep", "random"] + ["margin"] + ALT + ["v3", "oracle"]
    LBL = dict({s: SIGNALS[s][0] for s in SIGNALS}, keep="keep first answers", random="random gate (20 seeds)",
               v3="v3 gate (m and s1 thresholds)", oracle="oracle gate")
    P("## 1. Gate swap at matched budget (final accuracy after the second pass)\n")
    P("Accuracy over answered fields; share of the oracle gain = (gate - keep) / (oracle - keep), the part of the "
      "best attainable improvement at this budget that the gate realises; delta = signal minus margin in points, "
      "95% interval. A negative delta means the signal routes worse than the margin. The second pass itself has "
      "little headroom on this split (compare keep, reflect-all at 100% in section 2, and oracle), so accuracy "
      "differences between gates are bounded by a fraction of a point; section 3 measures the ranking quality "
      "directly.\n")
    headline = {}
    for m in MODELS + ["pooled"]:
        models = MODELS if m == "pooled" else [m]
        sigs = [s for s in ROWS if s in ("keep", "random", "v3", "oracle", "margin") or
                all(s in avail[x] for x in models)]
        res, n = group_rows(models, sigs, lambda x: budget[x])
        bud = np.mean([budget[x] for x in models]) if m != "pooled" else \
            sum(budget[x] * len(pop[x]) for x in models) / n
        P(f"### {'All four models' if m == 'pooled' else MNAME[m]} ({n:,} fields, budget {100 * bud:.1f}%)\n")
        P("| Gate | extra calls per field | accuracy | share of the oracle gain | delta vs margin [95% CI] |")
        P("|---|--:|--:|--:|--:|")
        k0, o0 = res["keep"][0], res["oracle"][0]
        for s in sigs:
            a, d, (lo, hi) = res[s]
            calls = SIGNALS[s][2] if s in SIGNALS else 0
            dd = "" if s == "margin" else f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            sh = f"{100 * (a - k0) / (o0 - k0):.0f}%" if o0 > k0 else ""
            P(f"| {LBL[s]} | {calls} | {a:.2f} | {sh} | {dd} |")
        P("")
        headline[m] = res
    if all("combo_paid" in avail[m] for m in MODELS):
        rows = []
        for m in MODELS:
            fs = pop[m]
            wm, wc = weights(m, "margin", budget[m]), weights(m, "combo_paid", budget[m])
            for f, a, b in zip(fs, wm, wc):
                if not f["base"]:
                    rows.append((a, b, f["shown"], f["bc"]))
        R_ = np.array(rows, float)
        w0 = R_[:, 0]
        ERRSTATS.update(UQAblErrRouted=f"{100 * w0.sum() / len(R_):.0f}",
                        UQAblErrInShown=f"{100 * (w0 * R_[:, 2]).sum() / max(w0.sum(), 1e-9):.0f}",
                        UQAblErrRepaired=f"{100 * (w0 * R_[:, 3]).sum() / max(w0.sum(), 1e-9):.0f}")
        for name, col in (("margin", 0), ("combination", 1)):
            w = R_[:, col]
            P(f"First-pass errors the {name} gate reflects at the matched budget: {w.sum():,.0f} of {len(R_):,}; "
              f"gold among the shown candidates for {100 * (w * R_[:, 2]).sum() / max(w.sum(), 1e-9):.0f}% of them, "
              f"repaired by the second pass {100 * (w * R_[:, 3]).sum() / max(w.sum(), 1e-9):.0f}%.")
        only = (R_[:, 1] - R_[:, 0]).clip(0)
        P(f"Errors the combination reflects and the margin does not ({only.sum():,.0f}): gold shown for "
          f"{100 * (only * R_[:, 2]).sum() / max(only.sum(), 1e-9):.0f}%, repaired {100 * (only * R_[:, 3]).sum() / max(only.sum(), 1e-9):.0f}%.\n")

    # pooled over the models that have each paid signal
    P("### Signals available on some models only, pooled over those models\n")
    P("| Signal | models | fields | budget | accuracy | margin accuracy | delta vs margin [95% CI] |")
    P("|---|---|--:|--:|--:|--:|--:|")
    subset_res = {}
    for s in ALT:
        models = [m for m in MODELS if s in avail[m]]
        if not models or len(models) == len(MODELS):
            continue
        res, n = group_rows(models, ["margin", s], lambda x: budget[x])
        bud = sum(budget[x] * len(pop[x]) for x in models) / n
        a, d, (lo, hi) = res[s]
        subset_res[s] = (models, n, res)
        P(f"| {SIGNALS[s][0]} | {', '.join(MNAME[x] for x in models)} | {n:,} | {100 * bud:.1f}% | {a:.2f} | "
          f"{res['margin'][0]:.2f} | {d:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    P("")

    # ---------------------------------------------------------------- budget sweep
    P("## 2. Budget sweep (all four models pooled, every model at the same share)\n")
    P("Final accuracy (%) when each gate reflects the given share of each model's answered fields. Signals that "
      "exist on some models only are in the per-model tables below.\n")
    common = [s for s in SIGNALS if all(s in avail[m] for m in MODELS)]
    sweep_cols = ["random"] + common + ["oracle"]
    P("| budget | " + " | ".join(LBL[s] for s in sweep_cols) + " |")
    P("|--:|" + "--:|" * len(sweep_cols))
    sweep_delta = {}
    for b in BUDGETS:
        vals = {s: np.concatenate([after(m, weights(m, s, b)) for m in MODELS]) for s in sweep_cols}
        P(f"| {int(round(100 * b))}% | " + " | ".join(f"{100 * vals[s].mean():.2f}" for s in sweep_cols) + " |")
        fields = [f for m in MODELS for f in pop[m]]
        for s in common:
            if s != "margin":
                d = vals[s] - vals["margin"]
                sweep_delta[(s, b)] = (100 * d.mean(), tuple(100 * x for x in ci(boot.mean(fields, d))))
    P("")
    P("Signal minus margin across the sweep, points [95% CI]:\n")
    P("| budget | " + " | ".join(LBL[s] for s in common if s != "margin") + " |")
    P("|--:|" + "--:|" * (len(common) - 1))
    for b in BUDGETS:
        P(f"| {int(round(100 * b))}% | " + " | ".join(
            f"{sweep_delta[(s, b)][0]:+.2f} [{sweep_delta[(s, b)][1][0]:+.2f}, {sweep_delta[(s, b)][1][1]:+.2f}]"
            for s in common if s != "margin") + " |")
    P("")
    for m in MODELS:
        cols = ["random"] + avail[m] + ["oracle"]
        P(f"### {MNAME[m]} ({len(pop[m]):,} fields)\n")
        P("| budget | " + " | ".join(LBL[s] for s in cols) + " |")
        P("|--:|" + "--:|" * len(cols))
        for b in BUDGETS:
            P(f"| {int(round(100 * b))}% | " + " | ".join(f"{100 * after(m, weights(m, s, b)).mean():.2f}"
                                                        for s in cols) + " |")
        P("")
    # area under the sweep (10%..100%), a single number per signal and model
    P("Mean final accuracy over the ten budgets (area under the sweep), per model:\n")
    P("| Signal | " + " | ".join(MNAME[m] for m in MODELS) + " |")
    P("|---|" + "--:|" * len(MODELS))
    for s in ["random"] + list(SIGNALS) + ["oracle"]:
        cells = []
        for m in MODELS:
            if s in ("random", "oracle") or s in avail[m]:
                cells.append(f"{100 * np.mean([after(m, weights(m, s, b)).mean() for b in BUDGETS]):.2f}")
            else:
                cells.append("")
        P(f"| {LBL[s]} | " + " | ".join(cells) + " |")
    P("")

    # ---------------------------------------------------------------- discrimination
    P("## 3. Discrimination of first-pass errors\n")
    P("AUROC and AUPRC treat first-pass errors as positives (higher is better); AURC is the area under the "
      "risk-coverage curve, most confident first (lower is better). Ties are scored by their expected value. The "
      "pooled rows rank each signal within its model (percentile rank) before pooling, so scale differences "
      "between models do not count. AUROC difference: signal minus margin, paired cluster bootstrap.\n")
    P("| Model | Signal | fields | AUROC | AUROC minus margin [95% CI] | AUPRC | AURC |")
    P("|---|---|--:|--:|--:|--:|--:|")
    disc = {}
    for m in MODELS + ["pooled"]:
        models = MODELS if m == "pooled" else [m]
        sigs = [s for s in SIGNALS if all(s in avail[x] for x in models)]
        fields = [f for x in models for f in pop[x]]
        err = 1 - np.concatenate([arr(pop[x], "base") for x in models])
        W = boot.field_w(fields)
        cv = {}
        for s in sigs:
            cv[s] = np.concatenate([pct_rank(conf(pop[x], s)) if m == "pooled" else conf(pop[x], s)
                                    for x in models])
        dm = auroc_draws(cv["margin"], err, W)
        for s in sigs:
            a = auroc_w(cv[s], err)
            if s == "margin":
                dd = ""
            else:
                lo, hi = ci(auroc_draws(cv[s], err, W) - dm)
                dd = f"{a - auroc_w(cv['margin'], err):+.3f} [{lo:+.3f}, {hi:+.3f}]"
            disc[(m, s)] = (a, auprc(cv[s], err), aurc(cv[s], err))
            P(f"| {'All four' if m == 'pooled' else MNAME[m]} | {SIGNALS[s][0]} | {len(fields):,} | {a:.3f} | {dd} | "
              f"{disc[(m, s)][1]:.3f} | {disc[(m, s)][2]:.4f} |")
    P("")
    P("Signals that exist on some models only, pooled over those models (within-model ranks):\n")
    P("| Signal | models | fields | AUROC | margin AUROC | AUROC minus margin [95% CI] | AUPRC | AURC |")
    P("|---|---|--:|--:|--:|--:|--:|--:|")
    for s in ALT:
        models = [m for m in MODELS if s in avail[m]]
        if not models or len(models) == len(MODELS):
            continue
        fields = [f for x in models for f in pop[x]]
        err = 1 - np.concatenate([arr(pop[x], "base") for x in models])
        W = boot.field_w(fields)
        cs = np.concatenate([pct_rank(conf(pop[x], s)) for x in models])
        cm = np.concatenate([pct_rank(conf(pop[x], "margin")) for x in models])
        lo, hi = ci(auroc_draws(cs, err, W) - auroc_draws(cm, err, W))
        a, am = auroc_w(cs, err), auroc_w(cm, err)
        disc[("subset", s)] = (a, am, lo, hi, models)
        P(f"| {SIGNALS[s][0]} | {', '.join(MNAME[x] for x in models)} | {len(fields):,} | {a:.3f} | {am:.3f} | "
          f"{a - am:+.3f} [{lo:+.3f}, {hi:+.3f}] | {auprc(cs, err):.3f} | {aurc(cs, err):.4f} |")
    P("")

    lp_models = [m for m in MODELS if "lp_mean" in avail[m]]
    if lp_models:
        P("Token log-probability signals on the fields where the regeneration's first ranked candidate is the stored "
          "first answer (so the log-probabilities describe that answer), against the margin on the same fields:\n")
        P("| Model | fields | margin AUROC | answer-token log-prob. | mean token log-prob. | mean token entropy |")
        P("|---|--:|--:|--:|--:|--:|")
        for m in lp_models:
            fs = [f for f in pop[m] if f["lp_agree"]]
            err = 1 - arr(fs, "base")
            P(f"| {MNAME[m]} | {len(fs):,} | " + " | ".join(
                f"{auroc_w(conf(fs, s), err):.3f}" for s in ("margin", "lp_answer", "lp_mean", "lp_entropy")) + " |")
        P("")

    # ---------------------------------------------------------------- cost
    cost = cost_table()
    P("## 4. Cost per field\n")
    P("Extra model calls a signal needs beyond the first pass; tokens and seconds per field are the per-call means "
      "of the token logs times the calls (seconds are the logged call latency, not wall time of a parallel run). "
      "The score-based signals are read from the first pass and cost nothing. For scale, the first pass itself and "
      "one Map-Reflect pass (reflection plus the check when it changes the answer, per reflected field) are listed.\n")
    P("| Model | signal | calls per field | input tokens per field | output tokens per field | seconds per field |")
    P("|---|---|--:|--:|--:|--:|")
    lab = {"first pass": "first pass (context)", "Map-Reflect": "Map-Reflect pass (context)",
           "verbalised": "stated confidence", "p_true": "P(True)", "samples": "5 samples (semantic entropy, "
           "distinct answers)", "logprob": "greedy regeneration with log-probabilities"}
    for m in MODELS:
        P(f"| {MNAME[m]} | margin, top score, ratio, score entropy | 0 | 0 | 0 | 0 |")
        for s in ("verbalised", "p_true", "logprob", "samples", "first pass", "Map-Reflect"):
            if (m, s) not in cost:
                continue
            k, i, o, sec = cost[(m, s)]
            P(f"| {MNAME[m]} | {lab[s]} | {'' if k is None else k} | {i:,.0f} | {o:,.0f} | {sec:.2f} |")
    P("")

    # ---------------------------------------------------------------- dev-fitted thresholds (free signals)
    dev = dev_threshold_section(boot)
    L.extend(dev)

    # ---------------------------------------------------------------- verdict and writer notes
    L.extend(verdict(headline, subset_res, disc, avail, budget, pop, sweep_delta))
    out = Path(ARGS.out)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


def dev_threshold_section(boot):
    """One threshold per model and free signal, fitted on development (v3 development BC outcomes), applied once."""
    L = ["## 5. Free signals with a threshold fitted on development\n"]
    rows_dev, rows_test = defaultdict(list), defaultdict(list)
    bad = 0
    for ds in DATASETS:
        man = json.load(open(f"data/splits/{ds}.json"))
        dev_ids, test_ids = set(man["dev"]), set(man["test"]) - set(man.get("old_dev_excluded") or [])
        for m in MODELS:
            f = f"results/naacl/v3/dev/given/{ds}__{m}/fields.jsonl"
            if not os.path.exists(f):
                L.append(f"Development rows missing for {ds} {m}; section skipped.\n")
                return L
            fp = first_pass(ds, m, dev_ids)
            for r in read(f):
                x = fp.get(r["key"])
                if x is None or not x["valid"] or "BC_ok" not in r:
                    continue
                if r.get("base") != x["base"]:
                    bad += 1
                    continue
                g = gold(ds, x["sample_id"], x["key"])
                if g is None:
                    continue
                bo, bc = x["base_ok"], int(r["BC_ok"])
                if g != "keep":
                    bo, bc = target_correct(x["base"], g), target_correct(r.get("BC"), g)
                rows_dev[m].append(dict(sig=score_signals(x), base=bo, bc=bc))
            fpt = first_pass(ds, m, test_ids)
            refl = {r["key"]: r for r in read(f"results/naacl/v2/reflect/given/{ds}__{m}/fields.jsonl")}
            for k, x in fpt.items():
                r = refl.get(k)
                if not x["valid"] or r is None or "BC_ok" not in r or r.get("base") != x["base"]:
                    continue
                g = gold(ds, x["sample_id"], k)
                if g is None:
                    continue
                bo, bc = x["base_ok"], int(r["BC_ok"])
                if g != "keep":
                    bo, bc = target_correct(x["base"], g), target_correct(r.get("BC"), g)
                rows_test[m].append(dict(sig=score_signals(x), base=bo, bc=bc, sample=f"{ds}|{x['sample_id']}"))
    L.append("Gate: reflect when the signal is at or below a threshold. The threshold maximises development accuracy "
             "after the second pass (ties go to fewer reflections), per model, using the v3 development Map-Reflect "
             "outcomes on every development field; it is then applied once to test. This uses the margin alone, "
             "not the v3 two-signal rule, so all four free signals are fitted the same way. The paid signals have no "
             f"development values. Development rows on another first answer, left out: {bad}.\n")
    L.append("| Model | signal | threshold | dev accuracy | test reflected | test accuracy | delta vs margin [95% CI] |")
    L.append("|---|---|--:|--:|--:|--:|--:|")
    sigs = ["margin", "s1", "ratio", "entropy"]
    pooled = defaultdict(list)
    for m in MODELS:
        D, T = rows_dev[m], rows_test[m]
        if not D or not T:
            continue
        db, dr = np.array([r["base"] for r in D], float), np.array([r["bc"] for r in D], float)
        tb, tr = np.array([r["base"] for r in T], float), np.array([r["bc"] for r in T], float)
        res = {}
        for s in sigs:
            dv = np.array([r["sig"][s] for r in D])
            best = (db.mean(), 0.0, -np.inf)  # (accuracy, -share, threshold): keep-all first
            for t in np.unique(dv):
                w = (dv <= t).astype(float)
                a = (db + w * (dr - db)).mean()
                if a > best[0] + 1e-12:
                    best = (a, -w.mean(), t)
            t = best[2]
            tv = np.array([r["sig"][s] for r in T])
            w = (tv <= t).astype(float)
            res[s] = (t, best[0], w.mean(), tb + w * (tr - tb))
        for s in sigs:
            t, da, share, v = res[s]
            pooled[s].append(v)
            d = v - res["margin"][3]
            lo, hi = ci(boot.mean(T, d)) if s != "margin" else (0, 0)
            dd = "" if s == "margin" else f"{100 * d.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"
            L.append(f"| {MNAME[m]} | {SIGNALS[s][0]} | {t:.3g} | {100 * da:.2f} | {100 * share:.1f}% | "
                     f"{100 * v.mean():.2f} | {dd} |")
        pooled["_fields"].extend(T)
    if pooled["_fields"]:
        for s in sigs:
            v = np.concatenate(pooled[s])
            d = v - np.concatenate(pooled["margin"])
            lo, hi = ci(boot.mean(pooled["_fields"], d))
            dd = "" if s == "margin" else f"{100 * d.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"
            L.append(f"| All four | {SIGNALS[s][0]} | per model | | | {100 * v.mean():.2f} | {dd} |")
    L.append("")
    return L


def verdict(headline, subset_res, disc, avail, budget, pop, sweep_delta):
    L = ["## 6. Verdict\n"]
    pooled = headline["pooled"]
    m_acc = pooled["margin"][0]
    better, worse, tie = [], [], []
    for s, (a, d, (lo, hi)) in pooled.items():
        if s not in ALT:
            continue
        (better if lo > 0 else worse if hi < 0 else tie).append((s, d, lo, hi))
    for s, (models, n, res) in subset_res.items():
        a, d, (lo, hi) = res[s]
        (better if lo > 0 else worse if hi < 0 else tie).append((s, d, lo, hi))

    def fmt(x):
        return ", ".join(f"{SIGNALS[s][0]} {d:+.2f} [{lo:+.2f}, {hi:+.2f}]" for s, d, lo, hi in x) or "none"
    L.append(f"At the matched budget, the margin gate reaches {m_acc:.2f}% pooled over the four models "
             f"(keep-all {pooled['keep'][0]:.2f}%, random {pooled['random'][0]:.2f}%, oracle {pooled['oracle'][0]:.2f}%). "
             "Each signal minus the margin, points with 95% interval (signals on some models only are compared with "
             "the margin on the same models):\n")
    L.append(f"- Worse than the margin: {fmt(worse)}.")
    L.append(f"- Not distinguishable from the margin: {fmt(tie)}.")
    L.append(f"- Better than the margin: {fmt(better)}.\n")
    L.append("Discrimination (pooled AUROC on within-model ranks): " + ", ".join(
        f"{SIGNALS[s][0]} {disc[('pooled', s)][0]:.3f}" for s in sorted(
            [s for s in SIGNALS if ("pooled", s) in disc], key=lambda s: -disc[("pooled", s)][0])) + ".")
    if ("pooled", "combo_paid") in disc and "combo_paid" in pooled:
        a, d, (lo, hi) = pooled["combo_paid"]
        L.append("")
        L.append(f"The cross-fitted combination of margin, top score, P(True) and stated confidence ranks first-pass "
                 f"errors better than the margin (pooled AUROC {disc[('pooled', 'combo_paid')][0]:.3f} against "
                 f"{disc[('pooled', 'margin')][0]:.3f}, AUPRC {disc[('pooled', 'combo_paid')][1]:.3f} against "
                 f"{disc[('pooled', 'margin')][1]:.3f}), but as a gate at the matched budget it reaches {a:.2f}% "
                 f"({d:+.2f} [{lo:+.2f}, {hi:+.2f}] against the margin) for two extra calls per field. Better error "
                 "detection does not translate into better routing at this budget: both gates already send almost "
                 "every first-pass error to the second pass (section 1 counts them), the gold target is among the "
                 "shown candidates for only about a third of those errors, and the second pass repairs about one in "
                 "twenty of them, so which signal picks the fields changes the final accuracy by hundredths of a point. "
                 "Across the sweep (section 2), budgets where its interval excludes zero in its favour: "
                 + (", ".join(f"{int(round(100 * b))}% ({d:+.2f} [{lo:+.3f}, {hi:+.3f}])"
                              for (sg, b), (d, (lo, hi)) in sorted(sweep_delta.items()) if sg == "combo_paid" and lo > 0)
                    or "none") + "; its largest lead is "
                 + max((f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}] at {int(round(100 * b))}%"
                        for (sg, b), (d, (lo, hi)) in sweep_delta.items() if sg == "combo_paid"),
                       key=lambda t: float(t.split()[0])) + ".")
    sub = [(s, v) for (g, s), v in disc.items() if g == "subset"]
    if sub:
        L.append("On the models where they exist: " + ", ".join(
            f"{SIGNALS[s][0]} {v[0]:.3f} against margin {v[1]:.3f}" for s, v in sub) + ".")
    L.append("")
    L.append("## 7. For the writer: proposed table and macros (not written to the paper)\n")
    L.append("Proposed table `tab:uq_ablation` (pooled over the four models; signals on some models only are compared "
             "with the margin on those models): signal, extra calls per field, pooled AUROC, final accuracy at the "
             "matched budget, difference from the margin with 95% interval.\n")
    L.append("| Signal | extra calls | AUROC | accuracy | minus margin [95% CI] |")
    L.append("|---|--:|--:|--:|--:|")
    mac = {}
    cname = {"margin": "Margin", "s1": "TopScore", "ratio": "Ratio", "entropy": "ScoreEnt", "lp_answer": "AnsLogp",
             "lp_mean": "MeanLogp", "lp_entropy": "TokEnt", "p_true": "PTrue", "verbalised": "Stated",
             "sem_entropy": "SemEnt", "n_distinct": "Distinct", "combo_free": "ComboFree",
             "combo_paid": "ComboPaid"}
    for s in SIGNALS:
        if s in pooled:
            a, d, (lo, hi) = pooled[s]
            au = disc.get(("pooled", s), (float("nan"),))[0]
        elif s in subset_res:
            a, d, (lo, hi) = subset_res[s][2][s]
            au = disc.get(("subset", s), (float("nan"),))[0]
        else:
            continue
        dd = "" if s == "margin" else f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        name = SIGNALS[s][0]
        if s in subset_res and s not in pooled:
            ms = subset_res[s][0]
            name += (f" ({', '.join(MNAME[x] for x in ms)} only; margin there: AUROC "
                     f"{disc[('subset', s)][1]:.3f}, accuracy {subset_res[s][2]['margin'][0]:.2f})")
        L.append(f"| {name} | {SIGNALS[s][2]} | {au:.3f} | {a:.2f} | {dd} |")
        mac[f"UQAbl{cname[s]}AUC"] = f"{au:.3f}"
        mac[f"UQAbl{cname[s]}Acc"] = f"{a:.2f}"
        if s != "margin":
            mac[f"UQAbl{cname[s]}Delta"] = f"{d:+.2f}"
            mac[f"UQAbl{cname[s]}Lo"] = f"{lo:+.2f}"
            mac[f"UQAbl{cname[s]}Hi"] = f"{hi:+.2f}"
    mac["UQAblKeepAcc"] = f"{pooled['keep'][0]:.2f}"
    mac["UQAblRandomAcc"] = f"{pooled['random'][0]:.2f}"
    mac["UQAblOracleAcc"] = f"{pooled['oracle'][0]:.2f}"
    n = sum(len(pop[m]) for m in MODELS)
    mac["UQAblBudget"] = f"{100 * sum(budget[m] * len(pop[m]) for m in MODELS) / n:.1f}"
    mac["UQAblFields"] = f"{n:,}".replace(",", "{,}")
    mac.update(ERRSTATS)
    words = {1: "one", 2: "two", 3: "three", 4: "four"}
    for sg, nm in (("lp_answer", "AnsLogp"), ("sem_entropy", "SemEnt"), ("n_distinct", "Distinct")):
        if sg in subset_res:
            mac[f"UQAbl{nm}Models"] = words[len(subset_res[sg][0])]
    L.append("")
    L.append("Macros (signed values plain, for math mode), written to `paper/numbers_uq.tex`; the table is written to "
             "`paper/tab_uqsignals.tex`:\n")
    L.append("```latex")
    for k, v in mac.items():
        L.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    L.append("```\n")
    return L


def write_paper(mac):
    """paper/numbers_uq.tex (every UQAbl macro) and paper/tab_uqsignals.tex (the writer's table, tab:uqsignals)."""
    paper = REPO / "paper"
    head = ("% Generated by scripts/naacl/uq_ablation.py from result files (docs/naacl/results/uq_ablation.md); "
            "do not edit by hand.\n")
    (paper / "numbers_uq.tex").write_text(head + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    need = ["MarginAUC", "TopScoreAUC", "RatioAUC", "ScoreEntAUC", "AnsLogpAUC", "StatedAUC", "PTrueAUC", "SemEntAUC",
            "DistinctAUC", "ComboPaidAUC", "TopScoreDelta", "RatioDelta", "ScoreEntDelta", "AnsLogpDelta",
            "StatedDelta", "PTrueDelta", "SemEntDelta", "DistinctDelta", "ComboPaidDelta", "SemEntModels",
            "AnsLogpModels"]
    missing = [n for n in need if f"UQAbl{n}" not in mac]
    if missing:
        sys.exit(f"cannot write tab_uqsignals.tex, macros missing: {missing}")
    tab = r"""% Generated by scripts/naacl/uq_ablation.py (writer's form); every cell is a macro from numbers_uq.tex. Source:
% docs/naacl/results/uq_ablation.md. Extra calls are per field. Routing: matched-budget gate swap, pooled.
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
\midrule
Token log-probability$^\dagger$ & 1 & \UQAblAnsLogpAUC & $\UQAblAnsLogpDelta$ \\
Stated confidence & 1 & \UQAblStatedAUC & $\UQAblStatedDelta$ \\
P(True) & 1 & \UQAblPTrueAUC & $\UQAblPTrueDelta$ \\
Semantic entropy$^\ast$ & 5 & \UQAblSemEntAUC & $\UQAblSemEntDelta$ \\
Distinct answers$^\ast$ & 5 & \UQAblDistinctAUC & $\UQAblDistinctDelta$ \\
\midrule
Logistic: margin, top score, & 2 & \UQAblComboPaidAUC & $\UQAblComboPaidDelta$ \\
\quad P(True), stated & & & \\
\bottomrule
\end{tabular}
\caption{The free margin is the best single error detector, and no paid signal routes reflection better. Error AUROC: how well the signal ranks wrong first answers above right ones. Routing: change in accuracy (points) when the signal, instead of the margin, picks the same number of fields to reflect. $^\dagger$Answer-token log-probability, \UQAblAnsLogpModels{} models (Llama-4 returns none). $^\ast$Five samples, \UQAblSemEntModels{} models; these rows compare with the margin on the same models.}
\label{tab:uqsignals}
\end{table}
"""
    (paper / "tab_uqsignals.tex").write_text(tab)
    print(f"wrote {paper / 'numbers_uq.tex'} ({len(mac)} macros) and {paper / 'tab_uqsignals.tex'}")


if __name__ == "__main__":
    if ARGS.v3:
        main()
    else:
        import uq_ablation_v4
        uq_ablation_v4.run(sys.modules[__name__])
