#!/usr/bin/env python3
"""The benefit estimate u(x) = P(repair) - P(break) as a UQ method: its probabilities, its alternatives and its
threshold, on the stored method v5 outputs. Offline; no model call. Exploratory (specified after the test results
were seen, in answer to an ARR review asking for UQ validation of the benefit classifier).

Everything reads stored outputs only: results/naacl/v6/v5/{dev,test} (both second-pass actions measured on every
field), the frozen benefit model and threshold (docs/naacl/results/v5_frozen.json) and the frozen development cost
estimates of the matched comparisons (docs/naacl/results/v5_matched_cost_frozen.json). Nothing is refit on test.

  probs    the frozen three-outcome model on the core test set and on the held-out sets (no dataset indicator there):
           outcome counts of the field's routed action under the either-order verifier; multiclass log loss and Brier
           score against a prior that predicts the development outcome frequencies; per outcome, the AUROC and the
           ECE (10 equal-width bins) of P(repair) and P(break); u(x) read as the expected change in correctness from
           reflecting: its AUROC for a repair, for a break among first-pass correct answers and for a first-pass error,
           its calibration error over 10 equal-mass bins of u, and the predicted against the observed change on the
           fields Map-Reflect selects and on all fields.
  alts     alternative estimates, each fitted on development only with the frozen features and standardisation, the
           same grouped 5-fold cross-validation for C and the same threshold rule (cross-fitted development accuracy,
           ties to fewer calls): P(error), a binary logistic model of first-pass error (the correctness estimate of
           standard UQ); P(repair) alone, from the frozen model; and leave-one-model-out transfer, where each model's
           test fields are scored by a benefit model fitted on the other three models' development fields without
           model indicators. Each is scored at Map-Reflect's calls per model (sequential whole-field fill with the
           frozen development cost estimates, as v5_controls.py) and at its own threshold, with paired split-group
           bootstrap intervals against Map-Reflect (B = 2000). The margin at Map-Reflect's calls is the reference that
           must reproduce F-VFiveCtlMarginMatchedVsMain.
  tau      test accuracy and calls when the frozen u(x) is thresholded at other values of tau, beside the cross-fitted
           development accuracy that chose tau.

Writes docs/naacl/results/v5_benefit_uq_facts.json (F-BenUQ*).
Usage: ITOPS_V21=<v2.1 label file> .venv/bin/python scripts/naacl/v5_benefit_uq.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
sys.argv = ["v5_matched.py", "--stage", "score", "--B", "2000"]
import v5_matched as VM  # noqa: E402  (v5 loader, frozen-model scoring, cost model and sequential fill)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

V5, GS, MODELS = VM.V5, VM.GS, VM.MODELS
FZ = VM.FZ
BM = FZ["c_benefit"]
MU, SD = np.array(BM["standardise"]["mu"]), np.array(BM["standardise"]["sd"])
S = FZ["intervention"]["outcome_field"]
E = json.loads(VM.COST_F.read_text())["est"]
FACTS_F = ROOT / "docs/naacl/results/v5_benefit_uq_facts.json"
CGRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
FACTS = []
DEF = ("Exploratory, specified after the test results were seen; offline on the stored method v5 outputs "
       "(results/naacl/v6/v5), frozen benefit model and tau (docs/naacl/results/v5_frozen.json); nothing refit on test.")


def fact(fid, claim, value, lo=None, hi=None):
    assert fid.isalpha(), fid
    FACTS.append(dict(id="F-BenUQ" + fid, claim=claim + ". " + DEF, value=str(value),
                      lo=None if lo is None else str(lo), hi=None if hi is None else str(hi),
                      src=str(FACTS_F.relative_to(ROOT)), fn="v5_benefit_uq", status="exploratory"))


def sg(x, nd=2):
    return f"{x:+.{nd}f}"


def design(c, drop_model=False):
    M_, D_ = V5.feats(c)
    X = np.hstack([(M_ - MU) / SD, D_])
    return X[:, :X.shape[1] - len(MODELS)] if drop_model else X


def probs(X, model):
    W, b = np.array(model["coef"]), np.array(model["intercept"])
    z = X @ W.T + b
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    cl = list(model["classes"])
    return p[:, cl.index(1)], p[:, cl.index(-1)]


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum() for b in range(bins) if (idx == b).any()) / len(p))


def ece_mass(p, y, bins=10):
    order = np.argsort(p, kind="stable")
    return float(sum(abs(p[ix].mean() - y[ix].mean()) * len(ix) for ix in np.array_split(order, bins)) / len(p))


def groups_of(cells, datasets, dup):
    g = GS.split_groups(datasets, dup)
    return np.unique([f"{c.ds}::{g[(c.ds, s)]}" for c in cells for s in c.sid], return_inverse=True)[1]


def fit_cv(X, y, gid, binary=False):
    """C by grouped 5-fold CV log loss (the frozen rule), the full fit and cross-fitted scores."""
    folds = list(GroupKFold(n_splits=5).split(X, y, gid))
    best = None
    for C in CGRID:
        ll = 0.0
        for tr, te in folds:
            clf = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])
            ll -= np.log(np.clip(p[np.arange(len(te)), np.searchsorted(clf.classes_, y[te])], 1e-12, 1)).sum()
        if best is None or ll / len(y) < best[0]:
            best = (ll / len(y), C)
    C = best[1]
    clf = LogisticRegression(C=C, max_iter=5000).fit(X, y)
    cf = {}
    for tr, te in folds:
        m_ = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr])
        p = m_.predict_proba(X[te])
        cl = list(m_.classes_)
        for k in cl:
            cf.setdefault(k, np.zeros(len(y)))[te] = p[:, cl.index(k)]
    return clf, C, cf


def score_of(clf, X, kind):
    p = clf.predict_proba(X)
    cl = list(clf.classes_)
    if kind == "u":
        return p[:, cl.index(1)] - p[:, cl.index(-1)]
    if kind == "repair":
        return p[:, cl.index(1)]
    return p[:, cl.index(1)]  # binary: P(error)


def tau_of(ucf, act, keep, calls):
    return max(V5.tau_grid(ucf, act, keep, calls))[2]


def main():
    dev, test = V5.load("dev"), V5.load("test")
    ho = V5.load("test", datasets=V5.HELDOUT)
    for cs in (dev, test, ho):
        VM.prepare(cs)
        for c in cs:
            c.X = design(c)
            c.pr, c.pb = probs(c.X, BM["model"])
            c.delta = c.act - c.keep
            c.lab = V5.label(c, S)
    N = sum(c.n for c in test)
    first = 100 * sum(c.keep.sum() for c in test) / N
    main_acc = 100 * sum(VM.outcome(c, c.r5).sum() for c in test) / N
    assert abs(main_acc - 93.05) < 0.005, main_acc  # F-VFiveMainAcc
    ydev = np.concatenate([c.lab for c in dev])
    prior = np.array([(ydev == k).mean() for k in (-1, 0, 1)])
    fact("DevPairs", "development field-model pairs (the classifier's training set)", f"{len(ydev):,}")
    fact("DevRepairs", "development repairs of the routed action under the selected strategy", f"{(ydev == 1).sum():,}")
    fact("DevBreaks", "development breaks of the routed action under the selected strategy", f"{(ydev == -1).sum():,}")

    # ------------------------------------------------------------------ probs: the frozen model as probabilities
    for tag, cs, lab in (("", test, "core test"), ("HO", ho, "held-out test")):
        y = np.concatenate([c.lab for c in cs])
        pr, pb = np.concatenate([c.pr for c in cs]), np.concatenate([c.pb for c in cs])
        pn = 1 - pr - pb
        u, d = pr - pb, np.concatenate([c.delta for c in cs])
        keep = np.concatenate([c.keep for c in cs])
        sel = np.concatenate([c.r5 for c in cs]) > 0
        n = len(y)
        P = np.column_stack([pb, pn, pr])
        Y = np.column_stack([y == -1, y == 0, y == 1]).astype(float)
        ll = float(-np.mean(np.log(np.clip((P * Y).sum(1), 1e-12, 1))))
        llp = float(-np.mean(np.log(np.clip((prior * Y).sum(1), 1e-12, 1))))
        br = float(np.mean(((P - Y) ** 2).sum(1)))
        brp = float(np.mean(((prior - Y) ** 2).sum(1)))
        fact(f"{tag}Pairs", f"{lab}: field-model pairs", f"{n:,}")
        fact(f"{tag}Repairs", f"{lab}: repairs when every field is reflected on", f"{(y == 1).sum():,}")
        fact(f"{tag}Breaks", f"{lab}: breaks when every field is reflected on", f"{(y == -1).sum():,}")
        fact(f"{tag}LogLoss", f"{lab}: multiclass log loss of the frozen benefit model (lower is better)", f"{ll:.3f}")
        fact(f"{tag}LogLossPrior", f"{lab}: multiclass log loss of a prior that predicts the development outcome "
             "frequencies", f"{llp:.3f}")
        fact(f"{tag}Brier", f"{lab}: multiclass Brier score of the frozen benefit model (lower is better)", f"{br:.3f}")
        fact(f"{tag}BrierPrior", f"{lab}: multiclass Brier score of the development outcome frequencies", f"{brp:.3f}")
        fact(f"{tag}RepairAUC", f"{lab}: AUROC of P(repair) for a repair", f"{roc_auc_score(y == 1, pr):.2f}")
        fact(f"{tag}RepairECE", f"{lab}: ECE of P(repair) (10 equal-width bins), points", f"{100 * ece(pr, (y == 1).astype(float)):.2f}")
        fact(f"{tag}RepairMean", f"{lab}: mean P(repair), %", f"{100 * pr.mean():.2f}")
        fact(f"{tag}RepairRate", f"{lab}: observed repair rate, %", f"{100 * (y == 1).mean():.2f}")
        ok = keep == 1
        fact(f"{tag}BreakAUC", f"{lab}: AUROC of P(break) for a break, among first-pass correct answers",
             f"{roc_auc_score(y[ok] == -1, pb[ok]):.2f}")
        fact(f"{tag}BreakECE", f"{lab}: ECE of P(break) (10 equal-width bins), points", f"{100 * ece(pb, (y == -1).astype(float)):.2f}")
        fact(f"{tag}BreakMean", f"{lab}: mean P(break), %", f"{100 * pb.mean():.2f}")
        fact(f"{tag}BreakRate", f"{lab}: observed break rate, %", f"{100 * (y == -1).mean():.2f}")
        fact(f"{tag}UAUCRepair", f"{lab}: AUROC of u(x) for a repair", f"{roc_auc_score(y == 1, u):.2f}")
        fact(f"{tag}UAUCError", f"{lab}: AUROC of u(x) for a first-pass error", f"{roc_auc_score(keep == 0, u):.2f}")
        fact(f"{tag}UAUCRepairAmongErrors", f"{lab}: AUROC of u(x) for a repair among first-pass errors (which errors "
             "reflection repairs)", f"{roc_auc_score(y[keep == 0] == 1, u[keep == 0]):.2f}")
        fact(f"{tag}UCalErr", f"{lab}: calibration error of u(x) as the expected change in correctness, 10 equal-mass "
             "bins of u, points", f"{100 * ece_mass(u, d):.2f}")
        fact(f"{tag}PredSel", f"{lab}: on the fields Map-Reflect selects, predicted change in accuracy (sum of u over "
             "all fields), points", f"{100 * u[sel].sum() / n:+.2f}")
        fact(f"{tag}ObsSel", f"{lab}: on the fields Map-Reflect selects, observed change in accuracy (repairs minus "
             "breaks over all fields), points", f"{100 * d[sel].sum() / n:+.2f}")
        fact(f"{tag}PredAll", f"{lab}: on all fields, predicted change in accuracy (sum of u), points", f"{100 * u.sum() / n:+.2f}")
        fact(f"{tag}ObsAll", f"{lab}: on all fields, observed change in accuracy, points", f"{100 * d.sum() / n:+.2f}")
        fact(f"{tag}SelRate", f"{lab}: Map-Reflect's reflection rate, %", f"{100 * sel.mean():.1f}")
        # equal-mass reliability of u, for the record
        order = np.argsort(u, kind="stable")
        rel = [(float(u[ix].mean()), float(d[ix].mean()), int(len(ix))) for ix in np.array_split(order, 10)]
        fact(f"{tag}UReliability", f"{lab}: reliability of u(x), 10 equal-mass bins: (mean u, mean observed change, n)",
             "; ".join(f"({a:+.3f}, {b:+.3f}, {k})" for a, b, k in rel))
        print(tag, "ll", ll, llp, "brier", br, brp, "rel", rel)

    # ------------------------------------------------------------------ alts: fitted on development only
    keep_d = np.concatenate([c.keep for c in dev])
    act_d = np.concatenate([c.act for c in dev])
    calls_d = np.concatenate([c.act_calls for c in dev])
    Xd = np.vstack([c.X for c in dev])
    gid = groups_of(dev, V5.DATASETS, V5.ITOPS_DUP)
    # reproduce the frozen fit (C, tau) before using the same rule for the alternatives
    clf_u, C_u, cf_u = fit_cv(Xd, ydev, gid)
    tau_u = tau_of(cf_u[1] - cf_u[-1], act_d, keep_d, calls_d)
    assert C_u == BM["model"]["C"] and abs(tau_u - BM["tau"]) < 1e-9, (C_u, tau_u)
    alts = {}
    # P(repair) alone: the frozen model's P(repair); its own tau on the cross-fitted P(repair)
    alts["Repair"] = dict(name="P(repair) alone", C=C_u, tau=tau_of(cf_u[1], act_d, keep_d, calls_d),
                          prio={id(c): c.pr for c in test})
    # P(error): binary model of first-pass error on the same features
    clf_e, C_e, cf_e = fit_cv(Xd, (keep_d == 0).astype(int), gid, binary=True)
    alts["Error"] = dict(name="P(error), correctness estimate", C=C_e, tau=tau_of(cf_e[1], act_d, keep_d, calls_d),
                         prio={id(c): score_of(clf_e, c.X, "error") for c in test})
    # leave-one-model-out: fit on the other three models' development fields without model indicators
    lomo, lomo_tau, lomo_C = {}, {}, {}
    for m in MODELS:
        dv = [c for c in dev if c.model != m]
        Xo = np.vstack([design(c, True) for c in dv])
        yo = np.concatenate([c.lab for c in dv])
        clf_o, C_o, cf_o = fit_cv(Xo, yo, groups_of(dv, V5.DATASETS, V5.ITOPS_DUP))
        lomo_tau[m] = tau_of(cf_o[1] - cf_o[-1], np.concatenate([c.act for c in dv]), np.concatenate([c.keep for c in dv]),
                             np.concatenate([c.act_calls for c in dv]))
        lomo_C[m] = C_o
        for c in test:
            if c.model == m:
                lomo[id(c)] = score_of(clf_o, design(c, True), "u")
    alts["Transfer"] = dict(name="u(x) fitted on the other three models", C=lomo_C, tau=lomo_tau, prio=lomo)
    alts["Margin"] = dict(name="margin (frozen refitted t_m)", prio={id(c): c.mprio for c in test}, own={id(c): c.rm for c in test})
    for k, a in alts.items():
        if "own" not in a:
            if isinstance(a["tau"], dict):
                a["own"] = {id(c): (a["prio"][id(c)] > a["tau"][c.model]).astype(float) for c in test}
            else:
                a["own"] = {id(c): (a["prio"][id(c)] > a["tau"]).astype(float) for c in test}
    GB = GS.GroupBootstrap(test, GS.split_groups(V5.DATASETS, V5.ITOPS_DUP), B=2000, seed=V5.SEED)
    bud = {m: float(sum(np.sum(c.r5 * VM.real_cost(c, "calls")) for c in test if c.model == m)) for m in MODELS}
    R5 = {id(c): c.r5 for c in test}
    calls5, _ = VM.cost_of(test, R5)
    fact("MainCallsPer", "core test, Map-Reflect at its frozen tau: second-pass calls per field (standalone accounting)", f"{calls5:.3f}")
    for k, a in alts.items():
        W, MIS = VM.seq_route(test, a["prio"], "calls", bud, E)
        for mode, WW in (("Matched", W), ("Own", a["own"])):
            acc = 100 * sum(VM.outcome(c, WW[id(c)]).sum() for c in test) / N
            dl, dd = GB.pooled(test, lambda c: VM.outcome(c, WW[id(c)]) - VM.outcome(c, c.r5))
            lo, hi = GS.interval(dd)
            calls, _ = VM.cost_of(test, WW)
            desc = f"core test, {a['name']}, " + ("at Map-Reflect's calls (matched)" if mode == "Matched" else "at its own development threshold")
            fact(f"Alt{k}{mode}Acc", f"{desc}: accuracy, %", f"{acc:.2f}")
            fact(f"Alt{k}{mode}VsMain", f"{desc}: minus Map-Reflect at its frozen tau, points (paired, unadjusted)", sg(dl), sg(lo), sg(hi))
            fact(f"Alt{k}{mode}Calls", f"{desc}: second-pass calls per field", f"{calls:.3f}")
            print(k, mode, f"{acc:.2f}", sg(dl), sg(lo), sg(hi), f"{calls:.3f}")
        if "C" in a:
            fact(f"Alt{k}C", f"{a['name']}: L2 inverse strength C chosen by grouped development CV", json.dumps(a["C"]))
            fact(f"Alt{k}Tau", f"{a['name']}: own threshold fitted on cross-fitted development scores",
                 json.dumps({m: round(v, 4) for m, v in a["tau"].items()}) if isinstance(a["tau"], dict) else f"{a['tau']:.4f}")
    # u's error-detection AUROC beside the margin's, same core test fields (all field-model pairs)
    keep_t = np.concatenate([c.keep for c in test])
    fact("MarginAUCError", "core test, all field-model pairs: AUROC of the margin ordering (declined first, then ascending "
         "margin) for a first-pass error", f"{roc_auc_score(keep_t == 0, np.concatenate([c.mprio for c in test])):.2f}")
    fact("ErrorAUCError", "core test: AUROC of the P(error) model for a first-pass error",
         f"{roc_auc_score(keep_t == 0, np.concatenate([alts['Error']['prio'][id(c)] for c in test])):.2f}")
    fact("ErrorAUCRepair", "core test: AUROC of the P(error) model for a repair",
         f"{roc_auc_score(np.concatenate([c.lab for c in test]) == 1, np.concatenate([alts['Error']['prio'][id(c)] for c in test])):.2f}")

    # ------------------------------------------------------------------ tau sensitivity
    ucf = cf_u[1] - cf_u[-1]
    rows = []
    for t in (-0.02, 0.0, BM["tau"], 0.02, 0.05, 0.1, 0.2):
        rd = ucf > t
        dacc = 100 * np.where(rd, act_d, keep_d).mean()
        W = {id(c): (c.u > t).astype(float) for c in test}
        acc = 100 * sum(VM.outcome(c, W[id(c)]).sum() for c in test) / N
        calls, _ = VM.cost_of(test, W)
        rate = 100 * sum(W[id(c)].sum() for c in test) / N
        rows.append((t, dacc, acc, calls, rate))
    fact("TauGrid", "frozen u(x) at other thresholds: (tau, cross-fitted development accuracy %, test accuracy %, test "
         "calls per field, test reflection rate %)", "; ".join(f"({t:.4f}, {a:.2f}, {b:.2f}, {c:.3f}, {r:.1f})" for t, a, b, c, r in rows))
    accs = [r[2] for r in rows]
    fact("TauTestMin", "core test accuracy over tau from -0.02 to 0.2 (frozen u), lowest, %", f"{min(accs):.2f}")
    fact("TauTestMax", "core test accuracy over tau from -0.02 to 0.2 (frozen u), highest, %", f"{max(accs):.2f}")
    tz = [r for r in rows if r[0] == 0.0][0]
    t2 = [r for r in rows if r[0] == 0.2][0]
    fact("TauZeroAcc", "core test accuracy with tau = 0 (reflect when the expected change is positive), %", f"{tz[2]:.2f}")
    fact("TauZeroCalls", "core test calls per field with tau = 0", f"{tz[3]:.2f}")
    fact("TauHighValue", "the largest threshold of the tau grid", f"{t2[0]:.1f}")
    fact("TauHighAcc", "core test accuracy with tau = 0.2, %", f"{t2[2]:.2f}")
    fact("TauHighCalls", "core test calls per field with tau = 0.2", f"{t2[3]:.2f}")
    print(rows)
    FACTS_F.write_text(json.dumps(dict(generated_by="scripts/naacl/v5_benefit_uq.py", status="exploratory", definition=DEF,
                                       first_accuracy=first, facts=FACTS), indent=1) + "\n")
    print("wrote", FACTS_F, len(FACTS))


if __name__ == "__main__":
    main()
