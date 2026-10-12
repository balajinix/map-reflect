#!/usr/bin/env python3
"""Budget-matched gate comparison (the fair Table 1).

For each dataset, gate the Granite->Llama escalation at a fixed budget (default
30% of fields) by each signal and compare downstream top-1 field accuracy:

  UQ           : ranker margin (gamma1 - gamma2), low margin = escalate
  error        : 1 - gamma1 (low top score = escalate)
  neighborhood : sibling-field target-table incoherence (a field whose predicted
                 target table disagrees with its source-table siblings is suspect)
  random       : random budget (mean over seeds)

All signals spend the SAME budget, so this isolates gate QUALITY from selectivity.
Reads campaign slots <base>/<ds>/esc_base__granite and esc_base__llama.

Usage: uq_gate_budget.py [results/campaign] [--budget=0.30]
"""
import sys, os, json, glob
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from eval.calc_accuracy import fields_match
except Exception:
    sys.path.insert(0, str(ROOT))
    from eval.calc_accuracy import fields_match


def _f(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def pred_table(pid):
    if "__" in pid: return pid.rsplit("__", 1)[0]
    if "." in pid: return pid.rsplit(".", 1)[0]
    return pid


def load_slot(slot):
    # The runner's final output first, a checkpoint only when nothing else exists. glob order is the file system's,
    # and a checkpoint lags the final output (it misses the last samples and any in-place re-scoring).
    fs = sorted(glob.glob(os.path.join(slot, "*.json")), key=lambda f: (os.path.basename(f).startswith("checkpoint_"), f))
    for f in fs:
        try:
            d = json.load(open(f))
            if isinstance(d, dict) and "results" in d:
                return d
        except Exception:
            continue
    return None


def index_run(d):
    """key -> {pred, correct, margin, g1, table, skey}. Also per-sample field lists."""
    out = {}
    def _id(x): return x.get("id", "") if isinstance(x, dict) else str(x or "")
    for r in d.get("results", []):
        skey = f"{_id(r.get('expected_source'))}|{_id(r.get('expected_target'))}|{r.get('sample_id','')}"
        gt = {str(k).lower(): (v if isinstance(v, list) else [v])
              for k, v in (r.get("ground_truth_mapping", {}) or {}).items()}
        for fk, fv in (r.get("predictions", {}) or {}).items():
            if not isinstance(fv, dict) or fk.lower() not in gt:
                continue
            ir = fv.get("initial_ranking") or []
            if not ir:
                continue
            pred = str(ir[0].get("id", ""))
            gts = [str(g) for g in gt[fk.lower()] if g]
            correct = int(any(fields_match(g, pred) for g in gts))
            sc = [_f(c.get("score")) for c in ir]
            g1 = sc[0] if sc and sc[0] is not None else 0.0
            g2 = sc[1] if len(sc) > 1 and sc[1] is not None else g1
            out[(skey, fk.lower())] = dict(pred=pred, correct=correct,
                                           margin=g1 - g2, g1=g1,
                                           table=pred_table(pred), skey=skey)
    return out


def neighborhood_incoherence(base, keys):
    """Per key: 1 - (fraction of same-sample fields predicting the same target table)."""
    by_sample = {}
    for k in keys:
        by_sample.setdefault(base[k]["skey"], []).append(k)
    inc = {}
    for skey, ks in by_sample.items():
        tabs = [base[k]["table"] for k in ks]
        n = len(tabs)
        for k in ks:
            same = sum(1 for t in tabs if t == base[k]["table"])
            inc[k] = 1.0 - (same / n if n else 1.0)
    return inc


def escalate_acc(bc, ec, suspicion, k, descending=True):
    order = np.argsort(-suspicion if descending else suspicion)
    idx = order[:k]
    acc = bc.copy(); acc[idx] = ec[idx]
    return acc.mean()


def escalate_acc_tb(bc, ec, suspicion, k, descending=True, n_shuffles=50):
    """escalate_acc as (mean, sd) over randomized tie-breaking.

    The verbalized scores are quantized, so a large share of fields share a suspicion
    value and which of them gets escalated is arbitrary. Permuting before a stable sort
    randomizes that choice; the sd says how much of the reported gain is tie-break luck.
    """
    s = -suspicion if descending else suspicion
    n = len(s)
    accs = []
    for seed in range(n_shuffles):
        perm = np.random.RandomState(seed).permutation(n)
        idx = perm[np.argsort(s[perm], kind="stable")][:k]
        acc = bc.copy(); acc[idx] = ec[idx]; accs.append(acc.mean())
    return float(np.mean(accs)), float(np.std(accs))


def eval_dataset(gpath, lpath, budget):
    base = index_run(json.load(open(gpath)))
    esc = index_run(json.load(open(lpath)))
    keys = sorted(set(base) & set(esc))
    if len(keys) < 10:
        return None
    bc = np.array([base[k]["correct"] for k in keys], float)
    ec = np.array([esc[k]["correct"] for k in keys], float)
    margin = np.array([base[k]["margin"] for k in keys], float)
    g1 = np.array([base[k]["g1"] for k in keys], float)
    inc = neighborhood_incoherence(base, keys)
    nbr = np.array([inc[k] for k in keys], float)
    n = len(keys); k = max(1, int(round(budget * n)))

    res = {"n": n, "base": bc.mean(), "esc_all": ec.mean()}
    res["uq"] = escalate_acc(bc, ec, -margin, k)       # low margin = suspect
    res["error"] = escalate_acc(bc, ec, 1 - g1, k)     # low g1 = suspect
    res["nbr"] = escalate_acc(bc, ec, nbr, k)          # high incoherence = suspect
    # tie-break error bars (the deltas above use one arbitrary tie order)
    res["uq_tb"], res["uq_sd"] = escalate_acc_tb(bc, ec, -margin, k)
    res["error_tb"], res["error_sd"] = escalate_acc_tb(bc, ec, 1 - g1, k)
    res["nbr_tb"], res["nbr_sd"] = escalate_acc_tb(bc, ec, nbr, k)
    rand = []
    for s in range(20):
        r = np.random.RandomState(s).permutation(n)[:k]
        rc = bc.copy(); rc[r] = ec[r]; rand.append(rc.mean())
    res["random"] = float(np.mean(rand))
    return res


def main():
    base_dir = "results/campaign"
    budget = 0.30
    for a in sys.argv[1:]:
        if a.startswith("--budget="): budget = float(a.split("=", 1)[1])
        elif not a.startswith("--"): base_dir = a
    base_dir = base_dir if os.path.isabs(base_dir) else str(ROOT / base_dir)

    ds_dirs = sorted(d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d))
    print(f"\n=== BUDGET-MATCHED GATE COMPARISON (escalate {budget:.0%} of fields, Granite->Llama) ===")
    hdr = (f"{'dataset':18s} {'n':>5s} {'base':>6s} {'esc-all':>7s} | "
           f"{'UQ':>6s} {'error':>6s} {'nbr':>6s} {'random':>6s} | {'UQ-rand':>7s} {'best':>6s}")
    print(hdr); print("-" * len(hdr))
    wins = 0; total = 0
    agg = {g: [] for g in ["uq", "error", "nbr", "random"]}
    tb = []
    for dd in ds_dirs:
        ds = os.path.basename(dd)
        g = os.path.join(dd, "esc_base__granite"); l = os.path.join(dd, "esc_base__llama")
        if not (load_slot(g) and load_slot(l)):
            continue
        gf = glob.glob(os.path.join(g, "*.json")); lf = glob.glob(os.path.join(l, "*.json"))
        r = eval_dataset(gf[0], lf[0], budget)
        if not r:
            continue
        gates = {k: r[k] for k in ["uq", "error", "nbr", "random"]}
        best = max(gates, key=gates.get)
        uq_rand = 100 * (r["uq"] - r["random"])
        wins += int(r["uq"] >= max(r["error"], r["nbr"], r["random"]) - 1e-9); total += 1
        for k in agg: agg[k].append(r[k] - r["base"])
        tb.append((ds, r))
        print(f"{ds:18s} {r['n']:5d} {r['base']:6.3f} {r['esc_all']:7.3f} | "
              f"{r['uq']:6.3f} {r['error']:6.3f} {r['nbr']:6.3f} {r['random']:6.3f} | "
              f"{uq_rand:+7.1f} {best:>6s}")
    print("-" * len(hdr))
    means = {k: 100 * np.mean(v) for k, v in agg.items() if v}
    if means:
        print(f"{'mean Δpp vs base':18s} {'':5s} {'':6s} {'':7s} | "
              f"{means['uq']:+6.1f} {means['error']:+6.1f} {means['nbr']:+6.1f} {means['random']:+6.1f}")
    print(f"\nUQ is best gate on {wins}/{total} datasets. All gates spend the same {budget:.0%} budget.")

    # Tie-break sensitivity: the table above breaks ties by key order, which is arbitrary.
    print(f"\n=== TIE-BREAK SENSITIVITY (mean Δpp ± sd over 50 randomized tie orders) ===")
    h2 = f"{'dataset':18s} {'UQ':>14s} {'error':>14s} {'nbr':>14s}"
    print(h2); print("-" * len(h2))
    for ds, r in tb:
        cells = ""
        for g in ("uq", "error", "nbr"):
            cells += f"  {100*(r[g+'_tb']-r['base']):+6.1f}±{100*r[g+'_sd']:4.2f}"
        print(f"{ds:18s}{cells}")
    if tb:
        for g in ("uq", "error", "nbr"):
            m = 100 * np.mean([r[g + "_tb"] - r["base"] for _, r in tb])
            s = 100 * np.mean([r[g + "_sd"] for _, r in tb])
            print(f"  mean {g:<6s} {m:+6.1f}  (mean per-dataset sd {s:4.2f})")


if __name__ == "__main__":
    main()
