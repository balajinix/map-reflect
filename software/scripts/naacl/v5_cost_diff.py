#!/usr/bin/env python3
"""The offline difference between v5's two cost accountings (scripts/naacl/v5.py standalone_costs). No model call.

  collection  what the collector spent: a widen outcome copied from NT (a fully shown type) costs nothing extra. The
              frozen v5 fit (tau's tie-break) and the scored v5 test numbers use it.
  standalone  what a deployed strategy pays: a copied widen outcome is charged the NT action's calls and tokens.

The script shows, from the committed outputs and the stored rows only:
  1. development: the copied widen outcomes, how many are the routed action, and every strategy's calls per field
     under both accountings;
  2. tau: rerunning the frozen fit's benefit model and cross-fitted u (v5.benefit_fit) and its tau grid (v5.tau_grid)
     under collection costs gives the frozen tau, and under standalone costs gives the same tau;
  3. test: the same counts and calls per field for every scored policy; accuracies do not depend on the accounting,
     and the committed v5_facts.json calls per field equal the collection values recomputed here.
Writes docs/naacl/results/v5_cost_diff.json and v5_cost_diff.md. Usage:
  ITOPS_V21=<v2.1 label file> .venv/bin/python scripts/naacl/v5_cost_diff.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
sys.argv = ["v5.py", "--stage", "score"]
import v5 as V5  # noqa: E402

FZ = json.loads((ROOT / "docs/naacl/results/v5_frozen.json").read_text())
S = FZ["intervention"]["outcome_field"]
OUT_J = ROOT / "docs/naacl/results/v5_cost_diff.json"
OUT_M = ROOT / "docs/naacl/results/v5_cost_diff.md"


def counts(cells):
    n = sum(c.n for c in cells)
    copied = int(sum(c.reuse.sum() for c in cells))
    routed_w = int(sum((c.reuse & (c.isnt == 0)).sum() for c in cells))
    return dict(fields=n, copied_widen=copied, copied_widen_routed=routed_w)


def per_field(cells, w, which):
    """Calls per field of a policy that routes weight w (per cell) to the routed action (act) or to widen (W)."""
    n = sum(c.n for c in cells)
    if which == "act":
        col = sum(np.nansum(w[id(c)] * c.act_calls) for c in cells) / n
        sta = sum(np.nansum(w[id(c)] * c.Ac) for c in cells) / n
    else:
        col = sum(np.nansum(w[id(c)] * c.W_calls) for c in cells) / n
        sta = sum(np.nansum(w[id(c)] * c.Wc) for c in cells) / n
    return round(float(col), 4), round(float(sta), 4)


def policies(cells):
    bm = FZ["c_benefit"]
    mu, sd = np.array(bm["standardise"]["mu"]), np.array(bm["standardise"]["sd"])
    out = {}
    w5, wm, w41, ws, we = {}, {}, {}, {}, {}
    for c in cells:
        M_, D_ = V5.feats(c)
        w5[id(c)] = (V5.u_of(np.hstack([(M_ - mu) / sd, D_]), bm["model"]) > bm["tau"]).astype(float)
        mr = FZ["margin_router"][c.model]
        wm[id(c)] = V5.route_margin(c, mr["t_m"], mr["t_s"], S)[1]
        w41[id(c)] = V5.route_margin(c, V5.TH41[c.model]["t_m"], V5.TH41[c.model]["t_s"], S)[1]
        ws[id(c)] = c.struct
        we[id(c)] = np.ones(c.n)
    out["v5 (benefit router)"] = per_field(cells, w5, "act")
    out["margin router (t_m refitted)"] = per_field(cells, wm, "act")
    out["margin router (v4.1 thresholds)"] = per_field(cells, w41, "act")
    out["struct (widen action)"] = per_field(cells, ws, "W")
    out["every field (routed action)"] = per_field(cells, we, "act")
    return out


def main():
    dev = V5.load("dev")
    test = V5.load("test")
    ho = V5.load("test", datasets=V5.HELDOUT)
    res = dict(source="scripts/naacl/v5_cost_diff.py; v5.standalone_costs; no model call",
               frozen="docs/naacl/results/v5_frozen.json", outcome_field=S)
    res["development"] = dict(counts(dev), calls_per_field=policies(dev))
    # tau under both accountings, from the frozen fit's own procedure
    model, ucf, mu, sd, C, best, y = V5.benefit_fit(dev, S)
    keep = np.concatenate([c.keep for c in dev])
    act = np.concatenate([getattr(c, f"act_{S}") for c in dev])
    g_col = max(V5.tau_grid(ucf, act, keep, np.concatenate([c.act_calls for c in dev])))
    g_sta = max(V5.tau_grid(ucf, act, keep, np.concatenate([c.Ac for c in dev])))
    ft = FZ["c_benefit"]["tau"]
    res["tau"] = dict(frozen=ft, refit_collection=g_col[2], refit_standalone=g_sta[2], C_frozen=FZ["c_benefit"]["model"]["C"],
                      C_refit=C, dev_accuracy_collection=g_col[0], dev_accuracy_standalone=g_sta[0],
                      unchanged=bool(g_col[2] == ft and g_sta[2] == ft and C == FZ["c_benefit"]["model"]["C"]))
    res["core_test"] = dict(counts(test), calls_per_field=policies(test))
    res["heldout_test"] = dict(counts(ho), calls_per_field=policies(ho))
    facts = {f["id"]: f["value"] for f in json.loads((ROOT / "docs/naacl/results/v5_facts.json").read_text())["facts"]}
    chk = {"F-VFiveMainCalls": "v5 (benefit router)", "F-VFiveMarginCalls": "margin router (t_m refitted)",
           "F-VFiveMarginOldCalls": "margin router (v4.1 thresholds)", "F-VFiveStructCalls": "struct (widen action)",
           "F-VFiveEveryCalls": "every field (routed action)"}
    res["scored_test_unchanged"] = {k: dict(fact=facts.get(k), recomputed_collection=res["core_test"]["calls_per_field"][v][0],
                                            agrees=facts.get(k) == f"{res['core_test']['calls_per_field'][v][0]:.2f}")
                                    for k, v in chk.items()}
    assert res["tau"]["unchanged"], res["tau"]
    assert all(v["agrees"] for v in res["scored_test_unchanged"].values()), res["scored_test_unchanged"]
    OUT_J.write_text(json.dumps(res, indent=1) + "\n")
    L = ["# v5 cost accounting: collection against standalone (offline)", "",
         "Collection costs (what the collector spent; a widen outcome copied from NT costs nothing extra) are what the "
         "frozen fit and the scored test use. Standalone costs charge a copied widen outcome the NT action's calls, as a "
         "deployed strategy would pay (v5.standalone_costs). No model call; nothing frozen or scored is rewritten.", "",
         f"Frozen tau {ft:.6f}. Refit under collection costs: {g_col[2]:.6f}; under standalone costs: {g_sta[2]:.6f} "
         f"(C {C}, frozen C {FZ['c_benefit']['model']['C']}). Unchanged: {res['tau']['unchanged']}.", ""]
    for name, key in (("Development", "development"), ("Core test", "core_test"), ("Held-out test", "heldout_test")):
        r = res[key]
        L += [f"## {name}", "", f"{r['fields']:,} field-model pairs; {r['copied_widen']:,} copied widen outcomes, "
              f"{r['copied_widen_routed']:,} of them where widen is the routed action.", "",
              "| strategy | calls per field, collection | calls per field, standalone |", "|---|--:|--:|"]
        for p, (a, b) in r["calls_per_field"].items():
            L.append(f"| {p} | {a:.4f} | {b:.4f} |")
        L.append("")
    L += ["## Scored test facts", "", "| fact | committed | recomputed (collection) |", "|---|--:|--:|"]
    for k, v in res["scored_test_unchanged"].items():
        L.append(f"| {k} | {v['fact']} | {v['recomputed_collection']:.4f} |")
    OUT_M.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
