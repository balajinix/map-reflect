"""Phase 2 of the v6 numbers (docs/naacl/v6/PLAN_PHASE2.md): the offline gap filling (R2), the offline UQ ablations
(U1 to U6, U8, U10) and the per-panel data of the chart pages (C1 to C13, C15). No model call.

Called by scripts/naacl/v6_numbers.py after its own analyses, with its globals (cells, the fact registry, the split-
group bootstrap, the v4 rows) and the population the UQ run used (uq_ablation_v4.run's locals, captured at return).
Every result is exploratory. Per-panel numbers go to paper/figures/data6/*.csv, next to the figures that
scripts/naacl/v6_charts.py draws from them.

Not offline, so listed as needing runs: U7 (the five answer samples per field were not stored, only their entropy and
distinct count), U9 (answer entropy for Granite-4 and Llama-4, token log-probabilities for Llama-4), and the held-out
AUROC of the answer-token log-probability and answer entropy (no held-out outputs).
"""
import csv
import re
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

E = "exploratory"
WORD = {10: "Ten", 20: "Twenty", 25: "TwentyFive", 30: "Thirty", 50: "Fifty", 100: "Hundred"}
NEEDS_RUNS = []  # (item, what is missing, estimated calls)


def setup(G):
    g = globals()
    for k, v in G.items():
        if not k.startswith("__") and k not in g:
            g[k] = v
    U = G["UQCAP"]["U"]
    # With the U9 rows (answer entropy for Granite-4 and Llama-4, log-probabilities for Llama-4) the UQ population is
    # the same fields in the same order, so every other signal and the bootstrap draws are unchanged and the two signals
    # cover all four models; v6_numbers checks the population equality before this switch.
    L9 = G.get("UQCAP9") or {}
    same = bool(L9) and all([(f["ds"], f["sig"]["_key"], f["base"], f["out"]) for f in L9["pop"][m]]
                            == [(f["ds"], f["sig"]["_key"], f["base"], f["out"]) for f in G["UQCAP"]["pop"][m]] for m in U.MODELS)
    g.update(U=U, L=L9 if same else G["UQCAP"], L0=G["UQCAP"], DATA=G["ROOT"] / "paper/figures/data6", U9ON=same)
    DATA.mkdir(parents=True, exist_ok=True)


def wcsv(name, header, rows):
    with open(DATA / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([(f"{x:.6g}" if isinstance(x, float) else x) for x in r])


V5SRC = "results/naacl/v6/v5 (stored outputs); docs/naacl/results/v5_facts.json, docs/naacl/results/v5_test.md"
V5FID = re.compile(r"VFive|PD(GptOss)?(HO)?(Main|Every)(?!PreFF)")  # method v5's numbers: its selection and all fields


def F(fid, claim, value, lo=None, hi=None, fn="v6_phase2", src=None, status=E):
    # default source: the stored v4 outputs, except method v5's numbers (rebuilt from results/naacl/v6/v5 by v5_setup)
    if src is None and V5 and V5FID.match(fid):
        src = (V5SRC, fn)
    fact(fid, claim, value, src or ("results/naacl/v4 (stored outputs)", fn), status, lo, hi, func=fn)


SIGORDER = ["margin", "s1", "ratio", "entropy", "lp_answer", "verbalised", "p_true", "sem_entropy", "combo_paid"]


def uqmods(s):
    return [m for m in U.MODELS if s in L["avail"][m]]


def mname(ms):
    return ", ".join(MSHORT[m] for m in ms)


def pooled_rank(models, s, filt=None):
    """Within-model percentile ranks of signal s (as uq_ablation_v4's pooled AUROC) and the errors, over models."""
    cs, err, fs = [], [], []
    for m in models:
        f_ = [f for f in L["pop"][m] if filt is None or filt(f)]
        cs.append(U.pct_rank(np.array([f["sig"][s] for f in f_], float)))
        err.append(1 - np.array([f["base"] for f in f_], float))
        fs += f_
    return np.concatenate(cs), np.concatenate(err), fs


def auc_ci(models, s, filt=None):
    c, e, fs = pooled_rank(models, s, filt)
    a = U.auroc_w(c, e)
    W = L["field_w"](fs)
    lo, hi = L["ci"](U.auroc_draws(c, e, W))
    return a, lo, hi


def share_of_oracle(models, s):
    res, n = L["group"](models, ["keep", "oracle", "margin", s])
    k0, o0 = res["keep"][0], res["oracle"][0]
    return 100 * (res[s][0] - k0) / (o0 - k0), 100 * (res["margin"][0] - k0) / (o0 - k0), res


# ======================================================================================= method v5 (phase 5)
V5 = {}


def v5_setup():
    """Method v5 on test, rebuilt in memory exactly as scripts/naacl/v5.py --stage score builds it (its load, features,
    benefit model and frozen thresholds, imported, not rewritten), with every policy's per-field outcome, calls and
    tokens, for Table 1, the router table and the charts. Asserted against docs/naacl/results/v5_facts.json."""
    fzf = ROOT / "docs/naacl/results/v5_frozen.json"
    if V5 or not fzf.exists() or not (ROOT / "results/naacl/v6/v5/test").exists() or VARIANT != "v4_1":
        return
    argv, sys.argv = sys.argv, ["v5.py", "--stage", "score"]
    import v5 as V5m
    sys.argv = argv
    fz = json.loads(fzf.read_text())
    s_ = fz["intervention"]["outcome_field"]
    bm = fz["c_benefit"]
    mu, sd = np.array(bm["standardise"]["mu"]), np.array(bm["standardise"]["sd"])
    core = V5m.load("test")
    V5m.v41_reference(core)
    ho = V5m.load("test", datasets=V5m.HELDOUT)
    V5m.v41_reference(ho)
    for c in core + ho:
        M_, D_ = V5m.feats(c)
        c.u = V5m.u_of(np.hstack([(M_ - mu) / sd, D_]), bm["model"])
        r5 = (c.u > bm["tau"]).astype(float)
        act, cc, tk = getattr(c, f"act_{s_}"), c.act_calls, c.act_tok
        wok, wc, wt = getattr(c, f"W_{s_}"), c.W_calls, c.W_tok
        mr = fz["margin_router"][c.model]
        _, rm = V5m.route_margin(c, mr["t_m"], mr["t_s"], s_)
        _, r41 = V5m.route_margin(c, V5m.TH41[c.model]["t_m"], V5m.TH41[c.model]["t_s"], s_)
        st = c.struct
        c.act5, c.r5 = act, r5
        nanv = np.full(c.n, np.nan)
        c.pol = {
            "keep": (c.keep, np.zeros(c.n), np.zeros(c.n), np.zeros(c.n)),
            "v5": (np.where(r5 > 0, act, c.keep), r5 * cc, r5 * tk, r5),
            "margin": (np.where(rm > 0, act, c.keep), rm * cc, rm * tk, rm),
            "margin41": (np.where(r41 > 0, act, c.keep), r41 * cc, r41 * tk, r41),
            "struct": (np.where(st > 0, wok, c.keep), st * wc, st * wt, st),
            "union": (np.where(r5 > 0, act, np.where(st > 0, wok, c.keep)), r5 * cc + (1 - r5) * st * wc,
                      r5 * tk + (1 - r5) * st * wt, np.maximum(r5, st)),
            "every": (act, cc, tk, np.ones(c.n)),
            "v41": (c.v41, nanv, nanv, nanv)}
    for group in (core, ho):  # random and oracle at v5's routed share per model, within each population
        for m in V5m.MODELS:
            cs = [c for c in group if c.model == m]
            if not cs:
                continue
            share = sum(c.r5.sum() for c in cs) / sum(c.n for c in cs)
            tot = sum(np.where(c.keep == 0, c.act5, 0.0).sum() for c in cs)
            k = share * sum(c.n for c in cs)
            for c in cs:
                c.pol["random"] = (c.keep + share * (c.act5 - c.keep), share * c.act_calls, share * c.act_tok, np.full(c.n, share))
                w = np.where(c.keep == 0, c.act5, 0.0) * (min(1.0, k / tot) if tot else 1.0)
                c.pol["oracle"] = (c.keep + w, w * c.act_calls, w * c.act_tok, w)
    GB5 = GS.GroupBootstrap(core, GS.split_groups(V5m.DATASETS, V5m.ITOPS_DUP), B=2000, seed=V5m.SEED)
    GH5 = GS.GroupBootstrap(ho, GS.split_groups(V5m.HELDOUT, None), B=2000, seed=V5m.SEED)
    V5.update(core=core, ho=ho, GB=GB5, GH=GH5, fz=fz, mod=V5m)
    # the rebuild reproduces the scored v5 facts
    jf = {x["id"][2:]: x["value"] for x in json.loads((ROOT / "docs/naacl/results/v5_facts.json").read_text())["facts"]}
    N5 = sum(c.n for c in core)
    for k, tg in (("keep", "Keep"), ("v5", "Main"), ("struct", "Struct"), ("every", "Every"), ("margin", "Margin"),
                  ("union", "Union"), ("random", "Random"), ("oracle", "Oracle")):
        a = 100 * sum(c.pol[k][0].sum() for c in core) / N5
        assert f"{a:.2f}" == jf[f"VFive{tg}Acc"], (k, a, jf[f"VFive{tg}Acc"])


def v5_policy(cs, k):
    """(accuracy %, calls per field, tokens per field) of policy k over cells cs."""
    n = sum(c.n for c in cs)
    return (100 * sum(c.pol[k][0].sum() for c in cs) / n, sum(np.nansum(c.pol[k][1]) for c in cs) / n,
            sum(np.nansum(c.pol[k][2]) for c in cs) / n)


def v5_router_table(own=None):
    """The router table on method v5: every policy uses the same v5 second pass; values are the F-VFive* facts
    (v5.py --stage score), the oracle's calls and tokens from the in-memory rebuild."""
    jf = {x["id"][2:]: x for x in json.loads((ROOT / "docs/naacl/results/v5_facts.json").read_text())["facts"]}
    _, ocl, otk = v5_policy(V5["core"], "oracle")
    ROWS5 = [("No reflection", "Keep"), ("Random selection", "Random"), ("Schema-size baseline", "Struct"),
             ("Margin, fixed thresholds", "MarginOld"), (r"\textbf{Map-Reflect}", "Main"),
             ("Benefit or schema size", "Union"), (r"Margin, $t_m$ refit", "Margin"),
             "Reference", ("Reflect on all", "Every"), ("Oracle", "Oracle"), "Margin-Reflect configuration"]
    body = []
    for r in ROWS5:  # no spacer rows between groups (paper/STYLE.md section 8); a string is an italic group header
        if isinstance(r, str):
            body.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{r}}}}} \\")
            if r == "Margin-Reflect configuration" and own is not None:  # Margin-Reflect, its own second pass, same 13,384 pairs
                a, e, lo, hi, _en, _em, cl_, tk_ = own
                body.append(f"Margin-Reflect & {a:.1f} & {tex_s(e)} {tci(lo, hi, 1)} & {cl_:.2f} & {tnum(tk_)} \\\\")
            continue
        lab, tg = r
        acc = float(jf[f"VFive{tg}Acc"]["value"])
        if tg == "Keep":
            body.append(f"{lab} & {acc:.1f} & ref. & 0.00 & 0 \\\\")
            continue
        g = jf[f"VFive{tg}Gain"]
        cl = f"{ocl:.2f}" if tg == "Oracle" else jf[f"VFive{tg}Calls"]["value"]
        tk = tnum(otk) if tg == "Oracle" else jf[f"VFive{tg}Tokens"]["value"].replace(",", "{,}")
        body.append(f"{lab} & {acc:.1f} & {tex_s(float(g['value']))} {tci(float(g['lo']), float(g['hi']), 1)} & {cl} & {tk} \\\\")
    # the main table carries no Margin-Reflect row; the ablation section's copy adds it (tab6_main_mr, DECISIONS row 57)
    cut = next((i for i, ln in enumerate(body) if "Margin-Reflect configuration" in ln), len(body))
    write_tab("main", "lrrrr", r"Selection & Acc. & Change [95\% CI] & Calls & Tokens \\", body[:cut],
              note_="Method v5: every router uses the same v5 second pass; source docs/naacl/results/v5_facts.json "
              "(v5.py --stage score), oracle cost from v6_phase2.v5_setup.")
    if cut < len(body):
        TABLE_ENV["main_mr"] = "table*"
        write_tab("main_mr", "lrrrr", r"Selection & Acc. & Change [95\% CI] & Calls & Tokens \\", body,
                  note_="The main router table with the Margin-Reflect configuration (its own second pass, "
                  "the same 13,384 pairs) as its last row group.", label="main_mr")


def v5_curves(cs, steps=21):
    """Random (share swept), margin threshold swept (around the refit t_m) and benefit threshold swept (over u), each
    with the v5 second pass: lists of (calls per field, accuracy %)."""
    n = sum(c.n for c in cs)
    f0 = 100 * sum(c.keep.sum() for c in cs) / n
    ev = sum(float(np.sum(c.act_calls)) for c in cs) / n
    dv = 100 * sum(float(np.sum(c.act5 - c.keep)) for c in cs) / n
    rnd = [(s_ * ev, f0 + s_ * dv) for s_ in np.linspace(0, 1, steps)]
    fz = V5["fz"]
    msw = []
    for dl in np.round(np.arange(-0.5, 0.55, 0.05), 2):
        a = cl = 0.0
        for c in cs:
            mr = fz["margin_router"][c.model]
            rt = router.routed(c.margin, c.validb, c.s1, max(0.0, mr["t_m"] + dl), mr["t_s"])
            a += float(np.sum(np.where(rt, c.act5, c.keep)))
            cl += float(np.sum(rt * c.act_calls))
        msw.append((cl / n, 100 * a / n))
    allu = np.concatenate([c.u for c in cs])
    bsw = []
    for q in np.linspace(0, 1, 41):
        tau = np.quantile(allu, q) if q < 1 else allu.max() + 1
        a = sum(float(np.sum(np.where(c.u > tau, c.act5, c.keep))) for c in cs)
        cl = sum(float(np.sum((c.u > tau) * c.act_calls)) for c in cs)
        bsw.append((cl / n, 100 * a / n))
    return rnd, msw, sorted(bsw)


def v5_chart_data():
    """C3, C4 and C13 on method v5 (every router with the v5 second pass), replacing their v4.1 CSVs."""
    core, ho = V5["core"], V5["ho"]
    rows = []
    for m in MODELS:
        cs = [c for c in core if c.model == m]
        rnd, msw, bsw = v5_curves(cs)
        rows += [(m, "random", x, y) for x, y in rnd] + [(m, "margin sweep", x, y) for x, y in msw]
        rows += [(m, "benefit sweep", x, y) for x, y in bsw]
        for k, lab in (("struct", "struct"), ("every", "every"), ("v5", "Map-Reflect"), ("keep", "keep")):
            acc, cl, _ = v5_policy(cs, k)
            rows.append((m, lab, cl, acc))
    wcsv("c3_accuracy_calls", ["model", "series", "calls_per_field", "accuracy"], rows)
    GB5 = V5["GB"]
    rows = []
    for fam, dss in FAMILY:
        rs = [c for c in core if c.ds in dss]
        for k, nm in (("v5", "Map-Reflect"), ("random", "random"), ("struct", "struct"), ("every", "every"), ("oracle", "oracle")):
            e, d = GB5.pooled(rs, lambda c, k=k: c.pol[k][0] - c.keep)
            lo, hi = GS.interval(d)
            rows.append((fam, nm, e, lo, hi))
            if k == "v5":  # the family gains of Figure chartsB(h), cited in the text on the headroom per family
                F(f"VFiveFam{FT[fam]}Gain", f"test, method v5 (benefit router), {fam} family: change over the first "
                  "pass, field-pooled within the family, points (C4)", sg(e), sg(lo), sg(hi), fn="v6_phase2.v5_chart_data")
    wcsv("c4_family_gain", ["family", "router", "change", "lo", "hi"], rows)
    GH5 = V5["GH"]
    TAGS = {"mimic_omop_rematch": "Mimic", "semtab_m2kg_r1": "SemTab", "oaei_conference": "OAEI",
            "bioml_omim_ordo": "BioML", "schemanet_cprd": "SchemaNet"}
    rows = []
    for ds, tg in sorted(TAGS.items(), key=lambda t: t[1]):
        rs = [c for c in ho if c.ds == ds]
        e, d = GH5.pooled(rs, lambda c: c.pol["v5"][0] - c.keep)
        r_, dr = GH5.pooled(rs, lambda c: c.pol["v5"][0] - c.pol["random"][0])
        rows.append((tg, e, *GS.interval(d), r_, *GS.interval(dr)))
    e, d = GH5.pooled(ho, lambda c: c.pol["v5"][0] - c.keep)
    r_, dr = GH5.pooled(ho, lambda c: c.pol["v5"][0] - c.pol["random"][0])
    rows.append(("All", e, *GS.interval(d), r_, *GS.interval(dr)))
    for tg, e_, lo_, hi_, r__, rlo, rhi in rows:
        F(f"VFiveHOChart{tg}Gain", f"held-out, method v5, {tg}: change over the first pass (C13)", sg(e_), sg(lo_), sg(hi_),
          fn="v6_phase2.v5_chart_data")
        F(f"VFiveHOChart{tg}VsRand", f"held-out, method v5, {tg}: v5 minus random at v5's held-out share (C13)", sg(r__),
          sg(rlo), sg(rhi), fn="v6_phase2.v5_chart_data")
    wcsv("c13_heldout", ["set", "change", "lo", "hi", "vs_random", "vs_random_lo", "vs_random_hi"], rows)


def v5_facts():
    """Fold docs/naacl/results/v5_facts.json (F-VFive*, exploratory, ids kept) with its provenance."""
    f = ROOT / "docs/naacl/results/v5_facts.json"
    if not f.exists() or VARIANT != "v4_1":
        return
    d = json.loads(f.read_text())
    for x in d["facts"]:
        fid = x["id"][2:]
        if fid.endswith("Definition"):
            note(fid, x["claim"], x["value"])
            continue
        fact(fid, x["claim"], x["value"], (x.get("src") or "docs/naacl/results/v5_facts.json", x.get("fn") or "v5.score"),
             x.get("status") or "exploratory", x.get("lo"), x.get("hi"), func=x.get("fn") or "v5.score")
    if V5:  # the oracle's calls and tokens, which v5.py leaves out: the calls of the fields it routes
        acc, cl, tk = v5_policy(V5["core"], "oracle")
        F("VFiveOracleCalls", "test, oracle at v5's share: second-pass calls per field (the calls of the fields it routes)",
          f"{cl:.2f}", fn="v6_phase2.v5_setup")
        F("VFiveOracleTokens", "test, oracle at v5's share: second-pass tokens per field", f"{tk:,.0f}", fn="v6_phase2.v5_setup")
    note("VFiveProvenance", "how method v5 was fitted and scored", "Method v5 (method v4.1 + the description repair, the "
         "improved reflection, the either-order verifier and the benefit router) was fitted and frozen on development in "
         "docs/naacl/results/v5_frozen.json, commit 463f9c4, before its single test scoring in bd25112 "
         "(scripts/naacl/v5.py --stage score, docs/naacl/results/v5_test.md). The F-VFive* facts are read from "
         "docs/naacl/results/v5_facts.json. v6 rebuilds the same policies in memory from the stored v5 outputs "
         "(results/naacl/v6/v5/test) for Table 1, the router table and the charts, and asserts that the rebuild "
         "reproduces the scored accuracies.")
    mf = ROOT / "docs/naacl/results/v5_matched_facts.json"
    if mf.exists():  # the matched-budget cost check of v5 (F-VFiveMatched*), ids kept
        dm = json.loads(mf.read_text())
        for x in dm["facts"]:
            fid = x["id"][2:]
            if fid.endswith("Definition"):
                note(fid, x["claim"], x["value"])
                continue
            fact(fid, x["claim"], x["value"], (x.get("src") or "docs/naacl/results/v5_matched_facts.json", x.get("fn") or "v5_matched"),
                 x.get("status") or "exploratory", x.get("lo"), x.get("hi"), func=x.get("fn") or "v5_matched")
        if dm.get("definition"):
            note("VFiveMatchedDefinition", "the v5 matched-budget check", str(dm["definition"]))
        note("VFiveMatchedProvenance", "how the v5 matched-budget check was fitted and scored", "Its development-estimated "
             "costs were frozen in commit c105867 before the single test scoring in 188d631; the F-VFiveMatched* facts are "
             "read from docs/naacl/results/v5_matched_facts.json, and Figure 2 draws its test curves "
             "(paper/figures/data6/v5_matched_curves.csv).")
    cf = ROOT / "docs/naacl/results/v5_controls_facts.json"
    if cf.exists():  # task C: free selection controls on the v5 second pass (F-VFiveCtl*), ids kept
        dc = json.loads(cf.read_text())
        for x in dc["facts"]:
            fid = x["id"][2:]
            if fid.endswith("Definition"):
                note(fid, x["claim"], x["value"])
                continue
            fact(fid, x["claim"], x["value"], (x.get("src") or "docs/naacl/results/v5_controls_facts.json", x.get("fn") or "v5_controls"),
                 x.get("status") or "exploratory", x.get("lo"), x.get("hi"), func=x.get("fn") or "v5_controls")
        if dc.get("definition"):
            note("VFiveCtlDefinition", "the free selection controls", str(dc["definition"]))
        note("VFiveCtlProvenance", "how the free selection controls were fitted and scored", "The lexical correctness "
             "classifier, the legacy free composite and the structural trigger were fitted and frozen on v5 development data "
             "in commit 6758c9a before their single test scoring in c56dec2 (scripts/naacl/v5_controls.py; "
             "paper/tab6_v5_controls.tex). The F-VFiveCtl* facts are read from docs/naacl/results/v5_controls_facts.json.")
    note("MethodVersions", "which method each fact describes", "F-VFive* and the v5 columns of Table 1 and of the router "
         "table are method v5, the paper's method. Facts without the VFive prefix are method v4.1 (the reference), except "
         "F-PDFirst/Main/Every and F-PDHOFirst/Main/Every, which Table 1 now takes from v5; their v4.1 values are "
         "F-PDVFourOne* and F-PDHOVFourOne*. The interventions (Table 2), signals (Table 3) and detection tables, and the "
         "charts built from them, still use the v4.1 second pass and are pending the R-X1 and R-UQ reruns.")


# ============================================================================================ R2 signal gaps
HOX = {}
HOU9 = {}


def heldout_signals():
    """Held-out error AUROC per model x dataset cell for every signal with stored held-out outputs, on the cells and
    fields of uq_ablation_v4.heldout_aurocs (answered, stated confidence and P(True) present, >= 5 errors and >= 5
    correct). Score-list signals come from the stored v2 first pass of the held-out sets; the logistic is the per-model
    core-development fit of newdata_analysis.fit_combo (no held-out label)."""
    import uq_ablation_v4 as UQ4
    argv, sys.argv = sys.argv, ["newdata_analysis.py"]
    import newdata_analysis as ND
    sys.argv = argv
    combos = ND.fit_combo()
    cells_, fields = [], defaultdict(list)
    miss = 0
    for ds in UQ4.HELDOUT:
        ids = set(json.load(open(ROOT / f"data/splits/{ds}.json"))["test"])
        for m in U.MODELS:
            f = ROOT / f"results/naacl/v4/test/given/{ds}__{m}/fields.jsonl"
            if not Path(str(f) + ".done").exists():
                continue
            uq = U.uq_rows(ds, m)
            fp = U.first_pass(ds, m, ids)
            nw = ho_u9_rows(ds, m)
            fs = []
            for r in U.read(f):
                if not r["valid"]:
                    continue
                u = uq.get(r["key"], {})
                if "verbalised" not in u or "p_true" not in u:
                    continue
                s1, mg = float(r["s1"]), router.canonical_margin(float(r["margin"]))
                rec = fp.get(r["key"])
                x = dict(base=U.target_correct(r["base"], r["gts"]), margin=mg, s1=s1,
                         ratio=math.log((s1 + 1e-3) / (s1 - mg + 1e-3)), verbalised=u["verbalised"], p_true=u["p_true"])
                if rec is not None and abs(rec["margin"] - mg) < 1e-9 and (rec["base"] or "") == (r["base"] or ""):
                    v = U.score_signals(rec)
                    x.update(entropy=v["entropy"], norm_margin=v["norm_margin"], gap13=v["gap13"])
                else:
                    miss += 1
                if m in combos:
                    x["combo_paid"] = float(combos[m][0].predict_proba(np.array([[mg, s1, u["p_true"], u["verbalised"]]]))[0, 1])
                x.update(nw.get(r["key"], {}))
                fs.append(x)
            e = sum(1 - x["base"] for x in fs)
            if e < 5 or len(fs) - e < 5:
                continue
            err = np.array([1 - x["base"] for x in fs], float)
            a = {}
            for s in ("margin", "s1", "ratio", "entropy", "verbalised", "p_true", "combo_paid", "norm_margin", "gap13",
                      "lp_answer", "sem_entropy"):
                ok = np.array([s in x for x in fs])
                if ok.sum() == len(fs):
                    a[s] = U.auroc_w(np.array([x[s] for x in fs], float), err)
            cells_.append((ds, m, len(fs), int(e), a))
            fields[m] += [dict(x, ds=ds) for x in fs]
    for s, tg in (("margin", "UQAblHOMarginMeanAUC"), ("verbalised", "UQAblHOStatedMeanAUC"), ("p_true", "UQAblHOPTrueMeanAUC"),
                  ("ratio", "UQAblHORatioMeanAUC")):
        assert f"{np.mean([c[4][s] for c in cells_]):.2f}" == MUQ[tg], (s, MUQ[tg])  # same cells as the v4 table
    HOX.update(cells=cells_, fields=fields, miss=miss)
    F("SigHOEntropyMissing", "held-out answered fields whose stored first-pass ranking does not match the v4 row (score "
      "entropy left out there)", f"{miss:,}", fn="v6_phase2.heldout_signals")
    out = {}
    for s in ("s1", "entropy", "combo_paid", "norm_margin", "gap13", "lp_answer", "sem_entropy"):
        vals = [c[4][s] for c in cells_ if s in c[4]]
        if vals:
            out[s] = (float(np.mean(vals)), len(vals), sum(c[4][s] > c[4]["margin"] for c in cells_ if s in c[4]),
                      float(np.mean([c[4]["margin"] for c in cells_ if s in c[4]])))
    return out


def ho_u9_rows(ds, m):
    """The U9 held-out rows (results/naacl/v6/u9ho, scripts/naacl/v6_calls/run_u9.sh with HELDOUT=1) read as
    scripts/naacl/v6_calls/ho_score.py reads them: answer-token log-probability (a .fill file replaces a missing value)
    and answer entropy (negated, higher = more confident)."""
    out = defaultdict(dict)
    d = ROOT / "results/naacl/v6/u9ho"
    for sig, s_, neg, suf in (("logprob", "lp_answer", False, ""), ("logprob", "lp_answer", False, ".fill"),
                              ("samples", "sem_entropy", True, "")):
        f = d / sig / f"{ds}__{m}{suf}.jsonl"
        if f.exists():
            for r in read(f):
                if r.get(s_) is not None:
                    out[r["key"]][s_] = -float(r[s_]) if neg else float(r[s_])
    return out


def ho_swaps():
    """Held-out routing swaps for answer-token log-probability and answer entropy, computed exactly as newdata_analysis
    computes F-SigHOSwap{Stated,Logit,PTrue}: per held-out cell, the answered fields with both paid signals; the gate
    routes the same number of fields Map-Reflect routes there; final accuracy after the v4.1 second pass; pooled over
    the five held-out sets (Magellan left out) with the split-group bootstrap (B = 2000). A signal is scored on the
    fields of that population where its stored U9 value exists, with the margin on the same fields; fields without it
    are counted, never filled. The stated swap is recomputed the same way and must reproduce F-SigHOSwapStated."""
    argv, sys.argv = sys.argv, ["newdata_analysis.py"]
    import newdata_analysis as ND
    sys.argv = argv
    from eval.calc_accuracy import load_gold_file
    present, base_cells = [], []
    for ds in ND.NAMES:
        man = json.load(open(ROOT / f"data/splits/{ds}.json"))
        gold = load_gold_file(ROOT / f"data/{ds}_dataset.jsonl")
        ids = set(man["test"])
        for m in MODELS:
            c = ND.cell(ds, m, ids, gold, TH[m], defaultdict(int))
            if c is None:
                continue
            rows_ = [r for r in read(ROOT / f"results/naacl/v4/test/given/{ds}__{m}/fields.jsonl") if str(r["sample_id"]) in ids]
            assert len(rows_) == c.n
            c.keys = [r["key"] for r in rows_]
            base_cells.append(c)
            if ds not in present:
                present.append(ds)
    groups = GS.split_groups(present)
    sets = [d for d in present if d != ND.SANITY]
    out, cover = {}, {}
    for sig, tg in (("verbalised", "Stated"), ("lp_answer", "AnsLogp"), ("sem_entropy", "SemEnt")):
        subs, missing, total = [], 0, 0
        for c in base_cells:
            pop = c.valid & ~np.isnan(c.verb) & ~np.isnan(c.ptrue)
            if sig == "verbalised":
                val = c.verb
                has = pop.copy()
            else:
                nw = ho_u9_rows(c.ds, c.model)
                val = np.array([nw.get(k, {}).get(sig, np.nan) for k in c.keys], float)
                has = pop & ~np.isnan(val)
            if c.ds != ND.SANITY:
                missing += int((pop & ~has).sum())
                total += int(pop.sum())
            if not has.any():
                continue
            k_ = float(c.routed[has].sum())
            b_, a_ = c.vec["keep all"][has], c.vec["reflect all, v4 action"][has]
            sc = type("C", (), {})()
            sc.ds, sc.model, sc.n, sc.sid = c.ds, c.model, int(has.sum()), c.sid[has]
            sc.vec = {"margin": b_ + U.gate_weights(c.margin[has], k_) * (a_ - b_),
                      sig: b_ + U.gate_weights(val[has], k_) * (a_ - b_)}
            subs.append(sc)
        GBs = GS.GroupBootstrap(subs, groups, 2000)
        pt, lo, hi, _ = ND.estimate(GBs, subs, sets, sig, "margin")
        out[tg] = (pt, lo, hi)
        cover[tg] = (missing, total)
    st = out.pop("Stated")
    assert f"{st[0]:+.2f}" == MHO["HOAllSwapStated"], (st, MHO["HOAllSwapStated"])  # reproduces newdata_analysis
    for tg, (pt, lo, hi) in out.items():
        miss, tot = cover[tg]
        F(f"SigHOSwap{tg}", f"held-out pooled: routing by {UQ4SIG({'AnsLogp': 'lp_answer', 'SemEnt': 'sem_entropy'}[tg])} minus "
          f"the margin at Map-Reflect's held-out share (v4.1; fields with the stored signal: {tot - miss:,} of {tot:,})",
          sg(pt), sg(lo), sg(hi), fn="v6_phase2.ho_swaps")
        F(f"SigHOSwap{tg}Missing", f"held-out swap population fields without a stored {tg} value (left out, not filled)",
          f"{miss:,} of {tot:,}", fn="v6_phase2.ho_swaps")


def r2_signals():
    ho = heldout_signals()
    ho_swaps()
    TAGS = {"s1": "TopScore", "entropy": "ScoreEnt", "combo_paid": "ComboPaid"}
    for s, tg in TAGS.items():
        mean, k, wins, _ = ho[s]
        F(f"Sig{tg}HOAUC", f"{UQ4SIG(s)}: mean held-out within-cell error AUROC ({k} cells; logistic: core-development fit)"
          if s == "combo_paid" else f"{UQ4SIG(s)}: mean held-out within-cell error AUROC ({k} cells)", f"{mean:.2f}",
          fn="v6_phase2.heldout_signals")
        F(f"Sig{tg}HOWins", f"{UQ4SIG(s)}: held-out cells where it beats the margin's error AUROC", f"{wins} of {k}",
          fn="v6_phase2.heldout_signals")
    u9 = ROOT / "docs/naacl/results/u9_heldout_facts.json"
    HOU9.update(json.loads(u9.read_text()) if u9.exists() else {})
    diffs = []
    for s, tg in (("lp_answer", "AnsLogp"), ("sem_entropy", "SemEnt")):
        if s not in ho:
            NEEDS_RUNS.append((f"held-out AUROC, {UQ4SIG(s)}", "no held-out outputs were collected", "see ho_score.py"))
            continue
        mean, k, wins, mm = ho[s]
        F(f"Sig{tg}HOAUC", f"{UQ4SIG(s)}: mean held-out within-cell error AUROC ({k} cells), recomputed under v4.1 on the "
          "current population from the stored U9 held-out rows", f"{mean:.2f}", fn="v6_phase2.heldout_signals")
        F(f"Sig{tg}HOWins", f"{UQ4SIG(s)}: held-out cells where it beats the margin's error AUROC", f"{wins} of {k}",
          fn="v6_phase2.heldout_signals")
        F(f"Sig{tg}HOCells", f"{UQ4SIG(s)}: held-out cells in that mean", f"{k}", fn="v6_phase2.heldout_signals")
        F(f"Sig{tg}HOMarginSameCells", f"the margin's mean held-out AUROC on the same cells", f"{mm:.2f}", fn="v6_phase2.heldout_signals")
        for key, mine in ((f"F-Sig{tg}HOAUC", f"{mean:.2f}"), (f"F-Sig{tg}HOWins", f"{wins} of {k}"),
                          (f"F-Sig{tg}HOMarginSameCells", f"{mm:.2f}")):
            if key in HOU9 and HOU9[key] != mine:
                diffs.append(f"{key} {HOU9[key]} (ho_score, v4) -> {mine} (v4.1)")
    if HOU9:
        HOX["u9diffs"] = diffs
    # share of the oracle gain for every signal, on the models that have it
    import uq_ablation_v4 as UQ4_
    for s in SIGORDER:
        ms = uqmods(s)
        sh, shm, _ = share_of_oracle(ms, s)
        tg = UQ4_.MAC[s]
        if len(ms) < 4:
            F(f"Sig{tg}OracleShare", f"{UQ4SIG(s)}: share of the oracle gain at v4's share, pooled over the models that have "
              f"it ({mname(ms)}), %", f"{sh:.0f}", fn="v6_phase2.r2_signals")
            F(f"Sig{tg}OracleShareMargin", f"the margin's share of the oracle gain on the same models ({mname(ms)}), %",
              f"{shm:.0f}", fn="v6_phase2.r2_signals")
        SIGX.setdefault(s, {})["share"] = f"{sh:.0f}%"
        SIGX[s]["share_models"] = ms
    if U9ON:  # the pre-U9 shares, on the models that had the signal then, kept for traceability
        for s, tg in (("lp_answer", "AnsLogp"), ("sem_entropy", "SemEnt")):
            ms0 = [m for m in U.MODELS if s in L0["avail"][m]]
            res, _ = L0["group"](ms0, ["keep", "oracle", "margin", s])
            k0, o0 = res["keep"][0], res["oracle"][0]
            F(f"Sig{tg}SubOracleShare", f"{UQ4SIG(s)}: share of the oracle gain before U9, on {mname(ms0)}, %",
              f"{100 * (res[s][0] - k0) / (o0 - k0):.0f}", fn="v6_phase2.r2_signals")
            F(f"Sig{tg}SubOracleShareMargin", f"the margin's share of the oracle gain on the same models ({mname(ms0)}), %",
              f"{100 * (res['margin'][0] - k0) / (o0 - k0):.0f}", fn="v6_phase2.r2_signals")
    for s, tg in TAGS.items():
        SIGX.setdefault(s, {})["hoauc"] = f"{ho[s][0]:.2f}"
    for s, tg in (("lp_answer", "AnsLogp"), ("sem_entropy", "SemEnt")):
        SIGX.setdefault(s, {})["hoauc"] = f"{ho[s][0]:.2f}" if s in ho else "not run"
    return ho


def UQ4SIG(s):
    import uq_ablation_v4 as UQ4_
    return UQ4_.SIG[s][0] if s in UQ4_.SIG else s


# ================================================================================================ U1 budgets
SHARES = [round(0.05 * i, 2) for i in range(21)]


def curve(models, s, shares=SHARES):
    """Accuracy (%) after the v4 second pass when signal s routes each share of every model's population."""
    out = []
    for b in shares:
        tot = n = 0.0
        for m in models:
            fs = L["pop"][m]
            base = np.array([f["base"] for f in fs], float)
            o = np.array([f["out"] for f in fs], float)
            k = b * len(fs)
            if s == "random":
                w = np.full(len(fs), b)
            elif s == "oracle":
                w = U.gate_weights(-(o - base), k)
            else:
                w = U.gate_weights(np.array([f["sig"][s] for f in fs], float), k)
            tot += float(np.sum(base + w * (o - base)))
            n += len(fs)
        out.append(100 * tot / n)
    return out


U1 = {}


def u1():
    rows = []
    for s in SIGORDER + ["norm_margin", "gap13"]:
        ms = uqmods(s) if s in SIGORDER else uqmods("margin")
        cv, cm = curve(ms, s), curve(ms, "margin")
        cr, co = curve(ms, "random"), curve(ms, "oracle")
        U1[s] = (ms, cv, cm, cr, co)
        for b, a, am, ar, ao in zip(SHARES, cv, cm, cr, co):
            rows.append((s, "+".join(ms), b, a, am, ar, ao))
    wcsv("c1_budget_curves", ["signal", "models", "share", "accuracy", "margin_accuracy_same_models", "random_accuracy",
                              "oracle_accuracy"], rows)
    import uq_ablation_v4 as UQ4_
    TB = []
    for s in SIGORDER:
        ms, cv, cm, cr, co = U1[s]
        k0, every = cv[0], cv[-1]
        gain = np.array(cv) - k0
        reach = next((b for b, x in zip(SHARES, gain) if x >= 0.95 * (every - k0)), 1.0)
        mean = float(np.mean(gain))
        tg = UQ4_.MAC[s]
        at = {b: cv[SHARES.index(b)] - k0 for b in (0.1, 0.2, 0.3, 0.5)}
        F(f"BudgetMean{tg}", f"{UQ4SIG(s)}: mean change over routed shares 0 to 100% in steps of 5 ({mname(ms)})", sg(mean),
          fn="v6_phase2.u1")
        if s != "margin":
            F(f"BudgetMarginMean{tg}", f"the margin's mean change over the same shares on the same models ({mname(ms)})",
              sg(float(np.mean(np.array(cm) - cm[0]))), fn="v6_phase2.u1")
        for b, x in at.items():
            F(f"Budget{tg}At{WORD[int(round(100 * b))]}", f"{UQ4SIG(s)}: change routing {int(round(100 * b))}% of fields ({mname(ms)})", sg(x),
              fn="v6_phase2.u1")
        F(f"Budget{tg}Reach", f"{UQ4SIG(s)}: smallest routed share reaching 95% of the every-field change, % ({mname(ms)})",
          f"{100 * reach:.0f}", fn="v6_phase2.u1")
        TB.append((SSHORT[s], ms, at, mean, float(np.mean(np.array(cm) - cm[0])), 100 * reach))
    ms, cv, cm, cr, co = U1["margin"]
    F("BudgetEvery", "every field (routed share 100%): change on the signal population (all four models)", sg(cv[-1] - cv[0]),
      fn="v6_phase2.u1")
    F("BudgetRandomMean", "random routing: mean change over routed shares 0 to 100% (all four models)",
      sg(float(np.mean(np.array(cr) - cr[0]))), fn="v6_phase2.u1")
    F("BudgetOracleMean", "oracle routing: mean change over routed shares 0 to 100% (all four models)",
      sg(float(np.mean(np.array(co) - co[0]))), fn="v6_phase2.u1")
    body = []
    for lab, ms, at, mean, mm, reach in TB:
        body.append(f"{lab} & {len(ms)} & " + " & ".join(tex_s(at[b]) for b in (0.1, 0.2, 0.3, 0.5))
                    + f" & {tex_s(mean)} & {tex_s(mm)} & {reach:.0f} \\\\")
    write_tab("budget", "lrrrrrrrr", r"UQ method & Models & 10\% & 20\% & 30\% & 50\% & Mean & Margin & "
              r"Rate for 95\% (\%) \\", body,
              note_="U1 (exploratory). Change over the first pass (points) when the signal routes each share of every "
              "model's fields to the v4 second pass; uq population.")


# ======================================================================================= U2 detection metrics
def ece_brier(p, y, bins=10):
    p = np.clip(np.asarray(p, float), 0, 1)
    y = np.asarray(y, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean()) for k in range(bins) if (b == k).any())
    return float(ece), float(np.mean((p - y) ** 2))


CURVECHK = []  # (curve, reported AUROC, curve AUROC, reported AURC, curve AURC)


def tie_curves(c, e, auc_rep, aurc_rep, name, max_pts=600):
    """Tie-aware ROC and risk-coverage vertices for confidence c (higher = more confident) and errors e (1 = wrong).
    ROC: one vertex per distinct threshold, adding each tied group whole (errors are flagged least confident first), so
    the trapezoid integral equals the tie-aware AUROC (ties count one half). Risk-coverage: most confident first, a tied
    group entered by expected fractional inclusion (uq_ablation.aurc), so the mean risk equals the AURC. Both are
    thinned for the CSV only where that keeps the integral within 0.005 of the reported value (asserted)."""
    u, gid = np.unique(c, return_inverse=True)
    err = np.bincount(gid, weights=e, minlength=len(u))
    cnt = np.bincount(gid, minlength=len(u)).astype(float)
    P, Nn = err.sum(), (cnt - err).sum()
    tp = np.concatenate([[0.0], np.cumsum(err)]) / P  # ascending confidence = flagged first
    fp = np.concatenate([[0.0], np.cumsum(cnt - err)]) / Nn
    auc_curve = float(np.trapezoid(tp, fp))
    risks, cov, E_, N_ = [], [], 0.0, 0
    for ge, gc in zip(err[::-1], cnt[::-1]):  # descending confidence
        j = np.arange(1, int(gc) + 1)
        risks.append((E_ + ge * j / gc) / (N_ + j))
        cov.append((N_ + j) / len(e))
        E_ += ge
        N_ += int(gc)
    risks, cov = np.concatenate(risks), np.concatenate(cov)
    aurc_curve = float(risks.mean())

    def thin(x, y, area):
        if len(x) <= max_pts:
            return x, y
        keep = np.unique(np.r_[np.linspace(0, len(x) - 1, max_pts).astype(int), len(x) - 1])
        xs, ys = x[keep], y[keep]
        return (xs, ys) if abs(area(xs, ys) - area(x, y)) <= 0.002 else (x, y)
    rx, ry = thin(fp, tp, lambda x, y: np.trapezoid(y, x))
    kx, ky = thin(cov, risks, lambda x, y: np.trapezoid(y, x) / (x[-1] - x[0]) if x[-1] > x[0] else y.mean())
    auc_csv = float(np.trapezoid(ry, rx))
    aurc_csv = float(np.trapezoid(ky, kx) / (kx[-1] - kx[0])) if kx[-1] > kx[0] else float(ky.mean())
    assert abs(auc_curve - auc_rep) < 1e-9, (name, auc_curve, auc_rep)
    assert abs(auc_csv - auc_rep) <= 0.005, (name, "ROC CSV", auc_csv, auc_rep)
    assert abs(aurc_curve - aurc_rep) < 1e-9, (name, aurc_curve, aurc_rep)
    assert abs(aurc_csv - aurc_rep) <= 0.005, (name, "risk-coverage CSV", aurc_csv, aurc_rep)
    CURVECHK.append((name, auc_rep, auc_csv, aurc_rep, aurc_csv, len(rx), len(kx)))
    return list(zip(map(float, rx), map(float, ry))), list(zip(map(float, kx), map(float, ky)))


def olddev_fields():
    """Per model: the core development fields that carry every signal the test population has (the 'old development'
    samples, manifest old_dev_excluded; the v4 development split has no paid signals). Full-identity scorer,
    ITOpsGraph v2.1."""
    import uq_ablation_v4 as UQ4_
    out = defaultdict(list)
    for ds in UQ4_.CORE10:
        man = json.load(open(ROOT / f"data/splits/{ds}.json"))
        od = set(man.get("old_dev_excluded") or [])
        if not od:
            continue
        for m in U.MODELS:
            fp = U.first_pass(ds, m, od)
            uq = U.uq_rows(ds, m)
            for k, rec in fp.items():
                if not rec["valid"]:
                    continue
                try:
                    g = U.gold(ds, rec["sample_id"], k)
                except KeyError:
                    continue
                if g is None:
                    continue
                gl = rec["gts"] if g == "keep" else g
                v = U.score_signals(rec)
                v.update({a: b for a, b in uq.get(k, {}).items() if not a.startswith("_")})
                out[m].append(dict(ds=ds, base=float(U.target_correct(rec["base"], gl)), sig=v))
    return out


def iso_fit(x, y):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(np.asarray(x, float), np.asarray(y, float))


DET = {}


def u2():
    dev = olddev_fields()
    F("DetDevFields", "core development fields used to fit the isotonic calibration (old development samples, answered, "
      "all models)", f"{sum(len(v) for v in dev.values()):,}", fn="v6_phase2.u2")
    rows, roc, rc = [], [], []
    import uq_ablation_v4 as UQ4_
    for s in SIGORDER:
        ms = uqmods(s)
        c, e, fs = pooled_rank(ms, s)
        a, lo, hi = auc_ci(ms, s)
        ap, ar = U.auprc(c, e), U.aurc(c, e)
        # isotonic calibration fitted on development, per model, applied to the test population
        ps, ys, cal_ok = [], [], True
        for m in ms:
            d = [f for f in dev[m] if s in f["sig"]]
            if len(d) < 200 or s == "combo_paid":
                cal_ok = False
                break
            iso = iso_fit([f["sig"][s] for f in d], [f["base"] for f in d])
            ps.append(iso.predict(np.array([f["sig"][s] for f in L["pop"][m]], float)))
            ys.append(np.array([f["base"] for f in L["pop"][m]], float))
        ece, brier = ece_brier(np.concatenate(ps), np.concatenate(ys)) if cal_ok else (None, None)
        tg = UQ4_.MAC[s]
        F(f"Det{tg}AUC", f"{UQ4SIG(s)}: pooled core-test error AUROC with interval ({mname(ms)})", f"{a:.3f}", f"{lo:.3f}", f"{hi:.3f}",
          fn="v6_phase2.u2")
        F(f"Det{tg}AUPRC", f"{UQ4SIG(s)}: pooled core-test error AUPRC (errors positive; base rate {100 * e.mean():.1f}%)", f"{ap:.3f}",
          fn="v6_phase2.u2")
        F(f"Det{tg}AURC", f"{UQ4SIG(s)}: pooled core-test area under the risk-coverage curve (lower is better)", f"{ar:.4f}",
          fn="v6_phase2.u2")
        if cal_ok:
            F(f"Det{tg}ECE", f"{UQ4SIG(s)}: ECE after isotonic calibration fitted on development (10 bins)", f"{ece:.3f}", fn="v6_phase2.u2")
            F(f"Det{tg}Brier", f"{UQ4SIG(s)}: Brier score after isotonic calibration fitted on development", f"{brier:.3f}", fn="v6_phase2.u2")
        DET[s] = dict(ms=ms, auc=(a, lo, hi), auprc=ap, aurc=ar, ece=ece, brier=brier, base=e.mean())
        # curves (pooled ranks): ROC and risk-coverage, tie-aware, checked against the reported AUROC and AURC
        r_, k_ = tie_curves(c, e, a, ar, f"{s} core")
        roc += [(s, "core", x, y) for x, y in r_]
        rc += [(s, "core", x, y) for x, y in k_]
        # per model x dataset
        for m in ms:
            for ds in UQ4_.CORE10:
                f_ = [f for f in L["pop"][m] if f["ds"] == ds]
                y = 1 - np.array([f["base"] for f in f_], float)
                if y.sum() < 5 or len(y) - y.sum() < 5:
                    continue
                x = np.array([f["sig"][s] for f in f_], float)
                rows.append((s, m, ds, len(y), int(y.sum()), U.auroc_w(x, y), U.auprc(x, y), U.aurc(x, y)))
    # held-out pooled (within-model ranks) for the signals with held-out outputs
    hof = HOX["fields"]
    for s in ("margin", "s1", "ratio", "entropy", "verbalised", "p_true", "combo_paid", "lp_answer", "sem_entropy"):
        cs, es = [], []
        for m in U.MODELS:
            fs = [f for f in hof[m] if s in f]
            if not fs:
                continue
            cs.append(U.pct_rank(np.array([f[s] for f in fs], float)))
            es.append(1 - np.array([f["base"] for f in fs], float))
        c, e = np.concatenate(cs), np.concatenate(es)
        a = U.auroc_w(c, e)
        tg = UQ4_.MAC[s]
        F(f"Det{tg}HOPooledAUC", f"{UQ4SIG(s)}: pooled held-out error AUROC (within-model ranks, answered fields with both paid "
          f"signals, {len(e):,} fields)", f"{a:.3f}", fn="v6_phase2.u2")
        DET[s]["ho"] = a
        r_, k_ = tie_curves(c, e, a, U.aurc(c, e), f"{s} held-out")
        roc += [(s, "heldout", x, y) for x, y in r_]
        rc += [(s, "heldout", x, y) for x, y in k_]
    if HOX.get("u9diffs") is not None:
        for s, tg in (("lp_answer", "AnsLogp"), ("sem_entropy", "SemEnt")):
            key = f"F-Det{tg}HOPooledAUC"
            if key in HOU9 and "ho" in DET[s] and HOU9[key] != f"{DET[s]['ho']:.3f}":
                HOX["u9diffs"].append(f"{key} {HOU9[key]} (ho_score, v4) -> {DET[s]['ho']:.3f} (v4.1)")
        note("SigHOUNineRecomputed", "held-out AUROC of token log-probability and answer entropy",
             "Recomputed under v4.1 (canonical margins, unresolved stated answers missing) on the current held-out "
             "population from the stored U9 held-out rows (results/naacl/v6/u9ho), read as scripts/naacl/v6_calls/ho_score.py "
             "reads them; docs/naacl/results/u9_heldout_facts.json (v4) is used only for this comparison. Differences: "
             + ("; ".join(HOX["u9diffs"]) or "none") + ".")
    note("CurveCheck", "C7 and C8 curves are tie-aware and integrate to the reported values",
         "ROC vertices at distinct thresholds (tied groups added whole); risk-coverage with expected fractional inclusion of "
         "a tied group. Trapezoid integral of each CSV curve minus the reported value, largest absolute: AUROC "
         f"{max(abs(x[2] - x[1]) for x in CURVECHK):.4f}, AURC {max(abs(x[4] - x[3]) for x in CURVECHK):.4f} "
         f"(asserted within 0.005; {len(CURVECHK)} curves).")
    wcsv("c7_roc", ["signal", "split", "false_positive_rate", "true_positive_rate"], roc)
    wcsv("c8_risk_coverage", ["signal", "split", "coverage", "risk"], rc)
    wcsv("u2_detection_cells", ["signal", "model", "dataset", "fields", "errors", "auroc", "auprc", "aurc"], rows)
    body = []
    for s in SIGORDER:
        d = DET[s]
        a, lo, hi = d["auc"]
        body.append(f"{SSHORT[s]} & {len(d['ms'])} & {a:.3f} [{lo:.3f}, {hi:.3f}] & "
                    f"{d['ho']:.3f} & " if "ho" in d else f"{SSHORT[s]} & {len(d['ms'])} & {a:.3f} [{lo:.3f}, {hi:.3f}] & not run & ")
        body[-1] += (f"{d['auprc']:.3f} & {d['aurc']:.4f} & "
                     + (f"{d['ece']:.3f} & {d['brier']:.3f}" if d["ece"] is not None else
                        "n/a & n/a") + r" \\")
    write_tab("detect", "lrrrrrrr", r"UQ method & Models & Test AUROC [95\% CI] & Held-out & AUPRC & AURC & ECE & "
              r"Brier \\", body, env="table*", note_="U2 (exploratory). Pooled over models with within-model ranks; "
              "isotonic calibration fitted per model on the old development fields.")


# ===================================================================================== U3 margin variants
def u3():
    ms = uqmods("margin")
    res, n = L["group"](ms, ["margin", "norm_margin", "ratio", "entropy", "gap13"])
    body = []
    lab = {"margin": "raw margin $s_1 - s_2$", "norm_margin": "normalised margin (scores sum to 1)",
           "ratio": r"ratio $\log(s_1/s_2)$", "entropy": "score entropy", "gap13": "top-1 minus top-3 gap"}
    tg = {"margin": "Raw", "norm_margin": "Norm", "ratio": "Ratio", "entropy": "Entropy", "gap13": "GapThree"}
    for s in ("margin", "norm_margin", "ratio", "entropy", "gap13"):
        a, lo, hi = auc_ci(ms, s)
        acc, d, (dlo, dhi) = res[s]
        F(f"MVar{tg[s]}AUC", f"margin variant '{lab[s]}': pooled core-test error AUROC", f"{a:.3f}", f"{lo:.3f}", f"{hi:.3f}",
          fn="v6_phase2.u3")
        if s != "margin":
            F(f"MVar{tg[s]}Delta", f"margin variant '{lab[s]}': routing at v4's share minus the raw margin, points", sg(d), sg(dlo),
              sg(dhi), fn="v6_phase2.u3")
        body.append(f"{lab[s]} & {a:.3f} [{lo:.3f}, {hi:.3f}] & {acc:.2f} & "
                    + ("ref." if s == "margin" else f"{tex_s(d)} {tci(dlo, dhi)}") + r" \\")
    write_tab("marginvar", "lrrr", r"Variant & Test AUROC [95\% CI] & Accuracy & $\Delta$ vs raw margin [95\% CI] \\", body,
              note_="U3 (exploratory). Uq population, all four models, routing at v4's share.")


def tci(lo, hi, nd=2):
    return "[" + tex_s(float(lo), nd) + ", " + tex_s(float(hi), nd) + "]"


# ====================================================================================== U4 router components
def dev_rows():
    """Per model: the v4 development fields (results/naacl/v4/dev/given, dev split), with each action's outcome, as
    v4_fit.load (full identity, ITOpsGraph v2.1) plus the sample id."""
    out = {}
    for m in MODELS:
        D = defaultdict(list)
        for ds in DATASETS:
            ids = set(json.load(open(ROOT / f"data/splits/{ds}.json"))["dev"])
            for r in read(ROOT / f"results/naacl/v4/dev/given/{ds}__{m}/fields.jsonl"):
                sid = str(r["sample_id"])
                if sid not in ids:
                    continue
                tg = V.gold_of(ds, sid, r["key"].rsplit("||", 1)[1], r["gts"])
                if tg is None:
                    continue
                ok = lambda p: float(target_correct(p, tg))  # noqa: E731
                D["sid"].append(f"{ds}|{sid}")
                D["m"].append(router.canonical_margin(float(r["margin"])))
                D["s1"].append(float(r["s1"]))
                D["valid"].append(bool(r["valid"]))
                D["b"].append(ok(r["base"]))
                for X in ("NT", "WT20", "WT50"):
                    D[f"{X}_ok"].append(ok(r.get(X)))
                    p = r.get(f"{X}_proposed")
                    D[f"{X}_ch"].append(float(bool(p) and p != r["base"]))
        out[m] = {k: np.array(v) for k, v in D.items()}
    return out


def route_dev(D, t_m, t_s, W, mode="v4"):
    """(correct, calls) per development field. mode v4: router.bands; topscore: routed when declined or s1 <= t_s."""
    if mode == "topscore":
        routed = ~D["valid"] | (D["s1"] <= t_s)
        near = routed & D["valid"] & (D["s1"] >= 1.0)
        wid = routed & ~near
    else:
        b = router.bands(D["m"], D["valid"], D["s1"], t_m, t_s)
        near, wid = b == router.NEAR_TIE, b == router.WIDEN
    ok = np.where(near, D["NT_ok"], np.where(wid, D[f"{W}_ok"], D["b"]))
    calls = near * (1 + D["NT_ch"]) + wid * (1 + D[f"{W}_ch"])
    return ok, calls


def fit_tm(D, W, t_s, grid, mode="v4", idx=None):
    best = None
    for t in grid:
        ok, c = route_dev(D, t, t_s, W, mode) if mode == "v4" else route_dev(D, None, t, W, mode)
        if idx is not None:
            ok, c = ok[idx], c[idx]
        cand = (ok.mean(), -c.sum(), -t)
        if best is None or cand[:2] > best[0][:2]:
            best = (cand, t)
    return best[1]


def test_vec(c, t_m, t_s, mode="v4"):
    """Per-field correctness and calls on a test cell with the frozen widen action and check."""
    if mode == "topscore":
        routed = ~c.valid | (c.s1 <= t_s)
        near = routed & c.valid & (c.s1 >= 1.0)
        wid = routed & ~near
    else:
        b = router.bands(c.margin, c.valid, c.s1, t_m, t_s)
        near, wid = b == router.NEAR_TIE, b == router.WIDEN
    ok = np.where(near, c.nt, np.where(wid, c.widen, c.keep))
    calls = near * (1 + c.ver_nt) + wid * (1 + c.ver_w)
    return ok, calls


U4 = {}


def u4():
    dev = dev_rows()
    tms = {m: np.unique(np.r_[0.0, np.round(np.quantile(dev[m]["m"], np.linspace(0, 1, 21)), 3)]) for m in MODELS}
    fine = np.round(np.arange(0, 1.0001, 0.05), 2)
    for c in cells:  # the frozen router reproduces Map-Reflect on test
        t = TH[c.model]
        ok, _ = test_vec(c, t["t_m"], t["t_s"])
        assert np.array_equal(ok, c.method), c.ds
    for m in MODELS:
        t = TH[m]
        ok, _ = route_dev(dev[m], t["t_m"], t["t_s"], t["widen"])
        U4.setdefault("devacc", {})[m] = ok.mean()
    nd = sum(len(dev[m]["b"]) for m in MODELS)
    devacc = sum(U4["devacc"][m] * len(dev[m]["b"]) for m in MODELS) / nd
    fz = json.loads((ROOT / FROZEN_REL).read_text())
    assert abs(devacc - fz["dev_accuracy"]) < 1e-9, (devacc, fz["dev_accuracy"])  # same development fields and scoring
    variants = {}
    # (a) frozen
    variants["frozen"] = {m: ("v4", TH[m]["t_m"], TH[m]["t_s"]) for m in MODELS}
    # (b) margin only: no top-score band, every routed field widens; t_m refit on development
    big = 9.0
    variants["marginonly"] = {m: ("v4", fit_tm(dev[m], TH[m]["widen"], big, tms[m]), big) for m in MODELS}
    # (c) top score only: routed when declined or s1 <= t; t refit on development over the s1 grid
    variants["topscore"] = {}
    for m in MODELS:
        grid = np.unique(np.r_[np.round(np.quantile(dev[m]["s1"], np.linspace(0, 1, 21)), 3), -1.0])
        variants["topscore"][m] = ("topscore", None, fit_tm(dev[m], TH[m]["widen"], None, grid, mode="topscore"))
    # (d) one t_m shared by every model (pooled development accuracy), frozen t_s and widen action
    best = None
    for t in fine:
        acc = sum(route_dev(dev[m], t, TH[m]["t_s"], TH[m]["widen"])[0].sum() for m in MODELS) / nd
        calls = sum(route_dev(dev[m], t, TH[m]["t_s"], TH[m]["widen"])[1].sum() for m in MODELS)
        if best is None or (acc, -calls) > best[0]:
            best = ((acc, -calls), t)
    variants["shared"] = {m: ("v4", float(best[1]), TH[m]["t_s"]) for m in MODELS}
    # (e) per-model t_m refit on the 0.05 grid on all development (the frozen fit used quantiles; this checks the grid)
    LAB = {"frozen": "Frozen thresholds (per-model $t_m$, $t_s$)", "marginonly": "Margin only (no top-score threshold)",
           "topscore": "Top score only (no margin)", "shared": "One $t_m$ shared by all models"}
    TG = {"frozen": "Frozen", "marginonly": "MarginOnly", "topscore": "TopOnly", "shared": "Shared"}
    rows = []
    for v, th in variants.items():
        vec = {}
        calls = 0.0
        for c in cells:
            mode, t_m, t_s = th[c.model]
            ok, cl = test_vec(c, t_m, t_s, mode)
            vec[id(c)] = ok
            calls += cl.sum()
        e, d = pooled(cells, lambda c: vec[id(c)] - c.keep)
        lo, hi = GS.interval(d)
        dd, ddr = pooled(cells, lambda c: vec[id(c)] - c.method)
        dlo, dhi = GS.interval(ddr)
        dacc = 0.0
        for m in MODELS:
            mode, t_m, t_s = th[m]
            dacc += route_dev(dev[m], t_m, t_s, TH[m]["widen"], mode)[0].sum()
        thr = "; ".join(f"{MSHORT[m]} " + (f"top score at most {th[m][2]:.3g}" if th[m][0] == "topscore" else f"t_m {th[m][1]:.3g}")
                        for m in MODELS)
        F(f"RPart{TG[v]}Gain", f"router variant '{LAB[v]}': test change over the first pass", sg(e), sg(lo), sg(hi), fn="v6_phase2.u4")
        if v != "frozen":
            F(f"RPart{TG[v]}VsMain", f"router variant '{LAB[v]}' minus Map-Reflect, test", sg(dd), sg(dlo), sg(dhi), fn="v6_phase2.u4")
        F(f"RPart{TG[v]}Calls", f"router variant '{LAB[v]}': second-pass calls per field, test", f"{calls / N:.2f}", fn="v6_phase2.u4")
        F(f"RPart{TG[v]}DevAcc", f"router variant '{LAB[v]}': development accuracy, %", f"{100 * dacc / nd:.2f}", fn="v6_phase2.u4")
        F(f"RPart{TG[v]}Thresholds", f"router variant '{LAB[v]}': thresholds fitted on development", thr.replace("t_m", "margin threshold"),
          fn="v6_phase2.u4")
        rows.append((LAB[v], 100 * dacc / nd, e, lo, hi, None if v == "frozen" else (dd, dlo, dhi), calls / N, thr))
    U4["rows"] = rows
    # (f) development size: t_m and t_s refit per model on 10/25/50/100% of the development samples (50 draws)
    rng = np.random.default_rng(20261010)
    tss = {m: np.unique(np.r_[0.0, np.round(np.quantile(dev[m]["s1"], np.linspace(0, 1, 21)), 3), dev[m]["s1"].max() + 0.01])
           for m in MODELS}
    size_rows = []
    for frac in (0.1, 0.25, 0.5, 1.0):
        gains, tdev = [], []
        reps = 1 if frac == 1.0 else 50
        for rep in range(reps):
            th = {}
            for m in MODELS:
                D = dev[m]
                sids = np.unique(D["sid"])
                pick = set(rng.choice(sids, max(1, int(round(frac * len(sids)))), replace=False)) if frac < 1 else set(sids)
                idx = np.array([s in pick for s in D["sid"]])
                best = None
                for t_s in tss[m]:
                    for t_m in tms[m]:
                        ok, cl = route_dev(D, t_m, t_s, TH[m]["widen"])
                        cand = (ok[idx].mean(), -cl[idx].sum())
                        if best is None or cand > best[0]:
                            best = (cand, float(t_m), float(t_s))
                th[m] = (best[1], best[2])
                tdev.append(abs(best[1] - TH[m]["t_m"]))
            tot = sum(float(np.sum(test_vec(c, *th[c.model])[0] - c.keep)) for c in cells)
            gains.append(100 * tot / N)
        g = np.array(gains)
        lo, hi = (np.percentile(g, 2.5), np.percentile(g, 97.5)) if len(g) > 1 else (g[0], g[0])
        pc = int(round(100 * frac))
        F(f"RPartDevSize{WORD[pc]}", f"t_m and t_s refit per model on {pc}% of the development samples ({reps} draw{'s' if reps > 1 else ''}): "
          + ("mean test change (2.5 and 97.5 percentiles over draws)" if reps > 1 else "test change"), sg(float(g.mean())),
          sg(lo) if reps > 1 else None, sg(hi) if reps > 1 else None, fn="v6_phase2.u4")
        F(f"RPartDevSize{WORD[pc]}TmDev", f"{pc}% of development: mean absolute distance of the refit t_m from the frozen t_m",
          f"{float(np.mean(tdev)):.2f}", fn="v6_phase2.u4")
        size_rows.append((frac, reps, float(g.mean()), lo, hi, float(np.mean(tdev))))
    wcsv("c11_devsize", ["dev_fraction", "draws", "mean_test_change", "p2_5", "p97_5", "mean_abs_tm_shift"], size_rows)
    # heat map data: development and test accuracy against t_m on the 0.05 grid (frozen t_s, widen action, check)
    hm = []
    for m in MODELS:
        cs = [c for c in cells if c.model == m]
        nm_ = sum(c.n for c in cs)
        for t in fine:
            da = route_dev(dev[m], t, TH[m]["t_s"], TH[m]["widen"])[0].mean()
            ta = sum(float(np.sum(test_vec(c, t, TH[m]["t_s"])[0])) for c in cs) / nm_
            hm.append((m, float(t), 100 * da, 100 * ta, float(abs(t - TH[m]["t_m"]) < 1e-9)))
    wcsv("c11_threshold_heatmap", ["model", "t_m", "dev_accuracy", "test_accuracy", "frozen"], hm)
    body = []
    for lab, da, e, lo, hi, vs, cl, thr in rows:
        body.append(f"{lab} & {da:.2f} & {tex_s(e)} {tci(lo, hi)} & "
                    + ("ref." if vs is None else f"{tex_s(vs[0])} {tci(vs[1], vs[2])}") + f" & {cl:.2f} \\\\")
    for frac, reps, gm, lo, hi, sh in size_rows:
        body.append(f"$t_m$, $t_s$ refit on {int(round(100 * frac))}\\% of dev. ({reps} draw{'s' if reps > 1 else ''}) & n/a & "
                    f"{tex_s(gm)}{(' ' + tci(lo, hi)) if reps > 1 else ''} & n/a & n/a \\\\")
    write_tab("routerparts", "lrrrr", r"Selection rule & Dev. acc. & Test change [95\% CI] & $-$Frozen [95\% CI] & Calls \\", body,
              note_="U4 (exploratory). Thresholds refit on development with each model's frozen widen action and check; "
              "dev-size rows give the mean over draws with 2.5 and 97.5 percentiles.")


# ========================================================================================= U5 coverage signals
def u5():
    st = V.STRUCT
    w_main = {id(c): c.routed for c in cells}

    def logistic_routes():
        """Cross-fitted (split groups, 2 folds x 5 repeats) P(error) from struct flag, declined flag and margin; routes
        Map-Reflect's per-model share by it (ties share the boundary)."""
        from sklearn.linear_model import LogisticRegression
        out = {}
        for m in MODELS:
            cs = [c for c in cells if c.model == m]
            X = np.concatenate([np.column_stack([st(c).astype(float), (~c.valid).astype(float), c.margin]) for c in cs])
            y = np.concatenate([1 - c.keep for c in cs])
            units = np.array([f"{c.ds}|{GB.unit(c.ds, s)}" for c in cs for s in c.sid])
            uu = {u: i for i, u in enumerate(sorted(set(units)))}
            ui = np.array([uu[u] for u in units])
            rng = np.random.default_rng(7)
            p = np.zeros(len(y))
            for _ in range(5):
                fold = rng.integers(0, 2, len(uu))[ui]
                for k in range(2):
                    tr, te = fold != k, fold == k
                    p[te] += LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] / 5
            w = U.gate_weights(-p, SHARE_M[m] * len(y))
            i = 0
            for c in cs:
                out[id(c)] = w[i:i + c.n]
                i += c.n
        return out
    wl = logistic_routes()
    R = {
        "Struct": ("Schema-size baseline", lambda c: st(c).astype(float), "widen"),
        "Margin": ("Map-Reflect", lambda c: c.routed, "action"),
        "Union": ("Schema size or margin", lambda c: ((c.routed > 0) | st(c)).astype(float), "union"),
        "Inter": ("Schema size and margin", lambda c: ((c.routed > 0) & st(c)).astype(float), "action"),
        "Logit": ("logistic (cross-fitted)", lambda c: wl[id(c)], "action"),
    }
    rows = []
    for tg, (lab, wf, act) in R.items():
        def gain(c, wf=wf, act=act):
            w = wf(c)
            if act == "widen":
                return w * (c.widen - c.keep)
            if act == "union":
                return np.where(c.routed > 0, c.action, np.where(st(c), c.widen, c.keep)) - c.keep
            return w * (c.action - c.keep)

        def cost(c, wf=wf, act=act):
            w = wf(c)
            T = call_tok_v4(c)
            if act == "widen":
                return w * T["widen"][0], w * T["widen"][1]
            if act == "union":
                s_ = st(c) & ~(c.routed > 0)
                return (np.where(c.routed > 0, T["action"][0], np.where(s_, T["widen"][0], 0.0)),
                        np.where(c.routed > 0, T["action"][1], np.where(s_, T["widen"][1], 0.0)))
            return w * T["action"][0], w * T["action"][1]
        e, d = pooled(cells, gain)
        lo, hi = GS.interval(d)
        dd, ddr = pooled(cells, lambda c, g=gain: g(c) - POL["Method"](c))
        dlo, dhi = GS.interval(ddr)
        cl = sum(float(np.sum(cost(c)[0])) for c in cells) / N
        tk = sum(float(np.sum(cost(c)[1])) for c in cells) / N
        rt = 100 * sum(float(np.sum(wf(c))) for c in cells) / N
        F(f"Cov{tg}Gain", f"coverage router '{lab}': change over the first pass", sg(e), sg(lo), sg(hi), fn="v6_phase2.u5")
        if tg != "Margin":
            F(f"Cov{tg}VsMain", f"coverage router '{lab}' minus Map-Reflect", sg(dd), sg(dlo), sg(dhi), fn="v6_phase2.u5")
        F(f"Cov{tg}Routed", f"coverage router '{lab}': share of fields routed, %", f"{rt:.1f}", fn="v6_phase2.u5")
        F(f"Cov{tg}Calls", f"coverage router '{lab}': calls per field", f"{cl:.2f}", fn="v6_phase2.u5")
        F(f"Cov{tg}Tokens", f"coverage router '{lab}': tokens per field", fmt_int(tk), fn="v6_phase2.u5")
        rows.append((lab, rt, e, lo, hi, None if tg == "Margin" else (dd, dlo, dhi), cl, tk))
    body = [f"{'Margin-Reflect' if lab == 'Map-Reflect' else lab} & {rt:.1f} & {tex_s(e)} {tci(lo, hi)} & " + ("ref." if vs is None else f"{tex_s(vs[0])} {tci(vs[1], vs[2])}")
            + f" & {cl:.2f} & {tnum(tk)} \\\\" for lab, rt, e, lo, hi, vs, cl, tk in rows]
    write_tab("coverage", "lrrrrr", r"Selection & Reflection rate (\%) & Change [95\% CI] & $-$Margin-Reflect [95\% CI] & Calls & Tokens \\", body,
              note_="U5 (exploratory). Struct routes to the widen action; the others use the frozen action of each routed "
              "field. The logistic routes Map-Reflect's per-model share.")


# ========================================================================================= U6 acceptance rules
U6ROWS = []


def u6():
    """Map-Reflect's routed fields and actions; the proposal is accepted with no check, by the pairwise check (v4), by
    the reflection's own margin (accept when the revised margin exceeds the first margin, as v4_fit's check+margin),
    or by both. Near-tie (NT) proposals carry no revised margin, so there the margin rule and 'both' fall back to the
    check."""
    vec = defaultdict(dict)
    nt_routed = 0
    for c in cells:
        rs = rows_of(c)
        base = c.keep
        out = {k: np.array(base, float) for k in ("none", "check", "rmargin", "both")}
        chg = {k: np.zeros(c.n) for k in out}
        calls = {k: np.zeros(c.n) for k in out}
        for i, r in enumerate(rs):
            if not c.routed[i]:
                continue
            a = c.act[i]
            prop = r.get(f"{a}_proposed") or r["base"] or ""
            changed = bool(prop) and prop != (r["base"] or "")
            fin = r.get(a)
            rm = r.get(f"{a}_rmargin")
            ok_p = float(target_correct(prop, r["_gold"]))
            ok_c = float(target_correct(fin, r["_gold"]))
            grow = (rm is not None) and float(rm) > float(r["margin"])
            out["none"][i] = ok_p
            out["check"][i] = ok_c
            if rm is None:
                nt_routed += 1
                out["rmargin"][i] = ok_c
                out["both"][i] = ok_c
            else:
                out["rmargin"][i] = ok_p if (changed and grow) else base[i] if changed else ok_p
                out["both"][i] = ok_c if grow else base[i] if changed else ok_c
            chk_call = changed and bool(r["base"])  # a check call follows a changed answer (none for a declined field)
            for k in out:
                calls[k][i] = 1 + (chk_call if (k in ("check", "both") or (rm is None and k == "rmargin")) else 0)
        for k in out:
            vec[k][id(c)] = (out[k], calls[k])
    assert all(np.array_equal(vec["check"][id(c)][0], c.method) for c in cells)
    assert all(np.allclose(vec["check"][id(c)][1], c.routed * (1 + c.ver_act)) for c in cells)  # Table 1's calls
    LAB = {"none": "no check (accept every proposal)", "check": "pairwise check (Margin-Reflect)",
           "rmargin": "reflection's own margin grows", "both": "check and own margin"}
    TG = {"none": "None", "check": "Check", "rmargin": "RMargin", "both": "Both"}
    F("AcceptNTRouted", "routed field-model pairs with the near-tie action (no revised margin stored)", f"{nt_routed:,}", fn="v6_phase2.u6")
    for k in ("none", "check", "rmargin", "both"):
        e, d = pooled(cells, lambda c: vec[k][id(c)][0] - c.keep)
        lo, hi = GS.interval(d)
        dd, ddr = pooled(cells, lambda c: vec[k][id(c)][0] - c.method)
        dlo, dhi = GS.interval(ddr)
        rep = int(sum(((c.keep == 0) & (vec[k][id(c)][0] == 1)).sum() for c in cells))
        brk = int(sum(((c.keep == 1) & (vec[k][id(c)][0] == 0)).sum() for c in cells))
        cl = sum(float(vec[k][id(c)][1].sum()) for c in cells) / N
        F(f"Accept{TG[k]}Gain", f"acceptance rule '{LAB[k]}' (Map-Reflect's routing): change over the first pass", sg(e), sg(lo), sg(hi),
          fn="v6_phase2.u6")
        if k != "check":
            F(f"Accept{TG[k]}VsMain", f"acceptance rule '{LAB[k]}' minus the check", sg(dd), sg(dlo), sg(dhi), fn="v6_phase2.u6")
        F(f"Accept{TG[k]}Repairs", f"acceptance rule '{LAB[k]}': repairs / breaks", f"{rep:,} / {brk:,}", fn="v6_phase2.u6")
        F(f"Accept{TG[k]}Calls", f"acceptance rule '{LAB[k]}': calls per field", f"{cl:.2f}", fn="v6_phase2.u6")
        U6ROWS.append((LAB[k], e, lo, hi, None if k == "check" else (dd, dlo, dhi), rep, brk, cl))
    body = [f"{lab} & {tex_s(e)} {tci(lo, hi)} & " + ("ref." if vs is None else f"{tex_s(vs[0])} {tci(vs[1], vs[2])}")
            + f" & {tnum(rep)} & {tnum(brk)} & {cl:.2f} \\\\" for lab, e, lo, hi, vs, rep, brk, cl in U6ROWS]
    write_tab("accept", "lrrrrr", r"Acceptance rule & Change [95\% CI] & $-$Check [95\% CI] & Repairs & Breaks & Calls \\", body,
              note_="U6 (exploratory). Map-Reflect's routing and actions; only the rule that accepts a proposal changes.")
    rows = [("intervention", r["name"], r["rep"], r["brk"]) for r in X1ROWS + X3ROWS]
    rows += [("acceptance", lab, rep, brk) for lab, e, lo, hi, vs, rep, brk, cl in U6ROWS]
    wcsv("c9_repairs_breaks", ["group", "label", "repairs", "breaks"], rows)


# ========================================================================================= U8 log-probabilities
def u8():
    ms = uqmods("lp_answer")
    lpmin = {}
    for ds in DATASETS:
        for m in ms:
            for f in (ROOT / f"results/naacl/uq_ablation/logprob/{ds}__{m}.jsonl", ROOT / f"results/naacl/uq_ablation/logprob/{ds}__{m}.fill.jsonl",
                      ROOT / f"results/naacl/v6/u9/logprob/{ds}__{m}.jsonl"):
                if f.exists():
                    for r in read(f):
                        if r.get("lp_answer_min") is not None:
                            lpmin[(m, r["key"])] = float(r["lp_answer_min"])
    miss = 0
    for m in ms:
        for f in L["pop"][m]:
            v = lpmin.get((m, f["sig"]["_key"]))
            if v is None:
                miss += 1
                v = f["sig"]["lp_answer"]
            f["sig"]["lp_answer_min"] = v
    F("LogpMinMissing", "fields of the log-probability models without a stored minimum token log-probability (sum used)", f"{miss:,}",
      fn="v6_phase2.u8")
    LAB = {"lp_answer": "candidate tokens, sum", "lp_answer_min": "candidate tokens, min.",
           "lp_mean": "all tokens, mean", "lp_entropy": "all tokens, entropy", "margin": "margin (same fields)"}
    TG = {"lp_answer": "Sum", "lp_answer_min": "Min", "lp_mean": "Mean", "lp_entropy": "Ent", "margin": "Margin"}
    sigs = ["margin", "lp_answer", "lp_answer_min", "lp_mean", "lp_entropy"]
    have = [s for s in sigs if all(s in f["sig"] for m in ms for f in L["pop"][m])]
    res, n = L["group"](ms, have)
    body = []
    for s in have:
        a, lo, hi = auc_ci(ms, s)
        sa, _, _ = auc_ci(ms, s, filt=lambda f: f["lp_agree"])
        acc, d, (dlo, dhi) = res[s]
        F(f"Logp{TG[s]}AUC", f"log-probability variant '{LAB[s]}': pooled error AUROC ({mname(ms)})", f"{a:.3f}", f"{lo:.3f}", f"{hi:.3f}",
          fn="v6_phase2.u8")
        F(f"Logp{TG[s]}SameAUC", f"log-probability variant '{LAB[s]}': error AUROC on fields whose regenerated answer is the scored one",
          f"{sa:.3f}", fn="v6_phase2.u8")
        if s != "margin":
            F(f"Logp{TG[s]}Delta", f"log-probability variant '{LAB[s]}': routing at v4's share minus the margin", sg(d), sg(dlo), sg(dhi),
              fn="v6_phase2.u8")
        body.append(f"{LAB[s]} & {a:.3f} [{lo:.3f}, {hi:.3f}] & {sa:.3f} & {acc:.2f} & "
                    + ("ref." if s == "margin" else f"{tex_s(d)} {tci(dlo, dhi)}") + r" \\")
    write_tab("logprob", "lrrrr", r"UQ method & Test AUROC [95\% CI] & Same answer & Accuracy & $\Delta$ vs margin [95\% CI] \\",
              body, note_=f"U8 (exploratory). {mname(ms)}; aggregates stored per field (per-token values were not stored).")


# ==================================================================================== U10 margin calibration
def u10():
    dev = dev_rows()
    rows, rel = [], []
    for m in MODELS:
        D = dev[m]
        va = D["valid"]
        iso_m = iso_fit(D["m"][va], D["b"][va])
        iso_s = iso_fit(D["s1"][va], D["b"][va])
        rs = masked([c for c in cells if c.model == m], lambda c: c.valid)
        mg = np.concatenate([c.margin for c in rs])
        s1 = np.concatenate([c.s1 for c in rs])
        y = np.concatenate([c.keep for c in rs])
        out = {}
        for nm, p in (("TopRaw", s1), ("MarginRaw", np.clip(mg, 0, 1)), ("MarginIso", iso_m.predict(mg)), ("TopIso", iso_s.predict(s1))):
            ece, br = ece_brier(p, y)
            out[nm] = (ece, br)
            F(f"MCal{nm}ECE{MT[m]}", f"{MSHORT[m]}: ECE of {nm.replace('Raw', ' as a probability').replace('Iso', ' after isotonic calibration on development').replace('Top', 'the top score').replace('Margin', 'the margin')} (answered test fields)",
              f"{ece:.3f}", fn="v6_phase2.u10")
            F(f"MCal{nm}Brier{MT[m]}", f"{MSHORT[m]}: Brier score of {nm} (answered test fields)", f"{br:.3f}", fn="v6_phase2.u10")
            b = np.minimum((np.clip(p, 0, 1) * 10).astype(int), 9)
            for k in range(10):
                if (b == k).any():
                    rel.append((m, nm, k, int((b == k).sum()), float(p[b == k].mean()), float(y[b == k].mean())))
        rows.append((m, len(y), out))
    wcsv("c5_reliability", ["model", "score", "bin", "fields", "mean_predicted", "accuracy"], rel)
    body = [f"{MSHORT[m]} & {tnum(n)} & " + " & ".join(f"{out[k][0]:.3f} & {out[k][1]:.3f}" for k in ("TopRaw", "MarginRaw", "MarginIso", "TopIso"))
            + r" \\" for m, n, out in rows]
    write_tab("margincal", "lrrrrrrrrr", r" & & \multicolumn{2}{c}{Top score} & \multicolumn{2}{c}{Margin, raw} & \multicolumn{2}{c}{Margin, isotonic} & "
              r"\multicolumn{2}{c}{Top score, isotonic} \\ Model & Answered & ECE & Brier & ECE & Brier & ECE & Brier & ECE & Brier \\",
              body, note_="U10 (exploratory). Isotonic maps fitted per model on the answered v4 development fields.")


# =========================================================================================== chart data
def chart_data():
    # C2: change against routed share for each intervention, margin vs random (pooled, per-model shares)
    rows = []
    groups = {"Doubt": "harmful", "SelfRefine": "harmful", "NeutralEvid": "harmful", "Neutral": "nothing to repair",
              "Reflect": "nothing to repair", "WidenReflect": "repairing", "RoutedAction": "repairing"}
    for tg, reg in groups.items():
        rs = masked(cells, lambda c, tg=tg: ~np.isnan(c.x1[tg]))
        n_ = sum(c.n for c in rs)
        for b in SHARES:
            w = per_model_weights(rs, lambda c: c.margin, b)
            em = 100 * sum(float(np.sum(w[id(c)] * (c.x1[tg] - c.keep))) for c in rs) / n_
            er = 100 * sum(float(np.sum(b * (c.x1[tg] - c.keep))) for c in rs) / n_
            rows.append((tg, reg, b, em, er))
    for b in SHARES:  # escalation
        w = per_model_weights(cells, lambda c: c.margin, b)
        em = 100 * sum(float(np.sum(w[id(c)] * (c.esc - c.keep))) for c in cells) / N
        er = 100 * sum(float(np.sum(b * (c.esc - c.keep))) for c in cells) / N
        rows.append(("Escalate", "repairing", b, em, er))
    wcsv("c2_interventions", ["intervention", "regime", "share", "change_margin", "change_random"], rows)
    # C3: accuracy against calls per model (margin threshold swept, random, every, struct, Map-Reflect, oracle)
    rows = []
    for m in MODELS:
        cs = [c for c in cells if c.model == m]
        n_ = sum(c.n for c in cs)
        f0 = 100 * sum(c.keep.sum() for c in cs) / n_
        ev = sum(float(np.sum(1 + c.ver_act)) for c in cs) / n_
        dv = 100 * sum(float(np.sum(c.action - c.keep)) for c in cs) / n_
        for s in SHARES:
            rows.append((m, "random", s * ev, f0 + s * dv))
        for dl in np.round(np.arange(-0.5, 0.55, 0.05), 2):
            a = cl = 0.0
            for c in cs:
                t = TH[m]
                rt = router.routed(c.margin, c.valid, c.s1, max(0.0, t["t_m"] + dl), t["t_s"])
                a += float(np.sum(np.where(rt, c.action, c.keep)))
                cl += float(np.sum(rt * (1 + c.ver_act)))
            rows.append((m, "margin sweep", cl / n_, 100 * a / n_))
        st = V.STRUCT
        rows.append((m, "struct", sum(float(np.sum(st(c) * (1 + c.ver_w))) for c in cs) / n_,
                     f0 + 100 * sum(float(np.sum(st(c) * (c.widen - c.keep))) for c in cs) / n_))
        rows.append((m, "every", ev, f0 + dv))
        rows.append((m, "Map-Reflect", sum(float(np.sum(c.routed * (1 + c.ver_act))) for c in cs) / n_, 100 * sum(c.method.sum() for c in cs) / n_))
        rows.append((m, "keep", 0.0, f0))
    wcsv("c3_accuracy_calls", ["model", "series", "calls_per_field", "accuracy"], rows)
    # C4: change by family for each router, with intervals
    rows = []
    st = V.STRUCT
    RT = {"Map-Reflect": POL["Method"], "random": POL["Random"], "struct": lambda c: st(c) * (c.widen - c.keep),
          "every": POL["ReflectAllAction"], "oracle": POL["Oracle"]}
    for fam, dss in FAMILY:
        rs = [c for c in cells if c.ds in dss]
        for nm, f in RT.items():
            e, d = pooled(rs, f)
            lo, hi = GS.interval(d)
            rows.append((fam, nm, e, lo, hi))
    wcsv("c4_family_gain", ["family", "router", "change", "lo", "hi"], rows)
    # C6: margin histograms, correct vs wrong first answers (answered test fields)
    rows = []
    edges = np.round(np.arange(0, 1.0001, 0.05), 2)
    for m in MODELS:
        rs = masked([c for c in cells if c.model == m], lambda c: c.valid)
        mg = np.concatenate([c.margin for c in rs])
        y = np.concatenate([c.keep for c in rs])
        for lab, sel in (("correct", y == 1), ("wrong", y == 0)):
            h, _ = np.histogram(np.clip(mg[sel], 0, 1), bins=edges)
            for lo_, k in zip(edges[:-1], h):
                rows.append((m, lab, float(lo_), int(k), float(k / max(sel.sum(), 1))))
    wcsv("c6_margin_hist", ["model", "first_answer", "bin_low", "fields", "share"], rows)
    # C10: where the gold lies for routed first-pass errors, by structural stratum (X6)
    wcsv("c10_gold_location", ["stratum", "gold", "pairs", "errors", "routed_errors", "repairs", "breaks"],
         [(s, g, nf, err, rerr, rep, brk) for s, g, nf, err, rerr, rep, brk, *_ in X6TAB])
    # C12: detection AUROC against routing delta, per signal x model
    rows = []
    for s in SIGORDER:
        if s == "margin":
            continue
        for m in uqmods(s):
            res, _ = L["group"]([m], ["margin", s])
            fs = L["pop"][m]
            err = 1 - np.array([f["base"] for f in fs], float)
            a = U.auroc_w(np.array([f["sig"][s] for f in fs], float), err)
            am = U.auroc_w(np.array([f["sig"]["margin"] for f in fs], float), err)
            rows.append((s, m, a, am, res[s][1], res[s][2][0], res[s][2][1]))
    wcsv("c12_auroc_vs_routing", ["signal", "model", "auroc", "margin_auroc", "routing_delta", "lo", "hi"], rows)
    am = np.array([r[2] - r[3] for r in rows])
    dl = np.array([r[4] for r in rows])
    F("DetVsRouteCorr", "signal x model points: Pearson correlation of (AUROC minus the margin's) with the routing delta",
      f"{float(np.corrcoef(am, dl)[0, 1]):.2f}", fn="v6_phase2.chart_data")
    F("DetVsRoutePoints", "signal x model points in that comparison", str(len(rows)), fn="v6_phase2.chart_data")
    # C13: held-out change per dataset with intervals
    sets = sorted({mm.group(1) for k in MHO for mm in [re.fullmatch(r"HO(\w+?)Fields", k)] if mm and mm.group(1) not in ("All", "Magellan")})
    wcsv("c13_heldout", ["set", "change", "lo", "hi", "vs_random", "vs_random_lo", "vs_random_hi"],
         [(s, float(MHO[f"HO{s}VThree"]), float(MHO[f"HO{s}VThreeLo"]), float(MHO[f"HO{s}VThreeHi"]),
           float(MHO[f"HO{s}VMinusRandom"]), float(MHO[f"HO{s}VMinusRandomLo"]), float(MHO[f"HO{s}VMinusRandomHi"])) for s in sets + ["All"]])
    if V5:
        v5_chart_data()


# ============================================================================================ R5 human labels
H1 = Path("<DATASETS_REPO>/data/itopsgraph_v2_1/human_review")


def r5():
    """ITOpsGraph rescored on the h1 human-agreed labels next to v2.1 on the same fields, and the h1 summary facts.
    Skips cleanly while the label file is absent."""
    lab = H1 / "itopsgraph_v2.1_h1_human.jsonl"
    if not lab.exists():
        note("ITOpsHumanStatus", "R5 human-agreed ITOpsGraph labels", f"not available yet ({lab} absent); R5 skipped")
        return False
    from eval.calc_accuracy import load_gold_file
    HG = load_gold_file(str(lab))
    summ = json.loads((H1 / "h1_summary.json").read_text()) if (H1 / "h1_summary.json").exists() else {}

    def flat(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v, pre + k + "_")
            elif isinstance(v, (int, float, str)) and not isinstance(v, bool):
                yield pre + k, v
    for k, v in flat(summ):
        tg = "".join(w.capitalize() for w in re.split(r"[^A-Za-z]+", k) if w)
        if tg:
            F(f"ITOpsHuman{tg}", f"h1 summary: {k} (h1_summary.json)", f"{v:.3f}" if isinstance(v, float) else str(v),
              src=(str(H1 / "h1_summary.json"), "h1_rescore_inputs.py"), fn="v6_phase2.r5")
    ic = [c for c in cells if c.ds == "itopsgraph"]
    res = {}
    for c in ic:
        rs = rows_of(c)
        keep_h, meth_h, keep_v, meth_v, scored = [], [], [], [], []
        for r in rs:
            sid, fld = str(r["sample_id"]), r["key"].rsplit("||", 1)[1]
            tg, rm, _ = HG.get(sid, ({}, {}, None))
            ok = fld in tg and fld not in rm
            scored.append(ok)
            if ok:
                keep_h.append(float(target_correct(r["base"], tg[fld])))
                meth_h.append(float(target_correct(r["v4"], tg[fld])))
                keep_v.append(float(target_correct(r["base"], r["_gold"])))
                meth_v.append(float(target_correct(r["v4"], r["_gold"])))
        res[c.model] = tuple(np.array(x) for x in (keep_h, meth_h, keep_v, meth_v)) + (int(np.sum(scored)), c.n)
    kh, mh, kv, mv = (np.concatenate([res[m][i] for m in res]) for i in range(4))
    nsc = sum(res[m][4] for m in res)
    F("ITOpsHumanFields", "ITOpsGraph test field-model pairs with an agreed human label", f"{nsc:,} of {sum(res[m][5] for m in res):,}", fn="v6_phase2.r5")
    for tg, a, b in (("V21", kv, mv), ("Human", kh, mh)):
        F(f"ITOpsHumanSame{tg}First", f"same fields, {'v2.1' if tg == 'V21' else 'human'} labels: first-pass accuracy, %", f"{100 * a.mean():.2f}", fn="v6_phase2.r5")
        F(f"ITOpsHumanSame{tg}Gain", f"same fields, {'v2.1' if tg == 'V21' else 'human'} labels: Map-Reflect change, points", sg(100 * (b - a).mean()), fn="v6_phase2.r5")
    rows = []
    for f_ in ("h1_c15_fields.csv", "h1_c15_summary.csv"):
        if (H1 / f_).exists():
            (DATA / f_.replace("h1_", "c15_")).write_text((H1 / f_).read_text())
    return True


# ======================================================================== ITOpsGraph package facts (F-ITOps*)
DSREPO = Path("<DATASETS_REPO>")
PKG = "ff79d3e"  # review/h1-tooling: the v6 paper package (read with git show; the checkout's branch is never switched)
ITOPS_FLAGS = []


def gshow(path):
    import subprocess
    return subprocess.run(["git", "-C", str(DSREPO), "show", f"{PKG}:{path}"], capture_output=True, text=True, check=True).stdout


def itops_facts():
    """Descriptive ITOpsGraph v2.1 facts, each read from the package files at schema-mapping-datasets@b201b13 (or the
    files they name) and cross-checked where two sources state the same number."""
    D = "descriptive"
    st = json.loads(gshow("docs/itopsgraph/package_v6/itopsgraph_v21_stats.json"))
    agr = json.loads(gshow("data/itopsgraph_v2_1/review/agreement.json"))
    adj = [l for l in gshow("data/itopsgraph_v2_1/review/adjudicated.jsonl").splitlines() if l.strip()]
    chg = [json.loads(l) for l in gshow("data/itopsgraph_v2_1/change_log.jsonl").splitlines() if l.strip()]
    h1m = json.loads(gshow("data/itopsgraph_v2_1/human_review/h1_manifest.json"))
    h1p = json.loads(gshow("data/itopsgraph_v2_1/human_review/h1_priority.json"))
    rows = [json.loads(l) for l in gshow("data/itopsgraph_v2_1/itopsgraph_v2.1_cross_product.jsonl").splitlines() if l.strip()]
    pool = json.load(open(ROOT / "data/itopsgraph_api_lookup.json"))
    man = json.load(open(ROOT / "data/splits/itopsgraph.json"))
    SJ = (f"schema-mapping-datasets@{PKG} docs/itopsgraph/package_v6/itopsgraph_v21_stats.json", "paper_package_stats.py")
    a, dv, te = st["all"], st["dev"], st["test"]
    # cross-checks against the files the statistics were computed from
    test_ids = set(man["test"]) - set(man.get("old_dev_excluded") or [])
    assert te["samples"] == len(test_ids) and dv["samples"] == len(man["dev"]) and a["samples"] == len(rows)
    assert te["scored"] == sum(c.n for c in cells if c.ds == "itopsgraph") // len(MODELS), "test scored differs from Table data"
    assert st["pool"]["fields"] == len(pool)
    pc = {}
    for v in pool.values():
        pc.setdefault({"base": "prod_f", "edge": "prod_f", "model": "prod_f", "resource": "prod_f",
                       "prod_h": "prod_h_tf"}.get(v["product_name"], v["product_name"]), []).append(v)
    assert len(pc) == st["pool"]["products"]
    stub = len(pc.get("azure", [])) + len(pc.get("gcp", []))
    F("ITOpsSamples", "ITOpsGraph v2.1 cross-product samples (one source type, one target type of another product, a "
      "one-to-one field mapping)", f"{a['samples']}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsSamplesManifest", "ITOpsGraph samples in the protocol v2 split manifest (development + test)",
      f"{st['in_manifest']['samples']}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsSamplesDev", "ITOpsGraph development samples", f"{dv['samples']}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsSamplesTest", "ITOpsGraph test samples (manifest test, which excludes the 70 an earlier protocol used for "
      "development)", f"{te['samples']}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsTestScored", "ITOpsGraph scored test gold fields (gold fields not in removed_fields; equals the per-model test "
      "fields of the paper)", f"{te['scored']}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsTestScoredSamples", "ITOpsGraph test samples with at least one scored field", f"{te['samples_with_scored']}", src=SJ,
      status=D, fn="v6_phase2.itops_facts")
    F("ITOpsDevScored", "ITOpsGraph scored development gold fields", f"{dv['scored']}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsDevScoredSamples", "ITOpsGraph development samples with at least one scored field", f"{dv['samples_with_scored']}",
      src=SJ, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsGoldAll", "ITOpsGraph scored gold fields, all 350 samples", f"{a['scored']}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsGoldUnscored", "ITOpsGraph gold fields removed from scoring by the v2.1 rules, all samples", f"{a['removed']}", src=SJ,
      status=D, fn="v6_phase2.itops_facts")
    F("ITOpsGoldTotal", "ITOpsGraph v2 gold mappings before the v2.1 relabel, all samples", f"{a['gold_v2']}", src=SJ, status=D,
      fn="v6_phase2.itops_facts")
    assert a["scored"] + a["removed"] == a["gold_v2"]
    F("ITOpsProducts", "products in the ITOpsGraph samples (packages merged: prod_f with its base, edge, model and "
      "resource packages; prod_h with prod_h_tf)", f"{a['n_products']}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsPoolPrefixes", "product prefixes in the ITOpsGraph candidate pool (packages merged), including Azure and GCP stubs",
      f"{st['pool']['products']}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsPoolStubFields", "fields of the Azure and GCP stubs in the pool", f"{stub}", src=("data/itopsgraph_api_lookup.json",
      "v6_phase2.itops_facts"), status=D, fn="v6_phase2.itops_facts")
    F("ITOpsTypesSource", "ITOpsGraph source types, all samples", f"{a['source_types']}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsTypesTarget", "ITOpsGraph target types, all samples", f"{a['target_types']}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsPoolTypes", "types in the ITOpsGraph candidate pool", f"{st['pool']['types']:,}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsPoolFields", "fields in the ITOpsGraph candidate pool", f"{st['pool']['fields']:,}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsPoolDescribed", "pool fields with a non-empty schema comment, %",
      f"{100 * st['pool']['fields_described'] / st['pool']['fields']:.0f}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    for tg, k, cl in (("ExactTest", "share_exact_name", "test scored gold with a same-named target (lower case, letters and digits)"),
                      ("NoTokenTest", "share_no_shared_token", "test scored gold sharing no word with any gold target"),
                      ("NestedTest", "share_scored_in_nested_pairs", "test scored fields in a pair with a nested type (a type "
                       "generated as a path through an API operation's input or response, or an embedded protobuf Base message)"),
                      ("DescribedTestSource", "share_source_described", "test scored source fields with a schema comment"),
                      ("DescribedTestTarget", "share_target_described", "test scored gold target fields with a schema comment")):
        F(f"ITOps{tg}", f"ITOpsGraph: {cl}, %", f"{100 * te[k]:.1f}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsManyToOneTest", "ITOpsGraph test scored fields whose target two or more scored source fields of one sample share, % "
      "(fields)", f"{100 * te['share_many_to_one']:.1f} ({te['many_to_one_fields']})", src=SJ, status=D, fn="v6_phase2.itops_facts")
    pt = te["paradigm"]
    cross = pt.get("protobuf to GraphQL", 0) + pt.get("GraphQL to protobuf", 0)
    F("ITOpsParadigmTest", "ITOpsGraph test samples: GraphQL to GraphQL / across the two paradigms / protobuf to protobuf",
      f"{pt['GraphQL to GraphQL']} / {cross} / {pt['protobuf to protobuf']}", src=SJ, status=D, fn="v6_phase2.itops_facts")
    SA = (f"schema-mapping-datasets@{PKG} data/itopsgraph_v2_1/review/agreement.json", "build_itopsgraph_v2_1.py")
    nconf = sum(agr["confusion"].values())
    F("ITOpsKappaWeighted", "Cohen's kappa of the two blind LLM labelling passes, four-way decision, weighted by samples",
      f"{agr['decision_kappa_sample_weighted']:.3f}", src=SA, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsKappaUnits", "review units (duplicate groups) the two labelling passes covered", f"{agr['units']}", src=SA, status=D,
      fn="v6_phase2.itops_facts")
    F("ITOpsKappaAgreement", "raw agreement of the two passes on the decision, %", f"{100 * agr['decision_agreement']:.1f}", src=SA,
      status=D, fn="v6_phase2.itops_facts")
    if nconf != agr["units"]:
        ITOPS_FLAGS.append(f"F-ITOpsKappa: the package calls 0.926 'per unit', but the confusion counts behind it sum to "
                           f"{nconf} mappings, not {agr['units']} units; the fact says 'over {nconf} mappings'")
    F("ITOpsAdjudicated", "mappings where the two passes disagreed (the off-diagonal counts over the 492 mappings), decided by a third "
      "pass of the same LLM under the same rules, with no human check", f"{len(adj)}", src=(f"schema-mapping-datasets@{PKG} data/itopsgraph_v2_1/review/adjudicated.jsonl",
      "build_itopsgraph_v2_1.py"), status=D, fn="v6_phase2.itops_facts")
    assert len(adj) == agr["disagreements"]
    from collections import Counter
    cc = Counter(r["decision"] for r in chg)
    SC = (f"schema-mapping-datasets@{PKG} data/itopsgraph_v2_1/change_log.jsonl", "build_itopsgraph_v2_1.py")
    F("ITOpsChangedRemoved", "v2.1 change log: gold fields removed from scoring", f"{cc['remove']}", src=SC, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsChangedCorrected", "v2.1 change log: gold targets corrected", f"{cc['corrected']}", src=SC, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsChangedWidened", "v2.1 change log: gold widened to a set of acceptable targets", f"{cc['set']}", src=SC, status=D,
      fn="v6_phase2.itops_facts")
    F("ITOpsChangedRows", "v2.1 change log rows", f"{len(chg)}", src=SC, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsChangedAdjudicated", "change-log rows that come from adjudicated units",
      f"{sum(r.get('resolved') == 'adjudicated' for r in chg)}", src=SC, status=D, fn="v6_phase2.itops_facts")
    assert cc["remove"] == a["removed"]
    SH = (f"schema-mapping-datasets@{PKG}:data/itopsgraph_v2_1/human_review/h1_manifest.json, "
          f"schema-mapping-datasets@{PKG}:data/itopsgraph_v2_1/human_review/h1_priority.json", "h1 tooling")
    tiers = {}
    for it in h1p["items"]:
        x = tiers.setdefault(it["tier"], [0, 0])
        x[0] += 1
        x[1] += it["fields_to_judge"]
    assert sum(x[0] for x in tiers.values()) == h1m["items"] and sum(x[1] for x in tiers.values()) == h1m["field_judgements"]
    F("ITOpsHOneItems", "h1 human review: items (one per v2.1 duplicate group with a scored mapping)", f"{h1m['items']}", src=SH, status=D,
      fn="v6_phase2.itops_facts")
    F("ITOpsHOneJudgements", "h1 human review: field judgements", f"{h1m['field_judgements']}", src=SH, status=D, fn="v6_phase2.itops_facts")
    F("ITOpsHOneTestItems", "h1 human review: test items (tiers 1 and 2)", f"{tiers[1][0] + tiers[2][0]}", src=SH, status=D,
      fn="v6_phase2.itops_facts")
    F("ITOpsHOneTestJudgements", "h1 human review: test field judgements (tiers 1 and 2)", f"{tiers[1][1] + tiers[2][1]}", src=SH,
      status=D, fn="v6_phase2.itops_facts")
    side = {}
    for r in rows:
        if r["id"] in test_ids:
            side[r["id"]] = r["source_schema"]
    leak = sum(v in ("base", "prod_h") for v in side.values())
    F("ITOpsLeakTest", "ITOpsGraph test samples whose source prefix (base, prod_h) belongs to a product the scope rules join to "
      "a development prefix (prod_f, product H)", f"{leak}", src=(f"schema-mapping-datasets@{PKG} labels + data/splits/"
      "itopsgraph.json", "v6_phase2.itops_facts"), status=D, fn="v6_phase2.itops_facts")
    # comparison table: test fields per model by benchmark, from the v4 cells
    per = {ds: next(c.n for c in cells if c.ds == ds) for ds in DATASETS}
    for tg, dss in (("Valentine", [d for d in DATASETS if d.startswith("valentine")]), ("OMOP", [d for d in DATASETS if d.startswith("omop")]),
                    ("NLGQL", ["nl2gql"]), ("ToolBench", ["toolbench"]), ("ITOps", ["itopsgraph"])):
        F(f"ITOpsCmp{tg}", f"comparison table: {tg} test fields per model (" + " + ".join(f"{per[d]}" for d in dss) + ")",
          f"{sum(per[d] for d in dss):,}", status=D, fn="v6_phase2.itops_facts")
    # descriptions: template or authored, per dataset's candidate pool
    def share(ds, pat):
        d = json.load(open(ROOT / f"data/{ds}_api_lookup.json"))
        xs = [(v.get("description") or "").strip() for v in d.values()]
        return sum(bool(re.match(pat, x)) for x in xs), len(xs)
    val = [share(d, r"^Column .+ \(.*\) in table \S+") for d in DATASETS if d.startswith("valentine")]
    nl = share("nl2gql", r"^[A-Za-z0-9 ()]+? for [A-Za-z0-9_]+$")
    tb = share("toolbench", r"^Parameter \S+ of ")
    om = [share(d, r"^Column .+ in table ") for d in DATASETS if d.startswith("omop")]
    F("DescTemplValentine", "Valentine pool descriptions that are the template 'Column X (type) in table Y', %",
      f"{100 * sum(x for x, _ in val) / sum(n for _, n in val):.0f}", src=(", ".join(f"data/{d}_api_lookup.json" for d in DATASETS if d.startswith("valentine")), "v6_phase2.itops_facts"),
      status=D, fn="v6_phase2.itops_facts")
    F("DescTemplNLGQL", "NL2GQL pool descriptions that are the template '<kind> for <table>', %", f"{100 * nl[0] / nl[1]:.0f}",
      src=("data/nl2gql_api_lookup.json", "v6_phase2.itops_facts"), status=D, fn="v6_phase2.itops_facts")
    F("DescTemplToolBench", "ToolBench pool descriptions that are the template 'Parameter X of Y', % (the rest are the APIs' own "
      "parameter documentation)", f"{100 * tb[0] / tb[1]:.0f}", src=("data/toolbench_api_lookup.json", "v6_phase2.itops_facts"),
      status=D, fn="v6_phase2.itops_facts")
    F("DescTemplOMOP", "OMOP pool descriptions that follow a column template, %", f"{100 * sum(x for x, _ in om) / sum(n for _, n in om):.0f}",
      src=(", ".join(f"data/{d}_api_lookup.json" for d in DATASETS if d.startswith("omop")), "v6_phase2.itops_facts"), status=D, fn="v6_phase2.itops_facts")
    if tb[0] / tb[1] < 0.9:
        ITOPS_FLAGS.append(f"descriptions: ToolBench is not template-only; {tb[0]} of {tb[1]} pool descriptions "
                           f"({100 * tb[0] / tb[1]:.0f}%) are 'Parameter X of Y', the rest are the APIs' own parameter documentation")
    note("DescTemplates", "where real descriptions exist", "Valentine and NL2GQL descriptions are name-derived templates "
         "(F-DescTemplValentine, F-DescTemplNLGQL); OMOP's are authored table and column documentation; ToolBench is mixed: "
         "about half are the template 'Parameter X of Y' and half the APIs' own parameter documentation (F-DescTemplToolBench); "
         "ITOpsGraph has authored schema comments on 27% of pool fields.")
    note("ITOpsFlags", "values that disagree with PAPER_PACKAGE_v6.md", " | ".join(ITOPS_FLAGS) or "none")


# ============================================================================ R12: Table 1, per-dataset accuracy
PDROWS = {}
HODIFF = []


def baseline_answers(path):
    """(sample_id, field lower) -> top answer of a stored baseline run (results list in the protocol v2 format)."""
    d = json.load(open(path))
    if isinstance(d, list):
        d = {"results": d}
    out = {}
    for rec in R2.records(d, {str(x.get("sample_id")) for x in d["results"]}):
        out[(rec["sample_id"], rec["field"].lower())] = rec["base"]
    return out


def r12():
    """Table 1 (PLAN_PHASE2.md, R12): field accuracy per core dataset on the v4 test fields, named target type, full-
    identity scorer, ITOpsGraph v2.1, on the one backbone the baselines share (Granite-4). A field a baseline did not
    answer counts as wrong. Prompt-Matcher fields outside its population keep the first answer (it changes only the
    fields it selects). Missing runs print 'not run'."""
    BB = "granite"

    def anytool_path(ds):
        # ToolBench: the target-first run (the named target tool's fields first in the solver's 100-candidate list)
        # replaces the capped run once it exists; the capped run stays as an exploratory fact below.
        # anytool_r: answers re-resolved from the stored raw mappings (bl_blank.py; the Bio-ML label change had dropped
        # SemTab answers containing " ("), preferred wherever it exists.
        rr = ROOT / f"results/naacl/v6/anytool_r/{ds}__granite.json"
        if rr.exists():
            return rr
        tf = ROOT / f"results/naacl/v6/anytool_tf/{ds}__granite.json"
        return tf if ds == "toolbench" and tf.exists() else ROOT / f"results/naacl/v6/anytool/{ds}__granite.json"
    pm_budget = min((0.3, 0.5), key=lambda b: abs(b - SHARE_M[BB]))

    def pm_path(ds):
        # R-PM (ff1_rerun_facts.json): the field-first description rerun replaces the stored run wherever it exists
        ff1 = ROOT / f"results/naacl/v2_baselines/prompt_matcher_ff1/{ds}__granite/preds.json"
        return ff1 if ff1.exists() else ROOT / f"results/naacl/v2_baselines/prompt_matcher/{ds}__granite/preds.json"
    SYSTEMS = [
        ("AnyTool", anytool_path, "anytool"),
        ("MatchMaker", lambda ds: ROOT / f"results/naacl/v2_baselines/bases/{ds}__matchmaker-granite/base.json", "base"),
        ("Magneto", lambda ds: ROOT / f"results/naacl/magneto/variants/{ds}__desc-bp.json", "base"),
        ("PromptMatcher", pm_path, "pm"),
    ]
    pm_arm = f"pm{int(round(100 * pm_budget))}"

    def pm_pre(ds, rs_, golds_, tag_, where):
        """R-PM: the stored (pre-ff1) Prompt-Matcher cell, kept as a fact where the ff1 run replaces it."""
        if not pm_path(ds).parent.parent.name.endswith("_ff1"):
            return
        P0 = json.loads((ROOT / f"results/naacl/v2_baselines/prompt_matcher/{ds}__granite/preds.json").read_text())[pm_arm]
        x = 100 * float(np.mean([target_correct(P0.get(r["key"], r["base"]), g) for r, g in zip(rs_, golds_)]))
        PMPRE[("HO" if where else "") + tag_] = x
        F(f"PD{'HO' if where else ''}PromptMatcherPreFF{tag_}", f"Table 1{' held-out block' if where else ''}, Granite-4, "
          f"{V.DNAME.get(ds, ds)}: Prompt-Matcher ({pm_arm}) before the ff1 rerun (stored run, Table 1 scorer), field "
          "accuracy, %", f"{x:.1f}", fn="v6_phase2.r12")
    LAB = {"AnyTool": "AnyTool", "MatchMaker": "MatchMaker", "Magneto": "Magneto", "PromptMatcher": "Prompt-Matcher",
           "First": "First pass", "Main": "Map-Reflect", "Every": "Every field", "VFourOne": "Margin-Reflect"}
    if V5:
        LAB.update(Main="Map-Reflect", Every="Reflect on all")
        V5G = {c.ds: c for c in V5["core"] + V5["ho"] if c.model == BB}

    def ours(ds, c, keep, method, every):
        """First pass, Map-Reflect and every field: method v5 when it has been scored, with v4.1 as the reference."""
        if not V5:
            return {"First": keep, "Main": method, "Every": every}
        c5 = V5G[ds]
        assert abs(100 * c5.keep.mean() - keep) < 1e-9, (ds, 100 * c5.keep.mean(), keep)  # the same first pass and fields
        return {"First": keep, "Main": 100 * c5.pol["v5"][0].mean(), "Every": 100 * c5.pol["every"][0].mean(), "VFourOne": method}
    cov = []
    for ds in DATASETS:
        c = next(x for x in cells if x.ds == ds and x.model == BB)
        rs = rows_of(c)
        keys = [(str(r["sample_id"]), r["key"].rsplit("||", 1)[1].lower()) for r in rs]
        vals = ours(ds, c, 100 * c.keep.mean(), 100 * c.method.mean(), 100 * c.action.mean())
        for nm, pf, kind in SYSTEMS:
            p = pf(ds)
            if not p.exists():
                vals[nm] = None
                cov.append(f"{nm} {DT[ds]}: not run")
                continue
            if kind == "pm":
                P = json.loads(p.read_text())[f"pm{int(round(100 * pm_budget))}"]
                ans = [P.get(r["key"], r["base"]) for r in rs]
                hit = sum(r["key"] in P for r in rs)
            else:
                A = baseline_answers(p)
                ans = [A.get(k) for k in keys]
                hit = sum(k in A for k in keys)
            vals[nm] = 100 * float(np.mean([target_correct(a, r["_gold"]) for a, r in zip(ans, rs)]))
            cov.append(f"{nm} {DT[ds]}: {hit}/{len(rs)} fields answered")
            if kind == "pm":
                pm_pre(ds, rs, [r["_gold"] for r in rs], DT[ds], False)
        PDROWS[ds] = vals
        capped = ROOT / f"results/naacl/v6/anytool/{ds}__granite.json"
        if ds == "toolbench" and anytool_path(ds) != capped and capped.exists():
            A = baseline_answers(capped)
            F("PDAnyToolToolBenchCapped", "Table 1, Granite-4, ToolBench: AnyTool field accuracy with its original candidate "
              "order (the 100-candidate cap can drop the named target tool), %",
              f"{100 * float(np.mean([target_correct(A.get(k), r['_gold']) for k, r in zip(keys, rs)])):.1f}",
              fn="v6_phase2.r12")
    cols = ["AnyTool", "MatchMaker", "Magneto", "PromptMatcher", "First", "Main", "Every"] + (["VFourOne"] if V5 else [])
    for ds, v in PDROWS.items():
        for k in cols:
            if v[k] is not None:
                F(f"PD{k}{DT[ds]}", f"Table 1, Granite-4, {V.DNAME[ds]}: {LAB[k]} field accuracy, %", f"{v[k]:.1f}", fn="v6_phase2.r12")
    avg = {}
    for k in cols:
        xs = [PDROWS[ds][k] for ds in DATASETS]
        avg[k] = None if any(x is None for x in xs) else float(np.mean(xs))
        if avg[k] is not None:
            F(f"PD{k}Avg", f"Table 1, Granite-4: {LAB[k]}, unweighted mean over the ten datasets, %", f"{avg[k]:.1f}", fn="v6_phase2.r12")
    F("PDPromptMatcherBudget", "Prompt-Matcher budget used in Table 1 (closest to Map-Reflect's Granite-4 routed share), %",
      f"{100 * pm_budget:.0f}", fn="v6_phase2.r12")
    note("PDDefinitions", "Table 1 definitions", "Field accuracy (%) on the v4 test fields of each core dataset (test minus old "
         "development), named target type, full-identity scorer, ITOpsGraph v2.1, backbone Granite-4-H-small for every LLM "
         "column. MatchMaker: the protocol v2 run (results/naacl/v2_baselines/bases/*__matchmaker-granite; Valentine with the "
         "expected_target fallback). Magneto: released code, zero-shot bipartite reranking with descriptions (desc-bp), no LLM. "
         "Prompt-Matcher: the protocol v2 Granite-4 run on the same first pass. AnyTool: the faithful implementation "
         "(anytool/anytool_faithful.py, scripts/naacl/v6_calls/anytool_t1.py), target given, on Granite-4. Stated "
         "adaptations: (1) the expected_target fallback, when the utterance or AnyTool's product parse yields no target "
         "(Valentine, OMOP), MatchMaker and AnyTool use the record's named target; (2) on ToolBench, AnyTool's solver lists "
         "the named target tool's fields first, so its 100-candidate cap cannot drop them (the run with the original "
         "order is F-PDAnyToolToolBenchCapped); (3) on the five held-out sets, AnyTool and MatchMaker see exactly the "
         "target candidates each record lists, the list Map-Reflect and Magneto see; on Bio-ML, whose source classes are bare OMIM numbers, AnyTool's "
         "solver sees each source as its id with its label. AnyTool answers are translated "
         "from its hierarchy names back to lookup ids. Prompt-Matcher on the held-out sets: its answer accuracy Pr comes "
         "from the pooled development fit of the ten core sets (SemTab and SchemaNet have no development split and "
         "MIMIC-OMOP's is thin), the transfer setting of Map-Reflect's thresholds, which are frozen from core development. "
         "A field a baseline did not answer counts as wrong. Average: unweighted mean over the ten datasets, printed only "
         "when the column is complete.")

    # held-out block: our three columns on Granite-4 from the v4 held-out test rows (newdata_analysis.cell, the scorer and
    # gold of the held-out analysis); baselines where a run exists at the same paths
    argv, sys.argv = sys.argv, ["newdata_analysis.py"]
    import newdata_analysis as ND
    sys.argv = argv
    from eval.calc_accuracy import load_gold_file
    HO = [("mimic_omop_rematch", "Mimic", "MIMIC-OMOP"), ("semtab_m2kg_r1", "SemTab", "SemTab"),
          ("oaei_conference", "OAEI", "OAEI Conference"), ("bioml_omim_ordo", "BioML", "Bio-ML"),
          ("schemanet_cprd", "SchemaNet", "SchemaNet")]
    fz = json.loads((ROOT / FROZEN_REL).read_text())["thresholds"]
    HOROWS = {}
    for ds, tg, nm in HO:
        man = json.load(open(ROOT / f"data/splits/{ds}.json"))
        gold = load_gold_file(ROOT / f"data/{ds}_dataset.jsonl")
        from collections import defaultdict as _dd
        c = ND.cell(ds, BB, set(man["test"]), gold, fz[BB], _dd(int))
        vals = ours(ds, None, 100 * float(np.mean(c.vec["keep all"])), 100 * float(np.mean(c.vec["v4"])),
                    100 * float(np.mean(c.vec["reflect all, v4 action"])))
        rows_ = [json.loads(l) for l in open(ROOT / f"results/naacl/v4/test/given/{ds}__{BB}/fields.jsonl")]
        rows_ = [r for r in rows_ if str(r["sample_id"]) in set(man["test"])]
        assert len(rows_) == c.n
        golds = [gold[str(r["sample_id"])][0][r["key"].rsplit("||", 1)[1]] for r in rows_]
        keys = [(str(r["sample_id"]), r["key"].rsplit("||", 1)[1].lower()) for r in rows_]
        for nmb, pf, kind in SYSTEMS:
            p = pf(ds)
            if not p.exists():
                vals[nmb] = None
                cov.append(f"{nmb} {tg} (held-out): not run")
                continue
            if kind == "pm":
                P = json.loads(p.read_text())[f"pm{int(round(100 * pm_budget))}"]
                ans = [P.get(r["key"], r["base"]) for r in rows_]
            else:
                A = baseline_answers(p)
                ans = [A.get(k) for k in keys]
            vals[nmb] = 100 * float(np.mean([target_correct(a, g) for a, g in zip(ans, golds)]))
            if kind == "pm":
                pm_pre(ds, rows_, golds, tg, True)
        HOROWS[ds] = (tg, nm, vals)
        k_, v_, e_ = c.vec["keep all"], c.vec["v4"], c.vec["reflect all, v4 action"]
        un = c.routed == 0
        HODIFF.append((nm, 100 * float(c.routed.mean()), int((v_ != e_).sum()), int(un.sum()), int((e_[un] != k_[un]).sum()), c.n))
        F(f"PDHORouted{tg}", f"Table 1 held-out block, Granite-4, {nm}: share of fields Map-Reflect routes, %",
          f"{100 * float(c.routed.mean()):.1f}", fn="v6_phase2.r12")
        for k in cols:
            if vals[k] is not None:
                F(f"PDHO{k}{tg}", f"Table 1 held-out block, Granite-4, {nm}: {LAB[k]} field accuracy, %", f"{vals[k]:.1f}", fn="v6_phase2.r12")
    note("PDHORoutedEqualsAll", "why Routed equals All fields in Table 1's held-out block",
         "Real, not a shared column: Routed is the v4 answer (newdata_analysis vec 'v4'), All fields the every-field second "
         "pass ('reflect all, v4 action'). Granite-4 routes " + ", ".join(f"{a} {b:.1f}%" for a, b, *_ in HODIFF)
         + " of the held-out fields, and on the fields it keeps the second pass changes no outcome ("
         + ", ".join(f"{a} {u_} kept, {ch} changed" for a, _, _, u_, ch, _ in HODIFF) + "), so the two columns differ on "
         + str(sum(d for _, _, d, *_ in HODIFF)) + " fields in all five sets.")
    hoavg = {}
    for k in cols:
        xs = [HOROWS[ds][2][k] for ds, _, _ in HO]
        hoavg[k] = None if any(x is None for x in xs) else float(np.mean(xs))
        if hoavg[k] is not None:
            F(f"PDHO{k}Avg", f"Table 1 held-out block, Granite-4: {LAB[k]}, unweighted mean over the five held-out sets, %",
              f"{hoavg[k]:.1f}", fn="v6_phase2.r12")
    note("PDCoverage", "Table 1 baseline coverage on the v4 Granite-4 test fields", "; ".join(cov))

    def cell(x, best):
        if x is None:
            return "not run"
        t = f"{x:.1f}"
        return rf"\textbf{{{t}}}" if abs(x - best) < 0.05 else t

    # Table 1 shows the baselines and Map-Reflect only; the margin configuration (F-PDVFourOne*) stays in the facts
    # and in the appendix per-model table. With the model-free baselines folded, three column groups: no LLM, LLM-based,
    # Map-Reflect (ours); their values are the facts of t1_retrieval_facts.json (rounded there to one decimal).
    nollm = [k for k in RETM if f"PD{k}Avg" in RET]
    if nollm:
        HOTAG = {ds: tg for ds, tg, _ in HO}
        for ds in PDROWS:
            for k in nollm:
                PDROWS[ds][k] = RET.get(f"PD{k}{DT[ds]}")
        for ds in HOROWS:
            for k in nollm:
                HOROWS[ds][2][k] = RET.get(f"PDHO{k}{HOTAG[ds]}")
        for k in nollm:
            avg[k], hoavg[k] = RET.get(f"PD{k}Avg"), RET.get(f"PDHO{k}Avg")
        LAB.update(BMTwentyFive="BM25", Dense="Dense", Hybrid="Hybrid", Cupid="Cupid", SimFlood="SimFl.")
    groups = ([("No LLM", nollm + ["Magneto"]), ("LLM-based", ["AnyTool", "MatchMaker", "PromptMatcher"])] if nollm
              else [("", ["AnyTool", "MatchMaker", "Magneto", "PromptMatcher"])])
    groups.append(("Map-Reflect (ours)", ["First", "Main", "Every"]))
    tcols = [k for _, ks in groups for k in ks]
    SHORTLAB = {"First": "First", "Main": "Selective", "Every": "All"}
    if nollm:  # thirteen numeric columns: two-line headers for the long baseline names and 3 pt column spacing
        SHORTLAB.update(MatchMaker=r"\shortstack{Match-\\Maker}", PromptMatcher=r"\shortstack{Prompt-\\Matcher}")
        TABLE_SEP["perdataset_granite"] = 3

    def line(name, v):
        have = [round(v[k], 1) for k in tcols if v.get(k) is not None]  # best of the printed values, ties all bold
        best = max(have) if have else 0
        return f"{name} & " + " & ".join(cell(v.get(k), best) for k in tcols) + r" \\"
    body = []
    for i, (fam, dss) in enumerate(FAMILY):  # no spacer rows between groups (paper/STYLE.md section 8)
        for ds in dss:
            body.append(line(V.DNAME[ds], PDROWS[ds]))
    body.append(rf"\multicolumn{{{len(tcols) + 1}}}{{l}}{{\emph{{Held-out sets}}}} \\")
    for ds, tg, nm in HO:
        body.append(line(nm, HOROWS[ds][2]))
    body.append(r"\midrule")
    for nm, av in (("Average, core", avg), ("Average, held-out", hoavg)):
        have = [round(av[k_], 1) for k_ in tcols if av.get(k_) is not None]
        best = max(have) if have else 0
        body.append(f"{nm} & " + " & ".join(cell(av[k], best) if av.get(k) is not None else "n/a"
                                            for k in tcols) + r" \\")
    top, rules, pos = [], [], 2
    for gname, ks in groups:
        top.append(rf"\multicolumn{{{len(ks)}}}{{c}}{{{gname}}}" if gname else " & ".join([""] * len(ks)))
        if gname:
            rules.append(rf"\cmidrule(lr){{{pos}-{pos + len(ks) - 1}}}")
        pos += len(ks)
    hdr = (" & " + " & ".join(top) + r" \\" + "\n" + "".join(rules) + "\n"
           + "Dataset & " + " & ".join(SHORTLAB.get(k, LAB[k]) for k in tcols) + r" \\")
    # DECISIONS row 90: the main-text Table 1 is the gpt-oss-120b table (r12_second); this Granite-4 table, unchanged,
    # moves to the appendix as tab6_perdataset_granite
    TABLE_ENV["perdataset_granite"] = "table*"
    write_tab("perdataset_granite", "l" + "r" * len(tcols), hdr, body,
              note_="R12 Table 1. Field accuracy (%), Granite-4 for every LLM column; definitions in FACTS.md F-PDDefinitions.",
              label="perdataset_granite")
    # appendix: the four models
    body = []
    for i, (fam, dss) in enumerate(FAMILY):
        for ds in dss:
            row = [V.DNAME[ds]]
            for m in MODELS:
                c = next(x for x in cells if x.ds == ds and x.model == m)
                row += [f"{100 * c.keep.mean():.1f}", f"{100 * c.method.mean():.1f}", f"{100 * c.action.mean():.1f}"]
            body.append(" & ".join(row) + r" \\")
    body.append(r"\midrule")
    row = ["Average"]
    for m in MODELS:
        for f_ in (lambda c: c.keep, lambda c: c.method, lambda c: c.action):
            row.append(f"{float(np.mean([100 * f_(next(x for x in cells if x.ds == ds and x.model == m)).mean() for ds in DATASETS])):.1f}")
    body.append(" & ".join(row) + r" \\")
    hdr = (" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{MSHORT[m]}}}" for m in MODELS) + r" \\" + "\n"
           + " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(MODELS))) + "\n"
           + "Dataset" + " & First & Selective & All" * len(MODELS) + r" \\")
    write_tab("perdataset_models", "l" + "rrr" * len(MODELS), hdr, body, env="table*",
              note_="R12 appendix. Map-Reflect (ours) per model: no second pass (first pass), routed, and all fields (every "
              "field); field accuracy (%); average is the unweighted mean over datasets.")
    wcsv("r12_perdataset", ["dataset"] + cols, [[ds] + [("" if PDROWS[ds][k] is None else round(PDROWS[ds][k], 2)) for k in cols]
                                                 for ds in DATASETS])


# ============================================================== margin on the full v4.1 test population (phase 4)
def margin_full_pop():
    """The margin on every answered v4.1 test field (no paid-signal restriction): pooled error AUROC on within-model
    ranks with the split-group bootstrap of the main analysis, and the margin gate at each model's routed share of its
    answered fields with the v4.1 second pass, against random and the oracle. Also the Table 3 population rule."""
    rs = masked(cells, lambda c: c.valid)
    ranks, errs, W = [], [], []
    for m in MODELS:
        cs = [c for c in rs if c.model == m]
        ranks.append(U.pct_rank(np.concatenate([c.margin for c in cs])))
        errs.append(np.concatenate([1 - c.keep for c in cs]))
        W.append(boot_weights(cs))
    r_, e_, W_ = np.concatenate(ranks), np.concatenate(errs), np.concatenate(W, axis=1)
    auc = wauc(-r_, e_, np.ones(len(e_)))
    lo, hi = GS.interval(np.array([wauc(-r_, e_, W_[b]) for b in range(W_.shape[0])]))
    n = sum(c.n for c in rs)
    F("SigMarginFullPopFields", "answered v4.1 test field-model pairs (the full population, no paid-signal restriction)",
      f"{n:,}", fn="v6_phase2.margin_full_pop")
    F("SigMarginFullPopAUC", "margin: pooled error AUROC on every answered test field (within-model ranks)", f"{auc:.3f}",
      f"{lo:.3f}", f"{hi:.3f}", fn="v6_phase2.margin_full_pop")
    share = {m: sum(c.routed.sum() for c in rs if c.model == m) / sum(c.n for c in rs if c.model == m) for m in MODELS}
    wm = per_model_weights(rs, lambda c: c.margin, share)
    fixes = {}
    for m in MODELS:
        cs = [c for c in rs if c.model == m]
        fix = np.concatenate([np.where(c.keep == 0, c.action, 0.0) for c in cs])
        k = share[m] * len(fix)
        w = U.gate_weights(-fix, k)
        i = 0
        for c in cs:
            fixes[id(c)] = w[i:i + c.n]
            i += c.n
    d = lambda c: c.action - c.keep  # noqa: E731
    em, _ = pooled(rs, lambda c: wm[id(c)] * d(c))
    er, _ = pooled(rs, lambda c: share[c.model] * d(c))
    eo, _ = pooled(rs, lambda c: fixes[id(c)] * d(c))
    dmr, ddr = pooled(rs, lambda c: wm[id(c)] * d(c) - share[c.model] * d(c))
    lo2, hi2 = GS.interval(ddr)
    first = 100 * sum(c.keep.sum() for c in rs) / n
    F("SigMarginFullPopRouted", "answered test fields the v4.1 router sends to the second pass, %",
      f"{100 * sum(c.routed.sum() for c in rs) / n:.1f}", fn="v6_phase2.margin_full_pop")
    F("SigMarginFullPopAcc", "margin gate at the router's share of answered fields: accuracy on answered fields, %",
      f"{first + em:.2f}", fn="v6_phase2.margin_full_pop")
    F("SigMarginFullPopGain", "margin gate at that share: change over the first pass on answered fields", sg(em), fn="v6_phase2.margin_full_pop")
    F("SigMarginFullPopVsRand", "margin gate minus random at the same share, answered fields", sg(dmr), sg(lo2), sg(hi2),
      fn="v6_phase2.margin_full_pop")
    F("SigMarginFullPopOracleShare", "margin gate: share of the oracle gain at that share, answered fields, %",
      f"{100 * em / eo:.0f}" if eo > 0 else "n/a", fn="v6_phase2.margin_full_pop")
    # the Table 3 population rule and its size
    parts = []
    for m in U.MODELS:
        nF, nP = len(L["F"][m]), len(L["pop"][m])
        F(f"SigPop{MT[m]}", f"{MSHORT[m]}: Table 3 population / answered test fields (share dropped, %)",
          f"{nP:,} / {nF:,} ({100 * (nF - nP) / nF:.1f})", fn="v6_phase2.margin_full_pop")
        parts.append(f"{MSHORT[m]} {nP:,} of {nF:,} ({100 * (nF - nP) / nF:.1f}% dropped)")
    note("SigPopulationRule", "Table 3 population rule", "Table 3 compares signals head to head on one population per model: "
         "the answered test fields (declined first passes excluded) on which every signal that model has carries a valid "
         "value, stated confidence included (an unresolved stated answer is missing). Sizes: " + "; ".join(parts) + ". "
         "F-SigMarginFullPop* gives the margin on every answered field without this restriction.")


# =========================================================================================== D2 fold (phase 3/4)
def d2_facts():
    """The D2 agent's facts (scripts/naacl/v6_d2.py -> docs/naacl/results/v6_d2_facts.json), read, never transcribed,
    with their ids, claims and intervals; folded only when they were fitted under the same method variant."""
    f = ROOT / "docs/naacl/results/v6_d2_facts.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    if VARIANT == "v4_1" and not (str(d.get("base_method", "")).replace(".", "_") in ("v4_1",) and d.get("canonical_margin")):
        note("DTwoStatus", "D2", "v6_d2_facts.json was not fitted under method v4.1; not folded")
        return
    for x in d["facts"]:
        fid = x["id"][2:] if x["id"].startswith("F-") else x["id"]
        if fid.endswith("Definition"):  # a definition: a note, no macro
            note(fid, x["claim"], x["value"])
            continue
        fact(fid, x["claim"], x["value"], (x.get("src") or "docs/naacl/results/v6_d2_facts.json", x.get("fn") or "v6_d2"),
             "exploratory", x.get("lo"), x.get("hi"), func=x.get("fn") or "v6_d2")
    note("DTwoProvenance", "how D2 was fitted and scored", "Refit and frozen on development under method v4.1 (canonical "
         "margins) in docs/naacl/results/v6_d2_1_frozen.json, committed in 55d8d46 before its single test scoring (fd4ac32); "
         f"base method {d.get('base_method')}, canonical margins {d.get('canonical_margin')}. The facts are read from "
         "docs/naacl/results/v6_d2_facts.json (" + str(len(d["facts"])) + " facts); its diff_vs_first_run section records "
         "the changes from the first D2 run.")
    if any(x["id"].startswith("F-DTwoCost") for x in d["facts"]):
        note("DTwoCostProvenance", "how the D2-cost check was fitted and scored", "The development-estimated action costs "
             "were frozen in commit 2832fc1 before the single development and test scoring in 006b400; the F-DTwoCost* "
             "facts are read from docs/naacl/results/v6_d2_facts.json, and F-DTwoCostDefinition defines the check.")


# ========================================================================================= BL-blank fold (phase 5)
def bl_blank_facts():
    """The baseline agent's classification of blank baseline answers in Table 1 (scripts/naacl/v6_calls/bl_blank.py ->
    docs/naacl/results/bl_blank_facts.json), read, never transcribed, each fact keeping its stated status."""
    f = ROOT / "docs/naacl/results/bl_blank_facts.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    for x in d["facts"]:
        fid = x["id"][2:] if x["id"].startswith("F-") else x["id"]
        if fid.endswith("Definition") or x.get("status") == "note":
            note(fid, x["claim"], x["value"])
            continue
        fact(fid, x["claim"], x["value"], (x.get("src") or "docs/naacl/results/bl_blank_facts.json", x.get("fn") or "bl_blank"),
             x.get("status") or "exploratory", x.get("lo"), x.get("hi"), func=x.get("fn") or "bl_blank")


# =============================================================================================== D1 (phase 5)
def d1_block():
    """D1, the strong first-pass control (scripts/naacl/v5.py --d1): the v5 second pass after a hybrid top-20 first pass,
    Qwen3.5 and Llama-4, ten core datasets. Its facts are read from docs/naacl/results/v6_d1_facts.json (F-DOne*, ids
    kept) and tab6_d1.tex is generated from them."""
    f = ROOT / "docs/naacl/results/v6_d1_facts.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    J = {}
    for x in d["facts"]:
        fid = x["id"][2:]
        J[fid] = x
        if fid.endswith("Definition"):
            note(fid, x["claim"], x["value"])
            continue
        fact(fid, x["claim"], x["value"], (x.get("src") or "docs/naacl/results/v6_d1_facts.json", x.get("fn") or "v5.score --d1"),
             x.get("status") or "exploratory", x.get("lo"), x.get("hi"), func=x.get("fn") or "v5.score --d1")
    note("DOneProvenance", "how D1 was fitted and scored", "D1 (the v5 second pass after a hybrid top-20 first pass, "
         "Qwen3.5 and Llama-4, ten core datasets) was fitted and frozen on development in docs/naacl/results/v6_d1_frozen.json, "
         "commit aa6aad1, before any D1 test call and its single test scoring in 5df2874 (scripts/naacl/v5.py --stage score "
         "--d1, docs/naacl/results/v6_d1_test.md). The F-DOne* facts are read from docs/naacl/results/v6_d1_facts.json.")
    v = lambda k: J[f"DOne{k}"]  # noqa: E731

    def ci(x):
        return f"{tex_s(float(x['value']))} {tci(float(x['lo']), float(x['hi']))}"
    sg_ = v("StructGain")
    ROWS = [("No reflection", "Keep"), ("Map-Reflect", "Main"), (r"Margin, $t_m$ refit", "Margin")]
    if "DOneMarginOldAcc" in J:
        ROWS.append(("Margin, fixed thresholds", "MarginOld"))
    ROWS += [("Schema size", "Struct"), ("Reflect on all", "Every"), ("Random selection", "Random"), ("Oracle", "Oracle")]
    body = []
    for lab, tg in ROWS:
        acc = f"{float(v(tg + 'Acc')['value']):.2f}"
        if tg == "Keep":
            gain = "ref."
            # no reflection minus schema size is the negated schema-size gain (the same paired bootstrap draws)
            vs = f"{tex_s(-float(sg_['value']))} {tci(-float(sg_['hi']), -float(sg_['lo']))}"
            calls, rb = "0.00", "0 / 0"
        else:
            gain = ci(v(tg + "Gain"))
            vs = "ref." if tg == "Struct" else ci(v(tg + "VsStruct"))
            calls = v(tg + "Calls")["value"] if f"DOne{tg}Calls" in J else "n/a"
            rb = v(tg + "Repairs")["value"].replace(",", "{,}")
        body.append(f"{lab} & {acc} & {gain} & {vs} & {calls} & {rb} \\\\")
    write_tab("d1", "lrrrrr", r"Selection & Acc. & Gain [95\% CI] & $-$Schema size [95\% CI] & Calls & Repairs / breaks \\",
              body, env="table*", note_="D1 (exploratory). Proposed caption for the writer: \"Accuracy on the ten core "
              "datasets after a hybrid top-20 first pass, Qwen3.5 and Llama-4, with the v5 reflection (exploratory).\" "
              "Source: docs/naacl/results/v6_d1_facts.json; Oracle calls are not comparable (n/a).", label="d1")


# ==================================================================== model-free Table 1 baselines (item A)
RET = {}
RETM = ["BMTwentyFive", "Dense", "Hybrid", "Cupid", "SimFlood"]


def t1_retrieval():
    """The model-free Table 1 baselines (BM25, dense, hybrid top-1, Cupid, Similarity Flooding), computed by another
    agent into docs/naacl/results/t1_retrieval_facts.json (no model call). Folded with their ids and statuses; their
    values fill Table 1's "No LLM" columns."""
    f = ROOT / "docs/naacl/results/t1_retrieval_facts.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    for x in d["facts"]:
        fid = x["id"][2:]
        if x.get("status") == "note" or fid.endswith("Definition") or fid.endswith("Definitions"):
            note(fid, x["claim"], x["value"])
            continue
        fact(fid, x["claim"], x["value"], (x.get("src") or "docs/naacl/results/t1_retrieval_facts.json", x.get("fn") or "t1_retrieval"),
             x.get("status") or "descriptive", x.get("lo"), x.get("hi"), func=x.get("fn") or "t1_retrieval")
        RET[fid] = float(x["value"]) if re.fullmatch(r"[+\-]?\d+(\.\d+)?", str(x["value"])) else x["value"]


def itops_v3_facts():
    """Two descriptive counts of ITOpsGraph v3, read from the itopsgraph repository at a fixed commit with git show
    (the checkout is never switched)."""
    repo, rev = ROOT.parent / "itopsgraph", "0a0231f"

    def show(path):
        return subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True, text=True,
                              check=True).stdout
    if not repo.is_dir():
        return
    xw = json.loads(show("data/itopsgraph_xw/build_report.json"))
    F("ITOpsVThreeCrosswalk", "ITOpsGraph v3: crosswalk samples", f"{xw['samples']}",
      src=(f"itopsgraph@{rev} data/itopsgraph_xw/build_report.json", "build_itopsgraph_xw.py"), status="descriptive", fn="v6_phase2.itops_v3_facts")
    # the test mappings whose v2.1 label the second, independent labelling disputes: metadata.v3_conflicts kind
    # "disputed" on a scored mapping of a cross-product test sample (build_report.json holds only the all-split total)
    rows = [json.loads(l) for l in show("data/itopsgraph_v3/itopsgraph_v3.jsonl").splitlines() if l.strip()]
    br = json.loads(show("data/itopsgraph_v3/build_report.json"))
    test = [r for r in rows if r["split"] == "test" and not r["id"].startswith("xw")]
    disp = sum(1 for r in test for f, x in ((r.get("metadata") or {}).get("v3_conflicts") or {}).items()
               if x["kind"] == "disputed" and f in r["mapping"])
    allsplit = sum(1 for r in rows for f, x in ((r.get("metadata") or {}).get("v3_conflicts") or {}).items()
                   if x["kind"] == "disputed" and f in r["mapping"])
    assert allsplit == sum(v for k, v in br["fields"].items() if k.startswith("disputed:")), allsplit
    assert sum(len(r["mapping"]) for r in test) == br["cross_product"]["test mappings"]
    F("ITOpsVThreeDisputedTest", "ITOpsGraph v3: scored test mappings (of F-ITOpsTestScored) whose v2.1 label the second, "
      "independent labelling disputes (metadata.v3_conflicts kind disputed)", f"{disp}",
      src=(f"itopsgraph@{rev} data/itopsgraph_v3/itopsgraph_v3.jsonl, data/itopsgraph_v3/build_report.json",
           "build_itopsgraph_v3.py"),
      status="descriptive", fn="v6_phase2.itops_v3_facts")


def itops_human_overlap():
    """Where earlier human annotations reach the scored ITOpsGraph mappings (scripts/naacl/itops_human_overlap.py,
    reading itopsgraph@0a0231f; no model call), folded as descriptive facts with their ids and sources."""
    f = ROOT / "docs/naacl/results/itops_human_overlap_facts.json"
    if not f.exists():
        return
    got = {}
    for x in json.loads(f.read_text())["facts"]:
        fid = x["id"][2:]
        fact(fid, x["claim"], x["value"], (x["src"], x.get("fn") or "itops_human_overlap"), x.get("status") or "descriptive",
             x.get("lo"), x.get("hi"), func=x.get("fn") or "itops_human_overlap")
        got[fid] = x["value"]
    # consistency: the denominator is the paper's scored test population (F-ITOpsTestScored); the disputed count is
    # F-ITOpsVThreeDisputedTest; agree, contradicted and other-targets are disjoint subsets of each split's mappings
    # (the json gives no total of the three, so they are bounded by the mappings), and contradicted test mappings
    # lie among the disputed
    n = {k: int(str(v).replace(",", "")) for k, v in got.items() if re.fullmatch(r"[\d,]+", str(v))}
    fv = {x["id"]: x["value"] for x in FACTS}
    for mine, theirs in (("ITOpsHumanTestMappings", "ITOpsTestScored"), ("ITOpsHumanTestDisputed", "ITOpsVThreeDisputedTest")):
        if mine in n and theirs in fv:
            assert n[mine] == int(str(fv[theirs]).replace(",", "")), (mine, n[mine], theirs, fv[theirs])
    for sp in ("Test", "Dev"):
        parts = [n.get(f"ITOpsHuman{sp}{k}") for k in ("Agree", "Contra", "Other")]
        if None not in parts and f"ITOpsHuman{sp}Mappings" in n:
            assert sum(parts) <= n[f"ITOpsHuman{sp}Mappings"], (sp, parts)
        if f"ITOpsHuman{sp}Total" in n:  # a total, if the json ever gives one, must equal the three parts
            assert sum(parts) == n[f"ITOpsHuman{sp}Total"], (sp, parts)
    if "ITOpsHumanTestDisputed" in n and "ITOpsHumanTestContra" in n:
        assert n["ITOpsHumanTestContra"] <= n["ITOpsHumanTestDisputed"]


def rev2_facts():
    """The second simulated review's offline analyses (scripts/naacl/rev2_*.py, read-only on the stored v5 outputs,
    research d66d150): every docs/naacl/results/rev2_*_facts.json folded with its ids, all exploratory (DECISIONS row 63)."""
    import glob as _glob
    for f in sorted(_glob.glob(str(ROOT / "docs/naacl/results/rev2_*_facts.json"))):
        for x in json.loads(Path(f).read_text())["facts"]:
            fid = x["id"][2:]
            assert re.fullmatch(r"[A-Za-z]+", fid), fid  # letters only; no mapping needed so far
            src = x.get("src") or str(Path(f).relative_to(ROOT))
            fact(fid, x["claim"], x["value"], (f"{src} (research d66d150)", x.get("fn") or Path(f).stem),
                 "exploratory", x.get("lo"), x.get("hi"), func=x.get("fn") or Path(f).stem)


def latency_facts():
    """The latency run (scripts/naacl/latency.py --stage analyse; commits 29d7ef5, 033f7ab, c1db422): every
    F-Latency* fact of docs/naacl/results/latency_facts.json, exploratory, cited to that file (it carries no per-fact
    source). Facts only: no latency table has been asked for."""
    f = ROOT / "docs/naacl/results/latency_facts.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    for x in d["facts"]:
        fid = x["id"][2:]
        assert re.fullmatch(r"Latency[A-Za-z]+", fid), fid
        fact(fid, x["claim"], x["value"], (x.get("src") or "docs/naacl/results/latency_facts.json",
             x.get("fn") or "latency.py --stage analyse"), x.get("status") or "exploratory", x.get("lo"), x.get("hi"),
             func=x.get("fn") or "latency.py --stage analyse")


def rb_facts():
    """R-b, the run-to-run repeat of Map-Reflect v5's second pass on Granite-4 and Qwen3.5 (scripts/naacl/rb.py,
    research f777c4e): every F-RB* fact of docs/naacl/results/rb_facts.json, exploratory, with its own source. Facts only."""
    f = ROOT / "docs/naacl/results/rb_facts.json"
    if not f.exists():
        return
    for x in json.loads(f.read_text())["facts"]:
        fid = x["id"][2:]
        assert re.fullmatch(r"RB[A-Za-z]+", fid), fid
        fact(fid, x["claim"], x["value"], (x.get("src") or "docs/naacl/results/rb_facts.json", x.get("fn") or "rb.score"),
             x.get("status") or "exploratory", x.get("lo"), x.get("hi"), func=x.get("fn") or "rb.score")


def evab_facts():
    """The evidence-format ablation and fresh against anchored reflection (scripts/naacl/evab.py --stage score,
    research f3b44bf): every F-EVAB* fact of docs/naacl/results/evab_facts.json, exploratory, with its own source; and
    the appendix table tab6_evab (DECISIONS row 81). Macro values are plain numbers (no version or run names)."""
    f = ROOT / "docs/naacl/results/evab_facts.json"
    if not f.exists():
        return
    E = {}
    for x in json.loads(f.read_text())["facts"]:
        fid = x["id"][2:]
        assert re.fullmatch(r"EVAB[A-Za-z]+", fid), fid
        assert not re.search(r"\bv\d|D3|R-X1|EvAb", str(x["value"])), (fid, x["value"])  # values print in the paper
        fact(fid, x["claim"], x["value"], (x.get("src") or "docs/naacl/results/evab_facts.json", x.get("fn") or "evab"),
             x.get("status") or "exploratory", x.get("lo"), x.get("hi"), func=x.get("fn") or "evab")
        E[fid[4:]] = (x["value"], x.get("lo"), x.get("hi"))

    # anchored (Map-Reflect) calls per selected field, B rows: evab.py's own counting rule (sum of the stored rows'
    # calls over the fields, as for the fresh ranking), from the rows its scorer reads; computed in a subprocess so
    # evab's argument parser and loaders stay out of this process. The fresh calls recomputed the same way must equal
    # the folded F-EVABB*FreshCalls.
    import subprocess as _sp
    code = ("import sys, json; sys.argv = ['evab.py', '--stage', 'score']; sys.path.insert(0, 'scripts/naacl'); "
            "import evab as E; out = {}\n"
            "for step, split, T in (('Bdev', 'dev', 'Dev'), ('Btest', 'test', 'Test')):\n"
            "    rb = E.load_rows(step, split); N = sum(r.n for r in rb)\n"
            "    out[T] = dict(N=N, anchored=sum(float(r.stored_calls.sum()) for r in rb) / N, "
            "fresh=sum(float(r.fresh_calls.sum()) for r in rb) / N)\n"
            "print('EVABCALLS ' + json.dumps(out))")
    run = _sp.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    line = next((ln for ln in run.stdout.splitlines() if ln.startswith("EVABCALLS ")), None)
    assert line, run.stderr[-2000:]
    calls = json.loads(line[len("EVABCALLS "):])
    for T, lab in (("Dev", "development"), ("Test", "core test")):
        c = calls[T]
        assert f"{c['fresh']:.2f}" == E[f"B{T}FreshCalls"][0], (T, c, E[f"B{T}FreshCalls"])
        fact(f"EVABB{T}AnchoredCalls", f"EvAb B, {lab}: anchored reflection (Map-Reflect, stored) calls per field with the "
             f"verifier, on the {c['N']:,} fields Map-Reflect selects (evab.py's counting rule, as for the fresh ranking)",
             f"{c['anchored']:.2f}", ("scripts/naacl/evab.py (load_rows: the stored rows' calls; computed in "
                                      "v6_phase2.evab_facts)", "v6_phase2.evab_facts"), "exploratory",
             func="v6_phase2.evab_facts")
        E[f"B{T}AnchoredCalls"] = (f"{c['anchored']:.2f}", None, None)

    def ci(k):
        e, lo, hi = E[k]
        return f"{tex_s(e)} {tci(lo, hi)}"
    body = [r"\multicolumn{5}{l}{\emph{Prompt factors, development (648 fields)}} \\"]
    for tg, lab in (("Scores", "Old ranking scores shown"), ("Capped", "Descriptions capped"),
                    ("Oldev", "Evidence for the old candidates only"), ("Hint", "Uniqueness hint instead of the rule")):
        body.append(f"{lab} & {ci('A' + tg + 'MinusRef')} & {E['A' + tg + 'Holm'][0]} & "
                    f"{E['A' + tg + 'Repairs'][0]} / {E['A' + tg + 'Breaks'][0]} & {E['A' + tg + 'Calls'][0]} \\\\")
    body.append(r"\multicolumn{5}{l}{\emph{Fresh ranking minus anchored reflection (calls: fresh / anchored)}} \\")
    for sp, lab, n in (("Dev", "Development", "1{,}051"), ("Test", "Core test", "2{,}189")):
        body.append(f"{lab} ({n} fields) & {ci('B' + sp + 'FreshMinusAnchored')} & n/a & "
                    f"{E['B' + sp + 'FreshRepairs'][0]} / {E['B' + sp + 'FreshBreaks'][0]} & {E['B' + sp + 'FreshCalls'][0]} / "
                    f"{E['B' + sp + 'AnchoredCalls'][0]} \\\\")
    TABLE_ENV.setdefault("evab", "table*")
    write_tab("evab", "lrrrr", r"Variant & Change [95\% CI] & Holm $p$ & Repairs / breaks & Calls \\", body, v5=True,
              note_="CAPTION: the writer's. Prompt factors: each variant minus Map-Reflect's prompt rerun on the same "
              "development fields, paired split-group bootstrap, Holm over the four; repairs / breaks and calls per field "
              "are the variant's own. Fresh ranking minus anchored reflection on the fields Map-Reflect selects, both with "
              "the either-order verifier; repairs / breaks are the fresh ranking's, calls per field fresh / anchored; Holm p is n/a "
              "(no family). Source docs/naacl/results/evab_facts.json (F-EVAB*).", label="evab")


NUMWORD = {"10": "Ten", "20": "Twenty", "30": "Thirty", "40": "Forty", "50": "Fifty"}
NOPRINT = re.compile(r"\bv\d|D3|R-X1|EvAb|\bXW\b|\bR-a\b|\bCR\b")  # names that must not reach a printed macro value


def fold_json(rel, default_fn, notes=()):
    """Fold every fact of a *_facts.json with its id (letters only: 10..50 -> Ten..Fifty, Nl2gql -> NLGQL), source and
    status; ids in notes become notes (no macro). Returns {id: (value, lo, hi)} and the id mapping applied."""
    f = ROOT / rel
    if not f.exists():
        return {}, {}
    got, mapped = {}, {}
    for x in json.loads(f.read_text())["facts"]:
        fid0 = x["id"][2:]
        fid = re.sub(r"(10|20|30|40|50)", lambda m: NUMWORD[m.group(1)], fid0.replace("Nl2gql", "NLGQL"))
        if fid != fid0:
            mapped[fid0] = fid
        assert re.fullmatch(r"[A-Za-z]+", fid), fid
        if fid in notes:
            note(fid, x["claim"], str(x["value"]))
            continue
        assert not NOPRINT.search(str(x["value"])), (fid, x["value"])  # macro values print in the paper
        fact(fid, x["claim"], x["value"], (x.get("src") or rel, x.get("fn") or default_fn),
             x.get("status") or "exploratory", x.get("lo"), x.get("hi"), func=x.get("fn") or default_fn)
        got[fid] = (str(x["value"]), x.get("lo"), x.get("hi"))
    return got, mapped


FOLD_MAPPED = {}


def xw_ra_cr():
    """The crosswalk test (xw_facts.json), R-a (ra_facts.json) and CR (cr_dev_facts.json, research 55221b0), each with
    its own source; the appendix table tab6_xw (DECISIONS row 88)."""
    X, m1 = fold_json("docs/naacl/results/xw_facts.json", "xw.score", notes=("XWPreregCommit", "XWFrozenCommit"))
    _, m2 = fold_json("docs/naacl/results/ra_facts.json", "ra.score")
    _, m3 = fold_json("docs/naacl/results/cr_dev_facts.json", "cr.score")
    FOLD_MAPPED.update(m1, **m2, **m3)
    if not X:
        return
    num = lambda k: float(X[k][0].replace(",", ""))  # noqa: E731
    # coverage diagnostic: mapped test fields whose gold lies only beyond the shown list (derived: fields minus shown)
    fact("XWGoldBeyondShown", "mapped crosswalk test fields whose gold target is not in the first pass's shown list "
         "(F-XWFields minus F-XWGoldShown)", f"{int(num('XWFields') - num('XWGoldShown'))}",
         ("docs/naacl/results/xw_facts.json", "v6_phase2.xw_ra_cr"), "descriptive (post hoc, not pre-registered)",
         func="v6_phase2.xw_ra_cr")

    def ci(k):
        e, lo, hi = X[k]
        return f"{tex_s(e)} {tci(lo, hi)}"
    rows = [("First pass", "Keep"), (r"\textbf{Map-Reflect}", "Main"), ("Every field", "Every"),
            ("Schema-size rule", "Struct"), ("Margin", "Margin"), ("Random, same rate", "Random")]
    body = []
    for lab, tg in rows:
        acc = f"{num('XW' + tg + 'Acc'):.1f}"
        gain = "ref." if tg == "Keep" else ci(f"XW{tg}Gain")
        minus = "ref." if tg == "Main" else (ci("XWMainGain") if tg == "Keep" else ci(f"XWMainMinus{tg}"))
        calls = "0.00" if tg == "Keep" else X[f"XW{tg}Calls"][0]  # the first pass makes no second-pass call
        body.append(f"{lab} & {acc} & {gain} & {minus} & {calls} \\\\")
    # second block (DECISIONS row 91): Table 1's baselines on the crosswalk test with gpt-oss-120b (xwb_facts.json,
    # research 457883f). Calls per field in this block: Map-Reflect's include its first pass, each baseline counts its
    # own model calls (Prompt-Matcher's 0.90 excludes the first pass its questions build on); 0 for the no-call methods.
    # Prompt-Matcher is the pm30 arm, as in Table 1. Gain column: empty in this block (DECISIONS row 94: there is no
    # gpt-oss per-model gain fact with an interval; F-XWMainGainGptoss has none). Prompt-Matcher's calls include the
    # first pass it builds on (F-XWBPMCallsWithFirst); AnyTool and MatchMaker have no separate first pass.
    XB, _ = fold_json("docs/naacl/results/xwb_facts.json", "xwb_score", notes=("XWBPreregCommit",))
    if XB:
        BL = [("BM25", "BMTwentyFive"), ("Dense", "Dense"), ("Hybrid", "Hybrid"), ("Cupid", "Cupid"),
              ("SimFl.", "SimFlood"), ("Magneto", "Magneto"), ("AnyTool", "AnyTool"), ("MatchMaker", "MatchMaker"),
              ("Prompt-Matcher", "PromptMatcherThirty")]
        NM = {"SimFl.": "Similarity Flooding"}

        def cib(k):
            e, lo, hi = XB[k]
            return f"{tex_s(e)} {tci(lo, hi)}"
        body.append(r"\multicolumn{5}{l}{\emph{Baselines (gpt-oss-120b; calls include the first pass)}} \\")
        first = next(float(str(x["value"]).replace(",", "")) for x in FACTS if x["id"] == "XWCallsFirstGptoss") / \
            float(XB["XWBFields"][0].replace(",", ""))
        pm_own = float(XB["XWBPromptMatcherThirtyCalls"][0])
        pm_all = pm_own + first
        assert abs(first - 1.0) < 1e-9 and f"{pm_all:.2f}" == "1.90", (first, pm_own, pm_all)
        fact("XWBPMCallsWithFirst", "crosswalk test, gpt-oss-120b: Prompt-Matcher (pm30) calls per field including the "
             "first pass its questions build on (F-XWBPromptMatcherThirtyCalls plus F-XWCallsFirstGptoss / F-XWBFields)",
             f"{pm_all:.2f}", ("docs/naacl/results/xwb_facts.json, docs/naacl/results/xw_facts.json", "v6_phase2.xw_ra_cr"),
             "exploratory", func="v6_phase2.xw_ra_cr")
        body.append(f"\\textbf{{Map-Reflect}} & {float(XB['XWBMainAcc'][0]):.1f} &  & ref. & {XB['XWBMainCalls'][0]} \\\\")
        for lab, tg in BL:
            cl = XB["XWB" + tg + "Calls"][0]
            cl = "0" if float(cl) == 0 else (f"{pm_all:.2f}" if tg == "PromptMatcherThirty" else cl)
            body.append(f"{lab} & {float(XB['XWB' + tg + 'Acc'][0]):.1f} &  & {cib('XWBMainMinus' + tg)} & {cl} \\\\")
        lab, tg = min(BL, key=lambda b_: float(XB["XWBMainMinus" + b_[1]][0]))
        e, lo, hi = XB["XWBMainMinus" + tg]
        fact("XWBClosestBaseName", "crosswalk test, gpt-oss-120b: the baseline closest to Map-Reflect (smallest lead)",
             NM.get(lab, lab), ("docs/naacl/results/xwb_facts.json", "v6_phase2.xw_ra_cr"), "exploratory",
             func="v6_phase2.xw_ra_cr")
        fact("XWBClosestBaseLead", f"crosswalk test, gpt-oss-120b: Map-Reflect minus the closest baseline ({NM.get(lab, lab)}), "
             "points (paired)", e, ("docs/naacl/results/xwb_facts.json", "v6_phase2.xw_ra_cr"), "exploratory", lo, hi,
             func="v6_phase2.xw_ra_cr")
        fact("XWBBaselineCount", "crosswalk test, gpt-oss-120b: baselines compared with Map-Reflect", f"{len(BL)}",
             ("docs/naacl/results/xwb_facts.json", "v6_phase2.xw_ra_cr"), "exploratory", func="v6_phase2.xw_ra_cr")
    TABLE_ENV.setdefault("xw", "table*")
    write_tab("xw", "lrrrr", r"Selection & Acc. & Gain [95\% CI] & Map-Reflect minus [95\% CI] & Calls \\", body, v5=True,
              note_=f"Crosswalk test (exploratory): {X['XWFields'][0]} mapped fields x 4 models = {X['XWPairs'][0]} "
              f"field-model pairs, {X['XWGroups'][0]} source types as bootstrap units. Gain over the first pass; "
              "Map-Reflect minus each row, paired; calls are second-pass calls per field (first pass: none). Source "
              "docs/naacl/results/xw_facts.json (F-XW*).", label="xw")


def compute_facts():
    """The compute report (scripts/naacl/compute_report.py: per-call token logs on COMPUTE_HOST, grouped with the registry;
    DECISIONS row 89): every F-Compute* fact of docs/naacl/results/compute_facts.json with its own source, and the
    appendix table tab6_compute. The runs the paper does not report, the runs without token records and the billed
    spend are notes with no macro; the fold asserts that none of them becomes a macro."""
    f = ROOT / "docs/naacl/results/compute_facts.json"
    if not f.exists():
        return
    K = {}
    for x in json.loads(f.read_text())["facts"]:
        fid = x["id"][2:]
        assert re.fullmatch(r"Compute[A-Za-z]+", fid), fid
        if x.get("status") == "note" or re.search(r"NotReported|NotMeasured|BilledSpend", fid):
            assert x.get("status") == "note", fid  # the not-reported bucket and the spend never become macros
            note(fid, x["claim"], str(x["value"]))
            continue
        assert not NOPRINT.search(str(x["value"])) and "$" not in str(x["value"]), (fid, x["value"])
        fact(fid, x["claim"], x["value"], (x["src"], x.get("fn") or "compute_report.py"), x.get("status") or "descriptive",
             func=x.get("fn") or "compute_report.py")
        K[fid] = int(str(x["value"]).replace(",", "")) if re.fullmatch(r"[\d,]+", str(x["value"])) else x["value"]
    # anonymity (DECISIONS row 90): endpoints print as Endpoint A (ENDPOINT_A) and Endpoint B (endpoint_b); macro names keep
    # the real names, which do not print
    rows = [("gpt-oss (Endpoint A)", "GptossEndpointA"), ("Qwen3.5 (Endpoint A)", "QwenEndpointA"), ("Granite-4 (Endpoint A)", "GraniteEndpointA"),
            ("Granite-4 (Endpoint B)", "GraniteEndpointB"), ("Llama-4 (Endpoint B)", "LlamaEndpointB")]
    grp = ["Main", "Baselines", "Ablations"]

    def cell(c, t):  # thousands of calls / millions of tokens (prompt plus completion)
        return "0 / 0" if c == 0 else f"{c / 1000:.1f} / {t / 1e6:.1f}"
    body = []
    for lab, tg in rows:
        cs = [(K[f"Compute{g}{tg}Calls"], K[f"Compute{g}{tg}PromptTokens"] + K[f"Compute{g}{tg}CompletionTokens"]) for g in grp]
        tot = (sum(c for c, _ in cs), sum(t for _, t in cs))
        body.append(f"{lab} & " + " & ".join(cell(*x) for x in cs) + f" & {cell(*tot)} \\\\")
    cs = [(K[f"Compute{g}Calls"], K[f"Compute{g}PromptTokens"] + K[f"Compute{g}CompletionTokens"]) for g in grp]
    body += [r"\midrule", "Total & " + " & ".join(cell(*x) for x in cs)
             + f" & {cell(K['ComputeTotalCalls'], K['ComputeTotalPromptTokens'] + K['ComputeTotalCompletionTokens'])} \\\\"]
    TABLE_ENV.setdefault("compute", "table*")  # 69 pt too wide for a column even with short labels at 3 pt
    write_tab("compute", "lrrrr", r"Model (endpoint) & Main & Baselines & Ablations & Total \\", body, v5=True,
              note_="Compute report: thousands of LLM calls / millions of tokens (prompt plus completion) behind the "
              "reported results, from the per-call token logs (docs/naacl/results/compute_facts.json, F-Compute*). "
              "Runs the paper does not report, and runs without token records, are notes in FACTS.md (no macro).",
              label="compute")


PMFF, PMPRE = {}, {}


def pm_ff1():
    """R-PM (writer decision, DECISIONS row 34): the Prompt-Matcher part of docs/naacl/results/ff1_rerun_facts.json,
    folded as a complete block. F-PMFF* are the ff1 rerun's cells, F-PMPreFF* the stored ones they replace. The R-UQ
    and R-X1 facts in the same file are not folded here; they fold together when R-X1's gpt-oss part lands."""
    f = ROOT / "docs/naacl/results/ff1_rerun_facts.json"
    if not f.exists():
        return
    for x in json.loads(f.read_text())["facts"]:
        fid = x["id"][2:]
        if not (fid.startswith("PMFF") or fid.startswith("PMPreFF")):
            continue
        fid = re.sub(r"PM30$", "PMThirty", re.sub(r"PM50$", "PMFifty", fid))  # fact ids are letters only (macro names)
        fact(fid, x["claim"], x["value"], (x.get("src") or "docs/naacl/results/ff1_rerun_facts.json", x.get("fn") or "ff1_rerun_score"),
             x.get("status") or "exploratory", None, None, func=x.get("fn") or "ff1_rerun_score")
        PMFF[fid] = x["value"]


def ff1_uq_x1():
    """R-UQ and R-X1 (DECISIONS row 58): the rest of docs/naacl/results/ff1_rerun_facts.json, folded as one complete
    block now that all four models are in (ff97845). F-UQFF* / F-XOneFF* are the ff1 rerun's values, F-UQPreFF* /
    F-XOnePreFF* the stored ones they replace; all exploratory. The R-UQ rows are the ones v5_signals merged (the same
    results/naacl/v6/ff1rr/uq rows, F-UQFFRerunFields = F-VFiveSigFFRows); the R-X1 OMOP rows are the OMOP strategy
    arms of v5_strategies."""
    f = ROOT / "docs/naacl/results/ff1_rerun_facts.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    if set(d.get("models", [])) != {"gptoss", "granite", "llama4", "qwen35"}:
        return  # fold only the complete block
    for x in d["facts"]:
        fid = x["id"][2:]
        if fid.startswith("PM"):
            continue  # R-PM is folded by pm_ff1
        fact(fid, x["claim"], x["value"], (x.get("src") or "docs/naacl/results/ff1_rerun_facts.json",
             x.get("fn") or "ff1_rerun_score"), x.get("status") or "exploratory", x.get("lo"), x.get("hi"),
             func=x.get("fn") or "ff1_rerun_score")
        PMFF[fid] = x["value"]


# ============================================================= Codex review 3926722: feature ablation and models
def featabl():
    """The score-feature ablation of the Codex review of code 3926722 (feature_ablation.py, rerun here and identical),
    exploratory: specified after the test results were seen, intervals unadjusted. Each variant refits the benefit
    model on development only (same grouped CV and threshold rule) and is scored on the same stored test actions at
    matched calls per field."""
    f = ROOT / "docs/naacl/v6/reviews/review_3926722/feature_ablation.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    SRC = ("docs/naacl/v6/reviews/review_3926722/feature_ablation.json", "review_3926722/feature_ablation.py")
    V_ = [("all_features", "Full", "All features", "all features"),
          ("without_score_features", "NoScore", "Without score features",
           "without the score features (margin, top score, maximum-score flag, margin times truncation)"),
          ("learned_schema_only", "SchemaOnly", "Schema features only",
           "schema features only (declined, number shown, truncation, type size) with dataset and model indicators"),
          ("without_dataset_indicators", "NoDataset", "Without dataset indicators", "without the dataset indicators")]
    PI = d["paired_intervals"]
    F("FeatAblFirst", "feature ablation: first-pass test accuracy, %", f"{d['first_accuracy']:.2f}", src=SRC, fn="feature_ablation.py")
    F("FeatAblSchemaSize", "feature ablation: schema-size baseline test accuracy, %", f"{d['struct_accuracy']:.2f}", src=SRC,
      fn="feature_ablation.py")
    rows = []
    for key, tg, lab, cl in V_:
        v = d["variants"][key]
        F(f"FeatAbl{tg}Acc", f"feature ablation, benefit model with {cl}: test accuracy at matched calls per field, %",
          f"{v['matched_calls_accuracy']:.2f}", src=SRC, fn="feature_ablation.py")
        F(f"FeatAbl{tg}Calls", f"feature ablation, {cl}: calls per field at the matched budget", f"{v['matched_calls']:.3f}", src=SRC,
          fn="feature_ablation.py")
        F(f"FeatAbl{tg}VsSchema", f"feature ablation, {cl}: minus the schema-size baseline at matched calls, points",
          sg(v["matched_vs_struct"]), src=SRC, fn="feature_ablation.py")
        F(f"FeatAbl{tg}OwnAcc", f"feature ablation, {cl}: test accuracy at its own development threshold, %",
          f"{v['own_accuracy']:.2f}", src=SRC, fn="feature_ablation.py")
        F(f"FeatAbl{tg}OwnCalls", f"feature ablation, {cl}: calls per field at its own development threshold",
          f"{v['own_calls']:.2f}", src=SRC, fn="feature_ablation.py")
        F(f"FeatAbl{tg}C", f"feature ablation, {cl}: L2 inverse strength C chosen by grouped development CV", f"{v['C']}",
          src=SRC, fn="feature_ablation.py")
        diff = PI.get(f"all_features_minus_{key}")
        if diff:
            lo, hi = diff["ci95"]
            F(f"FeatAbl{tg}Diff", f"feature ablation: all features minus {cl}, at matched calls, points (paired, unadjusted)",
              sg(diff["points"]), sg(lo), sg(hi), src=SRC, fn="feature_ablation.py")
        rows.append((lab, v, diff))
    note("FeatAblProvenance", "the feature ablation", "From the Codex review of code 3926722 "
         "(docs/naacl/v6/reviews/review_3926722/feature_ablation.py and .json). Exploratory: specified after the test "
         "results were seen; intervals are paired split-group bootstraps, unadjusted. Each variant refits the benefit "
         "model on development only, with the same grouped cross-validation for C and the same threshold rule, and is "
         "scored on the same stored test reflections at matched calls per field (frozen development cost estimates, "
         "whole-field stopping). The full-feature refit reproduces the frozen model's C, threshold and accuracy, and the "
         "development cost correction leaves the threshold unchanged. Rerun here: every value identical. " + d["definition"])
    body = []
    for lab, v, diff in rows:
        df = "ref." if diff is None else f"{tex_s(diff['points'])} {tci(diff['ci95'][0], diff['ci95'][1])}"
        body.append(f"{lab} & {v['matched_calls_accuracy']:.2f} & {df} & {tex_s(v['matched_vs_struct'])} & "
                    f"{v['own_accuracy']:.2f} & {v['own_calls']:.2f} \\\\")
    write_tab("featabl", "lrrrrr", r" & \multicolumn{3}{c}{Matched calls per field} & \multicolumn{2}{c}{Own threshold} \\" + "\n"
              + r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}" + "\n"
              + r"Confidence model & Acc. & All $-$ variant [95\% CI] & $-$Schema size & Acc. & Calls \\", body,
              note_="Feature ablation of the benefit confidence model (exploratory; specified after the test results; "
              "source docs/naacl/v6/reviews/review_3926722/feature_ablation.json).", label="featabl")


def benefit_uq():
    """The benefit estimate as a UQ method (scripts/naacl/v5_benefit_uq.py, offline on the stored v5 outputs):
    calibration and discrimination of P(repair), P(break) and u(x), alternative estimates at Map-Reflect's calls, and
    the threshold. Exploratory: specified after the test results were seen. Folded with ids, intervals and statuses."""
    f = ROOT / "docs/naacl/results/v5_benefit_uq_facts.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    for x in d["facts"]:
        fact(x["id"][2:], x["claim"], x["value"], (x.get("src") or "docs/naacl/results/v5_benefit_uq_facts.json",
             x.get("fn") or "v5_benefit_uq"), x.get("status") or "exploratory", x.get("lo"), x.get("hi"),
             func=x.get("fn") or "v5_benefit_uq")


def v5_models_table():
    """Map-Reflect (the current method) per dataset and model on test: first pass, selective reflection and reflection
    on all fields, from the stored outputs rebuilt by v5_setup; core and held-out blocks, averages within each block."""
    if not V5:
        return
    core, ho = V5["core"], V5["ho"]
    MS = V5["mod"].MODELS
    order = [m for m in MODELS if m in MS]
    HON = {"mimic_omop_rematch": "MIMIC-OMOP", "semtab_m2kg_r1": "SemTab", "oaei_conference": "OAEI Conference",
           "bioml_omim_ordo": "Bio-ML", "schemanet_cprd": "SchemaNet"}

    def trip(cs, ds, m):
        c = next(x for x in cs if x.ds == ds and x.model == m)
        return [100 * c.keep.mean(), 100 * c.pol["v5"][0].mean(), 100 * c.pol["every"][0].mean()]
    body, csvrows = [rf"\multicolumn{{{1 + 3 * len(order)}}}{{l}}{{\emph{{Core datasets}}}} \\"], []  # STYLE.md: no spacer rows
    for fam, dss in FAMILY:
        for ds in dss:
            vals = [x for m in order for x in trip(core, ds, m)]
            csvrows.append([ds] + [round(x, 2) for x in vals])
            body.append(f"{V.DNAME[ds]} & " + " & ".join(f"{x:.1f}" for x in vals) + r" \\")
    body.append(rf"\multicolumn{{{1 + 3 * len(order)}}}{{l}}{{\emph{{Held-out sets}}}} \\")
    for ds, nm in HON.items():
        vals = [x for m in order for x in trip(ho, ds, m)]
        csvrows.append([ds] + [round(x, 2) for x in vals])
        body.append(f"{nm} & " + " & ".join(f"{x:.1f}" for x in vals) + r" \\")
    body.append(r"\midrule")
    for lab, cs, dss in (("Average, core", core, DATASETS), ("Average, held-out", ho, list(HON))):
        vals = [float(np.mean([trip(cs, ds, m)[k] for ds in dss])) for m in order for k in range(3)]
        body.append(f"{lab} & " + " & ".join(f"{x:.1f}" for x in vals) + r" \\")
    hdr = (" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{MSHORT[m]}}}" for m in order) + r" \\" + "\n"
           + " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(order))) + "\n"
           + "Dataset" + " & First & Selective & All" * len(order) + r" \\")
    write_tab("v5_models", "l" + "rrr" * len(order), hdr, body, env="table*",
              note_="Map-Reflect per dataset and model on test: first pass, selective reflection (benefit confidence "
              "model) and reflection on all fields; field accuracy (%), averages unweighted over datasets within each block. "
              "Source: the stored outputs in results/naacl/v6/v5/test, rebuilt by v6_phase2.v5_setup.", label="v5_models")
    wcsv("v5_models", ["dataset"] + [f"{m}_{k}" for m in order for k in ("first", "selective", "all")], csvrows)
    # compact per-model table (DECISIONS row 90): one row per model, core and held-out averages (unweighted over
    # datasets, as above) of the first pass, selective reflection and reflection on all fields
    MORD = [m for m in ("gptoss", "llama4", "granite", "qwen35") if m in order]
    MTAGC = {"gptoss": "Gptoss", "llama4": "Llama", "granite": "Granite", "qwen35": "Qwen"}
    body = []
    for m in MORD:
        vals = []
        for blk, cs, dss in (("Core", core, DATASETS), ("HO", ho, list(HON))):
            for k, nmk in enumerate(("First", "Main", "Every")):
                x = float(np.mean([trip(cs, ds, m)[k] for ds in dss]))
                vals.append(x)
                what = ("first pass", "selective reflection", "reflection on all fields")[k]
                over = "ten core datasets" if blk == "Core" else "five held-out sets"
                F(f"VFiveModel{MTAGC[m]}{blk}{nmk}", f"Map-Reflect per model, {MSHORT[m]}: {what}, unweighted mean "
                  f"over the {over}, %", f"{x:.1f}", fn="v6_phase2.v5_models_table")
        body.append(f"{MSHORT[m]} & " + " & ".join(f"{x:.1f}" for x in vals) + r" \\")
    hdr = (r" & \multicolumn{3}{c}{Core average} & \multicolumn{3}{c}{Held-out average} \\" + "\n"
           + r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" + "\n" + r"Model & First & Sel. & All & First & Sel. & All \\")
    write_tab("models_compact", "lrrrrrr", hdr, body, v5=True,
              note_="Map-Reflect per model on test: first pass, selective reflection and reflection on all fields, "
              "field accuracy (%) averaged without weights over the ten core datasets and the five held-out sets "
              "(F-VFiveModel*; per-dataset values in tab6_v5_models).", label="models_compact")


# ============================================================================== U7 sample count for answer entropy
def u7():
    """Answer entropy from the first k of the five stored draws (k = 1 to 5), Granite-4 and Llama-4 only: the U9 runs
    kept every draw (results/naacl/v6/u9/samples, 'samples' list); gpt-oss and Qwen3.5 stored only the entropy. Draws are
    clustered by answer identity as uq_baselines_v3 does (a declined or unreadable draw, '__none_<i>', is its own
    cluster); entropy = -sum p log p over the k draws. AUROC on within-model ranks with the UQ run's split-group draws;
    routing swap at v4's share against the margin on the same fields."""
    import math
    from collections import Counter
    ms = [m for m in ("granite", "llama4") if m in U.MODELS]
    if not U9ON or not all((ROOT / "results/naacl/v6/u9/samples").glob(f"*__{m}.jsonl") for m in ms):
        return
    draws, short, mism = {}, 0, 0
    for m in ms:
        for f in (ROOT / "results/naacl/v6/u9/samples").glob(f"*__{m}.jsonl"):
            for r in read(f):
                draws[(m, r["key"])] = (r.get("samples") or [], r.get("sem_entropy"))
    for m in ms:
        for f in L["pop"][m]:
            d, ent5 = draws.get((m, f["sig"]["_key"]), ([], None))
            ans = [x.get("answer") if x.get("valid") and x.get("answer") else f"__none_{i}" for i, x in enumerate(d)]
            if len(ans) < 5:
                short += 1
            for k in range(1, 6):
                a = ans[:k]
                cnt = Counter(a)
                n = len(a) or 1
                f["sig"][f"sem_k{k}"] = -max(0.0, -sum(c / n * math.log(c / n) for c in cnt.values())) if a else 0.0
            if ent5 is not None and len(ans) == 5 and abs(-f["sig"]["sem_k5"] - float(ent5)) > 1e-9:
                mism += 1
    nf = sum(len(L["pop"][m]) for m in ms)
    F("SampleKFields", f"U7: answered core-test fields of {mname(ms)} in the comparison", f"{nf:,}", fn="v6_phase2.u7")
    F("SampleKShort", "U7: fields with fewer than five stored draws", f"{short:,}", fn="v6_phase2.u7")
    F("SampleKMismatch", "U7: fields whose five-draw entropy recomputed here differs from the stored sem_entropy", f"{mism:,}",
      fn="v6_phase2.u7")
    res, _ = L["group"](ms, ["margin"] + [f"sem_k{k}" for k in range(1, 6)])
    rows, body = [], []
    for k in range(1, 6):
        s_ = f"sem_k{k}"
        a, lo, hi = auc_ci(ms, s_)
        acc, d, (dlo, dhi) = res[s_]
        W_ = WORD5[k]
        F(f"SampleK{W_}AUC", f"U7: answer entropy from the first {k} draw(s), pooled error AUROC ({mname(ms)})", f"{a:.3f}",
          f"{lo:.3f}", f"{hi:.3f}", fn="v6_phase2.u7")
        F(f"SampleK{W_}Delta", f"U7: answer entropy from the first {k} draw(s), routing at v4's share minus the margin ({mname(ms)})",
          sg(d), sg(dlo), sg(dhi), fn="v6_phase2.u7")
        rows.append((k, a, lo, hi, acc, d, dlo, dhi))
        body.append(f"{k} & {a:.3f} [{lo:.3f}, {hi:.3f}] & {tex_s(d)} {tci(dlo, dhi)} \\\\")
    am, _, _ = auc_ci(ms, "margin")
    F("SampleKMarginAUC", f"U7: the margin's pooled error AUROC on the same fields ({mname(ms)})", f"{am:.3f}", fn="v6_phase2.u7")
    write_tab("samplecount", "lrr", r"Draws $k$ & AUROC [95\% CI] & $\Delta$ vs margin [95\% CI] \\", body,
              note_="U7 (exploratory). Granite-4 and Llama-4 only: their U9 runs stored every draw; gpt-oss and Qwen3.5 "
              "stored only the five-draw entropy. With k = 1 the entropy is zero for every field, so it cannot rank them.")
    wcsv("u7_sample_count", ["k", "auroc", "lo", "hi", "accuracy", "delta_vs_margin", "delta_lo", "delta_hi"], rows)
    note("SampleKDefinitions", "U7 definitions", "Answer entropy from the first k of the five stored draws per field "
         "(results/naacl/v6/u9/samples), k = 1 to 5, for Granite-4 and Llama-4 only; gpt-oss and Qwen3.5 stored only the "
         "five-draw entropy, so they are not in U7. Draws are clustered by answer identity, a declined or unreadable draw "
         "being its own cluster, as uq_baselines_v3 scores answer entropy; the five-draw value reproduces the stored one "
         "(F-SampleKMismatch). AUROC on within-model ranks with the UQ run's split-group bootstrap; the routing swap uses "
         "v4's routed share and second pass. With k = 1 every field has entropy 0: its routing is the random gate's "
         "(ties share the budget), and its AUROC differs from 0.5 only because ranks are pooled within each model.")
    U7DONE.append(True)


U7DONE = []
WORD5 = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}


# ======================================================================= C2: MatchMaker OMOP failure audit (phase 3)
def mm_audit():
    """Every v4 test field of the three OMOP cells of Table 1's MatchMaker column (Granite-4, target type named), from
    MatchMaker's raw per-field output (results/naacl/v2_baselines/mm/<ds>__granite/out/*.jsonl: all_results with
    abstain, candidates and selected_match) and its run log. Each field falls in one class: no record, abstained,
    correct, selected outside the named target table, wrong with the gold among MatchMaker's candidates, wrong with the
    gold absent from them. Parse warnings are counted from the log. The earlier Llama-3.3 OMOP bases, run before the
    type-hint fix, are classified the same way for comparison."""
    from collections import Counter
    TAGS = {"omop_cms": "OMOPCMS", "omop_mimic": "OMOPMIMIC", "omop_synthea": "OMOPSynthea"}
    rows = []
    for ds, tg in TAGS.items():
        d = ROOT / f"results/naacl/v2_baselines/mm/{ds}__granite"
        outs = sorted((d / "out").glob("*.jsonl")) if (d / "out").exists() else []
        if not outs:
            continue
        raw = {}
        for r in read(outs[0]):
            tgt = r.get("expected_target") or {}
            tid = tgt.get("id") if isinstance(tgt, dict) else str(tgt)
            for fld, v in (r.get("all_results") or {}).items():
                raw[(str(r["sample_id"]), fld.lower())] = (v, tid)
        c = next(x for x in cells if x.ds == ds and x.model == "granite")
        cnt, ncand, sem_none = Counter(), [], 0
        for r_ in rows_of(c):
            sid, fld = str(r_["sample_id"]), r_["key"].rsplit("||", 1)[1].lower()
            gold = r_["_gold"]
            got = raw.get((sid, fld))
            if got is None:
                cnt["no record"] += 1
                continue
            v, tid = got
            cands = [x.get("target_attribute") for x in (v.get("candidates") or []) if x.get("target_attribute")]
            ncand.append(len(cands))
            sem_none += int(not any(x.get("source") in ("semantic", "both") for x in (v.get("candidates") or [])))
            sel = v.get("selected_match")
            if v.get("abstain") or not sel:
                cnt["abstained"] += 1
                top = cands[0] if cands else None  # what Table 1's base (reflect_on_baseline.prepare_matchmaker) keeps
                if top and target_correct(top, gold):
                    cnt["abstained, top candidate correct"] += 1
            elif target_correct(sel, gold):
                cnt["correct"] += 1
            elif tid and not str(sel).startswith(str(tid) + "."):
                cnt["outside the named table"] += 1
            elif any(target_correct(x, gold) for x in cands):
                cnt["wrong, gold among its candidates"] += 1
            else:
                cnt["wrong, gold not among its candidates"] += 1
            if tid and not any(str(g).startswith(str(tid) + ".") for g in (gold if isinstance(gold, list) else [gold])):
                cnt["gold outside the named table (unreachable)"] += 1
        log = (d / "run.log").read_text(errors="ignore") if (d / "run.log").exists() else ""
        parse = log.count("Failed to parse")
        n = c.n
        acc = 100 * cnt["correct"] / n
        acc_t1 = 100 * (cnt["correct"] + cnt["abstained, top candidate correct"]) / n
        assert abs(acc_t1 - PDROWS[ds]["MatchMaker"]) < 0.05, (ds, acc_t1, PDROWS[ds]["MatchMaker"])  # reproduces Table 1
        F(f"MMAudit{tg}OwnAcc", f"MatchMaker OMOP audit, {V.DNAME[ds]}: accuracy counting its abstentions as no answer, % "
          "(Table 1 instead keeps the top candidate of an abstained field)", f"{acc:.1f}", fn="v6_phase2.mm_audit", status="descriptive")
        for k, v in cnt.items():
            key = "".join(w.capitalize() for w in re.split(r"[^A-Za-z]+", k) if w)
            F(f"MMAudit{tg}{key}", f"MatchMaker OMOP audit, {V.DNAME[ds]} (Granite-4, {n} fields): {k}", f"{v}",
              src=(str(outs[0].relative_to(ROOT)), "v6_phase2.mm_audit"), fn="v6_phase2.mm_audit", status="descriptive")
        F(f"MMAudit{tg}Parse", f"MatchMaker OMOP audit, {V.DNAME[ds]}: 'Failed to parse reasoning candidates' warnings in the run log "
          "(per LLM call, not per field)", f"{parse}", src=(str((d / 'run.log').relative_to(ROOT)), "v6_phase2.mm_audit"),
          fn="v6_phase2.mm_audit", status="descriptive")
        F(f"MMAudit{tg}Cands", f"MatchMaker OMOP audit, {V.DNAME[ds]}: mean candidates per field / fields with no semantic-index "
          "candidate", f"{np.mean(ncand):.2f} / {sem_none}", fn="v6_phase2.mm_audit", status="descriptive")
        rows.append((ds, n, dict(cnt), parse, float(np.mean(ncand)), sem_none))
    # pre-fix Llama-3.3 bases: selections outside the named table (the type-hint bug's signature)
    pre = []
    for ds, tg in TAGS.items():
        f = ROOT / f"results/naacl/v2_baselines/bases/{ds}__matchmaker-llama/base.json"
        if not f.exists():
            continue
        c = next(x for x in cells if x.ds == ds and x.model == "granite")
        A = baseline_answers(f)
        outside = answered = 0
        for r_ in rows_of(c):
            a = A.get((str(r_["sample_id"]), r_["key"].rsplit("||", 1)[1].lower()))
            if not a:
                continue
            answered += 1
            tgt = r_["key"].split("|")[1]
            outside += int(not str(a).startswith(tgt + "."))
        F(f"MMAuditPreFix{tg}", f"MatchMaker OMOP audit, {V.DNAME[ds]}: the earlier Llama-3.3 run (before the type-hint fix), "
          "answered fields / of them outside the named target table", f"{answered} / {outside}", fn="v6_phase2.mm_audit",
          status="descriptive")
        pre.append((ds, answered, outside))
    if rows:
        note("MMAuditFindings", "MatchMaker OMOP audit (Codex review C2)", "; ".join(
            f"{V.DNAME[ds]}: " + ", ".join(f"{k} {v}" for k, v in sorted(cn.items(), key=lambda t: -t[1])) + f" of {n}; "
            f"{p_} parse warnings; {mc:.2f} candidates per field, {sn} fields with no semantic-index candidate"
            for ds, n, cn, p_, mc, sn in rows) + ". Earlier Llama-3.3 run (before the fix): " + "; ".join(
            f"{V.DNAME[ds]} {o} of {a} answered fields outside the named table" for ds, a, o in pre)
            + ". Scorer: full identity (target_correct), the same v4 test fields as Table 1. Table 1's MatchMaker cells use "
            "the reflection base (reflect_on_baseline.prepare_matchmaker), which keeps the top candidate of an abstained "
            "field as its answer; correct plus 'abstained, top candidate correct' reproduces each Table 1 cell, and "
            "F-MMAudit*OwnAcc gives the accuracy with abstentions counted as no answer. Why the semantic step is silent: "
            "match_maker_agent._semantic_candidate_generation asks the index (the whole OMOP catalogue: source and target "
            "tables) for top_k + |source fields| + 10 neighbours and then keeps only fields of the named target table, so "
            "on OMOP almost no neighbour survives and MatchMaker chooses among its LLM reasoning candidates alone. No field "
            "selects outside the named table after the type-hint fix (commit 9663e8a); before it most did. Parse warnings "
            "are few and per call. Fixing the semantic window would need a rerun.")


# ======================================================================================= R6 run-to-run variance
def r6_facts():
    """Fold the run agent's R6 facts (scripts/naacl/v6_calls/r6_analysis.py -> docs/naacl/results/r6_facts.json) into
    the ledger, read, never transcribed. IDs keep the run agent's names with R6 spelled RSix (letters only); a Lo/Hi key
    becomes the interval of its base fact."""
    f = ROOT / "docs/naacl/results/r6_facts.json"
    if not f.exists():
        return
    R = json.loads(f.read_text())
    SRC = ("docs/naacl/results/r6_facts.json", "v6_calls/r6_analysis.py")
    MOD = {"Qwen": "Qwen3.5", "Llama": "Llama-4", "Pooled": "both models"}
    WHAT = {"FirstAns": "first-pass answer", "FirstOk": "first-pass correctness", "Band": "routing band",
            "Routed": "routed or kept", "FinalAns": "final answer", "FinalOk": "final correctness"}

    def claim(k):
        m = re.fullmatch(r"Agree(\w+?)(Qwen|Llama|Pooled)", k)
        if m:
            return f"R6: run 1 and run 2 agree on the {WHAT[m.group(1)]}, {MOD[m.group(2)]}, % of fields"
        m = re.fullmatch(r"Gain(One|Two|Change)(Qwen|Llama|Pooled)", k)
        if m:
            w = {"One": "Map-Reflect's gain in run 1", "Two": "Map-Reflect's gain in run 2",
                 "Change": "change in Map-Reflect's gain, run 2 minus run 1"}[m.group(1)]
            return f"R6: {w}, {MOD[m.group(2)]}, points"
        m = re.fullmatch(r"FirstChange(Qwen|Llama|Pooled)", k)
        if m:
            return f"R6: change in first-pass accuracy, run 2 minus run 1, {MOD[m.group(1)]}, points"
        return {"Fields": "R6: field-model pairs compared (Qwen3.5 and Llama-4, ten core datasets)",
                "SameRanked": "R6: fields whose ranked candidate set is identical in both runs, %",
                "Cells": "R6: model-dataset cells", "CellsChangeCoversZero": "R6: cells whose gain-change interval covers zero",
                "CellsSameSign": "R6: cells whose gain has the same sign in both runs",
                "CellGainCorr": "R6: correlation of the per-cell gain between run 1 and run 2"}.get(k, f"R6: {k}")
    base = [k for k in R if not re.search(r"(Lo|Hi)$", k)]
    for k in base:
        tag = k.replace("F-R6", "", 1)
        lo, hi = R.get(k + "Lo"), R.get(k + "Hi")
        F("RSix" + tag, claim(tag), R[k], lo, hi, src=SRC, fn="v6_calls/r6_analysis.py")
    left = set(R) - set(base) - {k + s for k in base for s in ("Lo", "Hi")}
    assert not left, left
    note("RSixDefinitions", "R6 definitions", "Run 1 is the v4 test output; run 2 reran the protocol v2 first pass (same "
         "flags, temperature 0, seed 42, no response cache) and Map-Reflect's frozen second pass on the fields run 2 "
         "routed, for Qwen3.5 and Llama-4 on the ten core datasets. Full-identity scorer, ITOpsGraph v2.1, split-group "
         "bootstrap. Source: docs/naacl/results/r6_variance.md and r6_facts.json (scripts/naacl/v6_calls/r6_analysis.py).")


def r12_second(tag="gptoss", label="gpt-oss-120b"):
    """Table 1's second-backbone block (docs/naacl/v6/BASELINE2_ESTIMATE.md): the LLM baselines on gpt-oss, core and
    held-out, scored as the Granite-4 block (v4 test fields of the backbone's own rows, named target type, full-identity
    scorer, ITOpsGraph v2.1), except ToolBench, scored on the fixed-seed draw (docs/naacl/results/
    t1_gptoss_toolbench_draw.json). Magneto-BP calls no model, so its predictions are the Granite-4 block's, rescored
    on this block's populations. Prompt-Matcher: the one arm the run holds. Facts only (F-PDGptOss*); the table shape
    is the numbers pipeline's. Silent until a run exists; the Granite-4 block is untouched."""
    if not any((ROOT / p).exists() for p in (f"results/naacl/v2_baselines/bases/itopsgraph__matchmaker-{tag}",
                                              f"results/naacl/v6/anytool/itopsgraph__{tag}.json")):
        return
    draw = set(json.loads((ROOT / "docs/naacl/results/t1_gptoss_toolbench_draw.json").read_text())["sample_ids"])
    T = "GptOss"

    def pm_path(ds):
        # treatment ff1 (field-first descriptions): a set with an ff1 lookup answers into prompt_matcher_ff1
        ff1 = ROOT / f"results/naacl/v2_baselines/prompt_matcher_ff1/{ds}__{tag}/preds.json"
        return ff1 if (ROOT / f"data/{ds}_reflect_lookup_ff1.json").exists() else \
            ROOT / f"results/naacl/v2_baselines/prompt_matcher/{ds}__{tag}/preds.json"

    def at_path(ds):
        tf = ROOT / f"results/naacl/v6/anytool_tf/{ds}__{tag}.json"
        return tf if ds == "toolbench" else ROOT / f"results/naacl/v6/anytool/{ds}__{tag}.json"
    SYS = [("AnyTool", at_path, "base"),
           ("MatchMaker", lambda ds: ROOT / f"results/naacl/v2_baselines/bases/{ds}__matchmaker-{tag}/base.json", "base"),
           ("Magneto", lambda ds: ROOT / f"results/naacl/magneto/variants/{ds}__desc-bp.json", "base"),
           ("PromptMatcher", pm_path, "pm")]
    LAB = {"AnyTool": "AnyTool", "MatchMaker": "MatchMaker", "Magneto": "Magneto", "PromptMatcher": "Prompt-Matcher"}
    NOLLM = {"BMTwentyFive": "BM25", "Dense": "Dense", "Hybrid": "Hybrid", "Cupid": "Cupid", "SimFlood": "Similarity Flooding"}
    LAB.update(NOLLM)
    if V5:
        LAB.update(First="Map-Reflect first pass", Main="Map-Reflect selective", Every="Map-Reflect on all fields")
    arms, cov = set(), []
    RANS = ROOT / "results/naacl/v6/t1_retrieval"  # t1_retrieval.py's per-field answers (no model call)
    V5T = {c.ds: c for c in (V5["core"] + V5["ho"] if V5 else []) if c.model == tag}

    def ours(ds, keys):
        """Map-Reflect (method v5) on exactly the fields the baselines are scored on (on ToolBench, the draw)."""
        c5 = V5T.get(ds)
        if c5 is None:
            return {}
        m = np.array([k in keys for k in c5.keys])
        assert m.sum() == len(keys), (ds, int(m.sum()), len(keys))
        return {k: 100 * float(c5.pol[p_][0][m].mean()) for k, p_ in (("First", "keep"), ("Main", "v5"), ("Every", "every"))}

    def score(ds, rows, golds, where):
        vals = {}
        keys = [(str(r["sample_id"]), r["key"].rsplit("||", 1)[1].lower()) for r in rows]
        for nm, pf, kind in SYS:
            p = pf(ds)
            if not p.exists():
                vals[nm] = None
                cov.append(f"{nm} {where}: not run")
                continue
            if kind == "pm":
                d = json.loads(p.read_text())
                arm = next(k for k in ("pm50", "pm30") if k in d)
                arms.add(arm)
                ans = [d[arm].get(r["key"], r["base"]) for r in rows]
            else:
                A = baseline_answers(p)
                ans = [A.get(k) for k in keys]
                cov.append(f"{nm} {where}: {sum(k in A for k in keys)}/{len(rows)} fields answered")
            vals[nm] = 100 * float(np.mean([target_correct(a, g) for a, g in zip(ans, golds)]))
        # the model-free baselines rescored offline on exactly this block's fields (on ToolBench, the 400-sample draw):
        # t1_retrieval.py's stored per-field answers, scored with this block's gold
        f_ = RANS / f"{ds}.json"
        if f_.exists():
            A = {(str(a["sid"]), a["fld"].lower()): a for a in json.loads(f_.read_text())["answers"]}
            miss = [k for k in keys if k not in A]
            assert not miss, (ds, len(miss), miss[:3])  # every scored field has a model-free answer
            for m in NOLLM:
                vals[m] = 100 * float(np.mean([target_correct(A[k].get(m), g) for k, g in zip(keys, golds)]))
        return vals
    core = {}
    for ds in DATASETS:
        c = next((x for x in cells if x.ds == ds and x.model == tag), None)
        if c is None:
            continue
        rs = [r for r in rows_of(c) if ds != "toolbench" or str(r["sample_id"]) in draw]
        core[ds] = score(ds, rs, [r["_gold"] for r in rs], DT[ds])
        core[ds].update(ours(ds, {r["key"] for r in rs}))
        if ds == "toolbench":
            F(f"PD{T}ToolBenchDraw", f"Table 1, {label} block, ToolBench: samples and fields of the fixed-seed draw it is "
              "scored on", f"{len({str(r['sample_id']) for r in rs})} samples, {len(rs)} fields", fn="v6_phase2.r12_second")
    from eval.calc_accuracy import load_gold_file
    ho = {}
    for ds, tg, nm in [("mimic_omop_rematch", "Mimic", "MIMIC-OMOP"), ("semtab_m2kg_r1", "SemTab", "SemTab"),
                       ("oaei_conference", "OAEI", "OAEI Conference"), ("bioml_omim_ordo", "BioML", "Bio-ML"),
                       ("schemanet_cprd", "SchemaNet", "SchemaNet")]:
        f = ROOT / f"results/naacl/v4/test/given/{ds}__{tag}/fields.jsonl"
        if not f.exists():
            continue
        keep = set(json.load(open(ROOT / f"data/splits/{ds}.json"))["test"])
        gold = load_gold_file(ROOT / f"data/{ds}_dataset.jsonl")
        rs = [r for r in (json.loads(l) for l in open(f)) if str(r["sample_id"]) in keep]
        ho[tg] = (nm, score(ds, rs, [gold[str(r["sample_id"])][0][r["key"].rsplit("||", 1)[1]] for r in rs], tg + " (held-out)"))
        ho[tg][1].update(ours(ds, {r["key"] for r in rs}))
    for blk, rows_, lab_of, avgname in (("", {DT[d]: (V.DNAME[d], v) for d, v in core.items()}, None, "ten core datasets"),
                                         ("HO", ho, None, "five held-out sets")):
        for tg, (nm, v) in rows_.items():
            for k, x in v.items():
                if x is not None:
                    F(f"PD{T}{blk}{k}{tg}", f"Table 1, {label} block, {nm}: {LAB[k]} field accuracy, %", f"{x:.1f}",
                      fn="v6_phase2.r12_second")
        full = 10 if blk == "" else 5
        for k in LAB:
            xs = [v[k] for _, v in rows_.values()]
            if len(xs) == full and all(x is not None for x in xs):
                F(f"PD{T}{blk}{k}Avg", f"Table 1, {label} block: {LAB[k]}, unweighted mean over the {avgname}, %",
                  f"{float(np.mean(xs)):.1f}", fn="v6_phase2.r12_second")
    gptoss_table(core, ho, label)
    if arms:
        F(f"PD{T}PromptMatcherArm", f"Table 1, {label} block: the Prompt-Matcher arm run", ", ".join(sorted(arms)),
          fn="v6_phase2.r12_second")
    note(f"PD{T}Definitions", f"Table 1 {label} block definitions", "The Granite-4 block's definitions (F-PDDefinitions) "
         "and every stated adaptation carry over unchanged: the expected_target fallback, MatchMaker's named target type, "
         "the target-first candidate order for AnyTool on ToolBench, the held-out listed candidates for AnyTool and "
         "MatchMaker, Bio-ML sources shown to AnyTool with their labels, and Prompt-Matcher's pooled core-development fit "
         "(here from gpt-oss's own core calibration). Backbone gpt-oss-120b on ENDPOINT_A (gpt-oss-120b-a100). ToolBench is "
         "scored on the fixed-seed 400-sample draw (docs/naacl/results/t1_gptoss_toolbench_draw.json) in every column of "
         "this block. Prompt-Matcher runs one arm, pm30, the arm closest to the method's frozen development routed share "
         "for gpt-oss (19.3%), under the field-first description treatment (ff1) where a set has an ff1 lookup. AnyTool "
         "and MatchMaker keep their own description cuts (80 and 100 characters), as in their published designs. Further "
         "adaptation: MatchMaker on Bio-ML and Val. OpenData ran as 5 and 4 parallel shards of the sorted test population "
         "(docs/naacl/results/t1_gptoss_mm_shards.json, fixed before any shard call) because the endpoint slowed and a "
         "MatchMaker run writes nothing until it ends; each shard ran its own one-pass self-improvement (9 passes instead "
         "of 2), and each merged run holds exactly the set's population. Magneto-BP calls no model; its Granite-4-block "
         "predictions are rescored on this block's populations.")
    note(f"PD{T}Coverage", f"Table 1 {label} block baseline coverage", "; ".join(cov))


def gptoss_table(core, ho, label):
    """Table 1 on gpt-oss-120b (DECISIONS row 90): the Granite-4 table's 12-column layout (No LLM: BM25, Dense, Hybrid,
    Cupid, SimFl., Magneto; LLM-based: AnyTool, MatchMaker, Prompt-Matcher; Map-Reflect: First, Selective, All), every
    LLM column gpt-oss-120b, ToolBench on the fixed-seed 400-sample draw in every column. Written as tab6_perdataset
    once every cell has a value; plus the comparison facts (best baseline average, rows where a baseline beats or ties
    Map-Reflect's selective reflection)."""
    groups = [("No LLM", ["BMTwentyFive", "Dense", "Hybrid", "Cupid", "SimFlood", "Magneto"]),
              ("LLM-based", ["AnyTool", "MatchMaker", "PromptMatcher"]), ("Map-Reflect (ours)", ["First", "Main", "Every"])]
    ks = [k for _, g in groups for k in g]
    SL = {"BMTwentyFive": "BM25", "Dense": "Dense", "Hybrid": "Hybrid", "Cupid": "Cupid", "SimFlood": "SimFl.",
          "Magneto": "Magneto", "AnyTool": "AnyTool", "MatchMaker": r"\shortstack{Match-\\Maker}",
          "PromptMatcher": r"\shortstack{Prompt-\\Matcher}", "First": "First", "Main": "Selective", "Every": "All"}
    NAMES = {"BMTwentyFive": "BM25", "Dense": "Dense", "Hybrid": "Hybrid", "Cupid": "Cupid", "SimFlood": "Similarity Flooding",
             "Magneto": "Magneto", "AnyTool": "AnyTool", "MatchMaker": "MatchMaker", "PromptMatcher": "Prompt-Matcher"}
    rows_core = [(DT[ds], V.DNAME[ds], core.get(ds, {})) for _, dss in FAMILY for ds in dss]
    rows_ho = [(tg, nm, v) for tg, (nm, v) in ho.items()]
    if len(rows_core) != 10 or len(rows_ho) != 5 or any(v.get(k) is None for _, _, v in rows_core + rows_ho for k in ks):
        return
    avg = {k: float(np.mean([v[k] for _, _, v in rows_core])) for k in ks}
    hoavg = {k: float(np.mean([v[k] for _, _, v in rows_ho])) for k in ks}
    T = "PDGptOss"
    # comparison facts: the best baseline average (all nine baseline columns) and the rows where a baseline beats or
    # ties Map-Reflect's selective reflection at the printed precision
    base = [k for k in ks if k not in ("First", "Main", "Every")]
    for blk, av, what in (("", avg, "ten core datasets"), ("HO", hoavg, "five held-out sets")):
        b = max(base, key=lambda k: round(av[k], 1))
        F(f"{T}{blk}BestBaseAvg", f"Table 1 ({label}): the best baseline's unweighted mean over the {what}, %",
          f"{av[b]:.1f}", fn="v6_phase2.gptoss_table")
        F(f"{T}{blk}BestBaseName", f"Table 1 ({label}): the baseline with the best mean over the {what}", NAMES[b],
          fn="v6_phase2.gptoss_table")
    beats = []
    for _, nm, v in rows_core + rows_ho:
        mr = round(v["Main"], 1)
        hit = [f"{NAMES[k]} {v[k]:.1f}" for k in base if round(v[k], 1) >= mr]
        if hit:
            beats.append(f"{nm} ({', '.join(hit)}; Map-Reflect {mr:.1f})")
    F(f"{T}BeatsMainN", f"Table 1 ({label}): rows (core and held-out) where a baseline beats or ties Map-Reflect's "
      "selective reflection at one decimal", f"{len(beats)}", fn="v6_phase2.gptoss_table")
    F(f"{T}BeatsMain", f"Table 1 ({label}): those rows, with the baselines that beat or tie it",
      "; ".join(beats) or "none", fn="v6_phase2.gptoss_table")

    def line(name, v):
        best = max(round(v[k], 1) for k in ks)  # best of the printed values per row, ties all bold
        return f"{name} & " + " & ".join(rf"\textbf{{{v[k]:.1f}}}" if abs(round(v[k], 1) - best) < 0.05 else f"{v[k]:.1f}"
                                         for k in ks) + r" \\"
    body = [line(nm, v) for _, nm, v in rows_core]
    body.append(rf"\multicolumn{{{len(ks) + 1}}}{{l}}{{\emph{{Held-out sets}}}} \\")
    body += [line(nm, v) for _, nm, v in rows_ho]
    body += [r"\midrule", line("Average, core", avg), line("Average, held-out", hoavg)]
    top, rules, pos = [], [], 2
    for gname, g in groups:
        top.append(rf"\multicolumn{{{len(g)}}}{{c}}{{{gname}}}")
        rules.append(rf"\cmidrule(lr){{{pos}-{pos + len(g) - 1}}}")
        pos += len(g)
    hdr = (" & " + " & ".join(top) + r" \\" + "\n" + "".join(rules) + "\n"
           + "Dataset & " + " & ".join(SL[k] for k in ks) + r" \\")
    TABLE_ENV["perdataset"] = "table*"
    TABLE_SEP["perdataset"] = 3
    write_tab("perdataset", "l" + "r" * len(ks), hdr, body, v5=True,
              note_=f"Table 1, {label} for every LLM column. ToolBench is scored on the fixed-seed 400-sample draw "
              "(docs/naacl/results/t1_gptoss_toolbench_draw.json) in every column, the model-free columns rescored "
              "offline on the same fields; definitions F-PDGptOssDefinitions and F-PDRetrievalDefinitions.",
              label="perdataset")


# ================================================================================================= entry
SIGX = {}


def run_r2(G):
    setup(G)
    G["SIGX"] = SIGX
    v5_setup()
    r2_signals()


VSIG = {}
VSIG_KEYS = [("margin", "Margin"), ("s1", "TopScore"), ("ratio", "Ratio"), ("entropy", "ScoreEnt"), ("lp_answer", "AnsLogp"),
             ("verbalised", "Stated"), ("p_true", "PTrue"), ("sem_entropy", "SemEnt"), ("combo_paid", "ComboPaid")]
VSIG_AVAIL = {"margin": "margin", "s1": "top score", "ratio": "top-2 ratio", "entropy": "score entropy",
              "lp_answer": "answer-token log-probability", "verbalised": "stated confidence", "p_true": "P(True)",
              "sem_entropy": "answer entropy", "combo_paid": "logistic"}


def v5_signals():
    """1a (DECISIONS row 51): the confidence estimates on Map-Reflect v5's second pass, computed by another agent into
    docs/naacl/results/v5_signals_facts.json (F-VFiveSig*, no model call). Folded with their ids, intervals and
    statuses; they fill the v5 versions of tab6_signals, tab6_budget and tab6_detect (the compact Table 4 and the
    held-out table are v6_compact's). The Margin-Reflect versions stay generable as <name>_mr. ECE and Brier need paid
    estimates on development fields, which v5 has none of, so the v5 detection table has no calibration columns."""
    f = V5SIG_JSON
    if not f.exists():
        return
    for x in json.loads(f.read_text())["facts"]:
        fid = x["id"][2:]
        claim = x["claim"].split(". Provenance:")[0]
        fact(fid, claim, x["value"], (x.get("src") or "docs/naacl/results/v5_signals_facts.json", x.get("fn") or "v5_signals"),
             x.get("status") or "exploratory", x.get("lo"), x.get("hi"), func=x.get("fn") or "v5_signals")
        VSIG[fid[len("VFiveSig"):]] = (x["value"], x.get("lo"), x.get("hi"))
    if "UQFFRerunFields" in PMFF:  # the R-UQ rows folded by ff1_uq_x1 are the rows v5_signals merged
        assert int(str(PMFF["UQFFRerunFields"]).replace(",", "")) == int(VSIG["FFRows"][0].replace(",", "")), \
            (PMFF["UQFFRerunFields"], VSIG["FFRows"])
    prov = json.loads(f.read_text()).get("provenance", "")
    note("VFiveSigDefinitions", "confidence estimates on Map-Reflect v5's second pass: provenance and definitions",
         prov.replace("Provenance: ", "") + ". AUROC, AUPRC and AURC on within-model ranks; selection at Map-Reflect v5's "
         "calls per model; the change against the margin is at the same calls and fields. No ECE or Brier: v5 has no paid "
         "estimates on development fields to fit a calibration map.")
    meta = {key: (fp, bw.replace("WB (BB for Llama-4)", "WB/BB"), calls) for key, _, _, fp, bw, calls, *_ in SIGTAB}
    meta.setdefault("combo_paid", ("paid", "WB/BB", "1"))

    def v(k):
        return VSIG[k][0]

    def nmods(key, ho=""):
        n = 0
        for m in ("Gptoss", "Llama", "Granite", "Qwen"):
            a = VSIG.get(f"{ho}Avail{m}")
            if a is None:
                continue
            items, depth, cur = [], 0, ""
            for ch in a[0]:  # split the list at top-level commas (the logistic's inputs sit in parentheses)
                depth += (ch == "(") - (ch == ")")
                if ch == "," and depth == 0:
                    items.append(cur.strip())
                    cur = ""
                else:
                    cur += ch
            items.append(cur.strip())
            n += any(i.split(" (")[0] == VSIG_AVAIL[key] for i in items)
        return n

    def dci(k):
        e, lo, hi = VSIG[k]
        return f"{tex_s(e)} {tci(lo, hi)}"
    # full signals table
    body = []
    for key, tg in VSIG_KEYS:
        fp, bw, calls = meta[key]
        nm = rf"\textbf{{{SSHORT[key]}}}" if key == "margin" else SSHORT[key]
        d, dh = ("ref.", "ref.") if key == "margin" else (dci(f"{tg}SelVsMargin"), dci(f"HO{tg}SelVsMargin"))
        body.append(f"{nm} & {fp} & {bw} & {v(tg + 'AUC')} & {v('HO' + tg + 'AUC')} & {d} & {dh} & "
                    f"{v(tg + 'SelOracleShare')}\\% \\\\")
    TABLE_SEP["signals"] = 3  # 4.2 pt too wide at 4 pt
    # no Calls column: it repeats Cost (a free estimate makes no call, a paid one makes one); 30.7 pt too wide with it
    write_tab("signals", "lllrrrrr", r"UQ method & Cost & Access & Test AUROC & Held-out & $\Delta$ vs margin [95\% CI] "
              r"& Held-out $\Delta$ [95\% CI] & Oracle share \\", body, env="table*", v5=True,
              note_="1a (exploratory). Map-Reflect v5's second pass: docs/naacl/results/v5_signals_facts.json "
              "(F-VFiveSig*); selection at v5's calls per model; definitions F-VFiveSigDefinitions.")
    # budget
    body = []
    for key, tg in VSIG_KEYS:
        r = [v(f"{tg}RateAt{w}") for w in ("Ten", "Twenty", "Thirty", "Fifty")] + [v(f"{tg}RateMean")]
        body.append(f"{SSHORT[key]} & {nmods(key)} & " + " & ".join(tex_s(x) for x in r) + f" & {v(tg + 'RateReach')} \\\\")
    write_tab("budget", "lrrrrrrr", r"UQ method & Models & 10\% & 20\% & 30\% & 50\% & Mean & Rate for 95\% (\%) \\", body,
              env="table*", v5=True, note_="1a (exploratory). Change over the first pass (points) when each estimate selects "
              "a share of every model's fields for Map-Reflect v5's second pass; core test, common valid population.")
    # detection
    body = []
    for key, tg in VSIG_KEYS:
        e, lo, hi = VSIG[tg + "AUC"]
        body.append(f"{SSHORT[key]} & {nmods(key)} & {e} [{lo}, {hi}] & {v('HO' + tg + 'AUC')} & {v(tg + 'AUPRC')} & "
                    f"{v(tg + 'AURC')} \\\\")
    write_tab("detect", "lrrrrr", r"UQ method & Models & Test AUROC [95\% CI] & Held-out & AUPRC & AURC \\", body, env="table*",
              v5=True, note_="1a (exploratory). Error detection on the common valid population of Map-Reflect v5's test "
              "rows, within-model ranks; no ECE or Brier (no paid estimates on v5 development fields).")


VSTRAT = {}
# the strategies, named by their parts (1b, DECISIONS row 53); Routed is v5's own second pass, the action its rule names
STRATS = [("Doubt", "Doubt"), ("SelfRefine", "Self-Refine"), ("Neutral", "Neutral"), ("NeutralCheck", "Neutral + check"),
          ("Evidence", "Neutral + evidence"), ("NeutralEvidenceCheck", "Neutral + evidence + check"),
          ("Widen", "Widen + reflect"), ("Routed", "Reflect or widen + reflect")]


def v5_strategies():
    """1b (DECISIONS row 53): the reflection strategies on Map-Reflect v5's test fields and selection, computed by
    another agent into docs/naacl/results/v5_strategies_facts.json (F-VFiveStrat*, scripts/naacl/v5_strategies.py).
    Folded with ids, intervals and statuses; they fill the v5 versions of tab6_interventions (Table 3) and
    tab6_interventions_counts and the C9 data. C2's curves (change against reflection rate, fields in u(x) order and
    in random order) need per-field outcomes, so they are rebuilt here from v5_setup's cells with v5_strategies.fill
    (the same arm rows), after checking that every strategy's every-field change equals the folded fact. Each
    dataset's arms are the same version for all four models (v5_strategies.complete); a dataset still on the stored
    arms is named with its input difference in F-VFiveStratInputs and in the tables' header comments."""
    f = ROOT / "docs/naacl/results/v5_strategies_facts.json"
    if not f.exists() or not V5:
        return
    d = json.loads(f.read_text())
    for x in d["facts"]:
        fid = x["id"][2:]
        fact(fid, x["claim"], x["value"], (x.get("src") or "docs/naacl/results/v5_strategies_facts.json",
             x.get("fn") or "v5_strategies"), x.get("status") or "exploratory", x.get("lo"), x.get("hi"),
             func=x.get("fn") or "v5_strategies")
        VSTRAT[fid[len("VFiveStrat"):]] = (x["value"], x.get("lo"), x.get("hi"))
    by_ds, nre = {}, {}
    for k, src in d["arm_sources"].items():
        # the rerun of a dataset covers each model's fields whose prompt changed, so the count can differ by model
        # (ITOpsGraph); the input version is the text without the count, which must be one per dataset
        ds_ = k.rsplit("__", 1)[0]
        m_ = re.search(r", (\d+) fields;", src)
        if m_:
            nre.setdefault(ds_, []).append(int(m_.group(1)))
            src = src.replace(m_.group(0), ", {n};")
        by_ds.setdefault(ds_, set()).add(src)
    assert all(len(v_) == 1 for v_ in by_ds.values()), by_ds  # each dataset on one input version for all four models
    scored = {VSTRAT[k][0] for k in VSTRAT if k.startswith("ITopsRerun")}
    if "itopsgraph" in nre and len(scored) == 1:  # the scored fields rerun (the arm-source count includes unscored ones)
        by_ds["itopsgraph"] = {next(iter(by_ds["itopsgraph"])).replace("{n}", f"{next(iter(scored))} scored fields")}
        del nre["itopsgraph"]
    for ds_, ns in nre.items():
        lo_, hi_ = min(ns), max(ns)
        by_ds[ds_] = {next(iter(by_ds[ds_])).replace("{n}", f"{lo_} fields" if lo_ == hi_ else f"{lo_} to {hi_} fields per model")}
    groups = {}
    for ds_, srcs in by_ds.items():
        groups.setdefault(next(iter(srcs)), []).append(V.DNAME.get(ds_, ds_))
    inputs = "; ".join(f"{', '.join(sorted(v_))}: {k}" for k, v_ in sorted(groups.items()))
    note("VFiveStratInputs", "reflection strategies on v5: each dataset's arm inputs (the same for all four models)", inputs)
    note("VFiveStratDefinitions", "reflection strategies on v5: definitions", "Each strategy's answer replaces the first "
         "answer on the fields it is applied to. Doubt, Self-Refine and the four neutral arms (neutral reflection, with "
         "the pairwise check, with symmetric evidence, with both) reflect over the first ranking's candidates. Widen + "
         "reflect is v5's second pass with widening (neutral reflection with symmetric evidence and the either-order "
         "check over the widened pool) on every field. Reflect or widen + reflect is v5's second pass with the action "
         "its rule names per field. Selections: every field; v5's own selection (u(x) above its frozen tau, "
         f"{VSTRAT['RateShare'][0]}% of fields); and a random selection of the same share in each cell (in expectation). "
         "Gains against the first pass on the v5 test fields, pooled over field-model pairs, split-group bootstrap "
         "(B = 2,000). Arm outcomes read with the protocol v2 parser except the rerun cells (v5's parser).")
    v = lambda k: VSTRAT[k][0]  # noqa: E731

    def dci(k):
        e, lo, hi = VSTRAT[k]
        return f"{tex_s(e)} {tci(lo, hi)}"
    hc = (("Caption: Every strategy runs on Map-Reflect's inputs (source metadata, target names and full descriptions). "
           f"Arm inputs per dataset: {inputs}.") if str(VSTRAT.get("CellsOnStoredArms", ("",))[0]).strip() == "none"
          else f"Arm inputs per dataset (state in the caption where not v5's): {inputs}.")
    body, body2, c9 = [], [], []
    for tg, lab in STRATS:
        body.append(f"{lab} & {dci(tg + 'EveryGain')} & {tex_s(v(tg + 'RateGain'))} & {tex_s(v(tg + 'RateRandomGain'))} & "
                    f"{dci(tg + 'RateMinusRandom')} \\\\")
        rep, brk = (int(x.replace(",", "")) for x in v(tg + "EveryRepairs").split(" / "))
        first = float(v(tg + "EveryAcc")) - float(v(tg + "EveryGain"))
        body2.append(f"{lab} & {first:.1f} & {float(v(tg + 'EveryAcc')):.1f} & {tex_s(v(tg + 'EveryGain'))} & {tnum(rep)} & "
                     f"{tnum(brk)} \\\\")
        c9.append(("strategy", lab, rep, brk))
    write_tab("interventions", "lrrrr", r"Strategy & Every field [95\% CI] & $u(x)$, same rate & Random, same rate & "
              r"$u(x) - $Random [95\% CI] \\", body, env="table*", v5=True,
              note_="1b (exploratory). Change over the first pass (points) on Map-Reflect v5's test fields: each strategy "
              f"on every field and on the fields v5's benefit model selects ({v('RateShare')}%), against a random "
              f"selection of the same share. Definitions F-VFiveStratDefinitions. {hc}")
    write_tab("interventions_counts", "lrrrrr", r"Strategy & First & Accuracy & Change & Repaired & Broken \\", body2,
              env="table*", v5=True, note_=f"1b (exploratory). Each strategy on every v5 test field ({v('Fields')} "
              f"field-model pairs): accuracy (%), change (points), repairs and breaks. {hc}")
    wcsv("v5_c9_strategies", ["group", "label", "repairs", "breaks"], c9)
    # C2: change against reflection rate, fields in u(x) order within each cell, against random order
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        import v5_strategies as VS
    finally:
        sys.argv = argv
    cells = V5["core"]
    for c in cells:
        VS.fill(c, "v5", {}, {})
    N = sum(c.n for c in cells)
    for tg, _ in STRATS:
        g = 100 * sum((c.arm[tg] - c.keep).sum() for c in cells) / N
        if abs(g - float(v(tg + "EveryGain"))) > 0.006:
            print(f"C2 not written: {tg} every-field change {g:+.3f} here, {v(tg + 'EveryGain')} in the facts "
                  "(the arm rows changed since v5_strategies_facts.json was written)")
            return
    rows = []
    for tg, lab in STRATS:
        gain = float(v(tg + "EveryGain"))
        reg = "harmful" if gain < -0.05 else ("repairing" if gain > 1.0 else "nothing to repair")
        for b in range(0, 101, 5):
            ch = rn = 0.0
            for c in cells:
                dlt = c.arm[tg] - c.keep
                k = int(round(b / 100 * c.n))
                order = np.argsort(-c.u, kind="mergesort")
                ch += dlt[order[:k]].sum()
                rn += b / 100 * dlt.sum()
            rows.append([tg, reg, b / 100, round(100 * ch / N, 6), round(100 * rn / N, 6)])
    wcsv("v5_c2_strategies", ["intervention", "regime", "share", "change_u", "change_random"], rows)


def run_rest(G):
    setup(G)
    itops_facts()
    itops_v3_facts()
    itops_human_overlap()
    rev2_facts()
    latency_facts()
    rb_facts()
    evab_facts()
    xw_ra_cr()
    compute_facts()
    v5_facts()
    t1_retrieval()
    pm_ff1()
    ff1_uq_x1()
    r12()
    r12_second()
    mm_audit()
    d1_block()
    featabl()
    benefit_uq()
    v5_models_table()
    margin_full_pop()
    d2_facts()
    bl_blank_facts()
    r6_facts()
    u7()
    u1()
    u2()
    u3()
    u4()
    u5()
    u6()
    u8()
    u10()
    chart_data()
    r5()
    v5_signals()
    v5_strategies()
    for fid, claim, text in (
        ("BudgetDefinitions", "U1 definitions", "Uq population (answered core-test fields with every signal of their model, "
         "uq_ablation_v4.run's own population and v4 second pass). Each signal routes its least confident share b of every "
         "model's fields, b = 0 to 100% in steps of 5, ties sharing the boundary (uq_ablation.gate_weights); random is the "
         "exact expectation; the oracle routes the fields the action repairs first. Signals on fewer models are curved on "
         "those models, with the margin's curve on the same models beside them. Reach: smallest b whose change is at least "
         "95% of the every-field change."),
        ("DetDefinitions", "U2 definitions", "AUROC, AUPRC (errors positive) and AURC (risk-coverage, most confident first) on "
         "within-model percentile ranks pooled over the models that have the signal; AUROC interval from the UQ run's own "
         "split-group bootstrap draws. ECE (10 equal-width bins) and Brier after an isotonic map fitted per model on the old "
         "development fields (manifest old_dev_excluded: the only development fields that carry the paid signals). The "
         "token log-probability has no development outputs and the logistic is cross-fitted on test, so neither is "
         "calibrated. Held-out pooled AUROC: the 20 held-out cells of uq_ablation_v4.heldout_aurocs, ranks within model."),
        ("MVarDefinitions", "U3 definitions", "From the first pass's sorted score list: raw margin s1 - s2; normalised margin "
         "(s1 - s2) / sum of the non-negative scores; ratio log((s1 + 0.001) / (s2 + 0.001)); score entropy (negated, "
         "normalised); top-1 minus top-3 gap s1 - s3. Routing at v4's share with the v4 second pass, uq population."),
        ("RPartDefinitions", "U4 definitions", "Thresholds refit on the v4 development split (results/naacl/v4/dev/given, "
         "full identity, ITOpsGraph v2.1; reproduces the frozen development accuracy exactly) with each model's frozen "
         "widen action and the check, then applied to test. Margin only: no near-tie band, every routed field widens, t_m "
         "refit on v4_fit's quantile grid. Top score only: a field is routed when declined or its top score is at most t, t "
         "refit on the top-score quantile grid. Shared: one t_m on the 0.05 grid for all models (pooled development "
         "accuracy, ties to fewer calls). Development size: t_m and t_s refit per model on v4_fit's grids using a random "
         "10, 25 or 50% of the development samples (50 draws each); 100% reproduces the frozen thresholds."),
        ("CovDefinitions", "U5 definitions", "Struct sends declined fields and fields of a named type wider than the 20 shown "
         "candidates to the frozen widen action; margin is Map-Reflect; union and intersection combine the two field sets "
         "(the union's struct-only fields widen); the logistic predicts a first-pass error from the struct flag, the "
         "declined flag and the margin, cross-fitted per model over split groups (2 folds x 5 repeats), and routes "
         "Map-Reflect's per-model share. Costs from the v4 token logs as Table 1."),
        ("AcceptDefinitions", "U6 definitions", "Map-Reflect's routing and per-field action; only the acceptance of the "
         "reflection's proposal (field_proposed) changes. No check: take the proposal. Check: the stored v4 answer. Own "
         "margin: take a changed proposal only when the reflection's revised margin exceeds the first-pass margin (v4_fit's "
         "'check+margin' criterion, without the check). Both: the check and the margin rule. Near-tie proposals store no "
         "revised margin, so there the last two rules use the check. A check call follows a changed answer."),
        ("LogpDefinitions", "U8 definitions", "Stored per field by uq_baselines_v3 (per-token values were not kept): "
         "lp_answer and lp_answer_min are the sum and minimum log-probability of the tokens spelling the first ranked "
         "candidate's id (candidate tokens only); lp_mean and lp_entropy are the mean token log-probability and mean top-5 "
         "entropy over every token up to the end of that candidate (scores, formatting and any preamble included). So the "
         "candidate-only signal an earlier review asked for is the paper's lp_answer; U8 compares it with the all-token "
         "versions."),
        ("SemEntSeeds", "answer entropy: sampling seeds", "The endpoint_b client sends a fixed RANDOM_SEED (42) with sampling "
         "decoding, so five draws of one prompt come back identical. We gave each draw its own seed (42 to 46) at temperature "
         "0.7, which makes them five independent draws, as the ENDPOINT_A draws for gpt-oss and Qwen3.5 are."),
        ("MCalDefinitions", "U10 definitions", "Isotonic maps fitted per model on the answered v4 development fields "
         "(margin, and the top score for comparison), applied to the answered test fields. Raw margin taken as a "
         "probability as is (it lies in [0, 1]). ECE with 10 equal-width bins; Brier = mean squared error."),
    ):
        note(fid, claim, text)
    NEEDS_RUNS.extend([
        ("U7 answer-entropy sample count k = 1 to 5", "the five samples per field were not stored (only entropy and count)",
         "5 samples per answered core-test field for gpt-oss and Qwen3.5: 2 x ~3,000 x 5 = ~30,000 calls (or store samples in U9)"),
        ("U9 answer entropy for Granite-4 and Llama-4", "never collected", "2 x ~3,200 answered core-test fields x 5 = ~32,000 calls"),
        ("U9 / R2 answer-token log-probability for Llama-4", "Llama-4 endpoint returned no log-probabilities",
         "1 greedy regeneration per answered field, ~3,300 calls, if a log-probability endpoint exists"),
        ("R6 run-to-run variance", "a second run", "first pass + second pass for 2 models x 10 datasets: ~2 x 3,350 first-pass "
         "calls + ~2 x 3,350 x 0.48 second-pass calls = ~10,000 calls"),
        ("C14 run 1 vs run 2 chart", "R6", "none beyond R6"),
    ])
    if U9ON:  # U9 has landed: answer entropy and log-probabilities cover all four models
        NEEDS_RUNS[:] = [x for x in NEEDS_RUNS if not x[0].startswith(("U9",))]
    if U7DONE:  # U7 computed offline from the stored U9 draws (Granite-4 and Llama-4)
        NEEDS_RUNS[:] = [x for x in NEEDS_RUNS if not x[0].startswith("U7")]
    if (ROOT / "docs/naacl/results/r6_facts.json").exists():  # R6 and C14 have landed
        NEEDS_RUNS[:] = [x for x in NEEDS_RUNS if not x[0].startswith(("R6", "C14"))]
    note("NeedsRuns", "items that need model calls (phase 2)", "; ".join(f"{a}: {b} ({c})" for a, b, c in NEEDS_RUNS))
    wcsv("needs_runs", ["item", "missing", "estimated_calls"], NEEDS_RUNS)
