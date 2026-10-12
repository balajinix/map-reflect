#!/usr/bin/env python3
"""Render a fixed sample of reflection prompts on every dataset, without any model call.

I use this to check the field-first description treatment (MAPREFLECT_FIELD_FIRST, scripts/naacl/field_first.py):
with the treatment off, every prompt must be byte-identical to the snapshot taken on the code before the treatment
existed (tests/fixtures/ff1_default_prompt_md5.json). With it on, the same records render the treatment's cards.

The sample, per dataset: up to N_PER records of the stored protocol v2 first pass (results/naacl/v2/first/given,
granite when present), evenly spaced over the development and test fields in key order. A dataset without a local
first pass gets synthetic records from its converted dataset (the first 20 target properties, fixed scores), which
exercise the same prompt code. Per record I render every reflection prompt the runs use:
  reflect_v2  naive, neutral (A), neutral with evidence (B), Self-Refine feedback and refine, verify, verify with evidence
  reflect_v3  neutral with evidence over the WT20 pool (data/v4_pools, reflect_v4.pool_of): the widen action
  v6_d3_run   symmetric with and without scores, the blind verifier (both orders)
under MAPREFLECT_V41 off and on (the target name changes the prompt).

Usage (from the repo root):
  .venv/bin/python scripts/naacl/ff1_render.py --snapshot tests/fixtures/ff1_default_prompt_md5.json
  .venv/bin/python scripts/naacl/ff1_render.py --check tests/fixtures/ff1_default_prompt_md5.json
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
import reflect_v2 as R2  # noqa: E402
from reflect_v3 import Prompts as P3  # noqa: E402
from reflect_v4 import pool_of  # noqa: E402
from v6_d3_run import D3Prompts  # noqa: E402

MODEL = "endpoint_b/vendor/granite-4-h-small"
N_PER = 10
TAGS = ("granite", "gptoss", "qwen35", "llama4")
FEEDBACK = "The previous answer may confuse a timestamp with a date."


def datasets():
    return sorted(p.stem for p in (ROOT / "data/splits").glob("*.json") if lookup_path(p.stem))


def lookup_path(ds, ff=False):
    """The reflection lookup the runs use (run_v41r.sh, run_d1.sh): the v4.1 one when it exists, else the api one.
    ff=True: the field-first lookup (data/<ds>_reflect_lookup_ff1.json), when it exists."""
    if ff and (ROOT / f"data/{ds}_reflect_lookup_ff1.json").exists():
        return ROOT / f"data/{ds}_reflect_lookup_ff1.json"
    for p in (ROOT / f"data/{ds}_reflect_lookup_v41.json", ROOT / f"data/{ds}_api_lookup.json"):
        if p.exists():
            return p
    return None


def first_pass(ds):
    for tag in TAGS:
        d = R2.load_slot(str(ROOT / f"results/naacl/v2/first/given/{ds}__{tag}"))
        if d is not None:
            return d, tag
    return synthetic(ds), "synthetic"


def synthetic(ds):
    """A first-pass shaped result built from the converted dataset: each gold source field ranks the first 20
    target properties with descending scores. Prompt code only; no accuracy meaning."""
    p = ROOT / f"data/{ds}_dataset.jsonl"
    if not p.exists():
        return None
    out = []
    for line in p.open():
        s = json.loads(line)
        ts = s.get("target_schema")
        if not isinstance(ts, dict):
            continue
        props = [x["name"] for x in ts.get("properties", [])][:R2.N_CAND]
        if len(props) < 2:
            continue
        rank = [{"id": c, "score": round(0.95 - 0.04 * i, 2), "reason": "fixed"} for i, c in enumerate(props)]
        out.append(dict(sample_id=s["id"], expected_source={"id": s.get("source_schema")},
                        expected_target={"id": ts.get("name")}, ground_truth_mapping=s.get("mapping", {}),
                        predictions={f: {"initial_ranking": rank} for f in s.get("mapping", {})}))
    return {"results": out}


def sample(ds, n=N_PER):
    d, tag = first_pass(ds)
    if d is None:
        return [], tag
    man = json.load(open(ROOT / f"data/splits/{ds}.json"))
    keep = set(man.get("dev", [])) | set(man.get("test_grouped", man.get("test", [])))
    if man.get("all_test") or tag == "synthetic":
        keep = {str(r.get("sample_id")) for r in d["results"]}
    recs = sorted((r for r in R2.records(d, keep) if r["valid"] and len({c["id"] for c in r["ranking"]}) >= 2),
                  key=lambda r: r["key"])
    if len(recs) > n:
        step = len(recs) / n
        recs = [recs[int(i * step)] for i in range(n)]
    return recs, tag


def prompts_for(ds, rec, lookup, pools):
    """name -> prompt text for one record."""
    out = {}
    p2, p3 = R2.Prompts(MODEL, lookup), P3(MODEL, lookup)
    try:
        d3 = D3Prompts(MODEL, lookup, ds)
    except KeyError:  # D3 states a rule only for its own datasets; any rule exercises the same cards
        d3 = D3Prompts(MODEL, lookup, "omop_heldout")
    alt = next(str(c["id"]) for c in rec["ranking"] if str(c["id"]) != rec["base"])
    out["naive"] = p2.naive(rec)[0]
    out["A"] = p2.neutral(rec)[0]
    out["B"] = p2.neutral(rec, with_evidence=True)[0]
    out["sr_feedback"] = p2.sr_feedback(rec)
    out["sr_refine"] = p2.sr_refine(rec, FEEDBACK)[0]
    out["verify"] = p2.verify(rec, rec["base"], alt)[0]
    out["verify_ev"] = p2.verify(rec, rec["base"], alt, with_evidence=True)[0]
    cl = pool_of("WT20", rec, pools) if pools is not None else None
    pp = pools.get(f"{rec['sample_id']}||{rec['field']}", {}) if pools is not None else {}
    r = dict(rec, cand_list=cl, shown=pp.get("shown") or [str(c["id"]) for c in rec["ranking"]])
    out["W_B"] = p3.neutral(r, with_evidence=True)[0]
    out["W_verify_ev"] = p3.verify(r, rec["base"], alt, with_evidence=True)[0]
    out["symS"] = d3.symmetric(r, scores=True)[0]
    out["symN"] = d3.symmetric(r, scores=False)[0]
    out["blind_c1"] = d3.blind(r, rec["base"], alt)
    out["blind_c2"] = d3.blind(r, alt, rec["base"])
    return out


def render(ds, ff_lookup=False, v41=False, n=N_PER):
    """{record key: {prompt name: text}} for one dataset under the current environment's treatment flags."""
    old = os.environ.get("MAPREFLECT_V41")
    os.environ["MAPREFLECT_V41"] = "1" if v41 else "0"
    try:
        recs, _ = sample(ds, n)
        lookup = json.load(open(lookup_path(ds, ff=ff_lookup)))
        pp = ROOT / f"data/v4_pools/{ds}.json"
        pools = json.load(open(pp)) if pp.exists() else None
        return {r["key"]: prompts_for(ds, r, lookup, pools) for r in recs}
    finally:
        if old is None:
            os.environ.pop("MAPREFLECT_V41", None)
        else:
            os.environ["MAPREFLECT_V41"] = old


def md5s(ds, **kw):
    return {k: {n: hashlib.md5(t.encode()).hexdigest() for n, t in v.items()} for k, v in render(ds, **kw).items()}


def snapshot():
    out = {}
    for ds in datasets():
        for v41 in (False, True):
            m = md5s(ds, v41=v41)
            if m:
                out[f"{ds}|v41={int(v41)}"] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--snapshot", help="write the md5 of every sampled prompt to this file")
    g.add_argument("--check", help="compare the current rendering with this snapshot")
    a = ap.parse_args()
    if os.environ.get("MAPREFLECT_FIELD_FIRST") == "1":
        sys.exit("the snapshot and the check are of the default prompts: unset MAPREFLECT_FIELD_FIRST")
    cur = snapshot()
    if a.snapshot:
        Path(a.snapshot).parent.mkdir(parents=True, exist_ok=True)
        Path(a.snapshot).write_text(json.dumps(cur, indent=1, sort_keys=True) + "\n")
        n = sum(len(v) * len(next(iter(v.values()))) for v in cur.values())
        print(f"wrote {n} prompt md5s over {len(cur)} dataset/flag cells to {a.snapshot}")
        return
    ref = json.load(open(a.check))
    bad = [(c, k, p) for c in ref for k in ref[c] for p in ref[c][k]
           if cur.get(c, {}).get(k, {}).get(p) != ref[c][k][p]]
    n = sum(len(v) * len(next(iter(v.values()))) for v in ref.values())
    print(f"{n - len(bad)}/{n} default prompts byte-identical over {len(ref)} dataset/flag cells")
    if bad or set(cur) != set(ref):
        print("DIFFER:", bad[:10], set(cur) ^ set(ref))
        sys.exit(1)


if __name__ == "__main__":
    main()
