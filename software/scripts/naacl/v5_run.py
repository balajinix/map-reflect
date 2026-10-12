#!/usr/bin/env python3
"""Method v5 second pass (docs/naacl/v6/PLAN_PHASE2.md, "Method v5"): the improved reflection and the either-order
verifier, on every field of a split, for the actions the routers and controls need.

v5 = method v4.1 (MAPREFLECT_V41=1, MAPREFLECT_CANONICAL_MARGIN=1) plus
  * the description repair (treatment ff1, MAPREFLECT_FIELD_FIRST=1) wherever a data/<ds>_reflect_lookup_ff1.json
    exists (the launcher sets the flag per dataset; other datasets keep their v4.1 or default lookup, whose prompts
    the treatment does not change);
  * the improved reflection: v6_d3_run.D3Prompts.symmetric(scores=False), symmetric evidence for the leading old and
    new candidates, the dataset's correspondence rule instead of the uniqueness hint, no old scores (D3 arm symN);
  * the either-order verifier: v6_d3_run.D3Prompts.blind in both option orders; a proposal that changes a valid
    first answer is accepted when either order picks it (D3 rule blind_either). A proposal on a declined field has no
    first answer to defend and stands, as in method v4.

Actions per field (pools as reflect_v4.pool_of over data/v4_pools; the first pass is the protocol v2 first pass):
  NT  over the first ranking's candidates (no call on an invalid first pass)
  W   the model's frozen widen action of method v4.1 (WT20 or WT50, docs/naacl/results/v4_1_frozen.json)
--actions both runs NT and W on every field (development: what the benefit router and every control need); --actions
test runs the routed action (router.action under the v4.1 thresholds) on every field, and W as well where that action
is NT (struct, the union and the widen outcome of the benefit model). Two actions whose candidate lists coincide are
called once and share the outcome. --with-check also calls the current pairwise check (reflect_v2.Prompts.verify) on
each judged proposal, so that "new prompt + current check" is measured on the same reflections (development only).

Row fields per action X in {NT, W}: X_pool, X_gold_in_pool, X_proposed, X_rmargin, X_parse, X_c1 and X_c2 (the option
each verifier order picked: c1 shows the first answer as option 1, c2 the proposal as option 1), X_v5 (the answer under
the either-order rule), X_check and X_chk (the current check's decision and answer, with --with-check), X_same_as.
Raw responses: <out>.raw.jsonl (with the ff1 fields when the treatment is on). Scoring is offline (v5.py).

Usage (COMPUTE_HOST, through run_v5.sh):
  MAPREFLECT_V41=1 MAPREFLECT_CANONICAL_MARGIN=1 [MAPREFLECT_FIELD_FIRST=1] TOKEN_LOG=... .venv/bin/python \
      scripts/naacl/v5_run.py --dataset omop_cms --model-tag qwen35 --split dev --actions both --with-check \
      --out results/naacl/v6/v5/dev/omop_cms__qwen35.jsonl
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
import reflect_v2 as R2  # noqa: E402
import router  # noqa: E402
from eval.calc_accuracy import target_correct  # noqa: E402
from map_reflect.map_reflect_runner import set_usage_context, set_usage_stage  # noqa: E402
from reflect_v3 import revised  # noqa: E402
from reflect_v4 import pool_of  # noqa: E402
import field_first as FF  # noqa: E402
import ff1_log  # noqa: E402
from v6_d3_run import MODELS, ENDPOINT_A_EP, V41_LOOKUP, D3Prompts  # noqa: E402
import run_guard as RG  # noqa: E402


def lookup_path(ds):
    if FF.on():
        return ROOT / f"data/{ds}_reflect_lookup_ff1.json"
    return ROOT / (f"data/{ds}_reflect_lookup_v41.json" if ds in V41_LOOKUP else f"data/{ds}_api_lookup.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-tag", required=True, choices=sorted(MODELS))
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--actions", choices=["both", "test"], required=True)
    ap.add_argument("--with-check", action="store_true")
    ap.add_argument("--keys", help="run only these record keys (one per line)")
    ap.add_argument("--frozen", default="docs/naacl/results/v4_1_frozen.json")
    ap.add_argument("--base", help="first-pass slot (default results/naacl/v2/first/given/<ds>__<model>; D1: its merged "
                                   "hybrid first pass)")
    ap.add_argument("--pools", help="pools json (default data/v4_pools/<ds>.json; D1: data/v6_d1_pools/<ds>.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if os.environ.get("MAPREFLECT_V41") != "1" or not router.CANONICAL:
        sys.exit("v5 runs on method v4.1: set MAPREFLECT_V41=1 and MAPREFLECT_CANONICAL_MARGIN=1")
    ds, tag = a.dataset, a.model_tag
    model = MODELS[tag]
    if tag in ENDPOINT_A_EP:
        os.environ["ENDPOINT_A_ENDPOINT"] = ENDPOINT_A_EP[tag]
    t = json.load(open(ROOT / a.frozen))["thresholds"][tag]
    W = t["widen"]
    man = json.load(open(ROOT / f"data/splits/{ds}.json"))
    keep = set(man["dev"]) if a.split == "dev" else set(man["test"])
    d = R2.load_slot(a.base or str(ROOT / f"results/naacl/v2/first/given/{ds}__{tag}"))
    if a.split == "test" and man.get("all_test"):  # a held-out set below 100 fields: every sample is test
        keep = {str(r.get("sample_id")) for r in d["results"]}
    lk = lookup_path(ds)
    lookup = json.load(open(lk))
    ff = FF.on()
    if ff:
        FF.require(lookup, lk)
    lookup_sha = hashlib.sha256(lk.read_bytes()).hexdigest()
    pools = json.load(open(a.pools or ROOT / f"data/v4_pools/{ds}.json"))
    recs = list(R2.records(d, keep))
    if a.keys:
        want = {l.strip() for l in open(a.keys) if l.strip()}
        recs = [r for r in recs if r["key"] in want]
    if a.limit:
        recs = recs[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = str(out) + ".raw.jsonl"
    # resume guard: rows already in <out> must come from this exact configuration (scripts/naacl/run_guard.py)
    RG.check(out, RG.fingerprint(
        dict(lookup=lk, pools=a.pools or f"data/v4_pools/{ds}.json", frozen=a.frozen, keys=a.keys,
             base=a.base or f"results/naacl/v2/first/given/{ds}__{tag}"),
        dict(script="v5_run.py", dataset=ds, model=model, split=a.split, actions=a.actions, with_check=a.with_check,
             widen=W, limit=a.limit)))
    done = {json.loads(l)["key"] for l in out.open()} if out.exists() else set()
    recs = [r for r in recs if r["key"] not in done]
    print(f"{ds}__{tag} {a.split}: {len(done)} done, {len(recs)} to run; lookup {lk.name}; ff1 {ff}; widen {W}; "
          f"actions {a.actions}; check {a.with_check}", flush=True)
    meta = dict(script="v5_run.py", base=a.base, pools=a.pools, dataset=ds, model=tag, split=a.split, actions=a.actions, widen=W,
                with_check=a.with_check, lookup=str(lk.relative_to(ROOT)), lookup_sha256=lookup_sha,
                flags={k: v for k, v in sorted(os.environ.items()) if k.startswith("MAPREFLECT_")},
                token_log=os.environ.get("TOKEN_LOG"))
    Path(str(out) + ".manifest.json").write_text(json.dumps(meta, indent=1) + "\n")
    if ff:
        FF.manifest(out, lk, dict(script="v5_run.py", dataset=ds, model=tag, split=a.split, actions=a.actions))
    client = R2.make_client(model)
    P = D3Prompts(model, lookup, ds)
    lock = threading.Lock()

    def log(line):
        with lock:
            with open(raw_path, "a") as fh:
                fh.write(json.dumps(line) + "\n")

    def call(prompt, stage, rec, action, cands, info):
        set_usage_stage(stage)
        sink = []
        set_usage_context(dict(key=rec["key"], action=action, prompt_md5=hashlib.md5(prompt.encode()).hexdigest(),
                               treatment="v5" + ("+ff1" if ff else ""), _sink=sink))
        try:
            resp = client.generate_batch([prompt], use_cache=False)[0]
        finally:
            set_usage_context(None)
        kind = stage.split(":", 1)[0]
        st = R2.revised_status(resp, cands) if kind == "reflect" else R2.choice_status(resp)
        info[stage] = st
        line = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), model=model, run=os.environ.get("TOKEN_RUN", ""),
                    key=rec["key"], action=action, stage=stage, prompt_md5=hashlib.md5(prompt.encode()).hexdigest(),
                    candidates=list(cands or []), response=resp, parse_status=st)
        if ff:
            line.update(ff1_log.call_fields(resp, kind, st, list(cands or []), sink, lookup_sha))
        else:
            line.update(lookup_sha256=lookup_sha, **ff1_log.usage_totals(sink))
        log(line)
        return resp

    def act(rec, r, name):
        """One action under v5: the proposal, the two verifier orders, the either-order answer, the current check."""
        base = rec["base"] or ""
        res = dict(proposed=None, rmargin=None, parse=None, c1=None, c2=None, v5=base, check=None, chk=base)
        if name == "NT" and not rec["valid"]:
            return res
        info = {}
        prompt, cands = P.symmetric(r, scores=False)
        ans, rm, status = revised(call(prompt, f"reflect:{name}", rec, name, cands, info), cands)
        if status == "unparsed" or (ans and ans not in cands):
            ans = base
        res.update(proposed=ans, rmargin=rm, parse=info.get(f"reflect:{name}"))
        changed = bool(ans) and ans != base
        if changed and base:
            for nm, (o1, o2) in (("c1", (base, ans)), ("c2", (ans, base))):
                c = R2.choice(call(P.blind(r, o1, o2), f"verify:{name}:{nm}", rec, name, (o1, o2), info))
                res[nm] = {1: o1, 2: o2}.get(c)
            res["v5"] = ans if ans in (res["c1"], res["c2"]) else base
            if a.with_check:
                vp, (o1, o2) = P.verify(r, base, ans, with_evidence=True)
                res["check"] = {1: o1, 2: o2}.get(R2.choice(call(vp, f"verify:{name}:check", rec, name, (o1, o2), info))) == ans
                res["chk"] = ans if res["check"] else base
        elif changed:
            res.update(v5=ans, check=True, chk=ans)
        log(dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), model=model, run=os.environ.get("TOKEN_RUN", ""),
                 key=rec["key"], action=name, stage=f"decision:{name}", base=base, base_margin=rec["margin"],
                 valid=rec["valid"], proposed=res["proposed"], selected_margin=rm, c1=res["c1"], c2=res["c2"],
                 final=res["v5"], check=res["check"], parse=info))
        return res

    def work(rec):
        sc = sorted((R2._num(c.get("score")) for c in rec["ranking"]), reverse=True)
        s1 = sc[0] if sc and sc[0] != float("-inf") else 0.0
        band, ract = router.route(rec["margin"], rec["valid"], s1, t, ("NT", W))
        names = ["NT", "W"] if a.actions == "both" else ([ract if ract == "NT" else "W"] + (["W"] if ract == "NT" else []))
        p = pools.get(f"{rec['sample_id']}||{rec['field']}", {})
        ok = lambda x: bool(x) and target_correct(x, rec["gts"])  # noqa: E731  (v2.1 gold is applied offline)
        row = dict(key=rec["key"], sample_id=rec["sample_id"], dataset=ds, model=tag, split=a.split, gts=rec["gts"],
                   margin=rec["margin"], s1=s1, valid=rec["valid"], base=rec["base"] or "", band=band,
                   action=("NT" if ract == "NT" else "W"), widen=W)
        ran = {}
        for name in names:
            cl = pool_of("NT" if name == "NT" else W, rec, pools)
            r = dict(rec, cand_list=cl, shown=p.get("shown") or [str(c["id"]) for c in rec["ranking"]])
            cands = tuple(P.cands(r))
            row[f"{name}_pool"] = len(cands)
            row[f"{name}_gold_in_pool"] = any(ok(c) for c in cands)
            if cands in ran:
                src = ran[cands]
                for k in ("proposed", "rmargin", "parse", "c1", "c2", "v5", "check", "chk"):
                    row[f"{name}_{k}"] = row[f"{src}_{k}"]
                row[f"{name}_same_as"] = src
                continue
            for k, v in act(rec, r, name).items():
                row[f"{name}_{k}"] = v
            ran[cands] = name
        return row

    n, t0 = 0, time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex, out.open("a") as fh:
        for fut in as_completed([ex.submit(work, r) for r in recs]):
            row = fut.result()
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            n += 1
            if n % 50 == 0:
                print(f"{n}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
