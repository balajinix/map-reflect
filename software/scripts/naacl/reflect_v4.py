#!/usr/bin/env python3
"""Method v4 (docs/naacl/METHOD_V4.md): the measured second pass on a frozen protocol v2 first pass.

Every requested action runs on every field (development and test alike), so every gate, including random, oracle and
reflect-all, is scored on measured outcomes. Routing is the shared rule in scripts/naacl/router.py; with frozen
--t-m and --t-s each row also records its band and the v4 answer under the frozen near-tie and widen actions.

Actions (the prompts, evidence, pairwise check and id resolution are reflect_v2's, as in method v3):
  NT       near-tie: Map-Reflect over the candidates of the first ranking (method v3's BC)
  WT<K>    widen inside the named target type: the shown candidates plus the type's next K fields in hybrid order
           (data/v4_pools: type_rest)
  WX<K>    widen beyond the type: the shown candidates plus the first K of the type's other fields followed by the
           target schema's fields outside the type (type_rest, then wider), in hybrid order
  W..n     the same pool with the neutral reflection prompt only: the previous answer, no schema evidence, no
           pairwise check (the method v5 action ablation)
An action whose candidate list equals another widen action's on the same field is called once and its outcome
shared (X_same_as names the action that ran). With --reuse-nt, NT is taken from a stored BC row on the same first
answer (method v3's BC rows, or protocol v2's BC arm: the same prompts, evidence and check) instead of called again;
NT_reused names the file, and a row whose base, margin or validity differs from the current first pass is not used. The pairwise check may accept only a target inside the action's own
candidate pool; anything else leaves the first answer.

Each action X writes X (the answer after the check), X_ok, X_check (true when the check confirmed a change, false
when it rejected one, null when nothing changed), X_rmargin (the revised ranking's margin), X_proposed, X_pool (the
pool size) and, for widen actions, X_gold_in_pool. Scoring is eval.calc_accuracy.target_correct (full identity).

Usage (from the repo root):
  TOKEN_LOG=... .venv/bin/python scripts/naacl/reflect_v4.py --base results/naacl/v2/first/given/omop_cms__gptoss \
      --manifest data/splits/omop_cms.json --split dev --pools data/v4_pools/omop_cms.json \
      --lookup data/omop_cms_api_lookup.json --model endpoint_a/openai/gpt-oss-120b-a100 --actions NT,WT20,WT50,WX20,WX50 \
      --out results/naacl/v4/dev/given/omop_cms__gptoss/fields.jsonl
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
import reflect_v2 as R2  # noqa: E402  (sets up .env, the runner imports and the prompts)
import router  # noqa: E402
from eval.calc_accuracy import target_correct  # noqa: E402
from map_reflect.map_reflect_runner import set_usage_context, set_usage_stage  # noqa: E402
from reflect_v3 import Prompts, revised  # noqa: E402
import field_first as FF  # noqa: E402  (treatment ff1, MAPREFLECT_FIELD_FIRST=1; off by default)
import ff1_log  # noqa: E402  (the ff1 raw-log fields)

ACTION = re.compile(r"^(NT|W[TX]\d+n?)$")


def pool_of(name, rec, pools):
    """The candidate list of a widen action on one field (None for NT, whose pool is the first ranking's)."""
    if name == "NT":
        return None
    p = pools.get(f"{rec['sample_id']}||{rec['field']}", {})
    shown = p.get("shown") or [str(c["id"]) for c in rec["ranking"]]
    k = int(name[2:].rstrip("n"))
    extra = p.get("type_rest", [])[:k] if name[1] == "T" else (p.get("type_rest", []) + p.get("wider", []))[:k]
    return list(dict.fromkeys(shown + extra))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", choices=["dev", "test", "test_grouped", "all"], required=True)
    ap.add_argument("--pools", required=True, help="data/v4_pools/<dataset>.json")
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--actions", required=True, help="comma list of NT, WT<K>, WX<K>")
    ap.add_argument("--t-m", type=float, help="frozen: record the band and the v4 answer")
    ap.add_argument("--t-s", type=float)
    ap.add_argument("--near", default="NT", help="frozen near-tie action")
    ap.add_argument("--widen", help="frozen widen action")
    ap.add_argument("--accept", choices=["check", "check+margin"], default="check")
    ap.add_argument("--reuse-nt", action="append", default=[],
                    help="fields.jsonl with BC outcomes on the same first pass (v3 dev/test rows or v2 reflect rows)")
    ap.add_argument("--keys-file", help="run only these record keys (one per line)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--raw-log", default=os.environ.get("MAPREFLECT_RAW_LOG"),
                    help="method v4.1: append every reflection and verifier call (key, action, stage, prompt hash, "
                         "candidate ids, raw response, parse status) to this jsonl, and write <X>_parse and "
                         "<X>_verify_parse on each row. Default: MAPREFLECT_RAW_LOG, else <out>.raw.jsonl when "
                         "MAPREFLECT_V41=1, else off (v4)")
    a = ap.parse_args()
    actions = [x for x in a.actions.split(",") if x]
    for x in actions:
        if not ACTION.match(x):
            sys.exit(f"unknown action {x}")
    if a.reuse_nt and os.environ.get("MAPREFLECT_V41") == "1":
        sys.exit("--reuse-nt takes NT outcomes produced by the v4 prompts; under MAPREFLECT_V41=1 NT must be called")
    if a.reuse_nt and FF.on():
        sys.exit("--reuse-nt takes NT outcomes produced by the default descriptions; under MAPREFLECT_FIELD_FIRST=1 NT "
                 "must be called")
    frozen = a.t_m is not None
    if frozen and (a.t_s is None or not a.widen or a.near not in actions or a.widen not in actions):
        sys.exit("with --t-m give --t-s and --widen, and run both frozen actions (--actions)")

    d = R2.load_slot(a.base)
    if d is None:
        sys.exit(f"no runner output in {a.base}")
    man = json.load(open(a.manifest))
    keep = set(man["dev"]) | set(man.get("test_grouped", man.get("test", []))) if a.split == "all" else set(man[a.split])
    if man.get("all_test"):
        keep = {str(r.get("sample_id")) for r in d["results"]}
    pools = json.load(open(a.pools))
    lookup = json.load(open(a.lookup))
    if FF.on():
        FF.require(lookup, a.lookup)
    recs = list(R2.records(d, keep))
    if a.keys_file:
        only = {l.strip() for l in open(a.keys_file) if l.strip()}
        recs = [r for r in recs if r["key"] in only]
    if a.limit:
        recs = recs[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = a.raw_log or (str(out) + ".raw.jsonl" if os.environ.get("MAPREFLECT_V41") == "1" or FF.on() else None)
    ff = FF.on()
    lookup_sha = hashlib.sha256(Path(a.lookup).read_bytes()).hexdigest() if ff else None
    done = {json.loads(l)["key"] for l in out.open()} if out.exists() else set()
    recs = [r for r in recs if r["key"] not in done]
    print(f"{len(done)} done, {len(recs)} to run", flush=True)
    if FF.on():
        print(f"treatment ff1: manifest {FF.manifest(out, a.lookup, dict(script='reflect_v4.py', base=a.base, split=a.split, actions=actions, pools=a.pools))}", flush=True)
    reuse = {}
    for f in a.reuse_nt:
        for line in open(f):
            x = json.loads(line)
            if "BC" in x and x["key"] not in reuse:
                reuse[x["key"]] = (x, f)
    client = R2.make_client(a.model)
    P = Prompts(a.model, lookup)
    lock = threading.Lock()

    def call(prompt, stage, rec=None, name=None, cands=None, info=None):
        """One model call. With raw logging (method v4.1) the raw response and its parse status are appended to
        raw_path and the status is stored in info[stage kind]."""
        set_usage_stage(stage)
        sink = []
        if ff and rec is not None:  # the usage record of this call names its field, action and prompt
            set_usage_context(dict(key=rec["key"], action=name, prompt_md5=hashlib.md5(prompt.encode()).hexdigest(),
                                   treatment=FF.TREATMENT, _sink=sink))
        try:
            resp = client.generate_batch([prompt], use_cache=False)[0]
        finally:
            set_usage_context(None)
        if raw_path and rec is not None:
            kind = stage.split(":", 1)[0]
            st = R2.revised_status(resp, cands) if kind == "reflect" else R2.choice_status(resp)
            if info is not None:
                info[kind] = st
            line = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), model=a.model, run=os.environ.get("TOKEN_RUN", ""),
                        key=rec["key"], action=name, stage=stage,
                        prompt_md5=hashlib.md5(prompt.encode()).hexdigest(), candidates=list(cands or []),
                        response=resp, parse_status=st)
            if ff:
                line.update(ff1_log.call_fields(resp, kind, st, list(cands or []), sink, lookup_sha))
                if info is not None:
                    info[f"{kind}_failure"] = line["failure"]
                    info.setdefault("usage", []).extend(sink)
            with lock:
                with open(raw_path, "a") as fh:
                    fh.write(json.dumps(line) + "\n")
        return resp

    def act(rec, name, cand_list, info=None):
        """(final answer after the check, check flag, revised margin, proposed, pool size). Under treatment ff1 the
        action's decision line is appended to the raw log (ff1_log)."""
        res = _act(rec, name, cand_list, info)
        if ff and raw_path:
            final, check, rm, prop, _ = res
            line = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), model=a.model, run=os.environ.get("TOKEN_RUN", ""),
                        key=rec["key"], action=name, stage=f"decision:{name}",
                        **ff1_log.decision_fields(rec, info or {}, lookup_sha, prop, rm, check, final))
            with lock:
                with open(raw_path, "a") as fh:
                    fh.write(json.dumps(line) + "\n")
        return res

    def _act(rec, name, cand_list, info=None):
        base = rec["base"] or ""
        if name == "NT" and not rec["valid"]:
            return base, None, None, None, 0  # nothing to reflect on: no call
        r = dict(rec, cand_list=cand_list)
        plain = name.endswith("n")  # ablation: neutral reflection with the previous answer, no evidence, no check
        prompt, cands = P.neutral(r, with_evidence=not plain)
        ans, rm, status = revised(call(prompt, f"reflect:{name}", rec, name, cands, info), cands)
        if status == "unparsed" or (ans and ans not in cands):
            ans = base  # unreadable, or outside the action's own pool: the first answer stands
        changed = bool(ans) and ans != base
        check = None
        if plain:
            return ans or base, None, rm, ans, len(cands)
        if changed and base:
            vp, (o1, o2) = P.verify(r, base, ans, with_evidence=True)
            check = {1: o1, 2: o2}.get(R2.choice(call(vp, f"verify:{name}", rec, name, (o1, o2), info))) == ans
        elif changed:
            check = True  # no first answer to defend
        final = ans if check else base
        if a.accept == "check+margin" and check and not (rm is not None and rm > rec["margin"]):
            final = base
        return final, check, rm, ans, len(cands)

    def work(rec):
        sc = sorted((R2._num(c.get("score")) for c in rec["ranking"]), reverse=True)
        s1 = sc[0] if sc and sc[0] != float("-inf") else 0.0
        ok = lambda p: target_correct(p, rec["gts"])
        row = dict(key=rec["key"], sample_id=rec["sample_id"], gts=rec["gts"], margin=rec["margin"], s1=s1,
                   valid=rec["valid"], base=rec["base"] or "", base_ok=ok(rec["base"]))
        ran = {}  # tuple(candidate list) -> action that ran it
        route = router.route(rec["margin"], rec["valid"], s1, dict(t_m=a.t_m, t_s=a.t_s), (a.near, a.widen)) \
            if frozen else (None, None)
        got = reuse.get(rec["key"])
        if "NT" in actions and got and got[0].get("base", "") == row["base"] and bool(got[0].get("valid", True)) == \
                rec["valid"] and abs(float(got[0].get("margin", 0)) - rec["margin"]) < 1e-9:
            x, f = got
            if "BC_check" in x:  # method v3 row
                chk, rm, prop = x.get("BC_check"), x.get("BC_rmargin"), x.get("BC_proposed")
            else:  # protocol v2 row: B is the proposal, BC what the check left
                prop = x.get("B")
                chk = None if (not prop or prop == row["base"]) else (x.get("BC") == prop)
                rm = None
            final = x.get("BC") or ""
            if a.accept == "check+margin" and chk and not (rm is not None and rm > rec["margin"]):
                final = row["base"]
            # a confirmed change on a stored v2 row has no revised margin to test, so check+margin calls NT again
            if a.accept == "check" or rm is not None or not chk:
                row.update(NT=final, NT_ok=ok(final), NT_check=chk, NT_rmargin=rm, NT_proposed=prop,
                           NT_pool=None, NT_reused=f)
        for name in actions:
            if name == "NT" and "NT_reused" in row:
                continue
            cl = pool_of(name, rec, pools)
            key = (name.endswith("n"), tuple(cl)) if cl is not None else None
            if key is not None and key in ran:
                src = ran[key]
                for suf in ("", "_ok", "_check", "_rmargin", "_proposed", "_pool", "_parse", "_verify_parse"):
                    if src + suf in row:
                        row[name + suf] = row[src + suf]
                row[f"{name}_same_as"] = src
            else:
                info = {"route": route}
                final, check, rm, prop, n = act(rec, name, cl, info)
                row[name], row[f"{name}_ok"], row[f"{name}_check"] = final, ok(final), check
                if raw_path:
                    row[f"{name}_parse"], row[f"{name}_verify_parse"] = info.get("reflect"), info.get("verify")
                row[f"{name}_rmargin"], row[f"{name}_proposed"], row[f"{name}_pool"] = rm, prop, n
                if key is not None:
                    ran[key] = name
            if cl is not None:
                row[f"{name}_gold_in_pool"] = any(ok(c) for c in cl)
        if frozen:
            b, x = router.route(rec["margin"], rec["valid"], s1, dict(t_m=a.t_m, t_s=a.t_s), (a.near, a.widen))
            row["band"] = b
            row["v4_action"] = None if b == router.KEEP else x
            row["v4"] = row["base"] if b == router.KEEP else row[x]
            row["v4_ok"] = ok(row["v4"])
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
