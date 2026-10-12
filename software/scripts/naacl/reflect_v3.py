#!/usr/bin/env python3
"""Method v3 (docs/naacl/METHOD_V3.md): an uncertainty-routed second pass on a frozen protocol v2 first pass.

Every field is routed by two free signals of its first pass, the margin m = s1 - s2 and the top score s1, with the
shared rule in scripts/naacl/router.py (the band order is stated there):
  keep        valid and m > t_m             keep the first answer, no call
  near_tie    m <= t_m, valid, s1 >= t_s     action BC: Map-Reflect over the candidates of the first ranking
                                            (neutral instruction + schema evidence, then the pairwise check)
  widen       m <= t_m, and s1 < t_s or no   action W<K>: the same over a widened list, the candidates the first
              valid ranking
                                            pass showed plus the hybrid top-K over the target schema
                                            (data/v3_widened, built by v3_widen.py)

Development (--route off): every requested action runs on every field, so routing and the accept rule are fitted
offline. Each action X writes X (the final answer after the pairwise check), X_ok, X_check (true when the check
confirmed a change, false when it rejected one, null when nothing changed), X_rmargin (the margin of the reflection's
own revised ranking, resolved against the candidates it was shown), X_proposed and, for widened actions,
X_gold_in_pool. The margin-growth accept rule is then: X_check and X_rmargin > margin. Without a valid first answer
there is nothing to defend, so a proposed answer counts as confirmed.

Test (--route on, with the frozen --t-m, --t-s, --widen-k and --accept): each field gets its band (keep,
near_tie, widen) and only the band's action runs; the row writes v3, v3_ok, v3_changed, v3_rmargin and, for widen
fields, W_gold_in_pool.

Prompts, the evidence, the verifier and the id resolution are reflect_v2's (imported unchanged), so action BC on a
field is Map-Reflect v2's BC with its revised scores logged. Calls are tagged reflect:<action> and verify:<action>
for the usage log (TOKEN_LOG).

Usage (from the repo root):
  TOKEN_LOG=... .venv/bin/python scripts/naacl/reflect_v3.py --base results/naacl/v2/first/given/spider__gptoss \
      --manifest data/splits/spider.json --split dev --widened data/v3_widened/spider.json \
      --lookup data/spider_api_lookup.json --model endpoint_a/openai/gpt-oss-120b-a100 --actions BC,W20,W50 \
      --out results/naacl/v3/dev/given/spider__gptoss/fields.jsonl
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
import reflect_v2 as R2  # noqa: E402
import router  # noqa: E402  (sets up .env, the runner imports and the prompts)
from map_reflect.map_reflect_runner import (  # noqa: E402
    ranking_margin, resolve_ranked_id, safe_parse_json_array, safe_parse_json_object, set_usage_stage)


class Prompts(R2.Prompts):
    """reflect_v2's prompts over an explicit candidate list (rec["cand_list"]) when one is given."""

    def cands(self, rec):
        if rec.get("cand_list") is not None:
            return sorted(dict.fromkeys(rec["cand_list"]))
        return super().cands(rec)


def revised(resp, cands):
    """(answer, revised margin, status): the first listed id naming a shown candidate, and the margin of the revised
    ranking resolved against the shown candidates. status: ok, abstain, invalid or unparsed (reflect_v2.top1)."""
    t = R2.top1(resp, cands)
    if t is R2.ABSTAIN:
        return "", 0.0, "abstain"
    if t is R2.INVALID:
        return "", 0.0, "invalid"
    if t is None:
        return None, 0.0, "unparsed"
    obj = safe_parse_json_object(resp or "", want="revised_ranking")
    arr = obj.get("revised_ranking") if isinstance(obj, dict) else None
    if not arr:
        arr = safe_parse_json_array(resp or "") or []
    items = []
    for it in arr:
        if isinstance(it, dict) and "id" in it:
            s = str(it["id"])
            rid = cands[int(s) - 1] if s.isdigit() and 1 <= int(s) <= len(cands) else resolve_ranked_id(s, cands)
            if rid:
                items.append({"id": rid, "score": it.get("score")})
    if R2.v41("MAPREFLECT_REVISED_SELECTED"):
        return t, selected_margin(t, items), "ok"
    _, m, ok = ranking_margin(items)
    return t, (m if ok else 0.0), "ok"


def selected_margin(sel, items):
    """Method v4.1 (MAPREFLECT_REVISED_SELECTED=1): the revised margin of the answer actually selected.

    The answer is the first listed shown candidate (reflect_v2.top1), so its confidence is its own score minus the
    best score of any other distinct candidate, with each candidate counted once at its first listed score (a
    repeated id neither erases nor creates a margin). When another candidate scores higher than the selected one
    the scores contradict the order; the margin is 0 (no dominance), never another candidate's margin. A selected
    answer without a numeric score has margin 0. One candidate: the other score is 0, as in ranking_margin.
    """
    first = {}
    for it in items:
        if it["id"] not in first:
            first[it["id"]] = R2._num(it.get("score"))
    s1 = first.get(sel, float("-inf"))
    if s1 == float("-inf"):
        return 0.0
    others = [v for k, v in first.items() if k != sel and v != float("-inf")]
    return max(0.0, s1 - (max(others) if others else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", choices=["dev", "test", "test_grouped"], required=True)
    ap.add_argument("--widened", required=True, help="data/v3_widened/<dataset>.json")
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--actions", default="BC,W20,W50", help="BC and/or W<K> (development)")
    ap.add_argument("--widen-k", type=int, help="test: K of the widened action (frozen)")
    ap.add_argument("--accept", choices=["check", "check+margin"], help="test: the frozen accept rule")
    ap.add_argument("--route", choices=["off", "on"], default="off",
                    help="on: run only the routed action (needs --t-m and --t-s); off: every action on every field")
    ap.add_argument("--t-m", type=float)
    ap.add_argument("--t-s", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.route == "on" and (a.t_m is None or a.t_s is None or a.widen_k is None or a.accept is None):
        sys.exit("--route on needs --t-m, --t-s, --widen-k and --accept (the frozen values in METHOD_V3.md)")
    actions = [x for x in a.actions.split(",") if x]
    for x in actions:
        if x != "BC" and not (x.startswith("W") and x[1:].isdigit()):
            sys.exit(f"unknown action {x}")

    d = R2.load_slot(a.base)
    keep = set(json.load(open(a.manifest))[a.split])
    widened = json.load(open(a.widened))
    lookup = json.load(open(a.lookup))
    recs = list(R2.records(d, keep))
    if a.limit:
        recs = recs[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["key"] for l in out.open()} if out.exists() else set()
    recs = [r for r in recs if r["key"] not in done]
    print(f"{len(done)} done, {len(recs)} to run", flush=True)
    client = R2.make_client(a.model)
    P = Prompts(a.model, lookup)
    lock = threading.Lock()
    ok = lambda p, gts: int(bool(p) and any(R2.fields_match(g, p) for g in gts))

    def call(prompt, stage):
        set_usage_stage(stage)
        return client.generate_batch([prompt], use_cache=False)[0]

    def band(rec, s1):
        if a.t_m is None or a.t_s is None:
            return None
        return router.band(rec["margin"], rec["valid"], s1, a.t_m, a.t_s)

    def act(rec, name):
        """Run one action; return (final answer after the check, check flag, revised margin, proposed, gold in pool)."""
        base = rec["base"] or ""
        if name == "BC":
            if not rec["valid"]:
                return base, None, None, None, None
            r, gold_in = dict(rec, cand_list=None), None
        else:
            k = int(name[1:])
            w = widened.get(f"{rec['sample_id']}||{rec['field']}", {})
            shown = w.get("shown") or [c["id"] for c in rec["ranking"]]
            r = dict(rec, cand_list=list(dict.fromkeys(shown + w.get("widen", [])[:k])))
            gold_in = any(ok(c, rec["gts"]) for c in r["cand_list"])
        prompt, cands = P.neutral(r, with_evidence=True)
        ans, rm, status = revised(call(prompt, f"reflect:{name}"), cands)
        if status == "unparsed":
            ans = base
        changed = bool(ans) and ans != base
        check = None
        if changed and base:
            vp, (o1, o2) = P.verify(r, base, ans, with_evidence=True)
            check = {1: o1, 2: o2}.get(R2.choice(call(vp, f"verify:{name}"))) == ans
        elif changed:
            check = True  # no first answer to defend
        final = ans if check else base
        return final, check, rm, ans, gold_in

    def work(rec):
        sc = sorted((R2._num(c.get("score")) for c in rec["ranking"]), reverse=True)
        s1 = sc[0] if sc and sc[0] != float("-inf") else 0.0
        row = dict(key=rec["key"], sample_id=rec["sample_id"], gts=rec["gts"], margin=rec["margin"], s1=s1,
                   valid=rec["valid"], base=rec["base"] or "", base_ok=ok(rec["base"], rec["gts"]))
        b = band(rec, s1)
        if a.route == "off":
            if b:
                row["band"] = b
            for name in actions:
                final, check, rm, prop, gold_in = act(rec, name)
                row[name], row[f"{name}_ok"], row[f"{name}_check"] = final, ok(final, rec["gts"]), check
                row[f"{name}_rmargin"], row[f"{name}_proposed"] = rm, prop
                if gold_in is not None:
                    row[f"{name}_gold_in_pool"] = gold_in
            return row
        row["band"] = b
        if b == router.KEEP:
            row.update(v3=row["base"], v3_ok=row["base_ok"], v3_changed=False, v3_rmargin=None)
            return row
        name = "BC" if b == "near_tie" else f"W{a.widen_k}"
        final, check, rm, prop, gold_in = act(rec, name)
        if a.accept == "check+margin" and check and not (rm is not None and rm > rec["margin"]):
            final = rec["base"] or ""
        row.update(v3=final, v3_ok=ok(final, rec["gts"]), v3_changed=final != (rec["base"] or ""), v3_rmargin=rm,
                   v3_action=name, v3_proposed=prop)
        if gold_in is not None:
            row["W_gold_in_pool"] = gold_in
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
