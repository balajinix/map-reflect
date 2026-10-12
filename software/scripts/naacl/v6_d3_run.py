#!/usr/bin/env python3
"""D3 of the v6 plan (docs/naacl/v6/PLAN_PHASE2.md, Phase 3; docs/naacl/codex_ideas.txt ideas 5 and 6): development
runs of a symmetric-evidence reflection prompt and an order-balanced pairwise verifier. Development only, exploratory.

Each selected development field (scripts/naacl/v6_d3.py --stage select) runs its routed action's pool (router.action
under the frozen method v4.1 thresholds: NT over the first ranking's candidates, else the frozen widen action WT<K>)
with up to four arms, all on the method v4.1 code (MAPREFLECT_V41=1: repaired lookups, the target name, robust JSON,
the selected answer's revised margin, raw-response logging):

  cur    the current reflection (reflect_v4: the neutral prompt with schema evidence for the source and the previous
         ranking's top 5, the one-to-one conflicts line, the previous ranking with scores) and the current pairwise check
  symS   idea 5 with old scores: the same instruction and candidate list, but symmetric evidence (entity, meaning, data
         type, neighbouring fields) for the leading old candidates (the previous ranking's top 3) and the leading new
         ones (the first 3 candidates the action adds, in hybrid order; for NT the previous ranking's 4th to 6th), listed
         alphabetically with no mark of where each came from; the dataset's correspondence rule instead of the
         one-to-one conflicts line; then the current pairwise check
  symN   symS without the old numerical scores (the previous ranking keeps its order and reasons)
  blind  idea 6, on fields where cur proposed a change to a valid first answer (the check fired): a pairwise verifier
         with symmetric evidence cards for the source and both options and the dataset's rule, never saying which
         option is the first answer, called twice with the options in both orders (blind_c1: the first answer is
         option 1; blind_c2: the proposal is option 1). The decision rules (one random order, both orders agree, either
         order) are applied offline by v6_d3.py.

Treatment ff1 (MAPREFLECT_FIELD_FIRST=1, scripts/naacl/field_first.py): the lookup is data/<ds>_reflect_lookup_ff1.json,
every card's meaning is the field's own definition and ETL notes, the table text is given once per table, and the raw
log gains the ff1_log fields. Off, the prompts and logs are as before.

Each arm X writes X (the answer after its check), X_proposed, X_check (true when the check confirmed a change, false when
it rejected one, null when nothing changed), X_rmargin, X_parse and X_verify_parse. Scoring is offline (v6_d3.py).

Usage (from the repo root, COMPUTE_HOST):
  MAPREFLECT_V41=1 MAPREFLECT_CANONICAL_MARGIN=1 TOKEN_LOG=... .venv/bin/python scripts/naacl/v6_d3_run.py \
      --dataset omop_cms --model-tag qwen35 --keys results/naacl/v6/d3/keys/omop_cms__qwen35.txt \
      --arms cur,symS,symN,blind --out results/naacl/v6/d3/dev/omop_cms__qwen35.jsonl
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
from map_reflect.map_reflect_runner import set_usage_context, set_usage_stage  # noqa: E402
from reflect_v3 import Prompts, revised  # noqa: E402
from reflect_v4 import pool_of  # noqa: E402
import field_first as FF  # noqa: E402  (treatment ff1, MAPREFLECT_FIELD_FIRST=1; off by default)
import ff1_log  # noqa: E402  (the ff1 raw-log fields)

MODELS = {"gptoss": "endpoint_a/openai/gpt-oss-120b-a100", "qwen35": "endpoint_a/Qwen/Qwen3.5-397B-A17B-FP8-JP",
          "llama4": "endpoint_b/meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
          "granite": "endpoint_b/vendor/granite-4-h-small"}
ENDPOINT_A_EP = {"gptoss": "gpt-oss-120b-a100", "qwen35": "qwen3-5-397b-a17b-fp8-jp"}
V41_LOOKUP = {"nl2gql", "valentine_chembl", "valentine_opendata", "valentine_tpcdi", "valentine_wikidata"}
N_OLD, N_NEW = 3, 3

# The dataset's correspondence rule. Many-to-one tasks (OMOP, ToolBench, ITOpsGraph) say so explicitly and carry no
# uniqueness hint; the one-to-one tasks (Valentine, NL2GQL) state their cardinality as a rule of the task.
RULES = {
    "itopsgraph": "map the source field to the target field that holds the same data about the same entity. Several "
                  "source fields may map to the same target field, and more than one target field can be acceptable.",
    "toolbench": "map the source API parameter to the target API parameter that receives the same value. Several "
                 "source parameters may map to the same target parameter.",
    "nl2gql": "map the source GraphQL field to the target field that holds the same data; each target field "
              "corresponds to at most one source field of the pair.",
    "omop": "map the OMOP column to the source-system field it is derived from in the ETL (the value may be "
            "reformatted, for example a date taken from a timestamp or a code from a coded column). Several OMOP "
            "columns may be derived from the same source-system field.",
    "valentine": "the two tables were derived from one original table; map the source column to the target column "
                 "with the same original column. Each target column corresponds to at most one source column.",
    # held-out sets (method v5 transfers the core-development configuration to them)
    "mimic_omop_rematch": "map the MIMIC column to the OMOP CDM column that receives its value in the ETL (the value "
                          "may be reformatted or recoded). Several source columns may feed the same OMOP column.",
    "schemanet_cprd": "map the CPRD column to the OMOP CDM column that receives its value in the ETL (the value may be "
                      "reformatted or recoded). Several source columns may feed the same OMOP column, and more than one "
                      "OMOP column can be acceptable.",
    "semtab_m2kg_r1": "map the web-table column to the knowledge-graph property that states the same relation for the "
                      "row's subject.",
    "oaei_conference": "map the source ontology's class or property to the equivalent class or property of the target "
                       "ontology.",
    "bioml_omim_ordo": "map the source disease concept to the equivalent concept of the target ontology.",
}


def rule_of(ds):
    return RULES["omop" if ds.startswith("omop_") else "valentine" if ds.startswith("valentine_") else ds]


class D3Prompts(Prompts):
    """reflect_v3's prompts plus the symmetric-evidence reflection and the order-balanced verifier."""

    def __init__(self, model, lookup, ds):
        super().__init__(model, lookup)
        self.ds, self.rule = ds, rule_of(ds)
        self.groups = {}
        for k in lookup:
            g = self.group(k)
            if g is not None:
                self.groups.setdefault(g, []).append(k)

    def group(self, k):
        m = self.meta(k)
        if m.get("type_name"):
            return ("t", m.get("product_name"), m.get("type_name"))
        if m.get("api"):
            return ("a", m.get("tool"), m.get("api"))
        return None

    def entity(self, k):
        m = self.meta(k)
        name = m.get("type_name") or m.get("api") or "?"
        owner = m.get("product_name") or m.get("tool") or m.get("database")
        desc = m.get("type_description") or m.get("table_description") or ""
        if FF.on() and FF.parts(m):
            desc = ""  # treatment ff1: the table text is given once per table in the prompt's table block
        return f"{name}" + (f" (in {owner})" if owner and owner != name else "") + (f": {str(desc)[:160]}" if desc else "")

    def neighbours(self, k):
        g = self.group(k)
        names = [self.meta(x).get("field_name") or self.meta(x).get("name") or x
                 for x in self.groups.get(g, []) if x != k][:R2.N_SIB] if g else []
        return ", ".join(str(n) for n in names) or "none listed"

    def card(self, k, indent="   "):
        return (f"{k}\n{indent}entity: {self.entity(k)}\n{indent}meaning: {self.desc(k)}\n"
                f"{indent}data type: {self.dtype(k)}\n{indent}neighbouring fields: {self.neighbours(k)}")

    def leaders(self, rec, cands):
        """The leading old candidates (previous ranking's top N_OLD) and leading new ones (the first N_NEW the action
        adds, in pool order; for NT the previous ranking's next N_NEW), alphabetically and without origin."""
        prev = []
        for c in rec["ranking"]:
            cid = str(c["id"])
            if cid in cands and cid not in prev:
                prev.append(cid)
        old = prev[:N_OLD]
        if rec.get("cand_list") is not None:
            shown = set(rec.get("shown") or [])
            new = [c for c in rec["cand_list"] if c not in shown and c not in old][:N_NEW]
        else:
            new = prev[N_OLD:N_OLD + N_NEW]
        return sorted(dict.fromkeys(old + new))

    def symmetric(self, rec, scores=True):
        cands = self.cands(rec)
        cand_block = "\n".join(f"{i+1}. id: {c}\n   type: {self.dtype(c)}\n   text: {c} -- {self.desc(c)}"
                               for i, c in enumerate(cands))
        if scores:
            prior = "\n".join(f"{i+1}. {c.get('id')} (score={c.get('score')}) - {c.get('reason', '')}"
                              for i, c in enumerate(rec["ranking"][:R2.N_CAND]))
        else:
            prior = "\n".join(f"{i+1}. {c.get('id')} - {c.get('reason', '')}"
                              for i, c in enumerate(rec["ranking"][:R2.N_CAND]))
        lead = "\n".join(f"- {self.card(c)}" for c in self.leaders(rec, cands))
        user = f"""You previously ranked candidate {rec['tgt']} properties for a {rec['src']} property.

{rec['src']} property: {self.card(rec['field'])}{self.tables([rec['field']] + cands)}

Candidates:
{cand_block}

Your previous ranking (best first):
{prior}

The rule of this task: {self.rule}

Evidence for the leading candidates (the same facts for each, listed alphabetically):
{lead}

Review your previous ranking against the descriptions and the evidence. Compare the leading candidates on the same
terms: the entity each belongs to, its meaning, its data type (with units and allowed values where the description
gives them) and its neighbouring fields. Keep your previous top answer unless you find a concrete reason it is wrong,
such as a different meaning, data type or entity than the source property. Do not change it only because another
candidate has a similar name.

""" + R2.JSON_SPEC
        return self.fmt("You are an expert schema mapper reviewing your own work.", user), cands

    def blind(self, rec, o1, o2):
        user = f"""Which target property is the correct mapping for this {rec['src']} property?

The rule of this task: {self.rule}

{rec['src']} property: {self.card(rec['field'])}{self.tables([rec['field'], o1, o2])}

Option 1: {self.card(o1)}

Option 2: {self.card(o2)}

Judge both options on the same terms: the entity each belongs to, its meaning, its data type (with units and allowed
values where the description gives them) and its neighbouring fields, not name similarity. Return a JSON object:
{{"choice": 1 or 2, "reason": "one sentence"}}"""
        return self.fmt("You are an expert schema mapper.", user)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-tag", required=True, choices=sorted(MODELS))
    ap.add_argument("--keys", required=True, help="record keys to run, one per line")
    ap.add_argument("--arms", default="cur,symS,symN,blind")
    ap.add_argument("--frozen", default="docs/naacl/results/v4_1_frozen.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if os.environ.get("MAPREFLECT_V41") != "1" or not router.CANONICAL:
        sys.exit("D3 runs on method v4.1: set MAPREFLECT_V41=1 and MAPREFLECT_CANONICAL_MARGIN=1")
    arms = [x for x in a.arms.split(",") if x]
    ds, tag = a.dataset, a.model_tag
    model = MODELS[tag]
    if tag in ENDPOINT_A_EP:
        os.environ["ENDPOINT_A_ENDPOINT"] = ENDPOINT_A_EP[tag]
    t = json.load(open(ROOT / a.frozen))["thresholds"][tag]
    man = json.load(open(ROOT / f"data/splits/{ds}.json"))
    d = R2.load_slot(str(ROOT / f"results/naacl/v2/first/given/{ds}__{tag}"))
    lk = ROOT / (f"data/{ds}_reflect_lookup_v41.json" if ds in V41_LOOKUP else f"data/{ds}_api_lookup.json")
    if FF.on():  # treatment ff1: the same lookup with each entry's description parts (scripts/naacl/ff1_lookups.py)
        lk = ROOT / f"data/{ds}_reflect_lookup_ff1.json"
    lookup = json.load(open(lk))
    ff = FF.on()
    if ff:
        FF.require(lookup, lk)
    lookup_sha = hashlib.sha256(lk.read_bytes()).hexdigest() if ff else None
    pools = json.load(open(ROOT / f"data/v4_pools/{ds}.json"))
    want = {l.strip() for l in open(a.keys) if l.strip()}
    recs = [r for r in R2.records(d, set(man["dev"])) if r["key"] in want]
    if len(recs) != len(want):
        sys.exit(f"{len(want) - len(recs)} keys are not development fields of {ds}__{tag}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = str(out) + ".raw.jsonl"
    # resume guard: rows already in <out> must come from this exact configuration (scripts/naacl/run_guard.py)
    import run_guard as RG
    RG.check(out, RG.fingerprint(
        dict(lookup=lk, pools=f"data/v4_pools/{ds}.json", frozen=a.frozen, keys=a.keys,
             base=f"results/naacl/v2/first/given/{ds}__{tag}"),
        dict(script="v6_d3_run.py", dataset=ds, model=model, arms=arms)))
    done = {json.loads(l)["key"] for l in out.open()} if out.exists() else set()
    recs = [r for r in recs if r["key"] not in done]
    print(f"{ds}__{tag}: {len(done)} done, {len(recs)} to run; lookup {lk.name}; arms {arms}", flush=True)
    if FF.on():
        print(f"treatment ff1: manifest {FF.manifest(out, lk, dict(script='v6_d3_run.py', dataset=ds, model=tag, keys=a.keys, arms=arms))}", flush=True)
    client = R2.make_client(model)
    P = D3Prompts(model, lookup, ds)
    lock = threading.Lock()

    def call(prompt, stage, rec, arm, cands, info):
        set_usage_stage(stage)
        sink = []
        if ff:  # the usage record of this call names its field, arm and prompt
            set_usage_context(dict(key=rec["key"], arm=arm, prompt_md5=hashlib.md5(prompt.encode()).hexdigest(),
                                   treatment=FF.TREATMENT, _sink=sink))
        try:
            resp = client.generate_batch([prompt], use_cache=False)[0]
        finally:
            set_usage_context(None)
        kind = stage.split(":", 1)[0]
        st = R2.revised_status(resp, cands) if kind == "reflect" else R2.choice_status(resp)
        info[kind] = st
        line = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), model=model, run=os.environ.get("TOKEN_RUN", ""),
                    key=rec["key"], arm=arm, stage=stage, prompt_md5=hashlib.md5(prompt.encode()).hexdigest(),
                    candidates=list(cands or []), response=resp, parse_status=st)
        if ff:
            line.update(ff1_log.call_fields(resp, kind, st, list(cands or []), sink, lookup_sha))
            info[f"{kind}_failure"] = line["failure"]
            info.setdefault("usage", []).extend(sink)
        with lock:
            with open(raw_path, "a") as fh:
                fh.write(json.dumps(line) + "\n")
        return resp

    def reflect(rec, r, arm, act, route=(None, None)):
        """reflect_v4.act for one arm: (final, check, revised margin, proposed, parse, verify parse). Under treatment
        ff1 the arm's decision line is appended to the raw log (ff1_log)."""
        info = {"route": route}
        res = _reflect(rec, r, arm, act, info)
        if ff:
            fin, chk, rm, prop, _, _ = res
            line = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), model=model, run=os.environ.get("TOKEN_RUN", ""),
                        key=rec["key"], arm=arm, stage=f"decision:{arm}",
                        **ff1_log.decision_fields(rec, info, lookup_sha, prop, rm, chk, fin))
            with lock:
                with open(raw_path, "a") as fh:
                    fh.write(json.dumps(line) + "\n")
        return res

    def _reflect(rec, r, arm, act, info):
        base = rec["base"] or ""
        if act == "NT" and not rec["valid"]:
            return base, None, None, None, None, None
        if arm == "cur":
            prompt, cands = P.neutral(r, with_evidence=True)
        else:
            prompt, cands = P.symmetric(r, scores=(arm == "symS"))
        ans, rm, status = revised(call(prompt, f"reflect:{arm}", rec, arm, cands, info), cands)
        if status == "unparsed" or (ans and ans not in cands):
            ans = base
        changed = bool(ans) and ans != base
        check = None
        if changed and base:
            vp, (o1, o2) = P.verify(r, base, ans, with_evidence=True)
            check = {1: o1, 2: o2}.get(R2.choice(call(vp, f"verify:{arm}", rec, arm, (o1, o2), info))) == ans
        elif changed:
            check = True
        return (ans if check else base), check, rm, ans, info.get("reflect"), info.get("verify")

    def work(rec):
        sc = sorted((R2._num(c.get("score")) for c in rec["ranking"]), reverse=True)
        s1 = sc[0] if sc and sc[0] != float("-inf") else 0.0
        band, act = router.route(rec["margin"], rec["valid"], s1, t, ("NT", t["widen"]))
        cl = pool_of(act, rec, pools)
        p = pools.get(f"{rec['sample_id']}||{rec['field']}", {})
        r = dict(rec, cand_list=cl, shown=p.get("shown") or [str(c["id"]) for c in rec["ranking"]])
        row = dict(key=rec["key"], sample_id=rec["sample_id"], dataset=ds, model=tag, gts=rec["gts"],
                   margin=rec["margin"], s1=s1, valid=rec["valid"], base=rec["base"] or "", band=band, action=act,
                   pool=len(P.cands(r)), pool_ids=P.cands(r))
        for arm in arms:
            if arm == "blind":
                continue
            fin, chk, rm, prop, ps, vs = reflect(rec, r, arm, act, (band, act))
            row.update({arm: fin, f"{arm}_check": chk, f"{arm}_rmargin": rm, f"{arm}_proposed": prop,
                        f"{arm}_parse": ps, f"{arm}_verify_parse": vs})
        if "blind" in arms:
            prop = row.get("cur_proposed")
            if prop and row["base"] and prop != row["base"]:
                for nm, (o1, o2) in (("blind_c1", (row["base"], prop)), ("blind_c2", (prop, row["base"]))):
                    info = {}
                    c = R2.choice(call(P.blind(r, o1, o2), f"verify:{nm}", rec, nm, (o1, o2), info))
                    row[nm] = {1: o1, 2: o2}.get(c)
                    row[f"{nm}_parse"] = info.get("verify")
        return row

    n, t0 = 0, time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex, out.open("a") as fh:
        for fut in as_completed([ex.submit(work, r) for r in recs]):
            row = fut.result()
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            n += 1
            if n % 25 == 0:
                print(f"{n}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
