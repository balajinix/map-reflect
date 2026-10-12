#!/usr/bin/env python3
"""Protocol v2 reflection: six arms on one frozen first pass, with matched context.

Every arm sees the same source property, the same candidate list (the first 20 distinct ids of the first-pass
ranking, alphabetical, with descriptions) and the same previous ranking (sorted by score, with scores and
reasons). Arms differ only in what the protocol says they differ in:

  naive        instruction that the previous top answer is likely wrong and should be changed boldly
  self_refine  two calls: feedback on the previous answer, then a refined answer using that feedback
               (Self-Refine; the two instructions are the intrinsic self-correction prompts of Huang et al.)
  A            neutral instruction: keep the previous answer unless there is a concrete reason it is wrong
  AC           A, then a pairwise verifier (old answer vs revision, order fixed by a hash); keep the revision
               only if the verifier picks it
  B            A plus schema evidence the first pass did not see: data types, sibling fields of the source and
               of the top 5 previous candidates, a same-table marker, and one-to-one conflicts with the other
               source fields' previous answers
  BC           B, then the verifier, which also sees each option's type and siblings

The first-pass answer and margin follow protocol v2 (map_reflect_runner.ranking_margin): scores sorted
descending, margin s1 - s2 (one candidate: s2 = 0; ties: 0; no usable score: invalid, counted wrong, margin 0).
One parser (top1) reads every arm. Development and test fields come from a protocol v2 split manifest.

Every call is tagged with a usage stage (reflect:<arm>, verify:<arm>) and, when TOKEN_LOG is set, logged with
its tokens and seconds by the runner's clients.

Usage (from the repo root):
  TOKEN_LOG=results/naacl/v2/tokens/spider__gptoss.jsonl .venv/bin/python scripts/naacl/reflect_v2.py \
      --base results/naacl/v2/first/spider__gptoss --manifest data/splits/spider.json --split test \
      --lookup data/spider_api_lookup.json --model endpoint_a/openai/gpt-oss-120b-a100 \
      --arms naive,self_refine,A,AC,B,BC --out results/naacl/v2/reflect/spider__gptoss/fields.jsonl
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
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "map_reflect" / ".env")

from map_reflect.map_reflect_runner import (  # noqa: E402
    MapReflectConfig, EndpointABatchClient, EndpointBBatchClient, first_json, format_prompt_for_model, normalize_field_id,
    ranking_margin, resolve_ranked_id, safe_parse_json_array, safe_parse_json_object, set_usage_stage)
from eval.calc_accuracy import fields_match  # noqa: E402
from uq_gate_budget import load_slot  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts" / "naacl"))
import field_first as FF  # noqa: E402  (treatment ff1, MAPREFLECT_FIELD_FIRST=1; off by default)

N_CAND = 20
N_SIB = 6
N_EVID_CANDS = 5
ALL_ARMS = ["naive", "self_refine", "A", "AC", "B", "BC"]
FEEDBACK_INSTR = "Review your previous answer and find problems with your answer."
REFINE_INSTR = "Based on the problems you found, improve your answer."
JSON_SPEC = ('Return a JSON object: {"analysis": "one or two sentences", "revised_ranking": '
             '[{"id": "...", "score": 0.0, "reason": "..."}, ...]}\nwith ids exactly as shown, best first.')


def v41(flag):
    """True when a method v4.1 repair is switched on: its own variable (flag=1) or MAPREFLECT_V41=1 for all of them.
    Read at call time. Every repair is off by default, so method v4 reproduces exactly.

      MAPREFLECT_TARGET_NAME     prompts name the target schema (_id: id or name) instead of the generic "TARGET"
      MAPREFLECT_REVISED_SELECTED  the revised margin is the selected answer's, ids counted once (reflect_v3)
      MAPREFLECT_ROBUST_JSON     JSON is read block by block with json.loads(strict=False) semantics, the first block
                                 of the expected shape wins, and a ranking keeps its well-formed entries
                                 (map_reflect_runner.robust_json)
    """
    return os.environ.get("MAPREFLECT_V41") == "1" or os.environ.get(flag) == "1"


def _key_id(x):
    """The schema part of a stored record key: a dict's "id" only. Record keys, resume and reuse joins and the
    hash that orders the verifier's options are built from this, so it must not change."""
    return x.get("id", "") if isinstance(x, dict) else str(x or "")


def _id(x):
    """A schema's name for prompts: a dict's "id", else its "name" (the Valentine and held-out targets are
    {"name": ..., "properties": [...]} with no "id"). Never used for record keys (_key_id)."""
    if isinstance(x, dict):
        return str(x.get("id") or x.get("name") or "")
    return str(x or "")


def records(d, keep_ids):
    """Fields with ground truth whose sample is in keep_ids, with the protocol v2 first pass and margin."""
    label = _id if v41("MAPREFLECT_TARGET_NAME") else _key_id
    for r in d["results"]:
        sid = str(r.get("sample_id", ""))
        if sid not in keep_ids:
            continue
        skey = f"{_key_id(r.get('expected_source'))}|{_key_id(r.get('expected_target'))}|{sid}"
        gt = {str(k).lower(): (v if isinstance(v, list) else [v])
              for k, v in (r.get("ground_truth_mapping", {}) or {}).items()}
        preds = r.get("predictions") or {}
        sorted_rank = {}
        for fk, fv in preds.items():
            if isinstance(fv, dict):
                ir = [c for c in (fv.get("initial_ranking") or []) if isinstance(c, dict) and c.get("id")]
                sorted_rank[fk] = sorted(enumerate(ir), key=lambda t: (-_num(t[1].get("score")), t[0]))
        tops = {}
        for fk, sr in sorted_rank.items():
            top, _, ok = ranking_margin([c for _, c in sr])
            tops[fk] = top if ok else None
        for fk, fv in preds.items():
            if not isinstance(fv, dict) or fk.lower() not in gt:
                continue
            ranking = [c for _, c in sorted_rank.get(fk, [])]
            top, margin, valid = ranking_margin(ranking)
            yield dict(key=f"{skey}||{fk.lower()}", sample_id=sid, field=fk,
                       gts=[str(g) for g in gt[fk.lower()] if g], ranking=ranking, base=top, margin=margin,
                       valid=valid, siblings={k: v for k, v in tops.items() if k != fk and v},
                       src=label(r.get("expected_source")) or "SOURCE", tgt=label(r.get("expected_target")) or "TARGET")


def _num(x):
    try:
        v = float(x)
        return v if v == v else float("-inf")
    except (TypeError, ValueError):
        return float("-inf")


class Prompts:
    def __init__(self, model, lookup):
        self.model, self.lookup = model, lookup
        self.by_type = {}
        for k, v in lookup.items():
            if isinstance(v, dict) and v.get("type_name"):
                self.by_type.setdefault((v.get("product_name"), v.get("type_name")), []).append(k)

    def meta(self, k):
        v = self.lookup.get(k, {})
        return v if isinstance(v, dict) else {}

    def desc(self, k):
        v = self.lookup.get(k, {})
        if FF.on() and FF.parts(v):
            return FF.field_text(v, k)  # the property's own definition and ETL notes first (field_first.py)
        return (v.get("description", k) if isinstance(v, dict) else str(v))[:300]

    def tables(self, keys):
        """Treatment ff1: the shared table descriptions of these properties, once per table; "" when it is off."""
        return FF.table_block([self.meta(k) for k in keys]) if FF.on() else ""

    def dtype(self, k):
        m = self.meta(k)
        t = m.get("field_type") or m.get("type") or "unknown"
        flags = [f for f, on in (("list", m.get("is_list")), ("required", m.get("is_required")))
                 if str(on).lower() == "true"]
        return t + (f" ({', '.join(flags)})" if flags else "")

    def siblings(self, k):
        m = self.meta(k)
        group = self.by_type.get((m.get("product_name"), m.get("type_name")), [])
        names = [self.meta(g).get("field_name") or g for g in group if g != k][:N_SIB]
        return m.get("type_name") or "?", names

    def fmt(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return format_prompt_for_model(msgs, self.model.replace("endpoint_b/", "").replace("endpoint_a/", "")
                                       .replace("litellm/", ""))[0]

    def cands(self, rec):
        seen, out = set(), []
        for c in rec["ranking"]:
            cid = str(c["id"])
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
            if len(out) == N_CAND:
                break
        return sorted(out)

    def evidence(self, rec):
        st, ss = self.siblings(rec["field"])
        lines = [f"Source {rec['field']}: type {self.dtype(rec['field'])}; belongs to {st}"
                 + (f" with fields {', '.join(ss)}" if ss else "")]
        top = []
        for c in rec["ranking"]:
            if c["id"] not in top:
                top.append(c["id"])
            if len(top) == N_EVID_CANDS:
                break
        src_group = (self.meta(rec["field"]).get("product_name"), self.meta(rec["field"]).get("type_name"))
        for c in top:
            ct, cs = self.siblings(c)
            same = (self.meta(c).get("product_name"), self.meta(c).get("type_name")) == src_group and src_group[1]
            lines.append(f"Candidate {c}: type {self.dtype(c)}; belongs to {ct}"
                         + (" (the same table/type as the source)" if same else "")
                         + (f" with fields {', '.join(cs)}" if cs else ""))
        clash = [f for f, t in rec["siblings"].items() if t == rec["base"]]
        lines.append("One-to-one conflicts: " + (
            f"your previous answer {rec['base']} is also your previous answer for {', '.join(clash)} in this "
            "sample; each target should be claimed by at most one source property" if clash else "none"))
        return "\n".join(lines)

    def context(self, rec, with_evidence=False):
        """The block every arm shares: task, source, candidates and the previous ranking."""
        cands = self.cands(rec)
        if with_evidence:
            cand_block = "\n".join(f"{i+1}. id: {c}\n   type: {self.dtype(c)}\n   text: {c} -- {self.desc(c)}"
                                   for i, c in enumerate(cands))
        else:
            cand_block = "\n".join(f"{i+1}. id: {c}\n   text: {c} -- {self.desc(c)}" for i, c in enumerate(cands))
        prior = "\n".join(f"{i+1}. {c.get('id')} (score={c.get('score')}) - {c.get('reason', '')}"
                          for i, c in enumerate(rec["ranking"][:N_CAND]))
        body = f"""You previously ranked candidate {rec['tgt']} properties for a {rec['src']} property.

{rec['src']} property: {rec['field']}
Description: {self.desc(rec['field'])}{self.tables([rec['field']] + cands)}

Candidates:
{cand_block}

Your previous ranking (best first):
{prior}"""
        if with_evidence:
            body += "\n\nEvidence you did not have in your first ranking:\n" + self.evidence(rec)
        return body, cands

    def naive(self, rec):
        body, cands = self.context(rec)
        user = body + """

Your previous top answer is likely wrong: it probably matches by name similarity instead of meaning. Assume it
is incorrect, look for the candidate that represents the same data, and change your top answer. Be bold.

""" + JSON_SPEC
        return self.fmt("You are an expert schema mapper reviewing your own work.", user), cands

    def neutral(self, rec, with_evidence=False):
        body, cands = self.context(rec, with_evidence)
        hint = " and the evidence" if with_evidence else ""
        user = body + f"""

Review your previous ranking against the descriptions{hint}. Keep your previous top answer unless you find a
concrete reason it is wrong, such as a different meaning, data type or entity than the source property.
Do not change it only because another candidate has a similar name.

""" + JSON_SPEC
        return self.fmt("You are an expert schema mapper reviewing your own work.", user), cands

    def sr_feedback(self, rec):
        body, _ = self.context(rec)
        return self.fmt("You are an expert schema mapper.",
                        f"{body}\n\n{FEEDBACK_INSTR} Answer in at most five sentences.")

    def sr_refine(self, rec, fb):
        body, cands = self.context(rec)
        user = (f"{body}\n\n{FEEDBACK_INSTR}\n\nYour feedback on your previous answer:\n{fb.strip()[:2000]}\n\n"
                f"{REFINE_INSTR}\n\n{JSON_SPEC}")
        return self.fmt("You are an expert schema mapper.", user), cands

    def verify(self, rec, a, b, with_evidence=False):
        first, second = (a, b) if int(hashlib.md5(rec["key"].encode()).hexdigest(), 16) % 2 == 0 else (b, a)

        def opt(x):
            if not with_evidence:
                return f"{x} -- {self.desc(x)}"
            t, sib = self.siblings(x)
            return f"{x} -- {self.desc(x)} [type {self.dtype(x)}; in {t}" + (f" with {', '.join(sib)}" if sib else "") + "]"
        src = self.desc(rec["field"]) + (f" [type {self.dtype(rec['field'])}]" if with_evidence else "")
        user = f"""Which target property is the correct mapping for this {rec['src']} property?

{rec['src']} property: {rec['field']}
Description: {src}{self.tables([rec['field'], first, second])}

Option 1: {opt(first)}
Option 2: {opt(second)}

Compare meaning, data type and entity, not name similarity. Return a JSON object:
{{"choice": 1 or 2, "reason": "one sentence"}}"""
        return self.fmt("You are an expert schema mapper.", user), (first, second)


ABSTAIN = object()
INVALID = object()


def top1(resp, cands=None):
    """The single parser for every arm: the first id of revised_ranking that names a shown candidate.

    Ids are resolved against the candidates shown in the prompt (map_reflect_runner.resolve_ranked_id: exact,
    case-insensitive, then a unique field-name match; a number is a position in the list). Returns the candidate;
    ABSTAIN for a well-formed answer with an empty revised_ranking (the model declines every candidate); INVALID
    when the ranking names no shown candidate (both counted wrong, as for the first pass); None when nothing can
    be read.
    """
    obj = safe_parse_json_object(resp or "", want="revised_ranking")
    if isinstance(obj, dict) and isinstance(obj.get("revised_ranking"), list) and not obj["revised_ranking"]:
        return ABSTAIN
    arr = obj.get("revised_ranking") if isinstance(obj, dict) else None
    if not arr:
        arr = safe_parse_json_array(resp or "")
    ids = [str(item["id"]) for item in arr or [] if isinstance(item, dict) and "id" in item]
    if not ids:
        # A long ranking can be cut off or carry a stray character; the first ranked id is still readable.
        m = re.search(r'"revised_ranking"\s*:\s*\[\s*\{\s*"id"\s*:\s*"([^"]+)"', resp or "")
        ids = [m.group(1)] if m else []
    if not ids:
        return None
    for x in ids:
        r = _resolve(x, cands)
        if r:
            return r
    return INVALID


def _resolve(s, cands):
    if cands and s.isdigit() and 1 <= int(s) <= len(cands):
        return cands[int(s) - 1]
    if not cands:
        return None
    return resolve_ranked_id(s, cands)


def choice(resp):
    obj = safe_parse_json_object(resp or "", want="choice")
    try:
        c = int(obj.get("choice")) if isinstance(obj, dict) else None
    except (TypeError, ValueError):
        c = None
    return c if c in (1, 2) else None


def revised_status(resp, cands=None):
    """Parse status of a reflection response (logged under method v4.1; it never changes the answer):
      ok            a JSON object with revised_ranking whose entries are all well-formed
      partial       read, but some entries were not objects with an id, or named no shown candidate
      quotes_fixed  read only after single quotes became double quotes
      array_only    no revised_ranking object, but a bare ranking array was read
      truncated     no complete JSON; the first ranked id was read from a cut-off response
      abstain       a well-formed empty revised_ranking
      invalid       the ranking names no shown candidate
      unparsed      nothing could be read (empty or not JSON)
    """
    t = top1(resp, cands)
    if t is ABSTAIN:
        return "abstain"
    if t is INVALID:
        return "invalid"
    if t is None:
        return "unparsed"
    obj, st = first_json(resp or "", dict, lambda o: "revised_ranking" in o)
    arr = obj.get("revised_ranking") if isinstance(obj, dict) else None
    if not isinstance(arr, list):
        arr, st = first_json(resp or "", list, lambda v: any(isinstance(x, dict) for x in v))
        if arr is None:
            return "truncated"
        st = "array_only" if st == "ok" else st
    good = [x for x in arr if isinstance(x, dict) and "id" in x and _resolve(str(x["id"]), cands)]
    if st != "ok":
        return st
    return "ok" if len(good) == len(arr) else "partial"


def choice_status(resp):
    """Parse status of a verifier response: ok, quotes_fixed, no_choice (JSON read, but choice is not 1 or 2) or
    unparsed (no JSON object). Without this a verifier response that cannot be read looks like a rejection."""
    obj, st = first_json(resp or "", dict, lambda o: "choice" in o)
    if obj is None:
        obj, st = first_json(resp or "", dict)
        if obj is None:
            return "unparsed"
    try:
        c = int(obj.get("choice"))
    except (TypeError, ValueError):
        return "no_choice"
    return st if c in (1, 2) else "no_choice"


def make_client(model):
    cfg = MapReflectConfig(use_cache=False, temperature=0.0)
    return (EndpointABatchClient if model.startswith("endpoint_a/") else EndpointBBatchClient)(model, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="first-pass run directory (runner output)")
    ap.add_argument("--manifest", required=True, help="protocol v2 split manifest")
    ap.add_argument("--split", choices=["dev", "test", "test_grouped"], default="test",
                    help="test excludes the earlier development split; test_grouped is the grouped test before that "
                         "exclusion (a superset, so the exclusion can be applied at analysis time)")
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--model", required=True, help="endpoint_b/<id>, endpoint_a/<org>/<id> or litellm/<id>")
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mask-target", action="store_true", help="retrieval: never name the target schema in a prompt")
    ap.add_argument("--redo-keys", help="with --repair: also recompute the field keys listed in this file")
    ap.add_argument("--repair", action="store_true",
                    help="recompute every row of --out that has an unparsed arm (rows written before abstentions "
                         "were told apart from unreadable responses), and rewrite the file")
    ap.add_argument("--keys-file", help="run only these record keys (one per line); default: every field of the split")
    ap.add_argument("--raw-log", default=os.environ.get("MAPREFLECT_RAW_LOG"),
                    help="append every call (key, arm, stage, prompt md5, candidates, raw response, parse status) to this "
                         "jsonl and write <out>.manifest.json (flags, lookup sha256, arguments); default "
                         "MAPREFLECT_RAW_LOG, else off (the rows and prompts are the same either way)")
    ap.add_argument("--keys-only", action="store_true",
                    help="with --repair: recompute only the --redo-keys rows; an unparsed response stays as the "
                         "model's output and is counted wrong")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    bad = set(arms) - set(ALL_ARMS)
    if bad:
        sys.exit(f"unknown arms {bad}")

    d = load_slot(a.base)
    if d is None:
        sys.exit(f"no runner output in {a.base}")
    keep = set(json.load(open(a.manifest))[a.split])
    lookup = json.load(open(a.lookup))
    if FF.on():  # treatment ff1 (the R-X1 rerun): an ff1 lookup, TOKEN_LOG and a manifest, as in reflect_v4
        FF.require(lookup, a.lookup)
    recs = list(records(d, keep))
    if a.mask_target:
        for r in recs:
            r["tgt"] = "TARGET"
    if a.keys_file:
        only = {l.strip() for l in open(a.keys_file) if l.strip()}
        recs = [r for r in recs if r["key"] in only]
    if a.limit:
        recs = recs[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    if a.repair and out.exists():
        old_rows = [json.loads(l) for l in out.open() if l.strip()]
        redo = set() if a.keys_only else {r["key"] for r in old_rows if any(r.get(f"{x}_unparsed") for x in ALL_ARMS)}
        if a.redo_keys:
            redo |= {l.strip() for l in open(a.redo_keys) if l.strip()}
        kept = [r for r in old_rows if r["key"] not in redo]
        recs = [r for r in recs if r["key"] in redo]
        tmp = out.with_suffix(".repair.tmp")
        with tmp.open("w") as fh:
            for r in kept:
                fh.write(json.dumps(r) + "\n")
        tmp.replace(out)  # the repaired rows are appended below
        print(f"repair: {len(redo)} rows to recompute", flush=True)
    done = {json.loads(l)["key"] for l in out.open()} if out.exists() else set()
    recs = [r for r in recs if r["key"] not in done]
    print(f"{len(done)} done, {len(recs)} to run", flush=True)
    if FF.on():
        print(f"treatment ff1: manifest {FF.manifest(out, a.lookup, dict(script='reflect_v2.py', base=a.base, split=a.split, arms=arms))}", flush=True)
    client = make_client(a.model)
    P = Prompts(a.model, lookup)
    lock = threading.Lock()
    ok = lambda p, gts: int(bool(p) and any(fields_match(g, p) for g in gts))

    raw_path = a.raw_log
    if raw_path:
        man = dict(script="reflect_v2.py", args=vars(a), lookup=a.lookup,
                   lookup_sha256=hashlib.sha256(Path(a.lookup).read_bytes()).hexdigest(),
                   flags={k: v for k, v in sorted(os.environ.items()) if k.startswith("MAPREFLECT_")},
                   token_log=os.environ.get("TOKEN_LOG"))
        Path(str(out) + ".manifest.json").write_text(json.dumps(man, indent=1) + "\n")

    def call(prompt, stage, rec=None, cands=None):
        set_usage_stage(stage)
        resp = client.generate_batch([prompt], use_cache=False)[0]
        if raw_path and rec is not None:
            kind = stage.split(":", 1)[0]
            st = ("text" if stage == "reflect:self_refine" and cands is None
                  else revised_status(resp, cands) if kind == "reflect" else choice_status(resp))
            line = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), model=a.model, run=os.environ.get("TOKEN_RUN", ""),
                        key=rec["key"], stage=stage, prompt_md5=hashlib.md5(prompt.encode()).hexdigest(),
                        candidates=list(cands or []), response=resp, parse_status=st)
            with lock:
                with open(raw_path, "a") as fh:
                    fh.write(json.dumps(line) + "\n")
        return resp

    def revise(rec, arm, prompt, cands, row):
        t = top1(call(prompt, f"reflect:{arm}", rec, cands), cands)
        base = rec["base"] or ""
        row[f"{arm}_abstain"] = t is ABSTAIN
        row[f"{arm}_invalid"] = t is INVALID
        if t is ABSTAIN or t is INVALID:
            row[arm], row[f"{arm}_unparsed"] = "", False
        else:
            row[arm], row[f"{arm}_unparsed"] = (t or base), t is None
        row[f"{arm}_ok"] = ok(row[arm], rec["gts"])

    def verified(rec, arm, src_arm, with_evidence, row):
        base = rec["base"] or ""
        row[arm] = base
        if row.get(f"{src_arm}_abstain") or row.get(f"{src_arm}_invalid"):
            pass  # no shown candidate to confirm in a pairwise check; the first answer stands
        elif row[src_arm] != base and base:
            vp, (o1, o2) = P.verify(rec, base, row[src_arm], with_evidence)
            picked = {1: o1, 2: o2}.get(choice(call(vp, f"verify:{arm}", rec, (o1, o2))))
            row[f"{arm}_verifier_pick"] = picked
            row[arm] = row[src_arm] if picked == row[src_arm] else base
        elif row[src_arm] != base:
            row[arm] = row[src_arm]  # invalid first pass: nothing to defend
        row[f"{arm}_ok"] = ok(row[arm], rec["gts"])

    def work(rec):
        base = rec["base"] or ""
        row = dict(key=rec["key"], sample_id=rec["sample_id"], gts=rec["gts"], margin=rec["margin"],
                   valid=rec["valid"], base=base, base_ok=ok(base, rec["gts"]))
        if not rec["valid"]:
            # No usable first-pass ranking: no candidate list to reflect over, so no call is made and every arm
            # is wrong, like the first pass.
            for x in arms + (["A"] if "AC" in arms else []) + (["B"] if "BC" in arms else []):
                row[x], row[f"{x}_ok"], row[f"{x}_unparsed"] = "", 0, False
            row["no_first_pass"] = True
            return row
        if "naive" in arms:
            revise(rec, "naive", *P.naive(rec), row)
        if "self_refine" in arms:
            fb = call(P.sr_feedback(rec), "reflect:self_refine", rec) or ""
            revise(rec, "self_refine", *P.sr_refine(rec, fb), row)
        if "A" in arms or "AC" in arms:
            revise(rec, "A", *P.neutral(rec), row)
            if "AC" in arms:
                verified(rec, "AC", "A", False, row)
        if "B" in arms or "BC" in arms:
            revise(rec, "B", *P.neutral(rec, with_evidence=True), row)
            if "BC" in arms:
                verified(rec, "BC", "B", True, row)
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
