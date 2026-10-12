#!/usr/bin/env python3
"""Method v5: the first-pass candidate list of every field, with coverage, computed once per dataset (no model call).

The named target type's fields (the runner's full candidate list for the field, the source field removed) are ordered
by hybrid retrieval (dense + BM25 with reciprocal rank fusion, the runner's retrieve_candidates; fields the retriever
does not reach follow in stored order). The first pass shows:
  the whole type      when it fits the prompt budget: at most MAX_FIELDS fields and MAX_TOKENS estimated tokens of
                      candidate text (characters / 4)
  the hybrid top-K    otherwise, the longest retrieval-ordered prefix that fits the same budget
Per field: shown (the list above), type_rest (the type's fields beyond it, retrieval order), truncated (True when
type_rest is not empty), wider (as in v4, unused unless a development rule allows it) and hybrid20.

Writes data/v5_pools/<dataset>.json in the v4 pool format, so reflect_v4.py widens over type_rest unchanged.
Usage: .venv/bin/python scripts/naacl/v5_pools.py <data dir for the index> [datasets...]
"""
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "naacl")]
logging.disable(logging.WARNING)
from map_reflect.map_reflect_runner import (  # noqa: E402
    MapReflectConfig, MapReflectRunner, index_chroma, load_dataset, retrieve_candidates)
from v3_widen import DSNAME, EVAL, _id, lookup_file, toolbench_tools  # noqa: E402

WIDER_MAX = 50
MAX_FIELDS = 60
MAX_TOKENS = 6000


def main():
    ddir = Path(sys.argv[1])
    todo = sys.argv[2:]
    out_dir = ROOT / "data/v5_pools"
    out_dir.mkdir(parents=True, exist_ok=True)
    tools = toolbench_tools()

    def tool_of(api):
        hits = [t for t in tools if api.lower().startswith(t + "_")]
        return max(hits, key=len) if hits else api.lower()

    for lk in todo:
        mk = lambda full: MapReflectRunner(
            property_lookup_file=str(ROOT / lookup_file(lk)), data_dir=str(ddir), model="endpoint_b/vendor/granite-4-h-small",
            dataset=DSNAME[lk], config=MapReflectConfig(use_cache=False, top_k=10, neighbor_k=0,
                                                        use_constrained_assignment=False, full_list=full))
        r, rf = mk(False), mk(True)
        L = r.property_lookup
        coll = f"v3_{lk}_catalogue"
        index_chroma(coll, L, r.chroma_dir)
        meta = lambda k: L.get(k, {}) if isinstance(L.get(k), dict) else {}
        samples = load_dataset(ROOT / "data", DSNAME[lk], None, ROOT / EVAL[lk] if lk in EVAL else None)
        man = json.load(open(ROOT / f"data/splits/{lk}.json"))
        keep = set(man.get("dev", [])) | set(man.get("test_grouped", man.get("test", [])))
        if man.get("all_test"):
            keep = {str(s.get("sample_id")) for s in samples}
        nl_schema = defaultdict(set)
        if lk == "nl2gql":
            for s in samples:
                nl_schema[_id(s.get("expected_source")).lower()].add(_id(s.get("expected_target")).lower())
        by_prod, by_type, by_tool = defaultdict(list), defaultdict(list), defaultdict(list)
        for k in L:
            by_prod[meta(k).get("product_name")].append(k)
            by_type[(meta(k).get("product_name"), meta(k).get("type_name"))].append(k)
            if lk == "toolbench":
                by_tool[tool_of(k.split(".", 1)[0])].append(k)
        out = {}
        for s in samples:
            sid = str(s.get("sample_id"))
            if sid not in keep:
                continue
            for fld in (s.get("ground_truth_mapping") or {}):
                shown = r.shown_candidate_ids(s, fld)  # v4's list; only used to find the target schema
                full, _ = rf._get_candidates_for_sample(fld, s, rf.collection_name(), rf._parsed_utterance(s))
                tpool = [c["id"] for c in full if c["id"] != fld]
                if lk == "nl2gql":
                    tables = nl_schema[_id(s.get("expected_source")).lower()]
                    schema = {k for k in L if str(meta(k).get("type_name", "")).lower() in tables}
                elif lk == "toolbench":
                    schema = set(by_tool[tool_of(_id(s.get("expected_target")))])
                elif lk.startswith("bioml"):
                    schema = set(s.get("target_fields") or [])
                elif lk.startswith("valentine"):
                    schema = {k for c in shown for k in by_type[(meta(c).get("product_name"), meta(c).get("type_name"))]}
                else:
                    schema = {k for c in shown for k in by_prod[meta(c).get("product_name")]}
                tset = set(tpool)
                wider_pool = schema - tset - {fld}
                desc = meta(fld).get("description", fld)
                order = [str(h.get("id")) for h in retrieve_candidates(f"{fld}: {desc}", coll, r.chroma_dir,
                                                                         min(len(L), 5000))]
                rank = {k: i for i, k in enumerate(order)}
                pos = {k: i for i, k in enumerate(tpool)}
                ordered = sorted(tpool, key=lambda k: (rank.get(k, len(order)), pos[k]))
                shown, toks = [], 0
                for c in ordered:
                    t = len(f"{c} -- {meta(c).get('description', c)}") / 4
                    if len(shown) >= MAX_FIELDS or (shown and toks + t > MAX_TOKENS):
                        break
                    shown.append(c)
                    toks += t
                type_rest = ordered[len(shown):]
                wider = [k for k in order if k in wider_pool][:WIDER_MAX]
                out[f"{sid}||{fld}"] = dict(shown=shown, type_rest=type_rest, truncated=bool(type_rest), wider=wider,
                                            hybrid20=ordered[:20])
        json.dump(out, open(out_dir / f"{lk}.json", "w"))
        n = len(out)
        sz = [len(v["shown"]) for v in out.values()]
        print(lk, n, "fields; whole type shown for", sum(1 for v in out.values() if not v["truncated"]),
              "; shown size", min(sz) if sz else 0, "to", max(sz) if sz else 0, flush=True)


if __name__ == "__main__":
    main()
