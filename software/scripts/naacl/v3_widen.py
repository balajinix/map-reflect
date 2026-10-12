#!/usr/bin/env python3
"""Method v3: the widened candidate lists, computed once per dataset (no model call).

For every field of every development and grouped-test sample: the candidates the first pass showed
(MapReflectRunner.shown_candidate_ids, given target), and the hybrid top-50 over the target schema from the runner's
own retrieval (retrieve_candidates: dense + BM25 with reciprocal rank fusion over the catalogue), restricted to the
target schema and with the shown candidates and the source field removed. reflect_v3 widens with the first K.

Target schema per dataset:
  ITOpsGraph, ITOpsGraph-MB, Spider, OMOP  the product (or database) the shown candidates belong to
  NL2GQL                                   the GraphQL schema: every type that is a target of a sample sharing the
                                           sample's schema
  ToolBench                                the target API's tool (longest tool name prefixing the API id)
  Valentine                                the target table (each suite pairs one table with its variants)
  Bio-ML                                   the record's own candidate list (the track's 101 candidates)

Writes data/v3_widened/<dataset>.json: {"<sample_id>||<field>": {"shown": [...], "widen": [... up to 50]}}.
Usage: .venv/bin/python scripts/naacl/v3_widen.py <data dir for the index> [datasets...]
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

DSNAME = dict(itopsgraph="ITOpsGraph", itgraph_mb="ITGraph_MB", nl2gql="NL2GQL", spider="Spider", toolbench="ToolBench",
              omop_cms="OMOP_CMS", omop_mimic="OMOP_MIMIC", omop_synthea="OMOP_Synthea",
              valentine_chembl="valentine_chembl", valentine_opendata="valentine_opendata",
              valentine_tpcdi="valentine_tpcdi", valentine_wikidata="valentine_wikidata",
              valentine_magellan="valentine_magellan", mimic_omop_rematch="mimic_omop_rematch",
              semtab_m2kg_r1="semtab_m2kg_r1", oaei_conference="oaei_conference",
              schemanet_cprd="schemanet_cprd", bioml_omim_ordo="bioml_omim_ordo",
              semtab_m2kg_r2="semtab_m2kg_r2", bioml_ncit_doid="bioml_ncit_doid", oaei_anatomy="oaei_anatomy",
              t2d_v2="t2d_v2", mimic4_omop_parciak="mimic4_omop_parciak")
EVAL = dict(itopsgraph="data/v2/itopsgraph_v2_cross_product.jsonl",
            itgraph_mb="data/v2/itgraph_mb_v3_cross_product_positives.jsonl")
K_MAX = 50


def lookup_file(lk):
    return "data/itopsgraph_api_lookup.json" if lk in ("itopsgraph", "itgraph_mb") else f"data/{lk}_api_lookup.json"


def _id(x):
    return x.get("id", x.get("name", "")) if isinstance(x, dict) else str(x or "")


def toolbench_tools():
    tools = set()
    for line in open(ROOT / "data/toolbench_dataset.jsonl"):
        r = json.loads(line)
        for side in ("source_api", "target_api"):
            n = (r.get(side) or {}).get("tool_name")
            if n:
                tools.add(n.replace(" ", "_").lower())
    return tools


def main():
    ddir = Path(sys.argv[1])
    todo = sys.argv[2:] or list(DSNAME)
    out_dir = ROOT / "data/v3_widened"
    out_dir.mkdir(parents=True, exist_ok=True)
    tools = toolbench_tools()

    def tool_of(api):
        hits = [t for t in tools if api.lower().startswith(t + "_")]
        return max(hits, key=len) if hits else api.lower()

    for lk in todo:
        cfg = MapReflectConfig(use_cache=False, top_k=10, neighbor_k=0, use_constrained_assignment=False)
        r = MapReflectRunner(property_lookup_file=str(ROOT / lookup_file(lk)), data_dir=str(ddir),
                             model="endpoint_b/vendor/granite-4-h-small", dataset=DSNAME[lk], config=cfg)
        L = r.property_lookup
        coll = f"v3_{lk}_catalogue"
        index_chroma(coll, L, r.chroma_dir)
        meta = lambda k: L.get(k, {}) if isinstance(L.get(k), dict) else {}
        samples = load_dataset(ROOT / "data", DSNAME[lk], None, ROOT / EVAL[lk] if lk in EVAL else None)
        man = json.load(open(ROOT / f"data/splits/{lk}.json"))
        keep = set(man["dev"]) | set(man.get("test_grouped", man["test"]))
        nl_schema = defaultdict(set)
        if lk == "nl2gql":
            for s in samples:
                nl_schema[_id(s.get("expected_source")).lower()].add(_id(s.get("expected_target")).lower())
        by_prod, by_type = defaultdict(list), defaultdict(list)
        for k in L:
            by_prod[meta(k).get("product_name")].append(k)
            by_type[(meta(k).get("product_name"), meta(k).get("type_name"))].append(k)
        by_tool = defaultdict(list)
        if lk == "toolbench":
            for k in L:
                by_tool[tool_of(k.split(".", 1)[0])].append(k)
        out = {}
        for s in samples:
            sid = str(s.get("sample_id"))
            if sid not in keep:
                continue
            for fld in (s.get("ground_truth_mapping") or {}):
                shown = r.shown_candidate_ids(s, fld)
                if lk == "nl2gql":
                    tables = nl_schema[_id(s.get("expected_source")).lower()]
                    pool = {k for k in L if str(meta(k).get("type_name", "")).lower() in tables}
                elif lk == "toolbench":
                    pool = set(by_tool[tool_of(_id(s.get("expected_target")))])
                elif lk.startswith("bioml"):
                    pool = set(s.get("target_fields") or [])  # the track's own 101 candidates for this source
                elif lk.startswith("valentine"):
                    pool = {k for c in shown for k in by_type[(meta(c).get("product_name"), meta(c).get("type_name"))]}
                else:
                    pool = {k for c in shown for k in by_prod[meta(c).get("product_name")]}
                pool -= set(shown) | {fld}
                widen = []
                if pool:
                    desc = meta(fld).get("description", fld)
                    hits = retrieve_candidates(f"{fld}: {desc}", coll, r.chroma_dir, min(len(L), 5000))
                    for h in hits:
                        hid = str(h.get("id"))
                        if hid in pool and hid not in widen:
                            widen.append(hid)
                            if len(widen) == K_MAX:
                                break
                out[f"{sid}||{fld}"] = dict(shown=shown, widen=widen)
        json.dump(out, open(out_dir / f"{lk}.json", "w"))
        sizes = [len(v["widen"]) for v in out.values()]
        print(lk, len(out), "fields; widened by", min(sizes) if sizes else 0, "to", max(sizes) if sizes else 0, flush=True)


if __name__ == "__main__":
    main()
