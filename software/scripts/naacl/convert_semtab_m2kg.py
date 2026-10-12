#!/usr/bin/env python3
"""SemTab 2024 Metadata to KG, round 1 (Zenodo record 14207376, metadata2kg.zip, CC BY 4.0) -> map_reflect format.

Input: the release's round1/ directory: r1_test_metadata.jsonl (one web-table column per line: id, label, table_id,
table_name, table_columns), r1_glossary.jsonl (2,881 DBpedia ontology properties: id, label, desc) and
r1_test_metadata_GT.csv (column id, property IRI; one target per column). Table names and headers only; the track
gives no cell values.

Task: column (source) to DBpedia property (target), one gold target each.
  - source field id  semtab_<table_id>__<label>, description 'Column "<label>" of table "<table_name>"; other
    columns: ...'; lookup type_name = the table, so the reflection evidence lists the table's other columns
  - target field id  dbpedia__<local name of the IRI>, description "<label>: <desc>", one type (the glossary)
  - the glossary is too large for one ranking prompt, so each column is its own record and its target_schema lists
    the glossary in the order of the runner's hybrid retriever (scripts/naacl/newdata_order.py, gold-blind) for the
    runner's own query "<field id>: <description>", cut to the first KEEP; the first pass shows the first 20, the full-list control shows all KEEP,
    and v3 widening draws from the whole glossary
  - grouped split unit: the table

Usage: convert_semtab_m2kg.py <round1 dir>   (writes data/semtab_m2kg_r1_{dataset.jsonl,api_lookup.json})
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from newdata_order import ROOT, order_pools  # noqa: E402

NAME, KEEP = "semtab_m2kg_r1", 100


def local(iri):
    return re.split(r"[/#]", iri.rstrip("/"))[-1]


def main():
    raw = Path(sys.argv[1])
    cols = [json.loads(l) for l in open(raw / "r1_test_metadata.jsonl")]
    gloss = [json.loads(l) for l in open(raw / "r1_glossary.jsonl")]
    gt = dict(r for r in csv.reader(open(raw / "r1_test_metadata_GT.csv")) if len(r) == 2)
    lookup, iri2id = {}, {}
    for g in gloss:
        fid = f"dbpedia__{local(g['id'])}"
        assert fid not in lookup, fid
        iri2id[g["id"]] = fid
        lookup[fid] = dict(product_name="dbpedia", type_name="dbpedia_properties", field_name=local(g["id"]),
                           name=g.get("label") or local(g["id"]), type="property", database="dbpedia",
                           description=f"{g.get('label', '')}: {g.get('desc', '')}".strip(": "), iri=g["id"],
                           _pool="glossary")
    queries, pools, rec = {}, {}, []
    for c in cols:
        fid = f"semtab_{c['table_id']}__{c['label']}"
        others = [x for x in c.get("table_columns", []) if x != c["label"]]
        lookup[fid] = dict(product_name="semtab_tables", type_name=c["table_id"], field_name=c["label"],
                           name=c["label"], type="column", database="web_tables",
                           description=f'Column "{c["label"]}" of table "{c["table_name"]}"; other columns: '
                                       + ", ".join(others))
        queries[fid], pools[fid] = f"{fid}: {lookup[fid]['description']}", "glossary"
        rec.append((c, fid))
    order = order_pools(queries, pools, lookup, keep=KEEP)
    for v in lookup.values():
        v.pop("_pool", None)
    out, in_keep, in_20 = [], 0, 0
    for c, fid in rec:
        gold = iri2id[gt[c["id"]]]
        props = order[fid]
        in_keep += gold in props
        in_20 += gold in props[:20]
        out.append({
            "id": f"{NAME}_{c['id']}".lower(),
            "utterance": f"Map columns from source table {c['table_name']} to DBpedia properties",
            "source_schema": f"semtab_{c['table_id']}", "source_type": f"semtab_{c['table_id']}",
            "expected_target_type": "dbpedia_properties",
            "target_schema": {"name": "dbpedia_properties", "properties": [{"name": p} for p in props]},
            "mapping": {fid: gold},
            "metadata": {"source_table": c["table_id"], "table_name": c["table_name"], "column_id": c["id"],
                         "target_tables": ["dbpedia_properties"], "num_fields": 1, "benchmark": NAME,
                         "gold_iri": gt[c["id"]], "gold_rank_in_order": props.index(gold) + 1 if gold in props else None},
        })
    json.dump(lookup, open(ROOT / f"data/{NAME}_api_lookup.json", "w"), indent=1)
    with open(ROOT / f"data/{NAME}_dataset.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{NAME}: {len(out)} columns over {len({c['table_id'] for c, _ in rec})} tables; glossary {len(gloss)}; "
          f"gold in the first 20 shown {in_20}, in the stored {KEEP} {in_keep}")


if __name__ == "__main__":
    main()
