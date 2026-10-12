#!/usr/bin/env python3
"""Convert OMAP datasets (from the api_mapping repo) into map_reflect input format.

OMAP layout (per dataset <name>, e.g. imdb_saki, omop_cms):
  <name>_property_lookup.json  {field_id: {database_name, table_name, field_name, description, ...}}
  <name>_ground_truth.json     {source_field_id: [target_field_id, ...]}

Source vs target databases are inferred from the ground truth: keys' db prefix = source,
values' db prefix = target. Emits (per-sample candidate pool = the full target schema,
same faithful scoping used for Valentine):
  data/omap_<name>_api_lookup.json   flat {tgt_field_id: {...}}
  data/omap_<name>_dataset.jsonl     one sample per source table

Usage: build_omap_dataset.py <omap_dir> <name> [<name2> ...]
  e.g. build_omap_dataset.py <LOCAL_PATH> imdb_saki omop_cms
"""
import sys, os, json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def db_of(fid):
    return str(fid).split(".", 1)[0]


def convert(omap_dir, name):
    lookup = json.load(open(Path(omap_dir) / f"{name}_property_lookup.json"))
    gt = json.load(open(Path(omap_dir) / f"{name}_ground_truth.json"))
    if not gt:
        print(f"{name}: empty ground truth"); return

    src_db = db_of(next(iter(gt)))
    tgt_dbs = {db_of(t) for tl in gt.values() for t in (tl if isinstance(tl, list) else [tl])}
    tgt_db = sorted(tgt_dbs)[0] if tgt_dbs else ""

    # flat target api_lookup from property_lookup entries in the target db
    api_lookup = {}
    for fid, meta in lookup.items():
        if db_of(fid) != tgt_db:
            continue
        if not isinstance(meta, dict):
            meta = {}
        api_lookup[fid] = {
            "product_name": tgt_db,
            "type_name": meta.get("table_name", ""),
            "field_name": meta.get("field_name", fid),
            "name": str(fid).split(".")[-1],
            "type": meta.get("type", "unknown"),
            "database": tgt_db,
            "description": meta.get("description", fid),
        }
    tgt_field_ids = list(api_lookup.keys())
    if not tgt_field_ids:
        print(f"{name}: no target fields for db '{tgt_db}'"); return

    # group source fields (with a GT) by source table
    by_table = defaultdict(dict)
    for sfid, targets in gt.items():
        if db_of(sfid) != src_db:
            continue
        tl = [t for t in (targets if isinstance(targets, list) else [targets])
              if t in api_lookup]
        if not tl:
            continue
        table = ".".join(str(sfid).split(".")[:2])  # db.table
        by_table[table][sfid] = tl

    samples = []
    for table, mapping in by_table.items():
        tid = f"tgt_{name}"
        samples.append({
            "id": f"omap_{name}_{table.replace('.', '_')}",
            "utterance": f"Map columns from {table} to the {tgt_db} schema",
            "source_schema": src_db,
            "source_type": table,
            "expected_target_type": tid,
            "target_schema": {"name": tid,
                              "properties": [{"name": t} for t in tgt_field_ids]},
            "mapping": mapping,
            "metadata": {"source_table": table, "num_fields": len(mapping),
                         "benchmark": "omap", "source_db": src_db, "target_db": tgt_db},
        })

    out = f"omap_{name}"
    json.dump(api_lookup, open(DATA / f"{out}_api_lookup.json", "w"), indent=1)
    with open(DATA / f"{out}_dataset.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    n_fields = sum(len(s["mapping"]) for s in samples)
    print(f"{out}: {len(samples)} source tables, {n_fields} GT field-matches, "
          f"{len(api_lookup)} target fields ({src_db} -> {tgt_db})")
    print(f"  -> data/{out}_api_lookup.json, data/{out}_dataset.jsonl")


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    omap_dir = sys.argv[1]
    for name in sys.argv[2:]:
        convert(omap_dir, name)


if __name__ == "__main__":
    main()
