#!/usr/bin/env python3
"""Check Spider data format for GNN graph construction."""
import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent

with open(BASE / "data/spider_dataset.jsonl") as f:
    samples = [json.loads(l) for l in f if l.strip()]

print(f"Spider samples: {len(samples)}")

# Check format
for s in samples[:3]:
    r = s["result"]
    props = r.get("properties", {})
    print(f"\nID: {s['id']}, DB: {s['database']}")
    print(f"  result.name: {r['name']}")
    print(f"  mapping_utterance: {s.get('mapping_utterance', '?')}")
    print(f"  properties: {len(props)}")
    for k, v in list(props.items())[:3]:
        print(f"    {k} -> {v}")

# Count databases
dbs = set(s["database"] for s in samples)
print(f"\nDatabases: {len(dbs)}")

# Count source/target tables
src_tables = set()
tgt_tables = set()
field_mappings = 0
for s in samples:
    r = s["result"]
    src_tables.add(r["name"])
    props = r.get("properties", {})
    for src_field, tgt_info in props.items():
        field_mappings += 1
        if isinstance(tgt_info, dict):
            tgt_table = tgt_info.get("mapTable", "")
            tgt_tables.add(tgt_table)

print(f"Source tables: {len(src_tables)}")
print(f"Target tables: {len(tgt_tables)}")
print(f"Field mappings: {field_mappings}")

# API lookup
with open(BASE / "data/spider_api_lookup.json") as f:
    lookup = json.load(f)
print(f"\nAPI lookup: {len(lookup)} fields")
k = list(lookup.keys())[0]
print(f"  Example: {k} = {json.dumps(lookup[k])}")
