#!/usr/bin/env python3
"""Find many-to-many and bipartite examples for the paper."""
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

with open(BASE / "data/itgraph_dataset_high_confidence.jsonl") as f:
    samples = [json.loads(l) for l in f if l.strip()]

# Many-to-many: one source type -> multiple target types
src_to_tgts = defaultdict(list)
for s in samples:
    src = s["source_schema"] + "." + s["metadata"]["source_type"]
    tgt = s["target_schema"] + "." + s["metadata"]["target_type"]
    src_to_tgts[src].append((tgt, s))

print("=" * 70)
print("MANY-TO-MANY EXAMPLES (1 source type -> multiple target types)")
print("=" * 70)
for src, tgt_list in sorted(src_to_tgts.items(), key=lambda x: -len(x[1])):
    if len(tgt_list) >= 3:
        print(f"\n{src} maps to {len(tgt_list)} target types:")
        for tgt, s in tgt_list:
            n_fields = len(s["mapping"])
            methods = s.get("metadata", {}).get("n_methods", "?")
            print(f"  -> {tgt} ({n_fields} fields, {methods} methods agree)")

# Also find reverse: multiple sources -> one target
tgt_to_srcs = defaultdict(list)
for s in samples:
    src = s["source_schema"] + "." + s["metadata"]["source_type"]
    tgt = s["target_schema"] + "." + s["metadata"]["target_type"]
    tgt_to_srcs[tgt].append((src, s))

print("\n" + "=" * 70)
print("MANY-TO-MANY (multiple sources -> 1 target)")
print("=" * 70)
for tgt, src_list in sorted(tgt_to_srcs.items(), key=lambda x: -len(x[1])):
    if len(src_list) >= 3:
        print(f"\n{tgt} is target of {len(src_list)} source types:")
        for src, s in src_list:
            print(f"  <- {src}")

# Bipartite field examples
print("\n" + "=" * 70)
print("BIPARTITE FIELD MAPPING EXAMPLES")
print("=" * 70)
count = 0
for s in samples:
    m = s["mapping"]
    if 5 <= len(m) <= 12:
        src = s["source_schema"] + "." + s["metadata"]["source_type"]
        tgt = s["target_schema"] + "." + s["metadata"]["target_type"]
        tgt_fields = [v.split(".")[-1] for v in m.values()]
        # Check bipartite
        if len(tgt_fields) == len(set(tgt_fields)):
            n_methods = s.get("metadata", {}).get("n_methods", "?")
            print(f"\n{src} -> {tgt}")
            print(f"  ({len(m)} fields, {n_methods} methods agree, bipartite-clean)")
            for sf, tf in sorted(m.items()):
                sf_short = sf.split(".")[-1]
                tf_short = tf.split(".")[-1]
                print(f"    {sf_short:30s} -> {tf_short}")
            count += 1
            if count >= 3:
                break
