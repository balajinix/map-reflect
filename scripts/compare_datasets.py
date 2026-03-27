#!/usr/bin/env python3
"""Compare ITOpsGraph vs ITGraph datasets."""
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ALL_PRODUCTS = ["instana","concert","turbonomic","hcp_terraform","infragraph",
                "packer","aws","azure","gcp","hcp","cloudability","ns1",
                "watsonx-orchestrate","base","model","edge","resource"]

def get_product(field):
    for p in sorted(ALL_PRODUCTS, key=len, reverse=True):
        if field.startswith(p + "_"):
            return p
    return field.split("_")[0]

def analyze_dataset(name, path):
    with open(path) as f:
        if str(path).endswith(".jsonl"):
            samples = [json.loads(l) for l in f if l.strip()]
        else:
            d = json.load(f)
            samples = d if isinstance(d, list) else d.get("results", d.get("samples", []))

    products = set()
    src_types = set()
    tgt_types = set()
    cross = same = 0
    total_fields = 0
    src_to_tgts = defaultdict(set)
    field_mapping_type = defaultdict(int)  # 1:1, many:1, 1:many

    for s in samples:
        mapping = s.get("mapping", s.get("ground_truth", {}))
        if not mapping:
            continue

        # Get products from first field keys
        src_prods = set()
        tgt_prods = set()
        src_type_set = set()
        tgt_type_set = set()

        for src_field, tgt_field in mapping.items():
            sp = get_product(src_field)
            tp = get_product(tgt_field)
            src_prods.add(sp)
            tgt_prods.add(tp)
            # Get type from field: product_Type.field -> product_Type
            src_full = src_field.split(".")[0] if "." in src_field else src_field
            tgt_full = tgt_field.split(".")[0] if "." in tgt_field else tgt_field
            src_type_set.add(src_full)
            tgt_type_set.add(tgt_full)

        products.update(src_prods)
        products.update(tgt_prods)
        src_types.update(src_type_set)
        tgt_types.update(tgt_type_set)

        is_cross = src_prods != tgt_prods
        if is_cross:
            cross += 1
        else:
            same += 1

        total_fields += len(mapping)

        for st in src_type_set:
            for tt in tgt_type_set:
                src_to_tgts[st].add(tt)

        # Check bipartite: are target fields unique?
        tgt_fields = list(mapping.values())
        if len(tgt_fields) == len(set(tgt_fields)):
            field_mapping_type["bipartite"] += 1
        else:
            field_mapping_type["many-to-one"] += 1

    many_to_many = sum(1 for v in src_to_tgts.values() if len(v) > 1)
    max_targets = max((len(v) for v in src_to_tgts.values()), default=0)
    avg_fields = total_fields / len(samples) if samples else 0

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Samples:              {len(samples)}")
    print(f"  Products:             {len(products)} {sorted(products)}")
    print(f"  Cross-product:        {cross} ({cross*100//max(len(samples),1)}%)")
    print(f"  Same-product:         {same}")
    print(f"  Total field mappings: {total_fields}")
    print(f"  Avg fields/sample:    {avg_fields:.1f}")
    print(f"  Unique src types:     {len(src_types)}")
    print(f"  Unique tgt types:     {len(tgt_types)}")
    print(f"  M:M src types:        {many_to_many} (max {max_targets} targets)")
    print(f"  Bipartite samples:    {field_mapping_type['bipartite']}")
    print(f"  Many-to-one samples:  {field_mapping_type['many-to-one']}")

    # Check overlap with other dataset
    return samples, src_to_tgts

datasets = [
    ("ITOpsGraph (521 eval)", BASE / "data/itopsgraph_dataset.jsonl"),
    ("ITGraph (full, 419)", BASE / "data/itgraph_dataset.jsonl"),
    ("ITGraph_HC (246)", BASE / "data/itgraph_dataset_high_confidence.jsonl"),
    ("ITOM_Hashicorp (95)", BASE / "data/itom_hashicorp_dataset.jsonl"),
]

all_data = {}
for name, path in datasets:
    if path.exists():
        samples, m2m = analyze_dataset(name, path)
        all_data[name] = (samples, m2m)

# Overlap analysis
print(f"\n{'='*60}")
print(f"  OVERLAP ANALYSIS")
print(f"{'='*60}")

# Get type pairs from each dataset
def get_type_pairs(samples):
    pairs = set()
    for s in samples:
        mapping = s.get("mapping", s.get("ground_truth", {}))
        if not mapping:
            continue
        src_types = set()
        tgt_types = set()
        for sf, tf in mapping.items():
            src_types.add(sf.split(".")[0] if "." in sf else sf)
            tgt_types.add(tf.split(".")[0] if "." in tf else tf)
        for st in src_types:
            for tt in tgt_types:
                pairs.add((st, tt))
    return pairs

if "ITOpsGraph (521 eval)" in all_data and "ITGraph_HC (246)" in all_data:
    itops_pairs = get_type_pairs(all_data["ITOpsGraph (521 eval)"][0])
    itg_pairs = get_type_pairs(all_data["ITGraph_HC (246)"][0])
    overlap = itops_pairs & itg_pairs
    itops_only = itops_pairs - itg_pairs
    itg_only = itg_pairs - itops_pairs
    print(f"\n  ITOpsGraph type pairs:  {len(itops_pairs)}")
    print(f"  ITGraph_HC type pairs: {len(itg_pairs)}")
    print(f"  Overlap:               {len(overlap)}")
    print(f"  ITOpsGraph only:       {len(itops_only)}")
    print(f"  ITGraph_HC only:       {len(itg_only)}")
    print(f"  Union:                 {len(itops_pairs | itg_pairs)}")
