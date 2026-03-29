#!/usr/bin/env python3
"""
Load ITOpsGraph schemas and ground truth into LLMatch's MongoDB format.

Creates:
  - OntologySchemaRewrite records for each field in the API lookup
  - OntologyAlignmentGroundTruth records for each evaluation pair

Usage:
    python llmatch/load_itopsgraph_to_mongodb.py
    python llmatch/load_itopsgraph_to_mongodb.py --dry-run
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Must set MONGODB_HOST before importing mongoengine
if "MONGODB_HOST" not in os.environ:
    os.environ["MONGODB_HOST"] = "mongodb://localhost:27017/llmatch"

from mongoengine import connect
connect(host=os.environ["MONGODB_HOST"])

sys.path.insert(0, str(Path(__file__).parent / "LLMatch"))
from schema_match.data_models.experiment_models import (
    OntologySchemaRewrite,
    OntologyAlignmentGroundTruth,
)

BASE = Path(__file__).resolve().parent.parent


def load_api_lookup():
    path = BASE / "data" / "itopsgraph_api_lookup.json"
    with open(path) as f:
        return json.load(f)


def load_dataset():
    path = BASE / "data" / "itopsgraph_dataset.jsonl"
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_schemas_to_mongo(api_lookup, dry_run=False):
    """Convert API lookup fields to OntologySchemaRewrite records."""
    llm_model = "original"
    records = []
    products = set()

    for field_key, meta in api_lookup.items():
        product = meta.get("product_name", "")
        type_name = meta.get("type_name", "")
        field_name = meta.get("field_name", "")
        description = meta.get("description", "")
        field_type = meta.get("field_type", "")

        if not product or not type_name or not field_name:
            continue

        products.add(product)

        record = {
            "database": product,
            "original_table": type_name,
            "original_column": field_name,
            "table": type_name,
            "column": field_name,
            "table_description": meta.get("type_description", f"{type_name} type in {product}"),
            "column_description": description or f"{field_name} field in {type_name}",
            "column_type": field_type or "STRING",
            "llm_model": llm_model,
            "is_primary_key": field_name.lower() in ("id", "uuid", "key"),
            "is_foreign_key": False,
            "linked_table": "",
            "linked_column": "",
        }
        records.append(record)

    print(f"Schema records: {len(records)} across {len(products)} products")
    print(f"Products: {sorted(products)}")

    if not dry_run:
        # Clear existing ITOpsGraph data
        for product in products:
            deleted = OntologySchemaRewrite.objects(
                database=product, llm_model=llm_model
            ).delete()
            if deleted:
                print(f"  Cleared {deleted} existing records for {product}")

        # Insert in batches
        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            OntologySchemaRewrite.upsert_many(batch)
            print(f"  Inserted batch {i // batch_size + 1} ({len(batch)} records)")

        total = OntologySchemaRewrite.objects(llm_model=llm_model).count()
        print(f"  Total records in MongoDB: {total}")

    return products


def load_ground_truth_to_mongo(samples, dry_run=False):
    """Convert evaluation samples to OntologyAlignmentGroundTruth records."""
    # Group samples by (source_product, target_product)
    pair_mappings = defaultdict(lambda: defaultdict(list))

    for s in samples:
        mapping = s.get("mapping", {})
        if not mapping:
            continue

        # Extract products from first field
        src_fields = list(mapping.keys())
        tgt_fields = list(mapping.values())
        if not src_fields:
            continue

        # Get product names
        src_product = src_fields[0].split("_")[0] if "_" in src_fields[0] else "unknown"
        tgt_product = tgt_fields[0].split("_")[0] if "_" in tgt_fields[0] else "unknown"

        # For ITOpsGraph, use the source_schema and target_schema fields
        src_product = s.get("source_schema", src_product)
        tgt_product = s.get("target_schema", tgt_product)

        pair_key = f"{src_product}-{tgt_product}"

        for src_field, tgt_field in mapping.items():
            # Convert from product_Type.field to Type.field
            if "." in src_field:
                src_parts = src_field.split(".", 1)
                src_type_field = src_parts[0].split("_", 1)[-1] + "." + src_parts[1]
            else:
                src_type_field = src_field

            if "." in tgt_field:
                tgt_parts = tgt_field.split(".", 1)
                tgt_type_field = tgt_parts[0].split("_", 1)[-1] + "." + tgt_parts[1]
            else:
                tgt_type_field = tgt_field

            pair_mappings[pair_key][src_type_field].append(tgt_type_field)

    print(f"\nGround truth pairs: {len(pair_mappings)}")
    for pair, mappings in sorted(pair_mappings.items()):
        n_src = len(mappings)
        n_tgt = sum(len(v) for v in mappings.values())
        print(f"  {pair}: {n_src} source fields, {n_tgt} mappings")

    if not dry_run:
        for pair_key, mappings in pair_mappings.items():
            # Delete existing
            OntologyAlignmentGroundTruth.objects(dataset=pair_key).delete()

            record = {
                "dataset": pair_key,
                "data": dict(mappings),
            }
            OntologyAlignmentGroundTruth.upsert(record)
            print(f"  Saved GT for {pair_key}")

    return list(pair_mappings.keys())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Loading ITOpsGraph into LLMatch MongoDB")
    print("=" * 60)

    api_lookup = load_api_lookup()
    print(f"API lookup: {len(api_lookup)} fields")

    samples = load_dataset()
    print(f"Dataset: {len(samples)} samples")

    print("\n--- Loading Schemas ---")
    products = load_schemas_to_mongo(api_lookup, dry_run=args.dry_run)

    print("\n--- Loading Ground Truth ---")
    pairs = load_ground_truth_to_mongo(samples, dry_run=args.dry_run)

    # Print experiment config to add
    print("\n--- Add to schema_match/constants.py ---")
    print(f"DATABASES += {sorted(products)}")
    print(f"EXPERIMENTS += {sorted(pairs)}")

    if args.dry_run:
        print("\n(Dry run — no data written)")


if __name__ == "__main__":
    main()
