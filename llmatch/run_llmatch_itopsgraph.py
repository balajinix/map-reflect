#!/usr/bin/env python3
"""
Run LLMatch on ITOpsGraph experiments using data loaded into MongoDB.

Prerequisites:
  - Run load_itopsgraph_to_mongodb.py first
  - MongoDB running with ITOpsGraph data

Usage:
    MONGODB_HOST=mongodb://localhost:27017/llmatch python llmatch/run_llmatch_itopsgraph.py
    MONGODB_HOST=mongodb://localhost:27017/llmatch python llmatch/run_llmatch_itopsgraph.py --max-experiments 5
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

if "MONGODB_HOST" not in os.environ:
    os.environ["MONGODB_HOST"] = "mongodb://localhost:27017/llmatch"

from mongoengine import connect
connect(host=os.environ["MONGODB_HOST"])

# Add LLMatch to path
sys.path.insert(0, str(Path(__file__).parent / "LLMatch"))

from schema_match.data_models.experiment_models import (
    OntologySchemaRewrite,
    OntologyAlignmentGroundTruth,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent

# ITOpsGraph experiment pairs (top cross-product pairs by field count)
ITOPSGRAPH_EXPERIMENTS = [
    "instana-concert",
    "instana-turbonomic",
    "instana-cloudability",
    "concert-instana",
    "concert-turbonomic",
    "turbonomic-instana",
    "turbonomic-concert",
    "cloudability-instana",
    "infragraph-packer",
    "infragraph-base",
    "hcp_terraform-hcp",
    "instana-ns1",
]


def load_ground_truth(source_db, target_db):
    """Load ground truth for an experiment pair."""
    dataset_name = f"{source_db}-{target_db}"
    gt_doc = OntologyAlignmentGroundTruth.objects(dataset=dataset_name).first()
    if not gt_doc:
        return {}
    return gt_doc.data


def get_schema_description(database):
    """Get all tables and columns for a database."""
    tables = {}
    for record in OntologySchemaRewrite.objects(database=database, llm_model="original"):
        table = record.table
        if table not in tables:
            tables[table] = []
        tables[table].append({
            "column": record.column,
            "description": record.column_description,
            "type": record.column_type,
        })
    return tables


def run_llm_column_matching(source_db, target_db, model="watsonx/meta-llama/llama-3-3-70b-instruct"):
    """Run LLMatch-style column matching using LLM."""
    import litellm

    ground_truth = load_ground_truth(source_db, target_db)
    if not ground_truth:
        return {"error": f"No ground truth for {source_db}-{target_db}"}

    source_tables = get_schema_description(source_db)
    target_tables = get_schema_description(target_db)

    if not source_tables or not target_tables:
        return {"error": f"No schema data for {source_db} or {target_db}"}

    # Build source and target field lists
    source_fields = []
    for table, columns in source_tables.items():
        for col in columns:
            source_fields.append(f"{table}.{col['column']}")

    target_fields = []
    target_desc = []
    for table, columns in target_tables.items():
        for col in columns:
            target_fields.append(f"{table}.{col['column']}")
            target_desc.append(f"{table}.{col['column']}: {col['description'][:60]}")

    # Only use source fields that appear in ground truth
    gt_sources = list(ground_truth.keys())

    # Build prompt with target field descriptions
    target_text = "\n".join(f"- {d}" for d in target_desc[:100])

    mappings = {}
    for src_field in gt_sources:
        prompt = f"""Map the source field to the best matching target field.

Source ({source_db}): {src_field}

Available target fields ({target_db}):
{target_text}

Return a JSON object with:
- "target": the best matching target field (exact format as shown above), or null if no match
- "reason": one sentence explanation

Return ONLY the JSON object."""

        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a schema mapping expert. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""

            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group())
                if result.get("target"):
                    mappings[src_field] = result["target"]
        except Exception as e:
            logger.warning(f"  Error mapping {src_field}: {e}")
            continue

    # Evaluate
    correct = 0
    total = len(gt_sources)
    for src, predicted in mappings.items():
        gt_targets = ground_truth.get(src, [])
        if predicted in gt_targets:
            correct += 1

    precision = correct / len(mappings) if mappings else 0
    recall = correct / total if total else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "source_db": source_db,
        "target_db": target_db,
        "correct": correct,
        "total_gt": total,
        "total_predicted": len(mappings),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mappings": mappings,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="watsonx/meta-llama/llama-3-3-70b-instruct")
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--output", default="llmatch/results/llmatch_itopsgraph.json")
    args = parser.parse_args()

    experiments = ITOPSGRAPH_EXPERIMENTS
    if args.max_experiments:
        experiments = experiments[:args.max_experiments]

    print(f"Running LLMatch on {len(experiments)} ITOpsGraph experiments")
    print(f"Model: {args.model}")

    results = {}
    for i, exp in enumerate(experiments):
        parts = exp.split("-", 1)
        source_db, target_db = parts[0], parts[1]
        print(f"\n[{i+1}/{len(experiments)}] {exp}...")

        result = run_llm_column_matching(source_db, target_db, model=args.model)
        results[exp] = result

        if "error" not in result:
            print(f"  F1={result['f1']:.3f}  P={result['precision']:.3f}  R={result['recall']:.3f}  "
                  f"({result['correct']}/{result['total_gt']})")
        else:
            print(f"  {result['error']}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")

    # Summary
    total_correct = sum(r.get("correct", 0) for r in results.values() if "error" not in r)
    total_gt = sum(r.get("total_gt", 0) for r in results.values() if "error" not in r)
    avg_f1 = sum(r.get("f1", 0) for r in results.values() if "error" not in r) / max(1, len([r for r in results.values() if "error" not in r]))
    print(f"\nOverall: {total_correct}/{total_gt} correct, avg F1={avg_f1:.3f}")


if __name__ == "__main__":
    main()
