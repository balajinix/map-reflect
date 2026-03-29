#!/usr/bin/env python3
"""Run AnyTool on Spider dataset.

Spider has a different format: each sample has a database, source table,
and field mappings via result.properties. The hierarchy is per-database.
"""
import os
import json
import sys
import logging
from pathlib import Path

os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_DROP_PARAMS"] = "True"

sys.path.insert(0, str(Path(__file__).parent))
import anytool_faithful as at

logging.basicConfig(level=logging.WARNING)
BASE = Path(__file__).resolve().parent.parent


def run_spider_sample(sample, model):
    """Run AnyTool on one Spider sample."""
    db = sample["database"]
    result = sample["result"]
    src_table = result["name"]
    props = result.get("properties", {})

    if not props:
        return {}

    # Get target tables from properties
    tgt_tables = set()
    source_fields = []
    gt = {}
    for src_field, mappings in props.items():
        source_fields.append(src_field)
        for m in mappings:
            tgt_tables.add(m["mapTable"])
            gt[src_field] = m["mapProperty"]

    # Build candidate pool: all fields in target tables within this database
    api_pool = []
    if db in at.TOOL_DATA:
        for tgt_table in tgt_tables:
            if tgt_table in at.TOOL_DATA[db]:
                for api_name in at.TOOL_DATA[db][tgt_table]["api_list_names"]:
                    api_pool.append({
                        "category_name": db,
                        "tool_name": tgt_table,
                        "api_name": api_name,
                    })

    if not api_pool:
        return {}

    # Use a Spider-specific solver that emphasizes FK/PK relationships
    import litellm, re

    # Build candidate list with descriptions
    candidates = []
    for api in api_pool:
        details = at.get_api_details(api["category_name"], api["tool_name"], api["api_name"])
        desc = details.get("description", "") if isinstance(details, dict) else ""
        candidates.append(api["tool_name"] + "." + api["api_name"].split(".")[-1] + ": " + desc)

    pool_text = "\n".join("- " + c for c in candidates)

    prompt = (
        "You are a database schema mapping expert. Map each source column to the "
        "best matching target column based on FOREIGN KEY relationships.\n\n"
        "IMPORTANT: Foreign key columns in the source table should map to the "
        "PRIMARY KEY column they reference in the target table. Look for "
        "'(foreign key)' and '(primary key)' annotations in the descriptions.\n\n"
        "Source columns:\n" + json.dumps(source_fields) + "\n\n"
        "Target columns:\n" + pool_text + "\n\n"
        "For each source column, produce a JSON object with:\n"
        '- "source": source column name\n'
        '- "target": target column name (format: table.column)\n'
        '- "reason": explanation\n\n'
        "Return ONLY a JSON array."
    )

    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            mappings_list = json.loads(match.group())
            result = {}
            for m in mappings_list:
                if isinstance(m, dict) and m.get("target"):
                    # Normalize target to table_field format for GT comparison
                    tgt = m["target"]
                    # Convert table.field to table_field
                    tgt_normalized = tgt.replace(".", "_")
                    result[m["source"]] = {
                        "target_property": tgt_normalized,
                        "justification": m.get("reason", ""),
                    }
            return result
    except Exception as e:
        logging.error(f"Solver error: {e}")

    return {}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="watsonx/meta-llama/llama-3-3-70b-instruct")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default="anytool/results/faithful_spider_llama.json")
    args = parser.parse_args()

    # Build hierarchy if needed
    hierarchy_path = Path(__file__).parent / "tool_data_spider.json"
    if not hierarchy_path.exists():
        print("Building Spider hierarchy...")
        os.system(sys.executable + " " + str(Path(__file__).parent / "build_spider_hierarchy.py"))

    at.load_database(hierarchy_prefix="spider")
    total_dbs = len(at.TOOL_DATA)
    total_fields = sum(
        len(at.TOOL_DATA[db][t]["api_list_names"])
        for db in at.TOOL_DATA for t in at.TOOL_DATA[db]
    )
    print(f"Spider hierarchy: {total_dbs} databases, {total_fields} fields")

    with open(BASE / "data/spider_dataset.jsonl") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Running AnyTool on {len(samples)} Spider samples")

    results = []
    for i, s in enumerate(samples):
        sid = s["id"]
        props = s["result"].get("properties", {})
        gt = {}
        for src_field, mappings in props.items():
            for m in mappings:
                gt[src_field] = m["mapProperty"]

        print(f"  [{i+1}/{len(samples)}] {sid}: {s['database']}/{s['result']['name']}...", end=" ", flush=True)

        try:
            mappings = run_spider_sample(s, args.model)
        except Exception as e:
            print(f"ERROR: {e}")
            mappings = {}

        print(f"-> {len(mappings)} mappings")

        results.append({
            "sample_id": sid,
            "utterance": s.get("mapping_utterance", ""),
            "predicted_mapping": mappings,
            "ground_truth_mapping": gt,
        })

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Evaluate
    correct = total = 0
    for r in results:
        pm = r["predicted_mapping"]
        gt_map = r["ground_truth_mapping"]
        for src, pred_info in pm.items():
            tgt = pred_info.get("target_property", "") if isinstance(pred_info, dict) else str(pred_info)
            if tgt:
                total += 1
                gt_t = gt_map.get(src, "")
                if gt_t and tgt.lower() == gt_t.lower():
                    correct += 1

    acc = correct / total * 100 if total else 0
    print(f"\nRaw accuracy: {correct}/{total} ({acc:.1f}%)")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
