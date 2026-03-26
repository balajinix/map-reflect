#!/usr/bin/env python3
"""
Build ITGraph dataset from method predictions (no human annotation needed).

Uses consensus across multiple methods/models as signal:
- Type-level: many-to-many (one source type can map to multiple target types)
- Field-level: bipartite (each target field used at most once per type pair)

For field mappings, aggregates votes from all methods and applies greedy
bipartite assignment to resolve conflicts.

Output:
  data/itgraph_dataset.jsonl           — full dataset (all 12 products)
  data/itgraph_type_mapping_summary.json — type-level mapping with confidence
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

ALL_PRODUCTS = {
    "instana", "concert", "turbonomic", "hcp_terraform",
    "infragraph", "packer", "aws", "azure", "gcp", "hcp",
}


def load_api_lookup():
    path = DATA_DIR / "itom_hashicorp_api_lookup_v2.json"
    if not path.exists():
        path = DATA_DIR / "itom_hashicorp_api_lookup.json"
    with open(path) as f:
        return json.load(f)


def field_to_type(field_key):
    """Extract (product, type) from field key like 'concert_Application.id'."""
    if "." in field_key:
        prefix = field_key.split(".")[0]
    else:
        return None, None
    # prefix is like 'concert_Application'
    for prod in sorted(ALL_PRODUCTS, key=len, reverse=True):
        if prefix.startswith(prod + "_"):
            type_name = prefix[len(prod) + 1:]
            return prod, type_name
    return None, None


def load_all_predictions():
    """Load field-level predictions from all result files.

    Returns:
      predictions: dict of
        (sample_id) -> {src_field -> [(tgt_field, score, method_model)]}
    """
    predictions = defaultdict(lambda: defaultdict(list))

    result_sources = [
        ("Map-Reflect", [
            BASE_DIR / "map_reflect" / "results",
            BASE_DIR / "results",
        ]),
        ("MatchMaker", [
            BASE_DIR / "match_maker" / "results",
        ]),
        ("AnyTool", [
            BASE_DIR / "anytool" / "results",
        ]),
    ]

    for method_name, dirs in result_sources:
        for result_dir in dirs:
            if not result_dir.exists():
                continue
            for result_file in sorted(result_dir.glob("*.json")):
                fname = result_file.name.lower()
                if "checkpoint" in fname:
                    continue
                # Must be ITOM_Hashicorp
                if "itom" not in fname and "ITOM" not in result_file.name:
                    continue

                # Determine model
                model = "Unknown"
                if "llama" in fname:
                    model = "Llama"
                elif "granite" in fname:
                    model = "Granite"
                elif "mistral" in fname:
                    model = "Mistral"
                method_model = f"{method_name} ({model})"

                try:
                    with open(result_file) as f:
                        data = json.load(f)
                except Exception:
                    continue

                results = data.get("results", []) if isinstance(data, dict) else data
                if not isinstance(results, list):
                    continue

                for result in results:
                    sid = result.get("sample_id", "")
                    if not sid:
                        continue

                    # Map-Reflect format: predictions -> initial_ranking
                    preds = result.get("predictions", {})
                    if isinstance(preds, dict) and preds:
                        for src_field, pred_data in preds.items():
                            if not isinstance(pred_data, dict):
                                continue
                            ranking = pred_data.get("initial_ranking", [])
                            for rank, r in enumerate(ranking[:3]):
                                tgt = r.get("id", "")
                                score = r.get("score", 0)
                                if tgt and tgt != "NO_MATCH":
                                    # Discount lower-ranked predictions
                                    adj_score = score * (1.0 - 0.2 * rank)
                                    predictions[sid][src_field].append(
                                        (tgt, adj_score, method_model)
                                    )

                    # MatchMaker format: predicted_mapping -> target_property
                    pred_map = result.get("predicted_mapping", {})
                    if pred_map and not preds:
                        for src_field, tgt_info in pred_map.items():
                            if isinstance(tgt_info, dict):
                                tgt = tgt_info.get("target_property", "")
                                justification = tgt_info.get("justification", "")
                                # Extract confidence from justification
                                score = 0.7
                                conf_match = re.search(r"(\d+\.?\d*)", justification)
                                if conf_match:
                                    score = float(conf_match.group(1)) / 100.0
                            else:
                                tgt = str(tgt_info)
                                score = 0.5
                            if tgt:
                                predictions[sid][src_field].append(
                                    (tgt, min(score, 1.0), method_model)
                                )

                print(f"  Loaded {result_file.name}: {method_model}")

    return predictions


def aggregate_field_votes(field_predictions):
    """For each source field, aggregate votes across methods.

    Returns {src_field: [(tgt_field, consensus_score, methods)]}
    """
    aggregated = {}

    for src_field, preds in field_predictions.items():
        # Group by target field
        target_votes = defaultdict(list)
        target_methods = defaultdict(set)

        for tgt_field, score, method_model in preds:
            target_votes[tgt_field].append(score)
            target_methods[tgt_field].add(method_model)

        # Score each target: combine method count and average score
        scored = []
        for tgt_field, scores in target_votes.items():
            n_methods = len(target_methods[tgt_field])
            avg_score = sum(scores) / len(scores)
            # Consensus score: weight by both method agreement and avg prediction score
            consensus = (0.4 * min(n_methods / 3.0, 1.0)) + (0.6 * avg_score)
            scored.append((tgt_field, consensus, n_methods, target_methods[tgt_field]))

        scored.sort(key=lambda x: -x[1])
        aggregated[src_field] = scored

    return aggregated


def bipartite_assign(aggregated_fields, min_consensus=0.3):
    """Greedy bipartite assignment: each target field used at most once.

    Returns {src_field: tgt_field} for matched fields,
            and list of unmatched source fields.
    """
    # Sort source fields by their best candidate's consensus score (descending)
    # so high-confidence mappings are assigned first
    src_by_confidence = []
    for src_field, candidates in aggregated_fields.items():
        if candidates:
            best_score = candidates[0][1]
            src_by_confidence.append((src_field, best_score, candidates))

    src_by_confidence.sort(key=lambda x: -x[1])

    assigned = {}
    used_targets = set()
    unmatched = []

    for src_field, _, candidates in src_by_confidence:
        matched = False
        for tgt_field, consensus, n_methods, methods in candidates:
            if consensus < min_consensus:
                break
            if tgt_field not in used_targets:
                assigned[src_field] = {
                    "target": tgt_field,
                    "consensus": round(consensus, 3),
                    "n_methods": n_methods,
                    "methods": sorted(methods),
                }
                used_targets.add(tgt_field)
                matched = True
                break
        if not matched:
            unmatched.append(src_field)

    return assigned, unmatched


def discover_type_pairs(predictions, api_lookup):
    """Discover type-level mappings from field predictions.

    A type pair is discovered when at least one method predicts a
    field mapping between them.
    """
    type_pairs = defaultdict(lambda: {
        "methods": set(),
        "field_count": 0,
        "n_methods": 0,
    })

    for sid, field_preds in predictions.items():
        for src_field, preds in field_preds.items():
            src_p, src_t = field_to_type(src_field)
            if not src_p:
                continue

            for tgt_field, score, method_model in preds:
                tgt_p, tgt_t = field_to_type(tgt_field)
                if not tgt_p or (src_p == tgt_p and src_t == tgt_t):
                    continue

                pair_key = (src_p, src_t, tgt_p, tgt_t)
                type_pairs[pair_key]["methods"].add(method_model)

    # Compute stats
    for pair_key, info in type_pairs.items():
        info["n_methods"] = len(info["methods"])
        info["methods"] = sorted(info["methods"])

    return type_pairs


def build_dataset(predictions, api_lookup, min_methods=1, min_consensus=0.3):
    """Build the full dataset with bipartite field mappings.

    Args:
        min_methods: minimum number of methods that must predict a type pair
        min_consensus: minimum consensus score for a field mapping
    """
    # Discover type pairs
    type_pairs = discover_type_pairs(predictions, api_lookup)
    print(f"\nDiscovered {len(type_pairs)} type pairs from predictions")

    # Filter by method count
    filtered_pairs = {
        k: v for k, v in type_pairs.items()
        if v["n_methods"] >= min_methods
    }
    print(f"After filtering (>= {min_methods} methods): {len(filtered_pairs)} type pairs")

    # Also load the type mapping lookup for additional type-level info
    lookup_path = DATA_DIR / "itopsgraph_type_mapping_lookup.json"
    type_lookup = {}
    if lookup_path.exists():
        with open(lookup_path) as f:
            type_lookup = json.load(f)

    # Build samples
    samples = []
    sample_id = 0
    stats = {
        "total_pairs": 0,
        "total_fields": 0,
        "matched_fields": 0,
        "unmatched_fields": 0,
        "by_method_count": Counter(),
    }

    # Group predictions by type pair
    pair_predictions = defaultdict(lambda: defaultdict(list))
    for sid, field_preds in predictions.items():
        for src_field, preds in field_preds.items():
            src_p, src_t = field_to_type(src_field)
            if not src_p:
                continue
            for tgt_field, score, method_model in preds:
                tgt_p, tgt_t = field_to_type(tgt_field)
                if not tgt_p:
                    continue
                pair_key = (src_p, src_t, tgt_p, tgt_t)
                if pair_key in filtered_pairs:
                    pair_predictions[pair_key][src_field].append(
                        (tgt_field, score, method_model)
                    )

    for pair_key in sorted(filtered_pairs.keys()):
        src_p, src_t, tgt_p, tgt_t = pair_key
        pair_info = filtered_pairs[pair_key]

        # Get field predictions for this pair
        field_preds = pair_predictions.get(pair_key, {})
        if not field_preds:
            continue

        # Aggregate votes and apply bipartite assignment
        aggregated = aggregate_field_votes(field_preds)
        assigned, unmatched = bipartite_assign(aggregated, min_consensus)

        if not assigned:
            continue

        stats["total_pairs"] += 1
        stats["matched_fields"] += len(assigned)
        stats["unmatched_fields"] += len(unmatched)
        stats["total_fields"] += len(assigned) + len(unmatched)
        stats["by_method_count"][pair_info["n_methods"]] += 1

        # Build mapping dict
        mapping = {}
        for src_field, assign_info in assigned.items():
            mapping[src_field] = assign_info["target"]

        # Get type-level methods from lookup
        src_key_prefixed = f"{src_p}_{src_t}"
        tgt_key_prefixed = f"{tgt_p}_{tgt_t}"
        type_methods = (
            type_lookup.get(src_t, {}).get(tgt_t, [])
            or type_lookup.get(src_key_prefixed, {}).get(tgt_key_prefixed, [])
            or pair_info["methods"]
        )

        # Determine relationship
        relationship = "maps_to"
        if src_p in {"instana"} and tgt_p in {"aws", "azure", "gcp", "infragraph"}:
            relationship = "monitors"
        elif src_p in {"turbonomic"} and tgt_p in {"aws", "azure", "gcp", "infragraph"}:
            relationship = "optimizes"
        elif src_p in {"concert"} and tgt_p in {"aws", "azure", "gcp", "infragraph", "packer", "hcp_terraform"}:
            relationship = "governs"

        utterance = f"Map {src_p} {src_t} to {tgt_p} schema"

        sample = {
            "id": f"itgraph_{sample_id:04d}",
            "utterance": utterance,
            "source_schema": src_p,
            "target_schema": tgt_p,
            "expected_target_type": f"{tgt_p}_{tgt_t}",
            "mapping": mapping,
            "metadata": {
                "source_type": src_t,
                "target_type": tgt_t,
                "relationship": relationship,
                "description": f"Predicted by {', '.join(pair_info['methods'][:3])}",
                "source_entity": src_t,
                "target_entity": tgt_t,
                "n_methods": pair_info["n_methods"],
                "methods": pair_info["methods"],
                "type_methods": type_methods,
                "field_assignments": {
                    src: {
                        "target": info["target"],
                        "consensus": info["consensus"],
                        "n_methods": info["n_methods"],
                    }
                    for src, info in assigned.items()
                },
                "unmatched_fields": unmatched,
                "dataset_source": "method_consensus",
                "timestamp": datetime.now().isoformat(),
            },
        }
        samples.append(sample)
        sample_id += 1

    return samples, stats


def main():
    print("=" * 60)
    print("Building ITGraph Dataset from Method Predictions")
    print("=" * 60)

    api_lookup = load_api_lookup()
    print(f"API lookup: {len(api_lookup)} fields")

    print("\nLoading predictions from all methods...")
    predictions = load_all_predictions()
    total_preds = sum(
        sum(len(preds) for preds in field_preds.values())
        for field_preds in predictions.values()
    )
    print(f"  {len(predictions)} samples, {total_preds} total field predictions")

    print("\nBuilding dataset (min_methods=1, min_consensus=0.3)...")
    samples, stats = build_dataset(predictions, api_lookup, min_methods=1, min_consensus=0.3)

    # Save dataset
    out_path = DATA_DIR / "itgraph_dataset.jsonl"
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"\nSaved: {out_path}")

    # Also build a higher-confidence dataset (>=2 methods)
    print("\nBuilding high-confidence dataset (min_methods=2)...")
    samples_hc, stats_hc = build_dataset(predictions, api_lookup, min_methods=2, min_consensus=0.4)

    out_path_hc = DATA_DIR / "itgraph_dataset_high_confidence.jsonl"
    with open(out_path_hc, "w") as f:
        for s in samples_hc:
            f.write(json.dumps(s) + "\n")
    print(f"Saved: {out_path_hc}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Dataset Summary (all predictions)")
    print(f"  Type pairs: {stats['total_pairs']}")
    print(f"  Total fields: {stats['total_fields']}")
    print(f"  Matched fields (bipartite): {stats['matched_fields']}")
    print(f"  Unmatched fields: {stats['unmatched_fields']}")
    print(f"  By method count: {dict(sorted(stats['by_method_count'].items()))}")

    print(f"\nHigh-Confidence Dataset (>=2 methods)")
    print(f"  Type pairs: {stats_hc['total_pairs']}")
    print(f"  Matched fields: {stats_hc['matched_fields']}")
    print(f"  By method count: {dict(sorted(stats_hc['by_method_count'].items()))}")

    # Product coverage
    products_seen = set()
    for s in samples:
        products_seen.add(s["source_schema"])
        products_seen.add(s["target_schema"])
    print(f"\nProducts covered: {sorted(products_seen)}")

    # Save type mapping summary
    type_summary = {}
    for s in samples:
        m = s["metadata"]
        key = f"{s['source_schema']}.{m['source_type']} -> {s['target_schema']}.{m['target_type']}"
        type_summary[key] = {
            "n_methods": m["n_methods"],
            "methods": m["methods"],
            "matched_fields": len(s["mapping"]),
            "unmatched_fields": len(m.get("unmatched_fields", [])),
        }

    summary_path = DATA_DIR / "itgraph_type_mapping_summary.json"
    with open(summary_path, "w") as f:
        json.dump(type_summary, f, indent=2)
    print(f"\nSaved type summary: {summary_path}")


if __name__ == "__main__":
    main()
