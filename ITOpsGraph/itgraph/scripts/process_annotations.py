#!/usr/bin/env python3
"""
Process human annotations into ground truth dataset.

Reads:
  - data/itom_hashicorp_annotation_v2.md (annotated by human)
  - data/itom_hashicorp_annotation_candidates.json (candidate metadata)
  - data/itom_hashicorp_api_lookup_v2.json (field metadata)

Outputs:
  - data/itom_hashicorp_dataset_annotated.jsonl (rebuilt dataset with proper field mappings)

Validates:
  - Bipartite constraint: each target field used at most once per type pair
  - All chosen targets exist in the api lookup
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def parse_annotation_markdown(md_path):
    """Parse the annotated markdown template.

    Expected format per type pair:
    ## itom_hashi_XXXX: product.Type → product.Type
    ...
    | Source Field | Type | Candidates | Your Choice |
    |...|...|...|...|
    | `field_name` | type | candidates | chosen_target |
    """
    with open(md_path) as f:
        content = f.read()

    annotations = {}
    current_id = None
    current_src = None
    current_tgt = None
    current_mappings = {}
    in_table = False

    for line in content.split("\n"):
        # Match section header: ## itom_hashi_XXXX: src.Type → tgt.Type
        header_m = re.match(
            r"^##\s+(itom_hashi_\d+):\s+(\w+\.\w+)\s*(?:→|->|-->)\s*(\w+\.\w+)",
            line,
        )
        if header_m:
            # Save previous section
            if current_id and current_mappings:
                annotations[current_id] = {
                    "source": current_src,
                    "target": current_tgt,
                    "mappings": dict(current_mappings),
                }
            current_id = header_m.group(1)
            current_src = header_m.group(2)
            current_tgt = header_m.group(3)
            current_mappings = {}
            in_table = False
            continue

        # Detect table header
        if "Source Field" in line and "Your Choice" in line:
            in_table = True
            continue

        # Skip separator line
        if in_table and line.strip().startswith("|---"):
            continue

        # Parse table row
        if in_table and line.strip().startswith("|"):
            cols = [c.strip() for c in line.split("|")]
            # cols[0] is empty (before first |), cols[-1] is empty (after last |)
            cols = [c for c in cols if c != ""]
            if len(cols) >= 4:
                src_field = cols[0].strip("`").strip()
                choice = cols[3].strip()

                if choice and choice != "--" and choice != "—":
                    current_mappings[src_field] = choice
                else:
                    current_mappings[src_field] = None  # No match
        elif in_table and not line.strip():
            in_table = False

    # Save last section
    if current_id and current_mappings:
        annotations[current_id] = {
            "source": current_src,
            "target": current_tgt,
            "mappings": dict(current_mappings),
        }

    return annotations


def resolve_target_field(choice, tgt_product, tgt_type, api_lookup):
    """Resolve a short field name to the full field key.

    The annotator may write just the field name (e.g., 'displayName')
    or the full key (e.g., 'instana_Entity.displayName').
    """
    if not choice:
        return None

    # Already a full key?
    if choice in api_lookup:
        return choice

    # Try with product_Type prefix
    full_key = f"{tgt_product}_{tgt_type}.{choice}"
    if full_key in api_lookup:
        return full_key

    # Try case-insensitive search
    choice_lower = choice.lower()
    prefix = f"{tgt_product}_{tgt_type}."
    for key in api_lookup:
        if key.startswith(prefix) and key.split(".")[-1].lower() == choice_lower:
            return key

    return None


def validate_bipartite(mappings, sample_id):
    """Check bipartite constraint: each target field used at most once."""
    target_claims = defaultdict(list)
    for src, tgt in mappings.items():
        if tgt:
            target_claims[tgt].append(src)

    violations = {t: srcs for t, srcs in target_claims.items() if len(srcs) > 1}
    if violations:
        print(f"  WARNING [{sample_id}]: Bipartite violations:")
        for tgt, srcs in violations.items():
            tgt_short = tgt.split(".")[-1]
            src_shorts = [s.split(".")[-1] for s in srcs]
            print(f"    {tgt_short} claimed by: {', '.join(src_shorts)}")
    return violations


def process_annotations(annotations, candidates_data, api_lookup, original_samples):
    """Process annotations into ground truth dataset."""

    # Build sample lookup
    sample_lookup = {s["id"]: s for s in original_samples}

    new_samples = []
    stats = {
        "total_pairs": 0,
        "annotated_pairs": 0,
        "total_fields": 0,
        "matched_fields": 0,
        "unmatched_fields": 0,
        "bipartite_violations": 0,
        "unresolved_targets": 0,
    }

    for sample_id, orig_sample in sample_lookup.items():
        annotation = annotations.get(sample_id)
        cand_data = candidates_data.get(sample_id, {})

        src_product = orig_sample["source_schema"]
        tgt_product = orig_sample["target_schema"]
        tgt_type = orig_sample["metadata"]["target_type"]

        stats["total_pairs"] += 1

        if not annotation:
            # Use original (synthetic) mapping as fallback
            new_samples.append(orig_sample)
            continue

        stats["annotated_pairs"] += 1

        # Resolve field names to full keys
        resolved_mappings = {}
        src_prefix = f"{src_product}_{orig_sample['metadata']['source_type']}."

        for src_field_short, choice in annotation["mappings"].items():
            # Find the full source key
            src_key = None
            for key in api_lookup:
                if key.startswith(src_prefix) and key.split(".")[-1] == src_field_short:
                    src_key = key
                    break
            if not src_key:
                src_key = f"{src_prefix}{src_field_short}"

            stats["total_fields"] += 1

            if choice is None:
                resolved_mappings[src_key] = None
                stats["unmatched_fields"] += 1
            else:
                tgt_key = resolve_target_field(choice, tgt_product, tgt_type, api_lookup)
                if tgt_key:
                    resolved_mappings[src_key] = tgt_key
                    stats["matched_fields"] += 1
                else:
                    print(f"  WARNING [{sample_id}]: Cannot resolve target '{choice}' for {src_field_short}")
                    resolved_mappings[src_key] = f"{tgt_product}_{tgt_type}.{choice}"
                    stats["unresolved_targets"] += 1
                    stats["matched_fields"] += 1

        # Validate bipartite
        violations = validate_bipartite(resolved_mappings, sample_id)
        if violations:
            stats["bipartite_violations"] += len(violations)

        # Build new mapping (exclude None values for unmatched)
        clean_mapping = {
            src: tgt for src, tgt in resolved_mappings.items() if tgt is not None
        }

        # Create new sample
        new_sample = dict(orig_sample)
        new_sample["mapping"] = clean_mapping
        new_sample["metadata"] = dict(orig_sample["metadata"])
        new_sample["metadata"]["annotation_source"] = "human_v2"
        new_sample["metadata"]["annotation_timestamp"] = datetime.now().isoformat()
        new_sample["metadata"]["unmatched_fields"] = [
            src for src, tgt in resolved_mappings.items() if tgt is None
        ]
        new_samples.append(new_sample)

    return new_samples, stats


def main():
    print("=== Processing ITOM-Hashicorp Annotations ===\n")

    # Load inputs
    md_path = DATA_DIR / "itom_hashicorp_annotation_v2.md"
    if not md_path.exists():
        print(f"ERROR: Annotation file not found: {md_path}")
        print("Run generate_annotation_candidates.py first, then annotate the template.")
        sys.exit(1)

    print(f"Parsing annotations from {md_path}...")
    annotations = parse_annotation_markdown(md_path)
    print(f"  {len(annotations)} type pairs parsed")

    # Load candidates
    cand_path = DATA_DIR / "itom_hashicorp_annotation_candidates.json"
    candidates = {}
    if cand_path.exists():
        with open(cand_path) as f:
            candidates = json.load(f)
        print(f"  {len(candidates)} candidate sets loaded")

    # Load API lookup
    v2_path = DATA_DIR / "itom_hashicorp_api_lookup_v2.json"
    v1_path = DATA_DIR / "itom_hashicorp_api_lookup.json"
    lookup_path = v2_path if v2_path.exists() else v1_path
    with open(lookup_path) as f:
        api_lookup = json.load(f)
    print(f"  {len(api_lookup)} lookup entries")

    # Load original dataset
    original = []
    with open(DATA_DIR / "itom_hashicorp_dataset.jsonl") as f:
        for line in f:
            if line.strip():
                original.append(json.loads(line))
    print(f"  {len(original)} original samples")

    # Process
    print("\nProcessing annotations...")
    new_samples, stats = process_annotations(annotations, candidates, api_lookup, original)

    # Save
    out_path = DATA_DIR / "itom_hashicorp_dataset_annotated.jsonl"
    with open(out_path, "w") as f:
        for s in new_samples:
            f.write(json.dumps(s) + "\n")
    print(f"\nSaved: {out_path}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"  Total type pairs: {stats['total_pairs']}")
    print(f"  Annotated pairs:  {stats['annotated_pairs']}")
    print(f"  Total fields:     {stats['total_fields']}")
    print(f"  Matched fields:   {stats['matched_fields']}")
    print(f"  Unmatched (--):   {stats['unmatched_fields']}")
    print(f"  Bipartite violations: {stats['bipartite_violations']}")
    print(f"  Unresolved targets: {stats['unresolved_targets']}")

    if stats["bipartite_violations"] > 0:
        print("\n  ⚠️  Fix bipartite violations before using this dataset!")


if __name__ == "__main__":
    main()
