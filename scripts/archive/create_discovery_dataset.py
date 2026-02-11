#!/usr/bin/env python3
"""
Create a discovery-focused dataset where the target mapping is NOT given in the utterance.

The model must discover the correct target field given only:
- Source product
- Source type
- Source field
- Target product (optionally)

This tests actual schema mapping ability, not just parsing.
"""

import json
import random
from pathlib import Path

# Utterance templates - target is NOT mentioned
TEMPLATES = [
    # Source field only - find equivalent in target product
    "What {target_product} field corresponds to {source_product} {source_type}.{source_field}?",
    "Find the {target_product} equivalent of {source_product} {source_field}",
    "Map {source_product} {source_type}.{source_field} to {target_product}",
    "What in {target_product} matches {source_product} {source_field}?",
    "Discover {target_product} mapping for {source_product} {source_type} {source_field}",

    # More natural phrasing
    "I need to map {source_field} from {source_product} {source_type} to {target_product}",
    "How does {source_product} {source_field} translate to {target_product}?",
    "Find {target_product} field equivalent to {source_product} {source_type}.{source_field}",
    "What's the {target_product} counterpart of {source_product} {source_field}?",
    "Link {source_product} {source_type}.{source_field} to appropriate {target_product} field",

    # Without target product (harder - must infer from context)
    "Find schema mapping for {source_product} {source_type}.{source_field}",
    "What maps to {source_product} {source_field}?",
    "Discover equivalent field for {source_product} {source_type} {source_field}",
]


def create_discovery_utterance(source_product: str, source_type: str, source_field: str,
                                target_product: str, template_idx: int = None) -> str:
    """Create an utterance that requires discovering the target mapping."""
    if template_idx is None:
        template_idx = random.randint(0, len(TEMPLATES) - 1)

    template = TEMPLATES[template_idx % len(TEMPLATES)]

    return template.format(
        source_product=source_product,
        source_type=source_type,
        source_field=source_field,
        target_product=target_product
    )


def main():
    input_file = Path("data/itopsgraph_utterance_dataset.jsonl")
    output_file = Path("data/itopsgraph_discovery_dataset.jsonl")

    samples = []
    with open(input_file) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from {input_file}")

    # Group by unique mapping to avoid duplicates
    unique_mappings = {}
    for sample in samples:
        for src_field, tgt_field in sample["mapping"].items():
            key = (src_field, tgt_field)
            if key not in unique_mappings:
                unique_mappings[key] = sample

    print(f"Found {len(unique_mappings)} unique field mappings")

    # Create discovery dataset
    discovery_samples = []
    for idx, ((src_full, tgt_full), sample) in enumerate(unique_mappings.items()):
        # Parse source: "turbonomic_Action.actionID" -> product, type, field
        src_parts = src_full.split(".")
        src_type_full = src_parts[0]  # "turbonomic_Action"
        src_field = src_parts[1] if len(src_parts) > 1 else src_parts[0]

        src_product = sample["source_schema"]  # "turbonomic"
        # Extract type name: "turbonomic_Action" -> "Action"
        src_type = src_type_full.replace(f"{src_product}_", "")

        # Parse target
        tgt_parts = tgt_full.split(".")
        tgt_product = sample["target_schema"]

        # Create multiple utterance variants
        for template_idx in range(3):  # 3 variants per mapping
            utterance = create_discovery_utterance(
                source_product=src_product,
                source_type=src_type,
                source_field=src_field,
                target_product=tgt_product,
                template_idx=(idx * 3 + template_idx) % len(TEMPLATES)
            )

            discovery_sample = {
                "id": f"itopsgraph_disc_{idx:04d}_{template_idx}",
                "utterance": utterance,
                "source_schema": src_product,
                "target_schema": tgt_product,
                "source_type": src_type,
                "source_field": src_field,
                "mapping": {src_full: tgt_full},
                "metadata": {
                    "original_id": sample["id"],
                    "template_idx": template_idx,
                    "task_type": "discovery"
                }
            }
            discovery_samples.append(discovery_sample)

    # Shuffle to mix template types
    random.seed(42)
    random.shuffle(discovery_samples)

    # Write output
    with open(output_file, "w") as f:
        for sample in discovery_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Created {len(discovery_samples)} discovery samples")
    print(f"Written to {output_file}")

    # Show examples
    print("\nExample utterances:")
    for sample in discovery_samples[:10]:
        print(f"  {sample['utterance']}")
        print(f"    -> {list(sample['mapping'].values())[0]}")
        print()


if __name__ == "__main__":
    main()
