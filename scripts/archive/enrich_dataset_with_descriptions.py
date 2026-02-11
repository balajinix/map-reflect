#!/usr/bin/env python3
"""
Enrich ITOpsGraph dataset with API descriptions from multiple sources.

This script adds type and field descriptions to dataset samples before
sending them for schema mapping, improving the quality of mappings.

Sources (in priority order):
1. GraphQL triple-quote descriptions (concert, instana, turbonomic, etc.)
2. api_lookup.json field descriptions
3. all_products_schema.json descriptions
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
NL2CYPHER_DIR = Path("/Users/balaji/code/solis-dal-dataset/nl2cypher")
GRAPHQL_SCHEMA_DIR = NL2CYPHER_DIR / "schemas" / "graphql"
ALL_PRODUCTS_SCHEMA = NL2CYPHER_DIR / "data" / "all_products_schema.json"


def extract_graphql_descriptions(schema_dir: Path) -> Dict[str, Dict]:
    """
    Extract type and field descriptions from GraphQL schema files.

    Returns:
        Dict: {product: {type_name: {description, fields: {field: desc}}}}
    """
    descriptions = {}

    if not schema_dir.exists():
        print(f"Warning: GraphQL schema directory not found: {schema_dir}")
        return descriptions

    # Pattern to match triple-quoted descriptions
    desc_pattern = re.compile(r'"""(.*?)"""', re.DOTALL)
    type_pattern = re.compile(r'(?:""".*?"""\s*)?(type|input|enum)\s+(\w+)', re.DOTALL)
    field_pattern = re.compile(r'(?:"""(.*?)"""\s*)?(\w+)\s*(?:\([^)]*\))?\s*:\s*(\[?\w+!?\]?!?)', re.DOTALL)

    for product_dir in schema_dir.iterdir():
        if not product_dir.is_dir():
            continue

        product = product_dir.name
        descriptions[product] = {}

        for graphql_file in product_dir.rglob("*.graphql"):
            try:
                content = graphql_file.read_text()

                # Find all type definitions with their descriptions
                # Split by type definitions
                type_blocks = re.split(r'(?=(?:""".*?"""\s*)?(?:type|input|enum)\s+\w+)', content)

                for block in type_blocks:
                    if not block.strip():
                        continue

                    # Extract type description and name
                    type_match = type_pattern.search(block)
                    if not type_match:
                        continue

                    type_name = type_match.group(2)

                    # Get type description (before the type keyword)
                    type_desc = ""
                    desc_match = desc_pattern.search(block[:type_match.start()])
                    if desc_match:
                        type_desc = desc_match.group(1).strip()

                    if type_name not in descriptions[product]:
                        descriptions[product][type_name] = {
                            "description": type_desc,
                            "fields": {}
                        }
                    elif type_desc and not descriptions[product][type_name]["description"]:
                        descriptions[product][type_name]["description"] = type_desc

                    # Extract field descriptions (inside the type block)
                    brace_start = block.find("{")
                    brace_end = block.rfind("}")
                    if brace_start != -1 and brace_end != -1:
                        fields_block = block[brace_start+1:brace_end]

                        # Split by lines and process each field
                        lines = fields_block.split("\n")
                        current_desc = ""

                        for line in lines:
                            line = line.strip()

                            # Check for description
                            if line.startswith('"""'):
                                if line.endswith('"""') and len(line) > 6:
                                    current_desc = line[3:-3].strip()
                                else:
                                    # Multi-line description
                                    desc_match = desc_pattern.search(line)
                                    if desc_match:
                                        current_desc = desc_match.group(1).strip()
                            elif ":" in line and not line.startswith("#"):
                                # This is a field definition
                                field_match = re.match(r'(\w+)\s*(?:\([^)]*\))?\s*:', line)
                                if field_match:
                                    field_name = field_match.group(1)
                                    if current_desc:
                                        descriptions[product][type_name]["fields"][field_name] = current_desc
                                    current_desc = ""
                            elif not line.startswith("#") and line:
                                current_desc = ""

            except Exception as e:
                print(f"Warning: Error processing {graphql_file}: {e}")
                continue

    return descriptions


def load_all_products_schema(schema_path: Path) -> Dict:
    """Load the all_products_schema.json file."""
    if not schema_path.exists():
        print(f"Warning: Schema file not found: {schema_path}")
        return {}

    with open(schema_path) as f:
        return json.load(f)


def load_api_lookup(lookup_path: Path) -> Dict:
    """Load the api_lookup.json file."""
    if not lookup_path.exists():
        print(f"Warning: API lookup file not found: {lookup_path}")
        return {}

    with open(lookup_path) as f:
        return json.load(f)


class DescriptionEnricher:
    """Enriches dataset samples with API descriptions."""

    def __init__(
        self,
        graphql_schema_dir: Optional[Path] = None,
        all_products_schema_path: Optional[Path] = None,
        api_lookup_path: Optional[Path] = None
    ):
        self.graphql_docs = {}
        self.schema_data = {}
        self.api_lookup = {}

        # Load GraphQL descriptions
        if graphql_schema_dir and graphql_schema_dir.exists():
            print(f"Loading GraphQL descriptions from {graphql_schema_dir}...")
            self.graphql_docs = extract_graphql_descriptions(graphql_schema_dir)
            total_types = sum(len(p) for p in self.graphql_docs.values())
            print(f"  Loaded descriptions for {total_types} types from {len(self.graphql_docs)} products")

        # Load all_products_schema
        if all_products_schema_path and all_products_schema_path.exists():
            print(f"Loading schema from {all_products_schema_path}...")
            self.schema_data = load_all_products_schema(all_products_schema_path)
            if "products" in self.schema_data:
                print(f"  Loaded {len(self.schema_data['products'])} products")

        # Load api_lookup
        if api_lookup_path and api_lookup_path.exists():
            print(f"Loading API lookup from {api_lookup_path}...")
            self.api_lookup = load_api_lookup(api_lookup_path)
            print(f"  Loaded {len(self.api_lookup)} field entries")

    def get_type_description(self, product: str, type_name: str) -> str:
        """Get description for a type."""
        # Try GraphQL docs first
        if product in self.graphql_docs:
            if type_name in self.graphql_docs[product]:
                desc = self.graphql_docs[product][type_name].get("description", "")
                if desc:
                    return desc

        # Try schema data
        if "products" in self.schema_data:
            product_data = self.schema_data["products"].get(product, {})
            for type_info in product_data.get("types", []):
                if type_info.get("name") == type_name:
                    return type_info.get("description", "")

        return ""

    def get_field_description(self, product: str, type_name: str, field_name: str) -> str:
        """Get description for a field."""
        # Try api_lookup first
        lookup_key = f"{product}_{type_name}.{field_name}"
        if lookup_key in self.api_lookup:
            desc = self.api_lookup[lookup_key].get("description", "")
            if desc:
                return desc

        # Try GraphQL docs
        if product in self.graphql_docs:
            if type_name in self.graphql_docs[product]:
                fields = self.graphql_docs[product][type_name].get("fields", {})
                if field_name in fields:
                    return fields[field_name]

        # Try schema data
        if "products" in self.schema_data:
            product_data = self.schema_data["products"].get(product, {})
            for type_info in product_data.get("types", []):
                if type_info.get("name") == type_name:
                    for field_info in type_info.get("fields", []):
                        if field_info.get("name") == field_name:
                            return field_info.get("description", "")

        return ""

    def parse_api_field(self, field_path: str) -> Tuple[str, str, str]:
        """
        Parse a field path like 'turbonomic_Action.actionID' into components.

        Returns:
            Tuple of (product, type_name, field_name)
        """
        # Handle format: product_Type.field or product.Type.field
        if "_" in field_path:
            # product_Type.field format
            parts = field_path.split("_", 1)
            product = parts[0]
            if len(parts) > 1 and "." in parts[1]:
                type_field = parts[1].split(".", 1)
                type_name = type_field[0]
                field_name = type_field[1] if len(type_field) > 1 else ""
            else:
                type_name = parts[1] if len(parts) > 1 else ""
                field_name = ""
        elif "." in field_path:
            # product.Type.field format
            parts = field_path.split(".")
            product = parts[0]
            type_name = parts[1] if len(parts) > 1 else ""
            field_name = ".".join(parts[2:]) if len(parts) > 2 else ""
        else:
            product = field_path
            type_name = ""
            field_name = ""

        return product, type_name, field_name

    def enrich_sample(self, sample: Dict) -> Dict:
        """
        Enrich a dataset sample with API descriptions.

        Adds:
        - source_type_description
        - target_type_description
        - source_field_descriptions (dict)
        - target_field_descriptions (dict)
        """
        enriched = sample.copy()

        # Get source and target info from mapping
        mapping = sample.get("mapping", {})

        source_type_desc = ""
        target_type_desc = ""
        source_field_descs = {}
        target_field_descs = {}

        for source_field, target_field in mapping.items():
            # Parse source
            src_product, src_type, src_field = self.parse_api_field(source_field)

            # Parse target
            tgt_product, tgt_type, tgt_field = self.parse_api_field(target_field)

            # Get type descriptions (only once per type)
            if src_type and not source_type_desc:
                source_type_desc = self.get_type_description(src_product, src_type)

            if tgt_type and not target_type_desc:
                target_type_desc = self.get_type_description(tgt_product, tgt_type)

            # Get field descriptions
            if src_field:
                desc = self.get_field_description(src_product, src_type, src_field)
                if desc:
                    source_field_descs[src_field] = desc

            if tgt_field:
                desc = self.get_field_description(tgt_product, tgt_type, tgt_field)
                if desc:
                    target_field_descs[tgt_field] = desc

        # Add enrichment to sample
        enriched["enrichment"] = {
            "source_type_description": source_type_desc,
            "target_type_description": target_type_desc,
            "source_field_descriptions": source_field_descs,
            "target_field_descriptions": target_field_descs,
        }

        return enriched

    def enrich_dataset(self, dataset_path: Path, output_path: Optional[Path] = None) -> int:
        """
        Enrich all samples in a dataset file.

        Args:
            dataset_path: Path to input JSONL dataset
            output_path: Path to output enriched dataset (defaults to input with _enriched suffix)

        Returns:
            Number of samples enriched
        """
        if output_path is None:
            output_path = dataset_path.with_suffix(".enriched.jsonl")

        enriched_count = 0
        samples_with_desc = 0

        with open(dataset_path) as f_in, open(output_path, "w") as f_out:
            for line in f_in:
                sample = json.loads(line)
                enriched = self.enrich_sample(sample)

                # Track stats
                enriched_count += 1
                enrichment = enriched.get("enrichment", {})
                if (enrichment.get("source_type_description") or
                    enrichment.get("target_type_description") or
                    enrichment.get("source_field_descriptions") or
                    enrichment.get("target_field_descriptions")):
                    samples_with_desc += 1

                f_out.write(json.dumps(enriched) + "\n")

        print(f"\nEnriched {enriched_count} samples")
        print(f"  Samples with descriptions: {samples_with_desc} ({100*samples_with_desc/enriched_count:.1f}%)")
        print(f"Output saved to: {output_path}")

        return enriched_count


def main():
    """Main entry point."""
    # Initialize enricher with available sources
    enricher = DescriptionEnricher(
        graphql_schema_dir=GRAPHQL_SCHEMA_DIR,
        all_products_schema_path=ALL_PRODUCTS_SCHEMA,
        api_lookup_path=DATA_DIR / "itopsgraph_api_lookup.json"
    )

    # Enrich the ITOpsGraph dataset
    dataset_path = DATA_DIR / "itopsgraph_dataset.jsonl"

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return 1

    enricher.enrich_dataset(dataset_path)

    # Show sample of enriched data
    print("\n" + "="*60)
    print("Sample enriched entry:")
    print("="*60)

    enriched_path = DATA_DIR / "itopsgraph_dataset.enriched.jsonl"
    with open(enriched_path) as f:
        for i, line in enumerate(f):
            if i >= 3:  # Show first 3 samples with descriptions
                break
            sample = json.loads(line)
            enrichment = sample.get("enrichment", {})
            if enrichment.get("source_type_description") or enrichment.get("target_type_description"):
                print(f"\nSample: {sample.get('id')}")
                print(f"  Utterance: {sample.get('utterance')[:80]}...")
                print(f"  Source type desc: {enrichment.get('source_type_description', '')[:100]}...")
                print(f"  Target type desc: {enrichment.get('target_type_description', '')[:100]}...")
                print(f"  Source field descs: {list(enrichment.get('source_field_descriptions', {}).keys())}")
                print(f"  Target field descs: {list(enrichment.get('target_field_descriptions', {}).keys())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
