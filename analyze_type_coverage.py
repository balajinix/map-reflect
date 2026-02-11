#!/usr/bin/env python3
"""
Analyze type mapping lookup coverage for ITOpsGraph.

Compares all types in ITOpsGraph (from graphql files) against the types
that have mappings in itops_type_mapping_lookup.json.
"""

import json
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Official 12 products in ITOpsGraph
# Maps file prefix to canonical product name
PRODUCT_MAPPING = {
    'aws': 'aws',
    'azure': 'azure',
    'cloudability': 'cloudability',
    'concert': 'concert',
    'gcp': 'gcp',
    'packer': 'hcp_packer',           # packer_* files -> hcp_packer
    'hcp_terraform': 'hcp_terraform',  # hcp_terraform_* files -> hcp_terraform
    'infragraph': 'infragraph',
    'base': 'infragraph',              # base_* files are part of infragraph
    'edge': 'infragraph',              # edge_* files are part of infragraph
    'model': 'infragraph',             # model_* files are part of infragraph
    'resource': 'infragraph',          # resource_* files are part of infragraph
    'hcp': 'infragraph',               # hcp_* (non-terraform) are shared types in infragraph
    'instana': 'instana',
    'ns1': 'ns1',
    'turbonomic': 'turbonomic',
    'watsonx-orchestrate': 'watsonx-orchestrate',
}

# Expected type counts per product (from documentation)
EXPECTED_TYPES = {
    'aws': 21,
    'azure': 2,
    'cloudability': 40,
    'concert': 274,
    'gcp': 1,
    'hcp_packer': 9,
    'hcp_terraform': 6,
    'infragraph': 69,
    'instana': 1128,
    'ns1': 13,
    'turbonomic': 59,
    'watsonx-orchestrate': 57,
}

# Product descriptions
PRODUCT_DESCRIPTIONS = {
    'aws': 'Amazon Web Services infrastructure resources',
    'azure': 'Microsoft Azure cloud infrastructure resources',
    'cloudability': 'IBM Cloudability - Cloud cost management and FinOps platform',
    'concert': 'IBM Concert - Application lifecycle management and vulnerability tracking',
    'gcp': 'Google Cloud Platform infrastructure resources',
    'hcp_packer': 'HashiCorp Packer - Machine image automation',
    'hcp_terraform': 'HashiCorp Terraform Cloud - Infrastructure as code management',
    'infragraph': 'Infragraph base types for cross-cloud infrastructure modeling',
    'instana': 'IBM Instana - Enterprise observability platform',
    'ns1': 'IBM NS1 Connect - Intelligent DNS and traffic management',
    'turbonomic': 'IBM Turbonomic - Application resource management and optimization',
    'watsonx-orchestrate': 'IBM watsonx Orchestrate - AI-powered automation and skill orchestration',
}

# Products with underscores that need special prefix matching
PRODUCTS_WITH_UNDERSCORES = {'hcp_terraform', 'watsonx-orchestrate'}


def extract_product_from_type(type_name: str) -> str:
    """Extract product name from type like 'concert_Application' -> 'concert'."""
    # Check for products with underscores first (longer prefixes)
    for known_prod in sorted(PRODUCTS_WITH_UNDERSCORES, key=len, reverse=True):
        if type_name.startswith(known_prod + '_'):
            return PRODUCT_MAPPING.get(known_prod, known_prod)

    if '_' in type_name:
        prefix = type_name.split('_', 1)[0]
        return PRODUCT_MAPPING.get(prefix, prefix)
    return 'unknown'


def get_all_itopsgraph_types() -> dict:
    """Get all types from ITOpsGraph graphql files, organized by product."""
    graphql_dir = BASE_DIR / 'data' / 'itopsgraph_graphql'
    types_by_product = defaultdict(set)

    for f in graphql_dir.glob('*.graphql'):
        type_name = f.stem  # Remove .graphql

        # Skip duplicate files: hcp_HcpTerraform* duplicates hcp_terraform_HcpTerraform*
        if type_name.startswith('hcp_HcpTerraform'):
            continue

        product = extract_product_from_type(type_name)
        types_by_product[product].add(type_name)

    return dict(types_by_product)


def get_types_with_mappings(lookup_file: Path) -> set:
    """Get all types that have at least one mapping in the lookup file."""
    with open(lookup_file, 'r') as f:
        lookup = json.load(f)
    return set(lookup.keys())


def normalize_type_name(type_name: str) -> str:
    """Normalize type name for comparison."""
    return type_name.lower().replace('-', '_')


def find_matching_type(type_name: str, all_types: set) -> str | None:
    """Find a matching type in all_types, handling case and normalization."""
    if type_name in all_types:
        return type_name

    # Try normalized matching
    norm_name = normalize_type_name(type_name)
    for t in all_types:
        if normalize_type_name(t) == norm_name:
            return t

    return None


def main():
    print("=" * 70)
    print("ITOpsGraph Type Mapping Lookup Coverage Analysis")
    print("=" * 70)
    print()

    # Get all ITOpsGraph types
    types_by_product = get_all_itopsgraph_types()
    all_types = set()
    for types in types_by_product.values():
        all_types.update(types)

    print(f"Total types in ITOpsGraph API: {len(all_types)}")
    print(f"Total products: {len(types_by_product)}")
    print()

    # Get types with mappings
    lookup_file = BASE_DIR / 'data' / 'itops_type_mapping_lookup.json'
    if not lookup_file.exists():
        print(f"ERROR: Lookup file not found: {lookup_file}")
        return

    with open(lookup_file, 'r') as f:
        lookup = json.load(f)

    mapped_types = set(lookup.keys())
    print(f"Total keys in lookup file: {len(mapped_types)}")
    print()

    # Find which ITOpsGraph types have mappings
    covered_by_product = defaultdict(set)
    uncovered_by_product = defaultdict(set)

    for product, types in types_by_product.items():
        for type_name in types:
            # Check if this type exists as a key in the lookup
            if type_name in mapped_types:
                covered_by_product[product].add(type_name)
            else:
                # Try normalized matching
                matched = find_matching_type(type_name, mapped_types)
                if matched:
                    covered_by_product[product].add(type_name)
                else:
                    uncovered_by_product[product].add(type_name)

    # Print per-product coverage
    print("=" * 70)
    print("Coverage by Product")
    print("=" * 70)
    print()
    print(f"{'Product':<25} {'Types':<10} {'Covered':<10} {'Coverage':<10}")
    print("-" * 55)

    total_types = 0
    total_covered = 0
    product_stats = []

    for product in sorted(types_by_product.keys(), key=lambda p: len(types_by_product[p]), reverse=True):
        types_count = len(types_by_product[product])
        covered_count = len(covered_by_product.get(product, set()))
        coverage_pct = (covered_count / types_count * 100) if types_count > 0 else 0

        total_types += types_count
        total_covered += covered_count

        product_stats.append({
            'product': product,
            'types': types_count,
            'covered': covered_count,
            'coverage': coverage_pct
        })

        print(f"{product:<25} {types_count:<10} {covered_count:<10} {coverage_pct:>6.1f}%")

    print("-" * 55)
    total_coverage = (total_covered / total_types * 100) if total_types > 0 else 0
    print(f"{'TOTAL':<25} {total_types:<10} {total_covered:<10} {total_coverage:>6.1f}%")
    print()

    # Coverage summary
    print("=" * 70)
    print("Coverage Summary")
    print("=" * 70)
    print()
    print(f"{'Metric':<45} {'Count':>10}")
    print("-" * 55)
    print(f"{'Total types in ITOpsGraph API':<45} {total_types:>10}")
    print(f"{'Types with at least one mapping':<45} {total_covered:>10}")
    print(f"{'Types WITHOUT any mapping':<45} {total_types - total_covered:>10}")
    print(f"{'Current coverage':<45} {total_coverage:>9.1f}%")
    print()

    # Products with no coverage
    zero_coverage_products = [p for p in product_stats if p['coverage'] == 0]
    if zero_coverage_products:
        print("=" * 70)
        print("Products with NO coverage (0%)")
        print("=" * 70)
        for p in zero_coverage_products:
            print(f"  - {p['product']} ({p['types']} types)")
        print()

    # Products with partial coverage
    partial_coverage_products = [p for p in product_stats if 0 < p['coverage'] < 100]
    if partial_coverage_products:
        print("=" * 70)
        print("Products with Partial Coverage")
        print("=" * 70)
        for p in sorted(partial_coverage_products, key=lambda x: x['coverage']):
            print(f"  - {p['product']}: {p['coverage']:.1f}% ({p['covered']}/{p['types']})")
        print()

    # Fully covered products
    full_coverage_products = [p for p in product_stats if p['coverage'] == 100]
    if full_coverage_products:
        print("=" * 70)
        print("Products with Full Coverage (100%)")
        print("=" * 70)
        for p in full_coverage_products:
            print(f"  - {p['product']} ({p['types']} types)")
        print()

    # Key insights
    print("=" * 70)
    print("Key Insight")
    print("=" * 70)
    print()
    print(f"The goal should be to have mappings for all {total_types} types,")
    print(f"not just the {total_covered} currently covered.")
    print()
    print("Priority products for coverage (by type count):")
    for p in sorted(partial_coverage_products + zero_coverage_products,
                    key=lambda x: x['types'], reverse=True)[:5]:
        missing = p['types'] - p['covered']
        print(f"  - {p['product']}: {missing} types missing ({p['coverage']:.1f}% covered)")
    print()

    # Summary table
    print("=" * 70)
    print("Summary Table")
    print("=" * 70)
    print()
    print(f"{'Question':<50} {'Answer':>15}")
    print("-" * 65)
    print(f"{'How many keys SHOULD exist?':<50} {total_types:>15}")
    print(f"{'How many keys currently exist?':<50} {len(mapped_types):>15}")
    print(f"{'Types WITH at least one mapping':<50} {total_covered:>15}")
    print(f"{'Types WITHOUT any mapping':<50} {total_types - total_covered:>15}")
    print(f"{'Current coverage':<50} {f'{total_coverage:.1f}%':>15}")
    print()

    # Save detailed results to JSON
    output_file = BASE_DIR / 'data' / 'type_coverage_analysis.json'
    results = {
        'summary': {
            'total_itopsgraph_types': total_types,
            'total_lookup_keys': len(mapped_types),
            'types_with_mappings': total_covered,
            'types_without_mappings': total_types - total_covered,
            'coverage_percentage': round(total_coverage, 2)
        },
        'by_product': {
            p['product']: {
                'types': p['types'],
                'covered': p['covered'],
                'coverage_percentage': round(p['coverage'], 2)
            }
            for p in product_stats
        },
        'uncovered_types_by_product': {
            product: sorted(list(types))
            for product, types in uncovered_by_product.items()
            if types
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
