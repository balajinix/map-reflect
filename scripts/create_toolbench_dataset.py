#!/usr/bin/env python3
"""
Create a schema mapping dataset from ToolBench REST APIs.

This script creates mapping pairs between APIs from different providers
in the same domain (e.g., Weather, Finance, Travel).

The task: Given a source API's parameters, find the corresponding
parameters in a target API from a different provider.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
import re

# Paths
TOOLBENCH_DIR = Path("/Users/balaji/code/api_mapping/toolbench")
OUTPUT_DIR = Path("/Users/balaji/code/graphql-schema-mapping/data")


def load_all_apis():
    """Load all APIs from ToolBench instruction files."""
    all_apis = []

    for filename in ['G1_instruction.json', 'G2_instruction.json', 'G3_instruction.json']:
        filepath = TOOLBENCH_DIR / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            for sample in data:
                for api in sample.get('api_list', []):
                    api['source_file'] = filename
                    all_apis.append(api)

    # Deduplicate by (tool_name, api_name)
    unique_apis = {}
    for api in all_apis:
        key = (api.get('tool_name', ''), api.get('api_name', ''))
        if key not in unique_apis:
            unique_apis[key] = api

    return list(unique_apis.values())


def extract_api_schema(api):
    """Extract a clean schema representation from an API."""
    params = []

    for p in api.get('required_parameters', []):
        params.append({
            'name': p.get('name', ''),
            'type': p.get('type', 'STRING'),
            'description': p.get('description', '')[:200] if p.get('description') else '',
            'required': True,
            'default': p.get('default', '')
        })

    for p in api.get('optional_parameters', []):
        params.append({
            'name': p.get('name', ''),
            'type': p.get('type', 'STRING'),
            'description': p.get('description', '')[:200] if p.get('description') else '',
            'required': False,
            'default': p.get('default', '')
        })

    return {
        'tool_name': api.get('tool_name', ''),
        'api_name': api.get('api_name', ''),
        'api_description': api.get('api_description', '')[:300] if api.get('api_description') else '',
        'category': api.get('category_name', ''),
        'method': api.get('method', 'GET'),
        'parameters': params
    }


def normalize_param_name(name):
    """Normalize parameter name for comparison."""
    # Convert to lowercase, remove underscores/hyphens
    name = name.lower().replace('_', '').replace('-', '')
    return name


def find_param_mappings(source_params, target_params):
    """
    Find potential mappings between source and target parameters.
    Returns a dict of source_param -> target_param mappings.
    """
    mappings = {}

    # Known semantic equivalences
    equivalences = {
        # Location
        'lat': ['latitude', 'lat', 'startlat', 'endlat'],
        'lon': ['longitude', 'lon', 'lng', 'long', 'startlon', 'endlon', 'startlng', 'endlng'],
        'location': ['location', 'place', 'address', 'loc'],
        'city': ['city', 'cityname', 'town', 'q', 'location'],
        'country': ['country', 'countrycode', 'countryname', 'cc'],
        'region': ['region', 'state', 'province', 'area'],
        'postcode': ['postcode', 'zipcode', 'zip', 'postalcode'],

        # Time
        'startdate': ['startdate', 'start', 'from', 'fromdate', 'begindate', 'starttime'],
        'enddate': ['enddate', 'end', 'to', 'todate', 'endtime', 'until'],
        'date': ['date', 'dt', 'datetime', 'timestamp', 'time'],

        # Identifiers
        'id': ['id', 'identifier', 'uuid', 'key'],
        'symbol': ['symbol', 'ticker', 'stock', 'q'],
        'name': ['name', 'title', 'label', 'displayname'],

        # Pagination
        'page': ['page', 'pagenum', 'pagenumber', 'p'],
        'limit': ['limit', 'count', 'size', 'perpage', 'pagesize', 'num', 'max'],
        'offset': ['offset', 'skip', 'start'],

        # Units
        'units': ['units', 'unit', 'tempunit', 'measureunit'],
        'currency': ['currency', 'curr', 'currencycode'],
        'language': ['language', 'lang', 'locale', 'hl'],

        # Format
        'format': ['format', 'output', 'type', 'responsetype'],
    }

    # Build reverse lookup
    param_to_canonical = {}
    for canonical, variants in equivalences.items():
        for v in variants:
            param_to_canonical[v] = canonical

    # Normalize source and target params
    source_normalized = {normalize_param_name(p['name']): p['name'] for p in source_params}
    target_normalized = {normalize_param_name(p['name']): p['name'] for p in target_params}

    # Find mappings
    for src_norm, src_orig in source_normalized.items():
        # Direct match
        if src_norm in target_normalized:
            mappings[src_orig] = target_normalized[src_norm]
            continue

        # Canonical form match
        src_canonical = param_to_canonical.get(src_norm)
        if src_canonical:
            for tgt_norm, tgt_orig in target_normalized.items():
                tgt_canonical = param_to_canonical.get(tgt_norm)
                if tgt_canonical == src_canonical:
                    mappings[src_orig] = tgt_orig
                    break

    return mappings


def create_mapping_samples(apis, category):
    """Create mapping samples for APIs in a category."""
    samples = []

    # Group by tool (provider)
    by_tool = defaultdict(list)
    for api in apis:
        if api.get('category_name') == category:
            tool = api.get('tool_name', '')
            by_tool[tool].append(api)

    tools = list(by_tool.keys())

    # Create pairs between different providers
    for i, tool1 in enumerate(tools):
        for tool2 in tools[i+1:]:
            for api1 in by_tool[tool1]:
                for api2 in by_tool[tool2]:
                    # Extract schemas
                    schema1 = extract_api_schema(api1)
                    schema2 = extract_api_schema(api2)

                    # Skip if either has no parameters
                    if not schema1['parameters'] or not schema2['parameters']:
                        continue

                    # Find potential mappings
                    mappings = find_param_mappings(schema1['parameters'], schema2['parameters'])

                    # Only include if there's at least one mapping
                    if mappings:
                        sample = {
                            'category': category,
                            'source_api': schema1,
                            'target_api': schema2,
                            'ground_truth_mappings': mappings,
                            'mapping_count': len(mappings)
                        }
                        samples.append(sample)

    return samples


def main():
    print("Loading ToolBench APIs...")
    apis = load_all_apis()
    print(f"Loaded {len(apis)} unique APIs")

    # Categories with good variety
    target_categories = [
        'Weather', 'Finance', 'Financial', 'Travel', 'Location',
        'Sports', 'Music', 'eCommerce', 'Social', 'News_Media',
        'Food', 'Health_and_Fitness', 'Entertainment', 'Media'
    ]

    all_samples = []

    for category in target_categories:
        cat_apis = [a for a in apis if a.get('category_name') == category]
        if len(cat_apis) < 2:
            continue

        samples = create_mapping_samples(apis, category)
        print(f"  {category}: {len(samples)} mapping samples")
        all_samples.extend(samples)

    print(f"\nTotal samples: {len(all_samples)}")

    # Save dataset
    output_file = OUTPUT_DIR / "toolbench_schema_mapping.json"
    with open(output_file, 'w') as f:
        json.dump({
            'description': 'REST API schema mapping dataset from ToolBench',
            'task': 'Given a source API schema, find corresponding parameters in target API schema',
            'total_samples': len(all_samples),
            'categories': list(set(s['category'] for s in all_samples)),
            'samples': all_samples
        }, f, indent=2)

    print(f"Saved to {output_file}")

    # Also save as JSONL for compatibility
    jsonl_file = OUTPUT_DIR / "toolbench_dataset.jsonl"
    with open(jsonl_file, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Saved JSONL to {jsonl_file}")

    # Print sample
    print("\n" + "=" * 80)
    print("SAMPLE MAPPING:")
    print("=" * 80)
    if all_samples:
        s = all_samples[0]
        print(f"Category: {s['category']}")
        print(f"Source: {s['source_api']['tool_name']} / {s['source_api']['api_name']}")
        print(f"  Params: {[p['name'] for p in s['source_api']['parameters']]}")
        print(f"Target: {s['target_api']['tool_name']} / {s['target_api']['api_name']}")
        print(f"  Params: {[p['name'] for p in s['target_api']['parameters']]}")
        print(f"Mappings: {s['ground_truth_mappings']}")


if __name__ == "__main__":
    main()
