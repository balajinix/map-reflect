#!/usr/bin/env python3
"""
Generate nl2gql_api_lookup.json and nl2gql_hierarchy.json from GraphQL schemas.

Matches the exact format used in nl2gql_dataset.jsonl:
- Property IDs: {table}_{field} (e.g., "concert_stadium_id")
- Type names: PascalCase (e.g., "Concert")
- Skips @materializer relationship fields
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Set


def parse_graphql_schema(file_path: str) -> Dict[str, Any]:
    """
    Parse GraphQL schema and extract scalar fields only.
    
    Skips:
    - Relationship fields with @materializer
    - Query and Mutation types
    - Complex relationship fields
    
    Returns:
        Dict with type information
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    types_info = {}
    
    # Extract type definitions (not Query/Mutation)
    type_pattern = r'type\s+(\w+)\s*\{([^}]+)\}'
    
    for type_match in re.finditer(type_pattern, content, re.MULTILINE):
        type_name = type_match.group(1)
        fields_block = type_match.group(2)
        
        # Skip Query, Mutation, Subscription types
        if type_name in ['Query', 'Mutation', 'Subscription']:
            continue
        
        # Parse scalar fields only
        scalar_fields = []
        
        for line in fields_block.split('\n'):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Skip fields with @materializer (these are relationships)
            if '@materializer' in line:
                continue
            
            # Parse field definition
            # Pattern: field_name: Type or field_name: Type!
            field_match = re.match(r'(\w+)\s*:\s*(\[)?(\w+)([!])?(\])?([!])?', line)
            
            if field_match:
                field_name = field_match.group(1)
                is_list = field_match.group(2) == '['
                field_type = field_match.group(3)
                is_required = field_match.group(4) == '!' or field_match.group(6) == '!'
                
                # Only include scalar types (database columns)
                scalar_types = {'ID', 'Int', 'String', 'Float', 'Boolean', 'DateTime'}
                
                # Skip relationship fields (references to other types)
                if field_type in scalar_types and not is_list:
                    scalar_fields.append({
                        'name': field_name,
                        'type': field_type,
                        'is_required': is_required
                    })
        
        if scalar_fields:
            types_info[type_name] = {
                'name': type_name,
                'fields': scalar_fields
            }
    
    return types_info


def generate_description(table_name: str, field_name: str, field_type: str) -> str:
    """Generate description for a field based on naming patterns."""
    # Convert to readable format
    readable_field = field_name.replace('_', ' ').title()
    
    # Common patterns
    patterns = {
        'id': f'Unique identifier',
        'name': f'Name',
        'title': f'Title',
        'description': f'Description',
        'date': f'Date value',
        'time': f'Time value',
        'year': f'Year value',
        'created': f'Creation timestamp',
        'updated': f'Update timestamp',
        'count': f'Count',
        'num': f'Number',
        'amount': f'Amount',
        'price': f'Price',
        'status': f'Status',
        'type': f'Type',
        'homepage': f'Homepage URL',
        'location': f'Location',
        'capacity': f'Capacity',
        'age': f'Age',
        'country': f'Country',
    }
    
    field_lower = field_name.lower()
    for pattern, desc in patterns.items():
        if pattern in field_lower:
            return f'{desc} for {table_name}'
    
    # Default description
    return f'{readable_field} ({field_type}) for {table_name}'


def generate_api_lookup(all_types: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Generate API lookup in the format matching nl2gql_dataset.jsonl.
    
    Property ID format: {table_lowercase}_{field}
    Example: "concert_stadium_id", "singer_name"
    """
    api_lookup = {}
    
    for type_name, type_info in sorted(all_types.items()):
        # Table name is lowercase version of type name
        table_name = type_name.lower()
        
        for field in type_info['fields']:
            field_name = field['name']
            
            # Property ID: {table}_{field}
            property_id = f"{table_name}_{field_name}"
            
            api_lookup[property_id] = {
                'product_name': 'nl2gql',
                'type_name': type_name,  # Keep PascalCase
                'field_name': field_name,
                'description': generate_description(type_name, field_name, field['type']),
                'data_type': field['type'],
                'is_required': field['is_required']
            }
    
    return api_lookup


def generate_hierarchy(all_types: Dict[str, Dict]) -> Dict[str, Dict[str, List]]:
    """
    Generate hierarchy in the format matching ITOps hierarchy.json.
    
    Format:
    {
      "nl2gql": {
        "Concert": [
          {
            "name": "stadium_id",
            "description": "...",
            "type": "Int",
            "schema_component": "Concert"
          }
        ]
      }
    }
    """
    hierarchy = {'nl2gql': {}}
    
    for type_name, type_info in sorted(all_types.items()):
        fields_list = []
        
        for field in type_info['fields']:
            fields_list.append({
                'name': field['name'],
                'description': generate_description(type_name, field['name'], field['type']),
                'type': field['type'],
                'schema_component': type_name
            })
        
        hierarchy['nl2gql'][type_name] = fields_list
    
    return hierarchy


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate nl2gql files from GraphQL schemas (matches nl2gql_dataset.jsonl format)'
    )
    parser.add_argument('--schema-dir', type=str, required=True,
                       help='Directory containing .graphql schema files')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory (default: current directory)')
    
    args = parser.parse_args()
    
    schema_dir = Path(args.schema_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not schema_dir.exists():
        print(f"❌ Error: Schema directory not found: {schema_dir}")
        return 1
    
    print("="*60)
    print("NL2GQL Schema Processing")
    print("="*60)
    print(f"\nSchema directory: {schema_dir}")
    
    # Process all .graphql files
    all_types = {}
    schema_files = sorted(schema_dir.glob("*.graphql"))
    
    if not schema_files:
        print(f"❌ Error: No .graphql files found in {schema_dir}")
        return 1
    
    print(f"Found {len(schema_files)} schema files")
    print()
    
    for schema_file in schema_files:
        print(f"  Processing: {schema_file.name}")
        
        try:
            types = parse_graphql_schema(str(schema_file))
            
            # Merge types
            for type_name, type_info in types.items():
                if type_name in all_types:
                    # Merge fields if type already exists
                    existing_fields = {f['name'] for f in all_types[type_name]['fields']}
                    for field in type_info['fields']:
                        if field['name'] not in existing_fields:
                            all_types[type_name]['fields'].append(field)
                else:
                    all_types[type_name] = type_info
        
        except Exception as e:
            print(f"    ⚠ Warning: Failed to parse {schema_file.name}: {e}")
            continue
    
    print(f"\n✓ Parsed {len(all_types)} types")
    
    # Generate API lookup
    print("\nGenerating API lookup...")
    api_lookup = generate_api_lookup(all_types)
    
    lookup_file = output_dir / 'nl2gql_api_lookup.json'
    with open(lookup_file, 'w') as f:
        json.dump(api_lookup, f, indent=4, sort_keys=True)
    
    print(f"✓ Generated {len(api_lookup)} properties")
    print(f"✓ Saved to: {lookup_file}")
    
    # Generate hierarchy
    print("\nGenerating hierarchy...")
    hierarchy = generate_hierarchy(all_types)
    
    hierarchy_file = output_dir / 'nl2gql_hierarchy.json'
    with open(hierarchy_file, 'w') as f:
        json.dump(hierarchy, f, indent=4)
    
    print(f"✓ Generated hierarchy for {len(all_types)} types")
    print(f"✓ Saved to: {hierarchy_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60)
    
    print(f"\nStatistics:")
    print(f"  Schema files processed: {len(schema_files)}")
    print(f"  Types extracted: {len(all_types)}")
    print(f"  Properties generated: {len(api_lookup)}")
    
    # Show sample entries
    print(f"\nSample API lookup entries:")
    for i, (prop_id, prop_data) in enumerate(list(api_lookup.items())[:5]):
        print(f"\n  {prop_id}:")
        print(f"    Type: {prop_data['type_name']}")
        print(f"    Field: {prop_data['field_name']}")
        print(f"    Description: {prop_data['description']}")
        if i >= 2:  # Show only first 3
            break
    
    # Show sample hierarchy
    print(f"\nSample hierarchy structure:")
    for type_name, fields in list(hierarchy['nl2gql'].items())[:3]:
        print(f"\n  {type_name}: {len(fields)} fields")
        for field in fields[:2]:
            print(f"    - {field['name']}: {field['type']}")
    
    print(f"\nGenerated files:")
    print(f"  📄 {lookup_file}")
    print(f"  📄 {hierarchy_file}")
    
    print(f"\nFormat matches nl2gql_dataset.jsonl:")
    print(f"  ✓ Property IDs: {{table}}_{{field}} (e.g., 'concert_stadium_id')")
    print(f"  ✓ Type names: PascalCase (e.g., 'Concert')")
    print(f"  ✓ Scalar fields only (no relationships)")
    
    print(f"\nNext steps:")
    print(f"  1. Review generated files")
    print(f"  2. Verify format matches your convert_nl2gql.py expectations")
    print(f"  3. Run MapReflect: ./run_map_reflect.sh --datasets nl2gql")
    
    return 0


if __name__ == '__main__':
    exit(main())
