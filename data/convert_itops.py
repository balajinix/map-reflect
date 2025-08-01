#!/usr/bin/env python3
"""
Convert ITOps raw dataset to proper SMT toolkit input format.
Creates full InputTableSchema structures that the SMT toolkit expects.
"""
import json

input_file = "itops_dataset_raw.jsonl"
output_file = "itops_dataset.jsonl"

def build_schema(schema_name, property_names, schema_type="source"):
    """
    Build a proper InputTableSchema structure.
    
    Args:
        schema_name: Name of the schema
        property_names: List of property names for this schema
        schema_type: "source" or "target" for logging
    
    Returns:
        Dictionary in InputTableSchema format
    """
    properties = []
    
    for prop_name in property_names:
        properties.append({
            "name": prop_name,
            "description": f"Field {prop_name} of {schema_name}",
            "metadata": {}
        })
    
    return {
        "id": schema_name,
        "name": schema_name,
        "description": f"Schema for {schema_name}",
        "metadata": {},
        "properties": properties
    }

def extract_schemas(result_name, properties):
    """Extract source and target schema names."""
    # Source schema is the prefix before "__" if present
    source_schema = result_name.split("__")[0] if "__" in result_name else result_name

    # Target schema: pick the first mapTable from properties
    target_schema = ""
    for mappings in properties.values():
        if mappings and isinstance(mappings, list):
            target_schema = mappings[0].get("mapTable", "")
            break

    return source_schema, target_schema

def convert_line(data, idx):
    """Convert a single line to SMT toolkit format."""
    utterance = data.get("utterance", "")
    result = data.get("result", {})
    result_name = result.get("name", "")
    properties = result.get("properties", {})

    # Extract source and target schema names
    source_schema_name, target_schema_name = extract_schemas(result_name, properties)
    
    if not source_schema_name or not target_schema_name:
        print(f"Skipping line {idx}: missing source or target schema")
        return None

    # Build source properties list (all source fields)
    source_properties = []
    flat_mapping = {}
    
    for src_field, mappings in properties.items():
        if mappings and isinstance(mappings, list):
            # Add to source properties
            source_properties.append(src_field)
            
            # Add to flat mapping (source → target)
            target_property = mappings[0].get("mapProperty", "")
            if target_property:
                flat_mapping[src_field] = target_property

    # Build target properties list (all unique target fields)
    target_properties = list(set(flat_mapping.values()))
    
    if not source_properties or not target_properties:
        print(f"Skipping line {idx}: no valid properties found")
        return None

    # Build full schema structures
    source_schema = build_schema(source_schema_name, source_properties, "source")
    target_schema = build_schema(target_schema_name, target_properties, "target")

    # Construct final output in SMT toolkit format
    converted = {
        "id": data.get("id", f"itops_{idx+1:03d}"),
        "utterance": utterance,
        "source_schema": source_schema,
        "target_schema": target_schema,
        "mapping": flat_mapping,
        "metadata": data.get("metadata", {})
    }

    return converted

def main():
    """Main conversion function."""
    print(f"Converting {input_file} to SMT toolkit format...")
    
    converted_count = 0
    skipped_count = 0
    
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for idx, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                converted = convert_line(data, idx)
                
                if converted:
                    outfile.write(json.dumps(converted) + "\n")
                    converted_count += 1
                else:
                    skipped_count += 1
                    
                # Progress update
                if idx % 100 == 0:
                    print(f"Processed {idx} lines...")
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error on line {idx}: {e}")
                skipped_count += 1
            except Exception as e:
                print(f"Error processing line {idx}: {e}")
                skipped_count += 1

    print(f"\n✅ Conversion completed!")
    print(f"   Converted: {converted_count} lines")
    print(f"   Skipped: {skipped_count} lines")
    print(f"   Output file: {output_file}")
    
    # Validate a sample output
    if converted_count > 0:
        print(f"\n🔍 Sample validation:")
        with open(output_file, "r") as f:
            first_line = json.loads(f.readline())
            
            print(f"   Sample ID: {first_line['id']}")
            print(f"   Source schema: {first_line['source_schema']['name']} ({len(first_line['source_schema']['properties'])} properties)")
            print(f"   Target schema: {first_line['target_schema']['name']} ({len(first_line['target_schema']['properties'])} properties)")
            print(f"   Mappings: {len(first_line['mapping'])}")
            
            # Check structure
            required_fields = ["id", "name", "description", "metadata", "properties"]
            source_schema = first_line['source_schema']
            target_schema = first_line['target_schema']
            
            source_valid = all(field in source_schema for field in required_fields)
            target_valid = all(field in target_schema for field in required_fields)
            
            print(f"   Source schema structure: {'✅ Valid' if source_valid else '❌ Invalid'}")
            print(f"   Target schema structure: {'✅ Valid' if target_valid else '❌ Invalid'}")
            
            # Check properties structure
            if source_schema['properties']:
                first_prop = source_schema['properties'][0]
                prop_valid = all(field in first_prop for field in ["name", "description", "metadata"])
                print(f"   Property structure: {'✅ Valid' if prop_valid else '❌ Invalid'}")

if __name__ == "__main__":
    main()
