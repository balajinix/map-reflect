"""
Script to test and create all possible mappings from a given directory of JSON files. 
"""

# import json
# import os
# from toolkit.solve import solve
# from dotenv import load_dotenv
# from toolkit.core.schemas.input_schemas import InputTableSchema


# load_dotenv()

# config = {
#     "credentials": {
#         "apikey": os.getenv("WATSONX_APIKEY"),
#         "project_id": os.getenv("WATSONX_PROJECT_ID"),
#     },
#     "model_name": "meta-llama/llama-3-3-70b-instruct",
#     "max_new_tokens": 4096,
# }

# # Load your source and target schemas
# with open("/home/gunjan/mymapping/HI/concert_vulnerableApplications.json") as f:
#     source_schema = json.load(f)

# #tests/test_data/solis/instana_applications.json
# with open("/home/gunjan/mymapping/HI/instana_applications.json") as f:
#     target_schema = json.load(f)

# # Execute the schema matching pipeline
# result = solve(
#     source_schema_dict=target_schema,
#     target_schema_dict=source_schema,
#     pipeline_name="graphql_pipeline",  # Choose a pipeline from available options
#     # Additional pipeline-specific parameters can be passed as kwargs
#      **config,
# )

# trimmed_result = {"name": result["name"], "properties": {}}
# for x in result["properties"]:
#     if result["properties"][x] != []:
#         trimmed_result["properties"][x] = result["properties"][x]

# print("Generated mappings:")
# print(json.dumps(trimmed_result, indent=4))


import json
import os
from itertools import permutations
from toolkit.solve import solve
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def generate_all_mappings(input_directory, output_directory):
    """
    Generate all permutations of schema mappings between files with specified prefixes.
    
    Args:
        input_directory (str): Path to directory containing schema files
        output_directory (str): Path to directory where mappings will be saved
    """
    
    # Configuration for the mapping tool
    config = {
        "credentials": {
            "apikey": os.getenv("WATSONX_APIKEY"),
            "project_id": os.getenv("WATSONX_PROJECT_ID"),
        },
        "model_name": "meta-llama/llama-3-3-70b-instruct",
        "max_new_tokens": 4096,
    }
    
    # Ensure output directory exists
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files that start with the specified prefixes
    valid_prefixes = ["concert_", "turbo_", "instana_"]
    schema_files = []
    
    for filename in os.listdir(input_directory):
        if filename.endswith('.json') and any(filename.startswith(prefix) for prefix in valid_prefixes):
            schema_files.append(filename)
    
    print(f"Found {len(schema_files)} schema files: {schema_files}")
    
    # Generate all permutations of pairs (source, target)
    file_pairs = list(permutations(schema_files, 2))
    
    # Filter out same-product mappings
    filtered_pairs = []
    for source_file, target_file in file_pairs:
        source_product = source_file.split('_')[0]
        target_product = target_file.split('_')[0]
        
        # Skip if same product
        if source_product == target_product:
            continue
            
        filtered_pairs.append((source_file, target_file))
    
    print(f"Generated {len(filtered_pairs)} valid mapping pairs")
    
    # Process each pair
    successful_mappings = 0
    failed_mappings = 0
    empty_mappings = 0
    
    for source_file, target_file in filtered_pairs:
        try:
            # Load source and target schemas
            source_path = os.path.join(input_directory, source_file)
            target_path = os.path.join(input_directory, target_file)
            
            with open(source_path, 'r') as f:
                source_schema = json.load(f)
            
            with open(target_path, 'r') as f:
                target_schema = json.load(f)
            
            print(f"\nProcessing: {source_file} -> {target_file}")
            
            # Execute the schema matching pipeline
            # Note: Based on your comment, we use target as source and source as target
            result = solve(
                source_schema_dict=target_schema,
                target_schema_dict=source_schema,
                pipeline_name="graphql_pipeline",
                **config,
            )
            
            # Trim the result to only include non-empty properties
            trimmed_result = {"name": result["name"], "properties": {}}
            for prop in result["properties"]:
                if result["properties"][prop] != []:
                    trimmed_result["properties"][prop] = result["properties"][prop]
            
            # Check if properties is empty - if so, discard this mapping
            if not trimmed_result["properties"]:
                print(f"⚠ Skipping empty mapping: {source_file} -> {target_file} (no properties mapped)")
                empty_mappings += 1
                continue
            
            # Generate output filename
            source_name = source_file.replace('.json', '')
            target_name = target_file.replace('.json', '')
            output_filename = f"{source_name}-{target_name}.json"
            output_path = os.path.join(output_directory, output_filename)
            
            # Save the mapping
            with open(output_path, 'w') as f:
                json.dump(trimmed_result, f, indent=4)
            
            print(f"✓ Successfully generated mapping: {output_filename}")
            successful_mappings += 1
            
        except Exception as e:
            print(f"✗ Failed to generate mapping {source_file} -> {target_file}: {str(e)}")
            failed_mappings += 1
    
    print(f"\n=== Summary ===")
    print(f"Total pairs processed: {len(filtered_pairs)}")
    print(f"Successful mappings: {successful_mappings}")
    print(f"Empty mappings (discarded): {empty_mappings}")
    print(f"Failed mappings: {failed_mappings}")
    print(f"Mappings saved to: {output_directory}")

if __name__ == "__main__":
    # Set your paths here
    input_dir = "/home/gunjan/TheToolkit/schema-mapping-toolkit/tests/test_data/solis"   
    output_dir = "/home/gunjan/mymapping/mappings_output"
    
    generate_all_mappings(input_dir, output_dir)