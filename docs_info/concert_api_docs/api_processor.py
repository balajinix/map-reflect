""" Test code to create a API lookup using Concert JSON files """

import json
import os
import sys
from typing import Dict, List
from pathlib import Path

# Import the RAG module
from create_descriptions import ConcertAPIDescriptionGenerator

class APILookupProcessor:
    def __init__(self, rag_generator: ConcertAPIDescriptionGenerator = None,
                 output_file: str = "api_lookup.json"):
        """
        Initialize the API Lookup Processor.
        
        Args:
            rag_generator: Instance of ConcertAPIDescriptionGenerator
            output_file: Path to output JSON file
        """
        self.rag_generator = rag_generator or ConcertAPIDescriptionGenerator()
        self.output_file = output_file
        self.api_lookup = {}
        
        # Load existing lookup file if it exists
        self.load_existing_lookup()
    
    def load_existing_lookup(self):
        """Load existing api_lookup.json if it exists."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    self.api_lookup = json.load(f)
                print(f"Loaded existing lookup file with {len(self.api_lookup)} entries")
            except Exception as e:
                print(f"Warning: Could not load existing lookup file: {e}")
                self.api_lookup = {}
        else:
            print("No existing lookup file found. Starting fresh.")
    
    def save_lookup(self):
        """Save the current api_lookup to file."""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.api_lookup, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved {len(self.api_lookup)} entries to {self.output_file}")
        except Exception as e:
            print(f"Error saving lookup file: {e}")
    
    def parse_fullname(self, fullname: str) -> Dict[str, str]:
        """
        Parse the fullname to extract components.
        
        Args:
            fullname: String in format "productname_typename.field1.field2..."
            
        Returns:
            Dict with parsed components
        """
        parts = fullname.split('.')
        
        if not parts:
            raise ValueError(f"Invalid fullname format: {fullname}")
        
        # Split the first part to get product name and type name
        first_part = parts[0]
        if '_' in first_part:
            product_name, type_name = first_part.rsplit('_', 1)
        else:
            product_name = first_part
            type_name = ""
        
        # Get field names (everything after the first part)
        field_names = parts[1:] if len(parts) > 1 else []
        field_name = '.'.join(field_names) if field_names else ""
        
        return {
            'product_name': product_name,
            'type_name': type_name,
            'field_name': field_name,
            'fullname': fullname
        }
    
    def process_json_file(self, json_file_path: str, skip_existing: bool = True) -> Dict[str, any]:
        """
        Process a JSON file and extract API descriptions.
        
        Args:
            json_file_path: Path to the JSON file to process
            skip_existing: Whether to skip entries that already exist in lookup
            
        Returns:
            Dictionary with processing results
        """
        if not os.path.exists(json_file_path):
            return {
                'success': False,
                'error': f"File not found: {json_file_path}"
            }
        
        try:
            # Read the JSON file
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract properties
            properties = data.get('properties', [])
            if not properties:
                return {
                    'success': False,
                    'error': "No 'properties' found in JSON file"
                }
            
            processed_count = 0
            skipped_count = 0
            error_count = 0
            errors = []
            
            print(f"Processing {len(properties)} properties from {json_file_path}")
            
            for i, prop in enumerate(properties, 1):
                fullname = prop.get('name', '')
                if not fullname:
                    print(f"Skipping property {i}: no 'name' field")
                    continue
                
                # Check if already exists
                if skip_existing and fullname in self.api_lookup:
                    print(f"Skipping {fullname} (already exists)")
                    skipped_count += 1
                    continue
                
                print(f"Processing ({i}/{len(properties)}): {fullname}")
                
                try:
                    # Parse the fullname
                    parsed = self.parse_fullname(fullname)
                    
                    # Get original description from the JSON
                    original_desc = prop.get('description', '').strip()
                    print(original_desc)
                    # Get enhanced description using original description
                    description = self.rag_generator.get_description_only(fullname, original_desc=original_desc)
                    
                    # Check if description generation was successful
                    if description.startswith("Error generating description:"):
                        print(f"Warning: RAG failed for {fullname}: {description}")
                        error_count += 1
                        errors.append({
                            'fullname': fullname,
                            'error': description
                        })
                        continue
                    
                    # Create the lookup entry
                    lookup_entry = {
                        'product_name': parsed['product_name'].title(),  # Capitalize product name
                        'type_name': parsed['type_name'],
                        'field_name': parsed['field_name'],
                        'description': description
                    }
                    
                    # Add to lookup
                    self.api_lookup[fullname] = lookup_entry
                    processed_count += 1
                    
                    print(f"✅ Successfully processed: {fullname}")
                    
                    # Save periodically (every 10 entries)
                    if processed_count % 10 == 0:
                        self.save_lookup()
                        print(f"Intermediate save: {processed_count} entries processed")
                
                except Exception as e:
                    error_msg = f"Error processing {fullname}: {str(e)}"
                    print(f"❌ {error_msg}")
                    error_count += 1
                    errors.append({
                        'fullname': fullname,
                        'error': error_msg
                    })
                    continue
            
            # Final save
            self.save_lookup()
            
            return {
                'success': True,
                'processed_count': processed_count,
                'skipped_count': skipped_count,
                'error_count': error_count,
                'errors': errors,
                'total_entries': len(self.api_lookup)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error reading JSON file: {str(e)}"
            }
    
    def process_multiple_files(self, json_files: List[str], skip_existing: bool = True) -> Dict[str, any]:
        """
        Process multiple JSON files.
        
        Args:
            json_files: List of JSON file paths
            skip_existing: Whether to skip entries that already exist in lookup
            
        Returns:
            Dictionary with overall processing results
        """
        overall_results = {
            'files_processed': 0,
            'total_processed': 0,
            'total_skipped': 0,
            'total_errors': 0,
            'file_results': {},
            'all_errors': []
        }
        
        for json_file in json_files:
            print(f"\n{'='*60}")
            print(f"Processing file: {json_file}")
            print(f"{'='*60}")
            
            result = self.process_json_file(json_file, skip_existing)
            
            if result['success']:
                overall_results['files_processed'] += 1
                overall_results['total_processed'] += result['processed_count']
                overall_results['total_skipped'] += result['skipped_count']
                overall_results['total_errors'] += result['error_count']
                overall_results['all_errors'].extend(result['errors'])
                
                print(f"✅ File processed successfully:")
                print(f"   - Processed: {result['processed_count']}")
                print(f"   - Skipped: {result['skipped_count']}")
                print(f"   - Errors: {result['error_count']}")
            else:
                print(f"❌ Failed to process file: {result['error']}")
            
            overall_results['file_results'][json_file] = result
        
        return overall_results

def main():
    """Main function to process all JSON files in a directory."""
    if len(sys.argv) != 2:
        print("Usage: python api_processor.py <directory_path>")
        print("Example: python api_processor.py input_schemas/")
        sys.exit(1)

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"❌ Error: {input_dir} is not a valid directory.")
        sys.exit(1)

    # Collect all JSON files in the directory
    json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".json")]

    if not json_files:
        print(f"❌ No JSON files found in directory: {input_dir}")
        sys.exit(1)

    try:
        print("📦 Initializing API Lookup Processor...")
        processor = APILookupProcessor()

        # Process all JSON files in the directory
        results = processor.process_multiple_files(json_files)

        print(f"\n{'='*60}")
        print("📊 BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"📁 Files processed: {results['files_processed']}/{len(json_files)}")
        print(f"✅ Total processed: {results['total_processed']} entries")
        print(f"⏭️  Total skipped: {results['total_skipped']} entries")
        print(f"❌ Total errors: {results['total_errors']} entries")
        print(f"📊 Final lookup size: {len(processor.api_lookup)} entries")

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()