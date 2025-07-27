""" Test code to create a API lookup using instana and turbo JSON files """

import json
import os
import sys
from typing import Dict, List
from pathlib import Path

# Import the RAG module
from instana_turbo_api_docs.create_descriptions import InstanaAPIDescriptionGenerator

class APILookupProcessor:
    def __init__(self, rag_generator: InstanaAPIDescriptionGenerator = None,
                 output_file: str = "/home/gunjan/TheToolkit/schema-mapping-toolkit/api_lookup.json"):
        """
        Initialize the API Lookup Processor.
        
        Args:
            rag_generator: Instance of InstanaAPIDescriptionGenerator
            output_file: Path to output JSON file
        """
        self.rag_generator = rag_generator or InstanaAPIDescriptionGenerator()
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
        Parse the fullname to extract product, type, and field names.

        Args:
            fullname: String in format "productname_typename_typename.field1.field2..."

        Returns:
            Dict with parsed components
        """
        if '.' not in fullname:
            raise ValueError(f"Invalid fullname format: {fullname}. Must include at least one '.' separating fields.")
        
        # Split at the first dot to isolate the full identifier (product_type) and the field path
        identifier, *field_parts = fullname.split('.', 1)
        fields_str = field_parts[0] if field_parts else ''
        fields = fields_str.split('.') if fields_str else []

        # Get product name and type name
        if '_' not in identifier:
            raise ValueError(f"Invalid identifier format: {identifier}. Must contain at least one underscore.")

        product_name, type_name = identifier.split('_', 1)

        return {
            'product_name': product_name,
            'type_name': type_name,
            'field_name': fields_str,
            'fields': fields,
            'fullname': fullname
        }

    def validate_product_collection_availability(self, fullname: str) -> bool:
        """
        Validate that the required collection is available for the product in the fullname.
        
        Args:
            fullname: The full API path string
            
        Returns:
            Boolean indicating if the collection is available
        """
        try:
            parsed = self.parse_fullname(fullname)
            product_name = parsed['product_name'].lower()
            
            # Check if the appropriate collection is available
            collection = self.rag_generator.get_collection_for_product(product_name)
            
            if collection is None:
                print(f"Warning: Collection not available for product '{product_name}' in {fullname}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validating collection for {fullname}: {e}")
            return False
    
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
            collection_unavailable_count = 0
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
                
                # Validate that the required collection is available
                if not self.validate_product_collection_availability(fullname):
                    print(f"Skipping {fullname} (collection not available)")
                    collection_unavailable_count += 1
                    continue
                
                try:
                    # Parse the fullname
                    parsed = self.parse_fullname(fullname)
                    
                    # Get original description from the JSON
                    original_desc = prop.get('description', '').strip()
                    print(f"Original description: {original_desc}")
                    
                    # Get enhanced description using original description
                    # The RAG generator will now automatically route to the correct collection
                    description = self.rag_generator.get_description_only(fullname, original_desc=original_desc)
                    
                    # Check if description generation was successful
                    if description.startswith("Error generating description:"):
                        print(f"Warning: RAG failed for {fullname}: {description}")
                        error_count += 1
                        errors.append({
                            'fullname': fullname,
                            'error': description,
                            'product': parsed['product_name']
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
                    
                    print(f"✅ Successfully processed: {fullname} (using {parsed['product_name']} collection)")
                    
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
                        'error': error_msg,
                        'product': 'unknown'
                    })
                    continue
            
            # Final save
            self.save_lookup()
            
            return {
                'success': True,
                'processed_count': processed_count,
                'skipped_count': skipped_count,
                'error_count': error_count,
                'collection_unavailable_count': collection_unavailable_count,
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
            'total_collection_unavailable': 0,
            'file_results': {},
            'all_errors': [],
            'products_processed': set()
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
                overall_results['total_collection_unavailable'] += result.get('collection_unavailable_count', 0)
                overall_results['all_errors'].extend(result['errors'])
                
                # Track which products were processed
                for error in result['errors']:
                    if 'product' in error and error['product'] != 'unknown':
                        overall_results['products_processed'].add(error['product'])
                
                print(f"✅ File processed successfully:")
                print(f"   - Processed: {result['processed_count']}")
                print(f"   - Skipped: {result['skipped_count']}")
                print(f"   - Collection unavailable: {result.get('collection_unavailable_count', 0)}")
                print(f"   - Errors: {result['error_count']}")
            else:
                print(f"❌ Failed to process file: {result['error']}")
            
            overall_results['file_results'][json_file] = result
        
        return overall_results


def main():
    """Main function to demonstrate usage."""
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python api_processor.py <json_file1> [json_file2] ...")
        print("Example: python api_processor.py concert_vulnerabilities.json")
        print("Note: The script now supports both 'instana' and 'turbo' product collections")
        sys.exit(1)
    
    json_files = sys.argv[1:]
    
    # Validate files exist
    for json_file in json_files:
        if not os.path.exists(json_file):
            print(f"Error: File not found: {json_file}")
            sys.exit(1)
    
    try:
        # Initialize the processor
        print("Supported products: instana (instana_api collection), turbo (turbo_api collection)")
        processor = APILookupProcessor()
        
        # Process the files
        if len(json_files) == 1:
            result = processor.process_json_file(json_files[0])
            
            if result['success']:
                print(f"\n{'='*60}")
                print("PROCESSING COMPLETE")
                print(f"{'='*60}")
                print(f"✅ Processed: {result['processed_count']} entries")
                print(f"⏭️  Skipped: {result['skipped_count']} entries")
                print(f"🗄️  Collection unavailable: {result.get('collection_unavailable_count', 0)} entries")
                print(f"❌ Errors: {result['error_count']} entries")
                print(f"📁 Total in lookup: {result['total_entries']} entries")
                
                if result['errors']:
                    print(f"\nErrors encountered:")
                    for error in result['errors'][:5]:  # Show first 5 errors
                        product_info = f" (product: {error.get('product', 'unknown')})" if 'product' in error else ""
                        print(f"  - {error['fullname']}: {error['error']}{product_info}")
                    if len(result['errors']) > 5:
                        print(f"  ... and {len(result['errors']) - 5} more errors")
            else:
                print(f"❌ Processing failed: {result['error']}")
        else:
            # Process multiple files
            results = processor.process_multiple_files(json_files)
            
            print(f"\n{'='*60}")
            print("BATCH PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"📁 Files processed: {results['files_processed']}/{len(json_files)}")
            print(f"✅ Total processed: {results['total_processed']} entries")
            print(f"⏭️  Total skipped: {results['total_skipped']} entries")
            print(f"🗄️  Total collection unavailable: {results['total_collection_unavailable']} entries")
            print(f"❌ Total errors: {results['total_errors']} entries")
            print(f"📊 Final lookup size: {len(processor.api_lookup)} entries")
            
            if results['products_processed']:
                print(f"🏷️  Products processed: {', '.join(sorted(results['products_processed']))}")

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()