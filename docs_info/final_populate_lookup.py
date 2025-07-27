""" 
Populates the API lookup. Traverses the input schema JSON files to get the name, improves the descriptions using RAG and then stores it as an entry into the api_lookup.json 
Handles routing of endpoints to the respective API products' chromaDB store - Concert, Instana or Turbo.
"""

import json
import os
import sys
from typing import Dict, List, Optional
from pathlib import Path

# Import the different RAG modules
from concert_api_docs.create_descriptions import ConcertAPIDescriptionGenerator
from instana_turbo_api_docs.create_descriptions import InstanaAPIDescriptionGenerator

class UnifiedAPILookupProcessor:
    def __init__(self, output_file: str = "api_lookup.json"):
        """
        Initialize the Unified API Lookup Processor.
        
        Args:
            output_file: Path to output JSON file
        """
        self.output_file = output_file
        self.api_lookup = {}
        
        # Initialize RAG generators (lazy loading)
        self._concert_generator = None
        self._instana_generator = None
        self._turbo_generator = None
        
        # Product routing configuration
        self.product_config = {
            'concert': {
                'generator_class': ConcertAPIDescriptionGenerator,
                'parse_method': 'parse_concert_fullname'
            },
            'instana': {
                'generator_class': InstanaAPIDescriptionGenerator,
                'parse_method': 'parse_instana_fullname'
            },
            'turbo': {
                'generator_class': InstanaAPIDescriptionGenerator,  # Same generator class but separate instance
                'parse_method': 'parse_instana_fullname'
            }
        }
        
        # Load existing lookup file if it exists
        self.load_existing_lookup()
    
    def get_generator(self, product_name: str):
        """
        Get the appropriate RAG generator for the product (lazy loading).
        Each product gets its own generator instance.
        
        Args:
            product_name: Name of the product
            
        Returns:
            RAG generator instance
        """
        if product_name == 'concert':
            if self._concert_generator is None:
                self._concert_generator = ConcertAPIDescriptionGenerator()
            return self._concert_generator
        elif product_name == 'instana':
            if self._instana_generator is None:
                self._instana_generator = InstanaAPIDescriptionGenerator()
            return self._instana_generator
        elif product_name == 'turbo':
            if self._turbo_generator is None:
                self._turbo_generator = InstanaAPIDescriptionGenerator()
            return self._turbo_generator
        else:
            raise ValueError(f"Unsupported product: {product_name}")
    
    def extract_product_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract product name from filename.
        
        Args:
            filename: The JSON filename
            
        Returns:
            Product name or None if not supported
        """
        # Get just the filename without path
        base_filename = os.path.basename(filename)
        
        # Skip wxo files
        if base_filename.startswith('wxo_'):
            return None
        
        # Extract product name (everything before first underscore)
        if '_' in base_filename:
            product_name = base_filename.split('_')[0].lower()
            if product_name in self.product_config:
                return product_name
        
        return None
    
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
    
    def parse_concert_fullname(self, fullname: str) -> Dict[str, str]:
        """
        Parse the fullname for Concert products.
        
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
    
    def parse_instana_fullname(self, fullname: str) -> Dict[str, str]:
        """
        Parse the fullname for Instana/Turbo products.
        
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
    
    def validate_product_collection_availability(self, product_name: str, fullname: str) -> bool:
        """
        Validate that the required collection is available for the product.
        
        Args:
            product_name: Name of the product
            fullname: The full API path string
            
        Returns:
            Boolean indicating if the collection is available
        """
        try:
            # Check for products that use InstanaAPIDescriptionGenerator
            if product_name in ['instana', 'turbo']:
                generator = self.get_generator(product_name)
                
                # Check if the appropriate collection is available for this specific product
                collection = generator.get_collection_for_product(product_name.lower())
                
                if collection is None:
                    print(f"Warning: Collection not available for product '{product_name}' in {fullname}")
                    return False
            
            # Concert products don't need collection validation
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
        
        # Extract product name from filename
        product_name = self.extract_product_from_filename(json_file_path)
        if product_name is None:
            return {
                'success': False,
                'error': f"Unsupported product or wxo file: {json_file_path}"
            }
        
        print(f"Detected product: {product_name}")
        
        try:
            # Get the appropriate generator and parser
            generator = self.get_generator(product_name)
            parse_method = getattr(self, self.product_config[product_name]['parse_method'])
            
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
                
                # Validate that the required collection is available (for instana/turbo products)
                if not self.validate_product_collection_availability(product_name, fullname):
                    print(f"Skipping {fullname} (collection not available for {product_name})")
                    collection_unavailable_count += 1
                    continue
                
                try:
                    # Parse the fullname using the appropriate method
                    parsed = parse_method(fullname)
                    
                    # Get original description from the JSON
                    original_desc = prop.get('description', '').strip()
                    print(f"Original description: {original_desc}")
                    
                    # Get enhanced description using the appropriate generator
                    description = generator.get_description_only(fullname, original_desc=original_desc)
                    
                    # Check if description generation was successful
                    if description.startswith("Error generating description:"):
                        print(f"Warning: RAG failed for {fullname}: {description}")
                        error_count += 1
                        errors.append({
                            'fullname': fullname,
                            'error': description,
                            'product': product_name
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
                    
                    print(f"✅ Successfully processed: {fullname} (using {product_name} generator)")
                    
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
                        'product': product_name
                    })
                    continue
            
            # Final save
            self.save_lookup()
            
            return {
                'success': True,
                'product_name': product_name,
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
            'products_processed': set(),
            'unsupported_files': []
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
                overall_results['products_processed'].add(result['product_name'])
                
                print(f"✅ File processed successfully ({result['product_name']}):")
                print(f"   - Processed: {result['processed_count']}")
                print(f"   - Skipped: {result['skipped_count']}")
                print(f"   - Collection unavailable: {result.get('collection_unavailable_count', 0)}")
                print(f"   - Errors: {result['error_count']}")
            else:
                if "Unsupported product or wxo file" in result['error']:
                    overall_results['unsupported_files'].append(json_file)
                    print(f"⏭️  Skipped unsupported file: {json_file}")
                else:
                    print(f"❌ Failed to process file: {result['error']}")
            
            overall_results['file_results'][json_file] = result
        
        return overall_results


def main():
    """Main function to process JSON files or directories."""
    if len(sys.argv) < 2:
        print("Usage: python api_processor.py <path1> [path2] ...")
        print("Examples:")
        print("  python api_processor.py concert_vulnerability.json")
        print("  python api_processor.py input_schemas/")
        print("  python api_processor.py file1.json file2.json")
        print("\nSupported products: concert, instana, turbo")
        print("Note: Each product (concert, instana, turbo) uses its own independent ChromaDB collection")
        print("Note: Files starting with 'wxo_' will be automatically skipped")
        sys.exit(1)
    
    paths = sys.argv[1:]
    json_files = []
    
    # Collect all JSON files from the provided paths
    for path in paths:
        if os.path.isfile(path):
            if path.endswith('.json'):
                json_files.append(path)
            else:
                print(f"Warning: Skipping non-JSON file: {path}")
        elif os.path.isdir(path):
            # Get all JSON files in the directory
            dir_json_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.json')]
            json_files.extend(dir_json_files)
            print(f"Found {len(dir_json_files)} JSON files in directory: {path}")
        else:
            print(f"Warning: Path not found: {path}")
    
    if not json_files:
        print("❌ No JSON files found to process")
        sys.exit(1)
    
    try:
        print(f"🚀 Initializing Unified API Lookup Processor...")
        print(f"📁 Total files to process: {len(json_files)}")
        processor = UnifiedAPILookupProcessor()
        
        # Process all files
        results = processor.process_multiple_files(json_files)
        
        print(f"\n{'='*60}")
        print("🎉 BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"📁 Files processed: {results['files_processed']}/{len(json_files)}")
        print(f"⏭️  Unsupported files skipped: {len(results['unsupported_files'])}")
        print(f"✅ Total processed: {results['total_processed']} entries")
        print(f"⏭️  Total skipped: {results['total_skipped']} entries")
        print(f"🗄️  Total collection unavailable: {results['total_collection_unavailable']} entries")
        print(f"❌ Total errors: {results['total_errors']} entries")
        print(f"📊 Final lookup size: {len(processor.api_lookup)} entries")
        
        if results['products_processed']:
            print(f"🏷️  Products processed: {', '.join(sorted(results['products_processed']))}")
        
        if results['unsupported_files']:
            print(f"\n📋 Unsupported files skipped:")
            for file in results['unsupported_files']:
                print(f"  - {file}")
        
        # Show summary of errors by product
        if results['all_errors']:
            product_errors = {}
            for error in results['all_errors']:
                product = error.get('product', 'unknown')
                if product not in product_errors:
                    product_errors[product] = 0
                product_errors[product] += 1
            
            print(f"\n📊 Error summary by product:")
            for product, count in product_errors.items():
                print(f"  - {product}: {count} errors")

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()