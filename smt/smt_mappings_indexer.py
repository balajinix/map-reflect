#!/usr/bin/env python3
"""
SMT ChromaDB Indexer
Indexes SMT mapping files into ChromaDB for efficient retrieval
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import re
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    logger.info("ChromaDB imported successfully")
except ImportError as e:
    logger.error("ChromaDB not found. Install with: pip install chromadb")
    exit(1)


class SMTMappingsIndexer:
    """
    Indexes SMT mapping files into ChromaDB for efficient semantic search and reuse.
    """
    
    def __init__(self, 
                 chroma_db_path: str = "./chroma_db",
                 collection_name: str = "smt_mappings"):
        """
        Initialize the ChromaDB indexer.
        
        Args:
            chroma_db_path: Path to store ChromaDB data
            collection_name: Name of the ChromaDB collection
        """
        self.chroma_db_path = Path(chroma_db_path)
        self.collection_name = collection_name
        
        # Ensure directory exists
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.chroma_db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            logger.info(f"ChromaDB client initialized at {self.chroma_db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": "SMT schema mapping files for reuse",
                    "created": datetime.now().isoformat()
                }
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def parse_mapping_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a single SMT mapping file and extract indexable information.
        
        Args:
            file_path: Path to the mapping JSON file
            
        Returns:
            Dictionary with parsed mapping information
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            
            # Extract source and target API names from filename
            filename = file_path.stem  # Remove .json extension
            if "-" in filename:
                source_api, target_api = filename.split("-", 1)
            else:
                source_api = target_api = filename
            
            # Extract mapping information
            mapping_name = mapping_data.get("name", source_api)
            properties = mapping_data.get("properties", {})
            
            # Create searchable text content
            searchable_content = self._create_searchable_content(
                source_api, target_api, mapping_name, properties
            )
            
            # Extract field mappings for structured access
            field_mappings = []
            for source_field, mappings in properties.items():
                if mappings and isinstance(mappings, list):
                    for mapping in mappings:
                        if isinstance(mapping, dict):
                            field_mappings.append({
                                "source_field": source_field,
                                "target_table": mapping.get("mapTable", target_api),
                                "target_field": mapping.get("mapProperty", ""),
                                "justification": mapping.get("justification", "")
                            })
            
            # Generate content hash for change detection
            content_hash = hashlib.md5(json.dumps(mapping_data, sort_keys=True).encode()).hexdigest()
            
            parsed_mapping = {
                "file_path": str(file_path),
                "filename": filename,
                "source_api": source_api,
                "target_api": target_api,
                "mapping_name": mapping_name,
                "num_mappings": len(field_mappings),
                "searchable_content": searchable_content,
                "field_mappings": field_mappings,
                "content_hash": content_hash,
                "raw_data": mapping_data
            }
            
            return parsed_mapping
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return {}
    
    def _create_searchable_content(self, 
                                 source_api: str, 
                                 target_api: str, 
                                 mapping_name: str, 
                                 properties: Dict[str, Any]) -> str:
        """
        Create searchable text content from mapping data.
        
        This content will be embedded and used for semantic search.
        """
        content_parts = []
        
        # Add API names in readable format
        content_parts.append(f"Source API: {self._make_readable(source_api)}")
        content_parts.append(f"Target API: {self._make_readable(target_api)}")
        content_parts.append(f"Mapping: {self._make_readable(mapping_name)}")
        
        # Add field mapping descriptions
        field_descriptions = []
        for source_field, mappings in properties.items():
            if mappings and isinstance(mappings, list):
                readable_source = self._make_readable(source_field)
                
                for mapping in mappings:
                    if isinstance(mapping, dict):
                        target_field = mapping.get("mapProperty", "")
                        justification = mapping.get("justification", "")
                        
                        if target_field:
                            readable_target = self._make_readable(target_field)
                            field_desc = f"{readable_source} maps to {readable_target}"
                            if justification:
                                field_desc += f" ({justification})"
                            field_descriptions.append(field_desc)
        
        if field_descriptions:
            content_parts.append(f"Field Mappings: {'; '.join(field_descriptions[:10])}")  # Limit to 10
        
        return "\n".join(content_parts)
    
    def _make_readable(self, text: str) -> str:
        """Convert technical names to readable format."""
        if not text:
            return ""
        
        # Handle camelCase and PascalCase
        readable = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Handle snake_case, kebab-case, and dots
        readable = readable.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        
        # Clean up multiple spaces
        readable = re.sub(r'\s+', ' ', readable).strip().lower()
        
        return readable
    
    def index_mapping_files(self, mappings_dir: str, force_reindex: bool = False) -> int:
        """Index all mapping files in the specified directory."""
        mappings_path = Path(mappings_dir)
        
        if not mappings_path.exists():
            logger.error(f"Mappings directory not found: {mappings_path}")
            return 0
        
        # Find all JSON mapping files
        mapping_files = list(mappings_path.glob("*-*.json"))
        logger.info(f"Found {len(mapping_files)} mapping files in {mappings_path}")
        
        if not mapping_files:
            logger.warning("No mapping files found (expected format: source-target.json)")
            return 0
        
        # Clear collection if force reindex
        if force_reindex:
            logger.info("Force reindex: clearing existing collection")
            try:
                self.client.delete_collection(self.collection_name)
            except:
                pass
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "SMT schema mapping files for reuse",
                    "created": datetime.now().isoformat()
                }
            )
        
        # Check existing documents
        existing_data = {}
        try:
            existing_results = self.collection.get()
            existing_data = {
                doc_id: metadata.get("content_hash", "")
                for doc_id, metadata in zip(
                    existing_results.get("ids", []),
                    existing_results.get("metadatas", [])
                )
            }
            logger.info(f"Found {len(existing_data)} existing mappings in collection")
        except Exception as e:
            logger.warning(f"Could not check existing documents: {e}")
        
        # Process each mapping file
        indexed_count = 0
        skipped_count = 0
        updated_count = 0
        
        for mapping_file in mapping_files:
            try:
                doc_id = mapping_file.stem
                
                # Parse mapping file
                parsed_mapping = self.parse_mapping_file(mapping_file)
                
                if not parsed_mapping:
                    logger.warning(f"Skipping empty/invalid file: {mapping_file}")
                    continue
                
                # Check if update needed
                current_hash = parsed_mapping["content_hash"]
                existing_hash = existing_data.get(doc_id, "")
                
                if not force_reindex and doc_id in existing_data and current_hash == existing_hash:
                    skipped_count += 1
                    continue
                
                # Prepare metadata
                metadata = {
                    "filename": parsed_mapping["filename"],
                    "source_api": parsed_mapping["source_api"],
                    "target_api": parsed_mapping["target_api"],
                    "mapping_name": parsed_mapping["mapping_name"],
                    "num_mappings": parsed_mapping["num_mappings"],
                    "file_path": parsed_mapping["file_path"],
                    "content_hash": current_hash,
                    "indexed_at": datetime.now().isoformat()
                }
                
                # Add or update in ChromaDB
                if doc_id in existing_data:
                    # Update existing
                    self.collection.update(
                        documents=[parsed_mapping["searchable_content"]],
                        metadatas=[metadata],
                        ids=[doc_id]
                    )
                    updated_count += 1
                else:
                    # Add new
                    self.collection.add(
                        documents=[parsed_mapping["searchable_content"]],
                        metadatas=[metadata],
                        ids=[doc_id]
                    )
                    indexed_count += 1
                
                if (indexed_count + updated_count) % 50 == 0:
                    logger.info(f"   Processed {indexed_count + updated_count}/{len(mapping_files)} files...")
                
            except Exception as e:
                logger.error(f"Error processing {mapping_file}: {e}")
                continue
        
        total_processed = indexed_count + updated_count
        logger.info(f"Indexing complete:")
        logger.info(f"   New mappings indexed: {indexed_count}")
        logger.info(f"   Mappings updated: {updated_count}")
        logger.info(f"   Skipped (unchanged): {skipped_count}")
        logger.info(f"   Total in collection: {len(existing_data) + indexed_count}")
        
        return total_processed
    
    def search_existing_mappings(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for existing mappings using semantic similarity.
        This is the main method expected by the API retriever.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Convert to expected format
            matches = []
            
            if results and 'documents' in results and results['documents']:
                for i in range(len(results['documents'][0])):
                    doc_id = results['ids'][0][i]
                    doc = results['documents'][0][i]
                    metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
                    distance = results['distances'][0][i] if 'distances' in results else 0.0
                    
                    similarity_score = 1.0 - distance
                    
                    # Apply minimum threshold
                    if similarity_score < 0.2:
                        continue
                    
                    match = {
                        "mapping_id": doc_id,
                        "description": doc,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                        "file_path": metadata.get("file_path", ""),
                        "source_api": metadata.get("source_api", ""),
                        "target_api": metadata.get("target_api", ""),
                        "num_mappings": metadata.get("num_mappings", 0)
                    }
                    matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching existing mappings: {e}")
            return []
    
    def get_mapping_by_apis(self, source_api: str, target_api: str) -> Optional[Dict[str, Any]]:
        """
        Get mapping by source and target API names.
        
        Args:
            source_api: Source API name
            target_api: Target API name
            
        Returns:
            Mapping metadata if found
        """
        try:
            # Try both directions
            possible_ids = [
                f"{source_api}-{target_api}",
                f"{target_api}-{source_api}"
            ]
            
            for doc_id in possible_ids:
                try:
                    result = self.collection.get(
                        ids=[doc_id],
                        include=["documents", "metadatas"]
                    )
                    if result.get("ids") and len(result["ids"]) > 0:
                        return {
                            "mapping_id": result["ids"][0],
                            "metadata": result["metadatas"][0],
                            "description": result["documents"][0],
                            "file_path": result["metadatas"][0].get("file_path", "")
                        }
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting mapping by APIs: {e}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection."""
        try:
            all_data = self.collection.get(include=["metadatas"])
            
            if not all_data["ids"]:
                return {"total_mappings": 0}
            
            metadatas = all_data["metadatas"]
            
            # Calculate statistics
            total_mappings = len(metadatas)
            api_pairs = set()
            source_apis = set()
            target_apis = set()
            total_field_mappings = 0
            
            for metadata in metadatas:
                source_api = metadata.get("source_api", "")
                target_api = metadata.get("target_api", "")
                
                if source_api and target_api:
                    api_pairs.add(f"{source_api}-{target_api}")
                    source_apis.add(source_api)
                    target_apis.add(target_api)
                
                total_field_mappings += metadata.get("num_mappings", 0)
            
            return {
                "total_mappings": total_mappings,
                "total_field_mappings": total_field_mappings,
                "unique_api_pairs": len(api_pairs),
                "unique_source_apis": len(source_apis),
                "unique_target_apis": len(target_apis),
                "avg_fields_per_mapping": total_field_mappings / total_mappings if total_mappings > 0 else 0,
                "collection_name": self.collection_name,
                "db_path": str(self.chroma_db_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}


def main():
    """Main entry point for the mappings indexer."""
    parser = argparse.ArgumentParser(description="SMT Mappings Indexer")
    parser.add_argument("--mappings-dir", required=True, help="Directory containing SMT mapping files")
    parser.add_argument("--chroma-db-path", default="./chroma_db", help="Path to store ChromaDB data")
    parser.add_argument("--collection-name", default="smt_mappings", help="ChromaDB collection name")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindex (clear existing)")
    parser.add_argument("--test-query", help="Test query to search after indexing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"SMT Mappings ChromaDB Indexer")
    logger.info(f"Mappings directory: {args.mappings_dir}")
    logger.info(f"ChromaDB path: {args.chroma_db_path}")
    logger.info(f"Collection name: {args.collection_name}")
    
    # Initialize indexer
    indexer = SMTMappingsIndexer(
        chroma_db_path=args.chroma_db_path,
        collection_name=args.collection_name
    )
    
    # Index mapping files
    processed_count = indexer.index_mapping_files(
        mappings_dir=args.mappings_dir,
        force_reindex=args.force_reindex
    )
    
    # Show collection statistics
    stats = indexer.get_collection_stats()
    print(f"\nCollection Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test search if query provided
    if args.test_query:
        print(f"\nTesting search with query: '{args.test_query}'")
        results = indexer.search_existing_mappings(args.test_query, n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n   Result {i}:")
            print(f"   Mapping ID: {result['mapping_id']}")
            print(f"   Similarity: {result['similarity_score']:.3f}")
            print(f"   Source API: {result['source_api']}")
            print(f"   Target API: {result['target_api']}")
            print(f"   Field Mappings: {result['num_mappings']}")
    
    print(f"\nIndexing completed successfully!")
    print(f"   Processed {processed_count} mapping files")
    print(f"   ChromaDB collection: {args.collection_name}")
    print(f"   Database location: {args.chroma_db_path}")


if __name__ == "__main__":
    main()
