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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    logger.info("✅ ChromaDB imported successfully")
except ImportError as e:
    logger.error("❌ ChromaDB not found. Install with: pip install chromadb")
    exit(1)


class SMTMappingsIndexer:
    """
    Indexes SMT mapping files into ChromaDB for efficient semantic search.
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
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"✅ Using existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "SMT schema mapping files"}
            )
            logger.info(f"✅ Created new collection: {collection_name}")
    
    def parse_mapping_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a single SMT mapping file and extract indexable information.
        
        Args:
            file_path: Path to the mapping JSON file
            
        Returns:
            Dictionary with parsed mapping information
        """
        try:
            with open(file_path, 'r') as f:
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
            
            parsed_mapping = {
                "file_path": str(file_path),
                "filename": filename,
                "source_api": source_api,
                "target_api": target_api,
                "mapping_name": mapping_name,
                "num_mappings": len(field_mappings),
                "searchable_content": searchable_content,
                "field_mappings": field_mappings,
                "raw_data": mapping_data
            }
            
            return parsed_mapping
            
        except Exception as e:
            logger.error(f"❌ Error parsing {file_path}: {e}")
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
        content_parts.append("Field mappings:")
        
        for source_field, mappings in properties.items():
            if mappings and isinstance(mappings, list):
                # Make source field readable
                readable_source = self._make_readable(source_field)
                content_parts.append(f"- {readable_source}")
                
                for mapping in mappings:
                    if isinstance(mapping, dict):
                        target_field = mapping.get("mapProperty", "")
                        justification = mapping.get("justification", "")
                        
                        readable_target = self._make_readable(target_field)
                        content_parts.append(f"  maps to {readable_target}")
                        
                        if justification:
                            content_parts.append(f"  reason: {justification}")
        
        # Add domain-specific keywords
        domain_keywords = self._extract_domain_keywords(source_api, target_api, properties)
        if domain_keywords:
            content_parts.append(f"Keywords: {' '.join(domain_keywords)}")
        
        return "\n".join(content_parts)
    
    def _make_readable(self, text: str) -> str:
        """Convert technical names to readable format."""
        if not text:
            return ""
        
        # Handle camelCase and PascalCase
        readable = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Handle snake_case and kebab-case
        readable = readable.replace('_', ' ').replace('-', ' ')
        
        # Handle dots (field separators)
        readable = readable.replace('.', ' ')
        
        # Clean up multiple spaces
        readable = re.sub(r'\s+', ' ', readable).strip()
        
        return readable.lower()
    
    def _extract_domain_keywords(self, 
                                source_api: str, 
                                target_api: str, 
                                properties: Dict[str, Any]) -> List[str]:
        """Extract domain-specific keywords for better search."""
        keywords = set()
        
        # Add API-specific keywords
        api_keywords = {
            "turbo": ["performance", "optimization", "virtual", "disk", "storage", "entity", "market"],
            "instana": ["monitoring", "infrastructure", "application", "performance", "apm", "observability"],
            "concert": ["security", "vulnerability", "scanning", "assessment", "compliance", "risk"]
        }
        
        for api in [source_api, target_api]:
            api_lower = api.lower()
            for keyword_api, keywords_list in api_keywords.items():
                if keyword_api in api_lower:
                    keywords.update(keywords_list)
        
        # Extract keywords from field names
        all_text = " ".join([
            source_api, target_api,
            " ".join(properties.keys()),
            " ".join([
                mapping.get("mapProperty", "") + " " + mapping.get("justification", "")
                for mappings in properties.values() 
                if isinstance(mappings, list)
                for mapping in mappings
                if isinstance(mapping, dict)
            ])
        ])
        
        # Common technical keywords
        tech_keywords = [
            "stats", "statistics", "metrics", "entities", "data", "information",
            "response", "request", "api", "endpoint", "service", "application",
            "count", "total", "hits", "results", "related", "num", "number"
        ]
        
        for keyword in tech_keywords:
            if keyword in all_text.lower():
                keywords.add(keyword)
        
        return list(keywords)
    
    def index_mapping_files(self, mappings_dir: str, force_reindex: bool = False) -> int:
        """
        Index all mapping files in the specified directory.
        
        Args:
            mappings_dir: Directory containing SMT mapping files
            force_reindex: Whether to clear and recreate the index
            
        Returns:
            Number of files successfully indexed
        """
        mappings_path = Path(mappings_dir)
        
        if not mappings_path.exists():
            logger.error(f"❌ Mappings directory not found: {mappings_path}")
            return 0
        
        # Find all JSON mapping files
        mapping_files = list(mappings_path.glob("*-*.json"))
        logger.info(f"📁 Found {len(mapping_files)} mapping files in {mappings_path}")
        
        if not mapping_files:
            logger.warning("⚠️  No mapping files found (expected format: source-target.json)")
            return 0
        
        # Clear collection if force reindex
        if force_reindex:
            logger.info("🗑️  Force reindex: clearing existing collection")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "SMT schema mapping files"}
            )
        
        # Check existing documents to avoid duplicates
        existing_ids = set()
        try:
            existing_data = self.collection.get()
            existing_ids = set(existing_data.get("ids", []))
            logger.info(f"📋 Found {len(existing_ids)} existing documents in collection")
        except Exception as e:
            logger.warning(f"Could not check existing documents: {e}")
        
        # Process each mapping file
        indexed_count = 0
        skipped_count = 0
        
        for mapping_file in mapping_files:
            try:
                # Create document ID from filename
                doc_id = mapping_file.stem
                
                # Skip if already indexed (unless force reindex)
                if not force_reindex and doc_id in existing_ids:
                    skipped_count += 1
                    continue
                
                # Parse mapping file
                parsed_mapping = self.parse_mapping_file(mapping_file)
                
                if not parsed_mapping:
                    logger.warning(f"⚠️  Skipping empty/invalid file: {mapping_file}")
                    continue
                
                # Prepare document for ChromaDB
                document_text = parsed_mapping["searchable_content"]
                metadata = {
                    "filename": parsed_mapping["filename"],
                    "source_api": parsed_mapping["source_api"],
                    "target_api": parsed_mapping["target_api"],
                    "mapping_name": parsed_mapping["mapping_name"],
                    "num_mappings": parsed_mapping["num_mappings"],
                    "file_path": parsed_mapping["file_path"],
                    "indexed_at": datetime.now().isoformat()
                }
                
                # Add to ChromaDB
                self.collection.add(
                    documents=[document_text],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
                
                indexed_count += 1
                
                if indexed_count % 50 == 0:
                    logger.info(f"   Indexed {indexed_count}/{len(mapping_files)} files...")
                
            except Exception as e:
                logger.error(f"❌ Error indexing {mapping_file}: {e}")
                continue
        
        logger.info(f"✅ Indexing complete:")
        logger.info(f"   Indexed: {indexed_count} files")
        logger.info(f"   Skipped: {skipped_count} files (already indexed)")
        logger.info(f"   Total in collection: {len(existing_ids) + indexed_count}")
        
        return indexed_count
    
    
    def search_existing_mappings(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for existing mappings using semantic similarity.
        This is the method expected by smt_api_retriever.py
        
        Args:
            query: Natural language query describing the mapping task
            n_results: Number of results to return
            
        Returns:
            List of mapping matches in the format expected by api_retriever
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Convert to format expected by api_retriever
            matches = []
            
            if results and 'documents' in results and results['documents']:
                for i in range(len(results['documents'][0])):
                    doc = results['documents'][0][i]
                    metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
                    distance = results['distances'][0][i] if 'distances' in results else 0.0
                    
                    match = {
                        "mapping_id": metadata.get("filename", f"mapping_{i}"),
                        "description": doc,
                        "metadata": metadata,
                        "similarity_score": 1.0 - distance,  # Convert distance to similarity
                        "file_path": metadata.get("file_path", "")
                    }
                    matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"❌ Error searching existing mappings: {e}")
            return []
    
    # this is not used anymore - TODO: delete
    def search_mappings(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant mappings using semantic similarity.
        
        Args:
            query: Natural language query
            n_results: Number of results to return
            
        Returns:
            List of matching mapping information
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Process results
            matches = []
            for i in range(len(results.get("ids", [[]])[0])):
                match = {
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "document": results["documents"][0][i]
                }
                matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"❌ Error searching mappings: {e}")
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
                    result = self.collection.get(ids=[doc_id])
                    if result.get("ids") and len(result["ids"]) > 0:
                        return {
                            "id": result["ids"][0],
                            "metadata": result["metadatas"][0],
                            "document": result["documents"][0]
                        }
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting mapping by APIs: {e}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection."""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to analyze
            sample_data = self.collection.get(limit=10)
            
            api_pairs = set()
            source_apis = set()
            target_apis = set()
            
            for metadata in sample_data.get("metadatas", []):
                source_api = metadata.get("source_api", "")
                target_api = metadata.get("target_api", "")
                
                if source_api and target_api:
                    api_pairs.add(f"{source_api}-{target_api}")
                    source_apis.add(source_api)
                    target_apis.add(target_api)
            
            return {
                "total_mappings": count,
                "sample_api_pairs": len(api_pairs),
                "sample_source_apis": len(source_apis),
                "sample_target_apis": len(target_apis),
                "collection_name": self.collection_name,
                "db_path": str(self.chroma_db_path)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting collection stats: {e}")
            return {}


def main():
    """Main entry point for the ChromaDB indexer."""
    parser = argparse.ArgumentParser(description="Index SMT mapping files into ChromaDB")
    parser.add_argument("--mappings-dir", required=True, help="Directory containing SMT mapping files")
    parser.add_argument("--chroma-db-path", default="./chroma_db", help="Path to store ChromaDB data")
    parser.add_argument("--collection-name", default="smt_mappings", help="ChromaDB collection name")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindex (clear existing)")
    parser.add_argument("--test-query", help="Test query to search after indexing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"SMT ChromaDB Indexer")
    logger.info(f"Mappings directory: {args.mappings_dir}")
    logger.info(f"ChromaDB path: {args.chroma_db_path}")
    logger.info(f"Collection name: {args.collection_name}")
    
    # Initialize indexer
    indexer = SMTMappingsIndexer(
        chroma_db_path=args.chroma_db_path,
        collection_name=args.collection_name
    )
    
    # Index mapping files
    indexed_count = indexer.index_mapping_files(
        mappings_dir=args.mappings_dir,
        force_reindex=args.force_reindex
    )
    
    # Show collection statistics
    stats = indexer.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test search if query provided
    if args.test_query:
        print(f"\n🔍 Testing search with query: '{args.test_query}'")
        results = indexer.search_mappings(args.test_query, n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n   Result {i}:")
            print(f"   ID: {result['id']}")
            print(f"   Distance: {result['distance']:.3f}")
            print(f"   Source API: {result['metadata'].get('source_api', 'N/A')}")
            print(f"   Target API: {result['metadata'].get('target_api', 'N/A')}")
            print(f"   Mappings: {result['metadata'].get('num_mappings', 0)}")
    
    print(f"\n✅ Indexing completed successfully!")
    print(f"   Indexed {indexed_count} files")
    print(f"   ChromaDB collection: {args.collection_name}")
    print(f"   Database location: {args.chroma_db_path}")


if __name__ == "__main__":
    main()
