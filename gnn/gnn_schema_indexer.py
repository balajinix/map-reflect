#!/usr/bin/env python3
"""
SMT GraphQL Schema Indexer
Indexes GraphQL schema files into ChromaDB for schema discovery
This is the renamed version of gnn_chromadb_indexer.py for clarity
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
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    logger.info("ChromaDB imported successfully")
except ImportError as e:
    logger.error("ChromaDB not found. Install with: pip install chromadb")
    exit(1)


class GraphQLSchemaIndexer:
    """
    Indexes GraphQL schema files into ChromaDB for efficient semantic search and discovery.
    """
    
    def __init__(self, 
                 chroma_db_path: str = None,
                 collection_name: str = None):
        """
        Initialize the GraphQL schema indexer.
        
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
            logger.info(f"Using existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "GraphQL schemas for schema mapping"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def parse_graphql_schema(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a GraphQL schema file and extract indexable information.
        
        Args:
            file_path: Path to the GraphQL schema file
            
        Returns:
            Dictionary with parsed schema information
        """
        try:
            with open(file_path, 'r') as f:
                schema_content = f.read()
            
            # Extract schema name from filename
            schema_name = file_path.stem
            
            # Parse GraphQL schema content
            types = self._extract_graphql_types(schema_content)
            fields = self._extract_all_fields(schema_content)
            
            # Create searchable text content
            searchable_content = self._create_searchable_content(
                schema_name, types, fields, schema_content
            )
            
            parsed_schema = {
                "file_path": str(file_path),
                "schema_name": schema_name,
                "types": types,
                "fields": fields,
                "searchable_content": searchable_content,
                "raw_content": schema_content[:2000],  # Truncate for storage
                "file_size": file_path.stat().st_size
            }
            
            return parsed_schema
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return {}
    
    def _extract_graphql_types(self, content: str) -> List[str]:
        """Extract GraphQL type names from schema content."""
        types = []
        
        # Match type definitions
        type_patterns = [
            r'type\s+(\w+)',
            r'interface\s+(\w+)',
            r'enum\s+(\w+)',
            r'input\s+(\w+)',
            r'scalar\s+(\w+)'
        ]
        
        for pattern in type_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            types.extend(matches)
        
        return list(set(types))  # Remove duplicates
    
    def _extract_all_fields(self, content: str) -> List[str]:
        """Extract all field names from GraphQL schema."""
        fields = []
        
        # Extract fields from type definitions
        # Pattern to match field definitions like "fieldName: Type" or "fieldName(args): Type"
        field_pattern = r'^\s*(\w+)(?:\([^)]*\))?\s*:\s*'
        
        lines = content.split('\n')
        in_type_definition = False
        
        for line in lines:
            line = line.strip()
            
            # Check if we're entering a type definition
            if re.match(r'(type|interface|input)\s+\w+', line, re.IGNORECASE):
                in_type_definition = True
                continue
            
            # Check if we're leaving a type definition
            if line == '}':
                in_type_definition = False
                continue
            
            # Extract field names if we're in a type definition
            if in_type_definition:
                match = re.match(field_pattern, line)
                if match:
                    field_name = match.group(1)
                    # Filter out common GraphQL keywords
                    if field_name not in ['query', 'mutation', 'subscription', 'schema']:
                        fields.append(field_name)
        
        return list(set(fields))  # Remove duplicates
    
    def _create_searchable_content(self, 
                                 schema_name: str, 
                                 types: List[str], 
                                 fields: List[str],
                                 raw_content: str) -> str:
        """
        Create searchable text content from GraphQL schema.
        """
        content_parts = []
        
        # Add schema name in readable format
        content_parts.append(f"Schema: {self._make_readable(schema_name)}")
        
        # Add types
        if types:
            readable_types = [self._make_readable(t) for t in types]
            content_parts.append(f"Types: {', '.join(readable_types)}")
        
        # Add fields 
        if fields:
            readable_fields = [self._make_readable(f) for f in fields]
            content_parts.append(f"Fields: {', '.join(readable_fields[:20])}")  # Limit to first 20
        
        # Add domain keywords based on schema name and content
        domain_keywords = self._extract_domain_keywords(schema_name, raw_content)
        if domain_keywords:
            content_parts.append(f"Domain: {', '.join(domain_keywords)}")
        
        # Add description of functionality
        content_parts.append(self._generate_functionality_description(schema_name, types, fields))
        
        return "\n".join(content_parts)
    
    def _make_readable(self, text: str) -> str:
        """Convert technical names to readable format."""
        if not text:
            return ""
        
        # Handle camelCase and PascalCase
        readable = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Handle snake_case and kebab-case
        readable = readable.replace('_', ' ').replace('-', ' ')
        
        # Clean up multiple spaces
        readable = re.sub(r'\s+', ' ', readable).strip()
        
        return readable.lower()
    
    def _extract_domain_keywords(self, schema_name: str, content: str) -> List[str]:
        """Extract domain-specific keywords."""
        keywords = set()
        
        # Domain mappings based on common patterns
        domain_mappings = {
            "turbo": ["performance", "optimization", "virtual", "disk", "storage", "entity", "market", "statistics", "metrics"],
            "instana": ["monitoring", "infrastructure", "application", "performance", "apm", "observability", "topology", "entities"],
            "concert": ["security", "vulnerability", "scanning", "assessment", "compliance", "risk", "applications"],
            "apptio": ["cost", "financial", "cloud", "billing", "optimization", "analysis"]
        }
        
        # Check schema name for domain indicators
        schema_lower = schema_name.lower()
        for domain, domain_keywords in domain_mappings.items():
            if domain in schema_lower:
                keywords.update(domain_keywords)
        
        # Extract keywords from content
        content_lower = content.lower()
        technical_keywords = [
            "data", "api", "service", "endpoint", "query", "mutation", "response",
            "request", "entity", "object", "field", "type", "interface", "schema"
        ]
        
        for keyword in technical_keywords:
            if keyword in content_lower:
                keywords.add(keyword)
        
        return list(keywords)
    
    def _generate_functionality_description(self, schema_name: str, types: List[str], fields: List[str]) -> str:
        """Generate a description of what this schema likely does."""
        schema_lower = schema_name.lower()
        
        # Generate description based on common patterns
        if "entity" in schema_lower or "entities" in schema_lower:
            return "Provides entity management and data retrieval capabilities"
        elif "vulnerability" in schema_lower or "security" in schema_lower:
            return "Handles security vulnerabilities and risk assessment data"
        elif "topology" in schema_lower or "infrastructure" in schema_lower:
            return "Manages infrastructure topology and monitoring information"
        elif "application" in schema_lower:
            return "Manages application data and performance metrics"
        elif "stats" in schema_lower or "statistics" in schema_lower:
            return "Provides statistical data and performance metrics"
        else:
            return f"Provides data management for {self._make_readable(schema_name)} operations"
    
    def index_graphql_schemas(self, schema_dir: str, force_reindex: bool = False) -> int:
        """
        Index all GraphQL schema files in the specified directory.
        
        Args:
            schema_dir: Directory containing GraphQL schema files
            force_reindex: Whether to clear and recreate the index
            
        Returns:
            Number of schemas successfully indexed
        """
        schema_path = Path(schema_dir)
        
        if not schema_path.exists():
            logger.error(f"Schema directory not found: {schema_path}")
            return 0
        
        # Find GraphQL schema files
        schema_files = []
        for ext in ['*.graphql', '*.gql', '*.schema']:
            schema_files.extend(schema_path.rglob(ext))
        
        logger.info(f"Found {len(schema_files)} GraphQL schema files in {schema_path}")
        
        if not schema_files:
            logger.warning("No GraphQL schema files found (.graphql, .gql, .schema)")
            return 0
        
        # Clear collection if force reindex
        if force_reindex:
            logger.info("Force reindex: clearing existing collection")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "GraphQL schemas for schema mapping"}
            )
        
        # Check existing documents
        existing_ids = set()
        try:
            existing_data = self.collection.get()
            existing_ids = set(existing_data.get("ids", []))
            logger.info(f"Found {len(existing_ids)} existing schemas in collection")
        except Exception as e:
            logger.warning(f"Could not check existing documents: {e}")
        
        # Process each schema file
        indexed_count = 0
        skipped_count = 0
        
        for schema_file in schema_files:
            try:
                # Create document ID from schema name
                doc_id = schema_file.stem
                
                # Skip if already indexed (unless force reindex)
                if not force_reindex and doc_id in existing_ids:
                    skipped_count += 1
                    continue
                
                # Parse schema file
                parsed_schema = self.parse_graphql_schema(schema_file)
                
                if not parsed_schema:
                    logger.warning(f"Skipping empty/invalid schema: {schema_file}")
                    continue
                
                # Prepare document for ChromaDB
                document_text = parsed_schema["searchable_content"]
                metadata = {
                    "schema_name": parsed_schema["schema_name"],
                    "file_path": parsed_schema["file_path"],
                    "num_types": len(parsed_schema["types"]),
                    "num_fields": len(parsed_schema["fields"]),
                    "file_size": parsed_schema["file_size"],
                    "indexed_at": datetime.now().isoformat()
                }
                
                # Add to ChromaDB
                self.collection.add(
                    documents=[document_text],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
                
                indexed_count += 1
                
                if indexed_count % 10 == 0:
                    logger.info(f"   Indexed {indexed_count}/{len(schema_files)} schemas...")
                
            except Exception as e:
                logger.error(f"Error indexing {schema_file}: {e}")
                continue
        
        logger.info(f"Indexing complete:")
        logger.info(f"   Indexed: {indexed_count} schemas")
        logger.info(f"   Skipped: {skipped_count} schemas (already indexed)")
        logger.info(f"   Total in collection: {len(existing_ids) + indexed_count}")
        
        return indexed_count
    
    def search_schemas(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant schemas using semantic similarity.
        
        Args:
            query: Natural language query
            n_results: Number of results to return
            
        Returns:
            List of matching schema information
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
                    "schema_name": results["ids"][0][i],
                    "similarity_score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "metadata": results["metadatas"][0][i],
                    "description": results["documents"][0][i]
                }
                matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching schemas: {e}")
            return []


def main():
    """Main entry point for the GraphQL schema indexer."""
    parser = argparse.ArgumentParser(description="Index GraphQL schemas into ChromaDB")
    parser.add_argument("--schema-dir", required=True, help="Directory containing GraphQL schema files")
    parser.add_argument("--chroma-db-path", default=None, help="Path to store ChromaDB data")
    parser.add_argument("--collection-name", default=None, help="ChromaDB collection name")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindex (clear existing)")
    parser.add_argument("--test-query", help="Test query to search after indexing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"GraphQL Schema ChromaDB Indexer")
    logger.info(f"Schema directory: {args.schema_dir}")
    logger.info(f"ChromaDB path: {args.chroma_db_path}")
    logger.info(f"Collection name: {args.collection_name}")
    
    # Initialize indexer
    indexer = GraphQLSchemaIndexer(
        chroma_db_path=args.chroma_db_path,
        collection_name=args.collection_name
    )
    
    # Index schema files
    indexed_count = indexer.index_graphql_schemas(
        schema_dir=args.schema_dir,
        force_reindex=args.force_reindex
    )
    
    # Test search if query provided
    if args.test_query:
        print(f"\n🔍 Testing search with query: '{args.test_query}'")
        results = indexer.search_schemas(args.test_query, n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n   Result {i}:")
            print(f"   Schema: {result['schema_name']}")
            print(f"   Similarity: {result['similarity_score']:.3f}")
            print(f"   Types: {result['metadata'].get('num_types', 0)}")
            print(f"   Fields: {result['metadata'].get('num_fields', 0)}")
    
    print(f"\nIndexing completed successfully!")
    print(f"   Indexed {indexed_count} GraphQL schemas")
    print(f"   ChromaDB collection: {args.collection_name}")
    print(f"   Database location: {args.chroma_db_path}")


if __name__ == "__main__":
    main()
