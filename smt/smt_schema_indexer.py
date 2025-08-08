#!/usr/bin/env python3
"""
SMT GraphQL Schema Indexer
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


class GraphQLSchemaIndexer:
    """
    GraphQL Schema Indexer for semantic search.
    Focuses on schema structure, types, fields, and comments.
    """
    
    def __init__(self, 
                 chroma_db_path: str = "./chroma_db",
                 collection_name: str = "graphql_schemas"):
        """Initialize the GraphQL schema indexer."""
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
                    "description": "GraphQL schemas for semantic search",
                    "created": datetime.now().isoformat()
                }
            )
            logger.info(f"Created new collection: {collection_name}")
        
        # Simple domain mappings for context
        self.domain_mappings = {
            "turbo": ["performance", "optimization", "virtual", "entity"],
            "instana": ["monitoring", "application", "performance", "infrastructure"],
            "concert": ["security", "vulnerability", "scanning", "compliance"],
        }
    
    def parse_graphql_schema(self, file_path: Path) -> Dict[str, Any]:
        """Parse GraphQL schema and extract structure."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                schema_content = f.read()
            
            schema_name = file_path.stem
            
            # Extract schema components
            types = self._extract_types(schema_content)
            fields = self._extract_fields(schema_content)
            comments = self._extract_comments(schema_content)
            
            # Create searchable content
            searchable_content = self._create_searchable_content(
                schema_name, types, fields, comments, schema_content
            )
            
            # Generate content hash for change detection
            content_hash = hashlib.md5(schema_content.encode()).hexdigest()
            
            parsed_schema = {
                "file_path": str(file_path),
                "schema_name": schema_name,
                "types": types,
                "fields": fields,
                "comments": comments,
                "searchable_content": searchable_content,
                "content_hash": content_hash,
                "file_size": file_path.stat().st_size,
                "domain": self._detect_domain(schema_name, schema_content)
            }
            
            return parsed_schema
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return {}
    
    def _extract_types(self, content: str) -> List[Dict[str, Any]]:
        """Extract GraphQL types from schema content."""
        types = []
        
        type_patterns = [
            (r'type\s+(\w+)\s*(?:implements\s+[\w\s,&]+)?\s*\{([^}]+)\}', 'type'),
            (r'interface\s+(\w+)\s*\{([^}]+)\}', 'interface'),
            (r'enum\s+(\w+)\s*\{([^}]+)\}', 'enum'),
            (r'input\s+(\w+)\s*\{([^}]+)\}', 'input'),
            (r'scalar\s+(\w+)', 'scalar'),
            (r'union\s+(\w+)\s*=\s*([^\n]+)', 'union')
        ]
        
        for pattern, type_kind in type_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                type_name = match.group(1)
                type_body = match.group(2) if len(match.groups()) > 1 else ""
                
                type_info = {
                    "name": type_name,
                    "kind": type_kind,
                    "readable_name": self._make_readable(type_name)
                }
                
                # Extract fields for complex types
                if type_kind in ['type', 'interface', 'input']:
                    type_info["fields"] = self._parse_type_fields(type_body)
                elif type_kind == 'enum':
                    type_info["values"] = self._parse_enum_values(type_body)
                
                types.append(type_info)
        
        return types
    
    def _parse_type_fields(self, type_body: str) -> List[Dict[str, str]]:
        """Parse fields within a type definition."""
        fields = []
        
        # Field pattern: fieldName(args): Type
        field_pattern = r'^\s*(\w+)(?:\([^)]*\))?\s*:\s*(.+?)(?:\s*@[^\n]*)?$'
        
        for line in type_body.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            match = re.match(field_pattern, line)
            if match:
                field_name = match.group(1)
                field_type = match.group(2).strip()
                
                fields.append({
                    "name": field_name,
                    "type": field_type,
                    "readable_name": self._make_readable(field_name)
                })
        
        return fields
    
    def _parse_enum_values(self, enum_body: str) -> List[str]:
        """Parse enum values."""
        values = []
        for line in enum_body.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract enum value (first word)
                value = line.split()[0]
                if value:
                    values.append(value)
        return values
    
    def _extract_fields(self, content: str) -> List[Dict[str, str]]:
        """Extract all fields from schema."""
        all_fields = []
        
        # Extract from all type definitions
        type_matches = re.finditer(
            r'(?:type|interface|input)\s+(\w+)\s*(?:implements\s+[\w\s,&]+)?\s*\{([^}]+)\}',
            content, re.IGNORECASE | re.DOTALL
        )
        
        for match in type_matches:
            type_name = match.group(1)
            type_body = match.group(2)
            
            fields = self._parse_type_fields(type_body)
            for field in fields:
                field["parent_type"] = type_name
                all_fields.append(field)
        
        return all_fields
    
    def _extract_comments(self, content: str) -> List[str]:
        """Extract comments from GraphQL schema."""
        comments = []
        
        # Single line comments
        comment_pattern = r'#\s*(.+)'
        for match in re.finditer(comment_pattern, content):
            comment_text = match.group(1).strip()
            if comment_text:
                comments.append(comment_text)
        
        # Multi-line string literals (often used as descriptions)
        description_pattern = r'"""([^"]+)"""'
        for match in re.finditer(description_pattern, content, re.DOTALL):
            description = match.group(1).strip()
            if description:
                comments.append(description)
        
        return comments
    
    def _create_searchable_content(self, schema_name: str, types: List[Dict], 
                                 fields: List[Dict], comments: List[str], 
                                 raw_content: str) -> str:
        """Create searchable content from schema components."""
        content_parts = []
        
        # Schema name
        readable_name = self._make_readable(schema_name)
        content_parts.append(f"Schema: {readable_name}")
        content_parts.append(f"Schema Name: {schema_name}")
        
        # Domain context
        domain = self._detect_domain(schema_name, raw_content)
        if domain != "general":
            content_parts.append(f"Domain: {domain}")
        
        # Types
        if types:
            type_names = []
            for type_info in types:
                type_names.append(f"{type_info['readable_name']} ({type_info['kind']})")
            content_parts.append(f"Types: {', '.join(type_names)}")
        
        if fields:
            field_descriptions = []
            for field in fields[:20]:  # Limit to first 20 fields
                field_desc = f"{field['readable_name']} in {self._make_readable(field['parent_type'])}"
                field_descriptions.append(field_desc)
            content_parts.append(f"Fields: {', '.join(field_descriptions)}")
        
        # Comments and descriptions
        if comments:
            all_comments = ' '.join(comments[:10])  # Limit to first 10 comments
            content_parts.append(f"Documentation: {all_comments}")
        
        # Query and Mutation operations
        query_ops = self._extract_operations(raw_content, "Query")
        if query_ops:
            content_parts.append(f"Query Operations: {', '.join(query_ops)}")
        
        mutation_ops = self._extract_operations(raw_content, "Mutation")
        if mutation_ops:
            content_parts.append(f"Mutation Operations: {', '.join(mutation_ops)}")
        
        return "\n".join(content_parts)
    
    def _extract_operations(self, content: str, operation_type: str) -> List[str]:
        """Extract operations from Query or Mutation types."""
        operations = []
        
        pattern = rf'type\s+{operation_type}\s*\{{([^}}]+)\}}'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        
        if match:
            operations_body = match.group(1)
            
            # Parse individual operations
            operation_pattern = r'^\s*(\w+)(?:\([^)]*\))?\s*:\s*(.+?)(?:\s*@[^\n]*)?$'
            
            for line in operations_body.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                op_match = re.match(operation_pattern, line)
                if op_match:
                    op_name = op_match.group(1)
                    operations.append(self._make_readable(op_name))
        
        return operations
    
    def _make_readable(self, text: str) -> str:
        """Convert technical names to readable format."""
        if not text:
            return ""
        
        # Handle camelCase and PascalCase
        readable = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Handle snake_case and kebab-case
        readable = readable.replace('_', ' ').replace('-', ' ')
        
        # Clean up multiple spaces
        readable = re.sub(r'\s+', ' ', readable).strip().lower()
        
        return readable
    
    def _detect_domain(self, schema_name: str, content: str) -> str:
        """Detect domain from schema name and content."""
        schema_lower = schema_name.lower()
        content_lower = content.lower()
        
        # Check domain mappings
        for domain, keywords in self.domain_mappings.items():
            if domain in schema_lower:
                return domain
            
            # Check for domain keywords in content
            keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
            if keyword_matches >= 2:
                return domain
        
        return "general"
    
    def index_graphql_schemas(self, schema_dir: str, force_reindex: bool = False) -> int:
        """Index GraphQL schemas."""
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
            try:
                self.client.delete_collection(self.collection_name)
            except:
                pass
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "GraphQL schemas for semantic search",
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
            logger.info(f"Found {len(existing_data)} existing schemas in collection")
        except Exception as e:
            logger.warning(f"Could not check existing documents: {e}")
        
        # Process each schema file
        indexed_count = 0
        skipped_count = 0
        updated_count = 0
        
        for schema_file in schema_files:
            try:
                doc_id = schema_file.stem
                
                # Parse schema
                parsed_schema = self.parse_graphql_schema(schema_file)
                
                if not parsed_schema:
                    logger.warning(f"Skipping empty/invalid schema: {schema_file}")
                    continue
                
                # Check if update needed
                current_hash = parsed_schema["content_hash"]
                existing_hash = existing_data.get(doc_id, "")
                
                if not force_reindex and doc_id in existing_data and current_hash == existing_hash:
                    skipped_count += 1
                    continue
                
                # Prepare metadata
                metadata = {
                    "schema_name": parsed_schema["schema_name"],
                    "file_path": parsed_schema["file_path"],
                    "content_hash": current_hash,
                    "num_types": len(parsed_schema["types"]),
                    "num_fields": len(parsed_schema["fields"]),
                    "num_comments": len(parsed_schema["comments"]),
                    "file_size": parsed_schema["file_size"],
                    "domain": parsed_schema["domain"],
                    "indexed_at": datetime.now().isoformat()
                }
                
                # Add or update in ChromaDB
                if doc_id in existing_data:
                    # Update existing
                    self.collection.update(
                        documents=[parsed_schema["searchable_content"]],
                        metadatas=[metadata],
                        ids=[doc_id]
                    )
                    updated_count += 1
                else:
                    # Add new
                    self.collection.add(
                        documents=[parsed_schema["searchable_content"]],
                        metadatas=[metadata],
                        ids=[doc_id]
                    )
                    indexed_count += 1
                
                if (indexed_count + updated_count) % 10 == 0:
                    logger.info(f"   Processed {indexed_count + updated_count}/{len(schema_files)} schemas...")
                
            except Exception as e:
                logger.error(f"Error processing {schema_file}: {e}")
                continue
        
        total_processed = indexed_count + updated_count
        logger.info(f"Indexing complete:")
        logger.info(f"   New schemas indexed: {indexed_count}")
        logger.info(f"   Schemas updated: {updated_count}")
        logger.info(f"   Skipped (unchanged): {skipped_count}")
        logger.info(f"   Total in collection: {len(existing_data) + indexed_count}")
        
        return total_processed
    
    def search_schemas(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search schemas with the given query."""
        try:
            # Simple query enhancement
            enhanced_query = self._enhance_query(query)
            
            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=n_results
            )
            
            # Process results
            matches = []
            for i in range(len(results.get("ids", [[]])[0])):
                similarity_score = 1 - results["distances"][0][i]
                
                # Apply minimum threshold
                if similarity_score < 0.2:
                    continue
                
                match = {
                    "schema_name": results["ids"][0][i],
                    "similarity_score": similarity_score,
                    "metadata": results["metadatas"][0][i],
                    "description": results["documents"][0][i][:200] + "..." if len(results["documents"][0][i]) > 200 else results["documents"][0][i]
                }
                matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching schemas: {e}")
            return []
    
    def _enhance_query(self, query: str) -> str:
        """Simple query enhancement with domain keywords."""
        query_lower = query.lower()
        enhanced_terms = [query]
        
        # Add domain-specific terms if detected
        for domain, keywords in self.domain_mappings.items():
            if domain in query_lower:
                enhanced_terms.extend(keywords[:3])  # Add top 3 keywords
        
        return " ".join(enhanced_terms)


def main():
    """Main entry point for the schema indexer."""
    parser = argparse.ArgumentParser(description="GraphQL Schema Indexer")
    parser.add_argument("--schema-dir", required=True, help="Directory containing GraphQL schema files")
    parser.add_argument("--chroma-db-path", default="./chroma_db", help="Path to store ChromaDB data")
    parser.add_argument("--collection-name", default="graphql_schemas", help="ChromaDB collection name")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindex (clear existing)")
    parser.add_argument("--test-query", help="Test query to search after indexing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
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
    processed_count = indexer.index_graphql_schemas(
        schema_dir=args.schema_dir,
        force_reindex=args.force_reindex
    )
    
    # Test search if query provided
    if args.test_query:
        print(f"\nTesting search with query: '{args.test_query}'")
        results = indexer.search_schemas(args.test_query, n_results=5)
        
        for i, result in enumerate(results, 1):
            print(f"\n   Result {i}:")
            print(f"   Schema: {result['schema_name']}")
            print(f"   Similarity: {result['similarity_score']:.3f}")
            print(f"   Domain: {result['metadata'].get('domain', 'unknown')}")
            print(f"   Types: {result['metadata'].get('num_types', 0)}")
            print(f"   Fields: {result['metadata'].get('num_fields', 0)}")
    
    print(f"\nIndexing completed successfully!")
    print(f"   Processed {processed_count} GraphQL schemas")
    print(f"   ChromaDB collection: {args.collection_name}")
    print(f"   Database location: {args.chroma_db_path}")


if __name__ == "__main__":
    main()
