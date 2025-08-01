#!/usr/bin/env python3
"""
Quick Fix for SMT GraphQL Schema Indexer
Minimal changes to existing indexer for better candidate quality
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
    IMPROVED GraphQL Schema Indexer with better semantic content.
    Drop-in replacement for the original with enhanced indexing.
    """
    
    def __init__(self, 
                 chroma_db_path: str = "../data/chroma_db",
                 collection_name: str = "graphql_schemas"):
        """Initialize the GraphQL schema indexer."""
        self.chroma_db_path = Path(chroma_db_path)
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = None
        try:
            self.client = chromadb.PersistentClient(path=str(self.chroma_db_path))
        except Exception as e:
            logger.warning(f"PersistentClient failed: {e}")

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Enhanced GraphQL schemas for schema mapping"}
            )
            logger.info(f"Created new collection: {collection_name}")
        
        # Initialize domain knowledge for better indexing
        self._init_domain_knowledge()
    
    def _init_domain_knowledge(self):
        """Initialize domain-specific knowledge."""
        self.domain_mappings = {
            "turbo": ["performance", "optimization", "virtual", "entity", "market", "statistics"],
            "instana": ["monitoring", "application", "performance", "infrastructure", "topology"],
            "concert": ["security", "vulnerability", "scanning", "applications", "compliance"],
            "apptio": ["cost", "financial", "cloud", "billing", "optimization"]
        }
        
        self.business_contexts = {
            "entity": "business data management",
            "application": "application lifecycle management", 
            "topology": "infrastructure mapping",
            "vulnerability": "security assessment",
            "group": "resource categorization",
            "metric": "performance monitoring"
        }
    
    def parse_graphql_schema(self, file_path: Path) -> Dict[str, Any]:
        """Enhanced GraphQL schema parsing."""
        try:
            with open(file_path, 'r') as f:
                schema_content = f.read()
            
            schema_name = file_path.stem
            
            # Enhanced parsing
            types = self._extract_graphql_types(schema_content)
            fields = self._extract_all_fields(schema_content)
            
            # Create enhanced searchable content
            searchable_content = self._create_enhanced_searchable_content(
                schema_name, types, fields, schema_content
            )
            
            parsed_schema = {
                "file_path": str(file_path),
                "schema_name": schema_name,
                "types": types,
                "fields": fields,
                "searchable_content": searchable_content,
                "raw_content": schema_content[:2000],
                "file_size": file_path.stat().st_size,
                "domain_context": self._analyze_domain_context(schema_name, schema_content)
            }
            
            return parsed_schema
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return {}
    
    def _extract_graphql_types(self, content: str) -> List[str]:
        """Extract GraphQL type names from schema content."""
        types = []
        
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
        
        return list(set(types))
    
    def _extract_all_fields(self, content: str) -> List[str]:
        """Extract all field names from GraphQL schema."""
        fields = []
        field_pattern = r'^\s*(\w+)(?:\([^)]*\))?\s*:\s*'
        
        lines = content.split('\n')
        in_type_definition = False
        
        for line in lines:
            line = line.strip()
            
            if re.match(r'(type|interface|input)\s+\w+', line, re.IGNORECASE):
                in_type_definition = True
                continue
            
            if line == '}':
                in_type_definition = False
                continue
            
            if in_type_definition:
                match = re.match(field_pattern, line)
                if match:
                    field_name = match.group(1)
                    if field_name not in ['query', 'mutation', 'subscription', 'schema']:
                        fields.append(field_name)
        
        return list(set(fields))
    
    def _create_enhanced_searchable_content(self, 
                                          schema_name: str, 
                                          types: List[str], 
                                          fields: List[str],
                                          raw_content: str) -> str:
        """Create enhanced searchable content with better semantic information."""
        content_parts = []
        
        # Enhanced schema name processing
        readable_name = self._make_readable(schema_name)
        content_parts.append(f"Schema: {readable_name}")
        
        # Add business context based on schema name
        business_context = self._get_business_context(schema_name)
        if business_context:
            content_parts.append(f"Business Context: {business_context}")
        
        # Enhanced type processing
        if types:
            # Filter out generic types and focus on business-relevant ones
            business_types = self._filter_business_relevant_types(types)
            if business_types:
                readable_types = [self._make_readable(t) for t in business_types]
                content_parts.append(f"Business Types: {', '.join(readable_types)}")
            
            # Add all types but with lower priority
            all_readable_types = [self._make_readable(t) for t in types]
            content_parts.append(f"All Types: {', '.join(all_readable_types)}")
        
        # Enhanced field processing
        if fields:
            # Prioritize important fields
            important_fields = self._filter_important_fields(fields)
            if important_fields:
                readable_fields = [self._make_readable(f) for f in important_fields]
                content_parts.append(f"Key Fields: {', '.join(readable_fields)}")
        
        # Add domain-specific keywords
        domain_keywords = self._extract_enhanced_domain_keywords(schema_name, raw_content)
        if domain_keywords:
            content_parts.append(f"Domain Keywords: {', '.join(domain_keywords)}")
        
        # Add functionality description
        functionality = self._generate_enhanced_functionality_description(schema_name, types, fields)
        content_parts.append(f"Functionality: {functionality}")
        
        # Add use case scenarios
        use_cases = self._generate_use_cases(schema_name, types)
        content_parts.append(f"Use Cases: {use_cases}")
        
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
    
    def _analyze_domain_context(self, schema_name: str, content: str) -> Dict[str, Any]:
        """Analyze domain context for better matching."""
        schema_lower = schema_name.lower()
        content_lower = content.lower()
        
        detected_domains = []
        confidence_scores = {}
        
        for domain, keywords in self.domain_mappings.items():
            score = 0
            if domain in schema_lower:
                score += 0.8
            
            keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
            score += (keyword_matches / len(keywords)) * 0.5
            
            if score > 0.3:
                detected_domains.append(domain)
                confidence_scores[domain] = score
        
        primary_domain = max(confidence_scores.items(), key=lambda x: x[1])[0] if confidence_scores else "general"
        
        return {
            "primary_domain": primary_domain,
            "detected_domains": detected_domains,
            "confidence_scores": confidence_scores
        }
    
    def _get_business_context(self, schema_name: str) -> str:
        """Get business context from schema name."""
        schema_lower = schema_name.lower()
        
        for concept, context in self.business_contexts.items():
            if concept in schema_lower:
                return context
        
        # Check for domain-specific contexts
        for domain, keywords in self.domain_mappings.items():
            if domain in schema_lower:
                if domain == "turbo":
                    return "infrastructure performance optimization"
                elif domain == "instana":
                    return "application performance monitoring"
                elif domain == "concert":
                    return "security vulnerability management"
                elif domain == "apptio":
                    return "cloud cost management"
        
        return "data management operations"
    
    def _filter_business_relevant_types(self, types: List[str]) -> List[str]:
        """Filter types that are most relevant for business operations."""
        relevant_keywords = [
            'entity', 'application', 'user', 'group', 'policy', 'alert', 
            'metric', 'event', 'topology', 'vulnerability', 'config', 'service'
        ]
        
        business_types = []
        for type_name in types:
            type_lower = type_name.lower()
            if any(keyword in type_lower for keyword in relevant_keywords):
                business_types.append(type_name)
        
        return business_types[:10]  # Limit to top 10
    
    def _filter_important_fields(self, fields: List[str]) -> List[str]:
        """Filter fields that are most important for matching."""
        important_keywords = [
            'id', 'name', 'type', 'status', 'description', 'tags', 
            'properties', 'config', 'data', 'info', 'details'
        ]
        
        important_fields = []
        for field_name in fields:
            field_lower = field_name.lower()
            if any(keyword in field_lower for keyword in important_keywords):
                important_fields.append(field_name)
        
        return important_fields[:15]  # Limit to top 15
    
    def _extract_enhanced_domain_keywords(self, schema_name: str, content: str) -> List[str]:
        """Extract enhanced domain-specific keywords."""
        keywords = set()
        schema_lower = schema_name.lower()
        content_lower = content.lower()
        
        # Add domain-specific keywords
        for domain, domain_keywords in self.domain_mappings.items():
            if domain in schema_lower:
                keywords.update(domain_keywords[:4])  # Top 4 keywords
        
        # Add generic technical keywords if they appear in content
        technical_keywords = [
            "api", "service", "endpoint", "query", "data", "entity", 
            "object", "management", "monitoring", "performance"
        ]
        
        for keyword in technical_keywords:
            if keyword in content_lower:
                keywords.add(keyword)
        
        return list(keywords)[:8]  # Limit to 8 keywords
    
    def _generate_enhanced_functionality_description(self, schema_name: str, types: List[str], fields: List[str]) -> str:
        """Generate enhanced functionality description."""
        schema_lower = schema_name.lower()
        all_text = f"{schema_name} {' '.join(types)}".lower()
        
        if "vulnerability" in all_text or "security" in all_text:
            return "security vulnerability assessment and risk management"
        elif "topology" in all_text or "infrastructure" in all_text:
            return "infrastructure topology mapping and monitoring" 
        elif "application" in all_text:
            return "application performance monitoring and management"
        elif "entity" in all_text or "entities" in all_text:
            return "business entity data management and operations"
        elif "group" in all_text:
            return "resource grouping and categorization"
        elif "metric" in all_text or "stats" in all_text:
            return "performance metrics and statistical analysis"
        else:
            return f"data operations and management for {self._make_readable(schema_name)}"
    
    def _generate_use_cases(self, schema_name: str, types: List[str]) -> str:
        """Generate likely use cases."""
        schema_lower = schema_name.lower()
        types_text = ' '.join(types).lower()
        
        if "vulnerability" in schema_lower or "security" in types_text:
            return "vulnerability scanning, security assessment, compliance reporting"
        elif "topology" in schema_lower or "infrastructure" in types_text:
            return "infrastructure monitoring, dependency mapping, topology analysis"
        elif "application" in schema_lower or "app" in types_text:
            return "application monitoring, performance tracking, service management"
        elif "entity" in schema_lower or "entities" in types_text:
            return "data retrieval, entity management, business operations"
        else:
            return f"data querying, information retrieval, {schema_lower} operations"
    
    def index_graphql_schemas(self, schema_dir: str, force_reindex: bool = False) -> int:
        """Index GraphQL schemas with enhanced content."""
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
                metadata={"description": "Enhanced GraphQL schemas for schema mapping"}
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
                doc_id = schema_file.stem
                
                if not force_reindex and doc_id in existing_ids:
                    skipped_count += 1
                    continue
                
                # Parse with enhanced analysis
                parsed_schema = self.parse_graphql_schema(schema_file)
                
                if not parsed_schema:
                    logger.warning(f"Skipping empty/invalid schema: {schema_file}")
                    continue
                
                # Enhanced metadata
                metadata = {
                    "schema_name": parsed_schema["schema_name"],
                    "file_path": parsed_schema["file_path"],
                    "num_types": len(parsed_schema["types"]),
                    "num_fields": len(parsed_schema["fields"]),
                    "file_size": parsed_schema["file_size"],
                    "indexed_at": datetime.now().isoformat(),
                    "primary_domain": parsed_schema["domain_context"]["primary_domain"],
                    "domain_confidence": parsed_schema["domain_context"]["confidence_scores"].get(
                        parsed_schema["domain_context"]["primary_domain"], 0.0
                    )
                }
                
                # Add to ChromaDB with enhanced content
                self.collection.add(
                    documents=[parsed_schema["searchable_content"]],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
                
                indexed_count += 1
                
                if indexed_count % 10 == 0:
                    logger.info(f"   Indexed {indexed_count}/{len(schema_files)} schemas...")
                
            except Exception as e:
                logger.error(f"Error indexing {schema_file}: {e}")
                continue
        
        logger.info(f"Enhanced indexing complete:")
        logger.info(f"   Indexed: {indexed_count} schemas")
        logger.info(f"   Skipped: {skipped_count} schemas (already indexed)")
        logger.info(f"   Total in collection: {len(existing_ids) + indexed_count}")
        
        return indexed_count
    
    def search_schemas(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Enhanced schema search with better query processing."""
        try:
            # Enhance query with domain-specific terms
            enhanced_query = self._enhance_query(query)
            
            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=n_results * 2  # Get more results to filter
            )
            
            # Process and filter results
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
            
            # Sort by similarity and return top results
            matches.sort(key=lambda x: x["similarity_score"], reverse=True)
            return matches[:n_results]
            
        except Exception as e:
            logger.error(f"Error searching schemas: {e}")
            return []
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query with domain-specific terms."""
        query_lower = query.lower()
        enhanced_terms = [query]
        
        # Add domain expansions
        if "turbo" in query_lower:
            enhanced_terms.extend(["performance", "optimization", "virtual", "entity"])
        if "instana" in query_lower:
            enhanced_terms.extend(["monitoring", "application", "performance", "infrastructure"])
        if "concert" in query_lower:
            enhanced_terms.extend(["security", "vulnerability", "scanning", "applications"])
        
        # Add concept expansions
        if "entity" in query_lower:
            enhanced_terms.extend(["business", "data", "object", "resource"])
        if "application" in query_lower:
            enhanced_terms.extend(["service", "system", "software", "app"])
        if "group" in query_lower:
            enhanced_terms.extend(["collection", "category", "cluster"])
        
        return " ".join(enhanced_terms[:10])  # Limit to avoid over-expansion


def main():
    """Main entry point for the enhanced schema indexer."""
    parser = argparse.ArgumentParser(description="Enhanced GraphQL Schema Indexer")
    parser.add_argument("--schema-dir", required=True, help="Directory containing GraphQL schema files")
    parser.add_argument("--chroma-db-path", default="../data/chroma_db", help="Path to store ChromaDB data")
    parser.add_argument("--collection-name", default="graphql_schemas", help="ChromaDB collection name")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindex (clear existing)")
    parser.add_argument("--test-query", help="Test query to search after indexing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    logger.info(f"Enhanced GraphQL Schema ChromaDB Indexer")
    logger.info(f"Schema directory: {args.schema_dir}")
    logger.info(f"ChromaDB path: {args.chroma_db_path}")
    logger.info(f"Collection name: {args.collection_name}")
    
    # Initialize enhanced indexer
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
        print(f"\nTesting enhanced search with query: '{args.test_query}'")
        results = indexer.search_schemas(args.test_query, n_results=5)
        
        for i, result in enumerate(results, 1):
            print(f"\n   Result {i}:")
            print(f"   Schema: {result['schema_name']}")
            print(f"   Similarity: {result['similarity_score']:.3f}")
            print(f"   Domain: {result['metadata'].get('primary_domain', 'unknown')}")
            print(f"   Types: {result['metadata'].get('num_types', 0)}")
            print(f"   Fields: {result['metadata'].get('num_fields', 0)}")
    
    print(f"\nEnhanced indexing completed successfully!")
    print(f"   Indexed {indexed_count} GraphQL schemas")
    print(f"   ChromaDB collection: {args.collection_name}")
    print(f"   Database location: {args.chroma_db_path}")


if __name__ == "__main__":
    main()
