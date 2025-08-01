#!/usr/bin/env python3
"""
SMT API Retriever
Handles API discovery using ChromaDB for schemas and existing mapping lookup
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from smt_schema_indexer import GraphQLSchemaIndexer
    from smt_mappings_indexer import SMTMappingsIndexer
except ImportError as e:
    logger.error(f"Cannot import ChromaDB indexers: {e}")
    # Create mock classes for testing
    class GraphQLSchemaIndexer:
        def __init__(self, *args, **kwargs):
            pass
        def search_schemas(self, query, n_results=5):
            return []
    
    class SMTMappingsIndexer:
        def __init__(self, *args, **kwargs):
            pass
        def search_existing_mappings(self, query, n_results=5):
            return []


class SMTAPIRetriever:
    """
    Enhanced API Retriever for SMT schema mapping with two-tier ChromaDB system:
    1. Schema ChromaDB: For discovering candidate schemas
    2. Mappings ChromaDB: For finding existing mappings to reuse
    """
    
    def __init__(self, 
                 dataset: str = None,
                 mappings_dir: str = None, 
                 chroma_db_path: str = "../data/chroma_db",
                 collection_name:  str= None):
        """
        Initialize the API retriever with two ChromaDB instances.
        
        Args:
            dataset: Dataset name for collection naming
            mappings_dir: Directory containing existing mapping files
            chroma_db_path: Path to ChromaDB database
            collection_name: Explicit collection name for schemas
        """
        self.mappings_dir = Path(mappings_dir) if mappings_dir else Path("../data/mappings_output")
        self.chroma_db_path = Path(chroma_db_path)
        self.collection_name = collection_name
        self.dataset = dataset

        # Determine collection names
        if collection_name:
            schema_collection_name = collection_name
        else:
            schema_collection_name = f"{dataset.lower()}_graphql_schemas" if dataset else "graphql_schemas"

        # For mappings, use shared collection (not dataset-specific for now)
        mappings_collection_name = "smt_mappings"
        
        # Initialize Schema ChromaDB indexer for schema discovery
        try:
            self.schema_indexer = GraphQLSchemaIndexer(
                chroma_db_path=str(chroma_db_path),
                collection_name=schema_collection_name
            )
            logger.info(f"Connected to Schema ChromaDB collection: {schema_collection_name}")
        except Exception as e:
            logger.warning(f"Could not connect to Schema ChromaDB: {e}")
            self.schema_indexer = None
        
        # Initialize Mappings ChromaDB indexer for existing mapping lookup
        try:
            self.mappings_indexer = SMTMappingsIndexer(
                chroma_db_path=str(chroma_db_path),
                collection_name=mappings_collection_name
            )
            logger.info(f"Connected to Mappings ChromaDB collection: {mappings_collection_name}")
        except Exception as e:
            logger.warning(f"Could not connect to Mappings ChromaDB: {e}")
            self.mappings_indexer = None
        
        # Load existing mapping files as fallback
        self.existing_mappings = self._load_existing_mappings()
        logger.info(f"Loaded {len(self.existing_mappings)} existing mapping files")
        
        # Build API name index from existing mappings
        self.api_names = self._extract_api_names()
        logger.info(f"Indexed {len(self.api_names)} unique API names")
    
    def update_schema_collection(self, collection_name: str):
        """Update the schema collection name for dataset-specific indexing."""
        self.collection_name = collection_name
        if self.schema_indexer:
            self.schema_indexer.collection_name = collection_name
            logger.info(f"Updated schema collection to: {collection_name}")
    
    def _load_existing_mappings(self) -> Dict[str, str]:
        """Load all existing mapping files and create lookup index."""
        mappings = {}
        
        # Look for mapping files in format: {source_api}-{target_api}.json
        for mapping_file in self.mappings_dir.glob("*-*.json"):
            try:
                # Extract source and target API names from filename
                filename = mapping_file.stem  # Remove .json extension
                if "-" in filename:
                    source_api, target_api = filename.split("-", 1)
                    mapping_key = f"{source_api}-{target_api}"
                    mappings[mapping_key] = str(mapping_file)
                    
                    # Also add reverse mapping possibility
                    reverse_key = f"{target_api}-{source_api}"
                    if reverse_key not in mappings:
                        mappings[reverse_key] = str(mapping_file)
                        
            except Exception as e:
                logger.warning(f"Could not process mapping file {mapping_file}: {e}")
        
        return mappings
    
    def _extract_api_names(self) -> List[str]:
        """Extract unique API names from existing mapping files."""
        api_names = set()
        
        for mapping_key in self.existing_mappings.keys():
            if "-" in mapping_key:
                source_api, target_api = mapping_key.split("-", 1)
                api_names.add(source_api)
                api_names.add(target_api)
        
        return list(api_names)
    
    def find_existing_mapping_for_utterance(self, utterance: str, top_k: int = 3) -> Optional[Dict[str, Any]]:
        """
        PRIORITY: Try to find existing mapping that matches the utterance.
        This is the main efficiency gain - reuse existing mappings.
        
        Args:
            utterance: Natural language description of mapping task
            top_k: Number of top candidates to consider
            
        Returns:
            Best existing mapping match or None
        """
        logger.info(f"Searching for existing mapping for: '{utterance}'")
        
        if not self.mappings_indexer:
            logger.warning("Mappings ChromaDB not available, checking file-based lookup")
            return self._fallback_existing_mapping_lookup(utterance)
        
        try:
            # Search existing mappings using semantic similarity
            mapping_results = self.mappings_indexer.search_existing_mappings(utterance, n_results=top_k)
            
            if not mapping_results:
                logger.info("No existing mappings found for this utterance")
                return None
            
            # Take the best match (highest similarity)
            best_match = mapping_results[0]
            similarity_score = best_match.get("similarity_score", 0.0)
            
            # Set a threshold for using existing mappings
            SIMILARITY_THRESHOLD = 0.6
            
            if similarity_score >= SIMILARITY_THRESHOLD:
                logger.info(f"Found existing mapping: {best_match.get('mapping_id', 'unknown')} (similarity: {similarity_score:.3f})")
                
                # Load the actual mapping file
                file_path = best_match.get("file_path", "")
                try:
                    with open(file_path, 'r') as f:
                        mapping_data = json.load(f)
                    
                    return {
                        "mapping_data": mapping_data,
                        "source_api": best_match.get("metadata", {}).get("source_api", ""),
                        "target_api": best_match.get("metadata", {}).get("target_api", ""),
                        "similarity_score": similarity_score,
                        "file_path": file_path,
                        "reused": True
                    }
                    
                except Exception as e:
                    logger.error(f"Could not load mapping file {file_path}: {e}")
                    return None
            else:
                logger.info(f"Best existing mapping similarity too low: {similarity_score:.3f} < {SIMILARITY_THRESHOLD}")
                return None
        
        except Exception as e:
            logger.error(f"Error searching existing mappings: {e}")
            return self._fallback_existing_mapping_lookup(utterance)
    
    def _fallback_existing_mapping_lookup(self, utterance: str) -> Optional[Dict[str, Any]]:
        """Fallback method when ChromaDB is not available."""
        # Parse utterance to find API hints
        source_hints, target_hints = self._parse_utterance_for_apis(utterance)
        
        # Try to find existing mapping by API names
        for source_hint in source_hints:
            for target_hint in target_hints:
                existing_mapping = self.get_existing_mapping(source_hint, target_hint)
                if existing_mapping:
                    return {
                        "mapping_data": existing_mapping,
                        "source_api": source_hint,
                        "target_api": target_hint,
                        "similarity_score": 0.5,  # Default score for fallback
                        "file_path": self.existing_mappings.get(f"{source_hint}-{target_hint}", ""),
                        "reused": True
                    }
        
        return None
    
    def find_apis_for_utterance(self, utterance: str, top_k: int = 5) -> Tuple[List[str], List[str]]:
        """
        Find candidate source and target APIs for a natural language utterance.
        This is used when no existing mapping is found.
        
        Args:
            utterance: Natural language description like "map concert vulnerabilities to turbo entities"
            top_k: Number of top candidate APIs to return
            
        Returns:
            Tuple of (source_candidates, target_candidates)
        """
        logger.info(f"Finding new API candidates for: '{utterance}'")
        
        # Parse utterance to identify mapping direction and API hints
        source_hints, target_hints = self._parse_utterance_for_apis(utterance)
        
        # Use Schema ChromaDB to find relevant schemas
        if self.schema_indexer:
            source_candidates = self._find_schemas_with_chromadb(source_hints, top_k)
            target_candidates = self._find_schemas_with_chromadb(target_hints, top_k)
        else:
            # Fallback to existing API names
            source_candidates = self._find_similar_apis_fallback(source_hints, top_k)
            target_candidates = self._find_similar_apis_fallback(target_hints, top_k)
        
        logger.info(f"   Source candidates: {source_candidates}")
        logger.info(f"   Target candidates: {target_candidates}")
        
        return source_candidates, target_candidates
    
    def _parse_utterance_for_apis(self, utterance: str) -> Tuple[List[str], List[str]]:
        """
        Parse utterance to extract API hints.
        
        Examples:
        - "map concert vulnerabilities to turbo entities" → (['concert'], ['turbo'])
        - "map turbo application metrics to instana performance" → (['turbo'], ['instana'])
        """
        utterance_lower = utterance.lower()
        
        # Common patterns for mapping utterances
        map_patterns = [
            r"map\s+(\w+)\s+.*?to\s+(\w+)",
            r"map\s+(\w+)\s+.*?\s+to\s+(\w+)",
            r"from\s+(\w+)\s+.*?to\s+(\w+)",
            r"(\w+)\s+.*?to\s+(\w+)\s+.*?mapping"
        ]
        
        source_hints = []
        target_hints = []
        
        # Try to extract using patterns
        for pattern in map_patterns:
            match = re.search(pattern, utterance_lower)
            if match:
                source_hints.append(match.group(1))
                target_hints.append(match.group(2))
                break
        
        # If no pattern match, extract potential API names from the utterance
        if not source_hints or not target_hints:
            # Look for known API names in the utterance
            found_apis = []
            for api_name in self.api_names:
                if api_name.lower() in utterance_lower:
                    found_apis.append(api_name)
            
            # If we found APIs, try to determine direction
            if len(found_apis) >= 2:
                # Simple heuristic: first mentioned is source, last is target
                source_hints = [found_apis[0]]
                target_hints = [found_apis[-1]]
            elif len(found_apis) == 1:
                # If only one API found, use it for both and let similarity search find the other
                source_hints = [found_apis[0]]
                target_hints = [found_apis[0]]
        
        # Fallback: extract all words as potential API hints
        if not source_hints and not target_hints:
            words = re.findall(r'\b\w+\b', utterance_lower)
            # Filter out common words
            common_words = {'map', 'to', 'from', 'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'as'}
            api_words = [w for w in words if w not in common_words and len(w) > 2]
            source_hints = api_words[:3]  # Take first few as source hints
            target_hints = api_words[-3:]  # Take last few as target hints
        
        return source_hints, target_hints
    
    def _find_schemas_with_chromadb(self, hints: List[str], top_k: int = 5) -> List[str]:
        """Find similar schemas using Schema ChromaDB semantic search."""
        if not hints or not self.schema_indexer:
            return []
        
        # Combine hints into a query
        query = " ".join(hints)
        
        try:
            # Search for relevant schemas
            schema_results = self.schema_indexer.search_schemas(query, n_results=top_k)
            
            # Extract schema names
            schema_names = [result.get("schema_name", "") for result in schema_results]
            
            return schema_names
            
        except Exception as e:
            logger.warning(f"Schema ChromaDB search failed: {e}")
            return self._find_similar_apis_fallback(hints, top_k)
    
    def _find_similar_apis_fallback(self, hints: List[str], top_k: int = 5) -> List[str]:
        """Fallback string matching when ChromaDB fails."""
        matches = []
        
        for api_name in self.api_names:
            api_lower = api_name.lower()
            for hint in hints:
                hint_lower = hint.lower()
                if hint_lower in api_lower or api_lower in hint_lower:
                    matches.append(api_name)
                    break
        
        return matches[:top_k]
    
    def get_existing_mapping(self, source_api: str, target_api: str) -> Optional[Dict[str, Any]]:
        """
        Check if a mapping file exists for the given API pair (file-based lookup).
        
        Args:
            source_api: Source API name
            target_api: Target API name
            
        Returns:
            Mapping data if file exists, None otherwise
        """
        mapping_key = f"{source_api}-{target_api}"
        reverse_key = f"{target_api}-{source_api}"
        
        # Check both directions
        mapping_file = None
        if mapping_key in self.existing_mappings:
            mapping_file = self.existing_mappings[mapping_key]
        elif reverse_key in self.existing_mappings:
            mapping_file = self.existing_mappings[reverse_key]
        
        if mapping_file:
            try:
                with open(mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                
                logger.info(f"Found existing mapping file: {Path(mapping_file).name}")
                return mapping_data
                
            except Exception as e:
                logger.error(f"Error loading mapping file {mapping_file}: {e}")
                return None
        
        return None
    
    def find_best_api_pair(self, utterance: str) -> Tuple[str, str, float]:
        """
        Find the best source-target API pair for an utterance.
        PRIORITY: First check existing mappings, then find new candidates.
        
        Args:
            utterance: Natural language mapping request
            
        Returns:
            Tuple of (source_api, target_api, confidence_score)
        """
        logger.info(f"Finding best API pair for: '{utterance}'")
        
        # STEP 1: Try to find existing mapping first (most efficient)
        existing_mapping = self.find_existing_mapping_for_utterance(utterance)
        
        if existing_mapping:
            source_api = existing_mapping["source_api"]
            target_api = existing_mapping["target_api"]
            confidence = existing_mapping["similarity_score"]
            
            logger.info(f"   Using existing mapping: {source_api} → {target_api} (confidence: {confidence:.3f})")
            return source_api, target_api, confidence
        
        # STEP 2: If no existing mapping, find new API candidates
        logger.info("   No suitable existing mapping found, searching for new candidates...")
        
        source_candidates, target_candidates = self.find_apis_for_utterance(utterance)
        
        if not source_candidates or not target_candidates:
            logger.warning("No API candidates found")
            # Return default APIs if available
            if self.api_names:
                return self.api_names[0], self.api_names[-1] if len(self.api_names) > 1 else self.api_names[0], 0.1
            return "default_source", "default_target", 0.0
        
        # Find the best combination based on similarity
        best_score = 0.0
        best_source = source_candidates[0]
        best_target = target_candidates[0]
        
        for source in source_candidates[:3]:  # Check top 3 sources
            for target in target_candidates[:3]:  # Check top 3 targets
                score = self._calculate_pair_score(source, target, utterance)
                if score > best_score:
                    best_score = score
                    best_source = source
                    best_target = target
        
        logger.info(f"   New candidates: {best_source} → {best_target} (score: {best_score:.3f})")
        return best_source, best_target, best_score
    
    def _calculate_pair_score(self, source_api: str, target_api: str, utterance: str) -> float:
        """Calculate confidence score for an API pair."""
        score = 0.0
        
        # Bonus for existing mapping file
        if self.get_existing_mapping(source_api, target_api):
            score += 0.5
        
        # Bonus for API names appearing in utterance
        utterance_lower = utterance.lower()
        if source_api.lower() in utterance_lower:
            score += 0.3
        if target_api.lower() in utterance_lower:
            score += 0.3
        
        # Basic similarity bonus
        source_readable = self._make_api_name_readable(source_api)
        target_readable = self._make_api_name_readable(target_api)
        
        utterance_words = set(utterance_lower.split())
        source_words = set(source_readable.split())
        target_words = set(target_readable.split())
        
        source_overlap = len(utterance_words.intersection(source_words))
        target_overlap = len(utterance_words.intersection(target_words))
        
        if source_overlap > 0:
            score += 0.1 * source_overlap
        if target_overlap > 0:
            score += 0.1 * target_overlap
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _make_api_name_readable(self, api_name: str) -> str:
        """Convert API name to readable format."""
        # Handle camelCase and snake_case
        readable = re.sub(r'([a-z])([A-Z])', r'\1 \2', api_name)  # camelCase
        readable = readable.replace('_', ' ')  # snake_case
        return readable.lower()
    
    def list_available_apis(self) -> List[str]:
        """List all available API names."""
        return sorted(self.api_names)
    
    def list_existing_mappings(self) -> List[str]:
        """List all existing mapping file pairs."""
        return sorted(self.existing_mappings.keys())
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever's capabilities."""
        stats = {
            "existing_mapping_files": len(self.existing_mappings),
            "unique_api_names": len(self.api_names),
            "schema_chromadb_available": self.schema_indexer is not None,
            "mappings_chromadb_available": self.mappings_indexer is not None,
            "schema_collection_name": self.collection_name,
            "dataset": self.dataset
        }
        
        if self.mappings_indexer and hasattr(self.mappings_indexer, 'get_collection_stats'):
            try:
                mapping_stats = self.mappings_indexer.get_collection_stats()
                stats.update({
                    "indexed_mappings": mapping_stats.get("total_mappings", 0),
                    "sample_api_pairs": mapping_stats.get("sample_api_pairs", 0)
                })
            except Exception as e:
                logger.warning(f"Could not get mapping stats: {e}")
        
        return stats

    def save_schemas_for_smt(self, source_api: str, target_api: str, output_dir: str) -> bool:
        """
        Generate and save source.json and target.json files for SMT toolkit.
        
        Args:
            source_api: Source API name
            target_api: Target API name  
            output_dir: Directory to save schema files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Generate mock schemas for now
            # In a real implementation, this would load actual schema files
            source_schema = {
                "type": "object",
                "title": source_api.title() + " Schema",
                "properties": {
                    f"{source_api}_id": {
                        "type": "string",
                        "description": f"Unique identifier for {source_api}"
                    },
                    f"{source_api}_name": {
                        "type": "string", 
                        "description": f"Name of the {source_api}"
                    },
                    f"{source_api}_data": {
                        "type": "object",
                        "description": f"Additional data for {source_api}",
                        "properties": {
                            "created_at": {"type": "string", "format": "date-time"},
                            "updated_at": {"type": "string", "format": "date-time"},
                            "status": {"type": "string", "enum": ["active", "inactive"]}
                        }
                    }
                },
                "required": [f"{source_api}_id", f"{source_api}_name"]
            }
            
            target_schema = {
                "type": "object", 
                "title": target_api.title() + " Schema",
                "properties": {
                    f"{target_api}_identifier": {
                        "type": "string",
                        "description": f"Unique ID for {target_api}"
                    },
                    f"{target_api}_title": {
                        "type": "string",
                        "description": f"Title of the {target_api}"
                    },
                    f"{target_api}_metadata": {
                        "type": "object", 
                        "description": f"Metadata for {target_api}",
                        "properties": {
                            "timestamp": {"type": "string", "format": "date-time"},
                            "last_modified": {"type": "string", "format": "date-time"},
                            "state": {"type": "string", "enum": ["enabled", "disabled"]}
                        }
                    }
                },
                "required": [f"{target_api}_identifier", f"{target_api}_title"]
            }
            
            # Save schemas
            source_file = output_path / "source.json"
            target_file = output_path / "target.json"
            
            with open(source_file, 'w') as f:
                json.dump(source_schema, f, indent=2)
                
            with open(target_file, 'w') as f:
                json.dump(target_schema, f, indent=2)
            
            logger.info(f"Generated schema files:")
            logger.info(f"  Source: {source_file}")
            logger.info(f"  Target: {target_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save schemas: {e}")
            return False


def main():
    """Main entry point for SMT API Retriever."""
    parser = argparse.ArgumentParser(description="SMT API Retriever")
    parser.add_argument("--utterance", required=True, help="Natural language utterance")
    parser.add_argument("--dataset", help="Target dataset")
    parser.add_argument("--model", help="Target model")
    parser.add_argument("--collection-name", help="ChromaDB schema collection name")
    parser.add_argument("--chroma-db-path", default="../data/chroma_db", help="ChromaDB database path")
    parser.add_argument("--mappings-dir", default="../data/mappings_output", help="Mappings directory")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results")
    parser.add_argument("--output-dir", default="mappings_output", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Determine collection name
    if args.collection_name:
        collection_name = args.collection_name
    elif args.dataset:
        collection_name = f"{args.dataset.lower()}_graphql_schemas"
    
    # Initialize retriever
    retriever = SMTAPIRetriever(
        dataset=args.dataset,
        mappings_dir=args.mappings_dir,
        chroma_db_path=args.chroma_db_path,
        collection_name=collection_name
    )
    
    print(f"SMT API Retriever")
    print(f"Utterance: {args.utterance}")
    print(f"Dataset: {args.dataset}")
    print(f"Schema Collection: {collection_name}")
    print(f"ChromaDB Path: {args.chroma_db_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Show retriever stats
    stats = retriever.get_retriever_stats()
    print(f"\nRetriever Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Find best API pair
    source_api, target_api, confidence = retriever.find_best_api_pair(args.utterance)
    
    print(f"\nSelected API pair: {source_api} → {target_api} (confidence: {confidence:.3f})")
    
    # Generate and save schemas
    success = retriever.save_schemas_for_smt(source_api, target_api, args.output_dir)
    
    if success:
        print(f"\nSchema files generated successfully in {args.output_dir}/")
        print(f"Next steps:")
        print(f"  1. Review the generated source.json and target.json files")
        print(f"  2. Run the SMT toolkit with: make run-smt")
        print(f"  3. Check results in results/ directory")
    else:
        print(f"Failed to generate schema files")
        exit(1)


if __name__ == "__main__":
    main()
