#!/usr/bin/env python3
"""
SMT API Retriever
Handles API discovery, mapping lookup, and schema retrieval for SMT evaluation
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SMTAPIRetriever:
    """
    API Retriever for SMT schema mapping.
    Handles vector-based API discovery and existing mapping lookup.
    """
    
    def __init__(self, 
                 mappings_dir: str = ".", 
                 vector_db_config: Dict[str, Any] = None,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the API retriever.
        
        Args:
            mappings_dir: Directory containing existing mapping files
            vector_db_config: Configuration for vector database (if available)
            embedding_model: Sentence transformer model for similarity
        """
        self.mappings_dir = Path(mappings_dir)
        self.vector_db_config = vector_db_config or {}
        
        # Initialize sentence transformer for API name similarity
        try:
            self.sentence_transformer = SentenceTransformer(embedding_model)
            logger.info(f"✅ Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            self.sentence_transformer = None
        
        # Load existing mapping files
        self.existing_mappings = self._load_existing_mappings()
        logger.info(f"📁 Loaded {len(self.existing_mappings)} existing mapping files")
        
        # Build API name index for similarity search
        self.api_names = self._extract_api_names()
        logger.info(f"🔍 Indexed {len(self.api_names)} unique API names")
    
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
    
    def find_apis_for_utterance(self, utterance: str, top_k: int = 5) -> Tuple[List[str], List[str]]:
        """
        Find candidate source and target APIs for a natural language utterance.
        
        Args:
            utterance: Natural language description like "map concert vulnerabilities to turbo entities"
            top_k: Number of top candidate APIs to return
            
        Returns:
            Tuple of (source_candidates, target_candidates)
        """
        logger.info(f"🔍 Finding APIs for utterance: '{utterance}'")
        
        # Parse utterance to identify mapping direction and API hints
        source_hints, target_hints = self._parse_utterance_for_apis(utterance)
        
        # Find candidate APIs using embedding similarity
        source_candidates = self._find_similar_apis(source_hints, top_k)
        target_candidates = self._find_similar_apis(target_hints, top_k)
        
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
    
    def _find_similar_apis(self, hints: List[str], top_k: int = 5) -> List[str]:
        """Find similar API names using embedding similarity."""
        if not hints or not self.sentence_transformer:
            return []
        
        # Combine hints into a single query
        query = " ".join(hints)
        
        try:
            # Get embedding for the query
            query_embedding = self.sentence_transformer.encode([query])[0]
            
            # Calculate similarities with all API names
            similarities = []
            for api_name in self.api_names:
                # Create readable version of API name for embedding
                api_readable = self._make_api_name_readable(api_name)
                api_embedding = self.sentence_transformer.encode([api_readable])[0]
                
                # Calculate cosine similarity
                cosine_sim = np.dot(query_embedding, api_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(api_embedding)
                )
                similarities.append((api_name, cosine_sim))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [api_name for api_name, _ in similarities[:top_k]]
            
        except Exception as e:
            logger.warning(f"Error in similarity search: {e}")
            # Fallback: simple string matching
            return self._fallback_string_matching(hints, top_k)
    
    def _make_api_name_readable(self, api_name: str) -> str:
        """Convert API name to readable format for embedding."""
        # Handle camelCase and snake_case
        readable = re.sub(r'([a-z])([A-Z])', r'\1 \2', api_name)  # camelCase
        readable = readable.replace('_', ' ')  # snake_case
        return readable.lower()
    
    def _fallback_string_matching(self, hints: List[str], top_k: int = 5) -> List[str]:
        """Fallback string matching when embeddings fail."""
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
        Check if a mapping file exists for the given API pair.
        
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
                
                logger.info(f"✅ Found existing mapping: {Path(mapping_file).name}")
                return mapping_data
                
            except Exception as e:
                logger.error(f"❌ Error loading mapping file {mapping_file}: {e}")
                return None
        
        logger.info(f"❌ No existing mapping found for {source_api} ↔ {target_api}")
        return None
    
    def get_api_schemas(self, source_api: str, target_api: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Retrieve API schemas from vector database or local storage.
        
        Args:
            source_api: Source API name
            target_api: Target API name
            
        Returns:
            Tuple of (source_schema, target_schema) dictionaries
        """
        logger.info(f"📋 Retrieving schemas for {source_api} → {target_api}")
        
        # TODO: Implement actual vector database integration
        # For now, return mock schemas based on API names
        
        source_schema = self._get_mock_schema(source_api)
        target_schema = self._get_mock_schema(target_api)
        
        return source_schema, target_schema
    
    def _get_mock_schema(self, api_name: str) -> Dict[str, Any]:
        """Generate mock schema for an API (placeholder until vector DB integration)."""
        
        # Basic schema templates based on common API patterns
        if "turbo" in api_name.lower():
            return {
                "name": f"{api_name}_schema",
                "type": "object",
                "properties": {
                    "Entity": {
                        "type": "object",
                        "properties": {
                            "entityId": {"type": "string"},
                            "entityName": {"type": "string"},
                            "stats": {
                                "type": "object",
                                "properties": {
                                    "numRelatedEntities": {"type": "integer"},
                                    "statistics": {"type": "array"}
                                }
                            }
                        }
                    }
                }
            }
        elif "instana" in api_name.lower():
            return {
                "name": f"{api_name}_schema", 
                "type": "object",
                "properties": {
                    "entities_200_response": {
                        "type": "object",
                        "properties": {
                            "totalHits": {"type": "integer"},
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "label": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        elif "concert" in api_name.lower():
            return {
                "name": f"{api_name}_schema",
                "type": "object", 
                "properties": {
                    "Vulnerability": {
                        "type": "object",
                        "properties": {
                            "vulnerabilityId": {"type": "string"},
                            "severity": {"type": "string"},
                            "description": {"type": "string"},
                            "affectedComponents": {"type": "array"}
                        }
                    }
                }
            }
        else:
            # Generic schema
            return {
                "name": f"{api_name}_schema",
                "type": "object",
                "properties": {
                    "GenericEntity": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "attributes": {"type": "object"}
                        }
                    }
                }
            }
    
    def find_best_api_pair(self, utterance: str) -> Tuple[str, str, float]:
        """
        Find the best source-target API pair for an utterance.
        
        Args:
            utterance: Natural language mapping request
            
        Returns:
            Tuple of (source_api, target_api, confidence_score)
        """
        logger.info(f"🎯 Finding best API pair for: '{utterance}'")
        
        # Get API candidates
        source_candidates, target_candidates = self.find_apis_for_utterance(utterance)
        
        if not source_candidates or not target_candidates:
            logger.warning("No API candidates found")
            return "", "", 0.0
        
        # Find the best combination based on existing mappings and similarity
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
        
        logger.info(f"   Best pair: {best_source} → {best_target} (score: {best_score:.3f})")
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
        
        # Bonus for reasonable API name similarity to utterance
        if self.sentence_transformer:
            try:
                utterance_embedding = self.sentence_transformer.encode([utterance])[0]
                pair_text = f"{self._make_api_name_readable(source_api)} to {self._make_api_name_readable(target_api)}"
                pair_embedding = self.sentence_transformer.encode([pair_text])[0]
                
                similarity = np.dot(utterance_embedding, pair_embedding) / (
                    np.linalg.norm(utterance_embedding) * np.linalg.norm(pair_embedding)
                )
                score += similarity * 0.4
            except:
                pass
        
        return min(score, 1.0)  # Cap at 1.0
    
    def list_available_apis(self) -> List[str]:
        """List all available API names."""
        return sorted(self.api_names)
    
    def list_existing_mappings(self) -> List[str]:
        """List all existing mapping file pairs."""
        return sorted(self.existing_mappings.keys())


def main():
    """Test the API retriever functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SMT API Retriever")
    parser.add_argument("--utterance", required=True, help="Natural language utterance to test")
    parser.add_argument("--mappings-dir", default=".", help="Directory containing mapping files")
    
    args = parser.parse_args()
    
    # Initialize retriever
    retriever = SMTAPIRetriever(mappings_dir=args.mappings_dir)
    
    print(f"\n{'='*60}")
    print(f"SMT API Retriever Test")
    print(f"{'='*60}")
    print(f"Utterance: {args.utterance}")
    print(f"Mappings directory: {args.mappings_dir}")
    print(f"Available APIs: {len(retriever.list_available_apis())}")
    print(f"Existing mappings: {len(retriever.list_existing_mappings())}")
    
    # Test API finding
    print(f"\n🔍 Finding APIs...")
    source_candidates, target_candidates = retriever.find_apis_for_utterance(args.utterance)
    print(f"Source candidates: {source_candidates}")
    print(f"Target candidates: {target_candidates}")
    
    # Test best pair finding
    print(f"\n🎯 Finding best API pair...")
    best_source, best_target, confidence = retriever.find_best_api_pair(args.utterance)
    print(f"Best pair: {best_source} → {best_target}")
    print(f"Confidence: {confidence:.3f}")
    
    # Test existing mapping lookup
    print(f"\n📁 Checking existing mapping...")
    existing_mapping = retriever.get_existing_mapping(best_source, best_target)
    if existing_mapping:
        print(f"✅ Found existing mapping with {len(existing_mapping.get('properties', {}))} properties")
    else:
        print(f"❌ No existing mapping found")
    
    # Test schema retrieval
    print(f"\n📋 Retrieving schemas...")
    source_schema, target_schema = retriever.get_api_schemas(best_source, best_target)
    print(f"Source schema: {source_schema['name'] if source_schema else 'None'}")
    print(f"Target schema: {target_schema['name'] if target_schema else 'None'}")


if __name__ == "__main__":
    main()
