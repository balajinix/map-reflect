#!/usr/bin/env python3
"""
GNN API Retriever
Handles API discovery using ChromaDB for schemas and GNN for field joinability analysis
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import torch
import pickle

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from gnn_schema_indexer import GraphQLSchemaIndexer
except ImportError as e:
    logger.warning(f"Cannot import schema indexer: {e}")
    # Create mock class for testing
    class GraphQLSchemaIndexer:
        def __init__(self, *args, **kwargs):
            pass
        def search_schemas(self, query, n_results=5):
            return []

try:
    from gnn_model import MappingKGGNN, load_model_safely
    GNN_MODEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Cannot import GNN model: {e}")
    GNN_MODEL_AVAILABLE = False
    
    def load_model_safely(path):
        return None


class GNNAPIRetriever:
    """
    GNN-based API Retriever for schema mapping.
    Uses ChromaDB for schema discovery and GNN for field joinability analysis.
    """
    
    def __init__(self, 
                 gnn_model_path: str = None,
                 graph_data_path: str = None,
                 chroma_db_path: str = None,
                 collection_name: str = None,
                 api_lookup_file: str = None):
        """
        Initialize the GNN API retriever.
        
        Args:
            gnn_model_path: Path to trained GNN model
            graph_data_path: Path to graph data
            chroma_db_path: Path to ChromaDB database
            api_lookup_file: Path to API lookup file
        """
        self.gnn_model_path = Path(gnn_model_path)
        self.graph_data_path = Path(graph_data_path)
        self.chroma_db_path = Path(chroma_db_path)
        self.api_lookup_file = Path(api_lookup_file)
        
        # Initialize Schema ChromaDB indexer for schema discovery
        try:
            self.schema_indexer = GraphQLSchemaIndexer(
                chroma_db_path=str(chroma_db_path),
                collection_name=str(collection_name)
            )
            logger.info("Connected to Schema ChromaDB")
        except Exception as e:
            logger.warning(f"Could not connect to Schema ChromaDB: {e}")
            self.schema_indexer = None
        
        # Load GNN model and data
        self.gnn_model = None
        self.graph = None
        self.node_list = None
        self.graph_data = None
        self._load_gnn_components()
        
        # Load API lookup data
        self.api_lookup = self._load_api_lookup()
        
        # Build API name index
        self.api_names = self._extract_api_names()
        logger.info(f"Indexed {len(self.api_names)} unique API names")
    
    def _load_gnn_components(self):
        """Load GNN model and graph data."""
        try:
            # Load GNN model
            if self.gnn_model_path.exists() and GNN_MODEL_AVAILABLE:
                try:
                    # Use safe loading method
                    self.gnn_model = load_model_safely(self.gnn_model_path)
                    
                    if self.gnn_model and hasattr(self.gnn_model, 'eval'):
                        self.gnn_model.eval()
                        logger.info("GNN model loaded successfully")
                    else:
                        logger.warning("GNN model loaded but may not be functional")
                        
                except Exception as e:
                    logger.warning(f"Error loading GNN model: {e}")
                    self.gnn_model = None
            else:
                if not self.gnn_model_path.exists():
                    logger.warning(f"GNN model file not found: {self.gnn_model_path}")
                if not GNN_MODEL_AVAILABLE:
                    logger.warning("GNN model dependencies not available")
                self.gnn_model = None
            
            # Load graph data
            if self.graph_data_path.exists():
                try:
                    with open(self.graph_data_path, 'rb') as f:
                        graph_data = pickle.load(f)
                    
                    self.graph_data = graph_data.get('data')
                    self.graph = graph_data.get('graph')
                    self.node_list = graph_data.get('node_list')
                    
                    logger.info("Graph data loaded successfully")
                except Exception as e:
                    logger.warning(f"Error loading graph data: {e}")
                    self.graph_data = None
                    self.graph = None
                    self.node_list = None
            else:
                logger.warning(f"Graph data file not found: {self.graph_data_path}")
                self.graph_data = None
                self.graph = None
                self.node_list = None
                
        except Exception as e:
            logger.error(f"Error loading GNN components: {e}")
            self.gnn_model = None
            self.graph = None
            self.graph_data = None
            self.node_list = None
    
    def _load_api_lookup(self) -> Dict[str, Any]:
        """Load API lookup data for field descriptions."""
        try:
            with open(self.api_lookup_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"API lookup file not found: {self.api_lookup_file}")
            return {}
    
    def _extract_api_names(self) -> List[str]:
        """Extract unique API/schema names from API lookup data.

        Returns both product names AND schema type names for better matching.
        E.g., ['Concert', 'concert_Application', 'concert_ApiKeyDate', ...]
        """
        api_names = set()
        schema_types = set()

        for field_name, meta in self.api_lookup.items():
            # Extract product name
            product_name = meta.get("product_name", "")
            if product_name:
                api_names.add(product_name)

            # Extract schema type from field name (e.g., "concert_Application.id" -> "concert_Application")
            parts = field_name.split('.')
            if parts:
                schema_type = parts[0]
                schema_types.add(schema_type)

        # Combine both - schema types are more specific and useful for matching
        all_names = list(schema_types) + list(api_names)
        return all_names
    
    def find_apis_for_utterance(self, utterance: str, top_k: int = 5) -> Tuple[List[str], List[str]]:
        """
        Find candidate source and target APIs for a natural language utterance.
        
        Args:
            utterance: Natural language description like "map concert vulnerabilities to turbo entities"
            top_k: Number of top candidate APIs to return
            
        Returns:
            Tuple of (source_candidates, target_candidates)
        """
        logger.info(f"Finding API candidates for: '{utterance}'")
        
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
        # Note: ITOps dataset uses "with" not "to" (e.g., "map concert X with turbo Y")
        map_patterns = [
            r"map\s+(\w+)\s+.*?with\s+(\w+)",      # "map concert X with turbo Y"
            r"map\s+(\w+)\s+.*?to\s+(\w+)",        # "map concert X to turbo Y"
            r"map\s+(\w+)\s+.*?\s+to\s+(\w+)",
            r"from\s+(\w+)\s+.*?to\s+(\w+)",
            r"(\w+)\s+.*?to\s+(\w+)\s+.*?mapping",
            r"(\w+)\s+.*?with\s+(\w+)\s+.*?mapping"
        ]
        
        source_hints = []
        target_hints = []
        
        # Try to extract using patterns
        for pattern in map_patterns:
            match = re.search(pattern, utterance_lower)
            if match:
                source_product = match.group(1)
                target_product = match.group(2)
                source_hints.append(source_product)
                target_hints.append(target_product)

                # Also extract domain terms from the query for better schema matching
                # E.g., "map concert api key with turbo entity" ->
                #   source: ['concert', 'api', 'key'], target: ['turbo', 'entity']
                words = re.findall(r'\b\w+\b', utterance_lower)
                common_words = {'map', 'to', 'from', 'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'as', 'data', 'schema', 'table', 'field', 'type'}

                # Find position of products in the word list
                try:
                    src_idx = words.index(source_product)
                    tgt_idx = words.index(target_product)

                    # Words between source product and target product are source domain terms
                    source_domain = [w for w in words[src_idx+1:tgt_idx] if w not in common_words and len(w) > 2]
                    source_hints.extend(source_domain[:3])

                    # Words after target product are target domain terms
                    target_domain = [w for w in words[tgt_idx+1:] if w not in common_words and len(w) > 2]
                    target_hints.extend(target_domain[:3])
                except ValueError:
                    pass

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
            common_words = {'map', 'to', 'from', 'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'as', 'data', 'schema', 'table', 'field', 'type'}
            api_words = [w for w in words if w not in common_words and len(w) > 2]
            source_hints = api_words[:3]  # Take first few as source hints
            target_hints = api_words[-3:]  # Take last few as target hints
        
        return source_hints, target_hints
    
    def _find_schemas_with_chromadb(self, hints: List[str], top_k: int = 5) -> List[str]:
        """Find similar schemas using Schema ChromaDB semantic search."""
        if not hints or not self.schema_indexer:
            return self._find_similar_apis_fallback(hints, top_k)

        # Combine hints into a query
        query = " ".join(hints)

        try:
            # Search for relevant schemas
            schema_results = self.schema_indexer.search_schemas(query, n_results=top_k)

            # Extract schema names
            schema_names = [result.get("schema_name", "") for result in schema_results]

            # Fall back to API lookup matching if ChromaDB returns empty
            if not schema_names:
                logger.info(f"ChromaDB returned empty results for '{query}', using fallback")
                return self._find_similar_apis_fallback(hints, top_k)

            return schema_names

        except Exception as e:
            logger.warning(f"Schema ChromaDB search failed: {e}")
            return self._find_similar_apis_fallback(hints, top_k)
    
    def _find_similar_apis_fallback(self, hints: List[str], top_k: int = 5) -> List[str]:
        """Fallback string matching to find schemas matching query hints.

        Improved matching for ITOps queries like:
        - 'concert api key' -> 'concert_ApiKeyDate'
        - 'concert applications' -> 'concert_Application', 'concert_ApplicationCollection'
        - 'concert exposures' -> 'concert_Exposures'
        """
        matches = []
        scored_matches = []

        # Normalize hints - combine multi-word hints
        hint_words = []
        for hint in hints:
            hint_words.extend(hint.lower().split())

        for api_name in self.api_names:
            api_lower = api_name.lower()
            score = 0

            # Check each hint word
            for hint in hint_words:
                hint_lower = hint.lower()

                # Exact substring match
                if hint_lower in api_lower:
                    score += 2

                # Handle common word variations
                # 'api key' -> 'apikey', 'applications' -> 'application'
                hint_normalized = hint_lower.rstrip('s')  # Remove plural 's'
                api_words = api_lower.replace('_', ' ').split()

                for api_word in api_words:
                    if hint_normalized in api_word or api_word in hint_normalized:
                        score += 1

            if score > 0:
                scored_matches.append((api_name, score))

        # Sort by score descending
        scored_matches.sort(key=lambda x: -x[1])

        # Return top matches
        matches = [m[0] for m in scored_matches[:top_k]]

        # If no matches, fall back to product-level matching
        if not matches:
            for api_name in self.api_names:
                api_lower = api_name.lower()
                for hint in hints:
                    if hint.lower() in api_lower:
                        matches.append(api_name)
                        break
                if len(matches) >= top_k:
                    break

        return matches[:top_k]
    
    def get_joinability_score(self, field1: str, field2: str, quiet: bool = False) -> float:
        """
        Get GNN-based joinability score between two fields.
        
        Args:
            field1: First field name (e.g., "vulnerabilities.id")
            field2: Second field name (e.g., "entities.entityId")
            quiet: If True, suppress warnings about missing fields
            
        Returns:
            Joinability score between 0 and 1
        """
        if not self.gnn_model or not self.graph_data or not self.node_list:
            if not quiet:
                logger.warning("GNN model or data not loaded - using fallback scoring")
            return self._fallback_joinability_score(field1, field2)
        
        try:
            # Find node indices for the fields
            idx1 = None
            idx2 = None
            
            # Search for field nodes in the graph
            for i, node_id in enumerate(self.node_list):
                if self.graph and node_id in self.graph.nodes:
                    node_data = self.graph.nodes[node_id]
                    if node_data.get('node_type') == 'field':
                        full_name = node_data.get('full_name', '')
                        if full_name == field1:
                            idx1 = i
                        elif full_name == field2:
                            idx2 = i
            
            if idx1 is None or idx2 is None:
                if not quiet:
                    logger.warning(f"Could not find field nodes for {field1} or {field2}")
                return self._fallback_joinability_score(field1, field2)
            
            # Create query edges tensor
            import torch
            query_edges = torch.tensor([[idx1], [idx2]], dtype=torch.long)

            # Get joinability score using semantic similarity of original features
            # This uses sentence-transformer embeddings directly, avoiding GNN over-smoothing
            with torch.no_grad():
                # Use semantic similarity (original features) - best for generalization
                if hasattr(self.gnn_model, 'predict_links_semantic'):
                    score = self.gnn_model.predict_links_semantic(
                        self.graph_data.x,
                        self.graph_data.edge_index,
                        query_edges
                    )
                elif hasattr(self.gnn_model, 'predict_links_cosine'):
                    # Fallback to GNN cosine similarity
                    score = self.gnn_model.predict_links_cosine(
                        self.graph_data.x,
                        self.graph_data.edge_index,
                        query_edges
                    )
                else:
                    # Fallback to original MLP method
                    score = self.gnn_model.predict_links(
                        self.graph_data.x,
                        self.graph_data.edge_index,
                        query_edges
                    )

            return score.item()
            
        except Exception as e:
            if not quiet:
                logger.warning(f"Error calculating GNN joinability score: {e}")
            return self._fallback_joinability_score(field1, field2)
    
    def _fallback_joinability_score(self, field1: str, field2: str) -> float:
        """
        Fallback joinability scoring when GNN is not available.
        Uses simple heuristics based on field names.
        """
        try:
            # Simple similarity based on field names
            field1_parts = field1.lower().split('.')
            field2_parts = field2.lower().split('.')
            
            # Check for similar field names
            field1_name = field1_parts[-1] if field1_parts else ""
            field2_name = field2_parts[-1] if field2_parts else ""
            
            # Basic similarity heuristics
            if field1_name == field2_name:
                return 0.8  # Exact match
            
            # Check for common ID patterns
            id_patterns = ['id', 'identifier', 'key', 'uuid']
            if any(pattern in field1_name for pattern in id_patterns) and \
               any(pattern in field2_name for pattern in id_patterns):
                return 0.6  # Both are ID-like fields
            
            # Check for name patterns
            name_patterns = ['name', 'title', 'label']
            if any(pattern in field1_name for pattern in name_patterns) and \
               any(pattern in field2_name for pattern in name_patterns):
                return 0.5  # Both are name-like fields
            
            # Default similarity based on string overlap
            common_chars = set(field1_name) & set(field2_name)
            if common_chars:
                similarity = len(common_chars) / max(len(field1_name), len(field2_name), 1)
                return min(similarity * 0.4, 0.4)  # Cap at 0.4 for fallback
            
            return 0.1  # Minimal similarity
            
        except Exception as e:
            logger.warning(f"Error in fallback scoring: {e}")
            return 0.1
    
    def find_best_api_pair(self, utterance: str) -> Tuple[str, str, float]:
        """
        Find the best source-target API pair for an utterance using GNN analysis.
        
        Args:
            utterance: Natural language mapping request
            
        Returns:
            Tuple of (source_api, target_api, confidence_score)
        """
        logger.info(f"Finding best API pair for: '{utterance}'")
        
        # Find API candidates
        source_candidates, target_candidates = self.find_apis_for_utterance(utterance)
        
        if not source_candidates or not target_candidates:
            logger.warning("No API candidates found")
            # Return default APIs if available
            if self.api_names:
                return self.api_names[0], self.api_names[-1] if len(self.api_names) > 1 else self.api_names[0], 0.1
            return "default_source", "default_target", 0.0
        
        # Find the best combination based on GNN joinability analysis
        best_score = 0.0
        best_source = source_candidates[0]
        best_target = target_candidates[0]
        
        for source in source_candidates[:3]:  # Check top 3 sources
            for target in target_candidates[:3]:  # Check top 3 targets
                score = self._calculate_api_pair_score(source, target, utterance)
                if score > best_score:
                    best_score = score
                    best_source = source
                    best_target = target
        
        logger.info(f"   Best pair: {best_source} → {best_target} (score: {best_score:.3f})")
        return best_source, best_target, best_score
    
    def _calculate_api_pair_score(self, source_api: str, target_api: str, utterance: str) -> float:
        """Calculate confidence score for an API pair using GNN analysis."""
        score = 0.0
        
        # Bonus for API names appearing in utterance
        utterance_lower = utterance.lower()
        if source_api.lower() in utterance_lower:
            score += 0.3
        if target_api.lower() in utterance_lower:
            score += 0.3
        
        # GNN-based joinability analysis
        if self.gnn_model and self.api_lookup:
            try:
                # Find representative fields from each API
                source_fields = [field for field, meta in self.api_lookup.items() 
                               if meta.get("product_name", "").lower() == source_api.lower()]
                target_fields = [field for field, meta in self.api_lookup.items() 
                               if meta.get("product_name", "").lower() == target_api.lower()]
                
                if source_fields and target_fields:
                    # Sample a few field pairs and get average joinability
                    max_samples = min(3, len(source_fields), len(target_fields))
                    joinability_scores = []
                    
                    for i in range(max_samples):
                        src_field = source_fields[i % len(source_fields)]
                        tgt_field = target_fields[i % len(target_fields)]
                        
                        joinability = self.get_joinability_score(src_field, tgt_field)
                        joinability_scores.append(joinability)
                    
                    if joinability_scores:
                        avg_joinability = sum(joinability_scores) / len(joinability_scores)
                        score += 0.4 * avg_joinability  # Weight GNN score
                        
            except Exception as e:
                logger.warning(f"Error in GNN analysis for {source_api}-{target_api}: {e}")
        
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
    
    def _extract_product_name(self, api_name: str) -> str:
        """
        Extract the product name from a schema/API name.

        Examples:
            concert__convertApplicationName -> concert
            instana_applications -> instana
            turbo_marketActionsFiltered -> turbo
            concert -> concert
        """
        # Handle double underscore format (from ChromaDB schema names)
        if '__' in api_name:
            return api_name.split('__')[0].lower()
        # Handle single underscore format
        elif '_' in api_name:
            return api_name.split('_')[0].lower()
        # Already a product name
        return api_name.lower()

    def _extract_schema_type(self, api_name: str) -> str:
        """
        Extract the schema type from a schema/API name.

        Examples:
            omop_PERSON -> PERSON
            mimic_PATIENTS -> PATIENTS
            concert_Application -> Application
            concert -> None (product only, no specific type)
        """
        # Handle underscore format: product_TYPE
        if '_' in api_name:
            parts = api_name.split('_', 1)
            if len(parts) > 1:
                return parts[1]
        return None

    def get_top_field_mappings(self, source_api: str, target_api: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get top field mappings between two APIs using GNN analysis.

        Args:
            source_api: Source API name (e.g., omop_PERSON, mimic_PATIENTS)
            target_api: Target API name
            top_k: Number of top mappings to return

        Returns:
            List of field mapping predictions with scores
        """
        if not self.gnn_model or not self.api_lookup:
            logger.warning("GNN model or API lookup not available")
            return []

        try:
            # Extract product names and schema types from API names
            source_product = self._extract_product_name(source_api)
            target_product = self._extract_product_name(target_api)
            source_type = self._extract_schema_type(source_api)
            target_type = self._extract_schema_type(target_api)

            logger.info(f"Looking for fields: source={source_product}/{source_type}, target={target_product}/{target_type}")

            # Find fields for each API by product AND schema type
            source_fields = []
            target_fields = []

            for field, meta in self.api_lookup.items():
                field_product = meta.get("product_name", "").lower()
                field_type = meta.get("type_name", "")

                # Match source fields
                if field_product == source_product:
                    if source_type is None or field_type.upper() == source_type.upper():
                        source_fields.append(field)

                # Match target fields
                if field_product == target_product:
                    if target_type is None or field_type.upper() == target_type.upper():
                        target_fields.append(field)

            logger.info(f"Found {len(source_fields)} source fields, {len(target_fields)} target fields")

            if not source_fields or not target_fields:
                logger.warning(f"No fields found for {source_api} (product: {source_product}) or {target_api} (product: {target_product})")
                return []
            
            # Filter to only fields that exist in the graph (to reduce warnings)
            if self.graph and self.node_list:
                graph_field_names = set()
                for node_id in self.node_list:
                    if node_id in self.graph.nodes:
                        node_data = self.graph.nodes[node_id]
                        if node_data.get('node_type') == 'field':
                            full_name = node_data.get('full_name', '')
                            if full_name:
                                graph_field_names.add(full_name)

                logger.info(f"Graph has {len(graph_field_names)} field names")

                # Filter fields to only those in the graph
                source_fields_before = len(source_fields)
                target_fields_before = len(target_fields)
                source_fields = [f for f in source_fields if f in graph_field_names]
                target_fields = [f for f in target_fields if f in graph_field_names]

                logger.info(f"After filtering: {len(source_fields)}/{source_fields_before} source, {len(target_fields)}/{target_fields_before} target")

                if not source_fields or not target_fields:
                    logger.info(f"No fields from {source_api} or {target_api} found in trained graph")
                    logger.info(f"Graph contains {len(graph_field_names)} field nodes")
                    return []
            
            # Calculate joinability scores for field pairs (limited to reduce computation)
            field_mappings = []
            max_combinations = min(50, len(source_fields) * len(target_fields))  # Limit combinations
            
            count = 0
            for src_field in source_fields:
                if count >= max_combinations:
                    break
                for tgt_field in target_fields:
                    if count >= max_combinations:
                        break
                    
                    score = self.get_joinability_score(src_field, tgt_field, quiet=True)  # Suppress warnings
                    
                    field_mappings.append({
                        "source_field": src_field,
                        "target_field": tgt_field,
                        "joinability_score": score,
                        "source_api": source_api,
                        "target_api": target_api
                    })
                    
                    count += 1
            
            # Sort by score and return top-k
            field_mappings.sort(key=lambda x: x["joinability_score"], reverse=True)

            logger.info(f"Calculated {len(field_mappings)} field mappings, returning top {top_k}")
            if field_mappings:
                logger.info(f"Top mapping: {field_mappings[0]['source_field']} -> {field_mappings[0]['target_field']} (score: {field_mappings[0]['joinability_score']:.4f})")

            return field_mappings[:top_k]
            
        except Exception as e:
            logger.error(f"Error getting field mappings: {e}")
            return []
    
    def list_available_apis(self) -> List[str]:
        """List all available API names."""
        return sorted(self.api_names)
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever's capabilities."""
        stats = {
            "unique_api_names": len(self.api_names),
            "schema_chromadb_available": self.schema_indexer is not None,
            "gnn_model_available": self.gnn_model is not None,
            "graph_data_available": self.graph_data is not None,
            "api_lookup_available": len(self.api_lookup) > 0
        }
        
        if self.graph:
            stats.update({
                "graph_nodes": self.graph.number_of_nodes(),
                "graph_edges": self.graph.number_of_edges(),
                "field_nodes": len([n for n, d in self.graph.nodes(data=True) 
                                  if d.get('node_type') == 'field'])
            })
        
        return stats


def main():
    """Main entry point for GNN API Retriever."""
    parser = argparse.ArgumentParser(description="GNN API Retriever")
    parser.add_argument("--utterance", required=True, help="Natural language utterance")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results")
    parser.add_argument("--gnn-model-path", default="./gnn_results/mapping_gnn_results.pkl", 
                       help="Path to GNN model")
    parser.add_argument("--graph-data-path", default="./graph_data.pkl", 
                       help="Path to graph data")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize retriever
    retriever = GNNAPIRetriever(
        gnn_model_path=args.gnn_model_path,
        graph_data_path=args.graph_data_path
    )
    
    print(f"GNN API Retriever")
    print(f"Utterance: {args.utterance}")
    
    # Show retriever stats
    stats = retriever.get_retriever_stats()
    print(f"Retriever Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Find best API pair
    source_api, target_api, confidence = retriever.find_best_api_pair(args.utterance)
    
    print(f"\nSelected API pair: {source_api} → {target_api} (confidence: {confidence:.3f})")
    
    # Get top field mappings
    field_mappings = retriever.get_top_field_mappings(source_api, target_api, top_k=args.top_k)
    
    if field_mappings:
        print(f"\nTop {len(field_mappings)} field mappings:")
        for i, mapping in enumerate(field_mappings, 1):
            print(f"{i:2d}. {mapping['source_field']} → {mapping['target_field']} "
                  f"(score: {mapping['joinability_score']:.4f})")
    else:
        print("\nNo field mappings found")


if __name__ == "__main__":
    main()
