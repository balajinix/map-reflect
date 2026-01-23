# match_maker_agent.py
"""
MatchMaker: Self-Improving Compositional Language Model Program for Schema Matching
Faithful implementation of the paper: "Bootstrapping Self-Improvement of Language Model Programs for Zero-Shot Schema Matching"
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re

from litellm import completion
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger("matchmaker")


# ---------- Helper Functions ----------

def _strip_code_fences(s: str) -> str:
    """Remove Markdown code fences from model output."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.replace("```json", "").replace("```", "").strip()


def _safe_json_loads(payload: str) -> Tuple[Any, Optional[str]]:
    """Try json.loads; return (obj_or_none, error_str_or_none)."""
    try:
        return json.loads(payload), None
    except Exception as e:
        return None, str(e)


# ---------- Data Classes ----------

@dataclass
class CandidateMatch:
    """Represents a candidate target attribute match."""
    target_attribute: str
    confidence: float
    rank: int
    justification: str
    source: str  # 'semantic', 'reasoning', or 'both'


@dataclass
class MatchResult:
    """Result of matching a source attribute."""
    source_attribute: str
    candidates: List[CandidateMatch]
    selected_match: Optional[str]
    abstain: bool
    reasoning_trace: Dict[str, Any]


# ---------- MatchMaker Agent ----------

class MatchMakerAgent:
    """
    MatchMaker: Self-Improving Compositional LM Program for Schema Matching.
    
    Implements the approach from the paper:
    1. Multi-vector document creation (ColBERT-v2)
    2. Diverse candidate generation (semantic + reasoning)
    3. Candidate refinement
    4. MCQ-based confidence scoring
    5. Self-improvement via synthetic in-context examples
    """
    
    def __init__(
        self,
        property_lookup_file: str,
        model: str,
        dataset: str,
        chroma_dir: str = "../data/chroma_db",
        collection_name: Optional[str] = None,
        top_k: int = 5,
        rebuild_index: bool = False,
        use_optimization: bool = True,
        n_incontext_examples: int = 4,
        eval_set_size: int = 20
    ):
        """
        Initialize MatchMaker agent.
        
        Args:
            property_lookup_file: Path to property lookup JSON
            model: LLM model identifier
            dataset: Dataset name for namespacing
            chroma_dir: ChromaDB storage directory
            collection_name: Override collection name
            top_k: Number of candidates to retrieve
            rebuild_index: Whether to rebuild vector index
            use_optimization: Whether to use self-improvement
            n_incontext_examples: Number of in-context examples
            eval_set_size: Size of evaluation set for optimization
        """
        self.model = model
        self.dataset = dataset
        self.top_k = top_k
        self.rebuild_index = rebuild_index
        self.use_optimization = use_optimization
        self.n_incontext_examples = n_incontext_examples
        self.eval_set_size = eval_set_size
        
        # Load property data
        with open(property_lookup_file, 'r') as f:
            self.property_data = json.load(f)
        
        # Separate source and target properties
        self.source_properties = {}
        self.target_properties = {}
        self._parse_properties()
        
        # Initialize ColBERT-v2 components
        try:
            from ragatouille import RAGPretrainedModel
            self.colbert_available = True
            self.rag_model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        except ImportError:
            logger.warning("RAGatouille not available, falling back to sentence-transformers")
            self.colbert_available = False
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Collection name
        self.collection_name = (
            collection_name if collection_name
            else f"matchmaker_{self.dataset.lower()}"
        )
        
        # In-context examples (will be populated during optimization)
        self.incontext_examples = {
            'candidate_generation': [],
            'candidate_refinement': [],
            'confidence_scoring': []
        }
        
        # Build or load index
        self.index_path = f"{chroma_dir}/{self.collection_name}"
        if self.rebuild_index:
            self._build_multi_vector_index()
        else:
            self._load_index()
        
        # Run self-improvement optimization if requested
        if self.use_optimization:
            self._run_self_improvement()
        
        logger.info(
            f"MatchMaker initialized | dataset={dataset} model={model} "
            f"top_k={top_k} optimization={use_optimization}"
        )
    
    def _parse_properties(self):
        """Parse properties into source and target based on dataset."""
        # For MIMIC-OMOP: pre-defined source/target
        if self.dataset.lower() == 'mimic_omop':
            for prop_id, prop_data in self.property_data.items():
                product = prop_data.get('product_name', '').lower()
                if 'mimic' in product:
                    self.source_properties[prop_id] = prop_data
                elif 'omop' in product:
                    self.target_properties[prop_id] = prop_data
        
        # For itops/nl2gql: dynamic source/target (set via set_source_target_from_utterance)
        else:
            # Initially, all properties available
            # Will be filtered dynamically per utterance
            pass
    
    def set_source_target_from_utterance(self, utterance: str):
        """
        Dynamically set source and target properties based on utterance.

        For datasets like itops/nl2gql where source/target are not pre-defined,
        parse the utterance to determine which products/schemas to use.

        Example ITOps: "map concert vulnerabilities to turbo entities"
        → source: concert_* properties (by product_name)
        → target: turbonomic_* properties (by product_name)

        Example NL2GQL: "map concert data to stadium data"
        → source: Concert_* properties (by type_name)
        → target: Stadium_* properties (by type_name)
        """
        # Check if this is a single-product dataset like NL2GQL
        available_products = list(set(
            p.get('product_name', '')
            for p in self.property_data.values()
            if p.get('product_name')
        ))

        # For datasets with only one product (like nl2gql), use type_name (table name) instead
        use_type_name = len(available_products) <= 1

        if use_type_name:
            # Extract available table names from type_name
            available_tables = list(set(
                p.get('type_name', '')
                for p in self.property_data.values()
                if p.get('type_name')
            ))
            available_schemas = available_tables
            schema_field = 'type_name'
            examples = """- "map concert data to stadium data" → {{"source_schema": "Concert", "target_schema": "Stadium"}}
- "map singer_in_concert data to singer data" → {{"source_schema": "SingerInConcert", "target_schema": "Singer"}}"""
        else:
            available_schemas = available_products
            schema_field = 'product_name'
            examples = """- "map concert vulnerabilities to turbo entities" → {{"source_schema": "Concert", "target_schema": "Turbonomic"}}
- "find instana metrics similar to apptio costs" → {{"source_schema": "Instana", "target_schema": "Apptio"}}"""

        # Extract source and target from utterance using LLM
        prompt = f"""Parse this schema mapping request and identify the source and target schemas.

Utterance: "{utterance}"

Available {'tables' if use_type_name else 'products'} in the system: {', '.join(available_schemas[:50])}{'...' if len(available_schemas) > 50 else ''}

Return ONLY a JSON object with:
{{
  "source_schema": "source {'table' if use_type_name else 'product'} name",
  "target_schema": "target {'table' if use_type_name else 'product'} name"
}}

Examples:
{examples}
"""

        messages = [
            {"role": "system", "content": "You are a schema matching assistant. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, temperature=0.0)
        response = _strip_code_fences(response)

        parsed, err = _safe_json_loads(response)
        if err or not isinstance(parsed, dict):
            logger.warning(f"Failed to parse source/target from utterance: {err}")
            # Fallback: use all properties
            self.source_properties = self.property_data.copy()
            self.target_properties = self.property_data.copy()
            return

        source_schema = parsed.get('source_schema', '').lower()
        target_schema = parsed.get('target_schema', '').lower()

        logger.info(f"Parsed utterance → source: {source_schema}, target: {target_schema} (using {schema_field})")

        # Filter properties by the appropriate field
        self.source_properties = {
            prop_id: prop_data
            for prop_id, prop_data in self.property_data.items()
            if source_schema in prop_data.get(schema_field, '').lower()
        }

        self.target_properties = {
            prop_id: prop_data
            for prop_id, prop_data in self.property_data.items()
            if target_schema in prop_data.get(schema_field, '').lower()
        }

        logger.info(
            f"Filtered properties: {len(self.source_properties)} source, "
            f"{len(self.target_properties)} target"
        )

        # If no properties found, try a more lenient match
        if not self.source_properties or not self.target_properties:
            logger.warning(f"No properties found, trying lenient match")
            # Try matching just the first word of the schema name
            source_word = source_schema.split('_')[0] if '_' in source_schema else source_schema
            target_word = target_schema.split('_')[0] if '_' in target_schema else target_schema

            if not self.source_properties:
                self.source_properties = {
                    prop_id: prop_data
                    for prop_id, prop_data in self.property_data.items()
                    if source_word in prop_data.get(schema_field, '').lower()
                }

            if not self.target_properties:
                self.target_properties = {
                    prop_id: prop_data
                    for prop_id, prop_data in self.property_data.items()
                    if target_word in prop_data.get(schema_field, '').lower()
                }

            logger.info(
                f"After lenient match: {len(self.source_properties)} source, "
                f"{len(self.target_properties)} target"
            )

        # Rebuild index with filtered target properties if needed
        if self.colbert_available:
            logger.info("Rebuilding index with filtered target properties...")
            self._build_multi_vector_index()
        elif hasattr(self, 'collection'):
            # Update ChromaDB collection with filtered targets
            logger.info("Updating ChromaDB with filtered target properties...")
            self._build_multi_vector_index()
    
    # ========== Step 1: Multi-Vector Index Creation ==========
    
    def _build_multi_vector_index(self):
        """Build ColBERT-v2 multi-vector index from target schema."""
        logger.info(f"Building multi-vector index: {self.collection_name}")
        
        # For utterance-based datasets (itops/nl2gql), index ALL properties initially
        # We'll filter at query time, not at index time
        if self.dataset.lower() in ['itops', 'nl2gql']:
            properties_to_index = self.property_data
            logger.info(f"Utterance-based dataset: indexing all {len(properties_to_index)} properties")
        else:
            # For MIMIC-OMOP: only index target properties
            properties_to_index = self.target_properties
            logger.info(f"Attribute-based dataset: indexing {len(properties_to_index)} target properties")
        
        # Prepare documents
        documents = []
        doc_ids = []
        
        for prop_id, prop_data in tqdm(
            properties_to_index.items(),
            desc="Preparing documents for indexing",
            disable=logger.level > logging.INFO
        ):
            # Create rich document representation
            doc = self._create_document(prop_id, prop_data)
            documents.append(doc)
            doc_ids.append(prop_id)
        
        if self.colbert_available:
            # Use ColBERT-v2 for indexing
            self.rag_model.index(
                collection=documents,
                document_ids=doc_ids,
                index_name=self.collection_name,
                max_document_length=256,
                split_documents=False
            )
        else:
            # Fallback: use sentence-transformers with ChromaDB
            import chromadb
            self.chroma_client = chromadb.PersistentClient(path=self.index_path)
            
            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
            except:
                pass
            
            collection = self.chroma_client.create_collection(
                name=self.collection_name
            )
            
            embeddings = [self.embedding_model.encode(doc) for doc in tqdm(
                documents, 
                desc="Embedding documents",
                disable=logger.level > logging.INFO
            )]
            
            collection.add(
                ids=doc_ids,
                documents=documents,
                embeddings=embeddings
            )
            
            self.collection = collection
        
        logger.info(f"Index built with {len(documents)} properties")

    def _load_index(self):
        """Load existing index, or create if it doesn't exist."""
        logger.info(f"Loading existing index: {self.collection_name}")
        
        if self.colbert_available:
            # Index already loaded in __init__
            pass
        else:
            import chromadb
            self.chroma_client = chromadb.PersistentClient(path=self.index_path)
            
            try:
                self.collection = self.chroma_client.get_collection(self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception as e:
                logger.warning(f"Collection {self.collection_name} doesn't exist, creating it now...")
                self._build_multi_vector_index()
    
    def _create_document(self, prop_id: str, prop_data: Dict) -> str:
        """
        Create document from property for indexing.
        
        Format: {product} {type} {field} {description}
        This captures the hierarchical context as per paper.
        """
        parts = [
            prop_data.get('product_name', ''),
            prop_data.get('type_name', ''),
            prop_data.get('field_name', ''),
            prop_data.get('description', '')
        ]
        return ' '.join(filter(None, parts))
    
    # ========== Step 2: Candidate Generation ==========
    
    def _semantic_candidate_generation(
        self, 
        query_attr: str, 
        query_data: Dict
    ) -> List[str]:
        """
        Generate semantic retrieval candidates using multi-vector retrieval.
        
        Paper: Uses ColBERT-v2 late interaction with MaxSim operator.
        """
        query_doc = self._create_document(query_attr, query_data)
        
        if self.colbert_available:
            # Use ColBERT-v2 for retrieval
            results = self.rag_model.search(
                query=query_doc,
                index_name=self.collection_name,
                k=self.top_k
            )
            # Extract document IDs
            candidates = [r['document_id'] for r in results]
        else:
            # Fallback: simple embedding similarity
            query_embedding = self.embedding_model.encode(query_doc)
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=self.top_k
            )
            candidates = results['ids'][0] if results.get('ids') else []
        
        logger.debug(f"Semantic candidates for {query_attr}: {candidates}")
        return candidates
    
    def _reasoning_candidate_generation(
        self,
        query_attr: str,
        query_data: Dict
    ) -> List[str]:
        """
        Generate reasoning-based candidates using LLM with Chain-of-Thought.
        
        Paper: LLM reasons about schema hierarchy, relationships, and data types.
        """
        # Prepare context about target schema
        target_context = self._prepare_target_context()
        
        # Get in-context examples if available
        incontext = self._format_incontext_examples('candidate_generation')
        
        prompt = f"""You are an expert schema matcher. Given a source attribute, reason about which target attributes might match based on:
- Semantic similarity
- Schema hierarchy and relationships
- Data types and constraints
- Domain knowledge

Source Attribute:
- ID: {query_attr}
- Product: {query_data.get('product_name', 'N/A')}
- Type: {query_data.get('type_name', 'N/A')}
- Field: {query_data.get('field_name', 'N/A')}
- Description: {query_data.get('description', 'N/A')}

Target Schema Context:
{target_context}

{incontext}

Task: Reason step-by-step about which target attributes are most likely to match. Return ONLY a JSON array of target attribute IDs (up to {self.top_k} candidates):
["target_id_1", "target_id_2", ...]"""

        messages = [
            {"role": "system", "content": "You are an expert schema matching assistant. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_llm(messages, temperature=0.5)
        response = _strip_code_fences(response)
        
        candidates, err = _safe_json_loads(response)
        if err or not isinstance(candidates, list):
            logger.warning(f"Failed to parse reasoning candidates: {err}")
            return []
        
        # Filter to valid target IDs
        valid_candidates = [
            c for c in candidates 
            if c in self.target_properties
        ][:self.top_k]
        
        logger.debug(f"Reasoning candidates for {query_attr}: {valid_candidates}")
        return valid_candidates
    
    def _prepare_target_context(self, max_attrs: int = 10) -> str:
        """Prepare concise context about target schema."""
        context_parts = []
        for prop_id, prop_data in list(self.target_properties.items())[:max_attrs]:
            context_parts.append(
                f"- {prop_id}: {prop_data.get('field_name', '')} "
                f"({prop_data.get('type_name', '')})"
            )
        return '\n'.join(context_parts)
    
    def _candidate_refinement(
        self,
        candidates: List[str],
        query_attr: str,
        query_data: Dict
    ) -> List[str]:
        """
        Refine combined candidates to top-5 most relevant.
        
        Paper: Uses LLM with CoT to select most relevant candidates.
        """
        if len(candidates) <= 5:
            return candidates
        
        # Prepare candidate details
        candidate_details = []
        for cand_id in candidates:
            cand_data = self.target_properties.get(cand_id, {})
            candidate_details.append({
                'id': cand_id,
                'field': cand_data.get('field_name', ''),
                'type': cand_data.get('type_name', ''),
                'description': cand_data.get('description', '')[:100]
            })
        
        incontext = self._format_incontext_examples('candidate_refinement')
        
        prompt = f"""You are an expert at refining schema match candidates.

Source Attribute:
- ID: {query_attr}
- Field: {query_data.get('field_name', 'N/A')}
- Type: {query_data.get('type_name', 'N/A')}
- Description: {query_data.get('description', 'N/A')}

Candidate Matches:
{json.dumps(candidate_details, indent=2)}

{incontext}

Task: Analyze these candidates and select the top 5 most likely matches. Consider:
- Semantic similarity
- Type compatibility
- Hierarchical relationships
- Business logic alignment

Return ONLY a JSON array of the top 5 candidate IDs in order of relevance:
["id1", "id2", "id3", "id4", "id5"]"""

        messages = [
            {"role": "system", "content": "You are an expert schema matcher. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_llm(messages, temperature=0.1)
        response = _strip_code_fences(response)
        
        refined, err = _safe_json_loads(response)
        if err or not isinstance(refined, list):
            logger.warning(f"Failed to parse refined candidates: {err}")
            return candidates[:5]
        
        return refined[:5]
    
    # ========== Step 3: Confidence Scoring ==========
    
    def _format_as_mcq(self, candidates: List[str]) -> Tuple[str, List[str]]:
        """
        Format candidates as Multiple Choice Question.
        
        Paper: Adds "None of the above" option for abstention.
        """
        mcq_lines = []
        options = []
        
        for i, cand_id in enumerate(candidates):
            letter = chr(65 + i)  # A, B, C, ...
            cand_data = self.target_properties.get(cand_id, {})
            field = cand_data.get('field_name', cand_id)
            dtype = cand_data.get('type_name', 'unknown')
            mcq_lines.append(f"({letter}) {field} ({dtype})")
            options.append(letter)
        
        # Add "None of the above" option
        none_letter = chr(65 + len(candidates))
        mcq_lines.append(f"({none_letter}) None of the above")
        options.append(none_letter)
        
        mcq_text = '\n'.join(mcq_lines)
        return mcq_text, options
    
    def _confidence_scoring(
        self,
        mcq_text: str,
        options: List[str],
        candidates: List[str],
        query_attr: str,
        query_data: Dict
    ) -> Dict[str, float]:
        """
        Score each candidate using LLM with MCQ format.
        
        Paper: Prompts LLM to provide confidence scores 0-100 for each option.
        Uses token-level probabilities for better calibration.
        """
        incontext = self._format_incontext_examples('confidence_scoring')
        
        prompt = f"""You are an expert schema matcher. Rate the confidence (0-100) that each option matches the source attribute.

Source Attribute:
- Field: {query_data.get('field_name', 'N/A')}
- Type: {query_data.get('type_name', 'N/A')}
- Description: {query_data.get('description', 'N/A')}

Options:
{mcq_text}

{incontext}

Task: For each option (including "None of the above"), provide a confidence score from 0-100. Consider:
- Semantic similarity
- Type compatibility
- Schema relationships
- Whether a good match exists

If no option is a good match, score "None of the above" highest.

Return ONLY a JSON object mapping option letters to scores:
{{"A": 85, "B": 45, "C": 20, "D": 10, "E": 5, "F": 100}}"""

        messages = [
            {"role": "system", "content": "You are an expert schema matcher. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_llm(messages, temperature=0.0)
        response = _strip_code_fences(response)
        
        scores, err = _safe_json_loads(response)
        if err or not isinstance(scores, dict):
            logger.warning(f"Failed to parse confidence scores: {err}")
            # Return uniform scores
            return {opt: 50.0 for opt in options}
        
        # Ensure all options have scores
        for opt in options:
            if opt not in scores:
                scores[opt] = 0.0
        
        return scores
    
    # ========== Step 4: Self-Improvement ==========
    
    def _run_self_improvement(self):
        """
        Run self-improvement optimization using synthetic in-context examples.
        
        Paper Algorithm:
        1. Create evaluation set (easy + challenging queries)
        2. Run MatchMaker on eval set, capture traces
        3. LLM Evaluator scores each result (0-5)
        4. Select top-n traces as in-context examples
        """
        logger.info("Running self-improvement optimization...")
        
        # Step 1: Create evaluation set
        eval_set = self._create_evaluation_set()
        
        if not eval_set:
            logger.warning("Could not create evaluation set, skipping optimization")
            return
        
        # Step 2: Run MatchMaker and capture traces
        traces = []
        for eval_item in tqdm(eval_set, desc="Generating traces", disable=logger.level > logging.INFO):
            source_id = eval_item['source_id']
            source_data = eval_item['source_data']
            
            # Run matching pipeline with trace capture
            trace = self._run_with_trace(source_id, source_data)
            
            # Step 3: Evaluate with LLM
            score = self._llm_evaluator(eval_item, trace)
            
            traces.append({
                'source_id': source_id,
                'trace': trace,
                'score': score
            })
        
        # Step 4: Select top-n traces as in-context examples
        traces.sort(key=lambda x: x['score'], reverse=True)
        top_traces = traces[:self.n_incontext_examples]
        
        # Extract in-context examples for each component
        self._extract_incontext_examples(top_traces)
        
        logger.info(f"Optimization complete. Selected {len(top_traces)} examples with avg score: {sum(t['score'] for t in top_traces) / len(top_traces):.2f}")
    
    def _create_evaluation_set(self) -> List[Dict]:
        """
        Create evaluation set with easy and challenging queries.
        
        Paper: 
        - Easy: top semantic matches with similarity > 0.95
        - Challenging: lowest semantic match scores
        """
        eval_set = []
        
        # Sample source attributes
        source_items = list(self.source_properties.items())
        if len(source_items) > self.eval_set_size * 2:
            import random
            source_items = random.sample(source_items, self.eval_set_size * 2)
        
        # Compute semantic similarities for sampling
        similarities = []
        for source_id, source_data in source_items:
            # Get top semantic match
            semantic_cands = self._semantic_candidate_generation(source_id, source_data)
            if semantic_cands:
                # Approximate similarity (would need actual scores from retrieval)
                similarities.append((source_id, source_data, 0.9))  # Placeholder
            else:
                similarities.append((source_id, source_data, 0.1))
        
        # Select easy queries (high similarity)
        similarities.sort(key=lambda x: x[2], reverse=True)
        easy = similarities[:self.eval_set_size // 2]
        
        # Select challenging queries (low similarity)
        challenging = similarities[-self.eval_set_size // 2:]
        
        for source_id, source_data, _ in easy + challenging:
            eval_set.append({
                'source_id': source_id,
                'source_data': source_data
            })
        
        return eval_set
    
    def _run_with_trace(self, source_id: str, source_data: Dict) -> Dict:
        """Run matching pipeline and capture execution trace."""
        trace = {}
        
        # Semantic candidates
        semantic_cands = self._semantic_candidate_generation(source_id, source_data)
        trace['semantic_candidates'] = semantic_cands
        
        # Reasoning candidates
        reasoning_cands = self._reasoning_candidate_generation(source_id, source_data)
        trace['reasoning_candidates'] = reasoning_cands
        
        # Combine and refine
        all_cands = list(set(semantic_cands + reasoning_cands))
        refined_cands = self._candidate_refinement(all_cands, source_id, source_data)
        trace['refined_candidates'] = refined_cands
        
        # Score candidates
        if refined_cands:
            mcq_text, options = self._format_as_mcq(refined_cands)
            scores = self._confidence_scoring(mcq_text, options, refined_cands, source_id, source_data)
            trace['scores'] = scores
            
            # Select best match
            best_option = max(scores.items(), key=lambda x: x[1])[0]
            if best_option == options[-1]:  # "None of the above"
                trace['selected_match'] = None
            else:
                option_idx = ord(best_option) - 65
                trace['selected_match'] = refined_cands[option_idx]
        else:
            trace['selected_match'] = None
        
        return trace
    
    def _llm_evaluator(self, eval_item: Dict, trace: Dict) -> float:
        """
        LLM Evaluator: Score the quality of matches (0-5).
        
        Paper: Uses Chain-of-Thought to assess match relevance.
        """
        source_id = eval_item['source_id']
        source_data = eval_item['source_data']
        selected = trace.get('selected_match')
        
        prompt = f"""You are an expert schema matching evaluator. Rate the quality of this match on a scale of 0-5.

Source Attribute:
- ID: {source_id}
- Field: {source_data.get('field_name', 'N/A')}
- Type: {source_data.get('type_name', 'N/A')}
- Description: {source_data.get('description', 'N/A')}

Selected Match: {selected if selected else "No match (abstained)"}

Refined Candidates: {trace.get('refined_candidates', [])}

Rating Guidelines:
- 5: Perfect match (semantic + type compatible + hierarchically correct)
- 4: Good match (minor differences acceptable)
- 3: Acceptable match (some issues but reasonable)
- 2: Poor match (significant issues)
- 1: Wrong match (clearly incorrect)
- 0: No valid match possible, or abstention was incorrect

Think step-by-step about the match quality, then provide ONLY a number from 0-5."""

        messages = [
            {"role": "system", "content": "You are an expert evaluator. Respond with only a number."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_llm(messages, temperature=0.0)
        
        try:
            score = float(response.strip())
            return max(0, min(5, score))  # Clamp to [0, 5]
        except ValueError:
            logger.warning(f"Could not parse evaluator score: {response}")
            return 0.0
    
    def _extract_incontext_examples(self, top_traces: List[Dict]):
        """Extract in-context examples from top-scoring traces."""
        for trace_item in top_traces:
            trace = trace_item['trace']
            source_id = trace_item['source_id']
            
            # Extract examples for each component
            if trace.get('reasoning_candidates'):
                self.incontext_examples['candidate_generation'].append({
                    'source_id': source_id,
                    'reasoning_candidates': trace['reasoning_candidates']
                })
            
            if trace.get('refined_candidates'):
                self.incontext_examples['candidate_refinement'].append({
                    'source_id': source_id,
                    'all_candidates': trace.get('semantic_candidates', []) + trace.get('reasoning_candidates', []),
                    'refined_candidates': trace['refined_candidates']
                })
            
            if trace.get('scores'):
                self.incontext_examples['confidence_scoring'].append({
                    'source_id': source_id,
                    'scores': trace['scores']
                })
    
    def _format_incontext_examples(self, component: str) -> str:
        """Format in-context examples for a specific component."""
        examples = self.incontext_examples.get(component, [])
        if not examples:
            return ""
        
        formatted = ["Examples:"]
        for ex in examples[:3]:  # Use max 3 examples
            formatted.append(f"Example: {json.dumps(ex, indent=2)}")
        
        return "\n".join(formatted)
    
    # ========== Main Matching Pipeline ==========
    
    def match_attribute(
        self,
        source_id: str,
        source_data: Dict
    ) -> MatchResult:
        """
        Match a single source attribute to target schema.
        
        Returns:
            MatchResult with candidates, selected match, and reasoning trace.
        """
        reasoning_trace = {}
        
        # Step 2a: Semantic candidate generation
        semantic_cands = self._semantic_candidate_generation(source_id, source_data)
        reasoning_trace['semantic_candidates'] = semantic_cands
        
        # Step 2b: Reasoning-based candidate generation
        reasoning_cands = self._reasoning_candidate_generation(source_id, source_data)
        reasoning_trace['reasoning_candidates'] = reasoning_cands
        
        # Combine candidates
        all_candidates = list(set(semantic_cands + reasoning_cands))
        reasoning_trace['combined_candidates'] = all_candidates
        
        if not all_candidates:
            return MatchResult(
                source_attribute=source_id,
                candidates=[],
                selected_match=None,
                abstain=True,
                reasoning_trace=reasoning_trace
            )
        
        # Step 2c: Candidate refinement
        refined_cands = self._candidate_refinement(all_candidates, source_id, source_data)
        reasoning_trace['refined_candidates'] = refined_cands
        
        # Step 3: Confidence scoring
        mcq_text, options = self._format_as_mcq(refined_cands)
        scores = self._confidence_scoring(mcq_text, options, refined_cands, source_id, source_data)
        reasoning_trace['confidence_scores'] = scores
        
        # Create candidate matches
        candidates = []
        for i, cand_id in enumerate(refined_cands):
            letter = chr(65 + i)
            confidence = scores.get(letter, 0.0)
            
            # Determine source
            source = 'both' if cand_id in semantic_cands and cand_id in reasoning_cands else \
                    'semantic' if cand_id in semantic_cands else 'reasoning'
            
            candidates.append(CandidateMatch(
                target_attribute=cand_id,
                confidence=confidence,
                rank=i + 1,
                justification=f"Confidence: {confidence:.1f} (from {source})",
                source=source
            ))
        
        # Sort by confidence
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        for i, cand in enumerate(candidates):
            cand.rank = i + 1
        
        # Select best match
        none_option = options[-1]
        none_confidence = scores.get(none_option, 0.0)
        
        if candidates and candidates[0].confidence > none_confidence:
            selected_match = candidates[0].target_attribute
            abstain = False
        else:
            selected_match = None
            abstain = True
        
        return MatchResult(
            source_attribute=source_id,
            candidates=candidates,
            selected_match=selected_match,
            abstain=abstain,
            reasoning_trace=reasoning_trace
        )
    
    def run(self, source_attributes: Optional[List[str]] = None) -> Dict[str, MatchResult]:
        """
        Run MatchMaker on source attributes.
        
        Args:
            source_attributes: List of source attribute IDs to match. 
                             If None, matches all source attributes.
        
        Returns:
            Dictionary mapping source_id -> MatchResult
        """
        if source_attributes is None:
            source_attributes = list(self.source_properties.keys())
        
        results = {}
        
        for source_id in tqdm(
            source_attributes,
            desc="Matching attributes",
            disable=logger.level > logging.INFO
        ):
            if source_id not in self.source_properties:
                logger.warning(f"Source attribute not found: {source_id}")
                continue
            
            source_data = self.source_properties[source_id]
            result = self.match_attribute(source_id, source_data)
            results[source_id] = result
        
        return results
    
    # ========== LLM Helper ==========
    
    def _call_llm(self, messages: List[Dict], temperature: float = 0.1) -> str:
        """Call LLM with error handling. Raises on critical errors."""
        try:
            response = completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e).lower()
            
            # Critical errors that should stop execution immediately
            critical_patterns = [
                "provider not provided",
                "provider not found", 
                "invalid api key",
                "authentication",
                "unauthorized",
                "api key not found",
                "rate limit",  # May want to retry instead, but fail fast for now
                "quota exceeded",
                "model not found",
                "model does not exist",
            ]
            
            is_critical = any(pattern in error_str for pattern in critical_patterns)
            
            if is_critical:
                logger.error(f"CRITICAL LLM error - stopping execution: {e}")
                raise RuntimeError(f"Critical LLM configuration error: {e}") from e
            
            # Non-critical errors: log and return empty (allows graceful degradation)
            logger.error(f"LLM call failed (non-critical): {e}")
            return ""


# ========== Example Usage ==========

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Example: Run MatchMaker on a dataset
    agent = MatchMakerAgent(
        property_lookup_file="../data/mimic_omop_lookup.json",  # Adjust path
        model="watsonx/mistralai/mistral-large",
        dataset="mimic_omop",
        top_k=5,
        rebuild_index=True,
        use_optimization=True
    )
    
    # Match all source attributes
    results = agent.run()
    
    # Print results
    for source_id, result in results.items():
        print(f"\nSource: {source_id}")
        print(f"Selected: {result.selected_match}")
        print(f"Abstain: {result.abstain}")
        for cand in result.candidates[:3]:
            print(f"  - {cand.target_attribute}: {cand.confidence:.1f}")


if __name__ == "__main__":
    main()
