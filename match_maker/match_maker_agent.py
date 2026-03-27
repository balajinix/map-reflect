# match_maker_agent.py
"""
MatchMaker: Self-Improving Compositional Language Model Program for Schema Matching
Faithful implementation of the paper: "Bootstrapping Self-Improvement of Language Model Programs for Zero-Shot Schema Matching"

Fixes applied vs original:
- _create_evaluation_set: uses real ChromaDB similarity scores, not placeholder 0.9
- _run_self_improvement: falls back to property_data when source_properties empty
  (handles utterance-based datasets where source/target are set per sample)
- optimization is now a no-op at init for utterance datasets; call
  run_optimization_once() after the first set_source_target_from_utterance()
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

    Key design note on optimization timing:
    - For OMOP / MIMIC datasets where source/target are pre-defined at __init__,
      optimization runs automatically at the end of __init__.
    - For utterance-based datasets (itops, nl2gql, toolbench) where source/target
      are set dynamically per sample via set_source_target_from_utterance(),
      optimization is deferred. Call run_optimization_once() explicitly after the
      first set_source_target_from_utterance() call (the runner handles this).
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
        eval_set_size: int = 20,
        abstention_mode: str = 'lenient',
        min_confidence_threshold: float = 20.0,
        doc_context_chromadb: Optional[str] = None,
        doc_context_collection: str = "itopsgraph_docs",
        doc_context_top_k: int = 2,
    ):
        self.model = model
        self.dataset = dataset
        self.top_k = top_k
        self.rebuild_index = rebuild_index
        self.use_optimization = use_optimization
        self.n_incontext_examples = n_incontext_examples
        self.eval_set_size = eval_set_size
        self.abstention_mode = abstention_mode
        self.min_confidence_threshold = min_confidence_threshold
        self._optimization_done = False  # track so we run at most once

        valid_modes = ['strict', 'lenient', 'disabled', 'threshold']
        if abstention_mode not in valid_modes:
            raise ValueError(f"abstention_mode must be one of {valid_modes}, got '{abstention_mode}'")

        with open(property_lookup_file, 'r') as f:
            self.property_data = json.load(f)

        # Normalize property data: ensure product_name/type_name exist
        # ToolBench uses 'tool'/'category'/'api' instead; derive standard fields
        self._normalize_property_data()

        self.source_properties = {}
        self.target_properties = {}
        self._parse_properties()

        try:
            from ragatouille import RAGPretrainedModel
            self.colbert_available = True
            self.rag_model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        except ImportError:
            logger.warning("RAGatouille not available, falling back to sentence-transformers")
            self.colbert_available = False
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.collection_name = (
            collection_name if collection_name
            else f"matchmaker_{self.dataset.lower()}"
        )

        self.incontext_examples = {
            'candidate_generation': [],
            'candidate_refinement': [],
            'confidence_scoring': []
        }

        # Documentation context retriever (enrich index with product docs)
        self._doc_retriever = None
        if doc_context_chromadb:
            try:
                import chromadb as _chromadb
                _doc_client = _chromadb.PersistentClient(path=doc_context_chromadb)
                self._doc_collection = _doc_client.get_collection(name=doc_context_collection)
                self._doc_context_top_k = doc_context_top_k
                self._doc_retriever = True
                logger.info(f"DocContext: loaded {self._doc_collection.count()} chunks from {doc_context_chromadb}")
            except Exception as e:
                logger.warning(f"DocContext: failed to load: {e}")

        self.index_path = f"{chroma_dir}/{self.collection_name}"
        if doc_context_chromadb:
            # Force rebuild when doc context is added
            self._build_multi_vector_index()
        elif self.rebuild_index:
            self._build_multi_vector_index()
        else:
            self._load_index()

        # Utterance-based datasets: source_properties is empty at this point.
        # Optimization will run lazily via run_optimization_once() after the first
        # set_source_target_from_utterance() call.
        utterance_datasets = ['itops', 'nl2gql', 'itopsgraph', 'itom_hashicorp',
                               'omop_cms', 'omop_mimic', 'omop_synthea', 'toolbench',
                               'spider']
        is_utterance_dataset = self.dataset.lower() in utterance_datasets

        if self.use_optimization and not is_utterance_dataset:
            # Pre-defined source/target (e.g. mimic_omop): run now.
            self._run_self_improvement()
        elif self.use_optimization and is_utterance_dataset:
            logger.info(
                "Utterance-based dataset: optimization deferred until first "
                "set_source_target_from_utterance() call."
            )

        logger.info(
            f"MatchMaker initialized | dataset={dataset} model={model} "
            f"top_k={top_k} optimization={use_optimization}"
        )

    def run_optimization_once(self):
        """
        Run self-improvement exactly once.

        Safe to call multiple times; subsequent calls are no-ops.
        For utterance-based datasets, call this after the first
        set_source_target_from_utterance() so source_properties is populated.
        If source_properties is still empty, falls back to a sample of all
        property_data (gives the LLM meaningful candidate generation examples).
        """
        if self._optimization_done:
            return
        if not self.use_optimization:
            return
        self._run_self_improvement()

    def _normalize_property_data(self):
        """Ensure all property entries have product_name and type_name fields.

        Some datasets (e.g., ToolBench) use 'tool'/'api' instead of
        'product_name'/'type_name'. This normalizes them so downstream
        code can use a single field name consistently.
        """
        sample = next(iter(self.property_data.values()), {})
        has_product = bool(sample.get('product_name'))
        if has_product:
            return  # Already normalized

        for prop_id, prop_data in self.property_data.items():
            if not prop_data.get('product_name'):
                # Derive type_name from key prefix: "Foreca_Weather_Nowcast.location" -> "Foreca_Weather_Nowcast"
                key_prefix = prop_id.split('.')[0] if '.' in prop_id else prop_id
                prop_data['type_name'] = prop_data.get('api', key_prefix)
                prop_data['product_name'] = key_prefix
                if not prop_data.get('field_name'):
                    prop_data['field_name'] = prop_id.split('.')[-1] if '.' in prop_id else prop_id

    def _parse_properties(self):
        """Parse properties into source and target based on dataset."""
        if self.dataset.lower() == 'mimic_omop':
            for prop_id, prop_data in self.property_data.items():
                product = prop_data.get('product_name', '').lower()
                if 'mimic' in product:
                    self.source_properties[prop_id] = prop_data
                elif 'omop' in product:
                    self.target_properties[prop_id] = prop_data
        else:
            pass  # set dynamically per utterance

    def set_source_target_from_utterance(self, utterance: str,
                                          source_hint: str = None,
                                          target_hint: str = None):
        """Dynamically set source and target properties based on utterance."""
        available_products = list(set(
            p.get('product_name', '')
            for p in self.property_data.values()
            if p.get('product_name')
        ))

        use_type_name = len(available_products) <= 1

        # If hints are table names that exist as type_name but not product_name,
        # switch to type-level matching (e.g., Spider: utterance says "map cite
        # data to publication data" but product_name is the database name)
        if not use_type_name and source_hint and target_hint:
            available_products_lower = [p.lower() for p in available_products]
            hints_are_products = (
                source_hint.lower() in available_products_lower and
                target_hint.lower() in available_products_lower
            )
            if not hints_are_products:
                available_types = set(
                    p.get('type_name', '').lower()
                    for p in self.property_data.values()
                    if p.get('type_name')
                )
                hints_are_types = (
                    source_hint.lower() in available_types and
                    target_hint.lower() in available_types
                )
                if hints_are_types:
                    use_type_name = True
                    logger.info(f"Hints match type_name not product_name, switching to type-level matching")

        if use_type_name:
            available_tables = list(set(
                p.get('type_name', '')
                for p in self.property_data.values()
                if p.get('type_name')
            ))
            available_schemas = available_tables
            schema_field = 'type_name'
            examples = (
                '- "map concert data to stadium data" → '
                '{"source_schema": "Concert", "target_schema": "Stadium"}\n'
                '- "map singer_in_concert data to singer data" → '
                '{"source_schema": "SingerInConcert", "target_schema": "Singer"}'
            )
        else:
            available_schemas = available_products
            schema_field = 'product_name'
            examples = (
                '- "map concert vulnerabilities to turbo entities" → '
                '{"source_schema": "Concert", "target_schema": "Turbonomic"}\n'
                '- "map omop visit occurrence with cms inpatientclaims" → '
                '{"source_schema": "OMOP", "target_schema": "CMS"}'
            )

        source_schema = None
        target_schema = None

        logger.warning(f"DEBUG: source_hint={source_hint}, target_hint={target_hint}, "
                       f"available_schemas[:5]={available_schemas[:5]}")

        if source_hint and target_hint:
            source_hint_lower = source_hint.lower()
            target_hint_lower = target_hint.lower()
            available_lower = [s.lower() for s in available_schemas]
            if source_hint_lower in available_lower and target_hint_lower in available_lower:
                source_schema = source_hint_lower
                target_schema = target_hint_lower
                logger.info(f"Using provided hints → source: {source_schema}, target: {target_schema}")

        if not source_schema or not target_schema:
            utterance_lower = utterance.lower()
            utterance_words = set(utterance_lower.replace('_', ' ').split())
            matched_schemas = []
            for schema in available_schemas:
                if schema.lower() in utterance_lower or schema.lower() in utterance_words:
                    matched_schemas.append(schema)
            if len(matched_schemas) == 2:
                pos1 = utterance_lower.find(matched_schemas[0].lower())
                pos2 = utterance_lower.find(matched_schemas[1].lower())
                if pos1 < pos2:
                    source_schema = matched_schemas[0].lower()
                    target_schema = matched_schemas[1].lower()
                else:
                    source_schema = matched_schemas[1].lower()
                    target_schema = matched_schemas[0].lower()
                logger.info(f"Rule-based extraction → source: {source_schema}, target: {target_schema}")

        if not source_schema or not target_schema:
            prompt = (
                f'Parse this schema mapping request and identify the source and target schemas.\n\n'
                f'Utterance: "{utterance}"\n\n'
                f'Available {"tables" if use_type_name else "products"} in the system: '
                f'{", ".join(available_schemas[:50])}'
                f'{"..." if len(available_schemas) > 50 else ""}\n\n'
                f'Return ONLY a JSON object with:\n'
                f'{{\n  "source_schema": "source name",\n  "target_schema": "target name"\n}}\n\n'
                f'Examples:\n{examples}\n'
            )
            messages = [
                {"role": "system", "content": "You are a schema matching assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ]
            response = self._call_llm(messages, temperature=0.0)
            response = _strip_code_fences(response)
            parsed, err = _safe_json_loads(response)
            if err or not isinstance(parsed, dict):
                logger.warning(f"Failed to parse source/target from utterance: {err}")
                self.source_properties = self.property_data.copy()
                self.target_properties = self.property_data.copy()
                return
            source_schema = parsed.get('source_schema', '').lower()
            target_schema = parsed.get('target_schema', '').lower()
            logger.info(f"LLM extraction → source: {source_schema}, target: {target_schema}")

        logger.info(f"Parsed utterance → source: {source_schema}, target: {target_schema} "
                    f"(using {schema_field})")

        source_schema_normalized = source_schema.replace(' ', '_')
        target_schema_normalized = target_schema.replace(' ', '_')

        # For multi-product datasets (OMOP), also extract table names from utterance
        # Utterance format: "map omop <source_table> with cms <target_table>"
        source_table = None
        target_table = None

        if not use_type_name:
            # Get all available table names
            available_tables = list(set(
                p.get('type_name', '').lower()
                for p in self.property_data.values()
                if p.get('type_name')
            ))

            # Try to extract table names from utterance
            utterance_lower = utterance.lower().replace('_', ' ')

            # Find tables mentioned in utterance
            matched_tables = []
            for table in available_tables:
                table_normalized = table.lower().replace('_', ' ')
                if table_normalized in utterance_lower:
                    # Get position in utterance
                    pos = utterance_lower.find(table_normalized)
                    matched_tables.append((table, pos))

            # Sort by position in utterance
            matched_tables.sort(key=lambda x: x[1])

            if len(matched_tables) >= 2:
                # First table after source namespace is source_table
                # First table after target namespace is target_table
                source_pos = utterance_lower.find(source_schema_normalized)
                target_pos = utterance_lower.find(target_schema_normalized)

                for table, pos in matched_tables:
                    if source_pos < pos < target_pos and source_table is None:
                        source_table = table
                    elif pos > target_pos and target_table is None:
                        target_table = table

                logger.info(f"Extracted tables → source_table: {source_table}, target_table: {target_table}")
            elif len(matched_tables) == 1:
                # Single table - determine if it's source or target based on position
                table, pos = matched_tables[0]
                source_pos = utterance_lower.find(source_schema_normalized)
                target_pos = utterance_lower.find(target_schema_normalized)
                if target_pos >= 0 and pos > target_pos:
                    target_table = table
                elif source_pos >= 0 and pos > source_pos:
                    source_table = table
                logger.info(f"Extracted single table → source_table: {source_table}, target_table: {target_table}")

        # Filter properties by namespace and optionally by table
        if source_table:
            self.source_properties = {
                prop_id: prop_data
                for prop_id, prop_data in self.property_data.items()
                if (source_schema_normalized in prop_data.get('product_name', '').lower().replace(' ', '_')
                    and prop_data.get('type_name', '').lower().replace(' ', '_') == source_table.lower().replace(' ', '_'))
            }
        else:
            self.source_properties = {
                prop_id: prop_data
                for prop_id, prop_data in self.property_data.items()
                if source_schema_normalized in prop_data.get(schema_field, '').lower().replace(' ', '_')
            }

        if target_table:
            self.target_properties = {
                prop_id: prop_data
                for prop_id, prop_data in self.property_data.items()
                if (target_schema_normalized in prop_data.get('product_name', '').lower().replace(' ', '_')
                    and prop_data.get('type_name', '').lower().replace(' ', '_') == target_table.lower().replace(' ', '_'))
            }
        else:
            self.target_properties = {
                prop_id: prop_data
                for prop_id, prop_data in self.property_data.items()
                if target_schema_normalized in prop_data.get(schema_field, '').lower().replace(' ', '_')
            }

        logger.info(
            f"Filtered properties: {len(self.source_properties)} source, "
            f"{len(self.target_properties)} target"
            f"{' (table-filtered)' if source_table or target_table else ''}"
        )

        if not self.source_properties or not self.target_properties:
            logger.warning("No properties found, trying lenient match")
            source_word = (source_schema_normalized.split('_')[0]
                           if '_' in source_schema_normalized else source_schema_normalized)
            target_word = (target_schema_normalized.split('_')[0]
                           if '_' in target_schema_normalized else target_schema_normalized)
            if not self.source_properties:
                self.source_properties = {
                    prop_id: prop_data
                    for prop_id, prop_data in self.property_data.items()
                    if source_word in prop_data.get(schema_field, '').lower().replace(' ', '_')
                }
            if not self.target_properties:
                self.target_properties = {
                    prop_id: prop_data
                    for prop_id, prop_data in self.property_data.items()
                    if target_word in prop_data.get(schema_field, '').lower().replace(' ', '_')
                }
            logger.info(
                f"After lenient match: {len(self.source_properties)} source, "
                f"{len(self.target_properties)} target"
            )

        utterance_datasets = ['itops', 'nl2gql', 'itopsgraph', 'itom_hashicorp',
                               'omop_cms', 'omop_mimic', 'omop_synthea', 'toolbench',
                               'spider']
        if self.dataset.lower() in utterance_datasets:
            logger.info("Utterance-based dataset: skipping index rebuild")
        elif self.colbert_available:
            self._build_multi_vector_index()
        elif hasattr(self, 'collection'):
            self._build_multi_vector_index()

    # ========== Step 1: Multi-Vector Index Creation ==========

    def _build_multi_vector_index(self):
        """Build ColBERT-v2 multi-vector index from target schema."""
        logger.info(f"Building multi-vector index: {self.collection_name}")

        utterance_datasets = ['itops', 'nl2gql', 'itopsgraph', 'omop_cms',
                               'omop_mimic', 'omop_synthea']
        if self.dataset.lower() in utterance_datasets:
            properties_to_index = self.property_data
            logger.info(f"Utterance-based dataset: indexing all {len(properties_to_index)} properties")
        else:
            properties_to_index = self.target_properties
            logger.info(f"Attribute-based dataset: indexing {len(properties_to_index)} target properties")

        documents = []
        doc_ids = []
        for prop_id, prop_data in tqdm(
            properties_to_index.items(),
            desc="Preparing documents for indexing",
            disable=logger.level > logging.INFO
        ):
            doc = self._create_document(prop_id, prop_data)
            documents.append(doc)
            doc_ids.append(prop_id)

        if self.colbert_available:
            self.rag_model.index(
                collection=documents,
                document_ids=doc_ids,
                index_name=self.collection_name,
                max_document_length=256,
                split_documents=False
            )
        else:
            import chromadb
            if not hasattr(self, 'chroma_client') or self.chroma_client is None:
                self.chroma_client = chromadb.PersistentClient(path=self.index_path)
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name
            )
            embeddings = [self.embedding_model.encode(doc) for doc in tqdm(
                documents,
                desc="Embedding documents",
                disable=logger.level > logging.INFO
            )]
            batch_size = 5000
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                collection.add(
                    ids=doc_ids[i:batch_end],
                    documents=documents[i:batch_end],
                    embeddings=embeddings[i:batch_end]
                )
            self.collection = collection

        logger.info(f"Index built with {len(documents)} properties")

    def _load_index(self):
        """Load existing index, or create if it doesn't exist."""
        logger.info(f"Loading existing index: {self.collection_name}")
        if self.colbert_available:
            pass
        else:
            import chromadb
            if not hasattr(self, 'chroma_client') or self.chroma_client is None:
                self.chroma_client = chromadb.PersistentClient(path=self.index_path)
            try:
                self.collection = self.chroma_client.get_collection(self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception as e:
                logger.warning(f"Collection {self.collection_name} doesn't exist, creating now...")
                self._build_multi_vector_index()

    def _create_document(self, prop_id: str, prop_data: Dict) -> str:
        """Create document from property for indexing.

        When doc context is available, enriches the document with
        relevant chunks from product documentation.
        """
        parts = [
            prop_data.get('product_name', ''),
            prop_data.get('type_name', ''),
            prop_data.get('field_name', ''),
            prop_data.get('description', '')
        ]
        base_doc = ' '.join(filter(None, parts))

        # Enrich with documentation context
        if self._doc_retriever and hasattr(self, '_doc_collection'):
            try:
                product = prop_data.get('product_name', '')
                query = f"{prop_data.get('type_name', '')} {prop_data.get('field_name', '')}"
                where_filter = {"product": product} if product else None
                results = self._doc_collection.query(
                    query_texts=[query],
                    n_results=self._doc_context_top_k,
                    where=where_filter,
                    include=["documents"],
                )
                if results and results["documents"] and results["documents"][0]:
                    doc_text = " ".join(
                        d[:200] for d in results["documents"][0]
                    )
                    base_doc = f"{base_doc} {doc_text}"
            except Exception:
                pass  # Fall back to base document

        return base_doc

    # ========== Step 2: Candidate Generation ==========

    def _semantic_candidate_generation(
        self,
        query_attr: str,
        query_data: Dict
    ) -> List[str]:
        """Generate semantic retrieval candidates using multi-vector retrieval."""
        query_doc = self._create_document(query_attr, query_data)

        if self.colbert_available:
            results = self.rag_model.search(
                query=query_doc,
                index_name=self.collection_name,
                k=self.top_k
            )
            candidates = [r['document_id'] for r in results]
        else:
            query_embedding = self.embedding_model.encode(query_doc)
            source_prop_count = len(self.source_properties) if self.source_properties else 0
            n_request = min(self.top_k + source_prop_count + 10, 500)
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_request
            )
            candidates = results['ids'][0] if results.get('ids') else []

        source_prop_ids = set(self.source_properties.keys()) if self.source_properties else set()
        target_prop_ids = set(self.target_properties.keys()) if self.target_properties else set()

        filtered_candidates = [
            c for c in candidates
            if c != query_attr
            and c not in source_prop_ids
            and (not target_prop_ids or c in target_prop_ids)
        ][:self.top_k]

        if not filtered_candidates and candidates:
            logger.warning(f"All {len(candidates)} semantic candidates were source properties")

        logger.debug(f"Semantic candidates for {query_attr}: {filtered_candidates}")
        return filtered_candidates

    def _semantic_candidate_generation_with_scores(
        self,
        query_attr: str,
        query_data: Dict
    ) -> List[Tuple[str, float]]:
        """
        Like _semantic_candidate_generation but returns (candidate_id, score) tuples.

        Used by the evaluation set creator to get real similarity scores for
        distinguishing easy vs challenging queries (paper Section 4.4).
        """
        if self.colbert_available:
            query_doc = self._create_document(query_attr, query_data)
            results = self.rag_model.search(
                query=query_doc,
                index_name=self.collection_name,
                k=self.top_k
            )
            candidates_with_scores = [(r['document_id'], r.get('score', 0.0)) for r in results]
        else:
            query_doc = self._create_document(query_attr, query_data)
            query_embedding = self.embedding_model.encode(query_doc)
            source_prop_count = len(self.source_properties) if self.source_properties else 0
            n_request = min(self.top_k + source_prop_count + 10, 500)
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_request,
                include=['distances']
            )
            ids = results['ids'][0] if results.get('ids') else []
            # ChromaDB returns L2 distances; convert to cosine-like score in [0,1]
            dists = results.get('distances', [[]])[0]
            scores = [max(0.0, 1.0 - d / 2.0) for d in dists]
            candidates_with_scores = list(zip(ids, scores))

        # Apply same filtering as the base method
        source_prop_ids = set(self.source_properties.keys()) if self.source_properties else set()
        target_prop_ids = set(self.target_properties.keys()) if self.target_properties else set()

        filtered = [
            (c, s) for c, s in candidates_with_scores
            if c != query_attr
            and c not in source_prop_ids
            and (not target_prop_ids or c in target_prop_ids)
        ][:self.top_k]

        return filtered

    def _reasoning_candidate_generation(
        self,
        query_attr: str,
        query_data: Dict
    ) -> List[str]:
        """Generate reasoning-based candidates using LLM with Chain-of-Thought."""
        target_context = self._prepare_target_context()
        incontext = self._format_incontext_examples('candidate_generation')

        prompt = (
            "You are an expert schema matcher. Given a source attribute, reason about which "
            "target attributes might match based on semantic similarity, schema hierarchy and "
            "relationships, data types and constraints, and domain knowledge.\n\n"
            f"Source Attribute:\n"
            f"- ID: {query_attr}\n"
            f"- Product: {query_data.get('product_name', 'N/A')}\n"
            f"- Type: {query_data.get('type_name', 'N/A')}\n"
            f"- Field: {query_data.get('field_name', 'N/A')}\n"
            f"- Description: {query_data.get('description', 'N/A')}\n\n"
            f"Target Schema Context:\n{target_context}\n\n"
            f"{incontext}\n\n"
            f"Task: Reason step-by-step about which target attributes are most likely to match. "
            f"Return ONLY a JSON array of target attribute IDs (up to {self.top_k} candidates):\n"
            f'["target_id_1", "target_id_2", ...]'
        )

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

        valid_candidates = [c for c in candidates if c in self.target_properties][:self.top_k]
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
        """Refine combined candidates to top-5 most relevant."""
        if len(candidates) <= 5:
            return candidates

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

        prompt = (
            "You are an expert at refining schema match candidates.\n\n"
            f"Source Attribute:\n"
            f"- ID: {query_attr}\n"
            f"- Field: {query_data.get('field_name', 'N/A')}\n"
            f"- Type: {query_data.get('type_name', 'N/A')}\n"
            f"- Description: {query_data.get('description', 'N/A')}\n\n"
            f"Candidate Matches:\n{json.dumps(candidate_details, indent=2)}\n\n"
            f"{incontext}\n\n"
            "Task: Analyze these candidates and select the top 5 most likely matches. "
            "Consider semantic similarity, type compatibility, hierarchical relationships, "
            "and business logic alignment.\n\n"
            "Return ONLY a JSON array of the top 5 candidate IDs in order of relevance:\n"
            '["id1", "id2", "id3", "id4", "id5"]'
        )

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
        """Format candidates as Multiple Choice Question with None-of-above option."""
        mcq_lines = []
        options = []

        for i, cand_id in enumerate(candidates):
            letter = chr(65 + i)
            cand_data = self.target_properties.get(cand_id, {})
            field = cand_data.get('field_name', cand_id)
            dtype = cand_data.get('type_name', 'unknown')
            mcq_lines.append(f"({letter}) {field} ({dtype})")
            options.append(letter)

        none_letter = chr(65 + len(candidates))
        mcq_lines.append(f"({none_letter}) None of the above")
        options.append(none_letter)

        return '\n'.join(mcq_lines), options

    def _confidence_scoring(
        self,
        mcq_text: str,
        options: List[str],
        candidates: List[str],
        query_attr: str,
        query_data: Dict
    ) -> Dict[str, float]:
        """Score each candidate using LLM with MCQ format (0-100 per option)."""
        incontext = self._format_incontext_examples('confidence_scoring')

        prompt = (
            "You are an expert schema matcher. Rate the confidence (0-100) that each option "
            "matches the source attribute.\n\n"
            f"Source Attribute:\n"
            f"- Field: {query_data.get('field_name', 'N/A')}\n"
            f"- Type: {query_data.get('type_name', 'N/A')}\n"
            f"- Description: {query_data.get('description', 'N/A')}\n\n"
            f"Options:\n{mcq_text}\n\n"
            f"{incontext}\n\n"
            "Task: For each option (including 'None of the above'), provide a confidence score "
            "from 0-100. Scoring guidelines:\n"
            "- 80-100: Strong match\n"
            "- 60-79: Good match\n"
            "- 40-59: Moderate match\n"
            "- 20-39: Weak match\n"
            "- 0-19: Poor match\n\n"
            "For 'None of the above': Only score it high (>50) if ALL other options are "
            "clearly wrong or incompatible. Prefer making a match when a reasonable candidate exists.\n\n"
            "Return ONLY a JSON object mapping option letters to scores:\n"
            '{"A": 85, "B": 45, "C": 20, "D": 10, "E": 5, "F": 15}'
        )

        messages = [
            {"role": "system", "content": "You are an expert schema matcher. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, temperature=0.0)
        response = _strip_code_fences(response)

        scores, err = _safe_json_loads(response)
        if err or not isinstance(scores, dict):
            logger.warning(f"Failed to parse confidence scores: {err}")
            return {opt: 50.0 for opt in options}

        for opt in options:
            if opt not in scores:
                scores[opt] = 0.0

        return scores

    # ========== Step 4: Self-Improvement ==========

    def _run_self_improvement(self):
        """
        Run self-improvement optimization using synthetic in-context examples.

        Paper Algorithm 1:
        1. Create evaluation set (easy + challenging queries)
        2. Run MatchMaker on eval set, capture traces
        3. LLM Evaluator scores each result (0-5)
        4. Select top-n traces as in-context examples for each pipeline stage

        For utterance-based datasets where source_properties may be empty,
        falls back to a random sample from property_data so the LLM gets
        meaningful examples of the candidate generation / refinement / scoring
        pipeline.
        """
        logger.info("Running self-improvement optimization...")

        eval_set = self._create_evaluation_set()

        if not eval_set:
            logger.warning("Could not create evaluation set, skipping optimization")
            self._optimization_done = True
            return

        traces = []
        for eval_item in tqdm(eval_set, desc="Generating optimization traces",
                              disable=logger.level > logging.INFO):
            source_id = eval_item['source_id']
            source_data = eval_item['source_data']
            trace = self._run_with_trace(source_id, source_data)
            score = self._llm_evaluator(eval_item, trace)
            traces.append({'source_id': source_id, 'trace': trace, 'score': score})

        traces.sort(key=lambda x: x['score'], reverse=True)
        top_traces = traces[:self.n_incontext_examples]

        self._extract_incontext_examples(top_traces)
        self._optimization_done = True

        avg_score = sum(t['score'] for t in top_traces) / len(top_traces) if top_traces else 0.0
        logger.info(
            f"Optimization complete. Selected {len(top_traces)} in-context examples "
            f"(avg evaluator score: {avg_score:.2f}/5.0)"
        )

    def _create_evaluation_set(self) -> List[Dict]:
        """
        Create evaluation set with easy and challenging queries.

        Paper: easy = top semantic matches with similarity > 0.95;
               challenging = lowest semantic match scores.

        Fix over original: uses real retrieval scores from ChromaDB/ColBERT
        instead of the placeholder 0.9 that made all queries look equally easy.

        Fallback: if source_properties is empty (utterance-based datasets at
        init time), samples from all property_data so optimization can still
        produce meaningful in-context examples.
        """
        # Determine property pool for eval set
        if self.source_properties:
            source_pool = list(self.source_properties.items())
            logger.info(f"Optimization eval pool: {len(source_pool)} source properties")
        else:
            # Utterance-based dataset — use a sample of all properties as a proxy
            all_items = list(self.property_data.items())
            import random
            random.shuffle(all_items)
            source_pool = all_items[:min(self.eval_set_size * 3, len(all_items))]
            logger.info(
                f"Optimization eval pool: {len(source_pool)} sampled from full property_data "
                f"(source_properties empty at optimization time)"
            )

        if not source_pool:
            return []

        # Sample down to a manageable size for the similarity pass
        import random
        if len(source_pool) > self.eval_set_size * 2:
            source_pool = random.sample(source_pool, self.eval_set_size * 2)

        # Compute real similarity scores from retrieval
        # (paper: "easy queries where top-n matches have similarity > 0.95")
        similarities = []
        for source_id, source_data in source_pool:
            scored_cands = self._semantic_candidate_generation_with_scores(source_id, source_data)
            # Use the top-1 similarity score as the query difficulty indicator
            top_score = scored_cands[0][1] if scored_cands else 0.0
            similarities.append((source_id, source_data, top_score))

        # Sort by score descending
        similarities.sort(key=lambda x: x[2], reverse=True)

        # Easy: highest similarity scores (paper threshold: > 0.95)
        easy = similarities[:self.eval_set_size // 2]

        # Challenging: lowest similarity scores
        challenging = similarities[-(self.eval_set_size // 2):]

        eval_set = []
        for source_id, source_data, score in easy + challenging:
            eval_set.append({'source_id': source_id, 'source_data': source_data,
                              'top_similarity': score})

        easy_avg = sum(x[2] for x in easy) / len(easy) if easy else 0.0
        hard_avg = sum(x[2] for x in challenging) / len(challenging) if challenging else 0.0
        logger.info(
            f"Eval set: {len(easy)} easy (avg sim {easy_avg:.3f}), "
            f"{len(challenging)} challenging (avg sim {hard_avg:.3f})"
        )

        return eval_set

    def _run_with_trace(self, source_id: str, source_data: Dict) -> Dict:
        """Run matching pipeline and capture execution trace."""
        trace = {}

        semantic_cands = self._semantic_candidate_generation(source_id, source_data)
        trace['semantic_candidates'] = semantic_cands

        reasoning_cands = self._reasoning_candidate_generation(source_id, source_data)
        trace['reasoning_candidates'] = reasoning_cands

        all_cands = list(set(semantic_cands + reasoning_cands))
        refined_cands = self._candidate_refinement(all_cands, source_id, source_data)
        trace['refined_candidates'] = refined_cands

        if refined_cands:
            mcq_text, options = self._format_as_mcq(refined_cands)
            scores = self._confidence_scoring(mcq_text, options, refined_cands, source_id, source_data)
            trace['scores'] = scores

            best_option = max(scores.items(), key=lambda x: x[1])[0]
            if best_option == options[-1]:
                trace['selected_match'] = None
            else:
                option_idx = ord(best_option) - 65
                trace['selected_match'] = refined_cands[option_idx]
        else:
            trace['selected_match'] = None

        return trace

    def _llm_evaluator(self, eval_item: Dict, trace: Dict) -> float:
        """LLM Evaluator: score match quality 0-5 (paper Section 4.4)."""
        source_id = eval_item['source_id']
        source_data = eval_item['source_data']
        selected = trace.get('selected_match')

        prompt = (
            "You are an expert schema matching evaluator. Rate the quality of this match "
            "on a scale of 0-5.\n\n"
            f"Source Attribute:\n"
            f"- ID: {source_id}\n"
            f"- Field: {source_data.get('field_name', 'N/A')}\n"
            f"- Type: {source_data.get('type_name', 'N/A')}\n"
            f"- Description: {source_data.get('description', 'N/A')}\n\n"
            f"Selected Match: {selected if selected else 'No match (abstained)'}\n\n"
            f"Refined Candidates: {trace.get('refined_candidates', [])}\n\n"
            "Rating Guidelines:\n"
            "- 5: Perfect match (semantic + type compatible + hierarchically correct)\n"
            "- 4: Good match (minor differences acceptable)\n"
            "- 3: Acceptable match (some issues but reasonable)\n"
            "- 2: Poor match (significant issues)\n"
            "- 1: Wrong match (clearly incorrect)\n"
            "- 0: No valid match possible, or abstention was incorrect\n\n"
            "Think step-by-step, then provide ONLY a number from 0-5."
        )

        messages = [
            {"role": "system", "content": "You are an expert evaluator. Respond with only a number."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, temperature=0.0)
        try:
            score = float(response.strip())
            return max(0.0, min(5.0, score))
        except ValueError:
            logger.warning(f"Could not parse evaluator score: {response}")
            return 0.0

    def _extract_incontext_examples(self, top_traces: List[Dict]):
        """Extract per-component in-context examples from top-scoring traces."""
        for trace_item in top_traces:
            trace = trace_item['trace']
            source_id = trace_item['source_id']

            if trace.get('reasoning_candidates'):
                self.incontext_examples['candidate_generation'].append({
                    'source_id': source_id,
                    'reasoning_candidates': trace['reasoning_candidates']
                })

            if trace.get('refined_candidates'):
                self.incontext_examples['candidate_refinement'].append({
                    'source_id': source_id,
                    'all_candidates': (trace.get('semantic_candidates', [])
                                       + trace.get('reasoning_candidates', [])),
                    'refined_candidates': trace['refined_candidates']
                })

            if trace.get('scores'):
                self.incontext_examples['confidence_scoring'].append({
                    'source_id': source_id,
                    'scores': trace['scores']
                })

    def _format_incontext_examples(self, component: str) -> str:
        """Format in-context examples for a specific pipeline stage."""
        examples = self.incontext_examples.get(component, [])
        if not examples:
            return ""
        formatted = ["Examples from prior runs:"]
        for ex in examples[:3]:
            formatted.append(f"Example: {json.dumps(ex, indent=2)}")
        return "\n".join(formatted)

    # ========== Main Matching Pipeline ==========

    def match_attribute(self, source_id: str, source_data: Dict) -> MatchResult:
        """Match a single source attribute to target schema."""
        reasoning_trace = {}

        semantic_cands = self._semantic_candidate_generation(source_id, source_data)
        reasoning_trace['semantic_candidates'] = semantic_cands

        reasoning_cands = self._reasoning_candidate_generation(source_id, source_data)
        reasoning_trace['reasoning_candidates'] = reasoning_cands

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

        refined_cands = self._candidate_refinement(all_candidates, source_id, source_data)
        reasoning_trace['refined_candidates'] = refined_cands

        mcq_text, options = self._format_as_mcq(refined_cands)
        scores = self._confidence_scoring(mcq_text, options, refined_cands, source_id, source_data)
        reasoning_trace['confidence_scores'] = scores

        candidates = []
        for i, cand_id in enumerate(refined_cands):
            letter = chr(65 + i)
            confidence = scores.get(letter, 0.0)
            source_type = (
                'both' if cand_id in semantic_cands and cand_id in reasoning_cands
                else 'semantic' if cand_id in semantic_cands else 'reasoning'
            )
            candidates.append(CandidateMatch(
                target_attribute=cand_id,
                confidence=confidence,
                rank=i + 1,
                justification=f"Confidence: {confidence:.1f} (from {source_type})",
                source=source_type
            ))

        candidates.sort(key=lambda x: x.confidence, reverse=True)
        for i, cand in enumerate(candidates):
            cand.rank = i + 1

        none_option = options[-1]
        none_confidence = scores.get(none_option, 0.0)
        best_confidence = candidates[0].confidence if candidates else 0.0

        if self.abstention_mode == 'disabled':
            abstain = not candidates
        elif self.abstention_mode == 'threshold':
            abstain = best_confidence < self.min_confidence_threshold
        elif self.abstention_mode == 'lenient':
            abstain = (best_confidence < self.min_confidence_threshold
                       and best_confidence <= none_confidence)
        else:  # strict — original paper behavior
            abstain = not candidates or best_confidence <= none_confidence

        if candidates and not abstain:
            selected_match = candidates[0].target_attribute
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
        """Run MatchMaker on source attributes."""
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
            critical_patterns = [
                "provider not provided", "provider not found",
                "invalid api key", "authentication", "unauthorized",
                "api key not found", "rate limit", "quota exceeded",
                "model not found", "model does not exist",
            ]
            if any(p in error_str for p in critical_patterns):
                logger.error(f"CRITICAL LLM error - stopping: {e}")
                raise RuntimeError(f"Critical LLM configuration error: {e}") from e
            logger.error(f"LLM call failed (non-critical): {e}")
            return ""
