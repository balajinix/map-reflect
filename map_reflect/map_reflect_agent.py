# schema_mapping/map_reflect_agent.py
import json
import networkx as nx

# Workaround for old sqlite3 on some systems
try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, TypedDict, Optional
import re
from dataclasses import dataclass
from collections import defaultdict
import logging
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Always import litellm (needed for Mistral models which work better with chat API)
from litellm import completion

# Try native SDK for Granite/Llama models
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
    from ibm_watsonx_ai.foundation_models import ModelInference
    USE_LITELLM = False
except ImportError:
    USE_LITELLM = True

load_dotenv()

# ---------- config loading ----------

def load_dataset_config(dataset: str = None, config_path: str = None) -> Dict[str, Any]:
    """Load dataset-specific configuration from config.yaml.

    Args:
        dataset: Dataset name (e.g., 'OMOP_MIMIC', 'ITOps')
        config_path: Path to config file (default: config.yaml in same directory)

    Returns:
        Configuration dict with hyperparameters
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    default_config = {
        "top_k": 15,
        "neighbor_k": 5,
        "n_results_multiplier": 10,
        "strict_schema_filter": False,
        "product_boost": 100,
        "product_penalty": -50,
        "type_exact_boost": 20,
        "type_partial_boost": 10,
        "type_field_boost": 5,
    }

    try:
        with open(config_path, 'r') as f:
            all_config = yaml.safe_load(f)

        # Get dataset-specific config or default
        if dataset and dataset in all_config:
            config = all_config[dataset]
            logger.info(f"Loaded config for dataset '{dataset}'")
        elif 'default' in all_config:
            config = all_config['default']
            logger.info(f"Using default config (dataset '{dataset}' not found)")
        else:
            config = default_config
            logger.warning(f"No config found, using hardcoded defaults")

        # Merge with defaults to ensure all keys exist
        return {**default_config, **config}

    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
        return default_config

# Stable logger namespace
logger = logging.getLogger("solis_ai_api.map_reflect")

# ---------- helpers ----------

def _strip_code_fences(s: str) -> str:
    """Remove Markdown code fences from model output before json.loads."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.replace("```json", "").replace("```", "").strip()

def _safe_json_loads(payload: str):
    """Try json.loads; return (obj_or_none, error_str_or_none)."""
    try:
        return json.loads(payload), None
    except Exception as e:
        return None, str(e)

# ---------- types ----------

class MappingState(TypedDict):
    user_utterance: str
    property_data: Dict[str, Any]
    source_properties: List[str]
    destination_properties: List[str]
    candidate_mappings: Dict[str, List[Dict[str, str]]]
    final_mappings: Dict[str, List[Dict[str, str]]]
    reflection_feedback: str
    graph: nx.Graph
    chroma_collection: Any
    source_context: str
    destination_context: str
    # New fields for improved schema selection
    source_product: str
    target_product: str
    source_type_hints: List[str]
    target_type_hints: List[str]

@dataclass
class PropertyMapping:
    source_property: str
    destination_property: str
    justification: str
    confidence_score: float = 0.0

# ---------- agent ----------

class MapReflectAgent:
    def __init__(
        self,
        property_lookup_file: str,
        rebuild_index: bool = False,
        model: str = None,
        dataset: Optional[str] = None,
        chroma_dir: str = "../data/chroma_db",
        collection_name: Optional[str] = None,
        top_k: int = None,
        neighbor_k: int = None,
    ):
        """
        Initialize MapReflect agent.

        Args:
            property_lookup_file: Path to property lookup JSON file (required)
            rebuild_index: Whether to rebuild the ChromaDB index
            model: LLM model to use (if None, uses default from environment or fallback)
            dataset: Dataset name for namespaced collection (optional)
            chroma_dir: Directory for ChromaDB storage
            collection_name: Override collection name (optional)
            top_k: Number of top results to retrieve (overrides config if provided)
            neighbor_k: Number of neighbors to expand (overrides config if provided)
        """
        self.model = model
        self.dataset = dataset
        self.rebuild_index = rebuild_index

        # Load dataset-specific config
        self.config = load_dataset_config(dataset)
        logger.info("Loaded config for dataset=%s: %s", dataset, self.config)

        # Use provided values or fall back to config
        self.top_k = max(1, int(top_k)) if top_k is not None else self.config["top_k"]
        self.neighbor_k = max(0, int(neighbor_k)) if neighbor_k is not None else self.config["neighbor_k"]
        self.n_results_multiplier = self.config["n_results_multiplier"]
        self.strict_schema_filter = self.config["strict_schema_filter"]

        # Chroma client & embeddings
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load properties
        with open(property_lookup_file, "r") as f:
            self.property_data = json.load(f)

        # Namespace collection per dataset unless overridden
        self.collection_name = (
            collection_name
            if collection_name
            else (f"property_index_{self.dataset.lower()}" if self.dataset else "property_index")
        )

        logger.info(
            "MapReflect init | dataset=%s model=%s chroma_dir=%s collection=%s props=%d",
            self.dataset or "default", self.model, chroma_dir, self.collection_name, len(self.property_data)
        )

        # Pre-build property graph once (O(n²) operation - caching saves ~10s per sample)
        self._cached_property_graph = self._build_property_graph_once()
        logger.info("Pre-built property graph | nodes=%d edges=%d",
                    self._cached_property_graph.number_of_nodes(),
                    self._cached_property_graph.number_of_edges())

        # Build workflow
        self.graph = self._build_graph()

    # ---- LLM wrapper ----
    def _format_prompt_for_model(self, messages, model_id):
        """Format messages based on model-specific chat template.

        Args:
            messages: List of dicts with 'role' and 'content' keys
            model_id: Model identifier string

        Returns:
            Tuple of (formatted_prompt, stop_sequences)
        """
        model_lower = model_id.lower()

        # Granite 4 models (granite-4-h-small, etc.)
        if "granite-4" in model_lower or "granite4" in model_lower:
            prompt_parts = []
            for m in messages:
                role = m.get('role', 'user')
                content = m.get('content', '')
                prompt_parts.append(f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n")
            prompt_parts.append("<|start_of_role|>assistant<|end_of_role|>")
            return "".join(prompt_parts), ["<|end_of_text|>"]

        # Mistral models
        elif "mistral" in model_lower:
            system_msg = ""
            user_content = []
            for m in messages:
                if m.get('role') == 'system':
                    system_msg = m.get('content', '')
                elif m.get('role') == 'user':
                    user_content.append(m.get('content', ''))

            combined_user = "\n\n".join(user_content)
            if system_msg:
                prompt = f"[INST] {system_msg}\n\n{combined_user} [/INST]"
            else:
                prompt = f"[INST] {combined_user} [/INST]"
            return prompt, ["[/INST]", "</s>"]

        # Llama 3 models
        elif "llama" in model_lower:
            prompt_parts = ["<|begin_of_text|>"]
            for m in messages:
                role = m.get('role', 'user')
                content = m.get('content', '')
                prompt_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
            prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(prompt_parts), ["<|eot_id|>", "<|end_of_text|>"]

        # Granite 3.x models (granite-3-2-8b, granite-3-3-8b, etc.)
        elif "granite-3" in model_lower or "granite3" in model_lower:
            prompt_parts = []
            for m in messages:
                role = m.get('role', 'user')
                content = m.get('content', '')
                prompt_parts.append(f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n")
            prompt_parts.append("<|start_of_role|>assistant<|end_of_role|>")
            return "".join(prompt_parts), ["<|end_of_text|>"]

        # Default fallback for unknown models
        else:
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(content)
            return "\n\n".join(prompt_parts), ["<|endoftext|>"]

    def call_llm(self, messages):
        """Call LLM using native WatsonX SDK or litellm fallback."""
        import time
        try:
            t0 = time.time()

            # Get credentials
            api_key = os.getenv("WATSONX_API_KEY") or os.getenv("WATSONX_APIKEY")
            project_id = os.getenv("WATSONX_PROJECT_ID")
            url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

            if not api_key or not project_id:
                raise RuntimeError("WATSONX_API_KEY and WATSONX_PROJECT_ID required")

            # Remove watsonx/ prefix if present
            model_id = self.model.replace("watsonx/", "")

            # Use litellm for Mistral models (works better with chat API) or if native SDK not available
            use_litellm_for_model = USE_LITELLM or 'mistral' in model_id.lower()

            if use_litellm_for_model:
                t1 = time.time()
                logger.warning(f"LLM_TIMING | prompt_format: {t1-t0:.2f}s (litellm mode)")

                # Convert to litellm model format
                litellm_model = f"watsonx/{model_id}"

                response = completion(
                    model=litellm_model,
                    messages=messages,
                    temperature=0,  # Use 0 for deterministic responses
                    max_tokens=8000,
                )
                t2 = time.time()
                logger.warning(f"LLM_TIMING | generate: {t2-t1:.2f}s | total: {t2-t0:.2f}s")

                return response.choices[0].message.content

            # Format prompt based on model type
            prompt, stop_sequences = self._format_prompt_for_model(messages, model_id)

            t1 = time.time()
            logger.warning(f"LLM_TIMING | prompt_format: {t1-t0:.2f}s")

            # Call WatsonX - reuse cached model if available
            if not hasattr(self, '_cached_model') or self._cached_model_id != model_id:
                parameters = {
                    GenParams.MAX_NEW_TOKENS: 8000,
                    GenParams.TEMPERATURE: 0.1,
                    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
                    GenParams.STOP_SEQUENCES: stop_sequences,
                }

                credentials = Credentials(url=url, api_key=api_key)
                self._cached_model = ModelInference(
                    model_id=model_id,
                    params=parameters,
                    credentials=credentials,
                    project_id=project_id
                )
                self._cached_model_id = model_id
                t2 = time.time()
                logger.warning(f"LLM_TIMING | model_init: {t2-t1:.2f}s (new)")
            else:
                t2 = time.time()
                logger.warning(f"LLM_TIMING | model_init: {t2-t1:.2f}s (cached)")

            response = self._cached_model.generate_text(prompt=prompt)
            t3 = time.time()
            logger.warning(f"LLM_TIMING | generate: {t3-t2:.2f}s | total: {t3-t0:.2f}s")

            return response[0] if isinstance(response, list) else response

        except Exception as e:
            error_str = str(e).lower()
            critical_patterns = ["authentication", "unauthorized", "api key", "project", "exceeded limit"]

            if any(p in error_str for p in critical_patterns):
                logger.error(f"CRITICAL LLM error - stopping execution: {e}")
                raise RuntimeError(f"Critical LLM error: {e}") from e

            logger.error(f"LLM error: {e}", exc_info=True)
            return f"Error calling LLM: {e}"


    # ---- LangGraph DAG ----
    def _build_graph(self):
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(MappingState)
        if self.rebuild_index:
            workflow.add_node("initialize_index", self._initialize_index)
        workflow.add_node("extract_contexts", self._extract_contexts)
        workflow.add_node("create_graph", self._create_property_graph)
        workflow.add_node("select_source_properties", self._select_source_properties)
        workflow.add_node("select_destination_properties", self._select_destination_properties)
        workflow.add_node("find_mappings", self._find_mappings)
        workflow.add_node("reflect_and_prune", self._reflect_and_prune)
        workflow.add_node("format_output", self._format_output)

        if self.rebuild_index:
            workflow.add_edge("initialize_index", "extract_contexts")
            workflow.set_entry_point("initialize_index")
        else:
            workflow.set_entry_point("extract_contexts")

        workflow.add_edge("extract_contexts", "create_graph")
        workflow.add_edge("create_graph", "select_source_properties")
        workflow.add_edge("select_source_properties", "select_destination_properties")
        workflow.add_edge("select_destination_properties", "find_mappings")
        # the following is taking too much time.
        #workflow.add_edge("find_mappings", "reflect_and_prune")
        #workflow.add_edge("reflect_and_prune", "format_output")
        workflow.add_edge("find_mappings", "format_output")
        workflow.add_edge("format_output", END)
        return workflow.compile()

    # ---- Nodes ----
    def _initialize_index(self, state: MappingState) -> MappingState:
        state["chroma_collection"] = self._create_collection_from_state(state)
        return state

    def _extract_contexts(self, state: MappingState) -> MappingState:
        user_utterance = state["user_utterance"]
        logger.info("Extracting contexts from utterance")

        # First, extract schema hints directly from utterance using pattern matching
        source_hints, target_hints = self._parse_utterance_for_schemas(user_utterance)

        prompt = f"""Analyze this schema mapping request: "{user_utterance}"

        Extract the specific source and destination schema/type information.

        The request follows the pattern: "map [source_product] [source_type] with [target_product] [target_type]"

        Examples:
        - "map concert api key with turbo entity" -> source: "concert ApiKey", dest: "turbo Entity"
        - "map instana applications with concert tokens" -> source: "instana Application", dest: "concert Token"

        Return ONLY a JSON object:
        {{"source_context": "[product] [specific type name]", "destination_context": "[product] [specific type name]", "source_product": "[product name]", "target_product": "[product name]", "source_types": ["type1", "type2"], "target_types": ["type1", "type2"]}}

        Your entire response must be ONLY the JSON object starting with {{ and ending with }}.
        """.strip()

        messages = [
            {"role": "system", "content": "You are an expert in API schema mapping. Extract specific schema type names, not general domains."},
            {"role": "user", "content": prompt},
        ]
        raw = self.call_llm(messages)
        raw = _strip_code_fences(raw)

        match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if match:
            ctx, err = _safe_json_loads(match.group())
        else:
            ctx, err = _safe_json_loads(raw)

        if err or not isinstance(ctx, dict):
            logger.warning("Context JSON parse failed; using parsed hints. err=%s raw=%s", err, raw[:200])
            state["source_context"] = " ".join(source_hints) if source_hints else user_utterance
            state["destination_context"] = " ".join(target_hints) if target_hints else user_utterance
            state["source_product"] = source_hints[0] if source_hints else ""
            state["target_product"] = target_hints[0] if target_hints else ""
            state["source_type_hints"] = source_hints[1:] if len(source_hints) > 1 else []
            state["target_type_hints"] = target_hints[1:] if len(target_hints) > 1 else []
        else:
            state["source_context"] = ctx.get("source_context", user_utterance)
            state["destination_context"] = ctx.get("destination_context", user_utterance)
            state["source_product"] = ctx.get("source_product", source_hints[0] if source_hints else "")
            state["target_product"] = ctx.get("target_product", target_hints[0] if target_hints else "")
            state["source_type_hints"] = ctx.get("source_types", source_hints[1:] if len(source_hints) > 1 else [])
            state["target_type_hints"] = ctx.get("target_types", target_hints[1:] if len(target_hints) > 1 else [])

        logger.info("Contexts | source=%r dest=%r src_product=%r tgt_product=%r",
                    state["source_context"], state["destination_context"],
                    state.get("source_product"), state.get("target_product"))
        return state

    def _parse_utterance_for_schemas(self, utterance: str) -> tuple:
        """Parse utterance to extract source and target schema hints.

        Handles patterns like:
        - "map concert api key with turbo entity"
        - "map instana applications with concert tokens"

        Returns (source_hints, target_hints) where each is [product, type1, type2, ...]
        """
        utterance_lower = utterance.lower().strip()

        # Known product names in ITOps dataset
        products = ['concert', 'turbo', 'instana']

        # Patterns: "map X ... with Y ..." or "map X ... to Y ..."
        patterns = [
            r"map\s+(\w+)\s+(.+?)\s+with\s+(\w+)\s+(.+?)$",
            r"map\s+(\w+)\s+(.+?)\s+to\s+(\w+)\s+(.+?)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, utterance_lower)
            if match:
                src_product, src_rest, tgt_product, tgt_rest = match.groups()

                # Extract type hints from the rest
                src_types = self._extract_type_hints(src_rest)
                tgt_types = self._extract_type_hints(tgt_rest)

                return ([src_product] + src_types, [tgt_product] + tgt_types)

        # Fallback: find product names and split
        found_products = []
        for prod in products:
            if prod in utterance_lower:
                found_products.append(prod)

        if len(found_products) >= 2:
            # Assume first is source, second is target
            src_idx = utterance_lower.find(found_products[0])
            tgt_idx = utterance_lower.find(found_products[1])

            if src_idx < tgt_idx:
                src_part = utterance_lower[src_idx:tgt_idx]
                tgt_part = utterance_lower[tgt_idx:]
            else:
                src_part = utterance_lower[tgt_idx:src_idx]
                tgt_part = utterance_lower[src_idx:]
                found_products = [found_products[1], found_products[0]]

            src_types = self._extract_type_hints(src_part.replace(found_products[0], ''))
            tgt_types = self._extract_type_hints(tgt_part.replace(found_products[1], ''))

            return ([found_products[0]] + src_types, [found_products[1]] + tgt_types)

        return ([], [])

    def _extract_type_hints(self, text: str) -> list:
        """Extract potential schema type names from text.

        Converts phrases like 'api key' -> 'ApiKey', 'entity actions' -> ['Entity', 'Action']
        """
        # Remove common words
        stop_words = {'map', 'to', 'from', 'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'as', 'filtered', 'stats'}

        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if w not in stop_words and len(w) > 1]

        # Convert to CamelCase type names
        type_hints = []
        i = 0
        while i < len(words):
            # Check for two-word type names (e.g., "api key" -> "ApiKey")
            if i + 1 < len(words):
                combined = words[i].capitalize() + words[i+1].capitalize()
                # Common two-word patterns
                if words[i] in ['api', 'call', 'market', 'entity', 'application']:
                    type_hints.append(combined)
                    i += 2
                    continue

            type_hints.append(words[i].capitalize())
            i += 1

        return type_hints

    def _build_property_graph_once(self) -> nx.Graph:
        """Build property graph once during init (O(n²) - cached for reuse)."""
        import time
        t0 = time.time()
        G = nx.Graph()
        props = self.property_data

        for k, v in props.items():
            G.add_node(k, **v)

        edge_count = 0
        keys = list(props.keys())
        for i, p1 in enumerate(keys):
            d1 = props[p1]
            for p2 in keys[i + 1:]:
                d2 = props[p2]
                weight = 0.0
                if d1.get("product_name") == d2.get("product_name"):
                    weight += 0.5
                if d1.get("type_name") == d2.get("type_name"):
                    weight += 0.3
                w1 = set(str(d1.get("field_name", "")).lower().split("."))
                w2 = set(str(d2.get("field_name", "")).lower().split("."))
                sim = len(w1 & w2) / max(len(w1 | w2), 1)
                weight += 0.2 * sim
                if weight > 0.1:
                    G.add_edge(p1, p2, weight=weight)
                    edge_count += 1

        logger.info("Built property graph in %.2fs | nodes=%d edges=%d",
                    time.time() - t0, G.number_of_nodes(), edge_count)
        return G

    def _create_property_graph(self, state: MappingState) -> MappingState:
        """Use pre-cached property graph instead of rebuilding."""
        state["graph"] = self._cached_property_graph
        logger.info("Using cached property graph | nodes=%d edges=%d",
                    self._cached_property_graph.number_of_nodes(),
                    self._cached_property_graph.number_of_edges())
        return state

    def _get_or_create_collection(self, state: MappingState):
        try:
            col = self.chroma_client.get_collection(name=self.collection_name)
            logger.info("Using existing Chroma collection: %s", self.collection_name)
            return col
        except Exception:
            return self._create_collection_from_state(state)

    def _select_source_properties(self, state: MappingState) -> MappingState:
        collection = state["chroma_collection"] or self._get_or_create_collection(state)
        state["chroma_collection"] = collection

        source_product = state.get("source_product", "").lower()
        type_hints = state.get("source_type_hints", [])

        # For strict schema filtering (OMOP), filter by product FIRST before semantic search
        if self.strict_schema_filter and source_product:
            # Get all properties matching the product
            matching_ids = self._get_properties_by_product(state["property_data"], source_product, type_hints)
            logger.info("Strict filter: found %d properties matching product=%s", len(matching_ids), source_product)

            if matching_ids:
                # Now do semantic search and intersect with type-filtered properties
                q = state["source_context"]
                q_emb = self.embedding_model.encode(q).tolist()
                # Search more results to increase chance of finding type-matching ones
                n_results = min(len(state["property_data"]), self.top_k * self.n_results_multiplier * 5)
                res = collection.query(query_embeddings=[q_emb], n_results=n_results)
                semantic_ids = res["ids"][0] if res.get("ids") else []

                # Intersect semantic results with product-filtered results
                filtered_ids = [pid for pid in semantic_ids if pid in matching_ids][:self.top_k]
                logger.info("Strict filter: %d semantic results intersected to %d", len(semantic_ids), len(filtered_ids))

                # If intersection is empty/too small, take type-filtered properties directly
                if len(filtered_ids) < self.top_k:
                    remaining = self.top_k - len(filtered_ids)
                    extras = [pid for pid in sorted(matching_ids) if pid not in filtered_ids][:remaining]
                    filtered_ids.extend(extras)
                    logger.info("Strict filter: added %d type-matched properties to reach %d", len(extras), len(filtered_ids))
            else:
                # Fallback if no product match
                logger.warning("Strict filter: no properties match product=%s, falling back to semantic", source_product)
                filtered_ids = []
        else:
            filtered_ids = []

        # Standard semantic search path (soft filtering or fallback)
        if not filtered_ids:
            q = state["source_context"]
            q_emb = self.embedding_model.encode(q).tolist()

            # Get more results initially, then filter by product/type hints
            n_results = min(self.top_k * self.n_results_multiplier, len(state["property_data"]))
            res = collection.query(query_embeddings=[q_emb], n_results=n_results)
            all_ids = res["ids"][0] if res.get("ids") else []

            filtered_ids = self._filter_properties_by_hints(
                all_ids, state["property_data"], source_product, type_hints
            )

            # If filtering returned nothing, fall back to top results
            if not filtered_ids:
                filtered_ids = set(all_ids[:self.top_k])
                logger.warning("No properties matched source hints, using top %d unfiltered", self.top_k)
            else:
                # Limit to top_k after filtering
                filtered_ids = set(list(filtered_ids)[:self.top_k])

        filtered_ids = set(filtered_ids)

        # 1-hop neighbors (only if neighbor_k > 0)
        nbrs = set()
        if self.neighbor_k > 0:
            for pid in filtered_ids:
                if pid in state["graph"]:
                    nbrs.update(state["graph"].neighbors(pid))
            # When strict filtering is enabled, filter neighbors by product AND type
            if self.strict_schema_filter and source_product:
                product_lower = source_product.lower()
                type_hints_lower = [h.lower() for h in (type_hints or [])]
                filtered_nbrs = set()
                for n in nbrs:
                    if not n.lower().startswith(product_lower + "_"):
                        continue
                    # Also filter by type hints if available
                    if type_hints_lower:
                        pdata = state["property_data"].get(n, {})
                        tname = pdata.get("type_name", "").lower()
                        if any(hint in tname for hint in type_hints_lower):
                            filtered_nbrs.add(n)
                    else:
                        filtered_nbrs.add(n)
                nbrs = filtered_nbrs
            nbrs = set(sorted(nbrs)[: self.neighbor_k])

        # Keep filtered_ids first (by semantic order), then append neighbors
        combined = list(filtered_ids) + [n for n in sorted(nbrs) if n not in filtered_ids]
        state["source_properties"] = combined
        logger.info("Selected source properties | filtered=%d +nbr=%d -> %d (product=%s, hints=%s, strict=%s)",
                    len(filtered_ids), len(nbrs), len(combined), source_product, type_hints, self.strict_schema_filter)
        return state

    def _get_properties_by_product(self, property_data: dict, product: str, type_hints: list = None) -> set:
        """Get all properties matching a product name (and optionally type hints).

        Used for strict schema filtering where we filter by product BEFORE semantic search.
        """
        matching = set()
        type_matched = set()
        product_lower = product.lower()
        type_hints_lower = [h.lower() for h in (type_hints or [])]

        for pid, pdata in property_data.items():
            pname = pdata.get("product_name", "").lower()
            tname = pdata.get("type_name", "").lower()

            # Product must match
            if product_lower not in pname and pname not in product_lower:
                continue

            matching.add(pid)

            # Track which ones match type hints
            if type_hints_lower:
                type_match = any(
                    hint in tname or tname in hint
                    for hint in type_hints_lower
                )
                if type_match:
                    type_matched.add(pid)

        # If type hints were provided and found matches, prefer those
        # Otherwise fall back to all product-matching properties
        if type_hints_lower and type_matched:
            logger.info("Type hint filter: %d of %d properties match type hints %s",
                       len(type_matched), len(matching), type_hints_lower)
            return type_matched

        return matching

    def _filter_properties_by_hints(self, property_ids: list, property_data: dict,
                                     product: str, type_hints: list) -> set:
        """Re-rank properties by product name and type hints.

        Uses a soft re-ranking approach: properties matching the product get boosted,
        but we don't strictly filter out non-matching properties (since the semantic
        relationship between utterances and schemas may not be literal).

        Args:
            property_ids: List of property IDs from ChromaDB (in semantic similarity order)
            property_data: Full property metadata dict
            product: Product name to filter by (e.g., 'concert')
            type_hints: List of type name hints (e.g., ['ApiKey', 'Application'])

        Returns:
            Set of re-ranked property IDs
        """
        # Get boost values from config
        product_boost = self.config.get("product_boost", 100)
        product_penalty = self.config.get("product_penalty", -50)
        type_exact_boost = self.config.get("type_exact_boost", 20)
        type_partial_boost = self.config.get("type_partial_boost", 10)
        type_field_boost = self.config.get("type_field_boost", 5)

        scored = []

        for i, pid in enumerate(property_ids):
            pdata = property_data.get(pid, {})
            pname = pdata.get("product_name", "").lower()
            tname = pdata.get("type_name", "").lower()
            field_name = pid.split('.')[0].lower() if '.' in pid else pid.lower()

            # Base score from semantic similarity (inverse of position)
            base_score = len(property_ids) - i

            boost = 0

            # Product match (highest priority) - MUST match for source/target selection
            if product:
                if product in pname:
                    boost += product_boost  # Strong boost for matching product
                else:
                    boost += product_penalty  # Penalty for wrong product

            # Type hint matching (softer - these are often inexact)
            for hint in type_hints:
                hint_lower = hint.lower()
                # Exact type name match
                if hint_lower == tname:
                    boost += type_exact_boost
                elif hint_lower in tname or tname in hint_lower:
                    boost += type_partial_boost
                # Type name in field path
                if hint_lower in field_name:
                    boost += type_field_boost

            total_score = base_score + boost
            scored.append((pid, total_score))

        # Sort by total score descending
        scored.sort(key=lambda x: -x[1])

        # Return top properties (those with positive scores, or top N if none positive)
        positive_scored = [(pid, s) for pid, s in scored if s > 0]
        if positive_scored:
            return set(pid for pid, _ in positive_scored)
        else:
            # Fallback to top half of results
            return set(pid for pid, _ in scored[:len(scored)//2])

    def _create_collection_from_state(self, state: MappingState):
        logger.info("Building Chroma collection: %s", self.collection_name)
        col = self.chroma_client.get_or_create_collection(name=self.collection_name, embedding_function=None)
        try:
            col.delete(where={})
        except Exception:
            pass

        ids, docs, metas, embs = [], [], [], []
        props = state["property_data"]

        for pid, pdata in tqdm(
            props.items(),
            desc=f"Embedding properties -> {self.collection_name}",
            disable=logger.level > logging.INFO,
        ):
            ids.append(pid)
            doc = f"{pdata.get('product_name','')} {pdata.get('type_name','')} {pdata.get('field_name','')} {pdata.get('description','')}"
            docs.append(doc)
            metas.append({
                "product_name": pdata.get("product_name", ""),
                "type_name": pdata.get("type_name", ""),
                "field_name": pdata.get("field_name", "")
            })
            embs.append(self.embedding_model.encode(doc))

        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        logger.info("Chroma collection populated | %d items", len(ids))
        return col

    def _select_destination_properties(self, state: MappingState) -> MappingState:
        collection = state["chroma_collection"]
        target_product = state.get("target_product", "").lower()
        type_hints = state.get("target_type_hints", [])

        # For strict schema filtering (OMOP), filter by product FIRST before semantic search
        if self.strict_schema_filter and target_product:
            # Get all properties matching the product
            matching_ids = self._get_properties_by_product(state["property_data"], target_product, type_hints)
            logger.info("Strict filter (dest): found %d properties matching product=%s", len(matching_ids), target_product)

            if matching_ids:
                # Now do semantic search and intersect with type-filtered properties
                q = state["destination_context"]
                q_emb = self.embedding_model.encode(q).tolist()
                # Search more results to increase chance of finding type-matching ones
                n_results = min(len(state["property_data"]), self.top_k * self.n_results_multiplier * 5)
                res = collection.query(query_embeddings=[q_emb], n_results=n_results)
                semantic_ids = res["ids"][0] if res.get("ids") else []

                # Intersect semantic results with product-filtered results
                filtered_ids = [pid for pid in semantic_ids if pid in matching_ids][:self.top_k]
                logger.info("Strict filter (dest): %d semantic results intersected to %d", len(semantic_ids), len(filtered_ids))

                # If intersection is empty/too small, take type-filtered properties directly
                if len(filtered_ids) < self.top_k:
                    remaining = self.top_k - len(filtered_ids)
                    extras = [pid for pid in sorted(matching_ids) if pid not in filtered_ids][:remaining]
                    filtered_ids.extend(extras)
                    logger.info("Strict filter (dest): added %d type-matched properties to reach %d", len(extras), len(filtered_ids))
            else:
                # Fallback if no product match
                logger.warning("Strict filter (dest): no properties match product=%s, falling back to semantic", target_product)
                filtered_ids = []
        else:
            filtered_ids = []

        # Standard semantic search path (soft filtering or fallback)
        if not filtered_ids:
            q = state["destination_context"]
            q_emb = self.embedding_model.encode(q).tolist()

            # Get more results initially, then filter by product/type hints
            n_results = min(self.top_k * self.n_results_multiplier, len(state["property_data"]))
            res = collection.query(query_embeddings=[q_emb], n_results=n_results)
            all_ids = res["ids"][0] if res.get("ids") else []

            filtered_ids = self._filter_properties_by_hints(
                all_ids, state["property_data"], target_product, type_hints
            )

            # If filtering returned nothing, fall back to top results
            if not filtered_ids:
                filtered_ids = set(all_ids[:self.top_k])
                logger.warning("No properties matched target hints, using top %d unfiltered", self.top_k)
            else:
                # Limit to top_k after filtering
                filtered_ids = set(list(filtered_ids)[:self.top_k])

        filtered_ids = set(filtered_ids)

        # 1-hop neighbors (only if neighbor_k > 0)
        nbrs = set()
        if self.neighbor_k > 0:
            for pid in filtered_ids:
                if pid in state["graph"]:
                    nbrs.update(state["graph"].neighbors(pid))
            # When strict filtering is enabled, filter neighbors by product AND type
            if self.strict_schema_filter and target_product:
                product_lower = target_product.lower()
                type_hints_lower = [h.lower() for h in (type_hints or [])]
                filtered_nbrs = set()
                for n in nbrs:
                    if not n.lower().startswith(product_lower + "_"):
                        continue
                    # Also filter by type hints if available
                    if type_hints_lower:
                        pdata = state["property_data"].get(n, {})
                        tname = pdata.get("type_name", "").lower()
                        if any(hint in tname for hint in type_hints_lower):
                            filtered_nbrs.add(n)
                    else:
                        filtered_nbrs.add(n)
                nbrs = filtered_nbrs
            nbrs = set(sorted(nbrs)[: self.neighbor_k])

        # Keep filtered_ids first (by semantic order), then append neighbors
        combined = list(filtered_ids) + [n for n in sorted(nbrs) if n not in filtered_ids]
        # Remove any that are already in source properties
        filtered = [x for x in combined if x not in state["source_properties"]]
        state["destination_properties"] = filtered
        logger.info("Selected destination properties | filtered=%d +nbr=%d -> %d (product=%s, hints=%s, strict=%s)",
                    len(filtered_ids), len(nbrs), len(filtered), target_product, type_hints, self.strict_schema_filter)
        return state

    def _find_mappings(self, state: MappingState) -> MappingState:
        srcs = state["source_properties"]
        dsts = state["destination_properties"]
        pdata = state["property_data"]

        if not srcs or not dsts:
            logger.warning("Insufficient properties for mapping | src=%d dst=%d", len(srcs), len(dsts))
            state["candidate_mappings"] = {}
            return state
    
        logger.info("Finding mappings | evaluating %d source × %d dest = %d pairs", 
                    len(srcs), len(dsts), len(srcs) * len(dsts))
   
        prompt = [
            "Evaluate these property mappings. Return ONLY a JSON array. No preamble, no explanation.",
            "Format: [{\"source_property\": \"id\", \"destination_property\": \"id\", \"valid\": true, \"confidence\": 0.9, \"justification\": \"reason\"}]",
            "Only include entries where valid=true.",
            "Your entire response must be ONLY the JSON array starting with [ and ending with ].",
            "\nCandidate Pairs:\n"
        ]

        for sp in srcs:
            si = pdata[sp]
            prompt.append(f"\nSource: {sp}\n - Product: {si['product_name']}\n - Type: {si['type_name']}\n - Field: {si['field_name']}\n - Description: {si['description']}")
            for dp in dsts:
                di = pdata[dp]
                prompt.append(
                    f"  Candidate Destination: {dp}\n"
                    f"   - Product: {di['product_name']}\n"
                    f"   - Type: {di['type_name']}\n"
                    f"   - Field: {di['field_name']}\n"
                    f"   - Description: {di['description']}"
                )

        messages = [
            {"role": "system", "content": "You are an expert in data mapping and schema alignment. Return only valid JSON with no explanatory text."},
            {"role": "user", "content": "\n".join(prompt)}
        ]

        logger.debug("Evaluating %d source x %d dest = %d pairs", len(srcs), len(dsts), len(srcs) * len(dsts))

        raw = self.call_llm(messages)

        # Debug logging (only at DEBUG level)
        logger.debug("LLM Response length: %d characters", len(raw) if raw else 0)
        if len(raw) < 1000:
            logger.debug("Full LLM response:\n%s", raw)
        else:
            logger.debug("LLM response preview (first 500):\n%s", raw[:500])

        raw = _strip_code_fences(raw)

        match = re.search(r'\[\s*\{.*?\}\s*\]', raw, re.DOTALL)
        if match:
            parsed, err = _safe_json_loads(match.group())
        else:
            parsed, err = _safe_json_loads(raw)

        if err or not isinstance(parsed, list):
            logger.warning("Mapping JSON parse failed; err=%s", err)
            logger.info("Response length: %d characters", len(raw))
            logger.info("Response preview (first 500 chars):\n%s", raw[:500])
            logger.info("Response end (last 200 chars):\n%s", raw[-200:] if len(raw) > 200 else raw)
            
            # Try to extract JSON from verbose response
            # Look for JSON array anywhere in the response
            json_match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', raw)
            if json_match:
                try:
                    extracted_json = json_match.group()
                    logger.info("Found JSON array in response, attempting to parse...")
                    parsed = json.loads(extracted_json)
                    logger.info("Successfully recovered JSON array from verbose response (%d items)", len(parsed))
                except Exception as e2:
                    logger.warning("Failed to parse extracted JSON: %s", e2)
                    logger.debug("Extracted text was: %s", extracted_json[:500])
                    parsed = []
            else:
                logger.warning("No JSON array pattern found in response")
                parsed = []
    
        cand = defaultdict(list)
        valid_count = 0
        invalid_count = 0
        
        for m in parsed:
            if isinstance(m, dict):
                if m.get("valid") is True:
                    valid_count += 1
                    src = m.get("source_property")
                    dst = m.get("destination_property")
                    if not src or not dst:
                        logger.warning("Skipping mapping with missing src or dst: %s", m)
                        continue
                    cand[src].append({
                        "mapProperty": dst,
                        "justification": m.get("justification", ""),
                        "confidence": float(m.get("confidence", 0.5))
                    })
                else:
                    invalid_count += 1
    
        state["candidate_mappings"] = dict(cand)
        total_mappings = sum(len(v) for v in cand.values())
        
        logger.info("Parsed response: %d valid, %d invalid mappings", valid_count, invalid_count)
        
        if cand:
            logger.info("Candidate mappings found | %d sources with %d total mappings", 
                        len(cand), total_mappings)
        else:
            logger.warning("No candidate mappings found")
        
        return state

    def _create_reflection_prompt(self, candidate_mappings: Dict, property_data: Dict) -> str:
        lines = ["Review the following property mappings and identify any that should be removed:"]
        for sp, maps in candidate_mappings.items():
            sdesc = property_data.get(sp, {}).get("description", "")
            lines.append(f"\nSource: {sp} ({sdesc[:100]}...)")
            for m in maps:
                dp = m["mapProperty"]
                ddesc = property_data.get(dp, {}).get("description", "")
                lines.append(f"  -> {dp} ({ddesc[:100]}...)\n     Justification: {m['justification']}\n     Confidence: {m['confidence']}")
        lines.append("\nConsider: type compatibility, purpose similarity, business value, inconsistencies.")
        return "\n".join(lines)

    def _reflect_and_prune(self, state: MappingState) -> MappingState:
        cand = state["candidate_mappings"]
        if not cand:
            state["final_mappings"] = {}
            state["reflection_feedback"] = "No candidate mappings were generated"
            logger.info("No candidates to reflect on")
            return state

        prompt = self._create_reflection_prompt(cand, state["property_data"])
        messages = [
            {"role": "system", "content": "You are an expert data mapper tasked with validating property mappings. Review each mapping critically and provide feedback."},
            {"role": "user", "content": prompt}
        ]
        feedback = self.call_llm(messages)

        prune_prompt = [
            f"Feedback: {feedback}",
            "\nBased on this feedback, prune the candidate mappings below.",
            "Return ONLY a JSON object. No text before or after.",
            'Format: {"source_id": [{"mapProperty": "dest_id", "justification": "reason"}]}',
            "\nYour entire response must be ONLY the JSON object starting with { and ending with }.",
            "\nCandidate mappings:"
        ]
        for sp, maps in cand.items():
            prune_prompt.append(f"\nSource: {sp}")
            for m in maps:
                prune_prompt.append(f"  -> {m['mapProperty']} (Confidence: {m['confidence']})\n     Justification: {m['justification']}")

        messages = [
            {"role": "system", "content": "You are an expert in schema mapping. Output only the final valid mappings in JSON."},
            {"role": "user", "content": "\n".join(prune_prompt)}
        ]

        raw = self.call_llm(messages)
        raw = _strip_code_fences(raw)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            final, err = _safe_json_loads(match.group())
        else:
            final, err = None, "No JSON found"

        if err or not isinstance(final, dict):
            logger.warning("Pruned JSON parse failed; falling back to confidence filter; err=%s raw=%s", err, raw[:200])
            final = {
                s: [{"mapProperty": m["mapProperty"], "justification": m.get("justification", "")}
                    for m in maps if float(m.get("confidence", 0.0)) >= 0.7]
                for s, maps in cand.items()
            }

        state["final_mappings"] = final
        state["reflection_feedback"] = str(feedback)
        logger.info("Final mappings after reflection | sources=%d", len(final))
        return state

    def _format_output(self, state: MappingState) -> MappingState:
        if not state.get("final_mappings"):
            state["final_mappings"] = state.get("candidate_mappings", {})
        return state

    # ---- public ----
    def run(self, user_utterance: str) -> Dict[str, Any]:
        import time
        logger.info("MapReflect run | utterance=%r model=%s", user_utterance, self.model)
        initial = MappingState(
            user_utterance=user_utterance,
            property_data=self.property_data,
            source_properties=[],
            destination_properties=[],
            candidate_mappings={},
            final_mappings={},
            reflection_feedback="",
            graph=nx.Graph(),
            chroma_collection=None,
            source_context="",
            destination_context="",
            source_product="",
            target_product="",
            source_type_hints=[],
            target_type_hints=[]
        )

        # Run with timing for each step
        state = initial
        steps = [
            ("extract_contexts", self._extract_contexts),
            ("create_graph", self._create_property_graph),
            ("select_source_properties", self._select_source_properties),
            ("select_destination_properties", self._select_destination_properties),
            ("find_mappings", self._find_mappings),
            ("format_output", self._format_output),
        ]

        total_start = time.time()
        for step_name, step_fn in steps:
            step_start = time.time()
            state = step_fn(state)
            step_time = time.time() - step_start
            logger.warning(f"TIMING | {step_name}: {step_time:.2f}s")
        total_time = time.time() - total_start
        logger.warning(f"TIMING | TOTAL: {total_time:.2f}s")

        out = {
            "mappings": state["final_mappings"],
            "source_properties": state["source_properties"],
            "destination_properties": state["destination_properties"],
            "reflection_feedback": state["reflection_feedback"]
        }
        logger.info("MapReflect done | final_sources=%d", len(out["mappings"]))
        return out

# Example usage
def main():
    logging.basicConfig(level=logging.WARNING)
    agent = MapReflectAgent(
        property_lookup_file="../data/itops_api_lookup.json",
        dataset="test",
        model="watsonx/mistralai/mistral-large"  # Example model specification
    )
    user_input = "map concert vulnerabilities to turbo entities"
    result = agent.run(user_input)
    print("\nFinal Mappings:")
    print(json.dumps(result["mappings"], indent=2))

if __name__ == "__main__":
    main()
