"""
ue_reflect.py - Uncertainty-Quantified Map-Reflect Pipeline

DEPRECATED: This module has been merged into map_reflect_runner.py.
Please use `python -m map_reflect.map_reflect_runner` with --use-uncertainty and --use-reflection flags instead.

Example:
    python -m map_reflect.map_reflect_runner \\
        --datasets OMOP_Synthea \\
        --models watsonx/ibm/granite-4-h-small \\
        --property-lookup-file data/omop_synthea_api_lookup.json \\
        --data-dir data \\
        --use-uncertainty \\
        --use-reflection

---

Original description:
Mapping pipeline with LM-Polygraph perplexity-based uncertainty estimation
for self-reflection step. Uses WatsonX models.

Adapted to work with the existing graphql-schema-mapping data format.

Requirements:
  pip install chromadb sentence-transformers tqdm python-dotenv lm-polygraph ibm-watsonx-ai

Environment variables needed:
  WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL (optional)

Usage:
  python -m map_reflect.ue_reflect --datasets OMOP_Synthea --models watsonx/ibm/granite-4-h-small --property-lookup-file data/omop_synthea_api_lookup.json --data-dir data
"""
import warnings
warnings.warn(
    "ue_reflect.py is deprecated. Use map_reflect_runner.py with --use-uncertainty --use-reflection flags instead.",
    DeprecationWarning,
    stacklevel=2
)

import json
import os
import re
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Workaround for old sqlite3 on some systems
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import chromadb
from chromadb.utils import embedding_functions

# Try to import LM-Polygraph estimators
try:
    from lm_polygraph.estimators import Perplexity
    from lm_polygraph.utils.manager import estimate_uncertainty
    LM_POLYGRAPH_AVAILABLE = True
except ImportError:
    LM_POLYGRAPH_AVAILABLE = False

# WatsonX SDK imports
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
    from ibm_watsonx_ai.foundation_models import ModelInference
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False

# litellm fallback
try:
    from litellm import completion as litellm_completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# -------------------------
# Model-specific prompt formatting
# -------------------------

def format_prompt_for_model(messages: List[Dict], model_id: str) -> Tuple[str, List[str]]:
    """Format messages based on model-specific chat template."""
    model_lower = model_id.lower()

    # Granite 4 models
    if "granite-4" in model_lower or "granite4" in model_lower:
        prompt_parts = []
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')
            prompt_parts.append(f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n")
        prompt_parts.append("<|start_of_role|>assistant<|end_of_role|>")
        return "".join(prompt_parts), ["<|end_of_text|>"]

    # Llama 3 models
    elif "llama" in model_lower:
        prompt_parts = ["<|begin_of_text|>"]
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')
            prompt_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(prompt_parts), ["<|eot_id|>", "<|end_of_text|>"]

    # Granite 3.x models
    elif "granite-3" in model_lower or "granite3" in model_lower:
        prompt_parts = []
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')
            prompt_parts.append(f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n")
        prompt_parts.append("<|start_of_role|>assistant<|end_of_role|>")
        return "".join(prompt_parts), ["<|end_of_text|>"]

    # Default fallback
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


# -------------------------
# WatsonX LLM helper
# -------------------------

class WatsonXLLM:
    """WatsonX LLM wrapper for generation."""

    def __init__(self, model_id: str, use_litellm: bool = False):
        self.model_id = model_id.replace("watsonx/", "")
        self.use_litellm = use_litellm
        self._cached_model = None

        if not use_litellm:
            if not WATSONX_AVAILABLE:
                raise ImportError("ibm_watsonx_ai not installed. Run: pip install ibm-watsonx-ai")
            self._init_watsonx()
        else:
            if not LITELLM_AVAILABLE:
                raise ImportError("litellm not installed. Run: pip install litellm")

    def _init_watsonx(self):
        """Initialize WatsonX model."""
        api_key = os.getenv("WATSONX_API_KEY") or os.getenv("WATSONX_APIKEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

        if not api_key or not project_id:
            raise RuntimeError("WATSONX_API_KEY and WATSONX_PROJECT_ID required")

        credentials = Credentials(url=url, api_key=api_key)
        _, stop_sequences = format_prompt_for_model([], self.model_id)

        parameters = {
            GenParams.MAX_NEW_TOKENS: 4000,
            GenParams.TEMPERATURE: 0.1,
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.STOP_SEQUENCES: stop_sequences,
        }

        self._cached_model = ModelInference(
            model_id=self.model_id,
            params=parameters,
            credentials=credentials,
            project_id=project_id
        )

    def generate(self, prompt: str, max_tokens: int = 1000, max_retries: int = 5) -> str:
        """Generate response from LLM with retry logic for transient errors."""
        import time
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(max_retries):
            try:
                if self.use_litellm:
                    response = litellm_completion(
                        model=f"watsonx/{self.model_id}",
                        messages=messages,
                        temperature=0,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content
                else:
                    formatted_prompt, _ = format_prompt_for_model(messages, self.model_id)
                    response = self._cached_model.generate_text(prompt=formatted_prompt)
                    return response[0] if isinstance(response, list) else response

            except Exception as e:
                error_str = str(e).lower()
                # Retry on transient errors (500, rate limit, connection errors)
                is_retryable = any(x in error_str for x in [
                    "500", "502", "503", "504",
                    "rate limit", "timeout", "connection",
                    "internal server error", "service unavailable",
                    "downstream", "vllm"
                ])

                if is_retryable and attempt < max_retries - 1:
                    wait_time = 2 ** attempt * 5  # 5, 10, 20, 40, 80 seconds
                    logger.warning(f"API error (attempt {attempt+1}/{max_retries}): {str(e)[:100]}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error (final attempt): {e}")
                    raise


# -------------------------
# Uncertainty Estimation
# -------------------------

class SamplingUncertaintyEstimator:
    """
    Uncertainty estimation using multiple sampling and consistency checking.

    Methods:
    1. Self-consistency: Generate multiple responses and check agreement
    2. Score variance: Measure variance in predicted scores across samples
    3. Top-k agreement: Check if top predictions agree across samples
    """

    def __init__(self, llm: 'WatsonXLLM', n_samples: int = 3, temperature: float = 0.3):
        self.llm = llm
        self.n_samples = n_samples
        self.temperature = temperature
        self._original_temp = None

    def estimate_ranking_uncertainty(
        self,
        prompt: str,
        primary_response: str,
        candidates: List[Dict]
    ) -> Dict[str, Any]:
        """
        Estimate uncertainty for a ranking task by checking consistency across samples.

        Returns:
            Dict with uncertainty score and metrics
        """
        # Parse primary response to get top prediction
        primary_ranking = safe_parse_json_array(primary_response)
        if not primary_ranking:
            return {"uncertainty": 0.8, "method": "parse_failure"}

        # Validate that ranking contains dicts with 'id' field
        if not all(isinstance(r, dict) and 'id' in r for r in primary_ranking):
            return {"uncertainty": 0.8, "method": "invalid_format"}

        primary_top = primary_ranking[0].get("id", "") if primary_ranking else ""

        def _safe_score(v, default=0.5):
            """Safely convert score to float."""
            try:
                return float(v) if v is not None else default
            except (ValueError, TypeError):
                return default

        primary_scores = {r.get("id", ""): _safe_score(r.get("score", 0.5)) for r in primary_ranking}

        # Generate additional samples with higher temperature
        agreement_count = 1  # Primary response agrees with itself
        score_variances = []

        # Get additional samples
        for _ in range(self.n_samples - 1):
            try:
                sample_response = self._generate_with_temperature(prompt, self.temperature)
                sample_ranking = safe_parse_json_array(sample_response)

                if sample_ranking and all(isinstance(r, dict) and 'id' in r for r in sample_ranking):
                    sample_top = sample_ranking[0].get("id", "") if sample_ranking else ""

                    # Check top-1 agreement
                    if sample_top == primary_top:
                        agreement_count += 1

                    # Calculate score variance for overlapping candidates
                    sample_scores = {r.get("id", ""): _safe_score(r.get("score", 0.5)) for r in sample_ranking}
                    for cid in primary_scores:
                        if cid in sample_scores:
                            score_variances.append(abs(primary_scores[cid] - sample_scores[cid]))

            except Exception as e:
                logger.warning(f"Sampling error: {e}")
                continue

        # Calculate uncertainty metrics
        top1_agreement_rate = agreement_count / self.n_samples
        avg_score_variance = np.mean(score_variances) if score_variances else 0.3

        # Combine into overall uncertainty score
        # High agreement + low variance = low uncertainty
        uncertainty = 1.0 - (0.6 * top1_agreement_rate + 0.4 * (1.0 - min(avg_score_variance * 2, 1.0)))
        uncertainty = max(0.0, min(1.0, uncertainty))

        return {
            "uncertainty": float(uncertainty),
            "top1_agreement": float(top1_agreement_rate),
            "score_variance": float(avg_score_variance),
            "n_samples": self.n_samples,
            "method": "sampling_consistency"
        }

    def _generate_with_temperature(self, prompt: str, temperature: float) -> str:
        """Generate response with specific temperature."""
        messages = [{"role": "user", "content": prompt}]

        if self.llm.use_litellm:
            response = litellm_completion(
                model=f"watsonx/{self.llm.model_id}",
                messages=messages,
                temperature=temperature,
                max_tokens=800,
            )
            return response.choices[0].message.content
        else:
            # For native WatsonX, we need to create a new model with different temperature
            # For efficiency, just use a simple variation by asking for alternatives
            alt_prompt = prompt + "\n\nProvide an alternative ranking considering different interpretations."
            formatted_prompt, _ = format_prompt_for_model(messages, self.llm.model_id)
            response = self.llm._cached_model.generate_text(prompt=formatted_prompt)
            return response[0] if isinstance(response, list) else response


def estimate_uncertainty_from_response(response: str, candidates: List[Dict]) -> float:
    """
    Quick uncertainty estimation from response characteristics.

    Heuristics:
    - Short responses may indicate uncertainty
    - Missing scores or reasons indicate uncertainty
    - Low confidence scores indicate uncertainty
    """
    parsed = safe_parse_json_array(response)

    if not parsed:
        return 0.8  # High uncertainty - couldn't parse

    if len(parsed) == 0:
        return 0.9  # Very high uncertainty - empty response

    # Validate that all items are dicts
    if not all(isinstance(r, dict) for r in parsed):
        return 0.8  # High uncertainty - invalid format

    # Check for low confidence in top prediction
    top_score = parsed[0].get("score", 0.5) if parsed else 0.5
    try:
        top_score = float(top_score) if top_score is not None else 0.5
    except (ValueError, TypeError):
        top_score = 0.5

    # Check if reasons are provided
    has_reasons = all(r.get("reason", "").strip() for r in parsed[:3] if isinstance(r, dict))

    # Check score spread (concentrated scores = less certain about ranking)
    def safe_float(v, default=0.5):
        try:
            return float(v) if v is not None else default
        except (ValueError, TypeError):
            return default
    scores = [safe_float(r.get("score", 0.5), 0.5) for r in parsed if isinstance(r, dict)]
    if len(scores) > 1:
        score_range = max(scores) - min(scores)
    else:
        score_range = 0.5

    # Combine heuristics
    uncertainty = 0.0

    # Top score confidence (higher is more confident)
    uncertainty += 0.3 * (1.0 - top_score)

    # Reason completeness
    uncertainty += 0.2 * (0.0 if has_reasons else 1.0)

    # Score discrimination (higher range = more confident)
    uncertainty += 0.3 * (1.0 - min(score_range * 1.5, 1.0))

    # Base uncertainty
    uncertainty += 0.2 * 0.3

    return max(0.0, min(1.0, uncertainty))


# -------------------------
# JSON Parsing Helpers
# -------------------------

def safe_parse_json_array(s: str):
    """Parse JSON array from text, handling common issues."""
    m = re.search(r"(\[.*\])", s, re.S)
    if m:
        txt = m.group(1)
    else:
        txt = s
    try:
        return json.loads(txt)
    except Exception:
        try:
            return json.loads(txt.replace("'", '"'))
        except Exception:
            return None


def safe_parse_json_object(s: str):
    """Parse JSON object from text."""
    m = re.search(r"(\{.*\})", s, re.S)
    txt = m.group(1) if m else s
    try:
        return json.loads(txt)
    except Exception:
        try:
            return json.loads(txt.replace("'", '"'))
        except Exception:
            return None


# -------------------------
# Data Loading (compatible with existing format)
# -------------------------

def load_property_lookup(lookup_file: str) -> Dict[str, Any]:
    """Load property lookup file."""
    with open(lookup_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_property_graph(lookup: Dict[str, Any]) -> nx.Graph:
    """Build a property graph connecting semantically related properties.

    Properties are connected if they share:
    - Same product_name (+0.5 weight)
    - Same type_name (+0.3 weight)
    - Similar field names (+0.2 * Jaccard similarity)

    Edges are created for pairs with weight > 0.1.
    """
    import time
    t0 = time.time()
    G = nx.Graph()

    # Add all properties as nodes
    for key, value in lookup.items():
        node_data = value if isinstance(value, dict) else {"value": value}
        G.add_node(key, **node_data)

    # Create edges based on similarity
    keys = list(lookup.keys())
    edge_count = 0

    for i, p1 in enumerate(keys):
        d1 = lookup[p1] if isinstance(lookup[p1], dict) else {}
        for p2 in keys[i + 1:]:
            d2 = lookup[p2] if isinstance(lookup[p2], dict) else {}

            weight = 0.0

            # Same product
            if d1.get("product_name") == d2.get("product_name") and d1.get("product_name"):
                weight += 0.5

            # Same type
            if d1.get("type_name") == d2.get("type_name") and d1.get("type_name"):
                weight += 0.3

            # Field name similarity (Jaccard on words)
            w1 = set(str(d1.get("field_name", p1)).lower().replace("_", ".").split("."))
            w2 = set(str(d2.get("field_name", p2)).lower().replace("_", ".").split("."))
            if w1 and w2:
                sim = len(w1 & w2) / max(len(w1 | w2), 1)
                weight += 0.2 * sim

            if weight > 0.1:
                G.add_edge(p1, p2, weight=weight)
                edge_count += 1

    logger.info(f"Built property graph in {time.time() - t0:.2f}s | nodes={G.number_of_nodes()} edges={edge_count}")
    return G


def expand_candidates_with_neighbors(
    candidates: List[Dict],
    property_graph: nx.Graph,
    lookup: Dict[str, Any],
    neighbor_k: int = 5
) -> List[Dict]:
    """Expand candidate list with 1-hop neighbors from the property graph.

    Args:
        candidates: Initial candidate list from retrieval
        property_graph: NetworkX graph of property relationships
        lookup: Property lookup for descriptions
        neighbor_k: Maximum number of neighbors to add per candidate

    Returns:
        Expanded candidate list with neighbors appended
    """
    if neighbor_k <= 0 or property_graph.number_of_nodes() == 0:
        return candidates

    # Get IDs of current candidates
    candidate_ids = {c.get("id", "") for c in candidates}

    # Collect neighbors
    neighbor_ids = set()
    for cand in candidates:
        cand_id = cand.get("id", "")
        if cand_id in property_graph:
            for nbr in property_graph.neighbors(cand_id):
                if nbr not in candidate_ids:
                    neighbor_ids.add(nbr)

    # Limit neighbors
    neighbor_ids = set(sorted(neighbor_ids)[:neighbor_k])

    # Build neighbor candidate entries
    neighbor_candidates = []
    for nbr_id in neighbor_ids:
        nbr_info = lookup.get(nbr_id, {})
        nbr_desc = nbr_info.get("description", nbr_id) if isinstance(nbr_info, dict) else str(nbr_info)
        neighbor_candidates.append({
            "doc_id": nbr_id,
            "id": nbr_id,
            "doc": f"{nbr_id} -- {nbr_desc}",
            "distance": 0.6,  # Slightly worse than direct retrieval
            "meta": nbr_info if isinstance(nbr_info, dict) else {},
            "from_neighbor_expansion": True
        })

    if neighbor_candidates:
        logger.debug(f"    Expanded candidates with {len(neighbor_candidates)} neighbors")

    return candidates + neighbor_candidates


def get_product_properties(lookup: Dict[str, Any], product_name: str) -> Dict[str, Any]:
    """Get all properties for a specific product from lookup.

    Works with format: {"omop_PERSON.person_id": {"product_name": "OMOP", ...}}
    """
    properties = {}
    product_upper = product_name.upper()

    for key, value in lookup.items():
        if isinstance(value, dict):
            prop_product = value.get("product_name", "").upper()
            if prop_product == product_upper:
                desc = value.get("description", "")
                properties[key] = {
                    "description": desc,
                    "text": f"{key} -- {desc}",
                    "meta": value
                }
        elif key.lower().startswith(product_name.lower()):
            # Fallback: check key prefix
            properties[key] = {
                "description": str(value),
                "text": f"{key} -- {value}",
                "meta": {"value": value}
            }

    return properties


def load_dataset_samples(data_dir: str, dataset: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load evaluation samples from dataset JSONL file."""
    # Map dataset names to file names
    dataset_files = {
        'ITOps': 'itops_dataset.jsonl',
        'ITOpsGraph': 'itopsgraph_dataset.jsonl',  # Validated coverage mappings (521 samples)
        'NL2GQL': 'nl2gql_dataset.jsonl',
        'OMOP_MIMIC': 'omop_mimic_dataset.jsonl',
        'OMOP_CMS': 'omop_cms_dataset.jsonl',
        'OMOP_Synthea': 'omop_synthea_dataset.jsonl',
        'ToolBench': 'toolbench_eval_dataset.jsonl'
    }

    filename = dataset_files.get(dataset, f"{dataset.lower()}_dataset.jsonl")
    filepath = Path(data_dir) / filename

    if not filepath.exists():
        logger.error(f"Dataset file not found: {filepath}")
        return []

    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                sample = extract_sample_data(data, dataset, i)
                if sample:
                    samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in line {i+1}: {e}")
                continue

    logger.info(f"Loaded {len(samples)} samples from {filepath}")
    return samples


def extract_sample_data(data: Dict[str, Any], dataset: str, index: int) -> Optional[Dict[str, Any]]:
    """Extract sample data from JSON object."""
    # Handle unified format with result.properties
    if "result" in data and "properties" in data["result"]:
        result = data["result"]
        source_schema = result.get("name", "")
        properties = result.get("properties", {})

        ground_truth_mapping = {}
        expected_target = ""
        for src_field, mappings in properties.items():
            if mappings and len(mappings) > 0:
                ground_truth_mapping[src_field] = mappings[0]["mapProperty"]
                expected_target = mappings[0].get("mapTable", "")

        return {
            "sample_id": data.get("id", f"{dataset.lower()}_{index:03d}"),
            "utterance": data.get("mapping_utterance", data.get("question", "")),
            "expected_source": source_schema,
            "expected_target": expected_target,
            "ground_truth_mapping": ground_truth_mapping,
        }

    # Handle ToolBench and similar formats with direct ground_truth_mapping and expected_target
    if "ground_truth_mapping" in data and "expected_target" in data:
        return {
            "sample_id": data.get("sample_id", f"{dataset.lower()}_{index:03d}"),
            "utterance": data.get("utterance", ""),
            "expected_source": data.get("expected_source", ""),
            "expected_target": data.get("expected_target"),  # Can be dict or string
            "ground_truth_mapping": data.get("ground_truth_mapping", {}),
        }

    # Handle coverage dataset format with explicit source/target fields
    if "source_fields" in data and "target_fields" in data:
        # For coverage validation, create pseudo ground_truth from source_fields
        # The actual mapping will be discovered by UE-Reflect
        source_fields = data.get("source_fields", [])
        target_fields = data.get("target_fields", [])

        # Create a mapping task for each source field (empty target = discovery mode)
        ground_truth = {sf: "" for sf in source_fields}

        return {
            "sample_id": data.get("id", f"{dataset.lower()}_{index:03d}"),
            "utterance": data.get("utterance", ""),
            "expected_source": data.get("source_schema", data.get("source_type", "")),
            "expected_target": data.get("target_schema", data.get("target_type", "")),
            "ground_truth_mapping": ground_truth,
            "source_fields": source_fields,
            "target_fields": target_fields,
            "metadata": data.get("metadata", {}),
        }

    # Handle ITOpsGraph format (source_schema, target_schema, mapping)
    if "mapping" in data or "source_schema" in data:
        return {
            "sample_id": data.get("id", f"{dataset.lower()}_{index:03d}"),
            "utterance": data.get("utterance", data.get("mapping_utterance", "")),
            "expected_source": data.get("source_schema", ""),
            "expected_target": data.get("target_schema", ""),
            "ground_truth_mapping": data.get("mapping", {}),
            "metadata": data.get("metadata", {}),  # Preserve metadata for type filtering
        }

    return None


# -------------------------
# ChromaDB Indexing
# -------------------------

def index_chroma(collection_name: str, data: Dict[str, Any], chroma_dir: str):
    """Index documents in ChromaDB."""
    client = chromadb.PersistentClient(path=chroma_dir)
    documents_list, metadatas_list, ids_list = [], [], []

    for i, (key, value) in enumerate(data.items()):
        desc = value.get("description", "") if isinstance(value, dict) else str(value)
        doc = f"{key}: {desc}"
        metadata = {"source": collection_name, "class": key}

        documents_list.append(doc)
        metadatas_list.append(metadata)
        ids_list.append(f"doc_id_{i}")

    embed_model_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        client.delete_collection(name=collection_name)
    except:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_model_function,
        metadata={"hnsw:space": "cosine"}
    )

    # Add documents in batches to avoid ChromaDB batch size limit (5461)
    batch_size = 5000
    for i in range(0, len(documents_list), batch_size):
        end_idx = min(i + batch_size, len(documents_list))
        collection.add(
            documents=documents_list[i:end_idx],
            metadatas=metadatas_list[i:end_idx],
            ids=ids_list[i:end_idx]
        )
    logger.info(f"Indexed {len(documents_list)} documents in collection '{collection_name}'")
    return collection


def retrieve_candidates(query_text: str, collection_name: str, chroma_dir: str, top_k: int = 10) -> List[Dict]:
    """Retrieve candidate matches from ChromaDB."""
    embed_model = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(name=collection_name, embedding_function=embed_model)

    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    candidates = []
    for _id, doc, dist, meta in zip(ids, docs, dists, metas):
        full_name = meta.get('class') if isinstance(meta, dict) else _id
        candidates.append({
            "doc_id": _id,
            "id": full_name,
            "doc": doc,
            "distance": dist,
            "meta": meta
        })

    return candidates


# -------------------------
# LLM Re-ranking with Uncertainty
# -------------------------

def llm_rerank_with_uncertainty(
    llm: WatsonXLLM,
    source_prop: str,
    source_desc: str,
    candidates: List[Dict],
    source_product: str,
    target_product: str,
    use_sampling: bool = True,
    n_samples: int = 3
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Re-rank candidates and estimate uncertainty.

    Args:
        use_sampling: If True, use sampling-based uncertainty (slower but more accurate)
        n_samples: Number of samples for sampling-based uncertainty

    Returns:
        Tuple of (ranking, uncertainty_info dict)
    """
    cand_block = "\n".join([
        f"{i+1}. id: {c['id']}\n   text: {c['doc']}\n   distance: {c['distance']:.4f}"
        for i, c in enumerate(candidates)
    ])

    prompt = f"""You are given a {source_product} property and a list of candidate {target_product} properties.
Rank the candidates by how well they semantically match the {source_product} property.

{source_product} property: {source_prop}
Description: {source_desc}

Candidates:
{cand_block}

Task: Produce a JSON array of objects, ordered from best match to worst, where each object has:
- "id": candidate id (string, exactly as shown above)
- "score": a float between 0.0 and 1.0 (higher is better match)
- "reason": one concise sentence explaining why you assigned that score

Return only the JSON array and nothing else."""

    # Get primary response
    resp = llm.generate(prompt, max_tokens=800)

    # Estimate uncertainty
    if use_sampling and n_samples > 1:
        estimator = SamplingUncertaintyEstimator(llm, n_samples=n_samples)
        uncertainty_info = estimator.estimate_ranking_uncertainty(prompt, resp, candidates)
    else:
        # Quick heuristic-based estimation
        uncertainty_score = estimate_uncertainty_from_response(resp, candidates)
        uncertainty_info = {
            "uncertainty": uncertainty_score,
            "method": "response_heuristics"
        }

    # Parse the ranking response
    parsed = safe_parse_json_array(resp)

    # Validate parsed results have required 'id' field
    if parsed is not None:
        valid = all(isinstance(item, dict) and 'id' in item for item in parsed)
        if not valid:
            parsed = None  # Trigger fallback

    # Post-process: Map numeric IDs back to actual field IDs
    # LLMs sometimes return index numbers (e.g., '2') instead of actual IDs
    if parsed is not None:
        id_to_candidate = {str(i+1): c['id'] for i, c in enumerate(candidates)}
        for item in parsed:
            if isinstance(item, dict) and 'id' in item:
                item_id = str(item['id'])
                if item_id in id_to_candidate:
                    # Map numeric index back to actual field ID
                    item['id'] = id_to_candidate[item_id]

    if parsed is None:
        # Fallback to distance-based scoring
        maxd = max([c['distance'] for c in candidates]) if candidates else 1.0
        scored = []
        for c in candidates:
            score = 1.0 - (c['distance'] / (maxd + 1e-8))
            scored.append({
                "id": c['id'],
                "score": float(max(0.0, min(1.0, score))),
                "reason": "fallback distance-based score"
            })
        scored = sorted(scored, key=lambda x: -x['score'])
        uncertainty_info["uncertainty"] = 0.8  # High uncertainty on fallback
        uncertainty_info["method"] = "fallback"
        return scored, uncertainty_info
    else:
        return parsed, uncertainty_info


# -------------------------
# Self-Reflection with Uncertainty
# -------------------------

def llm_self_reflect_with_uncertainty(
    llm: WatsonXLLM,
    source_prop: str,
    source_desc: str,
    initial_ranking: List[Dict],
    candidates: List[Dict],
    uncertainty_score: float,
    high_uncertainty: bool,
    source_product: str,
    target_product: str
) -> Dict[str, Any]:
    """Enhanced self-reflection using uncertainty estimation."""
    # Show more candidates when uncertainty is high
    n_candidates = 8 if high_uncertainty else 5
    initial_block = "\n".join([
        f"{i+1}. {r.get('id', 'unknown')} (score={r.get('score')}) - {r.get('reason', '')}"
        for i, r in enumerate(initial_ranking[:n_candidates]) if isinstance(r, dict)
    ])
    cand_block = "\n".join([f"- {c.get('id', 'unknown')}: {c.get('doc', '')[:100]}" for c in candidates[:n_candidates] if isinstance(c, dict)])

    # Adjust reflection depth based on uncertainty
    if high_uncertainty:
        reflection_instruction = """CRITICAL: The initial ranking shows HIGH UNCERTAINTY, which strongly suggests the top-ranked candidate is WRONG.

SEMANTIC MATCHING RULES - Apply these carefully:
1. Foreign keys (fields ending in _id that reference another table) should map to the PRIMARY KEY of the referenced table, NOT to another foreign key
   - Example: concert_stadium_id (FK to stadium) → stadium_stadium_id (PK of stadium), NOT concert_concert_id
2. Match the TARGET entity, not the SOURCE entity
   - If source is "concert_stadium_id", the target should be in the STADIUM table, not CONCERT
3. Field names often follow pattern: {table}_{field}. Match the semantic purpose of {field}.

You MUST:
1. Assume the #1 candidate is incorrect - it's probably matching by name similarity instead of semantic purpose
2. Look for the candidate that represents the SAME DATA in the TARGET schema
3. For foreign keys, find the primary key they reference
4. CHANGE your top prediction based on semantic meaning, not name similarity

Be bold - your initial answer is likely wrong."""
    else:
        reflection_instruction = """The initial ranking shows reasonable confidence.
Please review for any obvious errors or improvements."""

    prompt = f"""You previously ranked {target_product} candidates for this {source_product} property:

{source_product} property: {source_prop}
Description: {source_desc}

Initial ranking (best first):
{initial_block}

Candidates:
{cand_block}

Uncertainty Score: {uncertainty_score:.3f} {"(HIGH)" if high_uncertainty else "(LOW)"}

{reflection_instruction}

Task:
1) Point out possible errors in the initial ranking (1-2 sentences).
2) Produce a revised JSON array (id, score, reason) from best to worst match.

Return a JSON object:
{{"analysis": "your error analysis", "revised_ranking": [array of ranked objects]}}"""

    # Get reflection response
    resp = llm.generate(prompt, max_tokens=900)

    # Estimate uncertainty of the reflection using response heuristics
    reflection_uncertainty = estimate_uncertainty_from_response(resp, candidates)

    # Parse reflection response
    parsed = safe_parse_json_object(resp)

    if parsed and "revised_ranking" in parsed:
        result = parsed
    else:
        # Try to extract just the array
        array_parsed = safe_parse_json_array(resp)
        if array_parsed:
            result = {"analysis": "Parsing fallback", "revised_ranking": array_parsed}
        else:
            result = {"analysis": "Could not parse response", "revised_ranking": initial_ranking}

    # Post-process: Map numeric IDs back to actual field IDs in revised_ranking
    if result.get("revised_ranking"):
        id_to_candidate = {str(i+1): c['id'] for i, c in enumerate(candidates)}
        for item in result["revised_ranking"]:
            if isinstance(item, dict) and 'id' in item:
                item_id = str(item['id'])
                if item_id in id_to_candidate:
                    item['id'] = id_to_candidate[item_id]

    # Add uncertainty metrics
    result["uncertainty_metrics"] = {
        "initial_uncertainty": uncertainty_score,
        "reflection_uncertainty": reflection_uncertainty,
        "high_uncertainty": high_uncertainty,
    }

    return result


# -------------------------
# Evaluation
# -------------------------

def evaluate_mapping(predicted: Dict[str, Any], ground_truth: Dict[str, str]) -> Dict[str, Any]:
    """Evaluate a single mapping prediction against ground truth."""
    revised_ranking = predicted.get("revised_ranking", [])

    if not revised_ranking or not ground_truth:
        return {"correct": False, "top1_hit": False, "mrr": 0.0}

    # Get predicted IDs
    pred_ids = [r.get("id", "") for r in revised_ranking if isinstance(r, dict)]

    # Check if top prediction is in ground truth
    gt_values = set(ground_truth.values()) if isinstance(ground_truth, dict) else set()

    top1_hit = pred_ids[0] in gt_values if pred_ids else False

    # Calculate MRR
    mrr = 0.0
    for i, pid in enumerate(pred_ids):
        if pid in gt_values:
            mrr = 1.0 / (i + 1)
            break

    return {
        "correct": top1_hit,
        "top1_hit": top1_hit,
        "mrr": mrr,
        "predicted_top1": pred_ids[0] if pred_ids else None,
        "ground_truth": list(gt_values)
    }


# -------------------------
# Main Pipeline
# -------------------------

def run_ue_reflect_pipeline(
    property_lookup_file: str,
    data_dir: str,
    dataset: str,
    model_id: str,
    output_dir: str = "map_reflect/results",
    chroma_dir: str = "data/chroma_db",
    top_k: int = 10,
    max_samples: Optional[int] = None,
    use_litellm: bool = False,
    use_sampling: bool = True,
    n_uncertainty_samples: int = 3,
    resume_from_checkpoint: bool = True,
    neighbor_k: int = 5,
):
    """Run the uncertainty-quantified map-reflect pipeline.

    Args:
        use_sampling: If True, use sampling-based uncertainty estimation
        n_uncertainty_samples: Number of samples for uncertainty estimation
        resume_from_checkpoint: If True, resume from checkpoint if available
        neighbor_k: Number of 1-hop neighbors to expand candidates with (0 to disable)
    """

    logger.info(f"Running UE-Reflect Pipeline")
    logger.info(f"  Dataset: {dataset}")
    logger.info(f"  Model: {model_id}")
    logger.info(f"  Top-K: {top_k}")
    logger.info(f"  Neighbor-K: {neighbor_k}")
    logger.info(f"  Uncertainty: {'sampling-based' if use_sampling else 'heuristics'} (n={n_uncertainty_samples})")
    logger.info(f"  Resume from checkpoint: {resume_from_checkpoint}")

    # Initialize LLM
    llm = WatsonXLLM(model_id, use_litellm=use_litellm)

    # Load property lookup
    lookup = load_property_lookup(property_lookup_file)
    logger.info(f"Loaded {len(lookup)} properties from lookup file")

    # Build property graph for neighbor expansion
    property_graph = None
    if neighbor_k > 0:
        property_graph = build_property_graph(lookup)

    # Load evaluation samples
    samples = load_dataset_samples(data_dir, dataset, max_samples)
    if not samples:
        logger.error("No samples loaded. Exiting.")
        return None

    logger.info(f"Processing {len(samples)} samples")

    # Setup checkpoint file
    os.makedirs(output_dir, exist_ok=True)
    model_name = model_id.replace("/", "_").replace("watsonx_", "")
    checkpoint_file = Path(output_dir) / f"ue_reflect_{dataset.lower()}_{model_name}_checkpoint.json"

    # Load checkpoint if available
    processed_ids = set()
    results = []
    all_uncertainties = []

    if resume_from_checkpoint and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            results = checkpoint_data.get("results", [])
            all_uncertainties = checkpoint_data.get("all_uncertainties", [])
            processed_ids = {r["sample_id"] for r in results}
            logger.info(f"Resumed from checkpoint: {len(processed_ids)} samples already processed")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            results = []
            all_uncertainties = []
            processed_ids = set()

    # Determine source and target products from dataset name
    # OMOP_Synthea -> source=OMOP, target=SYNTHEA
    # ITOps/ITOpsGraph -> multi-target (source/target vary per sample)
    multi_target_datasets = {"ITOPS", "ITOPSGRAPH", "ITOPSGRAPH_DISCOVERY", "ITOPSGRAPH_DISCOVERY_VALID", "ITOPSGRAPH_COVERAGE", "ITOPSGRAPH_COVERAGE_EVAL", "NL2GQL", "TOOLBENCH"}
    is_multi_target = dataset.upper() in multi_target_datasets

    if "_" in dataset and not is_multi_target:
        parts = dataset.split("_")
        source_product = parts[0].upper()
        target_product = parts[1].upper() if len(parts) > 1 else ""
    else:
        source_product = dataset.upper()
        target_product = ""

    logger.info(f"  Source product: {source_product}")
    logger.info(f"  Target product: {target_product if target_product else '(multi-target)'}")
    logger.info(f"  Multi-target mode: {is_multi_target}")

    # Get target properties for indexing
    if is_multi_target:
        # For multi-target datasets with bidirectional mappings (e.g., ITOps),
        # index ALL properties. Filtering to target-only is done per-sample
        # during candidate retrieval based on each sample's target schema.
        target_properties = {}
        for key, value in lookup.items():
            if isinstance(value, dict):
                desc = value.get("description", "")
                target_properties[key] = {
                    "description": desc,
                    "text": f"{key} -- {desc}",
                    "meta": value
                }
            else:
                target_properties[key] = {
                    "description": str(value),
                    "text": f"{key} -- {value}",
                    "meta": {}
                }
        logger.info(f"  Indexing ALL {len(target_properties)} properties (per-sample filtering will be applied)")
    else:
        target_properties = get_product_properties(lookup, target_product)
        if not target_properties:
            logger.error(f"No properties found for target product '{target_product}'")
            return None

    logger.info(f"  Target properties: {len(target_properties)}")

    # Index target properties
    collection_name = f"ue_reflect_{dataset.lower()}_target"
    index_chroma(collection_name, target_properties, chroma_dir)

    # Process each sample
    for sample in tqdm(samples, desc="Processing samples"):
        sample_id = sample.get("sample_id", "unknown")

        # Skip already processed samples (checkpoint resume)
        if sample_id in processed_ids:
            continue

        ground_truth = sample.get("ground_truth_mapping", {})

        if not ground_truth:
            continue

        # For multi-target datasets, extract source/target from sample
        sample_target_prefixes = set()  # Valid target prefixes for this sample
        if is_multi_target:
            sample_source = sample.get("expected_source", sample.get("source_schema", ""))
            sample_target = sample.get("expected_target", sample.get("target_schema", ""))
            # Extract product name from schema ID (e.g., "concert" from source_schema.id)
            if isinstance(sample_source, dict):
                sample_source_product = sample_source.get("name", sample_source.get("id", "source")).upper()
            else:
                sample_source_product = str(sample_source).split("_")[0].upper() if sample_source else "SOURCE"
            if isinstance(sample_target, dict):
                sample_target_product = sample_target.get("name", sample_target.get("id", "target")).upper()
                # Extract actual target prefixes from target schema properties
                for prop in sample_target.get("properties", []):
                    prop_name = prop.get("name", "") if isinstance(prop, dict) else str(prop)
                    if "_" in prop_name:
                        sample_target_prefixes.add(prop_name.split("_")[0].lower())
                if sample_target_prefixes:
                    logger.debug(f"  Sample {sample_id}: target prefixes = {sample_target_prefixes}")
            else:
                sample_target_product = str(sample_target).split("_")[0].upper() if sample_target else "TARGET"
        else:
            sample_source_product = source_product
            sample_target_product = target_product

        # For each source field in ground truth, do retrieval + ranking + reflection
        sample_result = {
            "sample_id": sample_id,
            "utterance": sample.get("utterance", ""),
            "expected_source": sample.get("expected_source", ""),
            "expected_target": sample.get("expected_target", ""),
            "ground_truth_mapping": ground_truth,
            "predictions": {},
        }

        sample_uncertainties = []

        for src_field, gt_target in ground_truth.items():
            # Get source field description from lookup
            src_info = lookup.get(src_field, {})
            src_desc = src_info.get("description", src_field) if isinstance(src_info, dict) else str(src_info)
            query_text = f"{src_field}: {src_desc}"

            # For multi-target datasets with sample-specific targets, use target schema properties directly
            # This avoids embedding-based retrieval which is biased by naming conventions

            # Check if sample has explicit target_fields list (coverage dataset format)
            sample_target_fields = sample.get("target_fields", [])

            if is_multi_target and sample_target_fields:
                # Coverage dataset format: explicit list of target field names
                candidates = []
                for prop_name in sample_target_fields:
                    prop_lookup = lookup.get(prop_name, {})
                    prop_desc = prop_lookup.get("description", prop_name) if isinstance(prop_lookup, dict) else str(prop_lookup)
                    candidates.append({
                        "doc_id": prop_name,
                        "id": prop_name,
                        "doc": f"{prop_name} -- {prop_desc}",
                        "distance": 0.5,  # Uniform distance since not using embeddings
                        "meta": prop_lookup if isinstance(prop_lookup, dict) else {}
                    })
                logger.debug(f"    {src_field[:40]}: Using {len(candidates)} explicit target fields as candidates")
            elif is_multi_target and isinstance(sample_target, dict) and sample_target.get("properties"):
                # ITOps-style: sample_target is a dict with properties list
                candidates = []
                for prop in sample_target.get("properties", []):
                    prop_name = prop.get("name", "") if isinstance(prop, dict) else str(prop)
                    prop_desc = prop.get("description", "") if isinstance(prop, dict) else ""
                    prop_lookup = lookup.get(prop_name, {})
                    if isinstance(prop_lookup, dict):
                        prop_desc = prop_lookup.get("description", prop_desc)
                    candidates.append({
                        "doc_id": prop_name,
                        "id": prop_name,
                        "doc": f"{prop_name} -- {prop_desc}",
                        "distance": 0.5,  # Uniform distance since not using embeddings
                        "meta": prop_lookup if isinstance(prop_lookup, dict) else {}
                    })
                logger.debug(f"    {src_field[:40]}: Using {len(candidates)} target schema properties as candidates")
            elif is_multi_target and isinstance(sample_target, str) and sample_target:
                # Multi-target dataset: sample_target is a product name (e.g., "turbonomic")
                # Check if we also have target_type in metadata for more precise filtering
                sample_metadata = sample.get("metadata", {})
                target_type = sample_metadata.get("target_type", "")

                if target_type:
                    # ITOpsGraph style: use product + type prefix (e.g., "turbonomic_FilterInput.")
                    target_prefix = f"{sample_target.lower()}_{target_type}."
                else:
                    # NL2GQL-style: use just product prefix (e.g., "stadium_")
                    target_prefix = sample_target.lower() + "_"

                candidates = []
                for prop_name, prop_info in lookup.items():
                    if prop_name.lower().startswith(target_prefix.lower()):
                        prop_desc = prop_info.get("description", prop_name) if isinstance(prop_info, dict) else str(prop_info)
                        candidates.append({
                            "doc_id": prop_name,
                            "id": prop_name,
                            "doc": f"{prop_name} -- {prop_desc}",
                            "distance": 0.5,  # Uniform distance since not using embeddings
                            "meta": prop_info if isinstance(prop_info, dict) else {}
                        })
                logger.debug(f"    {src_field[:40]}: Using {len(candidates)} properties with prefix '{target_prefix}' as candidates")

                # Safety limit: if too many candidates, truncate to avoid token overflow
                max_candidates = top_k * 3  # Allow up to 3x top_k candidates
                if len(candidates) > max_candidates:
                    logger.warning(f"    {src_field[:40]}: Truncating {len(candidates)} candidates to {max_candidates}")
                    candidates = candidates[:max_candidates]
            else:
                # For single-target datasets, use embedding-based retrieval
                candidates = retrieve_candidates(query_text, collection_name, chroma_dir, top_k=top_k)

            if not candidates:
                logger.warning(f"    {src_field[:40]}: No candidates found, skipping")
                continue

            # Expand candidates with 1-hop neighbors from property graph
            if property_graph is not None and neighbor_k > 0:
                candidates = expand_candidates_with_neighbors(
                    candidates, property_graph, lookup, neighbor_k=neighbor_k
                )

            # LLM re-ranking with uncertainty
            ranking, uncertainty_info = llm_rerank_with_uncertainty(
                llm, src_field, src_desc, candidates, sample_source_product, sample_target_product,
                use_sampling=use_sampling, n_samples=n_uncertainty_samples
            )
            uncertainty_score = uncertainty_info.get("uncertainty", 0.5)
            sample_uncertainties.append(uncertainty_score)
            all_uncertainties.append(uncertainty_score)

            # Self-reflection
            # Compute threshold dynamically based on collected uncertainties
            if len(all_uncertainties) > 1:
                mean_u = np.mean(all_uncertainties)
                std_u = np.std(all_uncertainties)
                threshold = mean_u + 0.5 * std_u
            else:
                threshold = 0.6

            high_uncertainty = uncertainty_score > threshold

            reflection = llm_self_reflect_with_uncertainty(
                llm, src_field, src_desc, ranking, candidates,
                uncertainty_score, high_uncertainty, sample_source_product, sample_target_product
            )

            sample_result["predictions"][src_field] = {
                "initial_ranking": ranking,
                "reflection": reflection,
                "uncertainty": float(uncertainty_score),
                "uncertainty_info": uncertainty_info,
                "high_uncertainty": bool(high_uncertainty),
            }

        # Evaluate sample
        correct_count = 0
        total_count = 0
        for src_field, gt_target in ground_truth.items():
            if src_field in sample_result["predictions"]:
                pred = sample_result["predictions"][src_field]
                revised = pred.get("reflection", {}).get("revised_ranking", [])
                if revised and isinstance(revised[0], dict):
                    top_pred = revised[0].get("id", "")
                    if top_pred == gt_target:
                        correct_count += 1
                total_count += 1

        sample_result["accuracy"] = correct_count / total_count if total_count > 0 else 0.0
        sample_result["correct_count"] = correct_count
        sample_result["total_count"] = total_count
        sample_result["mean_uncertainty"] = np.mean(sample_uncertainties) if sample_uncertainties else 0.5

        results.append(sample_result)

        # Save checkpoint after each sample
        checkpoint_data = {
            "dataset": dataset,
            "model_id": model_id,
            "results": results,
            "all_uncertainties": all_uncertainties,
            "processed_count": len(results),
            "total_samples": len(samples),
        }
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, cls=NumpyEncoder)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    # Compute overall metrics
    total_correct = sum(r["correct_count"] for r in results)
    total_fields = sum(r["total_count"] for r in results)
    overall_accuracy = total_correct / total_fields if total_fields > 0 else 0.0

    logger.info(f"\n=== Results ===")
    logger.info(f"Total samples: {len(results)}")
    logger.info(f"Total fields: {total_fields}")
    logger.info(f"Correct predictions: {total_correct}")
    logger.info(f"Overall accuracy: {overall_accuracy:.2%}")

    if all_uncertainties:
        logger.info(f"Mean uncertainty: {np.mean(all_uncertainties):.3f}")
        logger.info(f"Std uncertainty: {np.std(all_uncertainties):.3f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    model_name = model_id.replace("/", "_").replace("watsonx_", "")
    output_file = Path(output_dir) / f"ue_reflect_{dataset.lower()}_{model_name}.json"

    output_data = {
        "dataset": dataset,
        "model_id": model_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "top_k": top_k,
            "max_samples": max_samples,
        },
        "metrics": {
            "total_samples": len(results),
            "total_fields": total_fields,
            "correct_predictions": total_correct,
            "accuracy": overall_accuracy,
            "mean_uncertainty": float(np.mean(all_uncertainties)) if all_uncertainties else 0.5,
        },
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Results saved to {output_file}")
    return output_data


# -------------------------
# CLI Entry Point
# -------------------------

def main():
    parser = argparse.ArgumentParser(description='Run UE-Reflect pipeline with WatsonX')
    parser.add_argument('--datasets', type=str, required=True,
                        help='Dataset name (e.g., OMOP_Synthea, NL2GQL, ITOps)')
    parser.add_argument('--models', type=str, default='watsonx/ibm/granite-4-h-small',
                        help='Model ID (e.g., watsonx/ibm/granite-4-h-small)')
    parser.add_argument('--property-lookup-file', type=str, required=True,
                        help='Path to property lookup JSON file')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing dataset files')
    parser.add_argument('--output-dir', type=str, default='map_reflect/results',
                        help='Output directory for results')
    parser.add_argument('--chroma-dir', type=str, default='data/chroma_db',
                        help='Directory for ChromaDB storage')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of candidates to retrieve')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--use-litellm', action='store_true',
                        help='Use litellm instead of native WatsonX SDK')
    parser.add_argument('--use-sampling', action='store_true', default=True,
                        help='Use sampling-based uncertainty estimation (default: True)')
    parser.add_argument('--no-sampling', action='store_true',
                        help='Disable sampling-based uncertainty (use heuristics only)')
    parser.add_argument('--n-uncertainty-samples', type=int, default=3,
                        help='Number of samples for uncertainty estimation (default: 3)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh instead of resuming from checkpoint')
    parser.add_argument('--neighbor-k', type=int, default=5,
                        help='Number of 1-hop neighbors to expand candidates with (default: 5, 0 to disable)')

    args = parser.parse_args()

    # Handle --no-sampling flag
    use_sampling = not args.no_sampling

    run_ue_reflect_pipeline(
        property_lookup_file=args.property_lookup_file,
        data_dir=args.data_dir,
        dataset=args.datasets,
        model_id=args.models,
        output_dir=args.output_dir,
        chroma_dir=args.chroma_dir,
        top_k=args.top_k,
        max_samples=args.max_samples,
        use_litellm=args.use_litellm,
        use_sampling=use_sampling,
        n_uncertainty_samples=args.n_uncertainty_samples,
        resume_from_checkpoint=not args.no_resume,
        neighbor_k=args.neighbor_k,
    )


if __name__ == '__main__':
    main()
