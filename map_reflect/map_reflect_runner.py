#!/usr/bin/env python3
"""
Unified Map-Reflect Runner

Consolidated implementation merging:
- batch_runner.py: Batch inference with 10-11x speedup
- ue_reflect.py: Uncertainty quantification and self-reflection
- map_reflect_runner.py: Original wrapper

Features:
- Batch inference by default (configurable batch size)
- Optional uncertainty quantification (--use-uncertainty)
- Optional self-reflection with adaptive depth (--use-reflection)
- Neighborhood expansion via property graph (--neighbor-k)
- Checkpoint/resume capability (--checkpoint-interval, --resume)
- Response caching for efficiency

Usage:
    # Default (fast, batch mode)
    python -m map_reflect.map_reflect_runner \
        --datasets ITOpsGraph \
        --models watsonx/meta-llama/llama-3-3-70b-instruct \
        --property-lookup-file data/itopsgraph_api_lookup.json \
        --data-dir data

    # Full features
    python -m map_reflect.map_reflect_runner \
        --datasets ITOpsGraph \
        --models watsonx/meta-llama/llama-3-3-70b-instruct \
        --property-lookup-file data/itopsgraph_api_lookup.json \
        --data-dir data \
        --use-uncertainty \
        --use-reflection \
        --neighbor-k 5
"""

import os
import sys
import json
import time
import re
import argparse
import hashlib
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import networkx as nx
from tqdm import tqdm
from dotenv import load_dotenv

# Suppress verbose logging before imports
os.environ['LITELLM_LOG'] = 'ERROR'

script_dir = Path(__file__).parent
load_dotenv(script_dir / ".env")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import WatsonX SDK
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
    from ibm_watsonx_ai.foundation_models import ModelInference
    HAS_WATSONX = True
except ImportError:
    HAS_WATSONX = False
    logger.warning("ibm_watsonx_ai not installed, batch mode unavailable")

# Try to import ChromaDB for retrieval
try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    logger.warning("chromadb not installed, retrieval-based candidate selection unavailable")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MapReflectConfig:
    """Configuration for the unified Map-Reflect pipeline."""
    # Batch processing
    batch_size: int = 50
    max_workers: int = 4

    # Candidate retrieval
    top_k: int = 10
    neighbor_k: int = 5  # 1-hop neighbors (0 to disable)

    # Optional features
    use_uncertainty: bool = False
    use_reflection: bool = False
    n_uncertainty_samples: int = 3

    # Checkpointing
    checkpoint_interval: int = 50
    resume: bool = False

    # Caching
    use_cache: bool = True
    cache_dir: Optional[str] = None

    # LLM settings
    max_tokens: int = 4000
    temperature: float = 0.1


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


# =============================================================================
# Shared Utilities
# =============================================================================

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


def safe_parse_json_array(s: str) -> Optional[List]:
    """Parse JSON array from text, handling common issues."""
    if not s:
        return None
    m = re.search(r"(\[.*\])", s, re.S)
    txt = m.group(1) if m else s
    try:
        return json.loads(txt)
    except Exception:
        try:
            return json.loads(txt.replace("'", '"'))
        except Exception:
            return None


def safe_parse_json_object(s: str) -> Optional[Dict]:
    """Parse JSON object from text."""
    if not s:
        return None
    m = re.search(r"(\{.*\})", s, re.S)
    txt = m.group(1) if m else s
    try:
        return json.loads(txt)
    except Exception:
        try:
            return json.loads(txt.replace("'", '"'))
        except Exception:
            return None


def build_property_graph(lookup: Dict[str, Any]) -> nx.Graph:
    """Build a property graph connecting semantically related properties.

    Properties are connected if they share:
    - Same product_name (+0.5 weight)
    - Same type_name (+0.3 weight)
    - Similar field names (+0.2 * Jaccard similarity)

    Edges are created for pairs with weight > 0.1.
    """
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
    candidate_ids = {c.get("id", c.get("name", "")) for c in candidates}

    # Collect neighbors
    neighbor_ids = set()
    for cand in candidates:
        cand_id = cand.get("id", cand.get("name", ""))
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
            "id": nbr_id,
            "name": nbr_id,
            "doc": f"{nbr_id} -- {nbr_desc}",
            "description": nbr_desc,
            "distance": 0.6,  # Slightly worse than direct retrieval
            "type": nbr_info.get("type", "") if isinstance(nbr_info, dict) else "",
            "from_neighbor_expansion": True
        })

    if neighbor_candidates:
        logger.debug(f"Expanded candidates with {len(neighbor_candidates)} neighbors")

    return candidates + neighbor_candidates


# =============================================================================
# LLM Client
# =============================================================================

class WatsonXBatchClient:
    """WatsonX client optimized for batch inference with caching."""

    def __init__(self, model_id: str, config: MapReflectConfig):
        if not HAS_WATSONX:
            raise RuntimeError("ibm_watsonx_ai required for batch processing")

        self.model_id = model_id.replace("watsonx/", "")
        self.config = config
        self._model = None
        self._lock = threading.Lock()

        # Response cache
        self._cache: Dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_model(self):
        """Get or create model instance (thread-safe)."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    api_key = os.getenv("WATSONX_API_KEY") or os.getenv("WATSONX_APIKEY")
                    project_id = os.getenv("WATSONX_PROJECT_ID")
                    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

                    if not api_key or not project_id:
                        raise RuntimeError("WATSONX_API_KEY and WATSONX_PROJECT_ID required")

                    _, stop_sequences = format_prompt_for_model([], self.model_id)

                    self._model = ModelInference(
                        model_id=self.model_id,
                        params={
                            GenParams.MAX_NEW_TOKENS: self.config.max_tokens,
                            GenParams.TEMPERATURE: self.config.temperature,
                            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
                            GenParams.STOP_SEQUENCES: stop_sequences,
                        },
                        credentials=Credentials(url=url, api_key=api_key),
                        project_id=project_id
                    )
                    logger.info(f"Initialized WatsonX model: {self.model_id}")

        return self._model

    def _cache_key(self, prompt: str) -> str:
        """Generate cache key for a prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def generate_batch(
        self,
        prompts: List[str],
        use_cache: bool = True,
        batch_size: int = 50
    ) -> List[str]:
        """
        Generate responses for multiple prompts using batch inference.

        Args:
            prompts: List of prompts to process
            use_cache: Whether to use response caching
            batch_size: Max prompts per API call

        Returns:
            List of responses in same order as prompts
        """
        if not prompts:
            return []

        model = self._get_model()
        results = [None] * len(prompts)
        uncached_indices = []
        uncached_prompts = []

        # Check cache
        if use_cache and self.config.use_cache:
            for i, prompt in enumerate(prompts):
                key = self._cache_key(prompt)
                if key in self._cache:
                    results[i] = self._cache[key]
                    self._cache_hits += 1
                else:
                    uncached_indices.append(i)
                    uncached_prompts.append(prompt)
                    self._cache_misses += 1

            if self._cache_hits > 0:
                logger.debug(f"Cache: {self._cache_hits} hits, {self._cache_misses} misses")
        else:
            uncached_indices = list(range(len(prompts)))
            uncached_prompts = prompts

        if not uncached_prompts:
            return results

        # Process in batches
        for batch_start in range(0, len(uncached_prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(uncached_prompts))
            batch_prompts = uncached_prompts[batch_start:batch_end]
            batch_indices = uncached_indices[batch_start:batch_end]

            t0 = time.time()
            try:
                # Use max concurrency (10) for parallel requests
                responses = model.generate_text(prompt=batch_prompts, concurrency_limit=10)

                if isinstance(responses, str):
                    responses = [responses]
                elif not isinstance(responses, list):
                    responses = list(responses)

                t1 = time.time()
                logger.info(f"Batch {len(batch_prompts)} prompts in {t1-t0:.2f}s ({len(batch_prompts)/(t1-t0):.1f} p/s)")

                # Store results and cache
                for idx, (orig_idx, response) in enumerate(zip(batch_indices, responses)):
                    results[orig_idx] = response
                    if use_cache and self.config.use_cache:
                        key = self._cache_key(batch_prompts[idx])
                        self._cache[key] = response

            except Exception as e:
                logger.error(f"Batch generation failed: {e}")
                # Fallback to sequential
                for idx, (orig_idx, prompt) in enumerate(zip(batch_indices, batch_prompts)):
                    try:
                        response = model.generate_text(prompt=prompt)
                        if isinstance(response, list):
                            response = response[0]
                        results[orig_idx] = response
                        if use_cache and self.config.use_cache:
                            self._cache[self._cache_key(prompt)] = response
                    except Exception as e2:
                        logger.error(f"Fallback failed: {e2}")
                        results[orig_idx] = f"Error: {e2}"

        return results

    def generate_single(self, prompt: str, temperature: float = None) -> str:
        """Generate a single response (for reflection step)."""
        model = self._get_model()

        # Check cache first
        if self.config.use_cache:
            key = self._cache_key(prompt)
            if key in self._cache:
                self._cache_hits += 1
                return self._cache[key]
            self._cache_misses += 1

        try:
            response = model.generate_text(prompt=prompt)
            if isinstance(response, list):
                response = response[0]

            if self.config.use_cache:
                self._cache[self._cache_key(prompt)] = response

            return response
        except Exception as e:
            logger.error(f"Single generation failed: {e}")
            return f"Error: {e}"

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache)
        }


# =============================================================================
# Uncertainty Estimation
# =============================================================================

class UncertaintyEstimator:
    """Sampling-based uncertainty estimation for rankings."""

    def __init__(self, client: WatsonXBatchClient, n_samples: int = 3):
        self.client = client
        self.n_samples = n_samples

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
            try:
                return float(v) if v is not None else default
            except (ValueError, TypeError):
                return default

        primary_scores = {r.get("id", ""): _safe_score(r.get("score", 0.5)) for r in primary_ranking}

        # Generate additional samples
        agreement_count = 1  # Primary response agrees with itself
        score_variances = []

        for _ in range(self.n_samples - 1):
            try:
                alt_prompt = prompt + "\n\nProvide an alternative ranking considering different interpretations."
                sample_response = self.client.generate_single(alt_prompt)
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
        uncertainty = 1.0 - (0.6 * top1_agreement_rate + 0.4 * (1.0 - min(avg_score_variance * 2, 1.0)))
        uncertainty = max(0.0, min(1.0, uncertainty))

        return {
            "uncertainty": float(uncertainty),
            "top1_agreement": float(top1_agreement_rate),
            "score_variance": float(avg_score_variance),
            "n_samples": self.n_samples,
            "method": "sampling_consistency"
        }


def estimate_uncertainty_from_response(response: str, candidates: List[Dict]) -> float:
    """
    Quick uncertainty estimation from response characteristics (heuristics).
    """
    parsed = safe_parse_json_array(response)

    if not parsed:
        return 0.8  # High uncertainty - couldn't parse

    if len(parsed) == 0:
        return 0.9  # Very high uncertainty - empty response

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

    # Check score spread
    def safe_float(v, default=0.5):
        try:
            return float(v) if v is not None else default
        except (ValueError, TypeError):
            return default

    scores = [safe_float(r.get("score", 0.5), 0.5) for r in parsed if isinstance(r, dict)]
    score_range = max(scores) - min(scores) if len(scores) > 1 else 0.5

    # Combine heuristics
    uncertainty = 0.0
    uncertainty += 0.3 * (1.0 - top_score)
    uncertainty += 0.2 * (0.0 if has_reasons else 1.0)
    uncertainty += 0.3 * (1.0 - min(score_range * 1.5, 1.0))
    uncertainty += 0.2 * 0.3  # Base uncertainty

    return max(0.0, min(1.0, uncertainty))


# =============================================================================
# Error Detection for Reflection
# =============================================================================

def detect_mapping_errors(
    source_field: str,
    source_product: str,
    target_product: str,
    initial_ranking: List[Dict],
    candidates: List[Dict]
) -> List[str]:
    """
    Analyze the initial ranking to detect specific error patterns.

    Returns a list of error messages describing potential problems.
    """
    errors = []

    if not initial_ranking or not isinstance(initial_ranking[0], dict):
        return ["Could not parse initial ranking"]

    top_pred = initial_ranking[0].get("id", "")
    top_reason = initial_ranking[0].get("reason", "")

    # Extract field name components
    source_parts = source_field.lower().replace(".", "_").split("_")
    top_pred_parts = top_pred.lower().replace(".", "_").split("_")

    # Error 1: Predicted source-side field instead of target-side
    if top_pred.lower().startswith(source_product.lower() + "_"):
        errors.append(
            f"ERROR: Top prediction '{top_pred}' is from SOURCE product '{source_product}', "
            f"but should be from TARGET product '{target_product}'. "
            "The model matched by name similarity instead of finding the semantic equivalent in the target schema."
        )

    # Error 2: Foreign key mapped to wrong table
    # Foreign keys typically end in _id and reference another table's primary key
    if source_field.endswith("_id") or ".id" in source_field.lower():
        # Check if this looks like a foreign key (contains another table name)
        # e.g., concert_stadium_id should map to stadium's primary key, not concert's
        for part in source_parts:
            if part not in ["id", source_product.lower()] and len(part) > 2:
                # This might be a referenced table name
                referenced_table = part
                # Check if top prediction is in the wrong table
                if top_pred_parts and top_pred_parts[0] != referenced_table:
                    if referenced_table in source_field.lower() and referenced_table not in top_pred.lower():
                        errors.append(
                            f"ERROR: Source field '{source_field}' appears to be a foreign key referencing '{referenced_table}'. "
                            f"The top prediction '{top_pred}' may be mapping to the wrong table. "
                            f"Foreign keys should map to the PRIMARY KEY of the referenced table (e.g., {referenced_table}_id or {referenced_table}.id)."
                        )
                        break

    # Error 3: Name similarity without semantic match
    # Check if top prediction has high lexical overlap but might be semantically wrong
    source_words = set(source_parts) - {"id", "input", "type", "filter", source_product.lower()}
    pred_words = set(top_pred_parts) - {"id", "input", "type", "filter", target_product.lower()}

    overlap = source_words & pred_words
    if overlap and len(overlap) >= len(source_words) * 0.5:
        # High name overlap - check if the reason mentions "name" or "similar"
        reason_lower = top_reason.lower()
        if any(word in reason_lower for word in ["name", "similar", "same", "match", "identical"]):
            errors.append(
                f"WARNING: Top prediction may be based on NAME SIMILARITY rather than semantic meaning. "
                f"Overlapping terms: {overlap}. "
                "Verify that the fields represent the SAME DATA, not just similar names."
            )

    # Error 4: Type mismatch indicators
    # Check for type indicators in field names that don't match
    source_type_hints = []
    pred_type_hints = []

    type_keywords = ["string", "int", "bool", "float", "date", "time", "enum", "list", "array"]
    for kw in type_keywords:
        if kw in source_field.lower():
            source_type_hints.append(kw)
        if kw in top_pred.lower():
            pred_type_hints.append(kw)

    if source_type_hints and pred_type_hints and set(source_type_hints) != set(pred_type_hints):
        errors.append(
            f"WARNING: Possible type mismatch. Source field suggests type '{source_type_hints}', "
            f"but prediction suggests type '{pred_type_hints}'. Verify type compatibility."
        )

    # Error 5: Score-based warning
    top_score = initial_ranking[0].get("score", 0.5)
    try:
        top_score = float(top_score)
    except (ValueError, TypeError):
        top_score = 0.5

    if top_score < 0.6:
        errors.append(
            f"WARNING: Low confidence score ({top_score:.2f}) on top prediction. "
            "The model is uncertain - consider alternative candidates."
        )

    # Error 6: Check if top predictions are too similar (ambiguous)
    if len(initial_ranking) >= 2:
        second_pred = initial_ranking[1]
        if isinstance(second_pred, dict):
            second_score = second_pred.get("score", 0)
            try:
                second_score = float(second_score)
            except (ValueError, TypeError):
                second_score = 0

            if abs(top_score - second_score) < 0.1:
                errors.append(
                    f"WARNING: Top two predictions have similar scores ({top_score:.2f} vs {second_score:.2f}). "
                    f"Ambiguous choice between '{top_pred}' and '{second_pred.get('id', 'unknown')}'. "
                    "Carefully compare semantic meanings."
                )

    return errors


# =============================================================================
# Data Loading
# =============================================================================

def load_property_lookup(lookup_file: str) -> Dict[str, Any]:
    """Load property lookup file."""
    with open(lookup_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_dataset(data_dir: Path, dataset: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load evaluation dataset."""
    dataset_lower = dataset.lower()

    # Try different file patterns
    patterns = [
        f"{dataset_lower}_dataset.jsonl",
        f"{dataset_lower}_eval_dataset.jsonl",
        f"{dataset_lower}_coverage_eval_dataset.jsonl",
        f"{dataset}_dataset.jsonl",
    ]

    filepath = None
    for pattern in patterns:
        candidate = data_dir / pattern
        if candidate.exists():
            filepath = candidate
            break

    if not filepath:
        raise FileNotFoundError(f"No dataset file found for {dataset} in {data_dir}")

    samples = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if line.strip():
                data = json.loads(line)
                sample = extract_sample_data(data, dataset, i)
                if sample:
                    samples.append(sample)

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

    # Handle ToolBench and similar formats
    if "ground_truth_mapping" in data and "expected_target" in data:
        return {
            "sample_id": data.get("sample_id", f"{dataset.lower()}_{index:03d}"),
            "utterance": data.get("utterance", ""),
            "expected_source": data.get("expected_source", ""),
            "expected_target": data.get("expected_target"),
            "ground_truth_mapping": data.get("ground_truth_mapping", {}),
        }

    # Handle coverage dataset format
    if "source_fields" in data and "target_fields" in data:
        source_fields = data.get("source_fields", [])
        target_fields = data.get("target_fields", [])
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

    # Handle ITOpsGraph format
    if "mapping" in data or "source_schema" in data:
        return {
            "sample_id": data.get("id", f"{dataset.lower()}_{index:03d}"),
            "utterance": data.get("utterance", data.get("mapping_utterance", "")),
            "expected_source": data.get("source_schema", ""),
            "expected_target": data.get("target_schema", ""),
            "ground_truth_mapping": data.get("mapping", {}),
            "metadata": data.get("metadata", {}),
        }

    return None


# =============================================================================
# ChromaDB Indexing
# =============================================================================

def index_chroma(collection_name: str, data: Dict[str, Any], chroma_dir: str):
    """Index documents in ChromaDB."""
    if not HAS_CHROMADB:
        logger.warning("ChromaDB not available, skipping indexing")
        return None

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
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_model_function,
        metadata={"hnsw:space": "cosine"}
    )

    # Add documents in batches
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
    if not HAS_CHROMADB:
        return []

    embed_model = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=chroma_dir)

    try:
        collection = client.get_collection(name=collection_name, embedding_function=embed_model)
    except Exception:
        return []

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
            "name": full_name,
            "doc": doc,
            "distance": dist,
            "meta": meta
        })

    return candidates


# =============================================================================
# Main Pipeline
# =============================================================================

class MapReflectRunner:
    """
    Unified Map-Reflect runner with batch processing, uncertainty quantification,
    and self-reflection.
    """

    def __init__(
        self,
        property_lookup_file: str,
        data_dir: str,
        model: str,
        dataset: str,
        config: MapReflectConfig = None
    ):
        self.property_lookup_file = Path(property_lookup_file)
        self.data_dir = Path(data_dir)
        self.model = model
        self.dataset = dataset
        self.config = config or MapReflectConfig()

        # Initialize WatsonX client
        self.client = WatsonXBatchClient(model, self.config)

        # Load property lookup
        with open(self.property_lookup_file) as f:
            self.property_lookup = json.load(f)

        # Build property graph for neighbor expansion
        self.property_graph = None
        if self.config.neighbor_k > 0:
            self.property_graph = build_property_graph(self.property_lookup)

        # Initialize uncertainty estimator if needed
        self.uncertainty_estimator = None
        if self.config.use_uncertainty:
            self.uncertainty_estimator = UncertaintyEstimator(
                self.client, self.config.n_uncertainty_samples
            )

        # Results directory
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Chroma directory
        self.chroma_dir = str(self.data_dir / "chroma_db")

        logger.info(f"MapReflectRunner initialized")
        logger.info(f"  Model: {model}")
        logger.info(f"  Dataset: {dataset}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Neighbor-K: {self.config.neighbor_k}")
        logger.info(f"  Uncertainty: {self.config.use_uncertainty}")
        logger.info(f"  Reflection: {self.config.use_reflection}")

    def _format_prompt(self, messages: List[Dict]) -> str:
        """Format messages into a prompt string using model-specific template."""
        formatted, _ = format_prompt_for_model(messages, self.model)
        return formatted

    def _build_ranking_prompt(
        self,
        source_prop: str,
        source_desc: str,
        candidates: List[Dict],
        source_product: str,
        target_product: str
    ) -> str:
        """Build prompt for ranking candidates."""
        cand_block = "\n".join([
            f"{i+1}. id: {c.get('id', c.get('name', ''))}\n   text: {c.get('doc', c.get('description', ''))}\n   distance: {c.get('distance', 0.5):.4f}"
            for i, c in enumerate(candidates[:20])  # Limit for prompt size
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

        messages = [
            {"role": "system", "content": "You are an expert in API schema mapping."},
            {"role": "user", "content": prompt}
        ]
        return self._format_prompt(messages)

    def _build_reflection_prompt(
        self,
        source_prop: str,
        source_desc: str,
        initial_ranking: List[Dict],
        candidates: List[Dict],
        uncertainty_score: float,
        high_uncertainty: bool,
        source_product: str,
        target_product: str,
        detected_errors: List[str] = None
    ) -> str:
        """Build prompt for self-reflection with error analysis."""
        n_candidates = 8 if high_uncertainty else 5
        initial_block = "\n".join([
            f"{i+1}. {r.get('id', 'unknown')} (score={r.get('score')}) - {r.get('reason', '')}"
            for i, r in enumerate(initial_ranking[:n_candidates]) if isinstance(r, dict)
        ])
        cand_block = "\n".join([
            f"- {c.get('id', 'unknown')}: {c.get('doc', '')[:100]}"
            for c in candidates[:n_candidates] if isinstance(c, dict)
        ])

        # Format detected errors
        error_block = ""
        if detected_errors:
            error_block = "\n\n=== DETECTED ERRORS ===\n" + "\n".join(detected_errors)

        # Semantic matching rules (always included)
        semantic_rules = """
SEMANTIC MATCHING RULES - Apply these carefully:
1. Foreign keys (fields ending in _id that reference another table) should map to the PRIMARY KEY of the referenced table, NOT to another foreign key
   - Example: concert_stadium_id (FK to stadium) -> stadium_stadium_id (PK of stadium), NOT concert_concert_id
2. Match the TARGET entity, not the SOURCE entity
   - If source is "concert_stadium_id", the target should be in the STADIUM table, not CONCERT
3. Field names often follow pattern: {table}_{field}. Match the semantic purpose of {field}.
4. Filter/Input types: Match by the semantic operation (eq->equals, like->pattern match, in->contains)
5. Do NOT match by name similarity alone - verify the fields represent the SAME DATA"""

        if high_uncertainty or detected_errors:
            reflection_instruction = f"""CRITICAL: The initial ranking shows HIGH UNCERTAINTY or has DETECTED ERRORS, which strongly suggests the top-ranked candidate may be WRONG.
{semantic_rules}

You MUST:
1. Carefully review the DETECTED ERRORS above (if any)
2. Assume the #1 candidate is likely incorrect if errors were detected
3. Look for the candidate that represents the SAME DATA in the TARGET schema
4. For foreign keys, find the primary key they reference
5. CHANGE your top prediction based on semantic meaning, not name similarity

Be bold - your initial answer is likely wrong if errors were detected."""
        else:
            reflection_instruction = f"""The initial ranking shows reasonable confidence.
{semantic_rules}

Please review for any obvious errors or improvements based on the semantic matching rules above."""

        prompt = f"""You previously ranked {target_product} candidates for this {source_product} property:

{source_product} property: {source_prop}
Description: {source_desc}

Initial ranking (best first):
{initial_block}

Candidates:
{cand_block}

Uncertainty Score: {uncertainty_score:.3f} {"(HIGH)" if high_uncertainty else "(LOW)"}{error_block}

{reflection_instruction}

Task:
1) Analyze the detected errors (if any) and point out problems in the initial ranking.
2) Produce a revised JSON array (id, score, reason) from best to worst match.

Return a JSON object:
{{"analysis": "your error analysis addressing the detected issues", "revised_ranking": [array of ranked objects]}}"""

        messages = [
            {"role": "system", "content": "You are an expert schema mapper performing self-reflection."},
            {"role": "user", "content": prompt}
        ]
        return self._format_prompt(messages)

    def _parse_ranking_response(self, response: str, candidates: List[Dict]) -> List[Dict]:
        """Parse ranking response and map numeric IDs back to actual field IDs."""
        parsed = safe_parse_json_array(response)

        if parsed is None or not all(isinstance(item, dict) and 'id' in item for item in parsed):
            # Fallback to distance-based scoring
            maxd = max([c.get('distance', 1.0) for c in candidates]) if candidates else 1.0
            scored = []
            for c in candidates:
                score = 1.0 - (c.get('distance', 0.5) / (maxd + 1e-8))
                scored.append({
                    "id": c.get('id', c.get('name', '')),
                    "score": float(max(0.0, min(1.0, score))),
                    "reason": "fallback distance-based score"
                })
            return sorted(scored, key=lambda x: -x['score'])

        # Map numeric IDs back to actual field IDs
        id_to_candidate = {str(i+1): c.get('id', c.get('name', '')) for i, c in enumerate(candidates)}
        for item in parsed:
            item_id = str(item['id'])
            if item_id in id_to_candidate:
                item['id'] = id_to_candidate[item_id]

        return parsed

    def _parse_reflection_response(self, response: str, candidates: List[Dict], initial_ranking: List[Dict]) -> Dict:
        """Parse reflection response."""
        parsed = safe_parse_json_object(response)

        if parsed and "revised_ranking" in parsed:
            result = parsed
        else:
            array_parsed = safe_parse_json_array(response)
            if array_parsed:
                result = {"analysis": "Parsing fallback", "revised_ranking": array_parsed}
            else:
                result = {"analysis": "Could not parse response", "revised_ranking": initial_ranking}

        # Map numeric IDs back to actual field IDs
        if result.get("revised_ranking"):
            id_to_candidate = {str(i+1): c.get('id', c.get('name', '')) for i, c in enumerate(candidates)}
            for item in result["revised_ranking"]:
                if isinstance(item, dict) and 'id' in item:
                    item_id = str(item['id'])
                    if item_id in id_to_candidate:
                        item['id'] = id_to_candidate[item_id]

        return result

    def _get_candidates_for_sample(
        self,
        src_field: str,
        sample: Dict,
        collection_name: str
    ) -> List[Dict]:
        """Get candidates for a source field."""
        src_info = self.property_lookup.get(src_field, {})
        src_desc = src_info.get("description", src_field) if isinstance(src_info, dict) else str(src_info)
        query_text = f"{src_field}: {src_desc}"

        # Check for explicit target fields in sample
        sample_target_fields = sample.get("target_fields", [])

        if sample_target_fields:
            # Use explicit target fields
            candidates = []
            for prop_name in sample_target_fields:
                prop_lookup = self.property_lookup.get(prop_name, {})
                prop_desc = prop_lookup.get("description", prop_name) if isinstance(prop_lookup, dict) else str(prop_lookup)
                candidates.append({
                    "id": prop_name,
                    "name": prop_name,
                    "doc": f"{prop_name} -- {prop_desc}",
                    "distance": 0.5,
                    "meta": prop_lookup if isinstance(prop_lookup, dict) else {}
                })
        else:
            # Check for target schema filtering (multi-target datasets like ITOpsGraph)
            target_schema = sample.get("expected_target", sample.get("target_schema", ""))
            sample_metadata = sample.get("metadata", {})
            target_type = sample_metadata.get("target_type", "")

            if target_schema and isinstance(target_schema, str):
                # Build target prefix for filtering
                if target_type:
                    # ITOpsGraph style: product + type prefix (e.g., "turbonomic_FilterInput.")
                    target_prefix = f"{target_schema.lower()}_{target_type}."
                else:
                    # Simple product prefix (e.g., "turbonomic_")
                    target_prefix = f"{target_schema.lower()}_"

                # Filter candidates from lookup by target prefix
                candidates = []
                for prop_name, prop_info in self.property_lookup.items():
                    if prop_name.lower().startswith(target_prefix.lower()):
                        prop_desc = prop_info.get("description", prop_name) if isinstance(prop_info, dict) else str(prop_info)
                        candidates.append({
                            "id": prop_name,
                            "name": prop_name,
                            "doc": f"{prop_name} -- {prop_desc}",
                            "distance": 0.5,
                            "meta": prop_info if isinstance(prop_info, dict) else {}
                        })

                # Limit candidates to avoid token overflow
                max_candidates = self.config.top_k * 3
                if len(candidates) > max_candidates:
                    logger.debug(f"Truncating {len(candidates)} candidates to {max_candidates}")
                    candidates = candidates[:max_candidates]
            else:
                # Fallback to retrieval-based candidates
                candidates = retrieve_candidates(query_text, collection_name, self.chroma_dir, self.config.top_k)

        # Expand with neighbors
        if self.property_graph is not None and self.config.neighbor_k > 0:
            candidates = expand_candidates_with_neighbors(
                candidates, self.property_graph, self.property_lookup, self.config.neighbor_k
            )

        return candidates

    def process_sample(
        self,
        sample: Dict,
        collection_name: str,
        all_uncertainties: List[float]
    ) -> Dict:
        """Process a single sample."""
        sample_id = sample.get("sample_id", "unknown")
        ground_truth = sample.get("ground_truth_mapping", {})

        # Determine source/target products
        expected_source = sample.get("expected_source", "")
        expected_target = sample.get("expected_target", "")

        if isinstance(expected_source, dict):
            source_product = expected_source.get("name", "SOURCE").upper()
        else:
            source_product = str(expected_source).split("_")[0].upper() if expected_source else "SOURCE"

        if isinstance(expected_target, dict):
            target_product = expected_target.get("name", "TARGET").upper()
        else:
            target_product = str(expected_target).split("_")[0].upper() if expected_target else "TARGET"

        sample_result = {
            "sample_id": sample_id,
            "utterance": sample.get("utterance", ""),
            "expected_source": expected_source,
            "expected_target": expected_target,
            "ground_truth_mapping": ground_truth,
            "predictions": {},
        }

        sample_uncertainties = []

        # Process each source field
        for src_field, gt_target in ground_truth.items():
            # Get candidates
            candidates = self._get_candidates_for_sample(src_field, sample, collection_name)

            if not candidates:
                logger.warning(f"No candidates for {src_field}, skipping")
                continue

            # Get source description
            src_info = self.property_lookup.get(src_field, {})
            src_desc = src_info.get("description", src_field) if isinstance(src_info, dict) else str(src_info)

            # Build ranking prompt
            ranking_prompt = self._build_ranking_prompt(
                src_field, src_desc, candidates, source_product, target_product
            )

            # Get ranking response
            response = self.client.generate_single(ranking_prompt)
            ranking = self._parse_ranking_response(response, candidates)

            # Estimate uncertainty
            if self.config.use_uncertainty and self.uncertainty_estimator:
                uncertainty_info = self.uncertainty_estimator.estimate_ranking_uncertainty(
                    ranking_prompt, response, candidates
                )
            else:
                uncertainty_score = estimate_uncertainty_from_response(response, candidates)
                uncertainty_info = {"uncertainty": uncertainty_score, "method": "heuristics"}

            uncertainty_score = uncertainty_info.get("uncertainty", 0.5)
            sample_uncertainties.append(uncertainty_score)
            all_uncertainties.append(uncertainty_score)

            # Detect mapping errors from initial ranking
            detected_errors = detect_mapping_errors(
                src_field, source_product, target_product, ranking, candidates
            )

            # Self-reflection (optional)
            reflection = None
            if self.config.use_reflection:
                # Compute dynamic threshold
                if len(all_uncertainties) > 1:
                    mean_u = np.mean(all_uncertainties)
                    std_u = np.std(all_uncertainties)
                    threshold = mean_u + 0.5 * std_u
                else:
                    threshold = 0.6

                high_uncertainty = uncertainty_score > threshold

                # Trigger reflection if high uncertainty OR errors detected
                should_reflect = high_uncertainty or len(detected_errors) > 0

                if should_reflect:
                    reflection_prompt = self._build_reflection_prompt(
                        src_field, src_desc, ranking, candidates,
                        uncertainty_score, high_uncertainty, source_product, target_product,
                        detected_errors=detected_errors
                    )
                    reflection_response = self.client.generate_single(reflection_prompt)
                    reflection = self._parse_reflection_response(reflection_response, candidates, ranking)
                    reflection["high_uncertainty"] = high_uncertainty
                    reflection["detected_errors"] = detected_errors

            # Store prediction
            sample_result["predictions"][src_field] = {
                "initial_ranking": ranking,
                "reflection": reflection,
                "uncertainty": float(uncertainty_score),
                "uncertainty_info": uncertainty_info,
                "detected_errors": detected_errors,
            }

        # Evaluate sample
        correct_count = 0
        total_count = 0
        for src_field, gt_target in ground_truth.items():
            if src_field in sample_result["predictions"]:
                pred = sample_result["predictions"][src_field]
                if self.config.use_reflection and pred.get("reflection"):
                    revised = pred["reflection"].get("revised_ranking", [])
                else:
                    revised = pred.get("initial_ranking", [])

                if revised and isinstance(revised[0], dict):
                    top_pred = revised[0].get("id", "")
                    if top_pred == gt_target:
                        correct_count += 1
                total_count += 1

        sample_result["accuracy"] = correct_count / total_count if total_count > 0 else 0.0
        sample_result["correct_count"] = correct_count
        sample_result["total_count"] = total_count
        sample_result["mean_uncertainty"] = float(np.mean(sample_uncertainties)) if sample_uncertainties else 0.5

        return sample_result

    def run(self, samples: List[Dict]) -> Dict:
        """
        Run the Map-Reflect pipeline on samples using batch inference.

        This method batches all ranking prompts together for efficient API usage,
        then processes reflection in a second batch pass if enabled.

        Args:
            samples: List of sample dicts

        Returns:
            Results dict with metrics and per-sample results
        """
        total_start = time.time()
        results = []
        all_uncertainties = []

        # Setup checkpoint file
        model_name = self.model.replace("/", "_").replace("watsonx_", "")
        checkpoint_file = self.results_dir / f"checkpoint_{self.dataset.lower()}_{model_name}.json"

        # Load checkpoint if resuming
        processed_ids = set()
        if self.config.resume and checkpoint_file.exists():
            try:
                with open(checkpoint_file) as f:
                    checkpoint_data = json.load(f)
                results = checkpoint_data.get("results", [])
                all_uncertainties = checkpoint_data.get("all_uncertainties", [])
                processed_ids = {r["sample_id"] for r in results}
                logger.info(f"Resumed from checkpoint: {len(processed_ids)} samples already processed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

        # Index target properties for retrieval
        collection_name = f"mapreflect_{self.dataset.lower()}_target"
        if HAS_CHROMADB:
            index_chroma(collection_name, self.property_lookup, self.chroma_dir)

        # Filter samples to process
        samples_to_process = []
        for i, sample in enumerate(samples):
            sample_id = sample.get("sample_id", f"sample_{i}")
            if sample_id in processed_ids:
                continue
            if not sample.get("ground_truth_mapping"):
                continue
            samples_to_process.append((i, sample))

        if not samples_to_process:
            logger.info("No samples to process")
            return self._build_final_results(results, all_uncertainties, time.time() - total_start)

        logger.info(f"Processing {len(samples_to_process)} samples in batch mode (batch_size={self.config.batch_size})")

        # =====================================================================
        # PHASE 1: Collect all ranking prompts
        # =====================================================================
        logger.info("Phase 1: Collecting ranking prompts...")
        prompt_tasks = []  # List of (sample_idx, sample, src_field, gt_target, candidates, prompt, source_product, target_product)

        for sample_idx, sample in tqdm(samples_to_process, desc="Collecting prompts"):
            ground_truth = sample.get("ground_truth_mapping", {})
            expected_source = sample.get("expected_source", "")
            expected_target = sample.get("expected_target", "")

            if isinstance(expected_source, dict):
                source_product = expected_source.get("name", "SOURCE").upper()
            else:
                source_product = str(expected_source).split("_")[0].upper() if expected_source else "SOURCE"

            if isinstance(expected_target, dict):
                target_product = expected_target.get("name", "TARGET").upper()
            else:
                target_product = str(expected_target).split("_")[0].upper() if expected_target else "TARGET"

            for src_field, gt_target in ground_truth.items():
                candidates = self._get_candidates_for_sample(src_field, sample, collection_name)
                if not candidates:
                    continue

                src_info = self.property_lookup.get(src_field, {})
                src_desc = src_info.get("description", src_field) if isinstance(src_info, dict) else str(src_info)

                ranking_prompt = self._build_ranking_prompt(
                    src_field, src_desc, candidates, source_product, target_product
                )

                prompt_tasks.append({
                    "sample_idx": sample_idx,
                    "sample": sample,
                    "src_field": src_field,
                    "gt_target": gt_target,
                    "candidates": candidates,
                    "prompt": ranking_prompt,
                    "src_desc": src_desc,
                    "source_product": source_product,
                    "target_product": target_product,
                })

        logger.info(f"Collected {len(prompt_tasks)} ranking prompts from {len(samples_to_process)} samples")

        # =====================================================================
        # PHASE 2: Batch inference for initial rankings
        # =====================================================================
        logger.info("Phase 2: Batch inference for rankings...")
        ranking_prompts = [t["prompt"] for t in prompt_tasks]
        ranking_responses = self.client.generate_batch(
            ranking_prompts,
            use_cache=True,
            batch_size=self.config.batch_size
        )

        # =====================================================================
        # PHASE 3: Process responses and collect reflection prompts
        # =====================================================================
        logger.info("Phase 3: Processing responses...")
        reflection_tasks = []  # Tasks that need reflection

        for task_idx, (task, response) in enumerate(zip(prompt_tasks, ranking_responses)):
            ranking = self._parse_ranking_response(response, task["candidates"])
            task["ranking"] = ranking
            task["response"] = response

            # Estimate uncertainty (heuristic - fast)
            uncertainty_score = estimate_uncertainty_from_response(response, task["candidates"])
            task["uncertainty"] = uncertainty_score
            task["uncertainty_info"] = {"uncertainty": uncertainty_score, "method": "heuristics"}
            all_uncertainties.append(uncertainty_score)

            # Detect mapping errors
            detected_errors = detect_mapping_errors(
                task["src_field"], task["source_product"], task["target_product"],
                ranking, task["candidates"]
            )
            task["detected_errors"] = detected_errors

            # Check if reflection is needed
            if self.config.use_reflection:
                if len(all_uncertainties) > 1:
                    mean_u = np.mean(all_uncertainties)
                    std_u = np.std(all_uncertainties)
                    threshold = mean_u + 0.5 * std_u
                else:
                    threshold = 0.6

                high_uncertainty = uncertainty_score > threshold
                should_reflect = high_uncertainty or len(detected_errors) > 0
                task["high_uncertainty"] = high_uncertainty

                if should_reflect:
                    reflection_prompt = self._build_reflection_prompt(
                        task["src_field"], task["src_desc"], ranking, task["candidates"],
                        uncertainty_score, high_uncertainty, task["source_product"], task["target_product"],
                        detected_errors=detected_errors
                    )
                    task["reflection_prompt"] = reflection_prompt
                    reflection_tasks.append(task_idx)

        # =====================================================================
        # PHASE 4: Batch inference for reflections (if any)
        # =====================================================================
        if reflection_tasks and self.config.use_reflection:
            logger.info(f"Phase 4: Batch inference for {len(reflection_tasks)} reflections...")
            reflection_prompts = [prompt_tasks[idx]["reflection_prompt"] for idx in reflection_tasks]
            reflection_responses = self.client.generate_batch(
                reflection_prompts,
                use_cache=True,
                batch_size=self.config.batch_size
            )

            for task_idx, response in zip(reflection_tasks, reflection_responses):
                task = prompt_tasks[task_idx]
                reflection = self._parse_reflection_response(response, task["candidates"], task["ranking"])
                reflection["high_uncertainty"] = task.get("high_uncertainty", False)
                reflection["detected_errors"] = task.get("detected_errors", [])
                task["reflection"] = reflection
        else:
            logger.info("Phase 4: Skipped (no reflections needed)")

        # =====================================================================
        # PHASE 5: Assemble results by sample
        # =====================================================================
        logger.info("Phase 5: Assembling results...")

        # Group tasks by sample
        sample_tasks = {}
        for task in prompt_tasks:
            sample_idx = task["sample_idx"]
            if sample_idx not in sample_tasks:
                sample_tasks[sample_idx] = []
            sample_tasks[sample_idx].append(task)

        # Build results for each sample
        for sample_idx, sample in samples_to_process:
            tasks = sample_tasks.get(sample_idx, [])
            if not tasks:
                continue

            sample = tasks[0]["sample"]
            ground_truth = sample.get("ground_truth_mapping", {})

            sample_result = {
                "sample_id": sample.get("sample_id", f"sample_{sample_idx}"),
                "utterance": sample.get("utterance", ""),
                "expected_source": sample.get("expected_source", ""),
                "expected_target": sample.get("expected_target", ""),
                "ground_truth_mapping": ground_truth,
                "predictions": {},
            }

            sample_uncertainties = []
            for task in tasks:
                src_field = task["src_field"]
                sample_uncertainties.append(task["uncertainty"])

                sample_result["predictions"][src_field] = {
                    "initial_ranking": task["ranking"],
                    "reflection": task.get("reflection"),
                    "uncertainty": float(task["uncertainty"]),
                    "uncertainty_info": task["uncertainty_info"],
                    "detected_errors": task.get("detected_errors", []),
                }

            # Evaluate sample
            correct_count = 0
            total_count = 0
            for src_field, gt_target in ground_truth.items():
                if src_field in sample_result["predictions"]:
                    pred = sample_result["predictions"][src_field]
                    if self.config.use_reflection and pred.get("reflection"):
                        revised = pred["reflection"].get("revised_ranking", [])
                    else:
                        revised = pred.get("initial_ranking", [])

                    if revised and isinstance(revised[0], dict):
                        top_pred = revised[0].get("id", "")
                        if top_pred == gt_target:
                            correct_count += 1
                    total_count += 1

            sample_result["accuracy"] = correct_count / total_count if total_count > 0 else 0.0
            sample_result["correct_count"] = correct_count
            sample_result["total_count"] = total_count
            sample_result["mean_uncertainty"] = float(np.mean(sample_uncertainties)) if sample_uncertainties else 0.5

            results.append(sample_result)

            # Checkpoint periodically
            if len(results) % self.config.checkpoint_interval == 0:
                checkpoint_data = {
                    "dataset": self.dataset,
                    "model": self.model,
                    "results": results,
                    "all_uncertainties": all_uncertainties,
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, cls=NumpyEncoder)
                logger.info(f"Checkpoint saved: {len(results)} samples")

        return self._build_final_results(results, all_uncertainties, time.time() - total_start)

    def _build_final_results(self, results: List[Dict], all_uncertainties: List[float], total_time: float) -> Dict:
        """Build the final results dictionary."""
        total_correct = sum(r["correct_count"] for r in results)
        total_fields = sum(r["total_count"] for r in results)
        overall_accuracy = total_correct / total_fields if total_fields > 0 else 0.0

        cache_stats = self.client.get_cache_stats()

        return {
            "dataset": self.dataset,
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "batch_size": self.config.batch_size,
                "neighbor_k": self.config.neighbor_k,
                "use_uncertainty": self.config.use_uncertainty,
                "use_reflection": self.config.use_reflection,
                "n_uncertainty_samples": self.config.n_uncertainty_samples,
            },
            "metrics": {
                "total_samples": len(results),
                "total_fields": total_fields,
                "correct_predictions": total_correct,
                "accuracy": overall_accuracy,
                "mean_uncertainty": float(np.mean(all_uncertainties)) if all_uncertainties else 0.5,
                "total_time_seconds": total_time,
                "samples_per_second": len(results) / total_time if total_time > 0 else 0,
            },
            "cache_stats": cache_stats,
            "results": results
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Map-Reflect Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument("--datasets", required=True, help="Comma-separated datasets")
    parser.add_argument("--models", required=True, help="Model ID")
    parser.add_argument("--property-lookup-file", required=True, help="Property lookup JSON")
    parser.add_argument("--data-dir", required=True, help="Data directory")

    # Batch processing
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size (default: 50)")
    parser.add_argument("--no-cache", action="store_true", help="Disable response caching")

    # Candidate retrieval
    parser.add_argument("--top-k", type=int, default=10, help="Top-K candidates (default: 10)")
    parser.add_argument("--neighbor-k", type=int, default=5, help="1-hop neighbors (default: 5, 0 to disable)")

    # Optional features
    parser.add_argument("--use-uncertainty", action="store_true", help="Enable uncertainty quantification")
    parser.add_argument("--use-reflection", action="store_true", help="Enable self-reflection")
    parser.add_argument("--n-uncertainty-samples", type=int, default=3, help="Samples for UQ (default: 3)")

    # Checkpointing
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Save every N samples (default: 50)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    # Limiting
    parser.add_argument("--max-samples", type=int, help="Limit samples for testing")

    # Output
    parser.add_argument("--output-dir", type=str, default="map_reflect/results", help="Output directory")

    args = parser.parse_args()

    config = MapReflectConfig(
        batch_size=args.batch_size,
        use_cache=not args.no_cache,
        top_k=args.top_k,
        neighbor_k=args.neighbor_k,
        use_uncertainty=args.use_uncertainty,
        use_reflection=args.use_reflection,
        n_uncertainty_samples=args.n_uncertainty_samples,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )

    for dataset in args.datasets.split(","):
        dataset = dataset.strip()
        logger.info(f"Processing dataset: {dataset}")

        # Load data
        samples = load_dataset(Path(args.data_dir), dataset, args.max_samples)

        # Initialize runner
        runner = MapReflectRunner(
            property_lookup_file=args.property_lookup_file,
            data_dir=args.data_dir,
            model=args.models,
            dataset=dataset,
            config=config
        )

        # Run evaluation
        results = runner.run(samples)

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = args.models.replace("/", "_").replace("watsonx_", "")
        output_file = output_dir / f"mapreflect_{dataset.lower()}_{model_name}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Results saved to {output_file}")
        logger.info(f"Metrics: {results['metrics']}")


if __name__ == "__main__":
    main()
