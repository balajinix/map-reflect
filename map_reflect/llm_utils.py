"""
LLM utilities with batching and caching support for WatsonX.

This module provides:
1. Batch inference - process multiple prompts in a single API call
2. Response caching - avoid redundant LLM calls for identical prompts
3. Concurrent processing - process multiple samples in parallel
"""

import os
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Global cache for LLM responses
_response_cache: Dict[str, str] = {}
_cache_lock = Lock()
_cache_file: Optional[Path] = None

# Model instance cache
_model_cache: Dict[str, Any] = {}
_model_lock = Lock()


def init_cache(cache_dir: str = None):
    """Initialize the response cache, optionally loading from disk."""
    global _response_cache, _cache_file

    if cache_dir:
        _cache_file = Path(cache_dir) / "llm_response_cache.json"
        if _cache_file.exists():
            try:
                with open(_cache_file, 'r') as f:
                    _response_cache = json.load(f)
                logger.info(f"Loaded {len(_response_cache)} cached responses from {_cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                _response_cache = {}


def save_cache():
    """Save the response cache to disk."""
    global _response_cache, _cache_file

    if _cache_file:
        try:
            _cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(_cache_file, 'w') as f:
                json.dump(_response_cache, f)
            logger.info(f"Saved {len(_response_cache)} responses to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


def _hash_prompt(prompt: str, model_id: str) -> str:
    """Create a hash key for a prompt."""
    content = f"{model_id}:{prompt}"
    return hashlib.md5(content.encode()).hexdigest()


def get_cached_response(prompt: str, model_id: str) -> Optional[str]:
    """Get a cached response if available."""
    key = _hash_prompt(prompt, model_id)
    with _cache_lock:
        return _response_cache.get(key)


def set_cached_response(prompt: str, model_id: str, response: str):
    """Cache a response."""
    key = _hash_prompt(prompt, model_id)
    with _cache_lock:
        _response_cache[key] = response


def get_watsonx_model(model_id: str):
    """Get or create a cached WatsonX model instance."""
    global _model_cache

    with _model_lock:
        if model_id in _model_cache:
            return _model_cache[model_id]

        try:
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
            from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
            from ibm_watsonx_ai.foundation_models import ModelInference
        except ImportError:
            raise RuntimeError("ibm_watsonx_ai not installed")

        api_key = os.getenv("WATSONX_API_KEY") or os.getenv("WATSONX_APIKEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

        if not api_key or not project_id:
            raise RuntimeError("WATSONX_API_KEY and WATSONX_PROJECT_ID required")

        # Remove watsonx/ prefix if present
        clean_model_id = model_id.replace("watsonx/", "")

        parameters = {
            GenParams.MAX_NEW_TOKENS: 8000,
            GenParams.TEMPERATURE: 0.1,
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.STOP_SEQUENCES: ["<|endoftext|>"],
        }

        credentials = Credentials(url=url, api_key=api_key)
        model = ModelInference(
            model_id=clean_model_id,
            params=parameters,
            credentials=credentials,
            project_id=project_id
        )

        _model_cache[model_id] = model
        logger.info(f"Created WatsonX model instance: {clean_model_id}")

        return model


def batch_generate(
    prompts: List[str],
    model_id: str,
    use_cache: bool = True,
    max_batch_size: int = 10
) -> List[str]:
    """
    Generate responses for multiple prompts using batch inference.

    Args:
        prompts: List of prompts to process
        model_id: WatsonX model ID (e.g., "meta-llama/llama-3-3-70b-instruct")
        use_cache: Whether to use response caching
        max_batch_size: Maximum number of prompts per batch call

    Returns:
        List of responses in the same order as prompts
    """
    if not prompts:
        return []

    results = [None] * len(prompts)
    uncached_indices = []
    uncached_prompts = []

    # Check cache first
    if use_cache:
        for i, prompt in enumerate(prompts):
            cached = get_cached_response(prompt, model_id)
            if cached:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_prompts.append(prompt)

        if uncached_indices:
            logger.info(f"Cache hit: {len(prompts) - len(uncached_indices)}/{len(prompts)}")
    else:
        uncached_indices = list(range(len(prompts)))
        uncached_prompts = prompts

    if not uncached_prompts:
        return results

    # Get model and generate
    model = get_watsonx_model(model_id)

    # Process in batches
    for batch_start in range(0, len(uncached_prompts), max_batch_size):
        batch_end = min(batch_start + max_batch_size, len(uncached_prompts))
        batch_prompts = uncached_prompts[batch_start:batch_end]
        batch_indices = uncached_indices[batch_start:batch_end]

        t0 = time.time()
        try:
            # WatsonX batch inference - pass list of prompts
            responses = model.generate_text(prompt=batch_prompts)

            # Handle response format
            if isinstance(responses, str):
                responses = [responses]
            elif not isinstance(responses, list):
                responses = list(responses)

            t1 = time.time()
            logger.info(f"Batch generated {len(batch_prompts)} responses in {t1-t0:.2f}s")

            # Store results and cache
            for idx, (orig_idx, response) in enumerate(zip(batch_indices, responses)):
                results[orig_idx] = response
                if use_cache:
                    set_cached_response(batch_prompts[idx], model_id, response)

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            # Fall back to sequential generation
            for idx, (orig_idx, prompt) in enumerate(zip(batch_indices, batch_prompts)):
                try:
                    response = model.generate_text(prompt=prompt)
                    if isinstance(response, list):
                        response = response[0]
                    results[orig_idx] = response
                    if use_cache:
                        set_cached_response(prompt, model_id, response)
                except Exception as e2:
                    logger.error(f"Sequential fallback failed: {e2}")
                    results[orig_idx] = f"Error: {e2}"

    return results


def concurrent_process(
    items: List[Any],
    process_fn: Callable[[Any], Any],
    max_workers: int = 4,
    desc: str = "Processing"
) -> List[Any]:
    """
    Process items concurrently using a thread pool.

    Args:
        items: List of items to process
        process_fn: Function to apply to each item
        max_workers: Maximum concurrent workers
        desc: Description for logging

    Returns:
        List of results in the same order as items
    """
    results = [None] * len(items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_fn, item): idx
            for idx, item in enumerate(items)
        }

        # Collect results
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"{desc} failed for item {idx}: {e}")
                results[idx] = {"error": str(e)}

            completed += 1
            if completed % 10 == 0:
                logger.info(f"{desc}: {completed}/{len(items)} complete")

    return results


class BatchProcessor:
    """
    Batch processor for Map-Reflect that collects prompts and processes them together.

    Usage:
        processor = BatchProcessor(model_id, batch_size=10)

        # Collect prompts
        for sample in samples:
            processor.add_prompt(sample.id, "extract_contexts", prompt)

        # Process batch
        responses = processor.process()

        # Get results
        for sample in samples:
            ctx_response = processor.get_result(sample.id, "extract_contexts")
    """

    def __init__(self, model_id: str, batch_size: int = 10, use_cache: bool = True):
        self.model_id = model_id
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.prompts: Dict[str, Dict[str, str]] = {}  # sample_id -> step -> prompt
        self.results: Dict[str, Dict[str, str]] = {}  # sample_id -> step -> response

    def add_prompt(self, sample_id: str, step: str, prompt: str):
        """Add a prompt for later batch processing."""
        if sample_id not in self.prompts:
            self.prompts[sample_id] = {}
        self.prompts[sample_id][step] = prompt

    def process(self) -> int:
        """Process all collected prompts in batches. Returns number processed."""
        # Flatten prompts
        all_prompts = []
        prompt_keys = []  # (sample_id, step) tuples

        for sample_id, steps in self.prompts.items():
            for step, prompt in steps.items():
                all_prompts.append(prompt)
                prompt_keys.append((sample_id, step))

        if not all_prompts:
            return 0

        # Batch generate
        responses = batch_generate(
            all_prompts,
            self.model_id,
            use_cache=self.use_cache,
            max_batch_size=self.batch_size
        )

        # Store results
        for (sample_id, step), response in zip(prompt_keys, responses):
            if sample_id not in self.results:
                self.results[sample_id] = {}
            self.results[sample_id][step] = response

        # Clear prompts
        self.prompts = {}

        return len(responses)

    def get_result(self, sample_id: str, step: str) -> Optional[str]:
        """Get a processed result."""
        return self.results.get(sample_id, {}).get(step)

    def clear(self):
        """Clear all prompts and results."""
        self.prompts = {}
        self.results = {}
