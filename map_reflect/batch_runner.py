#!/usr/bin/env python3
"""
Batch-enabled Map-Reflect Runner

Processes multiple samples in parallel using WatsonX batch inference.
Achieves 10-11x speedup over sequential processing.

Usage:
    python -m map_reflect.batch_runner \
        --datasets ITOpsGraph \
        --models watsonx/meta-llama/llama-3-3-70b-instruct \
        --property-lookup-file data/itopsgraph_api_lookup.json \
        --data-dir data \
        --batch-size 50
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
from tqdm import tqdm
import threading

# Suppress verbose logging
os.environ['LITELLM_LOG'] = 'ERROR'

from dotenv import load_dotenv

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


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 50
    max_workers: int = 4
    use_cache: bool = True
    cache_dir: Optional[str] = None


class WatsonXBatchClient:
    """WatsonX client optimized for batch inference."""

    def __init__(self, model_id: str, max_tokens: int = 8000):
        if not HAS_WATSONX:
            raise RuntimeError("ibm_watsonx_ai required for batch processing")

        self.model_id = model_id.replace("watsonx/", "")
        self.max_tokens = max_tokens
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

                    self._model = ModelInference(
                        model_id=self.model_id,
                        params={
                            GenParams.MAX_NEW_TOKENS: self.max_tokens,
                            GenParams.TEMPERATURE: 0.1,
                            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
                            GenParams.STOP_SEQUENCES: ["<|endoftext|>"],
                        },
                        credentials=Credentials(url=url, api_key=api_key),
                        project_id=project_id
                    )
                    logger.info(f"Initialized WatsonX model: {self.model_id}")

        return self._model

    def _cache_key(self, prompt: str) -> str:
        """Generate cache key for a prompt."""
        import hashlib
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
        if use_cache:
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
                    if use_cache:
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
                        if use_cache:
                            self._cache[self._cache_key(prompt)] = response
                    except Exception as e2:
                        logger.error(f"Fallback failed: {e2}")
                        results[orig_idx] = f"Error: {e2}"

        return results

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache)
        }


class BatchMapReflectRunner:
    """
    Batch-enabled Map-Reflect runner.

    Processes multiple samples in parallel by batching LLM calls.
    """

    def __init__(
        self,
        property_lookup_file: str,
        data_dir: str,
        model: str,
        dataset: str,
        batch_config: BatchConfig = None
    ):
        self.property_lookup_file = Path(property_lookup_file)
        self.data_dir = Path(data_dir)
        self.model = model
        self.dataset = dataset
        self.config = batch_config or BatchConfig()

        # Initialize WatsonX client
        self.client = WatsonXBatchClient(model)

        # Load property lookup
        with open(self.property_lookup_file) as f:
            self.property_lookup = json.load(f)

        # Results directory
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

        logger.info(f"BatchMapReflectRunner initialized")
        logger.info(f"  Model: {model}")
        logger.info(f"  Dataset: {dataset}")
        logger.info(f"  Batch size: {self.config.batch_size}")

    def _format_prompt(self, messages: List[Dict]) -> str:
        """Format messages into a single prompt string (simplified for batch)."""
        # For batch mode, use simple concatenation without role tags
        # This matches what works in direct testing
        parts = []
        for msg in messages:
            content = msg.get("content", "")
            parts.append(content)
        return "\n\n".join(parts)

    def _build_extract_context_prompt(self, utterance: str, source_hints: List[str], target_hints: List[str]) -> str:
        """Build prompt for context extraction."""
        prompt = f"""Given this schema mapping request: "{utterance}"

Available source hints: {source_hints}
Available target hints: {target_hints}

Extract the source and destination schema contexts. Return ONLY a JSON object:
{{"source_context": "[product] [type]", "destination_context": "[product] [type]", "source_product": "[product]", "target_product": "[product]", "source_types": ["type1"], "target_types": ["type1"]}}
"""
        messages = [
            {"role": "system", "content": "You are an expert in API schema mapping."},
            {"role": "user", "content": prompt}
        ]
        return self._format_prompt(messages)

    def _build_field_hints_prompt(self, utterance: str) -> str:
        """Build prompt for field hint extraction."""
        prompt = f"""Analyze this schema mapping request: "{utterance}"

Extract specific field names to map. Return ONLY JSON:
{{"source_field_hints": ["field1"], "target_field_hints": ["field1"]}}
"""
        messages = [
            {"role": "system", "content": "You are an expert in schema mapping."},
            {"role": "user", "content": prompt}
        ]
        return self._format_prompt(messages)

    def _build_mapping_prompt(
        self,
        source_props: List[Dict],
        target_props: List[Dict],
        source_context: str,
        target_context: str
    ) -> str:
        """Build prompt for finding mappings."""
        source_str = json.dumps(source_props[:20], indent=2)  # Limit for prompt size
        target_str = json.dumps(target_props[:20], indent=2)

        prompt = f"""Map fields between these schemas:

SOURCE ({source_context}):
{source_str}

TARGET ({target_context}):
{target_str}

Return JSON array of mappings:
[{{"source": "field_name", "target": "field_name", "confidence": 0.9, "reason": "..."}}]
"""
        messages = [
            {"role": "system", "content": "You are an expert schema mapper. Match semantically equivalent fields."},
            {"role": "user", "content": prompt}
        ]
        return self._format_prompt(messages)

    def process_batch(self, samples: List[Dict]) -> List[Dict]:
        """
        Process a batch of samples using batched LLM inference.

        Args:
            samples: List of sample dicts with utterance, ground_truth, etc.

        Returns:
            List of result dicts
        """
        n = len(samples)
        logger.info(f"Processing batch of {n} samples")

        # Stage 1: Extract contexts (batch)
        context_prompts = []
        for sample in samples:
            utterance = sample.get("utterance", "")
            source_hints = sample.get("source_hints", [])
            target_hints = sample.get("target_hints", [])
            context_prompts.append(self._build_extract_context_prompt(utterance, source_hints, target_hints))

        t0 = time.time()
        context_responses = self.client.generate_batch(
            context_prompts,
            use_cache=self.config.use_cache,
            batch_size=self.config.batch_size
        )
        logger.info(f"Stage 1 (contexts): {time.time()-t0:.2f}s for {n} samples")

        # Stage 2: Extract field hints (batch)
        field_prompts = [self._build_field_hints_prompt(s.get("utterance", "")) for s in samples]

        t0 = time.time()
        field_responses = self.client.generate_batch(
            field_prompts,
            use_cache=self.config.use_cache,
            batch_size=self.config.batch_size
        )
        logger.info(f"Stage 2 (fields): {time.time()-t0:.2f}s for {n} samples")

        # Stage 3: Find mappings (batch) - requires parsed context
        mapping_prompts = []
        for i, sample in enumerate(samples):
            # Parse context response
            ctx = self._parse_json_response(context_responses[i])
            source_ctx = ctx.get("source_context", sample.get("utterance", ""))
            target_ctx = ctx.get("destination_context", sample.get("utterance", ""))
            source_product = ctx.get("source_product", "")
            target_product = ctx.get("target_product", "")

            # Get properties for this sample
            source_props = self._get_properties_for_product(source_product)
            target_props = self._get_properties_for_product(target_product)

            mapping_prompts.append(self._build_mapping_prompt(
                source_props, target_props, source_ctx, target_ctx
            ))

        t0 = time.time()
        mapping_responses = self.client.generate_batch(
            mapping_prompts,
            use_cache=self.config.use_cache,
            batch_size=self.config.batch_size
        )
        logger.info(f"Stage 3 (mappings): {time.time()-t0:.2f}s for {n} samples")

        # Build results
        results = []
        for i, sample in enumerate(samples):
            mappings = self._parse_mapping_response(mapping_responses[i])

            result = {
                "sample_id": sample.get("sample_id", f"sample_{i}"),
                "utterance": sample.get("utterance", ""),
                "predicted_mapping": mappings,
                "ground_truth_mapping": sample.get("ground_truth_mapping", {}),
                "success": len(mappings) > 0,
                "context_response": context_responses[i][:200] if context_responses[i] else "",
            }

            # Calculate accuracy
            gt = sample.get("ground_truth_mapping", {})
            if gt:
                correct = sum(1 for s, t in mappings.items() if gt.get(s) == t)
                result["accuracy"] = correct / len(gt) if gt else 0.0
            else:
                result["accuracy"] = 0.0

            results.append(result)

        return results

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response."""
        if not response:
            return {}

        import re
        # Strip code fences
        response = re.sub(r"^```(?:json)?\s*", "", response.strip(), flags=re.IGNORECASE)
        response = re.sub(r"\s*```$", "", response.strip())

        # Find JSON object
        match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass

        try:
            return json.loads(response)
        except:
            return {}

    def _parse_mapping_response(self, response: str) -> Dict[str, str]:
        """Parse mapping response into dict."""
        if not response:
            return {}

        import re
        response = re.sub(r"^```(?:json)?\s*", "", response.strip(), flags=re.IGNORECASE)
        response = re.sub(r"\s*```$", "", response.strip())

        # Find JSON array
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            try:
                mappings_list = json.loads(match.group())
                return {m.get("source", ""): m.get("target", "") for m in mappings_list if m.get("source")}
            except:
                pass

        return {}

    def _get_properties_for_product(self, product: str) -> List[Dict]:
        """Get properties for a product from the lookup."""
        if not product:
            return []

        product_lower = product.lower().replace(" ", "_")
        properties = []

        for prop_name, prop_info in self.property_lookup.items():
            if product_lower in prop_name.lower():
                properties.append({
                    "name": prop_name,
                    "description": prop_info.get("description", ""),
                    "type": prop_info.get("type", "")
                })

        return properties[:50]  # Limit

    def run_evaluation(self, utterances: List[Dict]) -> Dict:
        """
        Run full evaluation with batch processing.

        Args:
            utterances: List of sample dicts

        Returns:
            Evaluation results dict
        """
        total_start = time.time()
        all_results = []

        # Process in batches
        n_batches = (len(utterances) + self.config.batch_size - 1) // self.config.batch_size

        with tqdm(total=len(utterances), desc=f"Batch processing {self.dataset}") as pbar:
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, len(utterances))
                batch = utterances[start_idx:end_idx]

                batch_results = self.process_batch(batch)
                all_results.extend(batch_results)

                pbar.update(len(batch))

                # Show stats
                successful = sum(1 for r in all_results if r["success"])
                pbar.set_postfix({
                    "success": f"{successful}/{len(all_results)}",
                    "batch": f"{batch_idx+1}/{n_batches}"
                })

        total_time = time.time() - total_start

        # Calculate metrics
        successful_results = [r for r in all_results if r["success"]]
        avg_accuracy = sum(r["accuracy"] for r in successful_results) / len(successful_results) if successful_results else 0.0

        cache_stats = self.client.get_cache_stats()

        return {
            "dataset": self.dataset,
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "batch_size": self.config.batch_size,
                "use_cache": self.config.use_cache,
            },
            "metrics": {
                "total_samples": len(utterances),
                "successful": len(successful_results),
                "avg_accuracy": avg_accuracy,
                "total_time_seconds": total_time,
                "samples_per_second": len(utterances) / total_time,
            },
            "cache_stats": cache_stats,
            "results": all_results
        }


def load_dataset(data_dir: Path, dataset: str) -> List[Dict]:
    """Load evaluation dataset."""
    dataset_lower = dataset.lower()

    # Try different file patterns
    patterns = [
        f"{dataset_lower}_dataset.jsonl",
        f"{dataset_lower}_eval_dataset.jsonl",
        f"{dataset}_dataset.jsonl",
    ]

    for pattern in patterns:
        filepath = data_dir / pattern
        if filepath.exists():
            samples = []
            with open(filepath) as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            logger.info(f"Loaded {len(samples)} samples from {filepath}")
            return samples

    raise FileNotFoundError(f"No dataset file found for {dataset} in {data_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch Map-Reflect Runner")
    parser.add_argument("--datasets", required=True, help="Comma-separated datasets")
    parser.add_argument("--models", required=True, help="Model ID")
    parser.add_argument("--property-lookup-file", required=True, help="Property lookup JSON")
    parser.add_argument("--data-dir", required=True, help="Data directory")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size (default: 50)")
    parser.add_argument("--no-cache", action="store_true", help="Disable response caching")
    parser.add_argument("--max-samples", type=int, help="Limit samples for testing")

    args = parser.parse_args()

    config = BatchConfig(
        batch_size=args.batch_size,
        use_cache=not args.no_cache,
    )

    for dataset in args.datasets.split(","):
        dataset = dataset.strip()
        logger.info(f"Processing dataset: {dataset}")

        # Load data
        utterances = load_dataset(Path(args.data_dir), dataset)
        if args.max_samples:
            utterances = utterances[:args.max_samples]

        # Initialize runner
        runner = BatchMapReflectRunner(
            property_lookup_file=args.property_lookup_file,
            data_dir=args.data_dir,
            model=args.models,
            dataset=dataset,
            batch_config=config
        )

        # Run evaluation
        results = runner.run_evaluation(utterances)

        # Save results
        output_file = runner.results_dir / f"batch_{dataset}_{args.models.replace('/', '_')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_file}")
        logger.info(f"Metrics: {results['metrics']}")


if __name__ == "__main__":
    main()
