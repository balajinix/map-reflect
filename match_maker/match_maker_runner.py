#!/usr/bin/env python3
"""
MatchMaker Runner - Evaluation framework for MatchMaker schema matching

Key fix over original:
- Agent is cached per (model, dataset) so it is created exactly once.
  Previously _get_agent_for_model() created a new MatchMakerAgent on every
  sample, meaning the self-improvement optimization ran N_samples x 20 times
  (thousands of wasted LLM calls) while producing zero usable in-context
  examples (source_properties was still empty at init for utterance datasets).

- For utterance-based datasets the optimization now runs exactly once, after
  the first set_source_target_from_utterance() call has populated
  source_properties.  This matches the paper's intent: optimize the program
  once on a representative set of queries, then apply the resulting in-context
  examples to all subsequent samples.
"""

import os
import sys
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Override sqlite3 with pysqlite3 on systems with old sqlite3 (< 3.35.0)
# Required by ChromaDB; must happen before any chromadb import
try:
    import match_maker.sqlite_override  # noqa: F401
except ImportError:
    pass

os.environ['LITELLM_LOG'] = 'ERROR'
os.environ['LITELLM_DROP_PARAMS'] = 'True'

from dotenv import load_dotenv

script_dir = Path(__file__).parent
env_file = script_dir / ".env"

if env_file.exists():
    load_dotenv(env_file)
    print(f"Loaded environment from {env_file}")
else:
    load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

for lib in ['litellm', 'LiteLLM', 'httpx', 'httpcore', 'urllib3', 'requests']:
    logging.getLogger(lib).setLevel(logging.ERROR)


class MatchMakerMappingAgent:
    """Wrapper for MatchMakerAgent to provide unified interface."""

    def __init__(
        self,
        property_lookup_file: str,
        data_dir: str,
        model: str,
        rebuild_index: bool,
        dataset: str,
        use_optimization: bool = True,
        top_k: int = 5,
        abstention_mode: str = 'lenient',
        min_confidence_threshold: float = 20.0
    ):
        self.model = model
        self.dataset = dataset
        self.property_lookup_file = Path(property_lookup_file)
        self.data_dir = Path(data_dir)
        self.rebuild_index = rebuild_index
        self.use_optimization = use_optimization
        self.top_k = top_k
        self.abstention_mode = abstention_mode
        self.min_confidence_threshold = min_confidence_threshold
        self.available = False

        # Flag: optimization runs once after first set_source_target call
        self._optimization_triggered = False

        try:
            self._setup_matchmaker_agent()
            self.available = True
            logger.info(f"MatchMaker agent initialized for dataset: {self.dataset}")
        except Exception as e:
            logger.error(f"Failed to initialize MatchMaker agent: {e}")
            self.available = False

    def _validate_model(self):
        """Validate that the LLM model is properly configured."""
        from litellm import completion
        test_messages = [{"role": "user", "content": "Reply with just the word 'ok'"}]
        try:
            response = completion(
                model=self.model,
                messages=test_messages,
                temperature=0.0,
                max_tokens=10
            )
            logger.info(f"Model validation successful: {self.model}")
            return True
        except Exception as e:
            raise RuntimeError(f"Model validation FAILED for {self.model}: {e}") from e

    def _setup_matchmaker_agent(self):
        """Set up MatchMaker agent with model validation."""
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))

        from match_maker_agent import MatchMakerAgent

        if not self.property_lookup_file.exists():
            raise FileNotFoundError(
                f"Property lookup file not found: {self.property_lookup_file}"
            )

        logger.info(f"Validating model configuration: {self.model}")
        self._validate_model()

        self.matchmaker_agent = MatchMakerAgent(
            property_lookup_file=str(self.property_lookup_file),
            model=self.model,
            dataset=self.dataset,
            chroma_dir=str(self.data_dir / f"chroma_db_{self.dataset.lower()}"),
            collection_name=None,
            top_k=self.top_k,
            rebuild_index=self.rebuild_index,
            use_optimization=self.use_optimization,
            abstention_mode=self.abstention_mode,
            min_confidence_threshold=self.min_confidence_threshold
        )

        logger.info(
            f"MatchMaker agent configured: model={self.model}, "
            f"dataset={self.dataset}, top_k={self.top_k}, "
            f"abstention_mode={self.abstention_mode}"
        )

    def _maybe_run_optimization(self):
        """
        Trigger optimization exactly once per agent lifetime, after source/target
        properties have been populated by set_source_target_from_utterance().

        For pre-defined datasets (mimic_omop) the agent's own __init__ already
        ran optimization, so _optimization_done will be True and this is a no-op.
        """
        if self._optimization_triggered:
            return
        self._optimization_triggered = True
        # Delegate to the agent — it guards against double-runs internally
        self.matchmaker_agent.run_optimization_once()

    def run_agent(
        self,
        utterance_or_source_attr: str,
        is_utterance: bool = False,
        ground_truth_fields: Optional[List[str]] = None,
        expected_source: Optional[str] = None,
        expected_target: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run MatchMaker on a source attribute or utterance."""
        if not self.available:
            return self._fallback_response(utterance_or_source_attr)

        try:
            if is_utterance:
                logger.info(f"Parsing utterance: {utterance_or_source_attr}")

                def extract_schema_name(hint):
                    if hint is None:
                        return None
                    if isinstance(hint, str):
                        return hint
                    if isinstance(hint, dict):
                        hint_id = hint.get('id', '')
                        return hint_id.split('_')[0] if '_' in hint_id else hint_id
                    return None

                source_hint = extract_schema_name(expected_source)
                target_hint = extract_schema_name(expected_target)

                self.matchmaker_agent.set_source_target_from_utterance(
                    utterance_or_source_attr,
                    source_hint=source_hint,
                    target_hint=target_hint
                )

                # Run optimization once now that source_properties is populated
                self._maybe_run_optimization()

                source_schema = ""
                target_schema = ""
                if self.matchmaker_agent.source_properties:
                    first_source = next(iter(self.matchmaker_agent.source_properties.values()), {})
                    all_products = set(
                        p.get('product_name', '')
                        for p in self.matchmaker_agent.property_data.values()
                    )
                    if len(all_products) <= 1:
                        source_schema = first_source.get('type_name', '')
                    else:
                        source_schema = first_source.get('product_name', '')
                if self.matchmaker_agent.target_properties:
                    first_target = next(iter(self.matchmaker_agent.target_properties.values()), {})
                    all_products = set(
                        p.get('product_name', '')
                        for p in self.matchmaker_agent.property_data.values()
                    )
                    if len(all_products) <= 1:
                        target_schema = first_target.get('type_name', '')
                    else:
                        target_schema = first_target.get('product_name', '')

                if ground_truth_fields:
                    source_attrs = []
                    for gt_field in ground_truth_fields:
                        gt_lower = gt_field.lower()
                        gt_parts = gt_lower.split('_', 1)
                        gt_table = gt_parts[0] if gt_parts else ''
                        gt_field_name = gt_parts[1] if len(gt_parts) > 1 else gt_lower

                        for source_id in self.matchmaker_agent.source_properties.keys():
                            source_lower = source_id.lower()
                            if source_lower == gt_lower or gt_lower in source_lower:
                                source_attrs.append(source_id)
                                break
                            if '.' in source_lower:
                                src_parts = source_lower.rsplit('.', 1)
                                src_table = src_parts[0].split('_')[-1]
                                src_field = src_parts[1]
                            else:
                                src_parts = source_lower.split('_')
                                if len(src_parts) >= 3:
                                    src_table = src_parts[-2]
                                    src_field = src_parts[-1]
                                else:
                                    continue
                            if gt_table == src_table and gt_field_name == src_field:
                                source_attrs.append(source_id)
                                break
                    logger.info(
                        f"Matching {len(source_attrs)} fields from ground truth "
                        f"(from {len(ground_truth_fields)} gt fields)"
                    )
                else:
                    source_attrs = list(self.matchmaker_agent.source_properties.keys())
                    logger.info(f"Matching all {len(source_attrs)} source attributes")

                if not source_attrs:
                    logger.warning("No source attributes found after parsing utterance")
                    return {
                        "utterance": utterance_or_source_attr,
                        "predicted_mapping": {},
                        "source_schema": source_schema,
                        "target_schema": target_schema,
                        "mapping_result": {},
                        "candidates": [],
                        "selected_match": None,
                        "abstain": True,
                        "error": "No source attributes identified"
                    }

                results = self.matchmaker_agent.run(source_attrs)

                predicted_mapping = {}
                for source_attr, result in results.items():
                    if result.selected_match and not result.abstain:
                        predicted_mapping[source_attr] = {
                            'target_property': result.selected_match,
                            'justification': (
                                f"Confidence: {result.candidates[0].confidence:.1f}"
                                if result.candidates else "MatchMaker match"
                            )
                        }

                mapping_result = self._convert_utterance_to_mapping_format(
                    results, utterance_or_source_attr
                )

                return {
                    "utterance": utterance_or_source_attr,
                    "predicted_mapping": predicted_mapping,
                    "source_schema": source_schema,
                    "target_schema": target_schema,
                    "mapping_result": mapping_result,
                    "all_results": {k: self._result_to_dict(v) for k, v in results.items()},
                    "total_mappings": len(predicted_mapping),
                    "error": None
                }

            else:
                source_attribute = utterance_or_source_attr
                logger.info(f"Running MatchMaker for: {source_attribute}")

                results = self.matchmaker_agent.run([source_attribute])

                if source_attribute not in results:
                    return {
                        "source_attribute": source_attribute,
                        "mapping_result": {},
                        "candidates": [],
                        "selected_match": None,
                        "abstain": True,
                        "error": "No result returned"
                    }

                result = results[source_attribute]
                candidates_list = [
                    {
                        "target_attribute": cand.target_attribute,
                        "confidence": cand.confidence,
                        "rank": cand.rank,
                        "justification": cand.justification,
                        "source": cand.source
                    }
                    for cand in result.candidates
                ]
                mapping_result = self._convert_to_mapping_format(result)

                return {
                    "source_attribute": source_attribute,
                    "mapping_result": mapping_result,
                    "candidates": candidates_list,
                    "selected_match": result.selected_match,
                    "abstain": result.abstain,
                    "reasoning_trace": result.reasoning_trace,
                    "error": None
                }

        except Exception as e:
            logger.error(f"MatchMaker failed: {e}")
            return {
                "input": utterance_or_source_attr,
                "mapping_result": {},
                "candidates": [],
                "selected_match": None,
                "abstain": True,
                "error": str(e)
            }

    def _fallback_response(self, source_attribute: str) -> Dict[str, Any]:
        return {
            "source_attribute": source_attribute,
            "mapping_result": {},
            "candidates": [],
            "selected_match": None,
            "abstain": True,
            "error": "MatchMaker agent not available"
        }

    def _convert_to_mapping_format(self, result) -> Dict[str, Any]:
        if result.abstain or not result.selected_match:
            return {"name": f"matchmaker_{result.source_attribute}", "properties": {}}
        properties = {
            result.source_attribute: [{
                "mapProperty": result.selected_match,
                "justification": (
                    f"Confidence: {result.candidates[0].confidence:.1f}"
                    if result.candidates else "Selected match"
                )
            }]
        }
        return {"name": f"matchmaker_{result.source_attribute}", "properties": properties}

    def _convert_utterance_to_mapping_format(self, results: Dict, utterance: str) -> Dict[str, Any]:
        properties = {}
        for source_attr, result in results.items():
            if not result.abstain and result.selected_match:
                properties[source_attr] = [{
                    "mapProperty": result.selected_match,
                    "justification": (
                        f"Confidence: {result.candidates[0].confidence:.1f}"
                        if result.candidates else "Selected match"
                    )
                }]
        return {
            "name": f"matchmaker_utterance_{len(properties)}_mappings",
            "properties": properties
        }

    def _result_to_dict(self, result) -> Dict[str, Any]:
        return {
            "source_attribute": result.source_attribute,
            "selected_match": result.selected_match,
            "abstain": result.abstain,
            "candidates": [
                {
                    "target_attribute": c.target_attribute,
                    "confidence": c.confidence,
                    "rank": c.rank,
                    "justification": c.justification,
                    "source": c.source
                }
                for c in result.candidates
            ],
            "reasoning_trace": result.reasoning_trace
        }


class MatchMakerEvaluationRunner:
    """Runner for MatchMaker evaluation across datasets."""

    def __init__(
        self,
        property_lookup_file: str,
        data_dir: str,
        schema_dir: Optional[str] = None,
        rebuild_index: bool = False,
        use_optimization: bool = True,
        top_k: int = 5,
        abstention_mode: str = 'lenient',
        min_confidence_threshold: float = 20.0,
        max_workers: int = 1
    ):
        self.property_lookup_file = Path(property_lookup_file)
        self.data_dir = Path(data_dir)
        self.schema_dir = Path(schema_dir) if schema_dir else None
        self.rebuild_index = rebuild_index
        self.use_optimization = use_optimization
        self.top_k = top_k
        self.abstention_mode = abstention_mode
        self.min_confidence_threshold = min_confidence_threshold
        self.max_workers = max_workers

        script_dir_local = Path(__file__).parent
        self.results_dir = script_dir_local / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Agent cache: keyed by (model, dataset)
        # One agent per combination; optimization runs at most once per agent.
        self._agent_cache: Dict[tuple, MatchMakerMappingAgent] = {}

        if not self.property_lookup_file.exists():
            raise FileNotFoundError(
                f"Property lookup file not found: {self.property_lookup_file}"
            )
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.dataset_loader = self._initialize_dataset_loader()
        logger.info("MatchMaker runner initialized")

    def _initialize_dataset_loader(self):
        if str(self.data_dir) not in sys.path:
            sys.path.insert(0, str(self.data_dir))
        try:
            from dataset_loader import DatasetLoader
            return DatasetLoader(str(self.data_dir))
        except ImportError as e:
            logger.error(f"Failed to import DatasetLoader: {e}")
            raise

    def _get_agent_for_model(
        self, model: str, dataset: str, worker_id: int = 0
    ) -> MatchMakerMappingAgent:
        """
        Return the cached agent for (model, dataset, worker_id), creating it if needed.

        Using a cache ensures:
        - The ColBERT/ChromaDB index is built once, not once per sample.
        - The self-improvement optimization runs once per dataset/model run,
          not once per sample.

        When max_workers > 1, each worker thread gets its own agent instance
        (keyed by worker_id) so that set_source_target_from_utterance() calls
        don't conflict across threads.
        """
        cache_key = (model, dataset, worker_id)
        if cache_key not in self._agent_cache:
            logger.info(
                f"Creating new MatchMaker agent for ({model}, {dataset}, worker={worker_id})"
            )
            self._agent_cache[cache_key] = MatchMakerMappingAgent(
                property_lookup_file=str(self.property_lookup_file),
                data_dir=str(self.data_dir),
                model=model,
                rebuild_index=self.rebuild_index,
                dataset=dataset,
                use_optimization=self.use_optimization,
                top_k=self.top_k,
                abstention_mode=self.abstention_mode,
                min_confidence_threshold=self.min_confidence_threshold
            )
        return self._agent_cache[cache_key]

    def load_evaluation_utterances(self, dataset: str) -> List[Dict[str, Any]]:
        return self.dataset_loader.load_evaluation_utterances(dataset)

    def calculate_accuracy_at_k(
        self,
        predicted: str,
        ground_truth: str,
        all_candidates: List[Dict],
        k: int
    ) -> float:
        if not all_candidates:
            return 0.0
        top_k_ids = [cand['target_attribute'] for cand in all_candidates[:k]]
        return 1.0 if ground_truth in top_k_ids else 0.0

    def run_single_evaluation(
        self,
        input_data: str,
        sample_id: str,
        model: str,
        dataset: str,
        ground_truth: Optional[str] = None,
        ground_truth_mapping: Optional[Dict] = None,
        expected_source: Optional[str] = None,
        expected_target: Optional[str] = None,
        is_utterance: bool = None,
        worker_id: int = 0
    ) -> Dict[str, Any]:
        """Run MatchMaker on a single input (attribute or utterance)."""
        start_time = time.time()

        if is_utterance is None:
            is_utterance = (
                dataset.lower() in ['itops', 'nl2gql', 'omop_cms', 'omop_mimic',
                                     'omop_synthea', 'toolbench', 'spider',
                                     'itom_hashicorp']
                or dataset.lower().startswith('itopsgraph')
            )

        try:
            logger.info(f"Running MatchMaker on sample {sample_id}")
            agent = self._get_agent_for_model(model, dataset, worker_id=worker_id)
            gt_fields = list(ground_truth_mapping.keys()) if ground_truth_mapping else None

            result = agent.run_agent(
                input_data,
                is_utterance=is_utterance,
                ground_truth_fields=gt_fields,
                expected_source=expected_source,
                expected_target=expected_target
            )

            execution_time = time.time() - start_time

            if is_utterance:
                return {
                    "sample_id": sample_id,
                    "utterance": input_data,
                    "predicted_mapping": result.get('predicted_mapping', {}),
                    "ground_truth_mapping": ground_truth_mapping or {},
                    "expected_source": (
                        {"id": expected_source}
                        if isinstance(expected_source, str) else expected_source
                    ),
                    "expected_target": (
                        {"id": expected_target}
                        if isinstance(expected_target, str) else expected_target
                    ),
                    "execution_time": execution_time,
                    "success": result.get('error') is None,
                    "error_message": result.get('error'),
                    "dataset": dataset,
                    "model": model,
                    "accuracy_metrics": {},
                    "abstain": not result.get('predicted_mapping'),
                    "total_mappings": len(result.get('predicted_mapping', {}))
                }
            else:
                accuracy_metrics = {}
                if ground_truth and result.get('candidates'):
                    for k in [1, 3, 5]:
                        acc_k = self.calculate_accuracy_at_k(
                            predicted=result.get('selected_match', ''),
                            ground_truth=ground_truth,
                            all_candidates=result['candidates'],
                            k=k
                        )
                        accuracy_metrics[f'accuracy@{k}'] = acc_k

                return {
                    "sample_id": sample_id,
                    "input": input_data,
                    "is_utterance": is_utterance,
                    "selected_match": result.get('selected_match'),
                    "abstain": result.get('abstain', False),
                    "candidates": result.get('candidates', []),
                    "mapping_result": result.get('mapping_result', {}),
                    "total_mappings": result.get(
                        'total_mappings', 0 if result.get('abstain') else 1
                    ),
                    "accuracy_metrics": accuracy_metrics,
                    "execution_time": execution_time,
                    "success": result.get('error') is None,
                    "error_message": result.get('error'),
                    "reasoning_trace": result.get('reasoning_trace', {}),
                    "dataset": dataset,
                    "model": model,
                    "ground_truth": ground_truth
                }

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"MatchMaker failed for {sample_id}: {e}")

            if is_utterance:
                return {
                    "sample_id": sample_id,
                    "utterance": input_data,
                    "predicted_mapping": {},
                    "ground_truth_mapping": ground_truth_mapping or {},
                    "expected_source": expected_source,
                    "expected_target": expected_target,
                    "execution_time": execution_time,
                    "success": False,
                    "error_message": str(e),
                    "dataset": dataset,
                    "model": model,
                    "accuracy_metrics": {},
                    "abstain": True,
                    "total_mappings": 0
                }
            else:
                return {
                    "sample_id": sample_id,
                    "input": input_data,
                    "is_utterance": is_utterance,
                    "selected_match": None,
                    "abstain": True,
                    "candidates": [],
                    "mapping_result": {},
                    "total_mappings": 0,
                    "accuracy_metrics": {},
                    "execution_time": execution_time,
                    "success": False,
                    "error_message": str(e),
                    "dataset": dataset,
                    "model": model,
                    "ground_truth": ground_truth
                }

    def _extract_sample_fields(self, item: Dict, is_utterance_dataset: bool):
        """Extract input fields from a sample item."""
        if is_utterance_dataset:
            return {
                'input_data': item.get('utterance', ''),
                'ground_truth': None,
                'ground_truth_mapping': item.get('ground_truth_mapping', {}),
                'expected_source': item.get('expected_source', '') or item.get('source_schema', ''),
                'expected_target': item.get('expected_target', '') or item.get('target_schema', ''),
            }
        else:
            return {
                'input_data': item.get('source_attribute', item.get('utterance', '')),
                'ground_truth': item.get('ground_truth_target', item.get('expected_target')),
                'ground_truth_mapping': None,
                'expected_source': None,
                'expected_target': None,
            }

    def _process_sample(
        self, item: Dict, model: str, dataset: str,
        is_utterance_dataset: bool, worker_id: int = 0
    ) -> Dict[str, Any]:
        """Process a single sample. Thread-safe: each worker_id uses its own agent."""
        sample_id = item['sample_id']
        fields = self._extract_sample_fields(item, is_utterance_dataset)

        try:
            result = self.run_single_evaluation(
                input_data=fields['input_data'],
                sample_id=sample_id,
                model=model,
                dataset=dataset,
                ground_truth=fields['ground_truth'],
                ground_truth_mapping=fields['ground_truth_mapping'],
                expected_source=fields['expected_source'],
                expected_target=fields['expected_target'],
                is_utterance=is_utterance_dataset,
                worker_id=worker_id
            )
            return result
        except Exception as e:
            logger.error(f"Unexpected error processing {sample_id}: {e}")
            return {
                "sample_id": sample_id,
                "success": False,
                "error_message": str(e),
                "abstain": True,
                "total_mappings": 0,
                "accuracy_metrics": {},
                "execution_time": 0,
                "dataset": dataset,
                "model": model,
            }

    def run_evaluation(self, dataset: str, model: str) -> Dict[str, Any]:
        """Run evaluation for dataset-model combination."""
        logger.info(f"Starting MatchMaker evaluation: {dataset} with {model}")

        utterances = self.load_evaluation_utterances(dataset)
        if not utterances:
            logger.error(f"No evaluation data for {dataset}")
            return {}

        logger.info(f"Loaded {len(utterances)} evaluation samples")

        is_utterance_dataset = (
            dataset.lower() in ['itops', 'nl2gql', 'omop_cms', 'omop_mimic',
                                  'omop_synthea', 'toolbench', 'spider',
                                  'itom_hashicorp']
            or dataset.lower().startswith('itopsgraph')
        )

        results = []
        total_start = time.time()
        MAX_CONSECUTIVE_FAILURES = 50

        from tqdm import tqdm

        if self.max_workers > 1:
            self._run_evaluation_parallel(
                utterances, dataset, model, is_utterance_dataset,
                results, MAX_CONSECUTIVE_FAILURES, tqdm
            )
        else:
            self._run_evaluation_sequential(
                utterances, dataset, model, is_utterance_dataset,
                results, MAX_CONSECUTIVE_FAILURES, tqdm
            )

        total_time = time.time() - total_start
        successful = [r for r in results if r["success"]]

        accuracy_at_k = {}
        for k in [1, 3, 5]:
            accs = [
                r['accuracy_metrics'].get(f'accuracy@{k}', 0.0)
                for r in results if r['accuracy_metrics']
            ]
            accuracy_at_k[f'accuracy@{k}'] = sum(accs) / len(accs) if accs else 0.0

        total_mappings = sum(r.get("total_mappings", 0) for r in results)
        total_abstentions = sum(1 for r in results if r["abstain"])

        logger.info(f"Evaluation complete for {dataset}")
        logger.info(f"  Total samples: {len(utterances)}")
        logger.info(f"  Successful: {len(successful)}")
        logger.info(f"  Accuracy@1: {accuracy_at_k.get('accuracy@1', 0):.3f}")

        return {
            "metadata": {
                "dataset": dataset,
                "method": "MatchMaker",
                "model": model,
                "is_utterance_dataset": is_utterance_dataset,
                "execution_time": total_time,
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(utterances),
                "successful_samples": len(successful),
                "total_mappings": total_mappings,
                "total_abstentions": total_abstentions,
                "accuracy_at_k": accuracy_at_k,
                "property_lookup_file": str(self.property_lookup_file),
                "rebuild_index": self.rebuild_index,
                "use_optimization": self.use_optimization,
                "top_k": self.top_k,
                "max_workers": self.max_workers
            },
            "results": results
        }

    def _run_evaluation_sequential(
        self, utterances, dataset, model, is_utterance_dataset,
        results, max_consecutive_failures, tqdm
    ):
        """Original sequential sample processing loop."""
        consecutive_failures = 0

        with tqdm(utterances, desc=f"Evaluating {dataset}", unit="sample") as pbar:
            for item in pbar:
                sample_id = item['sample_id']
                pbar.set_description(f"Processing {sample_id}")

                try:
                    result = self._process_sample(
                        item, model, dataset, is_utterance_dataset, worker_id=0
                    )
                    results.append(result)

                    if not result["success"]:
                        consecutive_failures += 1
                        error_msg = result.get("error_message", "Unknown error")
                        critical_patterns = [
                            "provider not provided", "provider not found",
                            "invalid api key", "authentication", "critical llm",
                        ]
                        if any(p in error_msg.lower() for p in critical_patterns):
                            raise RuntimeError(f"Critical error - stopping: {error_msg}")
                        if consecutive_failures >= max_consecutive_failures:
                            raise RuntimeError(
                                f"Stopping after {consecutive_failures} consecutive failures. "
                                f"Last error: {error_msg}"
                            )
                    else:
                        consecutive_failures = 0

                    status = "ok" if result["success"] else "FAIL"
                    pbar.set_postfix({
                        "status": status,
                        "mappings": result.get("total_mappings", 0),
                        "fails": consecutive_failures
                    })

                except RuntimeError:
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error processing {item['sample_id']}: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        raise RuntimeError(f"Too many failures: {e}") from e

                time.sleep(0.5)

    def _run_evaluation_parallel(
        self, utterances, dataset, model, is_utterance_dataset,
        results, max_consecutive_failures, tqdm
    ):
        """Parallel sample processing using ThreadPoolExecutor."""
        max_workers = self.max_workers
        print(f"Running parallel evaluation with {max_workers} workers")

        # Process first sample sequentially to trigger optimization
        first_item = utterances[0]
        print(f"Processing first sample sequentially (triggers optimization)...")
        first_result = self._process_sample(
            first_item, model, dataset, is_utterance_dataset, worker_id=0
        )
        results.append(first_result)

        if not first_result["success"]:
            error_msg = first_result.get("error_message", "Unknown error")
            critical_patterns = [
                "provider not provided", "provider not found",
                "invalid api key", "authentication", "critical llm",
            ]
            if any(p in error_msg.lower() for p in critical_patterns):
                raise RuntimeError(f"Critical error on first sample - stopping: {error_msg}")

        remaining_items = utterances[1:]
        if not remaining_items:
            return

        # Pre-create agent pool so all workers share the ChromaDB index build cost
        # Worker 0's agent already exists from the first sample above
        print(f"Creating agent pool ({max_workers} agents)...")
        for worker_id in range(1, max_workers):
            self._get_agent_for_model(model, dataset, worker_id=worker_id)
        print(f"Agent pool ready")

        # Thread-safe state for failure tracking and results
        lock = threading.Lock()
        consecutive_failures = 0
        stop_event = threading.Event()
        critical_error_msg = None

        # Map worker IDs to threads via thread-local storage
        worker_ids = list(range(max_workers))
        worker_id_queue = list(worker_ids)
        worker_id_lock = threading.Lock()
        thread_worker_map: Dict[int, int] = {}

        def get_worker_id():
            tid = threading.get_ident()
            if tid not in thread_worker_map:
                with worker_id_lock:
                    if tid not in thread_worker_map:
                        thread_worker_map[tid] = worker_id_queue.pop(0)
            return thread_worker_map[tid]

        def process_with_tracking(item):
            nonlocal consecutive_failures, critical_error_msg

            if stop_event.is_set():
                return None

            wid = get_worker_id()
            result = self._process_sample(
                item, model, dataset, is_utterance_dataset, worker_id=wid
            )

            with lock:
                if stop_event.is_set():
                    return None

                if not result["success"]:
                    consecutive_failures += 1
                    error_msg = result.get("error_message", "Unknown error")
                    critical_patterns = [
                        "provider not provided", "provider not found",
                        "invalid api key", "authentication", "critical llm",
                    ]
                    if any(p in error_msg.lower() for p in critical_patterns):
                        critical_error_msg = error_msg
                        stop_event.set()
                        return result
                    if consecutive_failures >= max_consecutive_failures:
                        critical_error_msg = (
                            f"Stopping after {consecutive_failures} consecutive failures. "
                            f"Last error: {error_msg}"
                        )
                        stop_event.set()
                        return result
                else:
                    consecutive_failures = 0

            return result

        completed = 0
        pbar = tqdm(
            total=len(remaining_items),
            desc=f"Evaluating {dataset} (parallel x{max_workers})",
            unit="sample",
            initial=0
        )

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_with_tracking, item): item
                    for item in remaining_items
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        completed += 1
                        pbar.update(1)

                        status = "ok" if result["success"] else "FAIL"
                        pbar.set_postfix({
                            "status": status,
                            "done": completed,
                            "fails": consecutive_failures
                        })

                    if stop_event.is_set():
                        # Cancel pending futures
                        for f in futures:
                            f.cancel()
                        break
        finally:
            pbar.close()

        if critical_error_msg:
            raise RuntimeError(f"Critical error - stopping: {critical_error_msg}")

    def save_results(self, results: Dict, filename: str):
        output_file = self.results_dir / filename
        try:
            with open(output_file, 'w') as f:
                json.dump(self._make_json_serializable(results), f, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def save_results_jsonl(self, results: Dict, filename: str):
        output_file = self.results_dir / filename
        try:
            result_list = results.get('results', [])
            with open(output_file, 'w') as f:
                for result in result_list:
                    f.write(json.dumps(self._make_json_serializable(result)) + '\n')
            logger.info(f"Results saved to {output_file} ({len(result_list)} samples)")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def _make_json_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)


def main():
    parser = argparse.ArgumentParser(
        description="MatchMaker Schema Matching Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--property-lookup-file", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--schema-dir", type=str, default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--no-optimization", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--abstention-mode", type=str, default="lenient",
        choices=["strict", "lenient", "disabled", "threshold"]
    )
    parser.add_argument("--min-confidence", type=float, default=20.0)
    parser.add_argument(
        "--max-workers", type=int, default=1,
        help="Number of parallel workers for sample processing (default: 1 = sequential)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        logger.setLevel(logging.INFO)

    datasets = [d.strip() for d in args.datasets.split(",")]
    models = [m.strip() for m in args.models.split(",")]

    try:
        runner = MatchMakerEvaluationRunner(
            property_lookup_file=args.property_lookup_file,
            data_dir=args.data_dir,
            schema_dir=args.schema_dir,
            rebuild_index=args.rebuild_index,
            use_optimization=not args.no_optimization,
            top_k=args.top_k,
            abstention_mode=args.abstention_mode,
            min_confidence_threshold=args.min_confidence,
            max_workers=args.max_workers
        )
    except Exception as e:
        logger.error(f"Failed to initialize runner: {e}")
        return 1

    all_results = {}
    total_start = time.time()

    for dataset in datasets:
        for model in models:
            key = f"{dataset}_{model.replace('/', '_')}"
            print(f"\n{'='*60}\nEvaluating: {dataset} with {model}\n{'='*60}\n")

            try:
                result = runner.run_evaluation(dataset, model)

                if result:
                    all_results[key] = result

                    model_lower = model.lower()
                    if 'granite' in model_lower:
                        model_short = 'granite'
                    elif 'llama' in model_lower:
                        model_short = 'llama'
                    elif 'mistral' in model_lower:
                        model_short = 'mistral'
                    else:
                        model_short = model.replace('/', '_').replace('watsonx_', '')

                    jsonl_filename = f"{dataset}_{model_short}.jsonl"
                    runner.save_results_jsonl(result, jsonl_filename)

                    json_filename = f"{dataset}_{model_short}_full.json"
                    runner.save_results(result, json_filename)

                    print(f"\nResults saved: {runner.results_dir / jsonl_filename}")
                    metadata = result.get('metadata', {})
                    print(f"  Samples:       {metadata.get('total_samples', 0)}")
                    print(f"  Successful:    {metadata.get('successful_samples', 0)}")
                    print(f"  Abstentions:   {metadata.get('total_abstentions', 0)}")
                    print(f"  Accuracy@1:    {metadata.get('accuracy_at_k', {}).get('accuracy@1', 0):.3f}")
                    print(f"  Accuracy@3:    {metadata.get('accuracy_at_k', {}).get('accuracy@3', 0):.3f}")
                    print(f"  Accuracy@5:    {metadata.get('accuracy_at_k', {}).get('accuracy@5', 0):.3f}")

            except Exception as e:
                logger.error(f"Evaluation failed for {dataset} with {model}: {e}")
                continue

            time.sleep(2)

    total_time = time.time() - total_start

    if all_results:
        print(f"\n{'='*60}\nMatchMaker Evaluation Complete!\n{'='*60}")
        print(f"  Total combinations: {len(all_results)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Results directory: {runner.results_dir}")
        return 0
    else:
        logger.error("No results generated")
        return 1


if __name__ == "__main__":
    sys.exit(main())
