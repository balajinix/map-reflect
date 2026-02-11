#!/usr/bin/env python3
"""
ReMatch Runner - Evaluation framework for ReMatch schema matching
Supports multiple datasets: itops, nl2gql, and MIMIC-OMOP
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

# Suppress LiteLLM logging
os.environ['LITELLM_LOG'] = 'ERROR'
os.environ['LITELLM_DROP_PARAMS'] = 'True'

from dotenv import load_dotenv

# Get script directory and load environment
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

# Disable verbose logging
for lib in ['litellm', 'LiteLLM', 'httpx', 'httpcore', 'urllib3', 'requests']:
    logging.getLogger(lib).setLevel(logging.ERROR)


class ReMatchMappingAgent:
    """Wrapper for ReMatchAgent to provide unified interface."""

    def __init__(
        self,
        property_lookup_file: str,
        data_dir: str,
        model: str,
        dataset: str,
        top_j: int = 2,
        top_k: int = 5,
        use_guidance: bool = False,
        guidance_mappings: Optional[Dict[str, tuple]] = None
    ):
        """Initialize ReMatch mapping agent."""
        self.model = model
        self.dataset = dataset
        self.property_lookup_file = Path(property_lookup_file)
        self.data_dir = Path(data_dir)
        self.top_j = top_j
        self.top_k = top_k
        self.use_guidance = use_guidance
        self.guidance_mappings = guidance_mappings or {}
        self.available = False

        try:
            self._setup_rematch_agent()
            self.available = True
            logger.info(f"ReMatch agent initialized for dataset: {self.dataset}")
        except Exception as e:
            logger.error(f"Failed to initialize ReMatch agent: {e}")
            self.available = False

    def _validate_model(self):
        """Validate that the LLM model is properly configured."""
        from litellm import completion

        test_messages = [
            {"role": "user", "content": "Reply with just the word 'ok'"}
        ]

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
            error_msg = f"Model validation FAILED for {self.model}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _setup_rematch_agent(self):
        """Set up ReMatch agent with model validation."""
        # Add current directory to path for import
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))

        from rematch_agent import ReMatchAgent

        if not self.property_lookup_file.exists():
            raise FileNotFoundError(
                f"Property lookup file not found: {self.property_lookup_file}"
            )

        # VALIDATE MODEL FIRST before doing expensive setup
        logger.info(f"Validating model configuration: {self.model}")
        self._validate_model()

        # Initialize ReMatch agent
        self.rematch_agent = ReMatchAgent(
            property_lookup_file=str(self.property_lookup_file),
            model=self.model,
            dataset=self.dataset,
            top_j=self.top_j,
            top_k=self.top_k,
            use_guidance=self.use_guidance,
            guidance_mappings=self.guidance_mappings
        )

        logger.info(
            f"ReMatch agent configured: model={self.model}, "
            f"dataset={self.dataset}, top_j={self.top_j}, top_k={self.top_k}"
        )

    def run_agent(self, source_attribute: str, source_table: str) -> Dict[str, Any]:
        """
        Run ReMatch on a single source attribute (attribute-based mode).

        Args:
            source_attribute: Source attribute ID to match
            source_table: Source table name

        Returns:
            Dict with match results
        """
        if not self.available:
            return self._fallback_response(source_attribute, source_table)

        try:
            logger.info(f"Running ReMatch for: {source_table}.{source_attribute}")

            # Run ReMatch
            result = self.rematch_agent.match_attribute(source_attribute, source_table)

            if not result:
                return {
                    "source_table": source_table,
                    "source_attribute": source_attribute,
                    "mapping_result": {},
                    "candidates": [],
                    "top_k_matches": [],
                    "error": "No result returned"
                }

            # Convert to evaluation format
            candidates_list = [
                {
                    "target_table": cand.target_table,
                    "target_attribute": cand.target_attribute,
                    "rank": cand.rank,
                    "justification": cand.justification
                }
                for cand in result.candidates
            ]

            # Format for evaluation
            mapping_result = self._convert_to_mapping_format(result)

            return {
                "source_table": source_table,
                "source_attribute": source_attribute,
                "mapping_result": mapping_result,
                "candidates": candidates_list,
                "top_k_matches": result.top_k_matches,
                "selected_match": result.top_k_matches[0] if result.top_k_matches else None,
                "error": None
            }

        except Exception as e:
            logger.error(f"ReMatch failed for {source_table}.{source_attribute}: {e}")
            return {
                "source_table": source_table,
                "source_attribute": source_attribute,
                "mapping_result": {},
                "candidates": [],
                "top_k_matches": [],
                "selected_match": None,
                "error": str(e)
            }

    def run_utterance(
        self,
        utterance: str,
        ground_truth_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run ReMatch on an utterance (utterance-based mode for itops/nl2gql).

        Args:
            utterance: Natural language mapping request
            ground_truth_fields: Optional list of source field IDs to match

        Returns:
            Dict with mapping results in evaluation format
        """
        if not self.available:
            return {
                "utterance": utterance,
                "predicted_mapping": {},
                "source_schema": "",
                "target_schema": "",
                "error": "ReMatch agent not available"
            }

        try:
            logger.info(f"Running ReMatch for utterance: {utterance[:100]}...")

            # Use the new run_utterance method from the agent
            result = self.rematch_agent.run_utterance(utterance, ground_truth_fields)

            # Convert to evaluation format expected by calc_accuracy.py
            predicted_mapping = {}
            # The agent returns 'predicted_mapping' (renamed from 'mappings')
            agent_mappings = result.get('predicted_mapping') or result.get('mappings') or {}
            for src_field, mapping_info in agent_mappings.items():
                if isinstance(mapping_info, dict):
                    predicted_mapping[src_field] = {
                        'target_property': mapping_info.get('target_property', ''),
                        'justification': mapping_info.get('justification', 'ReMatch match')
                    }
                else:
                    predicted_mapping[src_field] = {
                        'target_property': str(mapping_info),
                        'justification': 'ReMatch match'
                    }

            return {
                "utterance": utterance,
                "predicted_mapping": predicted_mapping,
                "source_schema": result.get('source_schema', ''),
                "target_schema": result.get('target_schema', ''),
                "total_mappings": len(predicted_mapping),
                "error": result.get('error')
            }

        except Exception as e:
            logger.error(f"ReMatch utterance processing failed: {e}")
            return {
                "utterance": utterance,
                "predicted_mapping": {},
                "source_schema": "",
                "target_schema": "",
                "error": str(e)
            }

    def _fallback_response(self, source_attribute: str, source_table: str) -> Dict[str, Any]:
        """Fallback when agent not available."""
        return {
            "source_table": source_table,
            "source_attribute": source_attribute,
            "mapping_result": {},
            "candidates": [],
            "top_k_matches": [],
            "selected_match": None,
            "error": "ReMatch agent not available"
        }

    def _convert_to_mapping_format(self, result) -> Dict[str, Any]:
        """Convert ReMatch result to evaluation mapping format."""
        if not result.top_k_matches:
            return {
                "name": f"rematch_{result.source_attribute}",
                "properties": {}
            }

        # Format as expected by evaluation framework
        properties = {
            result.source_attribute: [{
                "mapProperty": match,
                "justification": f"Rank {i+1} match"
            } for i, match in enumerate(result.top_k_matches)]
        }

        return {
            "name": f"rematch_{result.source_attribute}",
            "properties": properties
        }


class ReMatchEvaluationRunner:
    """Runner for ReMatch evaluation across datasets."""
    
    def __init__(
        self,
        property_lookup_file: str,
        data_dir: str,
        schema_dir: Optional[str] = None,
        top_j: int = 2,
        top_k: int = 5,
        use_guidance: bool = False,
        guidance_file: Optional[str] = None
    ):
        """
        Initialize ReMatch evaluation runner.
        
        Args:
            property_lookup_file: Path to property lookup JSON
            data_dir: Directory with shared data components
            schema_dir: Directory with GraphQL schemas (optional)
            top_j: Number of candidate target tables to retrieve
            top_k: Number of top matches to return
            use_guidance: Whether to use human guidance
            guidance_file: Path to guidance mappings JSON
        """
        self.property_lookup_file = Path(property_lookup_file)
        self.data_dir = Path(data_dir)
        self.schema_dir = Path(schema_dir) if schema_dir else None
        self.top_j = top_j
        self.top_k = top_k
        self.use_guidance = use_guidance
        self.guidance_mappings = {}
        # Save results relative to this script's directory
        script_dir = Path(__file__).parent
        self.results_dir = script_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load guidance mappings if provided
        if guidance_file and Path(guidance_file).exists():
            self._load_guidance_mappings(guidance_file)
        
        # Validate files
        if not self.property_lookup_file.exists():
            raise FileNotFoundError(
                f"Property lookup file not found: {self.property_lookup_file}"
            )
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Initialize dataset loader
        self.dataset_loader = self._initialize_dataset_loader()
        
        # Load property lookup to extract table information
        with open(self.property_lookup_file, 'r') as f:
            self.property_data = json.load(f)
        
        logger.info("ReMatch runner initialized")
        logger.info(f"  Property lookup: {self.property_lookup_file}")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Top-J: {self.top_j}")
        logger.info(f"  Top-K: {self.top_k}")
        logger.info(f"  Use guidance: {self.use_guidance}")
    
    def _load_guidance_mappings(self, guidance_file: str):
        """Load guidance mappings from JSON file."""
        try:
            with open(guidance_file, 'r') as f:
                data = json.load(f)
                # Expected format: {"source_table": ["target_table", "attribute_pair"]}
                self.guidance_mappings = {
                    k: tuple(v) for k, v in data.items()
                }
            logger.info(f"Loaded {len(self.guidance_mappings)} guidance mappings")
        except Exception as e:
            logger.warning(f"Failed to load guidance mappings: {e}")
    
    def _initialize_dataset_loader(self):
        """Initialize dataset loader."""
        if str(self.data_dir) not in sys.path:
            sys.path.insert(0, str(self.data_dir))
        
        try:
            from dataset_loader import DatasetLoader
            return DatasetLoader(str(self.data_dir))
        except ImportError as e:
            logger.error(f"Failed to import DatasetLoader: {e}")
            raise
    
    def _get_source_table_for_attribute(self, attr_id: str) -> str:
        """Extract source table name for attribute from property data."""
        attr_data = self.property_data.get(attr_id, {})
        return attr_data.get('type_name', 'unknown')
    
    def _get_agent_for_model(
        self,
        model: str,
        dataset: str
    ) -> ReMatchMappingAgent:
        """Get agent for specified model and dataset."""
        return ReMatchMappingAgent(
            property_lookup_file=str(self.property_lookup_file),
            data_dir=str(self.data_dir),
            model=model,
            dataset=dataset,
            top_j=self.top_j,
            top_k=self.top_k,
            use_guidance=self.use_guidance,
            guidance_mappings=self.guidance_mappings
        )
    
    def load_evaluation_utterances(self, dataset: str) -> List[Dict[str, Any]]:
        """Load evaluation utterances for dataset."""
        return self.dataset_loader.load_evaluation_utterances(dataset)
    
    def calculate_accuracy_at_k(
        self,
        ground_truth: str,
        all_candidates: List[str],
        k: int
    ) -> float:
        """
        Calculate accuracy@k for a prediction.
        
        Returns 1.0 if ground_truth is in top-k candidates, else 0.0.
        """
        if not all_candidates:
            return 0.0
        
        # Get top-k candidate IDs
        top_k_ids = all_candidates[:k]
        
        return 1.0 if ground_truth in top_k_ids else 0.0
    
    def run_single_evaluation(
        self,
        source_attribute: str,
        sample_id: str,
        model: str,
        dataset: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run ReMatch on a single source attribute."""
        start_time = time.time()
        
        try:
            # Get source table for the attribute
            source_table = self._get_source_table_for_attribute(source_attribute)
            
            logger.info(f"Running ReMatch on sample {sample_id}")
            logger.info(f"  Source: {source_table}.{source_attribute}")
            logger.info(f"  Dataset: {dataset}")
            logger.info(f"  Model: {model}")
            
            # Get agent
            agent = self._get_agent_for_model(model, dataset)
            
            # Run agent
            result = agent.run_agent(source_attribute, source_table)
            
            execution_time = time.time() - start_time
            
            # Calculate accuracy metrics
            accuracy_metrics = {}
            if ground_truth and result.get('top_k_matches'):
                for k in [1, 3, 5]:
                    acc_k = self.calculate_accuracy_at_k(
                        ground_truth=ground_truth,
                        all_candidates=result['top_k_matches'],
                        k=k
                    )
                    accuracy_metrics[f'accuracy@{k}'] = acc_k
            
            # Compile result
            return {
                "sample_id": sample_id,
                "source_table": source_table,
                "source_attribute": source_attribute,
                "selected_match": result.get('selected_match'),
                "top_k_matches": result.get('top_k_matches', []),
                "candidates": result.get('candidates', []),
                "accuracy_metrics": accuracy_metrics,
                "execution_time": execution_time,
                "success": result.get('error') is None,
                "error_message": result.get('error'),
                "dataset": dataset,
                "model": model,
                "ground_truth": ground_truth
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"ReMatch failed for {sample_id}: {e}")
            
            return {
                "sample_id": sample_id,
                "source_table": "unknown",
                "source_attribute": source_attribute,
                "selected_match": None,
                "top_k_matches": [],
                "candidates": [],
                "accuracy_metrics": {},
                "execution_time": execution_time,
                "success": False,
                "error_message": str(e),
                "dataset": dataset,
                "model": model,
                "ground_truth": ground_truth
            }
    
    def run_evaluation(self, dataset: str, model: str) -> Dict[str, Any]:
        """Run evaluation for dataset-model combination with utterance support."""
        logger.info(f"Starting ReMatch evaluation: {dataset} with {model}")

        # Load evaluation data
        utterances = self.load_evaluation_utterances(dataset)
        if not utterances:
            logger.error(f"No evaluation data for {dataset}")
            return {}

        logger.info(f"Loaded {len(utterances)} evaluation samples")

        # Determine if this is utterance-based or attribute-based
        is_utterance_dataset = dataset.lower() in ['itops', 'nl2gql'] or dataset.lower().startswith('itopsgraph')

        # Run ReMatch on each sample
        results = []
        total_start = time.time()

        # FAILURE TRACKING
        MAX_CONSECUTIVE_FAILURES = 3
        consecutive_failures = 0

        from tqdm import tqdm

        with tqdm(utterances, desc=f"Evaluating {dataset}", unit="sample") as pbar:
            for item in pbar:
                sample_id = item['sample_id']

                pbar.set_description(f"Processing {sample_id}")

                try:
                    if is_utterance_dataset:
                        # Utterance-based evaluation (itops/nl2gql)
                        result = self.run_utterance_evaluation(
                            item=item,
                            sample_id=sample_id,
                            model=model,
                            dataset=dataset
                        )
                    else:
                        # Attribute-based evaluation (OMOP)
                        source_attr = item.get('source_attribute', item.get('utterance', ''))
                        ground_truth = item.get('ground_truth_target', item.get('expected_target'))

                        result = self.run_single_evaluation(
                            source_attribute=source_attr,
                            sample_id=sample_id,
                            model=model,
                            dataset=dataset,
                            ground_truth=ground_truth
                        )

                    results.append(result)

                    # CHECK FOR CONSECUTIVE FAILURES
                    if not result.get("success", False):
                        consecutive_failures += 1
                        error_msg = result.get("error_message", "Unknown error")

                        # Check for critical errors
                        critical_patterns = [
                            "provider not provided",
                            "provider not found",
                            "invalid api key",
                            "authentication",
                            "critical llm",
                        ]
                        is_critical = any(
                            p in str(error_msg).lower()
                            for p in critical_patterns
                        )

                        if is_critical:
                            logger.error(f"CRITICAL ERROR detected: {error_msg}")
                            raise RuntimeError(f"Critical error - stopping: {error_msg}")

                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            logger.error(
                                f"Too many consecutive failures ({consecutive_failures}). "
                                f"Last error: {error_msg}"
                            )
                            raise RuntimeError(
                                f"Stopping after {consecutive_failures} consecutive failures. "
                                f"Last error: {error_msg}"
                            )
                    else:
                        consecutive_failures = 0  # Reset on success

                    # Update progress bar
                    status = "ok" if result.get("success", False) else "FAIL"
                    pred_mapping = result.get("predicted_mapping", {})
                    mappings = len(pred_mapping) if is_utterance_dataset else (1 if result.get("selected_match") else 0)
                    # DEBUG: log mapping info
                    if is_utterance_dataset and mappings == 0:
                        logger.warning(f"DEBUG: {sample_id} has 0 mappings. predicted_mapping={pred_mapping}, success={result.get('success')}, error={result.get('error_message')}")
                    pbar.set_postfix({
                        "status": status,
                        "mappings": mappings,
                        "fails": consecutive_failures
                    })

                except RuntimeError:
                    # Re-raise runtime errors (critical failures)
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error processing {sample_id}: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        raise RuntimeError(f"Too many failures: {e}") from e

                time.sleep(0.5)  # Rate limiting

        total_time = time.time() - total_start

        # Calculate aggregate metrics
        successful = [r for r in results if r.get("success", False)]

        # Calculate statistics
        if is_utterance_dataset:
            total_mappings = sum(len(r.get("predicted_mapping", {})) for r in results)
        else:
            total_mappings = sum(1 for r in results if r.get("selected_match"))

        logger.info(f"Evaluation complete for {dataset}")
        logger.info(f"  Total samples: {len(utterances)}")
        logger.info(f"  Successful: {len(successful)}")
        logger.info(f"  Total mappings: {total_mappings}")
        logger.info(f"  Total time: {total_time:.2f}s")

        return {
            "metadata": {
                "dataset": dataset,
                "method": "ReMatch",
                "model": model,
                "is_utterance_dataset": is_utterance_dataset,
                "execution_time": total_time,
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(utterances),
                "successful_samples": len(successful),
                "total_mappings": total_mappings,
                "property_lookup_file": str(self.property_lookup_file),
                "top_j": self.top_j,
                "top_k": self.top_k,
                "use_guidance": self.use_guidance
            },
            "results": results
        }

    def run_utterance_evaluation(
        self,
        item: Dict[str, Any],
        sample_id: str,
        model: str,
        dataset: str
    ) -> Dict[str, Any]:
        """Run ReMatch evaluation on an utterance (for itops/nl2gql)."""
        start_time = time.time()

        try:
            utterance = item.get('utterance', '')
            expected_source = item.get('expected_source', '')
            expected_target = item.get('expected_target', '')
            ground_truth_mapping = item.get('ground_truth_mapping', {})

            logger.info(f"Running ReMatch utterance evaluation for sample {sample_id}")

            # Get agent
            agent = self._get_agent_for_model(model, dataset)

            # Extract ground truth fields for efficient matching
            ground_truth_fields = list(ground_truth_mapping.keys()) if ground_truth_mapping else None

            # Run utterance-based matching (only on ground truth fields for efficiency)
            result = agent.run_utterance(utterance, ground_truth_fields)

            execution_time = time.time() - start_time

            # Build result in calc_accuracy.py expected format
            return {
                "sample_id": sample_id,
                "utterance": utterance,
                "predicted_mapping": result.get('predicted_mapping', {}),
                "ground_truth_mapping": ground_truth_mapping,
                "expected_source": {"id": expected_source} if isinstance(expected_source, str) else expected_source,
                "expected_target": {"id": expected_target} if isinstance(expected_target, str) else expected_target,
                "execution_time": execution_time,
                "success": result.get('error') is None,
                "error_message": result.get('error'),
                "dataset": dataset,
                "model": model
            }

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"ReMatch utterance evaluation failed for {sample_id}: {e}")

            return {
                "sample_id": sample_id,
                "utterance": item.get('utterance', ''),
                "predicted_mapping": {},
                "ground_truth_mapping": item.get('ground_truth_mapping', {}),
                "expected_source": item.get('expected_source', ''),
                "expected_target": item.get('expected_target', ''),
                "execution_time": execution_time,
                "success": False,
                "error_message": str(e),
                "dataset": dataset,
                "model": model
            }
    
    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file."""
        output_file = self.results_dir / filename

        try:
            # Make serializable
            serializable = self._make_json_serializable(results)

            with open(output_file, 'w') as f:
                json.dump(serializable, f, indent=2)

            logger.info(f"Results saved to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def save_results_jsonl(self, results: Dict, filename: str):
        """Save results in JSONL format (one result per line) for calc_accuracy.py."""
        output_file = self.results_dir / filename

        try:
            result_list = results.get('results', [])

            with open(output_file, 'w') as f:
                for result in result_list:
                    serializable = self._make_json_serializable(result)
                    f.write(json.dumps(serializable) + '\n')

            logger.info(f"Results saved to {output_file} ({len(result_list)} samples)")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ReMatch Schema Matching Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python rematch_runner.py \\
    --property-lookup-file data/properties.json \\
    --data-dir data \\
    --datasets mimic_omop \\
    --models watsonx/mistralai/mistral-large

  # With guidance (relevance feedback)
  python rematch_runner.py \\
    --property-lookup-file data/properties.json \\
    --data-dir data \\
    --datasets mimic_omop \\
    --models watsonx/mistralai/mistral-large \\
    --use-guidance \\
    --guidance-file data/guidance.json

  # Custom retrieval parameters
  python rematch_runner.py \\
    --property-lookup-file data/properties.json \\
    --data-dir data \\
    --datasets mimic_omop,synthea \\
    --models watsonx/mistralai/mistral-large,claude-3-opus \\
    --top-j 3 \\
    --top-k 7
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--property-lookup-file",
        type=str,
        required=True,
        help="Path to property lookup JSON file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with shared data components"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated list of datasets (e.g., 'itops,nl2gql,mimic_omop')"
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of models (e.g., 'watsonx/mistralai/mistral-large,claude-3-opus')"
    )
    
    # Optional arguments
    parser.add_argument(
        "--schema-dir",
        type=str,
        default=None,
        help="Directory with GraphQL schemas"
    )
    parser.add_argument(
        "--top-j",
        type=int,
        default=2,
        help="Number of candidate target tables to retrieve (default: 2)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top matches to return (default: 5)"
    )
    parser.add_argument(
        "--use-guidance",
        action="store_true",
        help="Enable human guidance (relevance feedback)"
    )
    parser.add_argument(
        "--guidance-file",
        type=str,
        default=None,
        help="Path to guidance mappings JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
    
    # Parse datasets and models
    datasets = [d.strip() for d in args.datasets.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    
    logger.info("Starting ReMatch evaluation")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Models: {models}")
    logger.info(f"  Top-J (tables): {args.top_j}")
    logger.info(f"  Top-K (matches): {args.top_k}")
    logger.info(f"  Use guidance: {args.use_guidance}")
    
    # Initialize runner
    try:
        runner = ReMatchEvaluationRunner(
            property_lookup_file=args.property_lookup_file,
            data_dir=args.data_dir,
            schema_dir=args.schema_dir,
            top_j=args.top_j,
            top_k=args.top_k,
            use_guidance=args.use_guidance,
            guidance_file=args.guidance_file
        )
    except Exception as e:
        logger.error(f"Failed to initialize runner: {e}")
        return 1
    
    # Run evaluations
    all_results = {}
    total_start = time.time()
    
    for dataset in datasets:
        for model in models:
            key = f"{dataset}_{model.replace('/', '_')}"
            
            print(f"\n{'='*60}")
            print(f"Evaluating: {dataset} with {model}")
            print(f"{'='*60}\n")
            
            try:
                result = runner.run_evaluation(dataset, model)

                if result:
                    all_results[key] = result

                    # Determine model short name for filename
                    model_lower = model.lower()
                    if 'granite' in model_lower:
                        model_short = 'granite'
                    elif 'llama' in model_lower:
                        model_short = 'llama'
                    elif 'mistral' in model_lower:
                        model_short = 'mistral'
                    else:
                        model_short = model.replace('/', '_').replace('watsonx_', '')

                    # Save in JSONL format for calc_accuracy.py
                    guidance_suffix = "_guided" if args.use_guidance else ""
                    jsonl_filename = f"{dataset}_{model_short}{guidance_suffix}.jsonl"
                    runner.save_results_jsonl(result, jsonl_filename)

                    # Also save full results as JSON
                    json_filename = f"{dataset}_{model_short}{guidance_suffix}_full.json"
                    runner.save_results(result, json_filename)

                    print(f"\nResults saved: {runner.results_dir / jsonl_filename}")
                    
                    # Log summary
                    metadata = result.get('metadata', {})
                    print(f"  Samples: {metadata.get('total_samples', 0)}")
                    print(f"  Successful: {metadata.get('successful_samples', 0)}")
                    print(f"  Matches: {metadata.get('total_matches', 0)}")
                    print(f"  Avg candidates: {metadata.get('avg_candidates_per_query', 0):.1f}")
                    print(f"  Accuracy@1: {metadata.get('accuracy_at_k', {}).get('accuracy@1', 0):.3f}")
                    print(f"  Accuracy@3: {metadata.get('accuracy_at_k', {}).get('accuracy@3', 0):.3f}")
                    print(f"  Accuracy@5: {metadata.get('accuracy_at_k', {}).get('accuracy@5', 0):.3f}")
                
            except Exception as e:
                logger.error(f"Evaluation failed for {dataset} with {model}: {e}")
                continue
            
            time.sleep(2)  # Delay between evaluations
    
    total_time = time.time() - total_start
    
    if all_results:
        print(f"\n{'='*60}")
        print("ReMatch Evaluation Complete!")
        print(f"{'='*60}")
        print(f"  Total combinations: {len(all_results)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Results directory: {runner.results_dir}")
        
        # Print summary comparison
        print(f"\n{'='*60}")
        print("Summary of Results")
        print(f"{'='*60}")
        for key, result in all_results.items():
            metadata = result.get('metadata', {})
            acc1 = metadata.get('accuracy_at_k', {}).get('accuracy@1', 0)
            acc5 = metadata.get('accuracy_at_k', {}).get('accuracy@5', 0)
            print(f"  {key}:")
            print(f"    Acc@1: {acc1:.3f} | Acc@5: {acc5:.3f}")
        
        return 0
    else:
        logger.error("No results generated")
        return 1


if __name__ == "__main__":
    sys.exit(main())