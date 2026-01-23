#!/usr/bin/env python3
"""
MatchMaker Runner - Evaluation framework for MatchMaker schema matching
Supports multiple datasets: itops, nl2gql, and MIMIC-OMOP
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
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
        top_k: int = 5
    ):
        """Initialize MatchMaker mapping agent."""
        self.model = model
        self.dataset = dataset
        self.property_lookup_file = Path(property_lookup_file)
        self.data_dir = Path(data_dir)
        self.rebuild_index = rebuild_index
        self.use_optimization = use_optimization
        self.top_k = top_k
        self.available = False
        
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
    def _setup_matchmaker_agent(self):
        """Set up MatchMaker agent with model validation."""
        # Add current directory to path for import
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        from match_maker_agent import MatchMakerAgent
        
        if not self.property_lookup_file.exists():
            raise FileNotFoundError(
                f"Property lookup file not found: {self.property_lookup_file}"
            )
        
        # VALIDATE MODEL FIRST before doing expensive setup
        logger.info(f"Validating model configuration: {self.model}")
        self._validate_model()
        
        # Initialize MatchMaker agent
        self.matchmaker_agent = MatchMakerAgent(
            property_lookup_file=str(self.property_lookup_file),
            model=self.model,
            dataset=self.dataset,
            chroma_dir=str(self.data_dir / "chroma_db"),
            collection_name=None,  # Auto-generate
            top_k=self.top_k,
            rebuild_index=self.rebuild_index,
            use_optimization=self.use_optimization
        )
        
        logger.info(
            f"MatchMaker agent configured: model={self.model}, "
            f"dataset={self.dataset}, top_k={self.top_k}"
        )
    
    def run_agent(
        self,
        utterance_or_source_attr: str,
        is_utterance: bool = False,
        ground_truth_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run MatchMaker on a source attribute or utterance.

        Args:
            utterance_or_source_attr: Either a source attribute ID (MIMIC-OMOP)
                                      or a full utterance (itops/nl2gql)
            is_utterance: Whether input is a full utterance needing parsing
            ground_truth_fields: For utterance mode, list of source field IDs to match
                                (from ground_truth_mapping keys)

        Returns:
            Dict with match results
        """
        if not self.available:
            return self._fallback_response(utterance_or_source_attr)
        
        try:
            # For itops/nl2gql: parse utterance to set source/target dynamically
            if is_utterance:
                logger.info(f"Parsing utterance: {utterance_or_source_attr}")
                self.matchmaker_agent.set_source_target_from_utterance(utterance_or_source_attr)

                # Get the parsed source and target schemas
                # For NL2GQL (single product), use type_name; for ITOps (multiple products), use product_name
                source_schema = ""
                target_schema = ""
                if self.matchmaker_agent.source_properties:
                    first_source = next(iter(self.matchmaker_agent.source_properties.values()), {})
                    # Check if it's a single-product dataset by seeing if all have same product_name
                    all_products = set(p.get('product_name', '') for p in self.matchmaker_agent.property_data.values())
                    if len(all_products) <= 1:
                        # Single product (like nl2gql) - use type_name
                        source_schema = first_source.get('type_name', '')
                    else:
                        # Multiple products (like itops) - use product_name
                        source_schema = first_source.get('product_name', '')
                if self.matchmaker_agent.target_properties:
                    first_target = next(iter(self.matchmaker_agent.target_properties.values()), {})
                    all_products = set(p.get('product_name', '') for p in self.matchmaker_agent.property_data.values())
                    if len(all_products) <= 1:
                        target_schema = first_target.get('type_name', '')
                    else:
                        target_schema = first_target.get('product_name', '')

                # Determine which source attributes to match
                if ground_truth_fields:
                    # Only match fields that are in the ground truth (efficient)
                    source_attrs = []
                    for gt_field in ground_truth_fields:
                        gt_lower = gt_field.lower()
                        for source_id in self.matchmaker_agent.source_properties.keys():
                            if source_id.lower() == gt_lower or gt_lower in source_id.lower():
                                source_attrs.append(source_id)
                                break
                    logger.info(f"Matching {len(source_attrs)} fields from ground truth")
                else:
                    # Match all source attributes (slow)
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

                # For utterance-based matching, match the selected source attrs to target
                results = self.matchmaker_agent.run(source_attrs)

                # Build predicted_mapping in the format expected by calc_accuracy.py
                predicted_mapping = {}
                all_candidates = []
                for source_attr, result in results.items():
                    if result.selected_match and not result.abstain:
                        predicted_mapping[source_attr] = {
                            'target_property': result.selected_match,
                            'justification': f"Confidence: {result.candidates[0].confidence:.1f}" if result.candidates else "MatchMaker match"
                        }
                        all_candidates.extend(result.candidates)

                mapping_result = self._convert_utterance_to_mapping_format(results, utterance_or_source_attr)

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
            
            # For MIMIC-OMOP: direct attribute matching
            else:
                source_attribute = utterance_or_source_attr
                logger.info(f"Running MatchMaker for: {source_attribute}")
                
                # Run MatchMaker
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
                
                # Convert to evaluation format
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
                
                # Format for evaluation
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
        """Fallback when agent not available."""
        return {
            "source_attribute": source_attribute,
            "mapping_result": {},
            "candidates": [],
            "selected_match": None,
            "abstain": True,
            "error": "MatchMaker agent not available"
        }
    
    def _convert_to_mapping_format(self, result) -> Dict[str, Any]:
        """Convert MatchMaker result to evaluation mapping format."""
        if result.abstain or not result.selected_match:
            return {
                "name": f"matchmaker_{result.source_attribute}",
                "properties": {}
            }
        
        # Format as expected by evaluation framework
        properties = {
            result.source_attribute: [{
                "mapProperty": result.selected_match,
                "justification": f"Confidence: {result.candidates[0].confidence:.1f}"
                if result.candidates else "Selected match"
            }]
        }
        
        return {
            "name": f"matchmaker_{result.source_attribute}",
            "properties": properties
        }
    
    def _convert_utterance_to_mapping_format(self, results: Dict, utterance: str) -> Dict[str, Any]:
        """Convert multiple MatchMaker results from utterance to mapping format."""
        properties = {}
        
        for source_attr, result in results.items():
            if not result.abstain and result.selected_match:
                properties[source_attr] = [{
                    "mapProperty": result.selected_match,
                    "justification": f"Confidence: {result.candidates[0].confidence:.1f}"
                    if result.candidates else "Selected match"
                }]
        
        return {
            "name": f"matchmaker_utterance_{len(properties)}_mappings",
            "properties": properties
        }
    
    def _result_to_dict(self, result) -> Dict[str, Any]:
        """Convert MatchResult object to dictionary."""
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
        top_k: int = 5
    ):
        """
        Initialize MatchMaker evaluation runner.
        
        Args:
            property_lookup_file: Path to property lookup JSON
            data_dir: Directory with shared data components
            schema_dir: Directory with GraphQL schemas (optional)
            rebuild_index: Whether to rebuild vector index
            use_optimization: Whether to use self-improvement
            top_k: Number of candidates to retrieve
        """
        self.property_lookup_file = Path(property_lookup_file)
        self.data_dir = Path(data_dir)
        self.schema_dir = Path(schema_dir) if schema_dir else None
        self.rebuild_index = rebuild_index
        self.use_optimization = use_optimization
        self.top_k = top_k
        # Save results relative to this script's directory
        script_dir = Path(__file__).parent
        self.results_dir = script_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Cache for agents (reuse across samples)
        self._agent_cache = {}
        
        # Validate files
        if not self.property_lookup_file.exists():
            raise FileNotFoundError(
                f"Property lookup file not found: {self.property_lookup_file}"
            )
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Initialize dataset loader
        self.dataset_loader = self._initialize_dataset_loader()
        
        logger.info("MatchMaker runner initialized")
        logger.info(f"  Property lookup: {self.property_lookup_file}")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Rebuild index: {self.rebuild_index}")
        logger.info(f"  Use optimization: {self.use_optimization}")
        logger.info(f"  Top-k: {self.top_k}")
    
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
    
    def _get_agent_for_model(
        self,
        model: str,
        dataset: str
    ) -> MatchMakerMappingAgent:
        """Get agent for specified model and dataset."""
        return MatchMakerMappingAgent(
            property_lookup_file=str(self.property_lookup_file),
            data_dir=str(self.data_dir),
            model=model,
            rebuild_index=self.rebuild_index,
            dataset=dataset,
            use_optimization=self.use_optimization,
            top_k=self.top_k
        )
    
    def load_evaluation_utterances(self, dataset: str) -> List[Dict[str, Any]]:
        """Load evaluation utterances for dataset."""
        return self.dataset_loader.load_evaluation_utterances(dataset)
    
    def calculate_accuracy_at_k(
        self,
        predicted: str,
        ground_truth: str,
        all_candidates: List[Dict],
        k: int
    ) -> float:
        """
        Calculate accuracy@k for a prediction.
        
        Returns 1.0 if ground_truth is in top-k candidates, else 0.0.
        """
        if not all_candidates:
            return 0.0
        
        # Get top-k candidate IDs
        top_k_ids = [
            cand['target_attribute'] 
            for cand in all_candidates[:k]
        ]
        
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
        is_utterance: bool = None
    ) -> Dict[str, Any]:
        """Run MatchMaker on a single input (attribute or utterance)."""
        start_time = time.time()

        # Auto-detect if utterance-based (itops/nl2gql) or attribute-based (MIMIC-OMOP)
        if is_utterance is None:
            is_utterance = dataset.lower() in ['itops', 'nl2gql']

        try:
            logger.info(f"Running MatchMaker on sample {sample_id}")
            logger.info(f"  Input: {input_data[:100]}...")
            logger.info(f"  Dataset: {dataset}")
            logger.info(f"  Is utterance: {is_utterance}")
            logger.info(f"  Model: {model}")

            # Get agent
            agent = self._get_agent_for_model(model, dataset)

            # Extract ground truth fields for efficient matching
            gt_fields = list(ground_truth_mapping.keys()) if ground_truth_mapping else None

            # Run agent
            result = agent.run_agent(input_data, is_utterance=is_utterance, ground_truth_fields=gt_fields)

            execution_time = time.time() - start_time

            # Build result in calc_accuracy.py expected format
            if is_utterance:
                # Utterance-based evaluation format
                return {
                    "sample_id": sample_id,
                    "utterance": input_data,
                    "predicted_mapping": result.get('predicted_mapping', {}),
                    "ground_truth_mapping": ground_truth_mapping or {},
                    "expected_source": {"id": expected_source} if isinstance(expected_source, str) else expected_source,
                    "expected_target": {"id": expected_target} if isinstance(expected_target, str) else expected_target,
                    "execution_time": execution_time,
                    "success": result.get('error') is None,
                    "error_message": result.get('error'),
                    "dataset": dataset,
                    "model": model
                }
            else:
                # Attribute-based evaluation format
                # Calculate accuracy metrics
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
                    "total_mappings": result.get('total_mappings', 0 if result.get('abstain') else 1),
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
                    "model": model
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
    
    def run_evaluation(self, dataset: str, model: str) -> Dict[str, Any]:
        """Run evaluation for dataset-model combination with failure tracking."""
        logger.info(f"Starting MatchMaker evaluation: {dataset} with {model}")
        
        # Load evaluation data
        utterances = self.load_evaluation_utterances(dataset)
        if not utterances:
            logger.error(f"No evaluation data for {dataset}")
            return {}
        
        logger.info(f"Loaded {len(utterances)} evaluation samples")
        
        # Determine if this is utterance-based or attribute-based
        is_utterance_dataset = dataset.lower() in ['itops', 'nl2gql']
        
        # Run MatchMaker on each sample
        results = []
        total_start = time.time()
        
        # FAILURE TRACKING
        MAX_CONSECUTIVE_FAILURES = 3
        consecutive_failures = 0
        
        from tqdm import tqdm
        
        with tqdm(utterances, desc=f"Evaluating {dataset}", unit="sample") as pbar:
            for item in pbar:
                sample_id = item['sample_id']

                # For utterance-based: use full utterance
                # For attribute-based: use source_attribute
                if is_utterance_dataset:
                    input_data = item.get('utterance', '')
                    ground_truth = None
                    ground_truth_mapping = item.get('ground_truth_mapping', {})
                    expected_source = item.get('expected_source', '')
                    expected_target = item.get('expected_target', '')
                else:
                    input_data = item.get('source_attribute', item.get('utterance', ''))
                    ground_truth = item.get('ground_truth_target', item.get('expected_target'))
                    ground_truth_mapping = None
                    expected_source = None
                    expected_target = None

                pbar.set_description(f"Processing {sample_id}")

                try:
                    result = self.run_single_evaluation(
                        input_data=input_data,
                        sample_id=sample_id,
                        model=model,
                        dataset=dataset,
                        ground_truth=ground_truth,
                        ground_truth_mapping=ground_truth_mapping,
                        expected_source=expected_source,
                        expected_target=expected_target,
                        is_utterance=is_utterance_dataset
                    )
                    
                    results.append(result)
                    
                    # CHECK FOR CONSECUTIVE FAILURES
                    if not result["success"]:
                        consecutive_failures += 1
                        error_msg = result.get("error_message", "Unknown error")
                        
                        # Check for critical errors in the error message
                        critical_patterns = [
                            "provider not provided",
                            "provider not found",
                            "invalid api key",
                            "authentication",
                            "critical llm",
                        ]
                        is_critical = any(
                            p in error_msg.lower() 
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
                    status = "ok" if result["success"] else "FAIL"
                    matches = result.get("total_mappings", 0)
                    pbar.set_postfix({
                        "status": status,
                        "mappings": matches,
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
        successful = [r for r in results if r["success"]]
        
        # Accuracy@k metrics
        accuracy_at_k = {}
        for k in [1, 3, 5]:
            accs = [
                r['accuracy_metrics'].get(f'accuracy@{k}', 0.0)
                for r in results
                if r['accuracy_metrics']
            ]
            accuracy_at_k[f'accuracy@{k}'] = (
                sum(accs) / len(accs) if accs else 0.0
            )
        
        # Other statistics
        total_mappings = sum(r.get("total_mappings", 0) for r in results)
        total_abstentions = sum(1 for r in results if r["abstain"])
        
        logger.info(f"Evaluation complete for {dataset}")
        logger.info(f"  Total samples: {len(utterances)}")
        logger.info(f"  Successful: {len(successful)}")
        logger.info(f"  Total mappings: {total_mappings}")
        logger.info(f"  Abstentions: {total_abstentions}")
        logger.info(f"  Accuracy@1: {accuracy_at_k.get('accuracy@1', 0):.3f}")
        logger.info(f"  Total time: {total_time:.2f}s")
        
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
                "top_k": self.top_k
            },
            "results": results
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
        description="MatchMaker Schema Matching Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        help="Comma-separated list of models"
    )
    
    # Optional arguments
    parser.add_argument(
        "--schema-dir",
        type=str,
        default=None,
        help="Directory with GraphQL schemas"
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild vector index from scratch"
    )
    parser.add_argument(
        "--no-optimization",
        action="store_true",
        help="Disable self-improvement optimization"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidates to retrieve (default: 5)"
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
    
    logger.info("Starting MatchMaker evaluation")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Models: {models}")
    logger.info(f"  Rebuild index: {args.rebuild_index}")
    logger.info(f"  Use optimization: {not args.no_optimization}")
    logger.info(f"  Top-k: {args.top_k}")
    
    # Initialize runner
    try:
        runner = MatchMakerEvaluationRunner(
            property_lookup_file=args.property_lookup_file,
            data_dir=args.data_dir,
            schema_dir=args.schema_dir,
            rebuild_index=args.rebuild_index,
            use_optimization=not args.no_optimization,
            top_k=args.top_k
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
                    jsonl_filename = f"{dataset}_{model_short}.jsonl"
                    runner.save_results_jsonl(result, jsonl_filename)

                    # Also save full results as JSON
                    json_filename = f"{dataset}_{model_short}_full.json"
                    runner.save_results(result, json_filename)

                    print(f"\nResults saved: {runner.results_dir / jsonl_filename}")
                    
                    # Log summary
                    metadata = result.get('metadata', {})
                    print(f"  Samples: {metadata.get('total_samples', 0)}")
                    print(f"  Successful: {metadata.get('successful_samples', 0)}")
                    print(f"  Total mappings: {metadata.get('total_mappings', 0)}")
                    print(f"  Abstentions: {metadata.get('total_abstentions', 0)}")
                    print(f"  Accuracy@1: {metadata.get('accuracy_at_k', {}).get('accuracy@1', 0):.3f}")
                    print(f"  Accuracy@3: {metadata.get('accuracy_at_k', {}).get('accuracy@3', 0):.3f}")
                    print(f"  Accuracy@5: {metadata.get('accuracy_at_k', {}).get('accuracy@5', 0):.3f}")
                    
                    if metadata.get('is_utterance_dataset'):
                        print(f"  [Utterance-based dataset: {dataset}]")
                
            except Exception as e:
                logger.error(f"Evaluation failed for {dataset} with {model}: {e}")
                continue
            
            time.sleep(2)  # Delay between evaluations
    
    total_time = time.time() - total_start
    
    if all_results:
        print(f"\n{'='*60}")
        print("MatchMaker Evaluation Complete!")
        print(f"{'='*60}")
        print(f"  Total combinations: {len(all_results)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Results directory: {runner.results_dir}")
        
        return 0
    else:
        logger.error("No results generated")
        return 1


if __name__ == "__main__":
    sys.exit(main())
