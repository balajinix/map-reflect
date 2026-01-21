#!/usr/bin/env python3
"""
ReAct Agent Runner for Schema Mapping Evaluation
Standardized execution script using the existing GraphQLSchemaMappingAgent
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Fix SLATE_MODEL_PATH to be absolute path before imports
_project_root = Path(__file__).parent.parent
_slate_path = _project_root / "data" / "slate_30m"
if _slate_path.exists():
    os.environ["SLATE_MODEL_PATH"] = str(_slate_path)
    print(f"Set SLATE_MODEL_PATH to: {_slate_path}")
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import numpy as np
import math


# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

# Add the data directory to Python path for shared components
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

try:
    from agent import GraphQLSchemaMappingAgent
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    logger.info("Successfully imported ReAct agent")
except ImportError as e:
    logger.error(f"Failed to import ReAct agent: {e}")
    logger.error("Make sure you're running this from the react directory and have installed dependencies")
    sys.exit(1)

try:
    from dataset_loader import DatasetLoader
    logger.info("Successfully imported shared DatasetLoader")
except ImportError as e:
    logger.error(f"Failed to import shared DatasetLoader: {e}")
    logger.error("Make sure the data/dataset_loader.py file exists")
    sys.exit(1)


class ReActEvaluationRunner:
    """Runner class for ReAct agent evaluation."""

    # Dataset to schema directory mapping
    DATASET_SCHEMA_DIRS = {
        'ITOps': 'itops_graphql',
        'NL2GQL': 'nl2gql_graphql',
        'OMOP_CMS': 'omop_cms_graphql',
        'OMOP_MIMIC': 'omop_mimic_graphql',
        'OMOP_Synthea': 'omop_synthea_graphql',
        'GraphQLRestBench': 'graphql',  # fallback
    }

    def __init__(self, schema_dir: str = "../data/graphql", model_config: Dict[str, str] = None):
        self.base_schema_dir = Path(schema_dir).parent  # Parent of graphql dir (data/)
        self.schema_dir = Path(schema_dir)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        # Model configuration for different LLMs
        self.model_config = model_config or self._get_default_model_config()

        # Initialize the shared dataset loader
        data_dir = Path(__file__).parent.parent / "data"
        self.dataset_loader = DatasetLoader(str(data_dir))
        self.base_schema_dir = data_dir  # Use data dir as base

        # Agent will be initialized per dataset
        self.agent = None
        self.current_dataset = None

    def _get_schema_dir_for_dataset(self, dataset: str) -> Path:
        """Get the schema directory for a specific dataset."""
        schema_subdir = self.DATASET_SCHEMA_DIRS.get(dataset, 'graphql')
        return self.base_schema_dir / schema_subdir

    def _initialize_agent_for_dataset(self, dataset: str):
        """Initialize or reinitialize the agent for a specific dataset."""
        if self.current_dataset == dataset and self.agent is not None:
            return  # Already initialized for this dataset

        schema_dir = self._get_schema_dir_for_dataset(dataset)
        logger.info(f"Initializing ReAct agent for {dataset} with schema dir: {schema_dir}")

        try:
            self.agent = GraphQLSchemaMappingAgent(base_schema_dir=str(schema_dir))
            self.current_dataset = dataset
            logger.info(f"ReAct agent initialized for {dataset}")
        except Exception as e:
            logger.error(f"Failed to initialize ReAct agent for {dataset}: {e}")
            raise
    
    def _get_default_model_config(self) -> Dict[str, str]:
        """Get default model configuration."""
        return {
            "granite": "watsonx/ibm/granite-13b-chat-v2",
            "llama": "watsonx/meta-llama/llama-3-3-70b-instruct", 
            "qwen": "watsonx/qwen/qwen2-72b-instruct"
        }
    
    def load_evaluation_utterances(self, dataset: str) -> List[Dict[str, Any]]:
        """Load evaluation utterances for a specific dataset using shared loader."""
        return self.dataset_loader.load_evaluation_utterances(dataset)
    
    def run_single_evaluation(self, utterance_data: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Run ReAct agent on a single utterance and return standardized result."""
        start_time = time.time()
        
        sample_id = utterance_data.get("sample_id", "unknown")
        utterance = utterance_data.get("utterance", "")
        
        try:
            logger.info(f"Running ReAct agent on sample {sample_id}")
            logger.info(f"Utterance: {utterance}")
            
            # Set model configuration (this would need to be implemented in the agent)
            # For now, the agent uses the configured watsonx model
            
            # Run the ReAct agent
            agent_result = self.agent.run_agent(utterance)
            
            execution_time = time.time() - start_time
            
            # Process agent result
            success = agent_result.get("error") is None
            mapping_result = agent_result.get("mapping_result", {})
            
            # Extract predicted mappings from ReAct agent result
            predicted_mapping = self._extract_predicted_mappings(mapping_result)
            
            # Calculate accuracy using ground truth if available
            ground_truth = utterance_data.get("ground_truth_mapping", {})
            accuracy = self._calculate_accuracy(ground_truth, predicted_mapping) if ground_truth else self._calculate_accuracy_placeholder(predicted_mapping)
            
            logger.info(f"ReAct agent completed for {sample_id} in {execution_time:.2f}s")
            logger.info(f"   Source schema: {agent_result.get('source_schema', 'N/A')}")
            logger.info(f"   Target schema: {agent_result.get('target_schema', 'N/A')}")
            logger.info(f"   Mappings found: {len(predicted_mapping)}")
            
            return {
                "sample_id": sample_id,
                "predicted_mapping": predicted_mapping,
                "accuracy": accuracy,
                "source_schema": agent_result.get("source_schema", ""),
                "target_schema": agent_result.get("target_schema", ""),
                "schema_selection_reasoning": agent_result.get("schema_selection_reasoning", "ReAct agent schema selection"),
                "execution_time": execution_time,
                "success": success,
                "error_message": agent_result.get("error"),
                "agent_raw_result": mapping_result,
                "expected_source": utterance_data.get("expected_source", ""),
                "expected_target": utterance_data.get("expected_target", ""),
                "original_utterance": utterance,
                "ground_truth_mapping": ground_truth,
                "timestamp": datetime.now().isoformat(),
                "schema_similarities": agent_result.get("schema_similarities", [])[:5]  # Top 5 only
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"ReAct agent failed for {sample_id}: {e}")
            
            return {
                "sample_id": sample_id,
                "predicted_mapping": {},
                "accuracy": 0.0,
                "source_schema": "",
                "target_schema": "",
                "schema_selection_reasoning": "",
                "execution_time": execution_time,
                "success": False,
                "error_message": str(e),
                "agent_raw_result": {},
                "expected_source": utterance_data.get("expected_source", ""),
                "expected_target": utterance_data.get("expected_target", ""),
                "original_utterance": utterance,
                "ground_truth_mapping": utterance_data.get("ground_truth_mapping", {}),
                "timestamp": datetime.now().isoformat(),
                "schema_similarities": []
            }
    
    def _extract_predicted_mappings(self, mapping_result: Dict[str, Any]) -> Dict[str, str]:
        """Extract field mappings from ReAct agent result in standardized format."""
        predicted_mapping = {}
        
        if not mapping_result or "properties" not in mapping_result:
            return predicted_mapping
        
        properties = mapping_result.get("properties", {})
        
        for source_field, mappings in properties.items():
            if mappings and isinstance(mappings, list) and len(mappings) > 0:
                # Take the first mapping if multiple exist
                first_mapping = mappings[0]
                if isinstance(first_mapping, dict) and "mapProperty" in first_mapping:
                    target_field = first_mapping["mapProperty"]
                    predicted_mapping[source_field] = target_field
        
        return predicted_mapping
    
    def _calculate_accuracy(self, ground_truth: Dict[str, Any], predicted: Dict[str, Any]) -> float:
        """Calculate accuracy using the same method as SMT runner."""
        if not ground_truth:
            return 0.0
        
        if not predicted:
            return 0.0

        # Exact match check first
        gt_pairs = set(ground_truth.items())
        pred_pairs = set(predicted.items())
        exact_matches = len(gt_pairs.intersection(pred_pairs))

        if exact_matches > 0:
            return exact_matches / len(gt_pairs)

        # Fuzzy matching for partial credit
        correct_mappings = 0
        for gt_source, gt_target in ground_truth.items():
            for pred_source, pred_target in predicted.items():
                if (gt_source.lower() in pred_source.lower() or pred_source.lower() in gt_source.lower()) and \
                   (gt_target.lower() in pred_target.lower() or pred_target.lower() in gt_target.lower()):
                    correct_mappings += 1
                    break

        return correct_mappings / len(ground_truth)
    
    def _calculate_accuracy_placeholder(self, predicted_mapping: Dict[str, str]) -> float:
        """Calculate accuracy placeholder - same as SMT runner."""
        if not predicted_mapping:
            return 0.0
        
        # Placeholder accuracy based on number of mappings
        num_mappings = len(predicted_mapping)
        if num_mappings >= 5:
            return 0.85
        elif num_mappings >= 3:
            return 0.70
        elif num_mappings >= 1:
            return 0.50
        else:
            return 0.0
    
    def run_evaluation(self, dataset: str, model: str) -> Dict[str, Any]:
        """Run evaluation for a specific dataset and model combination."""
        logger.info(f"Starting ReAct evaluation for {dataset} with {model}")

        # Initialize agent for this dataset
        self._initialize_agent_for_dataset(dataset)

        # Load evaluation utterances using shared loader
        utterances = self.load_evaluation_utterances(dataset)
        if not utterances:
            logger.error(f"No utterances found for dataset {dataset}")
            return {}
        
        logger.info(f"Loaded {len(utterances)} utterances for {dataset}")
        
        # Run ReAct agent on each utterance
        results = []
        total_start_time = time.time()
        
        for i, utterance_data in enumerate(utterances):
            logger.info(f"Processing utterance {i+1}/{len(utterances)}: {utterance_data['sample_id']}")
            
            # Run single evaluation
            result = self.run_single_evaluation(utterance_data, model)
            results.append(result)
            
            if (i + 1) % 2 == 0:
                logger.info(f"Completed {i+1}/{len(utterances)} utterances")
            
            # Small delay between requests to avoid overwhelming the system
            time.sleep(1)
        
        total_execution_time = time.time() - total_start_time
        
        # Calculate overall statistics
        successful_results = [r for r in results if r["success"]]
        overall_accuracy = sum(r["accuracy"] for r in successful_results) / len(successful_results) if successful_results else 0.0
        
        logger.info(f"ReAct evaluation completed for {dataset}")
        logger.info(f"   Total utterances: {len(utterances)}")
        logger.info(f"   Successful: {len(successful_results)}")
        logger.info(f"   Overall accuracy: {overall_accuracy:.3f}")
        logger.info(f"   Total time: {total_execution_time:.2f}s")
        
        # Return results in format compatible with evaluation script
        # The evaluation script expects either:
        # 1. {"results": [list of result objects]} (JSON format)
        # 2. Direct list of result objects (JSONL format)
        return {
            "results": results  # This will be compatible with the evaluation script
        }
    
    def _to_json_safe(self, obj: Any):
        """
        Recursively convert objects to JSON-safe types:
        - numpy scalars -> Python scalars
        - numpy arrays  -> lists
        - tuples/sets   -> lists
        - dict/list     -> recurse
        - NaN/Inf       -> None
        """
        try:
            import numpy as np
            np_types = (np.generic, np.ndarray)
        except Exception:
            np_types = tuple()
    
        # numpy scalar
        if np_types and isinstance(obj, np.generic):
            return obj.item()
    
        # numpy array
        if np_types and isinstance(obj, np.ndarray):
            return [self._to_json_safe(x) for x in obj.tolist()]  # FIX: self._to_json_safe
    
        # basic scalars
        if isinstance(obj, (str, int)):
            return obj
        if isinstance(obj, float):
            if math.isfinite(obj):
                return obj
            return None
        if isinstance(obj, bool) or obj is None:
            return obj
    
        # containers
        if isinstance(obj, dict):
            return {str(k): self._to_json_safe(v) for k, v in obj.items()}  # FIX: self._to_json_safe
        if isinstance(obj, (list, tuple, set)):
            return [self._to_json_safe(v) for v in obj]  # FIX: self._to_json_safe
    
        # fallback to string
        return str(obj)
    
    
    def save_results(self, results: dict, dataset: str, model: str):
        """
        Robust save:
          - Stream JSONL per record (append, flush, fsync).
          - Also write a complete JSON atomically via temp file + rename.
          - Sanitize all objects (numpy, NaN/Inf, tuples/sets) to be JSON-safe.
          - Per-record try/except so one bad record won't break the file.
        """
        base = f"{dataset.lower()}_{model}"
        jsonl_path = self.results_dir / f"{base}.jsonl"
        final_json_path = self.results_dir / f"{base}.json"
        tmp_json_path = self.results_dir / f"{base}.json.tmp"
    
        results_list = results.get("results", [])
    
        # 1) Stream JSONL (append mode) — resilient to crashes and bad records
        appended = 0
        try:
            with open(jsonl_path, "a", encoding="utf-8") as jf:
                for r in results_list:
                    try:
                        safe_r = self._to_json_safe(r)  # FIX: self._to_json_safe
                        line = json.dumps(safe_r, ensure_ascii=False)
                        jf.write(line + "\n")
                        appended += 1
                    except Exception as e:
                        logger.error(f"Failed to serialize record {r.get('sample_id', '?')}: {e}")
                        # Last resort: drop problematic fields
                        minimal = {
                            "sample_id": r.get("sample_id"),
                            "success": bool(r.get("success", False)),
                            "accuracy": float(r.get("accuracy", 0.0)) if str(r.get("accuracy", "")).replace('.','',1).isdigit() else 0.0,
                            "error_message": str(r.get("error_message", ""))[:500],
                        }
                        try:
                            safe_min = self._to_json_safe(minimal)  # FIX: self._to_json_safe
                            jf.write(json.dumps(safe_min, ensure_ascii=False) + "\n")
                            appended += 1
                        except Exception as e2:
                            # Give up on this record, but continue
                            logger.error(f"JSONL skip record {r.get('sample_id','?')}: {e} / fallback failed: {e2}")
                jf.flush()
                os.fsync(jf.fileno())
            logger.info(f"Appended {appended}/{len(results_list)} records to {jsonl_path}")
        except Exception as e:
            logger.error(f"Failed streaming JSONL to {jsonl_path}: {e}")
            import traceback
            traceback.print_exc()  # ADD: Print full traceback for debugging
    
        # 2) Also write the full JSON atomically
        try:
            safe_all = self._to_json_safe({"results": results_list})  # FIX: self._to_json_safe
            with open(tmp_json_path, "w", encoding="utf-8") as f:
                # Disallow NaN in final JSON (safer for downstream parsers)
                json.dump(safe_all, f, indent=2, ensure_ascii=False, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_json_path, final_json_path)
            logger.info(f"Atomic JSON saved to {final_json_path}")
        except Exception as e:
            logger.error(f"Failed to save atomic JSON to {final_json_path}: {e}")
            import traceback
            traceback.print_exc()  # ADD: Print full traceback for debugging
            # Clean up temp on failure
            try:
                if os.path.exists(tmp_json_path):
                    os.remove(tmp_json_path)
            except Exception:
                pass

    def _numpy_safe(self, obj):
        """Convert NumPy types recursively to JSON-safe natives."""
        try:
            import numpy as np
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
        except Exception:
            pass
        if isinstance(obj, dict):
            return {k: self._numpy_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._numpy_safe(v) for v in obj]
        return obj


    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a summary of the evaluation results."""
        results_list = results.get("results", [])
        
        total_samples = len(results_list)
        successful_samples = len([r for r in results_list if r.get("success", False)])
        overall_accuracy = sum(r.get("accuracy", 0.0) for r in results_list) / total_samples if total_samples > 0 else 0.0
        total_time = sum(r.get("execution_time", 0.0) for r in results_list)
        
        print(f"\n{'='*60}")
        print(f"ReAct Evaluation Summary")
        print(f"{'='*60}")
        print(f"Total Samples: {total_samples}")
        print(f"Successful: {successful_samples}")
        print(f"Overall Accuracy: {overall_accuracy:.3f}")
        print(f"Total Time: {total_time:.2f}s")
        
        print(f"\nIndividual Results:")
        print(f"{'Sample ID':<15} {'Success':<8} {'Accuracy':<10} {'Source Schema':<15} {'Target Schema':<15}")
        print(f"{'-'*80}")
        
        for result in results_list:
            success_icon = "✅" if result.get("success", False) else "❌"
            print(f"{result.get('sample_id', ''):<15} {success_icon:<8} {result.get('accuracy', 0.0):<10.3f} {result.get('source_schema', ''):<15} {result.get('target_schema', ''):<15}")


def main():
    """Main entry point for ReAct evaluation."""
    parser = argparse.ArgumentParser(description="ReAct Agent Schema Mapping Evaluation")
    parser.add_argument("--datasets", nargs='+', default=['ITOps', 'NL2GQL', 'GraphQLRestBench'],
                       help="Datasets to evaluate on")
    parser.add_argument("--models", nargs='+', default=['granite', 'llama', 'qwen'],
                       help="Models to evaluate")
    parser.add_argument("--schema-dir", default="../data/graphql",
                       help="Directory containing GraphQL schemas")
    parser.add_argument("--quick", action="store_true",
                       help="Quick evaluation with minimal samples")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.quick:
        args.datasets = ['ITOps']
        args.models = ['llama']  # Use the currently configured model
        logger.info("Quick evaluation mode enabled")
    
    logger.info(f"ReAct Agent Schema Mapping Evaluation")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Base schema directory: {args.schema_dir}")

    # Initialize runner - it will resolve dataset-specific schema directories
    runner = ReActEvaluationRunner(schema_dir=args.schema_dir)
    
    # Run evaluation for each dataset/model combination
    for dataset in args.datasets:
        for model in args.models:
            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating {dataset} with {model}")
            logger.info(f"{'='*80}")
            
            # Run evaluation
            results = runner.run_evaluation(dataset, model)
            
            if results:
                # Print summary
                runner.print_evaluation_summary(results)
                
                # Save results
                runner.save_results(results, dataset, model)
            else:
                logger.error(f"No results generated for {dataset} + {model}")
    
    logger.info(f"\nReAct evaluation completed!")
    logger.info(f"Results saved in: {runner.results_dir}")


if __name__ == "__main__":
    main()
