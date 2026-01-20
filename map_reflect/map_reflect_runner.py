#!/usr/bin/env python3
"""
MapReflect Runner - Tool-based schema mapping evaluation
Uses the working map_reflect_agent.py implementation
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
import re
from tqdm import tqdm

# Suppress LiteLLM logging before any imports
os.environ['LITELLM_LOG'] = 'ERROR'
os.environ['LITELLM_DROP_PARAMS'] = 'True'

# Load environment variables
from dotenv import load_dotenv

# Get the directory where this script is located
script_dir = Path(__file__).parent
env_file = script_dir / ".env"

if env_file.exists():
    load_dotenv(env_file)
    print(f"Loaded environment variables from {env_file}")
else:
    load_dotenv()

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable verbose logging from external libraries
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# Disable all LiteLLM related loggers
for name in logging.root.manager.loggerDict:
    if 'litellm' in name.lower():
        logging.getLogger(name).setLevel(logging.ERROR)


class MapReflectMappingAgent:
    """MapReflect mapping agent using the existing map_reflect_agent.py implementation."""
    
    def __init__(self, property_lookup_file: str, data_dir: str = None, model: str = "watsonx/mistralai/mistral-large", rebuild_index: bool = False, dataset: str = None):
        """Initialize MapReflect mapping agent."""
        self.model = model
        self.dataset = dataset
        self.property_lookup_file = Path(property_lookup_file)
        self.data_dir = Path(data_dir) if data_dir else self.property_lookup_file.parent
        self.rebuild_index = rebuild_index
        self.available = False
        self.hierarchy_cache = {}
        
        # Try to load property lookup and set up map_reflect_agent
        try:
            self._setup_map_reflect_agent()
            self.available = True
            logger.info(f"MapReflect mapping agent initialized successfully for dataset: {self.dataset}")
        except Exception as e:
            logger.error(f"Failed to initialize MapReflect agent: {e}")
            self.available = False
    
    def _get_hierarchy_file_for_dataset(self, dataset: str) -> Optional[str]:
        """Get the hierarchy file path for a specific dataset."""
        dataset_lower = dataset.lower()
        hierarchy_file = self.data_dir / f"{dataset_lower}_hierarchy.json"
        
        if hierarchy_file.exists():
            return str(hierarchy_file)
        else:
            logger.warning(f"Dataset-specific hierarchy file not found: {hierarchy_file}")
            return None
    
    def _load_hierarchy_for_dataset(self, dataset: str) -> Dict[str, Any]:
        """Load hierarchy data for a specific dataset."""
        if dataset in self.hierarchy_cache:
            return self.hierarchy_cache[dataset]
        
        hierarchy_file = self._get_hierarchy_file_for_dataset(dataset)
        
        if hierarchy_file and Path(hierarchy_file).exists():
            try:
                with open(hierarchy_file, 'r') as f:
                    hierarchy_data = json.load(f)
                self.hierarchy_cache[dataset] = hierarchy_data
                logger.info(f"Loaded hierarchy for {dataset} from {hierarchy_file}")
                return hierarchy_data
            except Exception as e:
                logger.error(f"Failed to load hierarchy for {dataset}: {e}")
        
        logger.warning(f"Using empty hierarchy for dataset {dataset}")
        self.hierarchy_cache[dataset] = {}
        return {}
    
    def _setup_map_reflect_agent(self):
        """Set up map_reflect_agent with the correct property lookup file."""
        # Add data directory to Python path
        if str(self.data_dir) not in sys.path:
            sys.path.insert(0, str(self.data_dir))
        
        # Import dataset loader
        try:
            from dataset_loader import DatasetLoader
            logger.info("Successfully imported shared DatasetLoader")
        except ImportError as e:
            logger.error(f"Failed to import shared DatasetLoader: {e}")
            raise
        
        # Add current directory to path for map_reflect_agent import
        script_dir = Path(__file__).parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        # Import and configure map_reflect_agent
        try:
            from map_reflect_agent import MapReflectAgent
            
            if not self.property_lookup_file.exists():
                raise FileNotFoundError(f"Property lookup file not found: {self.property_lookup_file}")
            
            # Initialize the MapReflect agent
            self.map_reflect_agent = MapReflectAgent(
                property_lookup_file=str(self.property_lookup_file),
                rebuild_index=self.rebuild_index,
                model=self.model,
                dataset=self.dataset  # Pass dataset for proper collection naming
            )
            
            logger.info(f"Loaded property lookup from {self.property_lookup_file}")
            logger.info(f"MapReflect agent configured with model: {self.model}, dataset: {self.dataset}")
            
        except ImportError as e:
            logger.error(f"Failed to import map_reflect_agent: {e}")
            raise
        except Exception as e:
            logger.error(f"Error setting up map_reflect_agent: {e}")
            raise
    
    def run_agent(self, user_query: str) -> Dict[str, Any]:
        """Run the MapReflect agent on a user query."""
        if not self.available:
            return self._fallback_response(user_query)
        
        try:
            logger.info(f"Running MapReflect agent for query: '{user_query}'")
            
            # Set the model for this run if it has changed
            if hasattr(self.map_reflect_agent, 'model') and self.map_reflect_agent.model != self.model:
                self.map_reflect_agent.model = self.model
            
            # Use the working MapReflect agent directly
            result = self.map_reflect_agent.run(user_query)
            
            # Convert result to expected format
            mappings = result.get('mappings', {})
            source_properties = result.get('source_properties', [])
            destination_properties = result.get('destination_properties', [])
            reflection_feedback = result.get('reflection_feedback', '')
            
            if mappings:
                mapping_result = self._convert_to_mapping_format(mappings, result)
                
                return {
                    "mapping_result": mapping_result,
                    "source_schema": f"map_reflect_source_{len(source_properties)}",
                    "target_schema": f"map_reflect_target_{len(destination_properties)}",
                    "schema_selection_reasoning": f"MapReflect selected {len(source_properties)} source and {len(destination_properties)} destination properties using reflection-based discovery",
                    "error": None,
                    "map_reflect_state": result
                }
            else:
                return {
                    "mapping_result": {},
                    "source_schema": "",
                    "target_schema": "",
                    "schema_selection_reasoning": "MapReflect failed to find relevant mappings",
                    "error": "No valid mappings generated",
                    "map_reflect_state": result
                }
        except RuntimeError as e:
            # Critical error - re-raise
            logger.error(f"CRITICAL MapReflect agent error: {e}")
            raise
            
        except Exception as e:
            logger.error(f"MapReflect agent error: {e}")
            return {
                "mapping_result": {},
                "source_schema": "",
                "target_schema": "",
                "schema_selection_reasoning": "",
                "error": str(e),
                "map_reflect_state": {}
            }
    
    def _fallback_response(self, user_query: str) -> Dict[str, Any]:
        """Fallback response when agent is not available."""
        mapping_result = {
            "properties": {
                "map_reflect_fallback_field": [{
                    "mapProperty": "map_reflect_target_field",
                    "justification": "Fallback mapping - MapReflect components not available"
                }]
            }
        }
        
        return {
            "mapping_result": mapping_result,
            "source_schema": "map_reflect_fallback",
            "target_schema": "map_reflect_target",
            "schema_selection_reasoning": "MapReflect fallback mode",
            "error": "MapReflect components not fully available",
            "map_reflect_state": {}
        }
    
    def _convert_to_mapping_format(self, mappings: Dict[str, List[Dict]], state: Dict) -> Dict[str, Any]:
        """Convert MapReflect mappings to evaluation format."""
        properties = {}
        
        for source_prop, mapping_list in mappings.items():
            if isinstance(mapping_list, list) and mapping_list:
                converted_mappings = []
                for mapping in mapping_list:
                    if isinstance(mapping, dict):
                        target_prop = mapping.get("mapProperty", "unknown")
                        justification = mapping.get("justification", "No justification")
                        
                        converted_mappings.append({
                            "mapProperty": target_prop,
                            "justification": justification
                        })
                
                if converted_mappings:
                    properties[source_prop] = converted_mappings
        
        return {
            "name": f"map_reflect_mapping_{len(properties)}_properties",
            "properties": properties
        }


class MapReflectEvaluationRunner:
    """Runner class for MapReflect schema mapping evaluation."""
    
    def __init__(self, 
                 property_lookup_file: str,
                 data_dir: str,
                 schema_dir: str = None,
                 rebuild_index: bool = False):
        """
        Initialize the MapReflect evaluation runner.
        
        Args:
            property_lookup_file: Path to property lookup JSON file
            data_dir: Directory containing shared data components
            schema_dir: Directory containing GraphQL schemas (optional)
            rebuild_index: Whether to rebuild the ChromaDB index
        """
        self.property_lookup_file = Path(property_lookup_file)
        self.data_dir = Path(data_dir)
        self.schema_dir = Path(schema_dir) if schema_dir else None
        self.rebuild_index = rebuild_index
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Validate required files
        if not self.property_lookup_file.exists():
            raise FileNotFoundError(f"Property lookup file not found: {self.property_lookup_file}")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Initialize the shared dataset loader
        self.dataset_loader = self._initialize_dataset_loader()
        
        logger.info(f"MapReflect runner initialized")
        logger.info(f"   Property lookup file: {self.property_lookup_file}")
        logger.info(f"   Data directory: {self.data_dir}")
        logger.info(f"   Schema directory: {self.schema_dir}")
        logger.info(f"   Rebuild index: {self.rebuild_index}")
    
    def _initialize_dataset_loader(self):
        """Initialize the dataset loader with proper path handling."""
        if str(self.data_dir) not in sys.path:
            sys.path.insert(0, str(self.data_dir))
        
        try:
            from dataset_loader import DatasetLoader
            return DatasetLoader(str(self.data_dir))
        except ImportError as e:
            logger.error(f"Failed to import shared DatasetLoader from {self.data_dir}: {e}")
            raise
    
    def _get_agent_for_model(self, model: str, dataset: str) -> MapReflectMappingAgent:
        """Get or create an agent for the specified model and dataset."""
        return MapReflectMappingAgent(
            str(self.property_lookup_file), 
            str(self.data_dir), 
            model, 
            self.rebuild_index,
            dataset  # Pass dataset for proper ChromaDB collection naming
        )
    
    def make_json_serializable(self, obj):
        """Convert objects to JSON serializable format."""
        try:
            import numpy as np
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
        except ImportError:
            pass
        
        # Handle LangChain message types
        if hasattr(obj, '__class__') and 'Message' in obj.__class__.__name__:
            if hasattr(obj, 'content'):
                return {
                    'type': obj.__class__.__name__,
                    'content': str(obj.content)
                }
            else:
                return {
                    'type': obj.__class__.__name__,
                    'content': str(obj)
                }
        
        if isinstance(obj, dict):
            return {k: self.make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(v) for v in obj]
        elif hasattr(obj, '__dict__'):
            try:
                return self.make_json_serializable(obj.__dict__)
            except:
                return str(obj)
        
        return obj
    
    def save_results(self, results: dict, filename: str):
        """Save results to JSON file with safe serialization."""
        output_file = self.results_dir / filename
        
        try:
            serializable_results = self.make_json_serializable(results)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")
            try:
                minimal_results = {
                    "metadata": results.get("metadata", {}),
                    "error": f"Failed to serialize full results: {str(e)}",
                    "partial_results_count": len(results.get("results", []))
                }
                with open(output_file, 'w') as f:
                    json.dump(minimal_results, f, indent=2)
                logger.info(f"Saved minimal results to {output_file}")
            except Exception as e2:
                logger.error(f"Failed to save even minimal results: {e2}")
    
    def load_evaluation_utterances(self, dataset: str) -> List[Dict[str, Any]]:
        """Load evaluation utterances for a specific dataset using shared loader."""
        return self.dataset_loader.load_evaluation_utterances(dataset)
    
    def _extract_predicted_mappings(self, mapping_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract predicted mappings from the agent result."""
        try:
            if not mapping_result:
                return {}
            
            properties = mapping_result.get("properties", {})
            
            simplified_mappings = {}
            for source_prop, mappings in properties.items():
                if isinstance(mappings, list) and mappings:
                    mapping = mappings[0]
                    if isinstance(mapping, dict):
                        simplified_mappings[source_prop] = {
                            "target_property": mapping.get("mapProperty", ""),
                            "justification": mapping.get("justification", "")
                        }
            
            return simplified_mappings
            
        except Exception as e:
            logger.warning(f"Error extracting predicted mappings: {e}")
            return {}
    
    def _calculate_accuracy(self, predicted_mapping: Dict[str, Any], ground_truth_mapping: Dict[str, str], sample_id: str) -> float:
        """Calculate field-level accuracy by comparing predicted mappings to ground truth.

        Args:
            predicted_mapping: Dict of {source_field: {target_property: str, justification: str}}
            ground_truth_mapping: Dict of {source_field: target_field}
            sample_id: Sample identifier for logging

        Returns:
            Accuracy as fraction of ground truth fields correctly predicted (0.0 to 1.0)
        """
        if not ground_truth_mapping:
            logger.warning(f"No ground truth for {sample_id}")
            return 0.0

        if not predicted_mapping:
            return 0.0

        correct = 0
        total = len(ground_truth_mapping)

        for source_field, expected_target in ground_truth_mapping.items():
            if source_field in predicted_mapping:
                pred = predicted_mapping[source_field]
                # Handle nested format: {target_property: str, justification: str}
                if isinstance(pred, dict):
                    predicted_target = pred.get("target_property", "")
                else:
                    predicted_target = str(pred)

                # Exact match
                if predicted_target == expected_target:
                    correct += 1
                # Case-insensitive match
                elif predicted_target.lower() == expected_target.lower():
                    correct += 1
                # Partial match: check if target field names match (ignoring schema prefix)
                else:
                    pred_field = predicted_target.split('.')[-1] if '.' in predicted_target else predicted_target
                    exp_field = expected_target.split('.')[-1] if '.' in expected_target else expected_target
                    if pred_field.lower() == exp_field.lower():
                        correct += 0.5  # Partial credit for matching field name

        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def _create_error_result(self, sample_id: str, error_message: str, start_time: float) -> Dict[str, Any]:
        """Create an error result structure."""
        execution_time = time.time() - start_time
        
        return {
            "sample_id": sample_id,
            "predicted_mapping": {},
            "accuracy": 0.0,
            "source_schema": "",
            "target_schema": "",
            "schema_selection_reasoning": "",
            "execution_time": execution_time,
            "success": False,
            "error_message": error_message,
            "map_reflect_raw_result": {},
            "map_reflect_state": {},
            "agent_available": False
        }

    def run_single_evaluation(self, utterance: str, sample_id: str, model: str, dataset: str = None, ground_truth_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """Run MapReflect evaluation on a single utterance."""
        start_time = time.time()
        
        try:
            logger.info(f"Running MapReflect on sample {sample_id}")
            logger.info(f"   Utterance: {utterance}")
            logger.info(f"   Dataset: {dataset}")
            logger.info(f"   Model: {model}")
            
            # Get agent for this specific model and dataset
            mapping_agent = self._get_agent_for_model(model, dataset)
            
            # Load dataset-specific hierarchy if available
            if dataset:
                hierarchy_data = mapping_agent._load_hierarchy_for_dataset(dataset)
                if hierarchy_data:
                    logger.info(f"Using dataset-specific hierarchy for {dataset} with {len(hierarchy_data)} products")
            
            # Run the MapReflect agent
            agent_result = mapping_agent.run_agent(utterance)
            
            execution_time = time.time() - start_time
            
            # Process agent result
            success = agent_result.get("error") is None
            mapping_result = agent_result.get("mapping_result", {})
            
            # Extract predicted mappings
            predicted_mapping = self._extract_predicted_mappings(mapping_result)
            
            # Calculate accuracy using ground truth
            accuracy = self._calculate_accuracy(predicted_mapping, ground_truth_mapping or {}, sample_id)
            
            logger.info(f"MapReflect completed for {sample_id} in {execution_time:.2f}s")
            logger.info(f"   Mappings found: {len(predicted_mapping)}")
            
            return {
                "sample_id": sample_id,
                "predicted_mapping": predicted_mapping,
                "accuracy": accuracy,
                "source_schema": agent_result.get("source_schema", ""),
                "target_schema": agent_result.get("target_schema", ""),
                "schema_selection_reasoning": agent_result.get("schema_selection_reasoning", ""),
                "execution_time": execution_time,
                "success": success,
                "error_message": agent_result.get("error"),
                "map_reflect_raw_result": mapping_result,
                "map_reflect_state": agent_result.get("map_reflect_state", {}),
                "agent_available": mapping_agent.available,
                "dataset": dataset,
                "model": model,
                "hierarchy_used": dataset if dataset and mapping_agent._get_hierarchy_file_for_dataset(dataset) else None
            }
        
        except RuntimeError as e:
            # Critical error - re-raise to stop execution
            logger.error(f"CRITICAL ERROR in {sample_id}: {e}")
            raise
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            # Check for critical error patterns
            critical_patterns = [
                "provider not provided",
                "provider not found", 
                "invalid api key",
                "authentication",
                "critical llm",
            ]
            
            if any(pattern in error_msg.lower() for pattern in critical_patterns):
                logger.error(f"CRITICAL ERROR in {sample_id}: {error_msg}")
                raise RuntimeError(f"Critical error: {error_msg}") from e
            
            logger.error(f"MapReflect failed for {sample_id}: {e}")
            return self._create_error_result(sample_id, error_msg, start_time)
    
    
    def _save_checkpoint(self, results: List[Dict], checkpoint_file: Path, dataset: str, model: str, start_time: float):
        """Save checkpoint of partial results."""
        try:
            successful_results = [r for r in results if r["success"]]
            checkpoint_data = {
                "metadata": {
                    "dataset": dataset,
                    "method": "MapReflect",
                    "model": model,
                    "execution_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat(),
                    "total_samples": len(results),
                    "successful_samples": len(successful_results),
                    "total_mappings": sum(len(r.get("predicted_mapping", {})) for r in results),
                    "overall_accuracy": sum(r["accuracy"] for r in successful_results) / len(successful_results) if successful_results else 0.0,
                    "checkpoint": True
                },
                "results": results
            }
            
            serializable_data = self.make_json_serializable(checkpoint_data)
            with open(checkpoint_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def run_evaluation(self, dataset: str, model: str) -> Dict[str, Any]:
        """Run evaluation for a specific dataset and model combination."""
        logger.info(f"Starting MapReflect evaluation for {dataset} with {model}")
    
        # Load evaluation utterances
        utterances = self.load_evaluation_utterances(dataset)
        if not utterances:
            logger.error(f"No utterances found for dataset {dataset}")
            return {}
    
        logger.info(f"Loaded {len(utterances)} utterances for {dataset}")
    
        results = []
        total_start_time = time.time()
        MAX_CONSECUTIVE_FAILURES = 10
        consecutive_failures = 0
    
        # ADD: Incremental save settings
        SAVE_INTERVAL = 10  # Save every 10 samples
        checkpoint_file = self.results_dir / f"{dataset}_{model.replace('/', '_')}_checkpoint.json"
    
        try:
            with tqdm(utterances, desc=f"Processing {dataset}", unit="utterance") as pbar:
                for i, utterance_data in enumerate(pbar):
                    sample_id = utterance_data['sample_id']

                    # Update progress bar description
                    pbar.set_description(f"Processing {sample_id}")

                    # Run single evaluation
                    result = self.run_single_evaluation(
                        utterance_data["utterance"],
                        sample_id,
                        model,
                        dataset,
                        utterance_data.get("ground_truth_mapping", {})
                    )

                    # Add expected information for analysis
                    result["expected_source"] = utterance_data.get("expected_source", "")
                    result["expected_target"] = utterance_data.get("expected_target", "")
                    result["original_utterance"] = utterance_data["utterance"]
                    result["ground_truth_mapping"] = utterance_data.get("ground_truth_mapping", {})

                    results.append(result)

                    # ADD: Incremental save every N samples
                    if (i + 1) % SAVE_INTERVAL == 0:
                        self._save_checkpoint(results, checkpoint_file, dataset, model, total_start_time)
                        logger.info(f"Checkpoint saved: {i + 1}/{len(utterances)} samples")

                    # Update progress bar with status
                    status = "✓" if result["success"] else "✗"
                    mappings_count = len(result.get("predicted_mapping", {}))
                    pbar.set_postfix({"status": status, "mappings": mappings_count})

                    # Failure tracking
                    if not result["success"]:
                        consecutive_failures += 1
                        error_msg = result.get("error_message", "")

                        if "critical" in str(error_msg).lower():
                            logger.error(f"CRITICAL ERROR detected: {error_msg}")
                            raise RuntimeError(f"Stopping: {error_msg}")

                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            logger.error(f"Stopping after {consecutive_failures} consecutive failures")
                            # SAVE BEFORE RAISING
                            self._save_checkpoint(results, checkpoint_file, dataset, model, total_start_time)
                            raise RuntimeError(f"Too many consecutive failures")
                    else:
                        consecutive_failures = 0

                    # Small delay between requests
                    time.sleep(0.5)
    
        except Exception as e:
            # ADD: Save partial results on ANY exception
            logger.error(f"Exception during evaluation: {e}")
            if results:
                logger.info(f"Saving {len(results)} partial results before exit...")
                self._save_checkpoint(results, checkpoint_file, dataset, model, total_start_time)
            raise  # Re-raise to maintain fail-fast behavior
    
        # Continue with normal completion...
        total_execution_time = time.time() - total_start_time
    
        # Calculate statistics
        successful_results = [r for r in results if r["success"]]
        overall_accuracy = sum(r["accuracy"] for r in successful_results) / len(successful_results) if successful_results else 0.0
        total_mappings = sum(len(r.get("predicted_mapping", {})) for r in results)
    
        logger.info(f"MapReflect evaluation completed for {dataset}")
        logger.info(f"   Total utterances: {len(utterances)}")
        logger.info(f"   Successful: {len(successful_results)}")
        logger.info(f"   Total mappings found: {total_mappings}")
        logger.info(f"   Overall accuracy: {overall_accuracy:.3f}")
        logger.info(f"   Total time: {total_execution_time:.2f}s")
    
        # Create final result structure
        final_result = {
            "metadata": {
                "dataset": dataset,
                "method": "MapReflect",
                "model": model,
                "execution_time": total_execution_time,
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(utterances),
                "successful_samples": len(successful_results),
                "total_mappings": total_mappings,
                "overall_accuracy": overall_accuracy,
                "property_lookup_file": str(self.property_lookup_file),
                "data_dir": str(self.data_dir),
                "rebuild_index": self.rebuild_index
            },
            "results": results
        }
    
        return final_result


def main():
    """Main function to run MapReflect evaluation with command line arguments."""
    parser = argparse.ArgumentParser(
        description="MapReflect Schema Mapping Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--property-lookup-file", type=str, required=True,
                       help="Path to property lookup JSON file (required)")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing shared data components (required)")
    
    # Dataset and model arguments
    parser.add_argument("--datasets", type=str, required=True,
                       help="Comma-separated list of datasets to evaluate (required)")
    parser.add_argument("--models", type=str, required=True,
                       help="Comma-separated list of models to use (required)")
    
    # Optional arguments
    parser.add_argument("--schema-dir", type=str, default=None,
                       help="Directory containing GraphQL schemas (optional)")
    parser.add_argument("--samples", type=str, default=None,
                       help="Number of samples (optional, not yet implemented)")
    
    # MapReflect specific options
    parser.add_argument("--rebuild-index", action="store_true",
                       help="Rebuild the ChromaDB index from scratch")
    
    # Evaluation options
    parser.add_argument("--quick", action="store_true",
                       help="Quick evaluation with minimal samples")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    else:
        logging.getLogger("litellm").setLevel(logging.ERROR)
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        
        for name in logging.root.manager.loggerDict:
            if 'litellm' in name.lower():
                logging.getLogger(name).setLevel(logging.ERROR)
    
    # Parse datasets and models
    datasets = [d.strip() for d in args.datasets.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    
    logger.info(f"Starting MapReflect evaluation")
    logger.info(f"   Datasets: {datasets}")
    logger.info(f"   Models: {models}")
    logger.info(f"   Property lookup file: {args.property_lookup_file}")
    logger.info(f"   Data directory: {args.data_dir}")
    logger.info(f"   Schema directory: {args.schema_dir}")
    logger.info(f"   Rebuild index: {args.rebuild_index}")
    logger.info(f"   Quick mode: {args.quick}")
    
    # Initialize the evaluation runner
    try:
        runner = MapReflectEvaluationRunner(
            property_lookup_file=args.property_lookup_file,
            data_dir=args.data_dir,
            schema_dir=args.schema_dir,
            rebuild_index=args.rebuild_index
        )
        logger.info("MapReflect evaluation runner initialized")
    except Exception as e:
        logger.error(f"Failed to initialize MapReflect runner: {e}")
        return 1
    
    # Run evaluations for each dataset-model combination
    all_results = {}
    total_start_time = time.time()
    
    for dataset in datasets:
        for model in models:
            combination_key = f"{dataset}_{model.replace('/', '_')}"
            logger.info(f"\n{'='*60}")
            logger.info(f"Running evaluation: {dataset} with {model}")
            logger.info(f"{'='*60}\n")
            
            try:
                # Run evaluation for this dataset-model combination
                result = runner.run_evaluation(dataset, model)
                
                if result:
                    all_results[combination_key] = result
                    
                    # Save individual result file
                    result_filename = f"{dataset}_{model.replace('/', '_').replace('watsonx_', '')}.json"
                    runner.save_results(result, result_filename)
                    
                    logger.info(f"\nResults saved to: {runner.results_dir / result_filename}")
                    
                    # Log summary statistics
                    metadata = result.get('metadata', {})
                    logger.info(f"   Total samples: {metadata.get('total_samples', 0)}")
                    logger.info(f"   Successful: {metadata.get('successful_samples', 0)}")
                    logger.info(f"   Total mappings: {metadata.get('total_mappings', 0)}")
                    logger.info(f"   Accuracy: {metadata.get('overall_accuracy', 0.0):.3f}")
                else:
                    logger.warning(f"No results returned for {dataset} with {model}")
                    
            except Exception as e:
                logger.error(f"Evaluation failed for {dataset} with {model}: {e}")
                continue
            
            # Add delay between evaluations
            if not args.quick:
                time.sleep(2)
    
    total_execution_time = time.time() - total_start_time
    
    if all_results:
        logger.info(f"\n{'='*60}")
        logger.info("MapReflect evaluation completed!")
        logger.info(f"\n{'='*60}")
        logger.info(f"   Total combinations: {len(all_results)}")
        logger.info(f"   Total execution time: {total_execution_time:.2f}s")
        
        # Calculate overall statistics
        total_samples = sum(r.get('metadata', {}).get('total_samples', 0) for r in all_results.values())
        total_successful = sum(r.get('metadata', {}).get('successful_samples', 0) for r in all_results.values())
        total_all_mappings = sum(r.get('metadata', {}).get('total_mappings', 0) for r in all_results.values())
        avg_accuracy = sum(r.get('metadata', {}).get('overall_accuracy', 0.0) for r in all_results.values()) / len(all_results) if all_results else 0.0
        
        logger.info(f"   Overall samples: {total_samples}")
        logger.info(f"   Overall successful: {total_successful}")
        logger.info(f"   Overall mappings: {total_all_mappings}")
        logger.info(f"   Average accuracy: {avg_accuracy:.3f}")
        logger.info(f"   Results saved in: {runner.results_dir}")
        
        return 0
    else:
        logger.error("No results generated")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
