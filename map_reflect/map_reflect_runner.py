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

# Suppress LiteLLM logging before any imports
import os
os.environ['LITELLM_LOG'] = 'ERROR'
os.environ['LITELLM_DROP_PARAMS'] = 'True'

# Load environment variables from map_reflect directory
from dotenv import load_dotenv

# Get the directory where this script is located (map_reflect directory)
script_dir = Path(__file__).parent
env_file = script_dir / ".env"

# Load .env file from map_reflect directory if it exists
if env_file.exists():
    load_dotenv(env_file)
    print(f"Loaded environment variables from {env_file}")
else:
    # Fallback to default behavior (current directory)
    load_dotenv()

# Configure logging
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
    
    def __init__(self, property_lookup_file: str, data_dir: str = None, model: str = "watsonx/mistralai/mistral-large", rebuild_index: bool = False):
        """Initialize MapReflect mapping agent."""
        self.model = model
        self.property_lookup_file = Path(property_lookup_file)
        self.data_dir = Path(data_dir) if data_dir else self.property_lookup_file.parent
        self.rebuild_index = rebuild_index
        self.available = False
        self.hierarchy_cache = {}  # Cache for dataset-specific hierarchies
        
        # Try to load property lookup and set up map_reflect_agent
        try:
            self._setup_map_reflect_agent()
            self.available = True
            logger.info("MapReflect mapping agent initialized successfully")
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
        
        # Return empty hierarchy if file not found or failed to load
        logger.warning(f"Using empty hierarchy for dataset {dataset}")
        self.hierarchy_cache[dataset] = {}
        return {}
    
    def _setup_map_reflect_agent(self):
        """Set up map_reflect_agent with the correct property lookup file."""
        # Add data directory to Python path for shared components
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
            
            # Check if property lookup file exists
            if not self.property_lookup_file.exists():
                raise FileNotFoundError(f"Property lookup file not found: {self.property_lookup_file}")
            
            # Initialize the MapReflect agent
            self.map_reflect_agent = MapReflectAgent(
                property_lookup_file=str(self.property_lookup_file),
                rebuild_index=self.rebuild_index,
                model=self.model  # Pass the model parameter
            )
            
            logger.info(f"Loaded property lookup from {self.property_lookup_file}")
            
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
                # Convert to mapping format
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
        # Add data directory to Python path
        if str(self.data_dir) not in sys.path:
            sys.path.insert(0, str(self.data_dir))
        
        try:
            from dataset_loader import DatasetLoader
            return DatasetLoader(str(self.data_dir))
        except ImportError as e:
            logger.error(f"Failed to import shared DatasetLoader from {self.data_dir}: {e}")
            raise
    
    def _get_agent_for_model(self, model: str) -> MapReflectMappingAgent:
        """Get or create an agent for the specified model."""
        return MapReflectMappingAgent(
            str(self.property_lookup_file), 
            str(self.data_dir), 
            model, 
            self.rebuild_index
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
        
        # Handle dictionaries
        if isinstance(obj, dict):
            return {k: self.make_json_serializable(v) for k, v in obj.items()}
        
        # Handle lists
        elif isinstance(obj, list):
            return [self.make_json_serializable(v) for v in obj]
        
        # Handle other non-serializable objects
        elif hasattr(obj, '__dict__'):
            try:
                return self.make_json_serializable(obj.__dict__)
            except:
                return str(obj)
        
        # Return as-is if already serializable
        return obj
    
    def save_results(self, results: dict, filename: str):
        """Save results to JSON file with safe serialization."""
        output_file = self.results_dir / filename
        
        try:
            # Convert to JSON serializable format
            serializable_results = self.make_json_serializable(results)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")
            # Try to save a minimal version
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
            
            # Extract mappings from the MapReflect result format
            properties = mapping_result.get("properties", {})
            
            # Convert to a simplified format for evaluation
            simplified_mappings = {}
            for source_prop, mappings in properties.items():
                if isinstance(mappings, list) and mappings:
                    # Take the first mapping if multiple exist
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
    
    def _calculate_accuracy_placeholder(self, predicted_mapping: Dict[str, Any], sample_id: str) -> float:
        """Calculate accuracy placeholder (to be implemented with ground truth)."""
        if not predicted_mapping:
            return 0.0
        
        # For now, return a basic score based on whether we found any mappings
        return 1.0 if len(predicted_mapping) > 0 else 0.0
    
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
    
    def run_single_evaluation(self, utterance: str, sample_id: str, model: str, dataset: str = None) -> Dict[str, Any]:
        """Run MapReflect evaluation on a single utterance."""
        start_time = time.time()
        
        try:
            logger.info(f"Running MapReflect on sample {sample_id}")
            logger.info(f"Utterance: {utterance}")
            logger.info(f"Dataset: {dataset}")
            
            # Get agent for this specific model
            mapping_agent = self._get_agent_for_model(model)
            
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
            
            # Calculate accuracy (placeholder)
            accuracy = self._calculate_accuracy_placeholder(predicted_mapping, sample_id)
            
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
                "hierarchy_used": dataset if dataset and mapping_agent._get_hierarchy_file_for_dataset(dataset) else None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"MapReflect failed for {sample_id}: {e}")
            
            return self._create_error_result(sample_id, str(e), start_time)
    
    def run_evaluation(self, dataset: str, model: str) -> Dict[str, Any]:
        """Run evaluation for a specific dataset and model combination."""
        logger.info(f"Starting MapReflect evaluation for {dataset} with {model}")
        
        # Load evaluation utterances using shared loader
        utterances = self.load_evaluation_utterances(dataset)
        if not utterances:
            logger.error(f"No utterances found for dataset {dataset}")
            return {}
        
        logger.info(f"Loaded {len(utterances)} utterances for {dataset}")
        
        # Run MapReflect on each utterance
        results = []
        total_start_time = time.time()
        
        for i, utterance_data in enumerate(utterances):
            logger.info(f"Processing utterance {i+1}/{len(utterances)}: {utterance_data['sample_id']}")
            
            # Run single evaluation
            result = self.run_single_evaluation(
                utterance_data["utterance"],
                utterance_data["sample_id"],
                model,
                dataset
            )
            
            # Add expected information for analysis
            result["expected_source"] = utterance_data.get("expected_source", "")
            result["expected_target"] = utterance_data.get("expected_target", "")
            result["original_utterance"] = utterance_data["utterance"]
            result["ground_truth_mapping"] = utterance_data.get("ground_truth_mapping", {})
            
            results.append(result)
            
            if (i + 1) % 2 == 0:
                logger.info(f"Completed {i+1}/{len(utterances)} utterances")
            
            # Small delay between requests to avoid overwhelming the LLM
            time.sleep(0.5)
        
        total_execution_time = time.time() - total_start_time
        
        # Calculate overall statistics
        successful_results = [r for r in results if r["success"]]
        overall_accuracy = sum(r["accuracy"] for r in successful_results) / len(successful_results) if successful_results else 0.0
        
        logger.info(f"MapReflect evaluation completed for {dataset}")
        logger.info(f"   Total utterances: {len(utterances)}")
        logger.info(f"   Successful: {len(successful_results)}")
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
    parser = argparse.ArgumentParser(description="MapReflect Schema Mapping Evaluation")
    
    # Dataset and model arguments
    parser.add_argument("--datasets", type=str, default="ITOps",
                       help="Comma-separated list of datasets to evaluate (default: ITOps)")
    parser.add_argument("--models", type=str, default="watsonx/mistralai/mistral-large",
                       help="Comma-separated list of models to use (default: watsonx/mistralai/mistral-large)")
    
    # File and directory arguments - now all required
    parser.add_argument("--property-lookup-file", type=str, required=True,
                       help="Path to property lookup JSON file (e.g., api_lookup.json)")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing shared data components (dataset_loader.py, etc.)")
    parser.add_argument("--schema-dir", type=str, default=None,
                       help="Directory containing GraphQL schemas (optional)")
    
    # MapReflect specific options
    parser.add_argument("--rebuild-index", action="store_true",
                       help="Rebuild the ChromaDB index from scratch")
    
    # Evaluation options
    parser.add_argument("--quick", action="store_true",
                       help="Quick evaluation with minimal samples")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
        # Even in verbose mode, suppress LiteLLM spam
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    else:
        # Disable verbose logging from external libraries when not in verbose mode
        logging.getLogger("litellm").setLevel(logging.ERROR)
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        
        # Suppress all LiteLLM related loggers
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
            logger.info(f"Running evaluation: {dataset} with {model}")
            
            try:
                # Run evaluation for this dataset-model combination
                result = runner.run_evaluation(dataset, model)
                
                if result:
                    all_results[combination_key] = result
                    
                    # Save individual result file
                    result_filename = f"{dataset}_{model.replace('/', '_').replace('watsonx_', '')}.json"
                    runner.save_results(result, result_filename)
                    
                    logger.info(f"Results saved to: {runner.results_dir / result_filename}")
                    
                    # Log summary statistics
                    metadata = result.get('metadata', {})
                    logger.info(f"   Total samples: {metadata.get('total_samples', 0)}")
                    logger.info(f"   Successful: {metadata.get('successful_samples', 0)}")
                    logger.info(f"   Accuracy: {metadata.get('overall_accuracy', 0.0):.3f}")
                else:
                    logger.warning(f"No results returned for {dataset} with {model}")
                    
            except Exception as e:
                logger.error(f"Evaluation failed for {dataset} with {model}: {e}")
                continue
            
            # Add delay between evaluations to avoid overwhelming the API
            if not args.quick:
                time.sleep(2)
    
    total_execution_time = time.time() - total_start_time
    
    if all_results:
        logger.info("MapReflect evaluation completed!")
        logger.info(f"   Total combinations: {len(all_results)}")
        logger.info(f"   Total execution time: {total_execution_time:.2f}s")
        
        # Calculate overall statistics
        total_samples = sum(r.get('metadata', {}).get('total_samples', 0) for r in all_results.values())
        total_successful = sum(r.get('metadata', {}).get('successful_samples', 0) for r in all_results.values())
        avg_accuracy = sum(r.get('metadata', {}).get('overall_accuracy', 0.0) for r in all_results.values()) / len(all_results) if all_results else 0.0
        
        logger.info(f"   Overall samples: {total_samples}")
        logger.info(f"   Overall successful: {total_successful}")
        logger.info(f"   Average accuracy: {avg_accuracy:.3f}")
        logger.info(f"Results saved in: {runner.results_dir}")
        
        return 0
    else:
        logger.error("No results generated")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
