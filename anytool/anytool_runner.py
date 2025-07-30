#!/usr/bin/env python3
"""
AnyTool Runner - Tool-based schema mapping evaluation
Uses the working anytool_agent.py implementation
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

# Load environment variables from anytool directory
from dotenv import load_dotenv

# Get the directory where this script is located (anytool directory)
script_dir = Path(__file__).parent
env_file = script_dir / ".env"

# Load .env file from anytool directory if it exists
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


class AnyToolMappingAgent:
    """AnyTool mapping agent using the existing anytool_agent.py implementation."""
    
    def __init__(self, hierarchy_file: str, data_dir: str = None, model: str = "mistral-large"):
        """Initialize AnyTool mapping agent."""
        self.model = model
        self.hierarchy_file = Path(hierarchy_file)
        self.data_dir = Path(data_dir) if data_dir else self.hierarchy_file.parent
        self.available = False
        
        # Try to load hierarchy and set up anytool_agent
        try:
            self._setup_anytool_agent()
            self.available = True
            logger.info("AnyTool mapping agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AnyTool agent: {e}")
            self.available = False
    
    def _setup_anytool_agent(self):
        """Set up anytool_agent with the correct hierarchy file."""
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
        
        # Add current directory to path for anytool_agent import
        script_dir = Path(__file__).parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        # Import and configure anytool_agent
        try:
            import anytool_agent
            
            # Load hierarchy with absolute path
            if not self.hierarchy_file.exists():
                raise FileNotFoundError(f"Hierarchy file not found: {self.hierarchy_file}")
            
            with open(self.hierarchy_file) as f:
                anytool_agent.hierarchy = json.load(f)
            
            # Import the search function
            from anytool_agent import search_properties
            self.search_properties = search_properties
            
            logger.info(f"Loaded hierarchy with {len(anytool_agent.hierarchy)} products from {self.hierarchy_file}")
            
        except ImportError as e:
            logger.error(f"Failed to import anytool_agent: {e}")
            raise
        except Exception as e:
            logger.error(f"Error setting up anytool_agent: {e}")
            raise
    
    def run_agent(self, user_query: str) -> Dict[str, Any]:
        """Run the AnyTool agent on a user query."""
        if not self.available:
            return self._fallback_response(user_query)
        
        try:
            logger.info(f"Running AnyTool agent for query: '{user_query}'")
            
            # Use the working search_properties function from anytool_agent
            result = self.search_properties(user_query)
            
            # Convert result to expected format
            reasoning = result.get('reasoning', [])
            selected_product = result.get('selected_product', '')
            
            if isinstance(reasoning, list) and reasoning:
                # Convert to mapping format
                mapping_result = self._convert_to_mapping_format(reasoning, result)
                
                return {
                    "mapping_result": mapping_result,
                    "source_schema": selected_product,
                    "target_schema": "anytool_target",
                    "schema_selection_reasoning": f"AnyTool selected {selected_product} using tool-based discovery",
                    "error": None,
                    "anytool_state": result
                }
            else:
                return {
                    "mapping_result": {},
                    "source_schema": "",
                    "target_schema": "",
                    "schema_selection_reasoning": "AnyTool failed to find relevant mappings",
                    "error": "No valid reasoning generated",
                    "anytool_state": result
                }
                
        except Exception as e:
            logger.error(f"AnyTool agent error: {e}")
            return {
                "mapping_result": {},
                "source_schema": "",
                "target_schema": "",
                "schema_selection_reasoning": "",
                "error": str(e),
                "anytool_state": {}
            }
    
    def _fallback_response(self, user_query: str) -> Dict[str, Any]:
        """Fallback response when agent is not available."""
        mapping_result = {
            "properties": {
                "anytool_fallback_field": [{
                    "mapProperty": "anytool_target_field",
                    "justification": "Fallback mapping - AnyTool components not available"
                }]
            }
        }
        
        return {
            "mapping_result": mapping_result,
            "source_schema": "anytool_fallback",
            "target_schema": "anytool_target",
            "schema_selection_reasoning": "AnyTool fallback mode",
            "error": "AnyTool components not fully available",
            "anytool_state": {}
        }
    
    def _convert_to_mapping_format(self, reasoning: List[Dict], state: Dict) -> Dict[str, Any]:
        """Convert AnyTool reasoning to mapping format."""
        properties = {}
        
        for item in reasoning:
            if isinstance(item, dict):
                source_prop = item.get("source_property", "unknown")
                mapped_prop = item.get("mapped_property", "unknown")
                justification = item.get("justification", "No justification")
                
                properties[source_prop] = [{
                    "mapProperty": mapped_prop,
                    "justification": justification
                }]
        
        return {
            "name": f"anytool_mapping_{state.get('selected_product', 'unknown')}",
            "properties": properties
        }


class AnyToolEvaluationRunner:
    """Runner class for AnyTool schema mapping evaluation."""
    
    def __init__(self, 
                 hierarchy_file: str,
                 data_dir: str,
                 schema_dir: str = None):
        """
        Initialize the AnyTool evaluation runner.
        
        Args:
            hierarchy_file: Path to hierarchy JSON file
            data_dir: Directory containing shared data components
            schema_dir: Directory containing GraphQL schemas (optional)
        """
        self.hierarchy_file = Path(hierarchy_file)
        self.data_dir = Path(data_dir)
        self.schema_dir = Path(schema_dir) if schema_dir else None
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Validate required files
        if not self.hierarchy_file.exists():
            raise FileNotFoundError(f"Hierarchy file not found: {self.hierarchy_file}")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Initialize the shared dataset loader
        self.dataset_loader = self._initialize_dataset_loader()
        
        logger.info(f"AnyTool runner initialized")
        logger.info(f"   Hierarchy file: {self.hierarchy_file}")
        logger.info(f"   Data directory: {self.data_dir}")
        logger.info(f"   Schema directory: {self.schema_dir}")
    
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
    
    def _get_agent_for_model(self, model: str) -> AnyToolMappingAgent:
        """Get or create an agent for the specified model."""
        return AnyToolMappingAgent(str(self.hierarchy_file), str(self.data_dir), model)
    
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
            
            # Extract mappings from the AnyTool result format
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
            "anytool_raw_result": {},
            "anytool_state": {},
            "agent_available": False
        }
    
    def run_single_evaluation(self, utterance: str, sample_id: str, model: str) -> Dict[str, Any]:
        """Run AnyTool evaluation on a single utterance."""
        start_time = time.time()
        
        try:
            logger.info(f"Running AnyTool on sample {sample_id}")
            logger.info(f"Utterance: {utterance}")
            
            # Get agent for this specific model
            mapping_agent = self._get_agent_for_model(model)
            
            # Run the AnyTool agent
            agent_result = mapping_agent.run_agent(utterance)
            
            execution_time = time.time() - start_time
            
            # Process agent result
            success = agent_result.get("error") is None
            mapping_result = agent_result.get("mapping_result", {})
            
            # Extract predicted mappings
            predicted_mapping = self._extract_predicted_mappings(mapping_result)
            
            # Calculate accuracy (placeholder)
            accuracy = self._calculate_accuracy_placeholder(predicted_mapping, sample_id)
            
            logger.info(f"AnyTool completed for {sample_id} in {execution_time:.2f}s")
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
                "anytool_raw_result": mapping_result,
                "anytool_state": agent_result.get("anytool_state", {}),
                "agent_available": mapping_agent.available
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"AnyTool failed for {sample_id}: {e}")
            
            return self._create_error_result(sample_id, str(e), start_time)
    
    def run_evaluation(self, dataset: str, model: str) -> Dict[str, Any]:
        """Run evaluation for a specific dataset and model combination."""
        logger.info(f"Starting AnyTool evaluation for {dataset} with {model}")
        
        # Load evaluation utterances using shared loader
        utterances = self.load_evaluation_utterances(dataset)
        if not utterances:
            logger.error(f"No utterances found for dataset {dataset}")
            return {}
        
        logger.info(f"Loaded {len(utterances)} utterances for {dataset}")
        
        # Run AnyTool on each utterance
        results = []
        total_start_time = time.time()
        
        for i, utterance_data in enumerate(utterances):
            logger.info(f"Processing utterance {i+1}/{len(utterances)}: {utterance_data['sample_id']}")
            
            # Run single evaluation
            result = self.run_single_evaluation(
                utterance_data["utterance"],
                utterance_data["sample_id"],
                model
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
        
        logger.info(f"AnyTool evaluation completed for {dataset}")
        logger.info(f"   Total utterances: {len(utterances)}")
        logger.info(f"   Successful: {len(successful_results)}")
        logger.info(f"   Overall accuracy: {overall_accuracy:.3f}")
        logger.info(f"   Total time: {total_execution_time:.2f}s")
        
        # Create final result structure
        final_result = {
            "metadata": {
                "dataset": dataset,
                "method": "AnyTool",
                "model": model,
                "execution_time": total_execution_time,
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(utterances),
                "successful_samples": len(successful_results),
                "overall_accuracy": overall_accuracy,
                "hierarchy_file": str(self.hierarchy_file),
                "data_dir": str(self.data_dir)
            },
            "results": results
        }
        
        return final_result


def main():
    """Main function to run AnyTool evaluation with command line arguments."""
    parser = argparse.ArgumentParser(description="AnyTool Schema Mapping Evaluation")
    
    # Dataset and model arguments
    parser.add_argument("--datasets", type=str, default="ITOps",
                       help="Comma-separated list of datasets to evaluate (default: ITOps)")
    parser.add_argument("--models", type=str, default="mistral-large",
                       help="Comma-separated list of models to use (default: mistral-large)")
    
    # File and directory arguments - make them optional with smart defaults
    parser.add_argument("--hierarchy-file", type=str, default=None,
                       help="Path to hierarchy.json file (default: auto-detect)")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Directory containing shared data components (default: auto-detect)")
    parser.add_argument("--schema-dir", type=str, default=None,
                       help="Directory containing GraphQL schemas (optional)")
    
    # Evaluation options
    parser.add_argument("--quick", action="store_true",
                       help="Quick evaluation with minimal samples")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Auto-detect paths if not provided
    script_dir = Path(__file__).parent
    
    # Set default hierarchy file path
    if args.hierarchy_file is None:
        # Try common locations
        possible_hierarchy_paths = [
            script_dir / "hierarchy.json",
            script_dir / "../data/hierarchy.json", 
            script_dir.parent / "data/hierarchy.json"
        ]
        
        for path in possible_hierarchy_paths:
            if path.exists():
                args.hierarchy_file = str(path)
                break
        
        if args.hierarchy_file is None:
            logger.error("Could not find hierarchy.json file. Please specify with --hierarchy-file")
            logger.error("Tried locations:")
            for path in possible_hierarchy_paths:
                logger.error(f"  {path}")
            return 1
    
    # Set default data directory
    if args.data_dir is None:
        # Try common locations
        possible_data_dirs = [
            script_dir / "data",
            script_dir / "../data",
            script_dir.parent / "data"
        ]
        
        for path in possible_data_dirs:
            if path.exists() and (path / "dataset_loader.py").exists():
                args.data_dir = str(path)
                break
        
        if args.data_dir is None:
            logger.error("Could not find data directory with dataset_loader.py. Please specify with --data-dir")
            logger.error("Tried locations:")
            for path in possible_data_dirs:
                logger.error(f"  {path}")
            return 1
    
    # Set default schema directory if not provided
    if args.schema_dir is None:
        data_dir_path = Path(args.data_dir)
        schema_path = data_dir_path / "graphql"
        if schema_path.exists():
            args.schema_dir = str(schema_path)
    
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
    
    logger.info(f"Starting AnyTool evaluation")
    logger.info(f"   Datasets: {datasets}")
    logger.info(f"   Models: {models}")
    logger.info(f"   Hierarchy file: {args.hierarchy_file}")
    logger.info(f"   Data directory: {args.data_dir}")
    logger.info(f"   Schema directory: {args.schema_dir}")
    logger.info(f"   Quick mode: {args.quick}")
    
    # Initialize the evaluation runner
    try:
        runner = AnyToolEvaluationRunner(
            hierarchy_file=args.hierarchy_file,
            data_dir=args.data_dir,
            schema_dir=args.schema_dir
        )
        logger.info("AnyTool evaluation runner initialized")
    except Exception as e:
        logger.error(f"Failed to initialize AnyTool runner: {e}")
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
                    
                    # Save individual result file using ReAct naming convention
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
        logger.info("AnyTool evaluation completed!")
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
