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
    
    def _setup_matchmaker_agent(self):
        """Set up MatchMaker agent."""
        # Add current directory to path for import
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        from match_maker_agent import MatchMakerAgent
        
        if not self.property_lookup_file.exists():
            raise FileNotFoundError(
                f"Property lookup file not found: {self.property_lookup_file}"
            )
        
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
    
    def run_agent(self, source_attribute: str) -> Dict[str, Any]:
        """
        Run MatchMaker on a single source attribute.
        
        Args:
            source_attribute: Source attribute ID to match
            
        Returns:
            Dict with match results
        """
        if not self.available:
            return self._fallback_response(source_attribute)
        
        try:
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
            logger.error(f"MatchMaker failed for {source_attribute}: {e}")
            return {
                "source_attribute": source_attribute,
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
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
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
        source_attribute: str,
        sample_id: str,
        model: str,
        dataset: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run MatchMaker on a single source attribute."""
        start_time = time.time()
        
        try:
            logger.info(f"Running MatchMaker on sample {sample_id}")
            logger.info(f"  Source attribute: {source_attribute}")
            logger.info(f"  Dataset: {dataset}")
            logger.info(f"  Model: {model}")
            
            # Get agent
            agent = self._get_agent_for_model(model, dataset)
            
            # Run agent
            result = agent.run_agent(source_attribute)
            
            execution_time = time.time() - start_time
            
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
            
            # Compile result
            return {
                "sample_id": sample_id,
                "source_attribute": source_attribute,
                "selected_match": result.get('selected_match'),
                "abstain": result.get('abstain', False),
                "candidates": result.get('candidates', []),
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
            
            return {
                "sample_id": sample_id,
                "source_attribute": source_attribute,
                "selected_match": None,
                "abstain": True,
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
        """Run evaluation for dataset-model combination."""
        logger.info(f"Starting MatchMaker evaluation: {dataset} with {model}")
        
        # Load evaluation data
        utterances = self.load_evaluation_utterances(dataset)
        if not utterances:
            logger.error(f"No evaluation data for {dataset}")
            return {}
        
        logger.info(f"Loaded {len(utterances)} evaluation samples")
        
        # Run MatchMaker on each sample
        results = []
        total_start = time.time()
        
        from tqdm import tqdm
        
        with tqdm(utterances, desc=f"Evaluating {dataset}", unit="sample") as pbar:
            for item in pbar:
                sample_id = item['sample_id']
                source_attr = item.get('source_attribute', item.get('utterance', ''))
                ground_truth = item.get('ground_truth_target', item.get('expected_target'))
                
                pbar.set_description(f"Processing {sample_id}")
                
                result = self.run_single_evaluation(
                    source_attribute=source_attr,
                    sample_id=sample_id,
                    model=model,
                    dataset=dataset,
                    ground_truth=ground_truth
                )
                
                results.append(result)
                
                # Update progress bar
                status = "✓" if result["success"] else "✗"
                pbar.set_postfix({
                    "status": status,
                    "match": "Yes" if result["selected_match"] else "No"
                })
                
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
        total_matches = sum(1 for r in results if r["selected_match"])
        total_abstentions = sum(1 for r in results if r["abstain"])
        
        logger.info(f"Evaluation complete for {dataset}")
        logger.info(f"  Total samples: {len(utterances)}")
        logger.info(f"  Successful: {len(successful)}")
        logger.info(f"  Matches found: {total_matches}")
        logger.info(f"  Abstentions: {total_abstentions}")
        logger.info(f"  Accuracy@1: {accuracy_at_k.get('accuracy@1', 0):.3f}")
        logger.info(f"  Total time: {total_time:.2f}s")
        
        return {
            "metadata": {
                "dataset": dataset,
                "method": "MatchMaker",
                "model": model,
                "execution_time": total_time,
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(utterances),
                "successful_samples": len(successful),
                "total_matches": total_matches,
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
                    
                    # Save individual result
                    filename = f"{dataset}_{model.replace('/', '_').replace('watsonx_', '')}.json"
                    runner.save_results(result, filename)
                    
                    print(f"\nResults saved: {runner.results_dir / filename}")
                    
                    # Log summary
                    metadata = result.get('metadata', {})
                    print(f"  Samples: {metadata.get('total_samples', 0)}")
                    print(f"  Successful: {metadata.get('successful_samples', 0)}")
                    print(f"  Matches: {metadata.get('total_matches', 0)}")
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
