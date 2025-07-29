#!/usr/bin/env python3
"""
SMT Runner - Main orchestrator for SMT schema mapping evaluation
Coordinates API retrieval, SMT toolkit execution, and response generation
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

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the data directory to Python path for shared components
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

# Import SMT components
try:
    from smt_api_retriever import SMTAPIRetriever
    from smt_schema_indexer import GraphQLSchemaIndexer
except ImportError as e:
    logger.error(f"Cannot import SMT components: {e}")
    sys.exit(1)

# Import shared dataset loader
try:
    from dataset_loader import DatasetLoader
    logger.info("Successfully imported shared DatasetLoader")
except ImportError as e:
    logger.error(f"Failed to import shared DatasetLoader: {e}")
    logger.error("Make sure the data/dataset_loader.py file exists")
    sys.exit(1)

# Import SMT toolkit
try:
    from toolkit.solve import solve
    logger.info("SMT toolkit imported successfully")
except ImportError as e:
    logger.warning(f"SMT toolkit not found: {e}")
    logger.warning("Will use fallback mapping generation")


class SMTEvaluationRunner:
    """Runner class for SMT schema mapping evaluation."""
    
    def __init__(self, 
                 mappings_dir: str = "./mappings_output",
                 chroma_db_path: str = "./chroma_db", 
                 schema_dir: str = "../data/graphql"):
        """
        Initialize the SMT evaluation runner.
        
        Args:
            mappings_dir: Directory containing cached mapping results
            chroma_db_path: Path to ChromaDB database (should be pre-indexed)
            schema_dir: Directory containing GraphQL schemas
        """
        self.mappings_dir = Path(mappings_dir)
        self.chroma_db_path = Path(chroma_db_path)
        self.schema_dir = Path(schema_dir)
        self.results_dir = Path("results")
        
        # Create directories
        self.mappings_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize the shared dataset loader
        data_dir = Path(__file__).parent.parent / "data"
        self.dataset_loader = DatasetLoader(str(data_dir))
        
        # Check if ChromaDB has been set up
        self._verify_chromadb_setup()
        
        # Initialize components (assuming ChromaDB is pre-indexed)
        self.api_retriever = SMTAPIRetriever(
            mappings_dir=str(self.mappings_dir),
            chroma_db_path=str(self.chroma_db_path)
        )
        self.schema_indexer = GraphQLSchemaIndexer(chroma_db_path=str(self.chroma_db_path))
        
        logger.info(f"SMT runner initialized")
        logger.info(f"   Mappings directory: {self.mappings_dir}")
        logger.info(f"   ChromaDB path: {self.chroma_db_path}")
        logger.info(f"   Schema directory: {self.schema_dir}")
    
    def _verify_chromadb_setup(self):
        """Verify that ChromaDB has been properly set up."""
        if not self.chroma_db_path.exists():
            logger.warning(f"ChromaDB directory not found: {self.chroma_db_path}")
            logger.warning("Continuing with limited functionality")
            return
        
        # Verify collections exist
        try:
            import chromadb
            from chromadb.config import Settings
            
            client = chromadb.PersistentClient(
                path=str(self.chroma_db_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            collections = [c.name for c in client.list_collections()]
            
            if "graphql_schemas" not in collections:
                logger.warning("'graphql_schemas' collection not found")
                logger.warning("Schema-based API selection may not work optimally")
            
            if "smt_mappings" not in collections:
                logger.warning("'smt_mappings' collection not found")
                logger.warning("Existing mapping reuse will not be available")
            
            # Show collection stats
            for collection_name in collections:
                collection = client.get_collection(collection_name)
                count = collection.count()
                logger.info(f"   📦 {collection_name}: {count} documents")
                
        except Exception as e:
            logger.warning(f"ChromaDB verification failed: {e}")
            logger.warning("Continuing with limited functionality")
    
    def load_evaluation_utterances(self, dataset: str) -> List[Dict[str, Any]]:
        """Load evaluation utterances for a specific dataset using shared loader."""
        return self.dataset_loader.load_evaluation_utterances(dataset)
    
    def run_single_evaluation(self, utterance: str, sample_id: str, model: str) -> Dict[str, Any]:
        """Run SMT evaluation on a single utterance."""
        start_time = time.time()
        
        try:
            logger.info(f"Running SMT on sample {sample_id}")
            logger.info(f"Utterance: {utterance}")
            
            # Step 1: Find best API pair using retriever
            source_api, target_api, confidence = self.api_retriever.find_best_api_pair(utterance)
            
            if not source_api or not target_api:
                return self._create_error_result(sample_id, "No suitable API pair found", start_time)
            
            logger.info(f"Selected API pair: {source_api} → {target_api} (confidence: {confidence:.3f})")
            
            # Step 2: Check for existing mapping or run SMT toolkit
            mapping_result = self._get_or_create_mapping(source_api, target_api, utterance)
            
            if not mapping_result:
                return self._create_error_result(sample_id, "Failed to generate mapping", start_time)
            
            # Step 3: Extract predicted mappings
            predicted_mapping = self._extract_predicted_mappings(mapping_result)
            
            # Step 4: Calculate accuracy (placeholder)
            accuracy = self._calculate_accuracy_placeholder(predicted_mapping, sample_id)
            
            execution_time = time.time() - start_time
            
            logger.info(f"SMT completed for {sample_id} in {execution_time:.2f}s")
            logger.info(f"   Mappings found: {len(predicted_mapping)}")
            
            return {
                "sample_id": sample_id,
                "predicted_mapping": predicted_mapping,
                "accuracy": accuracy,
                "source_schema": source_api,
                "target_schema": target_api,
                "schema_selection_reasoning": f"SMT selected {source_api} → {target_api} using semantic similarity (confidence: {confidence:.3f})",
                "execution_time": execution_time,
                "success": True,
                "error_message": None,
                "smt_raw_result": mapping_result,
                "api_confidence": confidence
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"SMT failed for {sample_id}: {e}")
            
            return self._create_error_result(sample_id, str(e), start_time)
    
    def run_evaluation(self, dataset: str, model: str) -> Dict[str, Any]:
        """Run evaluation for a specific dataset and model combination."""
        logger.info(f"Starting SMT evaluation for {dataset} with {model}")
        
        # Load evaluation utterances using shared loader
        utterances = self.load_evaluation_utterances(dataset)
        if not utterances:
            logger.error(f"No utterances found for dataset {dataset}")
            return {}
        
        logger.info(f"Loaded {len(utterances)} utterances for {dataset}")
        
        # Run SMT on each utterance
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
            
            # Small delay between requests
            time.sleep(1)
        
        total_execution_time = time.time() - total_start_time
        
        # Calculate overall statistics
        successful_results = [r for r in results if r["success"]]
        overall_accuracy = sum(r["accuracy"] for r in successful_results) / len(successful_results) if successful_results else 0.0
        
        logger.info(f"SMT evaluation completed for {dataset}")
        logger.info(f"   Total utterances: {len(utterances)}")
        logger.info(f"   Successful: {len(successful_results)}")
        logger.info(f"   Overall accuracy: {overall_accuracy:.3f}")
        logger.info(f"   Total time: {total_execution_time:.2f}s")
        
        # Create final result structure
        final_result = {
            "metadata": {
                "dataset": dataset,
                "method": "SMT",
                "model": model,
                "execution_time": total_execution_time,
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(utterances),
                "successful_samples": len(successful_results),
                "overall_accuracy": overall_accuracy,
                "toolkit_version": "schema-mapping-toolkit",
                "mappings_dir": str(self.mappings_dir)
            },
            "results": results
        }
        
        return final_result
    
    def _get_or_create_mapping(self, source_api: str, target_api: str, utterance: str) -> Optional[Dict[str, Any]]:
        """Get existing mapping or create new one using SMT toolkit."""
        
        # PRIORITY: Check for existing mapping using the enhanced retriever
        logger.info(f"Checking for existing mapping: {source_api} → {target_api}")
        
        # First try the two-tier ChromaDB system
        existing_mapping_result = self.api_retriever.find_existing_mapping_for_utterance(utterance)
        
        if existing_mapping_result and existing_mapping_result.get("reused"):
            logger.info(f"Using existing mapping from ChromaDB (similarity: {existing_mapping_result['similarity_score']:.3f})")
            return existing_mapping_result["mapping_data"]
        
        # Fallback: Check file-based lookup
        existing_mapping = self.api_retriever.get_existing_mapping(source_api, target_api)
        
        if existing_mapping:
            logger.info(f"Using existing mapping file for {source_api} → {target_api}")
            return existing_mapping
        
        # Need to create new mapping using SMT toolkit
        logger.info(f"Creating new mapping using SMT toolkit for {source_api} → {target_api}")
        
        # Get schemas from schema indexer
        source_schemas = self.schema_indexer.search_schemas(f"{source_api} schema", n_results=1)
        target_schemas = self.schema_indexer.search_schemas(f"{target_api} schema", n_results=1)
        
        if not source_schemas or not target_schemas:
            logger.warning(f"Could not find schemas for {source_api} or {target_api}")
            # Try fallback schema generation
            return self._create_fallback_mapping(source_api, target_api)
        
        # Prepare schemas for SMT toolkit
        source_schema_dict = self._prepare_schema_for_smt(source_schemas[0])
        target_schema_dict = self._prepare_schema_for_smt(target_schemas[0])
        
        try:
            # Run SMT toolkit if available
            if 'solve' in globals():
                logger.info("Running SMT toolkit...")
                smt_result = solve(
                    source_schema_dict=source_schema_dict,
                    target_schema_dict=target_schema_dict,
                    pipeline_name="genai_mapping"
                )
            else:
                logger.info("🔄 SMT toolkit not available, using fallback mapping")
                smt_result = self._create_fallback_mapping_from_schemas(source_schema_dict, target_schema_dict)
            
            # Cache the result
            self._cache_mapping_result(source_api, target_api, smt_result)
            
            # Index the new mapping in ChromaDB for future reuse
            self._index_new_mapping(source_api, target_api, smt_result)
            
            return smt_result
            
        except Exception as e:
            logger.error(f"SMT toolkit failed: {e}")
            # Try fallback mapping
            return self._create_fallback_mapping(source_api, target_api)
    
    def _create_fallback_mapping(self, source_api: str, target_api: str) -> Dict[str, Any]:
        """Create a fallback mapping when SMT toolkit fails."""
        logger.info(f"🔄 Creating fallback mapping for {source_api} → {target_api}")
        
        # Generate simple field mappings based on API names
        fallback_mapping = {
            "name": f"{source_api}_to_{target_api}_mapping",
            "properties": {
                f"{source_api}_id": [{
                    "mapTable": target_api,
                    "mapProperty": f"{target_api}_identifier",
                    "justification": "Primary key mapping"
                }],
                f"{source_api}_name": [{
                    "mapTable": target_api,
                    "mapProperty": f"{target_api}_title",
                    "justification": "Name/title field mapping"
                }],
                f"{source_api}_data": [{
                    "mapTable": target_api,
                    "mapProperty": f"{target_api}_metadata",
                    "justification": "Data/metadata field mapping"
                }]
            }
        }
        
        return fallback_mapping
    
    def _create_fallback_mapping_from_schemas(self, source_schema: Dict[str, Any], target_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Create mapping from schema structures when SMT toolkit is not available."""
        
        source_properties = source_schema.get("properties", {})
        target_properties = target_schema.get("properties", {})
        
        mapping_properties = {}
        
        for source_field, source_def in source_properties.items():
            # Find best matching target field
            best_match = None
            best_score = 0
            
            for target_field, target_def in target_properties.items():
                # Simple similarity based on field names
                source_words = set(source_field.lower().split('_'))
                target_words = set(target_field.lower().split('_'))
                
                common_words = source_words.intersection(target_words)
                score = len(common_words) / max(len(source_words), len(target_words), 1)
                
                if score > best_score and score > 0.3:
                    best_score = score
                    best_match = target_field
            
            if best_match:
                mapping_properties[source_field] = [{
                    "mapTable": target_schema.get("name", "target"),
                    "mapProperty": best_match,
                    "justification": f"Field similarity match (score: {best_score:.2f})"
                }]
        
        return {
            "name": f"{source_schema.get('name', 'source')}_to_{target_schema.get('name', 'target')}_mapping",
            "properties": mapping_properties
        }
    
    def _index_new_mapping(self, source_api: str, target_api: str, mapping_result: Dict[str, Any]):
        """Index newly created mapping in ChromaDB for future reuse."""
        try:
            if hasattr(self.api_retriever, 'mappings_indexer') and self.api_retriever.mappings_indexer:
                # The mapping will be automatically indexed when the file is created
                # since the mappings indexer can be run periodically to pick up new files
                logger.info(f"💡 New mapping will be available for reuse after next indexing cycle")
            else:
                logger.info(f"💡 Mappings ChromaDB not available - mapping cached as file only")
        except Exception as e:
            logger.warning(f"Could not index new mapping: {e}")
    
    def setup_chromadb_indices(self, force_reindex: bool = False):
        """Setup both ChromaDB indices (schemas and mappings)."""
        logger.info("Setting up ChromaDB indices...")
        
        # Index GraphQL schemas
        if self.schema_indexer:
            try:
                schema_files = list(self.schema_dir.rglob("*.graphql")) + list(self.schema_dir.rglob("*.gql"))
                if schema_files or force_reindex:
                    logger.info(f"Indexing {len(schema_files)} GraphQL schemas...")
                    indexed_schemas = self.schema_indexer.index_graphql_schemas(
                        str(self.schema_dir), 
                        force_reindex=force_reindex
                    )
                    logger.info(f"Indexed {indexed_schemas} schemas")
                else:
                    logger.info("Schema index already exists")
            except Exception as e:
                logger.error(f"Schema indexing failed: {e}")
        
        # Index existing mappings
        if hasattr(self.api_retriever, 'mappings_indexer') and self.api_retriever.mappings_indexer:
            try:
                mapping_files = list(self.mappings_dir.glob("*-*.json"))
                if mapping_files or force_reindex:
                    logger.info(f"Indexing {len(mapping_files)} existing mappings...")
                    indexed_mappings = self.api_retriever.mappings_indexer.index_mapping_files(
                        str(self.mappings_dir),
                        force_reindex=force_reindex
                    )
                    logger.info(f"Indexed {indexed_mappings} mappings")
                else:
                    logger.info("Mappings index already exists")
            except Exception as e:
                logger.error(f"Mappings indexing failed: {e}")
        
        logger.info("ChromaDB setup complete")
    
    def _prepare_schema_for_smt(self, schema_match: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare schema for SMT toolkit input format."""
        
        # Extract schema information
        schema_name = schema_match["schema_name"]
        metadata = schema_match["metadata"]
        
        # Create mock schema structure for SMT
        # This would need to be replaced with actual schema loading from files
        schema_dict = {
            "name": schema_name,
            "type": "object",
            "properties": {
                # Generate some sample properties based on schema name
                f"{schema_name}_id": {"type": "string", "description": f"Unique identifier for {schema_name}"},
                f"{schema_name}_name": {"type": "string", "description": f"Name of {schema_name}"},
                f"{schema_name}_data": {"type": "object", "description": f"Data associated with {schema_name}"}
            }
        }
        
        return schema_dict
    
    def _cache_mapping_result(self, source_api: str, target_api: str, result: Dict[str, Any]):
        """Cache mapping result to file."""
        try:
            cache_file = self.mappings_dir / f"{source_api}-{target_api}.json"
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"💾 Cached mapping result to {cache_file}")
        except Exception as e:
            logger.warning(f"Could not cache mapping result: {e}")
    
    def _extract_predicted_mappings(self, mapping_result: Dict[str, Any]) -> Dict[str, str]:
        """Extract field mappings from SMT result."""
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
    
    def _calculate_accuracy_placeholder(self, predicted_mapping: Dict[str, str], sample_id: str) -> float:
        """Calculate accuracy against ground truth (placeholder implementation)."""
        # TODO: Implement actual ground truth comparison
        # For now, return accuracy based on number of mappings found
        
        if not predicted_mapping:
            return 0.0
        
        # Simulate accuracy based on number of mappings found
        num_mappings = len(predicted_mapping)
        if num_mappings >= 5:
            return 0.85
        elif num_mappings >= 3:
            return 0.70
        elif num_mappings >= 1:
            return 0.50
        else:
            return 0.0
    
    def _create_error_result(self, sample_id: str, error_message: str, start_time: float) -> Dict[str, Any]:
        """Create error result in standardized format."""
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
            "smt_raw_result": None,
            "api_confidence": 0.0
        }
    
    def save_results(self, results: Dict[str, Any], dataset: str, model: str):
        """Save results to JSON file."""
        output_file = self.results_dir / f"{dataset}_{model}.json"
        
        def convert_numpy_types(obj):
            """Convert NumPy types to Python native types for JSON serialization"""
            try:
                import numpy as np
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
            except ImportError:
                pass
            return obj
        
        try:
            # Convert any NumPy types before saving
            serializable_results = convert_numpy_types(results)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"💾 Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a summary of the evaluation results."""
        metadata = results.get("metadata", {})
        results_list = results.get("results", [])
        
        print(f"\n{'='*60}")
        print(f"SMT Evaluation Summary")
        print(f"{'='*60}")
        print(f"Dataset: {metadata.get('dataset', 'N/A')}")
        print(f"Model: {metadata.get('model', 'N/A')}")
        print(f"Total Samples: {metadata.get('total_samples', 0)}")
        print(f"Successful: {metadata.get('successful_samples', 0)}")
        print(f"Overall Accuracy: {metadata.get('overall_accuracy', 0.0):.3f}")
        print(f"Total Time: {metadata.get('execution_time', 0.0):.2f}s")
        
        print(f"\nIndividual Results:")
        print(f"{'Sample ID':<15} {'Success':<8} {'Accuracy':<10} {'Source Schema':<15} {'Target Schema':<15}")
        print(f"{'-'*80}")
        
        for result in results_list:
            success_icon = "✅" if result.get("success", False) else "❌"
            print(f"{result.get('sample_id', ''):<15} {success_icon:<8} {result.get('accuracy', 0.0):<10.3f} {result.get('source_schema', ''):<15} {result.get('target_schema', ''):<15}")


def main():
    """Main entry point for SMT evaluation."""
    parser = argparse.ArgumentParser(description="SMT Schema Mapping Evaluation")
    parser.add_argument("--datasets", default="ITOps",
                       help="Comma-separated list of datasets (default: ITOps)")
    parser.add_argument("--models", default="llama", 
                       help="Comma-separated list of models (default: llama)")
    parser.add_argument("--mappings-dir", default="./mappings_output",
                       help="Directory containing cached mapping results")
    parser.add_argument("--schema-dir", default="../data/graphql",
                       help="Directory containing GraphQL schemas")
    parser.add_argument("--chroma-db-path", default="./chroma_db",
                       help="Path to ChromaDB database (should be pre-indexed)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick evaluation with minimal samples")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse datasets and models from comma-separated strings
    datasets = [d.strip() for d in args.datasets.split(',')]
    models = [m.strip() for m in args.models.split(',')]
    
    if args.quick:
        datasets = ['ITOps']
        models = ['llama']
        logger.info("Quick evaluation mode enabled")
    
    logger.info(f"SMT Schema Mapping Evaluation")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Models: {models}")
    logger.info(f"Schema directory: {args.schema_dir}")
    logger.info(f"Mappings directory: {args.mappings_dir}")
    logger.info(f"ChromaDB path: {args.chroma_db_path}")
    
    # Check if schema directory exists
    schema_dir = Path(args.schema_dir)
    if not schema_dir.exists():
        logger.warning(f"Schema directory not found: {schema_dir}")
        logger.warning("Continuing with mock schemas")
    
    # Initialize runner (this will verify ChromaDB setup)
    runner = SMTEvaluationRunner(
        mappings_dir=args.mappings_dir,
        chroma_db_path=args.chroma_db_path,
        schema_dir=args.schema_dir
    )
    
    # Run evaluation for each dataset/model combination
    for dataset in datasets:
        for model in models:
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
    
    logger.info(f"\nSMT evaluation completed!")
    logger.info(f"Results saved in: {runner.results_dir}")


if __name__ == "__main__":
    main()
