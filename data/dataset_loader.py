#!/usr/bin/env python3
"""
Dataset Loader for Schema Mapping Evaluation
Centralized loading of evaluation utterances from various datasets
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and process GraphQL and REST API datasets."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.datasets = {
            'ITOps': 'itops_dataset.jsonl',
            'ITOpsGraph': 'itopsgraph_dataset.jsonl',  # Validated coverage mappings (521 samples)
            'NL2GQL': 'nl2gql_dataset.jsonl',
            'GraphQLRestBench': 'graphqlrestbench_dataset.jsonl',
            'OMOP_MIMIC': 'omop_mimic_dataset.jsonl',
            'OMOP_CMS': 'omop_cms_dataset.jsonl',
            'OMOP_Synthea': 'omop_synthea_dataset.jsonl'
        }
    
    def load_evaluation_utterances(self, dataset: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load evaluation utterances for a specific dataset.
        
        Args:
            dataset: Name of the dataset to load
            max_samples: Maximum number of samples to load (None for all)
        
        Returns:
            List of utterance dictionaries
        """
        
        utterances = []
        # Try to load from actual dataset file
        dataset_file = self.data_dir / self.datasets.get(dataset, f"{dataset.lower()}_dataset.jsonl")
        
        if dataset_file.exists():
            utterances = self._load_from_jsonl(dataset_file, dataset)
        else:
            logger.error(f"Dataset file not found: {dataset_file}")
            logger.error(f"Looking in directory: {self.data_dir}")
            
            # List available files for debugging
            if self.data_dir.exists():
                logger.error(f"Available files in {self.data_dir}:")
                for file in self.data_dir.glob("*dataset.jsonl"):
                    logger.error(f"  - {file.name}")
                for file in self.data_dir.glob("*.jsonl"):
                    logger.error(f"  - {file.name} (other JSONL)")
            else:
                logger.error(f"Data directory does not exist: {self.data_dir}")
            
        
        # Apply sample limit if specified
        if max_samples is not None and len(utterances) > max_samples:
            logger.info(f"Limiting {dataset} from {len(utterances)} to {max_samples} samples")
            utterances = utterances[:max_samples]
        
        return utterances
    
    def _load_from_jsonl(self, file_path: Path, dataset: str) -> List[Dict[str, Any]]:
        """Load utterances from JSONL file."""
        utterances = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                        
                    try:
                        data = json.loads(line)
                        
                        # Extract relevant fields based on dataset format
                        utterance_data = self._extract_utterance_data(data, dataset, i)
                        if utterance_data:
                            utterances.append(utterance_data)
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in line {i+1} of {file_path}: {e}")
                        logger.warning(f"Line content: {line[:100]}...")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {i+1} of {file_path}: {e}")
                        continue
            
            logger.info(f"Loaded {len(utterances)} utterances from {file_path}")
            return utterances
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []
    
    def _extract_utterance_data(self, data: Dict[str, Any], dataset: str, index: int) -> Optional[Dict[str, Any]]:
        """Extract utterance data from JSON object based on dataset format."""
        try:
            # Unified format detection - check for new standardized format first
            if "result" in data and "properties" in data["result"]:
                result = data["result"]
                source_schema = result.get("name", "")
                properties = result.get("properties", {})

                ground_truth_mapping = {}
                expected_target = ""
                for src_field, mappings in properties.items():
                    if mappings and len(mappings) > 0:
                        ground_truth_mapping[src_field] = mappings[0]["mapProperty"]
                        expected_target = mappings[0]["mapTable"]

                utterance_data = {
                    "sample_id": data.get("id", f"{dataset.lower()}_{index:03d}"),
                    "utterance": data.get("mapping_utterance", data.get("question", "")),
                    "expected_source": source_schema,
                    "expected_target": expected_target,
                    "ground_truth_mapping": ground_truth_mapping,
                    "metadata": data.get("metadata", {})
                }
                
                # Validate the extracted data
                if self.validate_dataset_format(dataset, utterance_data):
                    return utterance_data
                else:
                    logger.warning(f"Invalid format in sample {index+1}, skipping")
                    return None

            # Legacy format detection - check for old format
            if "mapping" in data and "source_schema" in data:
                utterance_data = {
                    "sample_id": data.get("id", f"{dataset.lower()}_{index:03d}"),
                    "utterance": data.get("utterance", ""),
                    "expected_source": data.get("source_schema", ""),
                    "expected_target": data.get("target_schema", ""),
                    "ground_truth_mapping": data.get("mapping", {}),
                    "metadata": data.get("metadata", {})
                }
                
                if self.validate_dataset_format(dataset, utterance_data):
                    return utterance_data
                else:
                    logger.warning(f"Invalid legacy format in sample {index+1}, skipping")
                    return None

            # Dataset-specific format detection
            if dataset == "ITOps":
                utterance_data = {
                    "sample_id": data.get("id", f"itops_{index:03d}"),
                    "utterance": data.get("utterance", data.get("query", "")),
                    "expected_source": data.get("source_schema", ""),
                    "expected_target": data.get("target_schema", ""),
                    "ground_truth_mapping": data.get("mapping", {}),
                    "metadata": data.get("metadata", {})
                }

            elif dataset.startswith("ITOpsGraph"):
                # ITOpsGraph format (utterance and attribute variants):
                # - id: sample identifier
                # - utterance: natural language or attribute name
                # - source_schema: source product name
                # - target_schema: target product name
                # - mapping: {source_field: target_field}
                utterance_data = {
                    "sample_id": data.get("id", f"itopsgraph_{index:03d}"),
                    "utterance": data.get("utterance", ""),
                    "expected_source": data.get("source_schema", ""),
                    "expected_target": data.get("target_schema", ""),
                    "ground_truth_mapping": data.get("mapping", {}),
                    "metadata": data.get("metadata", {})
                }

            elif dataset == "NL2GQL":
                # NL2GQL format:
                # - mapping_utterance: "map concert data to stadium data"
                # - result.name: source table (e.g., "concert")
                # - result.properties: ground truth mapping
                result = data.get("result", {})
                properties = result.get("properties", {})

                # Extract ground truth mapping: convert from result format to expected format
                # Input: {"concert_stadium_id": [{"mapTable": "stadium", "mapProperty": "stadium_stadium_id"}]}
                # Output: {"concert_stadium_id": "stadium_stadium_id"}
                ground_truth = {}
                for source_field, mappings in properties.items():
                    if mappings and len(mappings) > 0:
                        target_field = mappings[0].get("mapProperty", "")
                        if target_field:
                            ground_truth[source_field] = target_field

                # Extract source and target from mapping_utterance or result
                source_table = result.get("name", "")
                # Target can be extracted from first mapping's mapTable
                target_table = ""
                for mappings in properties.values():
                    if mappings and len(mappings) > 0:
                        target_table = mappings[0].get("mapTable", "")
                        break

                utterance_data = {
                    "sample_id": data.get("id", f"nl2gql_{index:03d}"),
                    "utterance": data.get("mapping_utterance", data.get("utterance", "")),
                    "expected_source": source_table,
                    "expected_target": target_table,
                    "ground_truth_mapping": ground_truth,
                    "question": data.get("question", "")
                }
            
            elif dataset == "GraphQLRestBench":
                utterance_data = {
                    "sample_id": data.get("id", f"graphqlrest_{index:03d}"),
                    "utterance": data.get("description", data.get("utterance", "")),
                    "expected_source": data.get("source_schema", "product"),
                    "expected_target": data.get("target_schema", "item"),
                    "ground_truth_mapping": data.get("mapping", {}),
                    "rest_endpoint": data.get("rest_endpoint", ""),
                    "graphql_schema": data.get("graphql_schema", "")
                }

            elif dataset.startswith("OMOP_"):
                # Handle OMOP datasets (OMOP_MIMIC, OMOP_CMS, OMOP_Synthea)
                source_schema = data.get("source_schema", {})
                target_schema = data.get("target_schema", {})
                utterance_data = {
                    "sample_id": data.get("id", f"{dataset.lower()}_{index:03d}"),
                    "utterance": data.get("utterance", ""),
                    "expected_source": source_schema.get("id", "") if isinstance(source_schema, dict) else str(source_schema),
                    "expected_target": target_schema.get("id", "") if isinstance(target_schema, dict) else str(target_schema),
                    "ground_truth_mapping": data.get("mapping", {}),
                    "source_properties": source_schema.get("properties", []) if isinstance(source_schema, dict) else [],
                    "target_properties": target_schema.get("properties", []) if isinstance(target_schema, dict) else [],
                    "metadata": data.get("metadata", {})
                }

            else:
                # Generic fallback format
                utterance_data = {
                    "sample_id": data.get("id", f"{dataset.lower()}_{index:03d}"),
                    "utterance": data.get("utterance", data.get("query", "")),
                    "expected_source": data.get("source_schema", ""),
                    "expected_target": data.get("target_schema", ""),
                    "ground_truth_mapping": data.get("mapping", {})
                }
            
            # Validate before returning
            if self.validate_dataset_format(dataset, utterance_data):
                return utterance_data
            else:
                logger.warning(f"Invalid format in sample {index+1}, skipping")
                return None

        except Exception as e:
            logger.warning(f"Error extracting utterance data from line {index+1}: {e}")
            return None
    
    def get_expected_samples_count(self, dataset_name: str) -> int:
        """Get expected number of samples for a dataset."""
        expected_counts = {
            'ITOps': 100,
            'ITOpsGraph': 521,  # Validated coverage mappings
            'NL2GQL': 50,
            'GraphQLRestBench': 75,
            'OMOP_MIMIC': 46,
            'OMOP_CMS': 38,
            'OMOP_Synthea': 12
        }
        return expected_counts.get(dataset_name, 0)
    
    def validate_dataset_format(self, dataset: str, sample_data: Dict[str, Any]) -> bool:
        """Validate that a sample has the expected format for the dataset."""
        required_fields = ["sample_id", "utterance", "expected_source", "expected_target"]
        
        for field in required_fields:
            if field not in sample_data:
                logger.error(f"Missing required field '{field}' in sample: {sample_data.get('sample_id', 'unknown')}")
                return False
            
            # Check for empty values
            if not sample_data[field] or sample_data[field] == "":
                logger.warning(f"Empty value for field '{field}' in sample: {sample_data.get('sample_id', 'unknown')}")
                # Don't fail validation for empty values, just warn
        
        return True
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets based on existing files."""
        available = []
        for dataset_name, filename in self.datasets.items():
            if (self.data_dir / filename).exists():
                available.append(dataset_name)
        
        # If no standard datasets found, check for any dataset files
        if not available:
            for file in self.data_dir.glob("*dataset.jsonl"):
                dataset_name = file.stem.replace("_dataset", "").title()
                available.append(dataset_name)
        
        return available
    
    def dataset_info(self, dataset: str) -> Dict[str, Any]:
        """Get information about a specific dataset."""
        dataset_file = self.data_dir / self.datasets.get(dataset, f"{dataset.lower()}_dataset.jsonl")
        
        if not dataset_file.exists():
            return {
                "name": dataset,
                "file": str(dataset_file),
                "exists": False,
                "sample_count": 0,
                "expected_count": self.get_expected_samples_count(dataset)
            }
        
        # Count lines in the file
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                sample_count = sum(1 for line in f if line.strip())
        except Exception as e:
            logger.error(f"Error counting samples in {dataset_file}: {e}")
            sample_count = 0
        
        return {
            "name": dataset,
            "file": str(dataset_file),
            "exists": True,
            "sample_count": sample_count,
            "expected_count": self.get_expected_samples_count(dataset)
        }

# Test function for standalone execution
if __name__ == "__main__":
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Test with different data directories
    test_dirs = ["../data", "./data", "."]
    
    for data_dir in test_dirs:
        print(f"\n=== Testing with data directory: {data_dir} ===")
        loader = DatasetLoader(data_dir)
        
        # Check available datasets
        available = loader.get_available_datasets()
        print(f"Available datasets: {available}")
        
        # Test loading ITOps dataset
        try:
            utterances = loader.load_evaluation_utterances("ITOps", max_samples=3)
            print(f"Loaded {len(utterances)} utterances")
            
            if utterances:
                print("Sample utterance:")
                for key, value in utterances[0].items():
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
        
        # Show dataset info
        info = loader.dataset_info("ITOps")
        print(f"Dataset info: {info}")
        
        if info["exists"]:
            break
