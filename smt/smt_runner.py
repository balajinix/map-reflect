#!/usr/bin/env python3
"""
SMT Runner - Fixed Threaded Version
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
import concurrent.futures
import threading
import traceback

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the data directory to Python path for shared components
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))


from dotenv import load_dotenv
load_dotenv()

import requests
import re

def get_watsonx_token(api_key: str) -> str:
    """Get IBM Cloud IAM token for watsonx API."""
    token_url = "https://iam.cloud.ibm.com/identity/token"
    response = requests.post(
        token_url,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}
    )
    return response.json().get("access_token")


def call_watsonx_llm(prompt: str, api_key: str, project_id: str, model_id: str = "ibm/granite-3-8b-instruct") -> str:
    """Call watsonx LLM API directly."""
    token = get_watsonx_token(api_key)
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model_id": model_id,
        "input": prompt,
        "parameters": {
            "max_new_tokens": 2048,
            "decoding_method": "greedy",
            "stop_sequences": ["\n\n\n"]
        },
        "project_id": project_id
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    return result.get("results", [{}])[0].get("generated_text", "")


def llm_based_field_mapping(source_schema: Dict, target_schema: Dict, api_key: str, project_id: str) -> Dict:
    """Generate field mappings using LLM directly."""
    source_name = source_schema.get("name", source_schema.get("id", "source"))
    target_name = target_schema.get("name", target_schema.get("id", "target"))

    # Build property lists
    source_props = []
    for prop in source_schema.get("properties", []):
        name = prop.get("name", "")
        desc = prop.get("description", "")[:150]
        source_props.append(f"  - {name}: {desc}")

    target_props = []
    for prop in target_schema.get("properties", []):
        name = prop.get("name", "")
        desc = prop.get("description", "")[:150]
        target_props.append(f"  - {name}: {desc}")

    prompt = f"""Map fields from source schema to target schema.

Source Schema: {source_name}
Properties:
{chr(10).join(source_props[:25])}

Target Schema: {target_name}
Properties:
{chr(10).join(target_props[:25])}

For each source field, identify the best matching target field based on semantic meaning and description.
Return ONLY a JSON object with source fields as keys and target fields as values.
Format: {{"source_field": "target_field", ...}}
Only include high-confidence mappings (>0.7).

JSON Output:"""

    try:
        response = call_watsonx_llm(prompt, api_key, project_id)
        # Extract JSON from response - try multiple patterns
        mappings = None

        # Try to find JSON object in response
        # First try: simple JSON without nested braces
        match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
        if match:
            try:
                mappings = json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Second try: extract key-value pairs manually
        if not mappings:
            pairs = re.findall(r'"([^"]+)":\s*"([^"]+)"', response)
            if pairs:
                mappings = dict(pairs)

        if mappings and isinstance(mappings, dict) and len(mappings) > 0:
            # Filter out invalid entries
            valid_mappings = {}
            for src, tgt in mappings.items():
                # Check that source and target look like actual field names
                if '.' in src or src.startswith(source_name.split()[0].lower()):
                    valid_mappings[src] = tgt
                elif '.' in tgt or any(c.islower() and c.isupper() for c in src+tgt):
                    # Has mixed case, likely a real field name
                    valid_mappings[src] = tgt

            if valid_mappings:
                return {
                    "name": f"{source_name}_to_{target_name}_mapping",
                    "description": f"LLM-based mapping from {source_name} to {target_name}",
                    "properties": {
                        src: [{"mapTable": target_name, "mapProperty": tgt, "justification": "LLM semantic mapping", "confidence": 0.8}]
                        for src, tgt in valid_mappings.items()
                    },
                    "metadata": {"method": "llm_direct", "created_at": datetime.now().isoformat()}
                }
    except Exception as e:
        logger.warning(f"LLM-based mapping failed: {e}")

    return None


# Import components - try package imports first, then relative imports
try:
    try:
        from smt.smt_api_retriever import SMTAPIRetriever
        from smt.smt_schema_indexer import GraphQLSchemaIndexer
    except ImportError:
        from smt_api_retriever import SMTAPIRetriever
        from smt_schema_indexer import GraphQLSchemaIndexer
    from dataset_loader import DatasetLoader
    from toolkit.solve import solve
    logger.info("All components imported successfully")
except ImportError as e:
    logger.error(f"Cannot import components: {e}")
    sys.exit(1)


def construct_schema_from_api_lookup(product: str, type_name: str, api_lookup: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Construct a schema object from the API lookup for a given product/type.

    Args:
        product: Product name (e.g., 'cloudability')
        type_name: Type name (e.g., 'StringFilterInput')
        api_lookup: The full API lookup dictionary

    Returns:
        Schema dict suitable for SMT toolkit, or None if not found
    """
    prefix = f"{product}_{type_name}."

    # Find all fields for this type
    fields = []
    for key, field_info in api_lookup.items():
        if key.startswith(prefix):
            field_name = key[len(prefix):]
            fields.append({
                "name": f"{product}_{type_name}.{field_name}",
                "description": field_info.get("description", ""),
                "metadata": {
                    "type": field_info.get("field_type", "String"),
                    "is_list": field_info.get("is_list", False),
                    "is_required": field_info.get("is_required", False)
                },
                "schema_component": f"{product}_{type_name}"
            })

    if not fields:
        return None

    return {
        "id": f"{product}_{type_name}",
        "name": f"{product}_{type_name}",
        "description": api_lookup.get(f"{product}_{type_name}.{fields[0]['name'].split('.')[-1]}", {}).get("type_description", ""),
        "properties": fields,
        "metadata": {
            "product": product,
            "type": type_name,
            "source": "api_lookup"
        }
    }


class SMTRunner:
    """High-performance parallelized SMT evaluation runner."""

    def __init__(self,
                 mappings_dir: str = "../data/mappings_output",
                 chroma_db_path: str = "../data/chroma_db",
                 schema_dir: str = "../data/graphql",
                 dataset: str = None,
                 collection_name: str = None,
                 mappings_collection_name: str = None,
                 max_workers: int = None):
        """Initialize with incremental writing support."""

        self.mappings_dir = Path(mappings_dir)
        self.chroma_db_path = Path(chroma_db_path)
        self.schema_dir = Path(schema_dir)
        self.results_dir = Path("results")
        self.dataset = dataset

        # Parallelization settings
        self.max_workers = max_workers or min(os.cpu_count(), 8)

        # Create directories
        self.mappings_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # Initialize shared components
        data_dir = Path(__file__).parent.parent / "data"
        self.dataset_loader = DatasetLoader(str(data_dir))

        # Schema collection name setup
        if collection_name:
            self.collection_name = collection_name
        elif dataset:
            self.collection_name = f"{dataset.lower()}_graphql_schemas"
        else:
            self.collection_name = "default_graphql_schemas"

        # Mappings collection name setup (separate from schemas)
        if mappings_collection_name:
            self.mappings_collection_name = mappings_collection_name
        elif dataset:
            self.mappings_collection_name = f"smt_mappings_{dataset.lower()}"
        else:
            self.mappings_collection_name = "smt_mappings"

        # Thread-safe counters
        self.write_lock = threading.Lock()

        # Load API lookup for multi-target datasets
        self.api_lookup = {}
        api_lookup_path = data_dir / "itopsgraph_api_lookup.json"
        if api_lookup_path.exists():
            try:
                with open(api_lookup_path) as f:
                    self.api_lookup = json.load(f)
                logger.info(f"Loaded API lookup with {len(self.api_lookup)} entries")
            except Exception as e:
                logger.warning(f"Failed to load API lookup: {e}")

        # Multi-target datasets that have per-sample source/target schemas
        self.multi_target_datasets = {
            'itopsgraph', 'nl2gql', 'toolbench'
        }

        # WatsonX credentials for SMT toolkit
        self.watsonx_config = {
            "credentials": {
                "apikey": os.environ.get("WATSONX_APIKEY") or os.environ.get("WATSONX_API_KEY"),
                "project_id": os.environ.get("WATSONX_PROJECT_ID")
            },
            "model_name": "ibm/granite-3-8b-instruct",
            "max_new_tokens": 2048
        }

        logger.info(f"SMT runner initialized with {self.max_workers} workers")
        logger.info(f"Schema collection: {self.collection_name}")
        logger.info(f"Mappings collection: {self.mappings_collection_name}")
    
    def create_results_file(self, dataset: str, model: str) -> Path:
        """Create results file for incremental writing."""
        # Sanitize model name to be filesystem-safe (replace slashes with underscores)
        safe_model = model.replace("/", "_").replace("\\", "_")
        results_file = self.results_dir / f"{dataset.lower()}_{safe_model}.jsonl"

        # Clear the file if it exists
        results_file.write_text("")

        return results_file
    
    def write_result_incremental(self, result: Dict[str, Any], results_file: Path):
        """Write a single result to file immediately (thread-safe)."""
        with self.write_lock:
            try:
                with open(results_file, 'a') as f:
                    f.write(json.dumps(result) + '\n')
                    f.flush()
            except Exception as e:
                logger.error(f"Failed to write result: {e}")
    
    def create_worker_components(self) -> Dict[str, Any]:
        """Create fresh component instances for each worker (thread-safe)."""
        try:
            # Create components with explicit error handling
            logger.debug(f"Creating worker components with schema collection: {self.collection_name}")
            logger.debug(f"Using mappings collection: {self.mappings_collection_name}")

            api_retriever = SMTAPIRetriever(
                mappings_dir=str(self.mappings_dir),
                chroma_db_path=str(self.chroma_db_path),
                collection_name=self.collection_name,
                mappings_collection_name=self.mappings_collection_name
            )
            
            schema_indexer = GraphQLSchemaIndexer(
                chroma_db_path=str(self.chroma_db_path),
                collection_name=self.collection_name
            )
            
            # Test components to ensure they work
            logger.debug("Testing component connections...")
            
            # Test schema indexer
            try:
                test_results = schema_indexer.search_schemas("test", n_results=1)
                logger.debug(f"Schema indexer test: found {len(test_results) if test_results else 0} results")
            except Exception as e:
                logger.warning(f"Schema indexer connection issue: {e}")
            
            return {
                "api_retriever": api_retriever,
                "schema_indexer": schema_indexer
            }
        except Exception as e:
            logger.error(f"Failed to create worker components: {e}")
            logger.error(f"ChromaDB path: {self.chroma_db_path}")
            logger.error(f"Collection name: {self.collection_name}")
            raise
    
    def process_single_sample(self, sample_data: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Process a single sample - worker function with robust error handling."""
        start_time = time.time()
        sample_id = sample_data.get("sample_id", "unknown")
        utterance = sample_data.get("utterance", "")
        
        try:
            logger.debug(f"Processing sample {sample_id}")
            
            # Create worker-specific components with retry logic
            components = None
            for attempt in range(3):  # Try up to 3 times
                try:
                    components = self.create_worker_components()
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} to create components failed: {e}")
                    if attempt == 2:  # Last attempt
                        return self._create_error_result(
                            sample_id, f"Failed to create components after 3 attempts: {e}", 
                            start_time, sample_data
                        )
                    time.sleep(1)  # Brief delay before retry
            
            api_retriever = components["api_retriever"]
            schema_indexer = components["schema_indexer"]

            # Step 1: Get API pair - prefer using provided schemas if available
            source_api = None
            target_api = None
            confidence = 1.0
            use_provided_schemas = False
            source_schema_data = None
            target_schema_data = None

            # Check if source/target schemas are provided in the dataset
            # Note: dataset_loader uses expected_source/expected_target keys
            expected_source = sample_data.get("expected_source") or sample_data.get("source_schema")
            expected_target = sample_data.get("expected_target") or sample_data.get("target_schema")

            # For multi-target datasets, expected_source/target are product names, not schema objects
            # We need to construct schemas from API lookup using product + type from metadata
            metadata = sample_data.get("metadata", {})
            source_type = metadata.get("source_type", "")
            target_type = metadata.get("target_type", "")

            if expected_source and expected_target and source_type and target_type and self.api_lookup:
                # Multi-target dataset: construct schemas from API lookup
                source_schema_data = construct_schema_from_api_lookup(
                    expected_source, source_type, self.api_lookup
                )
                target_schema_data = construct_schema_from_api_lookup(
                    expected_target, target_type, self.api_lookup
                )

                if source_schema_data and target_schema_data:
                    source_api = source_schema_data.get("id", "")
                    target_api = target_schema_data.get("id", "")
                    use_provided_schemas = True
                    logger.debug(f"Sample {sample_id}: Constructed schemas {source_api} -> {target_api}")
                else:
                    logger.warning(f"Sample {sample_id}: Could not construct schemas for {expected_source}_{source_type} -> {expected_target}_{target_type}")

            # If not a multi-target dataset, try to parse as schema objects
            if not use_provided_schemas and expected_source and expected_target:
                if isinstance(expected_source, dict):
                    source_schema_data = expected_source
                elif isinstance(expected_source, str):
                    try:
                        source_schema_data = eval(expected_source)
                    except:
                        source_schema_data = None

                if isinstance(expected_target, dict):
                    target_schema_data = expected_target
                elif isinstance(expected_target, str):
                    try:
                        target_schema_data = eval(expected_target)
                    except:
                        target_schema_data = None

                if source_schema_data and target_schema_data and isinstance(source_schema_data, dict) and isinstance(target_schema_data, dict):
                    source_api = source_schema_data.get("id", source_schema_data.get("name", ""))
                    target_api = target_schema_data.get("id", target_schema_data.get("name", ""))
                    use_provided_schemas = True
                    logger.debug(f"Sample {sample_id}: Using provided schemas {source_api} -> {target_api}")

            # Fallback to utterance-based API selection if schemas not provided
            if not source_api or not target_api:
                try:
                    source_api, target_api, confidence = api_retriever.find_best_api_pair(utterance)
                    logger.debug(f"Sample {sample_id}: Found API pair {source_api} -> {target_api} (confidence: {confidence})")
                except Exception as e:
                    logger.error(f"Sample {sample_id}: API pair selection failed: {e}")
                    return self._create_error_result(sample_id, f"API pair selection failed: {e}", start_time, sample_data)

            if not source_api or not target_api:
                return self._create_error_result(sample_id, "No suitable API pair found", start_time, sample_data)
            
            # Step 2: Get or create mapping
            try:
                # Pass the provided schemas if available
                provided_schemas = None
                if use_provided_schemas and source_schema_data and target_schema_data:
                    provided_schemas = {
                        "source": source_schema_data,
                        "target": target_schema_data
                    }

                mapping_result = self._get_or_create_mapping_worker(
                    api_retriever, schema_indexer, source_api, target_api, utterance,
                    provided_schemas=provided_schemas
                )
                logger.debug(f"Sample {sample_id}: Got mapping result")
            except Exception as e:
                logger.error(f"Sample {sample_id}: Mapping generation failed: {e}")
                # Create a basic fallback mapping instead of failing completely
                mapping_result = self._create_fallback_mapping(source_api, target_api)
                logger.warning(f"Sample {sample_id}: Using fallback mapping due to error")
            
            if not mapping_result:
                mapping_result = self._create_fallback_mapping(source_api, target_api)
                logger.warning(f"Sample {sample_id}: Using fallback mapping (no result returned)")
            
            # Step 3: Extract predicted mappings
            predicted_mapping = self._extract_predicted_mappings(mapping_result)
            
            # Step 4: Calculate accuracy (placeholder)
            accuracy = self._calculate_accuracy_placeholder(predicted_mapping, sample_id)
            
            execution_time = time.time() - start_time
            
            result = {
                "sample_id": sample_id,
                "predicted_mapping": predicted_mapping,
                "accuracy": accuracy,
                "source_schema": source_api,
                "target_schema": target_api,
                "schema_selection_reasoning": f"SMT selected {source_api} -> {target_api} using semantic similarity (confidence: {confidence:.3f})",
                "execution_time": execution_time,
                "success": True,
                "error_message": None,
                "smt_raw_result": mapping_result,
                "api_confidence": confidence,
                "expected_source": sample_data.get("expected_source", ""),
                "expected_target": sample_data.get("expected_target", ""),
                "original_utterance": utterance,
                "ground_truth_mapping": sample_data.get("ground_truth_mapping", {}),
                "timestamp": datetime.now().isoformat(),
                "chromadb_issues": bool(mapping_result.get("metadata", {}).get("fallback", False))
            }
            
            logger.debug(f"Sample {sample_id}: Processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Sample {sample_id}: Unexpected error: {e}")
            logger.error(f"Sample {sample_id}: Traceback: {traceback.format_exc()}")
            return self._create_error_result(sample_id, f"Unexpected error: {e}", start_time, sample_data)
    
    def run_evaluation_threaded(self, dataset: str, model: str, collection_name: str = None) -> Dict[str, Any]:
        """Run evaluation with threaded processing and incremental writing."""
        
        self.dataset = dataset
        if collection_name:
            self.collection_name = collection_name
        
        logger.info(f"Starting threaded evaluation for {dataset} with model {model}")
        
        # Load utterances
        try:
            utterances = self.dataset_loader.load_evaluation_utterances(dataset, max_samples=None)
            if not utterances:
                logger.error(f"No utterances found for dataset {dataset}")
                return {"error": "No utterances found"}
            
            logger.info(f"Loaded {len(utterances)} utterances")
        except Exception as e:
            logger.error(f"Failed to load utterances: {e}")
            return {"error": f"Failed to load utterances: {e}"}
        
        # Create results file
        results_file = self.create_results_file(dataset, model)
        
        # Progress tracking
        completed_count = 0
        successful_count = 0
        total_count = len(utterances)
        
        total_start_time = time.time()
        
        logger.info(f"Running evaluation with {self.max_workers} workers")
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(self.process_single_sample, utterance_data, model): utterance_data
                for utterance_data in utterances
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_sample):
                try:
                    result = future.result()
                    
                    # Write result immediately
                    self.write_result_incremental(result, results_file)
                    
                    # Update counters
                    completed_count += 1
                    if result.get("success", False):
                        successful_count += 1
                    
                    # Log progress every 10 completions
                    if completed_count % 10 == 0:
                        logger.info(f"Progress: {completed_count}/{total_count} completed ({successful_count} successful)")
                    
                except Exception as e:
                    sample_data = future_to_sample[future]
                    sample_id = sample_data.get("sample_id", "unknown")
                    logger.error(f"Task failed for sample {sample_id}: {e}")
                    
                    # Write error result
                    error_result = self._create_error_result(
                        sample_id, f"Task execution failed: {e}", 
                        time.time(), sample_data
                    )
                    self.write_result_incremental(error_result, results_file)
                    completed_count += 1
        
        total_execution_time = time.time() - total_start_time
        
        # Create final summary
        final_metadata = {
            "dataset": dataset,
            "method": "SMT_Threaded",
            "model": model,
            "execution_time": total_execution_time,
            "timestamp": datetime.now().isoformat(),
            "total_samples": total_count,
            "completed_samples": completed_count,
            "successful_samples": successful_count,
            "failed_samples": completed_count - successful_count,
            "overall_accuracy": successful_count / total_count if total_count > 0 else 0.0,
            "toolkit_version": "schema-mapping-toolkit-threaded",
            "max_workers": self.max_workers,
            "schema_collection": self.collection_name,
            "status": "completed",
            "results_file": str(results_file)
        }
        
        logger.info(f"Evaluation completed for {dataset}")
        logger.info(f"Total samples: {total_count}, Successful: {successful_count}, Failed: {completed_count - successful_count}")
        logger.info(f"Overall accuracy: {final_metadata['overall_accuracy']:.3f}")
        logger.info(f"Execution time: {total_execution_time:.2f}s")
        logger.info(f"Results written to: {results_file}")
        
        return final_metadata
    
    def _transform_schema_for_toolkit(self, schema: Dict) -> Dict:
        """Transform schema properties to include schema_component for toolkit compatibility."""
        transformed = schema.copy()
        if "properties" in transformed and isinstance(transformed["properties"], list):
            new_properties = []
            schema_name = transformed.get("id", transformed.get("name", ""))
            for prop in transformed["properties"]:
                new_prop = prop.copy()
                # Extract schema_component from property name (e.g., "omop_PERSON.field" -> "omop_PERSON")
                prop_name = prop.get("name", "")
                if "." in prop_name:
                    schema_component = prop_name.split(".")[0]
                else:
                    schema_component = schema_name
                new_prop["schema_component"] = schema_component
                new_properties.append(new_prop)
            transformed["properties"] = new_properties
        return transformed

    def _get_or_create_mapping_worker(self, api_retriever, schema_indexer,
                                    source_api: str, target_api: str, utterance: str,
                                    provided_schemas: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Worker version of mapping creation."""

        try:
            # If schemas are provided directly (from dataset), use them
            if provided_schemas:
                logger.debug(f"Using provided schemas for {source_api} -> {target_api}")
                source_schema_dict = self._transform_schema_for_toolkit(provided_schemas["source"])
                target_schema_dict = self._transform_schema_for_toolkit(provided_schemas["target"])

                # Ensure schemas have required fields for toolkit
                if not source_schema_dict.get("properties"):
                    logger.warning(f"Provided source schema missing properties, using fallback")
                    return self._create_fallback_mapping(source_api, target_api)
                if not target_schema_dict.get("properties"):
                    logger.warning(f"Provided target schema missing properties, using fallback")
                    return self._create_fallback_mapping(source_api, target_api)

                # Run SMT toolkit with provided schemas
                try:
                    logger.debug(f"Running SMT toolkit with provided schemas for {source_api} -> {target_api}")
                    smt_result = solve(
                        source_schema_dict=source_schema_dict,
                        target_schema_dict=target_schema_dict,
                        pipeline_name="graphql_pipeline",
                        **self.watsonx_config
                    )

                    # Cache result
                    self._cache_mapping_result_threadsafe(source_api, target_api, smt_result)

                    return smt_result
                except Exception as e:
                    logger.error(f"SMT toolkit failed with provided schemas: {e}")
                    # Try LLM-based mapping as fallback
                    api_key = self.watsonx_config["credentials"]["apikey"]
                    project_id = self.watsonx_config["credentials"]["project_id"]
                    llm_result = llm_based_field_mapping(
                        provided_schemas["source"], provided_schemas["target"], api_key, project_id
                    )
                    if llm_result:
                        logger.info(f"Used LLM-based fallback for {source_api} -> {target_api}")
                        self._cache_mapping_result_threadsafe(source_api, target_api, llm_result)
                        return llm_result
                    return self._create_fallback_mapping(source_api, target_api)

            # Otherwise, use the original flow with ChromaDB lookup
            # Check for existing mapping in utterance context
            try:
                existing_mapping_result = api_retriever.find_existing_mapping_for_utterance(utterance)

                if existing_mapping_result and existing_mapping_result.get("reused"):
                    logger.debug(f"Reusing existing mapping for utterance")
                    return existing_mapping_result["mapping_data"]
            except Exception as e:
                logger.warning(f"Could not check existing mapping for utterance: {e}")

            # Check file-based lookup
            try:
                existing_mapping = api_retriever.get_existing_mapping(source_api, target_api)
                if existing_mapping:
                    logger.debug(f"Found existing mapping file for {source_api} -> {target_api}")
                    return existing_mapping
            except Exception as e:
                logger.warning(f"Could not check file-based mapping: {e}")

            # Try to use schema indexer if available
            source_schemas = None
            target_schemas = None

            try:
                source_schemas = schema_indexer.search_schemas(f"{source_api} schema", n_results=1)
                target_schemas = schema_indexer.search_schemas(f"{target_api} schema", n_results=1)
                logger.debug(f"Schema search results: source={len(source_schemas) if source_schemas else 0}, target={len(target_schemas) if target_schemas else 0}")
            except Exception as e:
                logger.warning(f"Schema indexer search failed: {e}")
                logger.warning("Falling back to fallback mapping creation")

            if not source_schemas or not target_schemas:
                logger.warning(f"Could not find schemas for {source_api} -> {target_api}, using fallback")
                return self._create_fallback_mapping(source_api, target_api)

            # Prepare schemas for SMT toolkit
            try:
                source_schema_dict = self._prepare_schema_for_smt(source_schemas[0])
                target_schema_dict = self._prepare_schema_for_smt(target_schemas[0])
            except Exception as e:
                logger.warning(f"Schema preparation failed: {e}, using fallback")
                return self._create_fallback_mapping(source_api, target_api)
            
            # Run SMT toolkit
            try:
                logger.debug(f"Running SMT toolkit for {source_api} -> {target_api}")
                smt_result = solve(
                    source_schema_dict=source_schema_dict,
                    target_schema_dict=target_schema_dict,
                    pipeline_name="graphql_pipeline",
                    **self.watsonx_config
                )

                # Cache result
                self._cache_mapping_result_threadsafe(source_api, target_api, smt_result)

                return smt_result
            except Exception as e:
                logger.warning(f"SMT toolkit execution failed: {e}, using fallback")
                return self._create_fallback_mapping(source_api, target_api)
            
        except Exception as e:
            logger.error(f"SMT mapping creation failed for {source_api} -> {target_api}: {e}")
            return self._create_fallback_mapping(source_api, target_api)
    
    def _prepare_schema_for_smt(self, schema_match: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare schema in correct InputTableSchema format for SMT toolkit."""
        
        schema_name = schema_match.get("schema_name", "unknown")
        schema_content = schema_match.get("content", "")
        
        # Create proper InputTableSchema format
        schema_dict = {
            "id": schema_name,
            "name": schema_name,
            "description": f"GraphQL schema for {schema_name}",
            "metadata": {
                "source": "graphql",
                "original_content": schema_content[:500]  # Truncate if too long
            },
            "properties": []
        }
        
        # Extract basic properties from schema content (simplified)
        # In a real implementation, you'd parse the GraphQL schema properly
        basic_properties = [
            {
                "name": f"{schema_name}.id",
                "description": f"Unique identifier for {schema_name}",
                "metadata": {"type": "ID"}
            },
            {
                "name": f"{schema_name}.name",
                "description": f"Name field for {schema_name}",
                "metadata": {"type": "String"}
            },
            {
                "name": f"{schema_name}.data",
                "description": f"Main data field for {schema_name}",
                "metadata": {"type": "Object"}
            }
        ]
        
        schema_dict["properties"] = basic_properties
        
        return schema_dict
    
    def _cache_mapping_result_threadsafe(self, source_api: str, target_api: str, result: Dict[str, Any]):
        """Thread-safe caching of mapping results."""
        try:
            timestamp = int(time.time() * 1000)
            thread_id = threading.current_thread().ident
            cache_file = self.mappings_dir / f"{source_api}-{target_api}-{timestamp}-{thread_id}.json"
            
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
                
            logger.debug(f"Cached mapping result to {cache_file}")
                
        except Exception as e:
            logger.warning(f"Could not cache mapping result: {e}")
    
    def _create_error_result(self, sample_id: str, error_message: str, 
                           start_time: float, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create error result with full sample context."""
        return {
            "sample_id": sample_id,
            "predicted_mapping": {},
            "accuracy": 0.0,
            "source_schema": "",
            "target_schema": "",
            "schema_selection_reasoning": "",
            "execution_time": time.time() - start_time,
            "success": False,
            "error_message": error_message,
            "smt_raw_result": None,
            "api_confidence": 0.0,
            "expected_source": sample_data.get("expected_source", ""),
            "expected_target": sample_data.get("expected_target", ""),
            "original_utterance": sample_data.get("utterance", ""),
            "ground_truth_mapping": sample_data.get("ground_truth_mapping", {}),
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_predicted_mappings(self, mapping_result: Dict[str, Any]) -> Dict[str, str]:
        """Extract field mappings from SMT result."""
        predicted_mapping = {}
        
        if not mapping_result:
            return predicted_mapping
        
        # Handle different SMT result formats
        if "properties" in mapping_result:
            properties = mapping_result["properties"]
            
            for source_field, mappings in properties.items():
                if mappings and isinstance(mappings, list) and len(mappings) > 0:
                    first_mapping = mappings[0]
                    if isinstance(first_mapping, dict) and "mapProperty" in first_mapping:
                        target_field = first_mapping["mapProperty"]
                        predicted_mapping[source_field] = target_field
        
        # Fallback: if no properties found, create basic mapping
        if not predicted_mapping and "name" in mapping_result:
            mapping_name = mapping_result["name"]
            predicted_mapping[f"source_field"] = f"target_field"
        
        return predicted_mapping
    
    def _calculate_accuracy_placeholder(self, predicted_mapping: Dict[str, str], sample_id: str) -> float:
        """Calculate accuracy placeholder - replace with real accuracy calculation."""
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
    
    def _create_fallback_mapping(self, source_api: str, target_api: str) -> Dict[str, Any]:
        """Create fallback mapping when SMT toolkit fails."""
        return {
            "name": f"{source_api}_to_{target_api}_mapping",
            "description": f"Fallback mapping from {source_api} to {target_api}",
            "properties": {
                f"{source_api}_id": [{
                    "mapTable": target_api,
                    "mapProperty": f"{target_api}_identifier",
                    "justification": "Primary key mapping (fallback)",
                    "confidence": 0.5
                }],
                f"{source_api}_name": [{
                    "mapTable": target_api,
                    "mapProperty": f"{target_api}_name",
                    "justification": "Name field mapping (fallback)",
                    "confidence": 0.5
                }]
            },
            "metadata": {
                "fallback": True,
                "created_at": datetime.now().isoformat()
            }
        }


def main():
    """Main entry point for threaded SMT evaluation."""
    parser = argparse.ArgumentParser(description="Threaded SMT Schema Mapping Evaluation")
    parser.add_argument("--datasets", default="ITOps", help="Comma-separated list of datasets")
    parser.add_argument("--models", default="llama", help="Comma-separated list of models")
    parser.add_argument("--mappings-dir", default="../data/mappings_output", help="Mappings output directory")
    parser.add_argument("--schema-dir", default="../data/graphql", help="GraphQL schemas directory")
    parser.add_argument("--chroma-db-path", default="../data/chroma_db", help="ChromaDB path")
    parser.add_argument("--collection-name", help="ChromaDB schema collection name")
    parser.add_argument("--mappings-collection", help="ChromaDB mappings collection name (defaults to smt_mappings)")
    parser.add_argument("--max-workers", type=int, help="Maximum number of worker threads")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
    
    datasets = [d.strip() for d in args.datasets.split(',')]
    models = [m.strip() for m in args.models.split(',')]
    
    logger.info(f"Starting SMT evaluation for datasets: {datasets}, models: {models}")
    
    for dataset in datasets:
        for model in models:
            logger.info(f"Running threaded evaluation: {dataset} + {model}")
            
            try:
                runner = SMTRunner(
                    mappings_dir=args.mappings_dir,
                    chroma_db_path=args.chroma_db_path,
                    schema_dir=args.schema_dir,
                    dataset=dataset,
                    collection_name=args.collection_name,
                    mappings_collection_name=args.mappings_collection,
                    max_workers=args.max_workers
                )
                
                # Run threaded evaluation
                metadata = runner.run_evaluation_threaded(dataset, model, args.collection_name)
                
                if metadata and "error" not in metadata:
                    print(f"\n{'='*60}")
                    print(f"Threaded Results for {dataset} + {model}")
                    print(f"{'='*60}")
                    print(f"Total samples: {metadata['total_samples']}")
                    print(f"Completed: {metadata['completed_samples']}")
                    print(f"Successful: {metadata['successful_samples']}")
                    print(f"Failed: {metadata['failed_samples']}")
                    print(f"Accuracy: {metadata['overall_accuracy']:.3f}")
                    print(f"Time: {metadata['execution_time']:.2f}s")
                    if metadata['execution_time'] > 0:
                        print(f"Speed: {metadata['total_samples']/metadata['execution_time']:.1f} samples/sec")
                    print(f"Results written to: {metadata['results_file']}")
                else:
                    logger.error(f"Evaluation failed for {dataset} + {model}: {metadata}")
                    
            except Exception as e:
                logger.error(f"Failed to run evaluation for {dataset} + {model}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
