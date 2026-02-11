import os
import sys
import json
import requests
import re
import numpy as np
from pathlib import Path

# Increase recursion limit for large schemas (NL2GQL schemas can have 750+ properties)
sys.setrecursionlimit(3000)
from typing import Dict, Any, List, Optional, TypedDict, Annotated
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from sentence_transformers import SentenceTransformer
from utils.watsonx_utils import get_wxai_response
from utils.schema_indexer import build_schema_index
from toolkit.solve import solve
from utils.toolkit_utils import process

from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from litellm import completion
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger("solis_ai_api")

# Timeout and retry configuration
LLM_TIMEOUT = 120  # seconds
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 5  # seconds

import time

# Helper functions for LLM-based field mapping fallback
def get_watsonx_token(api_key: str, timeout: int = 30) -> str:
    """Get IBM Cloud IAM token for watsonx API."""
    token_url = "https://iam.cloud.ibm.com/identity/token"
    try:
        response = requests.post(
            token_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key},
            timeout=timeout
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.Timeout:
        logger.error(f"Token request timed out after {timeout}s")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Token request failed: {e}")
        raise

def call_watsonx_llm_direct(prompt: str, api_key: str, project_id: str, model_id: str = "ibm/granite-3-8b-instruct", timeout: int = LLM_TIMEOUT, max_retries: int = LLM_MAX_RETRIES) -> str:
    """Call watsonx LLM API directly with timeout and retry handling."""
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

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

    last_error = None
    for attempt in range(max_retries):
        try:
            # Get fresh token for each attempt
            token = get_watsonx_token(api_key)
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("results", [{}])[0].get("generated_text", "")

        except requests.exceptions.Timeout:
            last_error = f"Request timed out after {timeout}s"
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: {last_error}")
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: Request failed: {last_error}")

        if attempt < max_retries - 1:
            delay = LLM_RETRY_DELAY * (attempt + 1)  # Exponential backoff
            logger.info(f"Retrying in {delay}s...")
            time.sleep(delay)

    logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
    return ""

def llm_based_field_mapping(source_schema: Dict, target_schema: Dict, api_key: str, project_id: str) -> Dict:
    """Generate field mappings using LLM directly."""
    from datetime import datetime
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
        response = call_watsonx_llm_direct(prompt, api_key, project_id)
        # Extract JSON from response
        mappings = None

        # Try to find JSON object in response
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
            return {
                "name": f"{source_name}_to_{target_name}_mapping",
                "description": f"LLM-based mapping from {source_name} to {target_name}",
                "properties": {
                    src: [{"mapTable": target_name, "mapProperty": tgt, "justification": "LLM semantic mapping", "confidence": 0.8}]
                    for src, tgt in mappings.items()
                },
                "metadata": {"method": "llm_direct", "created_at": datetime.now().isoformat()}
            }
    except Exception as e:
        logger.warning(f"LLM-based mapping failed: {e}")

    return None

# LLM wrapper for IBM watsonx.ai via litellm
def call_llm(messages, model="watsonx/meta-llama/llama-3-3-70b-instruct", timeout: int = LLM_TIMEOUT, max_retries: int = LLM_MAX_RETRIES):
    """Call IBM watsonx.ai via litellm with timeout and retry handling."""
    last_error = None

    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=1000,
                timeout=timeout  # litellm supports timeout parameter
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = str(e)
            logger.warning(f"LLM call attempt {attempt + 1}/{max_retries} failed: {last_error}")

            if attempt < max_retries - 1:
                delay = LLM_RETRY_DELAY * (attempt + 1)
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)

    logger.error(f"All {max_retries} LLM call attempts failed. Last error: {last_error}")
    return f"Error calling LLM after {max_retries} attempts: {last_error}"

# State definition for LangGraph
class AgentState(TypedDict):
    messages: Annotated[List[Any], "The messages in the conversation using LangChain message types"]
    utterance: str
    schemas: List[str]
    schema_map: Dict[str, str]
    schema_similarities: Optional[List[tuple]]
    source_schema: Optional[str]
    target_schema: Optional[str]
    source_schema_content: Optional[str]
    target_schema_content: Optional[str]
    schema_selection_reasoning: Optional[str]
    mapping_result: Optional[Dict[str, Any]]
    error: Optional[str]

def split_camel_case(name: str):
    split = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
    return " ".join(split)

def find_mappings(source_graphql_schema, source_name, target_graphql_schema, target_name):
    """Generate schema mappings using the SMT toolkit with LLM fallback."""
    api_key = os.getenv("WATSONX_APIKEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")

    try:
        print(f"DEBUG: Starting mapping generation {source_name} -> {target_name}")
        print(f"DEBUG: SLATE_MODEL_PATH = {os.getenv('SLATE_MODEL_PATH', 'NOT SET')}")
        print(f"DEBUG: WATSONX_APIKEY = {'SET' if api_key else 'NOT SET'}")

        source_schema = process(source_graphql_schema, source_name)
        target_schema = process(target_graphql_schema, target_name)

        config = {
            "credentials": {
                "apikey": api_key,
                "project_id": project_id,
            },
            "model_name": "meta-llama/llama-3-3-70b-instruct",
            "max_new_tokens": 4096,
        }

        print(f"DEBUG: Running solve() with config")
        print(f"DEBUG: Source schema has {len(source_schema.get('properties', []))} properties")
        print(f"DEBUG: Target schema has {len(target_schema.get('properties', []))} properties")

        # Try SMT toolkit first
        try:
            result = solve(
                source_schema_dict=source_schema,
                target_schema_dict=target_schema,
                pipeline_name="graphql_pipeline",
                **config,
            )

            print(f"DEBUG: solve() returned: {type(result)}, keys: {result.keys() if isinstance(result, dict) else 'not dict'}")

            # Clean up result to only include non-empty mappings
            trimmed_result = {"name": result["name"], "properties": {}}
            for x in result["properties"]:
                if result["properties"][x] != []:
                    trimmed_result["properties"][x] = result["properties"][x]

            print(f"DEBUG: Trimmed result has {len(trimmed_result['properties'])} non-empty properties")

            # If toolkit returned results, use them
            if trimmed_result["properties"]:
                return trimmed_result

        except Exception as toolkit_error:
            print(f"DEBUG: SMT toolkit failed: {toolkit_error}")
            logger.warning(f"SMT toolkit failed: {toolkit_error}")

        # Fallback to LLM-based mapping
        print(f"DEBUG: Trying LLM-based field mapping fallback...")
        if api_key and project_id:
            llm_result = llm_based_field_mapping(source_schema, target_schema, api_key, project_id)
            if llm_result and llm_result.get("properties"):
                print(f"DEBUG: LLM fallback found {len(llm_result['properties'])} mappings")
                return llm_result

        print(f"DEBUG: No mappings found from either toolkit or LLM fallback")
        return {
            "name": f"{source_name}_to_{target_name}_mapping",
            "properties": {},
        }

    except Exception as e:
        print(f"DEBUG: Error in find_mappings: {e}")
        logger.error(f"Error in find_mappings: {e}")
        return {
            "name": f"{source_name}_to_{target_name}_mapping",
            "properties": {},
            "error": str(e)
        }

class GraphQLSchemaMappingAgent:
    """
    A GraphQL Mapping Agent using LangGraph for orchestration.
    """
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, base_schema_dir: str = "./data/graphql"):
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

        # Debug schema loading
        print(f"Loading schemas from: {base_schema_dir}")
        print(f"Directory exists: {os.path.exists(base_schema_dir)}")

        # Load schemas directly from base directory (not using utils.schema_indexer)
        self.schema_map = self._load_schemas_directly(base_schema_dir)

        self.graph = self._create_graph()

        # Debug: Print available schemas
        if self.schema_map:
            schema_names = [k.replace(".graphql", "") for k in self.schema_map.keys()]
            logger.info(f"Loaded {len(schema_names)} schemas: {schema_names[:10]}{'...' if len(schema_names) > 10 else ''}")
            print(f"Successfully loaded {len(schema_names)} schemas")
        else:
            logger.error("No schemas loaded - this will cause all evaluations to fail")
            print("ERROR: No schemas loaded")

    def _load_schemas_directly(self, base_schema_dir: str) -> Dict[str, str]:
        """Load schemas directly from base directory without subdirectory structure."""
        schema_map = {}
        try:
            schema_dir = Path(base_schema_dir)
            if schema_dir.exists():
                graphql_files = list(schema_dir.glob("*.graphql"))
                print(f"Found {len(graphql_files)} .graphql files")

                for schema_file in graphql_files:
                    try:
                        with open(schema_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Use just the filename as the key
                            schema_map[schema_file.name] = content
                            print(f"  Loaded: {schema_file.name}")
                    except Exception as e:
                        print(f"  Failed to load {schema_file.name}: {e}")

                print(f"Successfully loaded {len(schema_map)} schemas")
            else:
                print(f"Schema directory does not exist: {schema_dir}")
        except Exception as e:
            print(f"Error loading schemas: {e}")

        return schema_map

    def _create_graph(self):
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("find_schemas", self._find_relevant_schemas_node)
        workflow.add_node("select_schemas", self._select_schemas_node)
        workflow.add_node("generate_mappings", self._generate_mappings_node)

        # Add edges
        workflow.set_entry_point("find_schemas")
        workflow.add_edge("find_schemas", "select_schemas")
        workflow.add_edge("select_schemas", "generate_mappings")
        workflow.add_edge("generate_mappings", END)

        return workflow.compile()

    def _find_relevant_schemas_node(self, state: AgentState) -> AgentState:
        """Node to find relevant schemas based on user utterance"""
        try:
            utterance = state["utterance"]
            schemas, schema_map, similarities = self._find_relevant_schema(utterance)

            if not schemas:
                state["error"] = "No relevant schemas found"
                return state

            # Add system message about schema finding with detailed analysis
            similarity_info = "\n".join([f"  - {schema}: {score:.3f}" for schema, score in similarities[:5]])
            system_content = f"""Found {len(schemas)} relevant schemas for utterance: '{utterance}'

Top 5 schema similarities:
{similarity_info}

Total schemas analyzed: {len(similarities)}"""

            system_msg = SystemMessage(content=system_content)
            state["messages"].append(system_msg)
            state["schemas"] = schemas
            state["schema_map"] = schema_map
            state["schema_similarities"] = similarities

        except Exception as e:
            state["error"] = f"Error finding schemas: {str(e)}"
            logger.error(f"Error in _find_relevant_schemas_node: {e}")

        return state

    def _select_schemas_node(self, state: AgentState) -> AgentState:
        """Node to select source and target schemas using LLM"""
        try:
            if state.get("error"):
                return state

            utterance = state["utterance"]
            schemas = state["schemas"]
            similarities = state.get("schema_similarities", [])

            if not schemas:
                state["error"] = "No schemas available for selection"
                return state

            # Use the LLM wrapper for schema selection with detailed reasoning
            source_schema, target_schema, reasoning = self._select_schemas_with_llm(utterance, schemas, similarities)

            if not source_schema or not target_schema:
                state["error"] = "Failed to select valid schemas"
                return state

            state["source_schema"] = source_schema
            state["target_schema"] = target_schema
            state["source_schema_content"] = state["schema_map"].get(source_schema, "")
            state["target_schema_content"] = state["schema_map"].get(target_schema, "")
            state["schema_selection_reasoning"] = reasoning

            # Add detailed AI message about schema selection with reasoning
            ai_content = f"""Schema Selection Complete:

**Selected Schemas:**
- Source Schema: `{source_schema}`
- Target Schema: `{target_schema}`

**Selection Reasoning:**
{reasoning}

**Available Schemas Considered:**
{', '.join(schemas[:10])}{'...' if len(schemas) > 10 else ''}

**Similarity Scores for Selected Schemas:**"""

            # Add similarity scores for selected schemas
            for schema, score in similarities:
                if schema in [source_schema, target_schema]:
                    ai_content += f"\n- {schema}: {score:.3f}"

            ai_msg = AIMessage(content=ai_content)
            state["messages"].append(ai_msg)

        except Exception as e:
            state["error"] = f"Error selecting schemas: {str(e)}"
            logger.error(f"Error in _select_schemas_node: {e}")
            error_msg = AIMessage(content=f"Error during schema selection: {str(e)}")
            state["messages"].append(error_msg)

        return state

    def _generate_mappings_node(self, state: AgentState) -> AgentState:
        """Node to generate mappings between schemas"""
        try:
            if state.get("error"):
                return state

            source_schema = state["source_schema"]
            target_schema = state["target_schema"]
            source_content = state["source_schema_content"]
            target_content = state["target_schema_content"]

            if not source_content or not target_content:
                state["error"] = "Missing schema content for mapping generation"
                return state

            logger.info(f"Generating mappings: {source_schema} -> {target_schema}")

            mapping_result = find_mappings(
                source_content, source_schema,
                target_content, target_schema
            )

            if mapping_result.get("error"):
                state["error"] = f"Mapping generation failed: {mapping_result['error']}"
                return state

            state["mapping_result"] = mapping_result

            # Add detailed AI message about mapping completion
            ai_content = f"""Mapping Generation Complete

**Schemas Mapped:**
- From: `{source_schema}` 
- To: `{target_schema}`

**Mapping Summary:**
- Mapping Name: {mapping_result.get('name', 'N/A')}
- Properties Mapped: {len(mapping_result.get('properties', {}))}
- Non-empty Mappings: {len([k for k, v in mapping_result.get('properties', {}).items() if v])}

**Generated Mapping Structure:**
```json
{json.dumps(mapping_result, indent=2)[:500]}{'...' if len(json.dumps(mapping_result)) > 500 else ''}
```"""

            ai_msg = AIMessage(content=ai_content)
            state["messages"].append(ai_msg)

        except Exception as e:
            error_content = f"Error generating mappings: {str(e)}"
            state["error"] = error_content
            logger.error(f"Error in _generate_mappings_node: {e}")
            error_msg = AIMessage(content=error_content)
            state["messages"].append(error_msg)

        return state

    def _parse_utterance_for_products(self, utterance: str) -> tuple:
        """Parse utterance to extract source and target product names.

        Handles patterns like:
        - "map concert api key with turbo entity"
        - "map instana applications with concert tokens"

        Returns (source_product, target_product) or (None, None) if not found.
        """
        utterance_lower = utterance.lower().strip()

        # Known product names (ITOpsGraph + other datasets)
        products = ['concert', 'turbonomic', 'turbo', 'instana', 'aws', 'azure', 'gcp',
                    'cloudability', 'edge', 'hcp', 'hcp_terraform', 'infragraph',
                    'model', 'ns1', 'packer', 'resource', 'watsonx-orchestrate', 'base',
                    'omop', 'mimic', 'synthea', 'cms']

        # Pattern: "map X ... with Y ..." or "map X ... to Y ..."
        patterns = [
            r"map\s+(\w+)\s+.+?\s+with\s+(\w+)\s+",
            r"map\s+(\w+)\s+.+?\s+to\s+(\w+)\s+",
        ]

        for pattern in patterns:
            match = re.search(pattern, utterance_lower)
            if match:
                src_product, tgt_product = match.groups()
                if src_product in products and tgt_product in products:
                    logger.info(f"Extracted products from utterance: {src_product} -> {tgt_product}")
                    return src_product, tgt_product

        # Fallback: find any two products mentioned in order
        found_products = [p for p in products if p in utterance_lower]
        if len(found_products) >= 2:
            # Determine order based on position in utterance
            positions = [(p, utterance_lower.find(p)) for p in found_products]
            positions.sort(key=lambda x: x[1])
            logger.info(f"Extracted products (fallback): {positions[0][0]} -> {positions[1][0]}")
            return positions[0][0], positions[1][0]

        return None, None

    def _extract_type_hints_from_utterance(self, utterance: str) -> tuple[list, list]:
        """Extract source and target type hints from utterance.

        Parses patterns like:
        - "map turbonomic Action.actionID to concert PackageRecommendation.recommendation_id"
        - "map concert ApiKey with instana Application"

        Returns (source_type_hints, target_type_hints) where each is a list of type names.
        """
        utterance_lower = utterance.lower().strip()

        # Patterns to match: "map [product] [type.field] to/with [product] [type.field]"
        patterns = [
            r"map\s+(\w+)\s+(.+?)\s+to\s+(\w+)\s+(.+?)$",
            r"map\s+(\w+)\s+(.+?)\s+with\s+(\w+)\s+(.+?)$",
            r"find\s+(\w+)\s+fields?\s+equivalent\s+to\s+(\w+)\s+(.+?)$",
        ]

        source_hints = []
        target_hints = []

        for pattern in patterns:
            match = re.match(pattern, utterance_lower)
            if match:
                groups = match.groups()
                if len(groups) == 4:
                    src_product, src_rest, tgt_product, tgt_rest = groups
                    source_hints = self._extract_type_names(src_rest)
                    target_hints = self._extract_type_names(tgt_rest)
                elif len(groups) == 3:
                    # "find X fields equivalent to Y Z" pattern
                    tgt_product, src_product, src_rest = groups
                    source_hints = self._extract_type_names(src_rest)
                break

        logger.info(f"Extracted type hints: source={source_hints}, target={target_hints}")
        return source_hints, target_hints

    def _extract_type_names(self, text: str) -> list:
        """Extract type names from text like 'Action.actionID' or 'PackageRecommendation.recommendation_id'.

        Returns list of potential type names in original case.
        """
        type_names = []

        # Pattern 1: Type.field format (e.g., "Action.actionID")
        dot_pattern = r'(\w+)\.\w+'
        matches = re.findall(dot_pattern, text)
        for m in matches:
            # Convert to proper case (first letter uppercase)
            type_name = m[0].upper() + m[1:] if m else m
            if type_name and len(type_name) > 1:
                type_names.append(type_name)

        # Pattern 2: CamelCase or PascalCase words
        camel_pattern = r'\b([A-Z][a-zA-Z]+)\b'
        matches = re.findall(camel_pattern, text)
        for m in matches:
            if m not in type_names and len(m) > 1:
                type_names.append(m)

        # Pattern 3: Words that look like type names (capitalized in original utterance)
        # Handle lowercase input by looking for multi-word type names
        words = text.split()
        for word in words:
            # Remove field part if present
            word = word.split('.')[0]
            # Clean punctuation
            word = re.sub(r'[^\w]', '', word)
            if word and len(word) > 1:
                # Capitalize first letter
                type_name = word[0].upper() + word[1:]
                if type_name not in type_names:
                    type_names.append(type_name)

        return type_names

    def _find_relevant_schema(self, user_question: str, recall_limit=20) -> tuple[list, dict, list]:
        """Find relevant schemas using embeddings + type hint boosting and return similarities"""
        if not self.schema_map:
            logger.error("No schema map available")
            return [], {}, []

        # Extract products from utterance for filtering
        src_product, tgt_product = self._parse_utterance_for_products(user_question)

        # Extract type hints from utterance for boosting
        src_type_hints, tgt_type_hints = self._extract_type_hints_from_utterance(user_question)
        all_type_hints = src_type_hints + tgt_type_hints
        logger.info(f"Type hints for boosting: {all_type_hints}")

        all_keys = list(self.schema_map.keys())

        # Filter schemas by product if products were extracted
        if src_product and tgt_product:
            filtered_keys = [k for k in all_keys
                           if k.lower().startswith(src_product) or k.lower().startswith(tgt_product)]
            if filtered_keys:
                logger.info(f"Product filter: {len(all_keys)} -> {len(filtered_keys)} schemas (products: {src_product}, {tgt_product})")
                all_keys = filtered_keys
            else:
                logger.warning(f"Product filter returned no schemas, using all {len(all_keys)}")

        logger.info(f"Searching through {len(all_keys)} schemas")

        user_embedding = self.sentence_transformer.encode([user_question], show_progress_bar=False)[0]
        key_scores = []

        # Type hint boost value (large enough to override semantic similarity)
        TYPE_HINT_EXACT_BOOST = 0.5   # Exact type name match
        TYPE_HINT_PARTIAL_BOOST = 0.3  # Partial/substring match

        for k in all_keys:
            try:
                # Extract schema name from filename (no subdirectories)
                schema_name = k.replace(".graphql", "")
                doc = schema_name.replace("_", " ")
                doc = split_camel_case(doc)

                key_embedding = self.sentence_transformer.encode([doc], show_progress_bar=False)[0]
                cosine_sim = np.dot(user_embedding, key_embedding) / (
                    np.linalg.norm(user_embedding) * np.linalg.norm(key_embedding)
                )

                # Apply type hint boosting
                boost = 0.0
                schema_name_lower = schema_name.lower()
                # Extract the type part (after product_)
                schema_parts = schema_name.split('_', 1)
                schema_type = schema_parts[1] if len(schema_parts) > 1 else schema_name
                schema_type_lower = schema_type.lower()

                for hint in all_type_hints:
                    hint_lower = hint.lower()
                    # Exact type match (e.g., "PackageRecommendation" matches "concert_PackageRecommendation")
                    if schema_type_lower == hint_lower:
                        boost = max(boost, TYPE_HINT_EXACT_BOOST)
                        logger.debug(f"Exact type hint match: {schema_name} matches hint '{hint}'")
                    # Partial match (hint is substring of type or vice versa)
                    elif hint_lower in schema_type_lower or schema_type_lower in hint_lower:
                        boost = max(boost, TYPE_HINT_PARTIAL_BOOST)
                        logger.debug(f"Partial type hint match: {schema_name} ~ hint '{hint}'")

                final_score = cosine_sim + boost
                key_scores.append((schema_name, final_score, cosine_sim, boost))
            except Exception as e:
                logger.warning(f"Error processing schema {k}: {e}")
                continue

        # Sort by final score (similarity + boost)
        key_scores_sorted = sorted(key_scores, key=lambda x: x[1], reverse=True)

        # Log boosted schemas for debugging
        boosted = [(s, f, b) for s, f, c, b in key_scores_sorted[:20] if b > 0]
        if boosted:
            logger.info(f"Type-hint boosted schemas: {[(s, f'{f:.3f}', f'boost={b:.1f}') for s, f, b in boosted[:5]]}")

        # Return format: (schema_name, final_score) for compatibility
        key_scores_for_return = [(s, f) for s, f, c, b in key_scores_sorted]

        best_keys_with_scores = key_scores_for_return[:recall_limit]
        best_keys = [x[0] for x in best_keys_with_scores]

        # Build schema map for selected schemas
        best_schema_map = {}
        for k in all_keys:
            schema_name = k.replace(".graphql", "")
            if schema_name in best_keys:
                best_schema_map[schema_name] = self.schema_map[k]

        logger.info(f"Found {len(best_keys)} relevant schemas: {best_keys[:5]}{'...' if len(best_keys) > 5 else ''}")

        return best_keys, best_schema_map, key_scores_for_return

    def _extract_direction_from_utterance(self, utterance: str, schemas: List[str]) -> tuple[Optional[str], Optional[str]]:
        """Extract source and target schema names from utterance pattern.

        Pattern: "map [source] with/to [target]"
        The FIRST schema mentioned is SOURCE, SECOND is TARGET.

        Returns (source_schema, target_schema) or (None, None) if not extractable.
        """
        utterance_lower = utterance.lower().strip()

        # Pattern: "map X ... with/to Y ..."
        # Extract the two schema name parts
        patterns = [
            r"map\s+(.+?)\s+with\s+(.+?)$",
            r"map\s+(.+?)\s+to\s+(.+?)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, utterance_lower)
            if match:
                source_part = match.group(1).strip()
                target_part = match.group(2).strip()

                # Find best matching schema for each part
                source_schema = self._find_best_matching_schema(source_part, schemas)
                target_schema = self._find_best_matching_schema(target_part, schemas)

                if source_schema and target_schema and source_schema != target_schema:
                    logger.info(f"Extracted direction: '{source_part}' -> {source_schema}, '{target_part}' -> {target_schema}")
                    return source_schema, target_schema

        return None, None

    def _find_best_matching_schema(self, text_part: str, schemas: List[str]) -> Optional[str]:
        """Find the best matching schema name for a text part.

        Example: "omop person" should match "omop_PERSON"
        """
        text_normalized = text_part.lower().replace(" ", "_").replace("-", "_")

        # Try exact match first (case-insensitive)
        for schema in schemas:
            if schema.lower() == text_normalized:
                return schema

        # Try prefix match (e.g., "omop person" matches "omop_PERSON")
        for schema in schemas:
            schema_normalized = schema.lower()
            # Check if all words in the text part appear in the schema name
            words = text_part.lower().split()
            if all(word in schema_normalized for word in words):
                return schema

        # Try partial match - find schema that contains key parts
        best_match = None
        best_score = 0
        for schema in schemas:
            schema_lower = schema.lower()
            words = text_part.lower().split()
            matches = sum(1 for word in words if word in schema_lower)
            if matches > best_score:
                best_score = matches
                best_match = schema

        if best_score > 0:
            return best_match

        return None

    def _select_schemas_with_llm(self, utterance: str, schemas: List[str], similarities: List[tuple]) -> tuple[str, str, str]:
        """Select schemas using the LLM wrapper with detailed reasoning"""

        if len(schemas) < 2:
            if schemas:
                return schemas[0], schemas[0], "Only one schema available - using as both source and target"
            else:
                return "", "", "No schemas available for selection"

        # Prepare schema information with similarities
        schema_info = []
        similarity_dict = {schema: score for schema, score in similarities}

        for schema in schemas[:10]:  # Limit to top 10 for context
            score = similarity_dict.get(schema, 0.0)
            schema_info.append(f"- {schema} (similarity: {score:.3f})")

        # Try to extract direction from utterance pattern first
        extracted_source, extracted_target = self._extract_direction_from_utterance(utterance, schemas)
        direction_hint = ""
        if extracted_source and extracted_target:
            direction_hint = f"""
        IMPORTANT DIRECTION HINT:
        Based on the utterance pattern, the mapping direction should be:
        - Source (FROM): {extracted_source}
        - Target (TO): {extracted_target}
        Use these if the extracted schemas are in the available list.
        """

        schema_selection_prompt = f"""
        Given a user utterance and a list of schemas for mapping, choose a source and a target schema.

        User utterance: "{utterance}"

        CRITICAL DIRECTION RULE:
        The utterance follows the pattern "map [SOURCE] with/to [TARGET]".
        - The FIRST schema mentioned is the SOURCE (what you're mapping FROM)
        - The SECOND schema mentioned is the TARGET (what you're mapping TO)
        - Example: "map omop person with mimic patients" means:
          - Source = omop_PERSON (mapping FROM)
          - Target = mimic_PATIENTS (mapping TO)
        {direction_hint}

        Available schemas with similarity scores:
        {chr(10).join(schema_info)}

        Please analyze the utterance and select:
        1. A source schema (what you're mapping FROM) - this is the FIRST schema in the utterance
        2. A target schema (what you're mapping TO) - this is the SECOND schema in the utterance

        Your response should be a JSON with the following fields:
        1. "source_schema": the name of the source schema (FIRST mentioned)
        2. "target_schema": the name of the target schema (SECOND mentioned)
        3. "reasoning": detailed explanation of why these schemas were chosen

        Respond ONLY with valid JSON, no other text.
        """

        messages = [
            {"role": "system", "content": "You are a schema selection assistant. Analyze the user request carefully and provide detailed reasoning for your schema selection choices. Respond only with valid JSON."},
            {"role": "user", "content": schema_selection_prompt}
        ]

        try:
            response = call_llm(messages)

            # Clean the response
            response = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(response)

            source = data.get("source_schema", "")
            target = data.get("target_schema", "")
            reasoning = data.get("reasoning", "No reasoning provided")

            # Validate that selected schemas exist
            if source not in schemas:
                # Use extracted direction if available
                if extracted_source and extracted_source in schemas:
                    source = extracted_source
                    reasoning += f"\n(Note: Using extracted source from utterance pattern: {source})"
                else:
                    source = schemas[0]
                    reasoning += f"\n(Note: Original source selection not found, using {source})"
            if target not in schemas:
                # Use extracted direction if available
                if extracted_target and extracted_target in schemas:
                    target = extracted_target
                    reasoning += f"\n(Note: Using extracted target from utterance pattern: {target})"
                else:
                    target = schemas[1] if len(schemas) > 1 else schemas[0]
                    reasoning += f"\n(Note: Original target selection not found, using {target})"

            # Double-check: if we have extracted direction and LLM got it swapped, correct it
            if extracted_source and extracted_target:
                if source == extracted_target and target == extracted_source:
                    logger.warning(f"LLM swapped direction, correcting: {source}<->{target} to {extracted_source}->{extracted_target}")
                    source, target = extracted_source, extracted_target
                    reasoning += f"\n(Note: Direction corrected based on utterance pattern)"

            return source, target, reasoning

        except Exception as e:
            logger.warning(f"LLM schema selection failed: {e}")
            # Fallback: use extracted direction if available, otherwise similarity-based
            if extracted_source and extracted_target:
                fallback_reasoning = f"""Fallback selection using utterance pattern extraction (LLM error: {str(e)})

Extracted from utterance pattern "map [SOURCE] with/to [TARGET]":
- Source: {extracted_source}
- Target: {extracted_target}"""
                return extracted_source, extracted_target, fallback_reasoning
            else:
                fallback_reasoning = f"""Fallback selection due to LLM error: {str(e)}

Selected based on highest similarity scores:
- Source: {schemas[0]} (score: {similarity_dict.get(schemas[0], 0.0):.3f})
- Target: {schemas[1]} (score: {similarity_dict.get(schemas[1], 0.0):.3f})

This is an automatic fallback when LLM response cannot be parsed."""
                return schemas[0], schemas[1], fallback_reasoning

    def run_agent(self, utterance: str) -> Dict[str, Any]:
        """Run the agent with a user utterance"""
        initial_state = {
            "messages": [HumanMessage(content=utterance)],
            "utterance": utterance,
            "schemas": [],
            "schema_map": {},
            "schema_similarities": None,
            "source_schema": None,
            "target_schema": None,
            "source_schema_content": None,
            "target_schema_content": None,
            "schema_selection_reasoning": None,
            "mapping_result": None,
            "error": None
        }

        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)

            return {
                "mapping_result": final_state.get("mapping_result"),
                "source_schema": final_state.get("source_schema"),
                "target_schema": final_state.get("target_schema"),
                "schema_selection_reasoning": final_state.get("schema_selection_reasoning"),
                "error": final_state.get("error"),
                "messages": final_state.get("messages", []),
                "schema_similarities": final_state.get("schema_similarities", [])
            }
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                "mapping_result": None,
                "source_schema": "",
                "target_schema": "",
                "schema_selection_reasoning": "",
                "error": f"Agent execution failed: {str(e)}",
                "messages": [],
                "schema_similarities": []
            }

    def print_conversation(self, result: Dict[str, Any]):
        """Print the conversation in a readable format"""
        print("\n" + "="*80)
        print("CONVERSATION FLOW")
        print("="*80)

        for i, message in enumerate(result.get("messages", [])):
            if isinstance(message, HumanMessage):
                print(f"\nUSER MESSAGE {i+1}:")
                print("-" * 40)
                print(message.content)
            elif isinstance(message, SystemMessage):
                print(f"\nSYSTEM MESSAGE {i+1}:")
                print("-" * 40)
                print(message.content)
            elif isinstance(message, AIMessage):
                print(f"\nAI MESSAGE {i+1}:")
                print("-" * 40)
                print(message.content)
        
        print("\n" + "="*80)

# Tool wrapper for LangChain integration
class GraphQLMappingTool(BaseModel):
    """Tool for GraphQL schema mapping"""
    utterance: str = Field(description="The user's natural language query for schema mapping")

def create_graphql_mapping_tool(agent: GraphQLSchemaMappingAgent):
    """Create a LangChain tool from the agent"""
    def mapping_function(utterance: str) -> str:
        result = agent.run_agent(utterance)
        if result["error"]:
            return f"Error: {result['error']}"

        # Include reasoning in the tool output
        output = {
            "mapping_result": result["mapping_result"],
            "source_schema": result["source_schema"],
            "target_schema": result["target_schema"],
            "selection_reasoning": result["schema_selection_reasoning"]
        }
        return json.dumps(output, indent=2)

    return StructuredTool.from_function(
        func=mapping_function,
        name="graphql_mapping",
        description="Map between GraphQL schemas based on natural language description",
        args_schema=GraphQLMappingTool
    )
