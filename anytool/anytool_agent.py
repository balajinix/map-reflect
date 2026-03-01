import os
os.environ['LITELLM_LOG'] = 'ERROR'
os.environ['LITELLM_DROP_PARAMS'] = 'True'
import json
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator
import re
from dotenv import load_dotenv
load_dotenv()

# Try to import WatsonX SDK, fall back to litellm
HAS_WATSONX_SDK = False
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
    from ibm_watsonx_ai.foundation_models import ModelInference
    HAS_WATSONX_SDK = True
except ImportError:
    print("ibm_watsonx_ai not available, using litellm")

# Import litellm as fallback
try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    print("WARNING: litellm not available")


class ToolConverter:
    @staticmethod
    def to_langchain_tool(fn):
        langchain_tool = StructuredTool.from_function(func=fn)
        return langchain_tool


hierarchy = {}
MODEL = "watsonx/meta-llama/llama-3-3-70b-instruct"

# Cached WatsonX model instances (avoid re-creating on every call)
_model_cache: Dict[str, Any] = {}


# Tool functions
def get_properties_for_type(product, graphql_type):
    """Get all properties with their descriptions and data types for a specific product and GraphQL type."""
    try:
        product_data = hierarchy.get(product, {})
        if not isinstance(product_data, dict):
            print(f"Warning: Product '{product}' data is not a dictionary")
            return {'type_name': graphql_type, 'properties': []}

        properties = product_data.get(graphql_type, [])
        if not isinstance(properties, list):
            print(f"Warning: Properties for type '{graphql_type}' is not a list")
            return {'type_name': graphql_type, 'properties': []}

        return {
            'type_name': graphql_type,
            'properties': [
                {
                    'name': prop.get('name', 'unknown'),
                    'description': prop.get('description', 'No description available'),
                    'data_type': prop.get('type', 'unknown')
                }
                for prop in properties if isinstance(prop, dict)
            ]
        }
    except Exception as e:
        print(f"Error in get_properties_for_type: {str(e)}")
        return {'type_name': graphql_type, 'properties': []}


def get_types_for_product(product):
    """Get all GraphQL types with their descriptions for a specific product."""
    try:
        product_types = hierarchy.get(product, {})

        if not isinstance(product_types, dict):
            print(f"Warning: Product '{product}' data is not a dictionary: {type(product_types)}")
            return []

        result = []
        for type_name, properties in product_types.items():
            if not isinstance(properties, list):
                print(f"Warning: Properties for type '{type_name}' is not a list: {type(properties)}")
                continue

            property_names = [prop.get('name', 'unknown') for prop in properties if isinstance(prop, dict)]
            description = f'GraphQL type containing {len(properties)} properties: {", ".join(property_names)}'

            result.append({
                'type_name': type_name,
                'description': description,
                'property_count': len(properties)
            })

        return result
    except Exception as e:
        print(f"Error in get_types_for_product: {str(e)}")
        return []


def get_products():
    """Get all products."""
    try:
        if not isinstance(hierarchy, dict):
            print(f"Warning: Hierarchy is not a dictionary: {type(hierarchy)}")
            return []

        # FIX BUG 8: Only filter Mutation keys for GraphQL schemas; preserve all product keys.
        # Mutation is a GraphQL convention not a product name, but we should only exclude
        # it when it appears at the top level as a schema artefact (i.e. it has no subkeys
        # that look like real product types).
        products = []
        for key in hierarchy.keys():
            if key.startswith("Mutation"):
                # Include anyway if the value has product-like structure
                val = hierarchy[key]
                if isinstance(val, dict) and val:
                    products.append(key)
                # else skip it (bare Mutation stub with no content)
            else:
                products.append(key)
        print(products)
        return products
    except Exception as e:
        print(f"Error in get_products: {str(e)}")
        return []


# Create LangChain tools
get_products_tool = ToolConverter.to_langchain_tool(get_products)
get_types_for_product_tool = ToolConverter.to_langchain_tool(get_types_for_product)
get_properties_for_type_tool = ToolConverter.to_langchain_tool(get_properties_for_type)

tools = [get_products_tool, get_types_for_product_tool, get_properties_for_type_tool]
tool_node = ToolNode(tools)


# FIX BUG 1: AgentState now tracks source and target separately.
class AgentState(TypedDict):
    messages: Annotated[List[Any], operator.add]
    user_query: str
    current_step: str
    available_products: List[str]
    # Source schema fields
    selected_source_product: str
    available_source_types: List[Dict]
    selected_source_type: str
    source_properties: Dict
    # Target schema fields
    selected_target_product: str
    available_target_types: List[Dict]
    selected_target_type: str
    target_properties: Dict
    # Output
    final_mappings: List[Dict]
    reasoning: str


def format_prompt_for_model(messages, model_id):
    """Format messages based on model-specific chat template."""
    model_lower = model_id.lower()

    if "granite-4" in model_lower or "granite4" in model_lower:
        prompt_parts = []
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')
            prompt_parts.append(f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n")
        prompt_parts.append("<|start_of_role|>assistant<|end_of_role|>")
        return "".join(prompt_parts), ["<|end_of_text|>"]

    elif "mistral" in model_lower:
        system_msg = ""
        user_content = []
        for m in messages:
            if m.get('role') == 'system':
                system_msg = m.get('content', '')
            elif m.get('role') == 'user':
                user_content.append(m.get('content', ''))

        combined_user = "\n\n".join(user_content)
        if system_msg:
            prompt = f"[INST] {system_msg}\n\n{combined_user} [/INST]"
        else:
            prompt = f"[INST] {combined_user} [/INST]"
        return prompt, ["[/INST]", "</s>"]

    elif "llama" in model_lower:
        prompt_parts = ["<|begin_of_text|>"]
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')
            prompt_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(prompt_parts), ["<|eot_id|>", "<|end_of_text|>"]

    elif "granite-3" in model_lower or "granite3" in model_lower:
        prompt_parts = []
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')
            prompt_parts.append(f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n")
        prompt_parts.append("<|start_of_role|>assistant<|end_of_role|>")
        return "".join(prompt_parts), ["<|end_of_text|>"]

    else:
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(content)
        return "\n\n".join(prompt_parts), ["<|endoftext|>"]


def call_llm(messages, model=None):
    """Call IBM watsonx.ai using native SDK or litellm fallback."""
    if model is None:
        print("################ Why is model None? ###############")
        model = MODEL

    if not HAS_WATSONX_SDK:
        if not HAS_LITELLM:
            raise RuntimeError("Neither ibm_watsonx_ai nor litellm available")
        return call_llm_litellm(messages, model)

    try:
        api_key = os.getenv("WATSONX_API_KEY") or os.getenv("WATSONX_APIKEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

        if not api_key or not project_id:
            raise RuntimeError("WATSONX_API_KEY and WATSONX_PROJECT_ID required")

        model_id = model.replace("watsonx/", "")
        prompt, stop_sequences = format_prompt_for_model(messages, model_id)

        # Reuse cached model object to avoid re-creating credentials on every call
        cache_key = f"{model_id}:{url}"
        if cache_key not in _model_cache:
            parameters = {
                GenParams.MAX_NEW_TOKENS: 2000,
                GenParams.TEMPERATURE: 0.1,
                GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
                GenParams.STOP_SEQUENCES: stop_sequences,
            }
            credentials = Credentials(url=url, api_key=api_key)
            _model_cache[cache_key] = ModelInference(
                model_id=model_id,
                params=parameters,
                credentials=credentials,
                project_id=project_id
            )

        model_obj = _model_cache[cache_key]
        response = model_obj.generate_text(prompt=prompt)
        return response[0] if isinstance(response, list) else response

    except Exception as e:
        error_str = str(e).lower()
        critical_patterns = ["authentication", "unauthorized", "api key", "project", "provider not", "exceeded limit"]

        if any(p in error_str for p in critical_patterns):
            print(f"CRITICAL LLM error - stopping execution: {e}")
            raise RuntimeError(f"Critical LLM error: {e}") from e

        print(f"LLM Error: {str(e)}")
        return f"Error calling LLM: {str(e)}"


def call_llm_litellm(messages, model):
    """Call LLM using litellm (fallback when WatsonX SDK unavailable)."""
    try:
        if not model.startswith("watsonx/"):
            model = f"watsonx/{model}"

        litellm_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                litellm_messages.append(msg)
            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                litellm_messages.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg, SystemMessage):
                litellm_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                litellm_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                litellm_messages.append({"role": "assistant", "content": msg.content})

        response = litellm.completion(
            model=model,
            messages=litellm_messages,
            temperature=0.1,
            max_tokens=2000,
        )

        return response.choices[0].message.content

    except Exception as e:
        error_str = str(e).lower()
        critical_patterns = ["authentication", "unauthorized", "api key", "project", "provider not", "exceeded limit"]

        if any(p in error_str for p in critical_patterns):
            print(f"CRITICAL LLM error - stopping execution: {e}")
            raise RuntimeError(f"Critical LLM error: {e}") from e

        print(f"LLM Error (litellm): {str(e)}")
        return f"Error calling LLM: {str(e)}"


# ── Node functions ────────────────────────────────────────────────────────────

def get_products_node(state: AgentState):
    """Step 1: Get all available products."""
    print("Step 1: Getting available products...")

    products = get_products()
    state["available_products"] = products
    state["current_step"] = "product_selection"

    products_msg = f"Available products: {', '.join(products)}"
    print(products_msg)
    state["messages"].append(AIMessage(content=f"Found {len(products)} products: {products_msg}"))
    return state


def _extract_products_from_utterance(utterance: str, available_products: list) -> tuple:
    """Extract source and target products from utterance.

    Returns (source_product, target_product) or (None, None) if not found.
    """
    utterance_lower = utterance.lower().strip()
    products_lower = [p.lower() for p in available_products]

    patterns = [
        r"map\s+(\w+)\s+.+?\s+with\s+(\w+)\s+",
        r"map\s+(\w+)\s+.+?\s+to\s+(\w+)\s+",
    ]

    for pattern in patterns:
        match = re.search(pattern, utterance_lower)
        if match:
            src, tgt = match.groups()
            if src in products_lower and tgt in products_lower:
                src_idx = products_lower.index(src)
                tgt_idx = products_lower.index(tgt)
                print(f"Extracted products from utterance: {available_products[src_idx]} -> {available_products[tgt_idx]}")
                return available_products[src_idx], available_products[tgt_idx]

    # Fallback: find products mentioned in utterance by position
    found = []
    for i, p in enumerate(products_lower):
        pos = utterance_lower.find(p)
        if pos != -1:
            found.append((available_products[i], pos))

    if len(found) >= 2:
        found.sort(key=lambda x: x[1])
        print(f"Extracted products (fallback): {found[0][0]} -> {found[1][0]}")
        return found[0][0], found[1][0]

    return None, None


def select_product_node(state: AgentState):
    """Step 2: Identify both source and target products from the utterance."""
    print("Step 2: Selecting source and target products...")

    user_query = state["user_query"]
    available_products = state["available_products"]

    src_product, tgt_product = _extract_products_from_utterance(user_query, available_products)
    if src_product and tgt_product:
        state["selected_source_product"] = src_product
        state["selected_target_product"] = tgt_product
        reasoning = f"Extracted source='{src_product}', target='{tgt_product}' from query"
        print(reasoning)
        state["messages"].append(AIMessage(content=reasoning))
        return state

    # LLM fallback: ask for both products in one call
    system_prompt = f"""You are identifying the SOURCE and TARGET products for a schema mapping task.

Available products: {available_products}

RULES:
1. Output ONLY a JSON object: {{"source": "<product>", "target": "<product>"}}
2. Both values must be exact names from the available products list
3. No explanation, no extra text

Example: {{"source": "instana", "target": "concert"}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {user_query}\n\nIdentify source and target products:"}
    ]

    raw_response = call_llm(messages, model=MODEL).strip()
    print(f"LLM response: '{raw_response[:200]}'")

    src_product = None
    tgt_product = None

    try:
        json_match = re.search(r'\{[^}]+\}', raw_response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            src_candidate = str(parsed.get("source", "")).strip()
            tgt_candidate = str(parsed.get("target", "")).strip()

            products_lower = [p.lower() for p in available_products]
            if src_candidate.lower() in products_lower:
                src_product = available_products[products_lower.index(src_candidate.lower())]
            if tgt_candidate.lower() in products_lower:
                tgt_product = available_products[products_lower.index(tgt_candidate.lower())]
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        print(f"Could not parse product JSON: {e}")

    # Position-based fallback within the LLM response
    if not src_product or not tgt_product:
        response_lower = raw_response.lower()
        positions = {}
        for p in available_products:
            pos = response_lower.find(p.lower())
            if pos != -1:
                positions[p] = pos
        if len(positions) >= 2:
            sorted_prods = sorted(positions.items(), key=lambda x: x[1])
            src_product = src_product or sorted_prods[0][0]
            tgt_product = tgt_product or sorted_prods[1][0]

    # Final fallback
    if not src_product:
        src_product = available_products[0]
        print("Warning: Could not identify source product - using first available")
    if not tgt_product:
        tgt_product = available_products[1] if len(available_products) > 1 else available_products[0]
        print("Warning: Could not identify target product - using second available")

    state["selected_source_product"] = src_product
    state["selected_target_product"] = tgt_product
    reasoning = f"Selected source='{src_product}', target='{tgt_product}'"
    print(reasoning)
    state["messages"].append(AIMessage(content=reasoning))
    return state


def _select_type_with_llm(user_query: str, product: str, available_types: list, role: str) -> str:
    """Use LLM to select the most relevant type for a given product and role (source/target)."""
    if not available_types:
        return ""

    type_descriptions = []
    valid_types = []
    for type_info in available_types:
        if isinstance(type_info, dict) and 'type_name' in type_info:
            type_descriptions.append(
                f"- {type_info['type_name']}: {type_info.get('description', 'No description')} "
                f"({type_info.get('property_count', 0)} properties)"
            )
            valid_types.append(type_info['type_name'])

    if not valid_types:
        return ""

    system_prompt = f"""You are selecting the most relevant GraphQL type from the {role} schema '{product}'.

RULES:
1. Output ONLY the exact type name from the available types
2. No explanation, no quotes, no punctuation
3. Choose the type most relevant to the mapping task"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"Mapping query: {user_query}\n\n"
            f"Available types in '{product}':\n" + "\n".join(type_descriptions) +
            "\n\nOutput the single most relevant type name:"
        )}
    ]

    raw_response = call_llm(messages, model=MODEL).strip()
    print(f"LLM {role} type response: '{raw_response[:150]}'")

    # Exact match first
    if raw_response in valid_types:
        return raw_response

    # Find first mentioned
    first_positions = {}
    for type_name in valid_types:
        pos = raw_response.find(type_name)
        if pos != -1:
            first_positions[type_name] = pos

    if first_positions:
        selected = min(first_positions, key=first_positions.get)
        print(f"Extracted {role} type '{selected}' from response")
        return selected

    # Fallback
    print(f"Warning: Could not parse {role} type - using first available")
    return valid_types[0]


def get_source_types_node(state: AgentState):
    """Step 3a: Get available types for source product."""
    print("Step 3a: Getting types for source product...")
    product = state["selected_source_product"]
    try:
        types_data = get_types_for_product(product)
        if not isinstance(types_data, list):
            types_data = []
        state["available_source_types"] = types_data
        print(f"Found {len(types_data)} source types in '{product}'")
        state["messages"].append(AIMessage(content=f"Source product '{product}': {len(types_data)} types"))
    except Exception as e:
        print(f"Error in get_source_types_node: {str(e)}")
        state["available_source_types"] = []
    state["current_step"] = "source_type_selection"
    return state


def select_source_type_node(state: AgentState):
    """Step 4a: Select the most relevant source type."""
    print("Step 4a: Selecting source type...")
    selected = _select_type_with_llm(
        state["user_query"],
        state["selected_source_product"],
        state["available_source_types"],
        "source"
    )
    state["selected_source_type"] = selected
    print(f"Selected source type: '{selected}'")
    state["messages"].append(AIMessage(content=f"Source type: '{selected}'"))
    state["current_step"] = "source_property_retrieval"
    return state


def get_source_properties_node(state: AgentState):
    """Step 5a: Get properties for selected source type."""
    print("Step 5a: Getting source properties...")
    product = state["selected_source_product"]
    type_name = state["selected_source_type"]
    if not type_name:
        state["source_properties"] = {"type_name": "", "properties": []}
    else:
        try:
            state["source_properties"] = get_properties_for_type(product, type_name)
        except Exception as e:
            print(f"Error getting source properties: {e}")
            state["source_properties"] = {"type_name": type_name, "properties": []}
    print(f"Source properties: {len(state['source_properties'].get('properties', []))} fields")
    state["current_step"] = "target_type_retrieval"
    return state


def get_target_types_node(state: AgentState):
    """Step 3b: Get available types for target product."""
    print("Step 3b: Getting types for target product...")
    product = state["selected_target_product"]
    try:
        types_data = get_types_for_product(product)
        if not isinstance(types_data, list):
            types_data = []
        state["available_target_types"] = types_data
        print(f"Found {len(types_data)} target types in '{product}'")
        state["messages"].append(AIMessage(content=f"Target product '{product}': {len(types_data)} types"))
    except Exception as e:
        print(f"Error in get_target_types_node: {str(e)}")
        state["available_target_types"] = []
    state["current_step"] = "target_type_selection"
    return state


def select_target_type_node(state: AgentState):
    """Step 4b: Select the most relevant target type."""
    print("Step 4b: Selecting target type...")
    selected = _select_type_with_llm(
        state["user_query"],
        state["selected_target_product"],
        state["available_target_types"],
        "target"
    )
    state["selected_target_type"] = selected
    print(f"Selected target type: '{selected}'")
    state["messages"].append(AIMessage(content=f"Target type: '{selected}'"))
    state["current_step"] = "target_property_retrieval"
    return state


def get_target_properties_node(state: AgentState):
    """Step 5b: Get properties for selected target type."""
    print("Step 5b: Getting target properties...")
    product = state["selected_target_product"]
    type_name = state["selected_target_type"]
    if not type_name:
        state["target_properties"] = {"type_name": "", "properties": []}
    else:
        try:
            state["target_properties"] = get_properties_for_type(product, type_name)
        except Exception as e:
            print(f"Error getting target properties: {e}")
            state["target_properties"] = {"type_name": type_name, "properties": []}
    print(f"Target properties: {len(state['target_properties'].get('properties', []))} fields")
    state["current_step"] = "mapping_analysis"
    return state


def analyze_properties_node(state: AgentState):
    """Step 6: Cross-map source fields to target fields.

    FIX BUG 2+3: This node now has BOTH source and target properties and produces
    genuine source→target field mappings instead of source→source self-mappings.
    """
    print("Step 6: Cross-mapping source fields to target fields...")

    user_query = state["user_query"]
    source_props = state["source_properties"]
    target_props = state["target_properties"]
    source_product = state["selected_source_product"]
    target_product = state["selected_target_product"]
    source_type = state["selected_source_type"]
    target_type = state["selected_target_type"]

    # Build property description strings
    source_descriptions = []
    for prop in source_props.get('properties', []):
        source_descriptions.append(
            f"  - {prop['name']} ({prop['data_type']}): {prop['description']}"
        )

    target_descriptions = []
    for prop in target_props.get('properties', []):
        target_descriptions.append(
            f"  - {prop['name']} ({prop['data_type']}): {prop['description']}"
        )

    if not source_descriptions:
        print("No source properties available - cannot produce mappings")
        state["final_mappings"] = []
        state["reasoning"] = []
        state["current_step"] = "complete"
        state["messages"].append(AIMessage(content="No source properties to map"))
        return state

    if not target_descriptions:
        print("No target properties available - cannot produce mappings")
        state["final_mappings"] = []
        state["reasoning"] = []
        state["current_step"] = "complete"
        state["messages"].append(AIMessage(content="No target properties to map"))
        return state

    system_prompt = """You are a schema mapping expert. Your task is to map fields from a SOURCE schema to semantically equivalent fields in a TARGET schema.

For each source field, identify the best matching target field based on semantic meaning, not just name similarity.
A source field may have no match in the target (omit it), or may map to a differently named target field with the same meaning.

You must respond with ONLY a valid JSON array in this exact format:
[
    {
        "source_property": "<exact field name from SOURCE schema>",
        "target_property": "<exact field name from TARGET schema>",
        "justification": "<brief explanation of why these fields are semantically equivalent>"
    }
]

Rules:
- source_property must be an exact name from the SOURCE properties list
- target_property must be an exact name from the TARGET properties list
- Only include mappings where there is genuine semantic equivalence
- Omit source fields that have no meaningful target equivalent
- Return only the JSON array, no additional text"""

    user_content = f"""Mapping task: {user_query}

SOURCE schema: {source_product}.{source_type}
SOURCE properties:
{chr(10).join(source_descriptions)}

TARGET schema: {target_product}.{target_type}
TARGET properties:
{chr(10).join(target_descriptions)}

Map the SOURCE fields to semantically equivalent TARGET fields:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    print(f"Mapping {len(source_props.get('properties', []))} source fields "
          f"against {len(target_props.get('properties', []))} target fields")

    raw_response = call_llm(messages, model=MODEL).strip()

    # Parse JSON response
    try:
        json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = raw_response.strip()

        analysis_result = json.loads(json_str)

        if not isinstance(analysis_result, list):
            raise ValueError("Response must be a JSON array")

        # Validate and filter entries
        valid_source_names = {p['name'] for p in source_props.get('properties', [])}
        valid_target_names = {p['name'] for p in target_props.get('properties', [])}

        filtered_result = []
        for item in analysis_result:
            required_keys = ["source_property", "target_property", "justification"]
            if not all(key in item for key in required_keys):
                print(f"Skipping item missing required keys: {item}")
                continue
            src_name = item["source_property"]
            tgt_name = item["target_property"]
            # Only keep mappings with valid field names from both schemas
            if src_name not in valid_source_names:
                print(f"Skipping mapping: '{src_name}' not in source schema")
                continue
            if tgt_name not in valid_target_names:
                print(f"Skipping mapping: '{tgt_name}' not in target schema")
                continue
            filtered_result.append(item)

        print(f"Produced {len(filtered_result)} valid field mappings")
        analysis_result = filtered_result

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {raw_response[:500]}")
        analysis_result = []

    state["final_mappings"] = analysis_result
    state["reasoning"] = analysis_result
    state["current_step"] = "complete"
    state["messages"].append(AIMessage(content=json.dumps(analysis_result, indent=2)))
    return state


# ── Graph construction ────────────────────────────────────────────────────────

def create_property_search_graph():
    """Create and return the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("get_products", get_products_node)
    workflow.add_node("select_product", select_product_node)
    workflow.add_node("get_source_types", get_source_types_node)
    workflow.add_node("select_source_type", select_source_type_node)
    workflow.add_node("get_source_properties", get_source_properties_node)
    workflow.add_node("get_target_types", get_target_types_node)
    workflow.add_node("select_target_type", select_target_type_node)
    workflow.add_node("get_target_properties", get_target_properties_node)
    workflow.add_node("analyze_properties", analyze_properties_node)

    # Sequential: source path first, then target path, then cross-mapping
    workflow.add_edge("get_products", "select_product")
    workflow.add_edge("select_product", "get_source_types")
    workflow.add_edge("get_source_types", "select_source_type")
    workflow.add_edge("select_source_type", "get_source_properties")
    workflow.add_edge("get_source_properties", "get_target_types")
    workflow.add_edge("get_target_types", "select_target_type")
    workflow.add_edge("select_target_type", "get_target_properties")
    workflow.add_edge("get_target_properties", "analyze_properties")
    workflow.add_edge("analyze_properties", END)

    workflow.set_entry_point("get_products")
    return workflow.compile()


def search_properties(user_query: str):
    """Main function to execute the property search."""
    print(f"Starting property search for query: '{user_query}'")
    print("=" * 60)

    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "current_step": "initialization",
        "available_products": [],
        "selected_source_product": "",
        "available_source_types": [],
        "selected_source_type": "",
        "source_properties": {},
        "selected_target_product": "",
        "available_target_types": [],
        "selected_target_type": "",
        "target_properties": {},
        "final_mappings": [],
        "reasoning": ""
    }

    graph = create_property_search_graph()
    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("Search completed!")
    n_mappings = len(final_state.get('final_mappings', []))
    print(f"Mappings produced: {n_mappings}")

    return final_state


# Example usage
if __name__ == "__main__":
    data = {
        "name": "instana_ServiceEndpoint.id",
        "description": "Unique ID of the endpoint.",
        "type": "ID",
        "schema_component": "instana_ServiceEndpoint",
        "metadata": {
            "product": "instana",
            "graphql_type": "ServiceEndpoint"
        }
    }
    test_queries = [
        "find a property in concert which can be mapped directly to the following property:\n"
        f"{data['name']}: {data['description']}"
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Testing query: {query}")
        print('='*80)

        try:
            result = search_properties(query)
        except Exception as e:
            print(f"Error processing query: {e}")

        print("\n" + "="*80)
