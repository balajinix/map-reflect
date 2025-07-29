#!/usr/bin/env python3
"""
AnyTool Runner - Tool-based schema mapping evaluation
Uses LangGraph workflow with external tools for schema discovery and mapping
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

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable verbose logging from external libraries
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# Add the data directory to Python path for shared components
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

try:
    from dataset_loader import DatasetLoader
    logger.info("Successfully imported shared DatasetLoader")
except ImportError as e:
    logger.error(f"Failed to import shared DatasetLoader: {e}")
    sys.exit(1)

# Import LangGraph components
try:
    from typing import TypedDict, Annotated
    from langchain_core.tools import StructuredTool
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from litellm import completion
    import operator
    
    # Disable LiteLLM verbose output
    import litellm
    litellm.set_verbose = False
    
    LANGGRAPH_AVAILABLE = True
    logger.info("LangGraph components imported successfully")
except ImportError as e:
    logger.warning(f"LangGraph components not available: {e}")
    LANGGRAPH_AVAILABLE = False


class ToolConverter:
    @staticmethod
    def to_langchain_tool(fn):
        if LANGGRAPH_AVAILABLE:
            return StructuredTool.from_function(func=fn)
        else:
            return None


class AnyToolHierarchyLoader:
    """Load and manage hierarchy data for AnyTool."""
    
    def __init__(self, hierarchy_file: str = "../data/hierarchy.json"):
        """Initialize with hierarchy file path."""
        self.hierarchy_file = Path(hierarchy_file)
        self.hierarchy = self._load_hierarchy()
    
    def _load_hierarchy(self) -> Dict[str, Any]:
        """Load hierarchy data with error handling."""
        try:
            if not self.hierarchy_file.exists():
                logger.warning(f"Hierarchy file not found: {self.hierarchy_file}")
                return {}
            
            with open(self.hierarchy_file) as f:
                hierarchy = json.load(f)
            
            logger.info(f"Loaded hierarchy with {len(hierarchy)} products")
            return hierarchy
            
        except Exception as e:
            logger.error(f"Error loading hierarchy: {e}")
            return {}
    
    def get_products(self):
        """Get all products."""
        try:
            if not isinstance(self.hierarchy, dict):
                logger.warning(f"Hierarchy is not a dictionary: {type(self.hierarchy)}")
                return []
            
            products = [x for x in list(self.hierarchy.keys()) if not x.startswith("Mutation")]
            logger.debug(f"Found products: {products}")
            return products
        except Exception as e:
            logger.error(f"Error in get_products: {e}")
            return []
    
    def get_types_for_product(self, product):
        """Get all GraphQL types with their descriptions for a specific product."""
        try:
            product_types = self.hierarchy.get(product, {})
            
            if not isinstance(product_types, dict):
                logger.warning(f"Product '{product}' data is not a dictionary: {type(product_types)}")
                return []
            
            result = []
            for type_name, properties in product_types.items():
                if not isinstance(properties, list):
                    logger.warning(f"Properties for type '{type_name}' is not a list: {type(properties)}")
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
            logger.error(f"Error in get_types_for_product: {e}")
            return []
    
    def get_properties_for_type(self, product, graphql_type):
        """Get all properties with their descriptions and data types for a specific product and GraphQL type."""
        try:
            product_data = self.hierarchy.get(product, {})
            if not isinstance(product_data, dict):
                logger.warning(f"Product '{product}' data is not a dictionary")
                return {'type_name': graphql_type, 'properties': []}
                
            properties = product_data.get(graphql_type, [])
            if not isinstance(properties, list):
                logger.warning(f"Properties for type '{graphql_type}' is not a list")
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
            logger.error(f"Error in get_properties_for_type: {e}")
            return {'type_name': graphql_type, 'properties': []}


class AnyToolLLMClient:
    """LLM client for AnyTool using litellm."""
    
    def __init__(self, model: str = "watsonx/mistralai/mistral-large"):
        """Initialize LLM client."""
        self.model = self._get_full_model_name(model)
    
    def _get_full_model_name(self, model_short_name: str) -> str:
        """Convert short model names to full litellm model names."""
        model_mapping = {
            "granite": "watsonx/ibm/granite-13b-chat-v2",
            "llama": "watsonx/meta-llama/llama-3-3-70b-instruct",
            "qwen": "watsonx/qwen/qwen2-72b-instruct",
            "mistral-large": "watsonx/mistralai/mistral-large"
        }
        
        # Return mapped name or use as-is if already full path
        return model_mapping.get(model_short_name, model_short_name)
    
    def call_llm(self, messages, temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """Call LLM via litellm."""
        try:
            if not LANGGRAPH_AVAILABLE:
                # Fallback response when LangGraph is not available
                return "Fallback response - LangGraph not available"
            
            response = completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"LLM Error: {e}")
            return f"Error calling LLM: {str(e)}"


class AgentState(TypedDict):
    """State definition for AnyTool agent."""
    messages: Annotated[List[Any], operator.add]
    user_query: str
    current_step: str
    available_products: List[str]
    selected_product: str
    available_types: List[Dict]
    selected_type: str
    final_properties: Dict
    reasoning: str


class AnyToolMappingAgent:
    """AnyTool mapping agent using LangGraph workflow."""
    
    def __init__(self, hierarchy_file: str = "../data/hierarchy.json", model: str = "mistral-large"):
        """Initialize AnyTool mapping agent."""
        self.hierarchy_loader = AnyToolHierarchyLoader(hierarchy_file)
        self.llm_client = AnyToolLLMClient(model)
        self.available = LANGGRAPH_AVAILABLE and len(self.hierarchy_loader.hierarchy) > 0
        
        if self.available:
            self._setup_workflow()
            logger.info("AnyTool mapping agent initialized")
        else:
            logger.warning("AnyTool agent not fully available")
    
    def _setup_workflow(self):
        """Setup LangGraph workflow."""
        if not LANGGRAPH_AVAILABLE:
            return
        
        # Create tools
        get_products_tool = ToolConverter.to_langchain_tool(self.hierarchy_loader.get_products)
        get_types_tool = ToolConverter.to_langchain_tool(self.hierarchy_loader.get_types_for_product)
        get_properties_tool = ToolConverter.to_langchain_tool(self.hierarchy_loader.get_properties_for_type)
        
        # Setup workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("get_products", self._get_products_node)
        workflow.add_node("select_product", self._select_product_node)
        workflow.add_node("get_types", self._get_types_node)
        workflow.add_node("select_type", self._select_type_node)
        workflow.add_node("get_properties", self._get_properties_node)
        workflow.add_node("analyze_properties", self._analyze_properties_node)
        
        # Add edges
        workflow.add_edge("get_products", "select_product")
        workflow.add_edge("select_product", "get_types")
        workflow.add_edge("get_types", "select_type")
        workflow.add_edge("select_type", "get_properties")
        workflow.add_edge("get_properties", "analyze_properties")
        workflow.add_edge("analyze_properties", END)
        
        # Set entry point
        workflow.set_entry_point("get_products")
        
        self.workflow = workflow.compile()
    
    def run_agent(self, user_query: str) -> Dict[str, Any]:
        """Run the AnyTool agent on a user query."""
        if not self.available:
            return self._fallback_response(user_query)
        
        try:
            logger.info(f"Running AnyTool agent for query: '{user_query}'")
            
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=user_query)],
                "user_query": user_query,
                "current_step": "initialization",
                "available_products": [],
                "selected_product": "",
                "available_types": [],
                "selected_type": "",
                "final_properties": {},
                "reasoning": ""
            }
            
            # Run workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Process results
            reasoning = final_state.get('reasoning', [])
            
            if isinstance(reasoning, list) and reasoning:
                # Convert to mapping format
                mapping_result = self._convert_to_mapping_format(reasoning, final_state)
                
                return {
                    "mapping_result": mapping_result,
                    "source_schema": final_state.get('selected_product', ''),
                    "target_schema": "anytool_target",  # AnyTool maps to discovered properties
                    "schema_selection_reasoning": f"AnyTool selected {final_state.get('selected_product', '')} using tool-based discovery",
                    "error": None,
                    "anytool_state": final_state
                }
            else:
                return {
                    "mapping_result": {},
                    "source_schema": "",
                    "target_schema": "",
                    "schema_selection_reasoning": "AnyTool failed to find relevant mappings",
                    "error": "No valid reasoning generated",
                    "anytool_state": final_state
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
        # Simple keyword-based fallback
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
    
    # Workflow node implementations
    def _get_products_node(self, state: AgentState):
        """First node: Get all available products."""
        logger.debug("Step 1: Getting available products...")
        
        products = self.hierarchy_loader.get_products()
        state["available_products"] = products
        state["current_step"] = "product_selection"
        
        products_msg = f"Available products: {', '.join(products)}"
        logger.debug(products_msg)
        state["messages"].append(AIMessage(content=f"Found {len(products)} products: {products_msg}"))
        
        return state
    
    def _select_product_node(self, state: AgentState):
        """Second node: Select the most relevant product using LLM."""
        logger.debug("🎯 Step 2: Selecting most relevant product...")
        
        user_query = state["user_query"]
        available_products = state["available_products"]
        
        system_prompt = """You are an intelligent assistant that helps users find relevant properties in a product hierarchy.
        Your task is to select the most relevant product based on the user's query.
        
        Analyze the user query and select the product that is most likely to contain the information they're looking for.
        Respond with ONLY the product name, nothing else."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User query: '{user_query}'\nAvailable products: {available_products}\n\nSelect the most relevant product:"}
        ]
        
        selected_product = self.llm_client.call_llm(messages).strip()
        
        # Validate selection
        if selected_product not in available_products and available_products:
            selected_product = available_products[0]
        
        state["selected_product"] = selected_product
        state["current_step"] = "type_retrieval"
        
        reasoning = f"Selected product '{selected_product}' as most relevant to query: '{user_query}'"
        logger.debug(reasoning)
        state["messages"].append(AIMessage(content=reasoning))
        
        return state
    
    def _get_types_node(self, state: AgentState):
        """Third node: Get available types for selected product."""
        logger.debug("Step 3: Getting available types for selected product...")
        
        selected_product = state["selected_product"]
        
        try:
            types_data = self.hierarchy_loader.get_types_for_product(selected_product)
            state["available_types"] = types_data
            state["current_step"] = "type_selection"
            
            if types_data:
                types_summary = f"Found {len(types_data)} types in product '{selected_product}'"
            else:
                types_summary = f"No types found in product '{selected_product}'"
            
            logger.debug(types_summary)
            state["messages"].append(AIMessage(content=types_summary))
            
        except Exception as e:
            logger.error(f"Error in get_types_node: {e}")
            state["available_types"] = []
            state["messages"].append(AIMessage(content=f"Error getting types: {str(e)}"))
        
        return state
    
    def _select_type_node(self, state: AgentState):
        """Fourth node: Select the most relevant type using LLM."""
        logger.debug("🎯 Step 4: Selecting most relevant type...")
        
        user_query = state["user_query"]
        available_types = state["available_types"]
        
        if not available_types:
            logger.debug("No types available for selection")
            state["selected_type"] = ""
            state["current_step"] = "property_retrieval"
            state["messages"].append(AIMessage(content="No types available for selection"))
            return state
        
        # Create detailed type descriptions for LLM
        type_descriptions = []
        for type_info in available_types:
            if isinstance(type_info, dict) and 'type_name' in type_info:
                type_descriptions.append(
                    f"- {type_info['type_name']}: {type_info.get('description', 'No description')} ({type_info.get('property_count', 0)} properties)"
                )
        
        if not type_descriptions:
            logger.debug("No valid type descriptions found")
            state["selected_type"] = ""
            state["current_step"] = "property_retrieval"
            state["messages"].append(AIMessage(content="No valid types found"))
            return state
        
        system_prompt = """You are an intelligent assistant that helps users find relevant properties in a product hierarchy.
        Your task is to select the most relevant GraphQL type based on the user's query and type descriptions.
        
        Analyze the user query and the available types with their descriptions to select the type most likely to contain the relevant properties.
        Respond with ONLY the type name, nothing else."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User query: '{user_query}'\n\nAvailable types:\n" + "\n".join(type_descriptions) + "\n\nSelect the most relevant type:"}
        ]
        
        selected_type = self.llm_client.call_llm(messages).strip()
        
        # Validate selection
        valid_types = [t.get('type_name', '') for t in available_types if isinstance(t, dict)]
        if selected_type not in valid_types and valid_types:
            selected_type = valid_types[0]
        
        state["selected_type"] = selected_type
        state["current_step"] = "property_retrieval"
        
        reasoning = f"Selected type '{selected_type}' as most relevant to query"
        logger.debug(reasoning)
        state["messages"].append(AIMessage(content=reasoning))
        
        return state
    
    def _get_properties_node(self, state: AgentState):
        """Fifth node: Get properties for selected type."""
        logger.debug("Step 5: Getting properties for selected type...")
        
        selected_product = state["selected_product"]
        selected_type = state["selected_type"]
        
        if not selected_type:
            logger.debug("No type selected, cannot get properties")
            state["final_properties"] = {"type_name": "", "properties": []}
            state["current_step"] = "property_analysis"
            return state
        
        try:
            properties_data = self.hierarchy_loader.get_properties_for_type(selected_product, selected_type)
            
            if not isinstance(properties_data, dict):
                properties_data = {"type_name": selected_type, "properties": []}
            
            state["final_properties"] = properties_data
            state["current_step"] = "property_analysis"
            
        except Exception as e:
            logger.error(f"Error getting properties: {e}")
            state["final_properties"] = {"type_name": selected_type, "properties": []}
            state["current_step"] = "property_analysis"
        
        return state
    
    def _analyze_properties_node(self, state: AgentState):
        """Final node: Analyze and present relevant properties."""
        logger.debug("🎯 Step 6: Analyzing and presenting relevant properties...")
        
        user_query = state["user_query"]
        properties_data = state["final_properties"]
        
        # Create detailed property descriptions for analysis
        property_descriptions = []
        for prop in properties_data.get('properties', []):
            property_descriptions.append(
                f"- {prop['name']} ({prop['data_type']}): {prop['description']}"
            )
        
        system_prompt = """You are an intelligent assistant that analyzes user queries and maps them to relevant properties.
        Your task is to identify the most relevant properties and return them in a specific JSON format.
        
        For each relevant property you identify, you need to:
        1. Infer the source property from the user's query (what they're asking about)
        2. Map it to the selected product and type
        3. Generate an appropriate mapped property name
        4. Provide justification for why this mapping is relevant
        
        You must respond with ONLY a valid JSON array in this exact format:
        [
            {
                "source_property": "<property name inferred from user query>",
                "mapped_product": "<selected_product>",
                "mapped_type": "<selected_type>",
                "mapped_property": "<property name from available properties>",
                "justification": "<explanation of why this mapping is relevant>"
            }
        ]
        
        Important:
        - Only include properties that are directly relevant to the user's query
        - The source_property should reflect what the user is asking about
        - The mapped_property should be the exact property name from the available properties
        - Provide clear, concise justification for each mapping
        - Return only the JSON array, no additional text"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""User query: '{user_query}'

Selected product: {state['selected_product']}
Selected type: {state['selected_type']}

Available properties:
{chr(10).join(property_descriptions)}

Analyze the user query and return the relevant property mappings in the required JSON format."""}
        ]
        
        # Call LLM and get response
        raw_response = self.llm_client.call_llm(messages)
        
        # Parse and validate JSON response
        try:
            # Extract JSON from response if it contains additional text
            json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = raw_response.strip()
            
            # Parse JSON
            analysis_result = json.loads(json_str)
            
            # Validate structure
            if not isinstance(analysis_result, list):
                raise ValueError("Response must be a JSON array")
            
            for item in analysis_result:
                required_keys = ["source_property", "mapped_product", "mapped_type", "mapped_property", "justification"]
                if not all(key in item for key in required_keys):
                    raise ValueError(f"Missing required keys in item: {item}")
            
            logger.debug(f"Successfully parsed {len(analysis_result)} property mappings")
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Error parsing JSON response: {e}")
            logger.debug(f"Raw response: {raw_response}")
            
            # Fallback: create a basic structure if parsing fails
            analysis_result = [{
                "source_property": "unknown",
                "mapped_product": state['selected_product'],
                "mapped_type": state['selected_type'],
                "mapped_property": properties_data.get('properties', [{}])[0].get('name', 'unknown') if properties_data.get('properties') else "unknown",
                "justification": "Failed to parse LLM response, using fallback mapping"
            }]
        
        # Store the structured result
        state["reasoning"] = analysis_result
        state["current_step"] = "complete"
        state["messages"].append(AIMessage(content=json.dumps(analysis_result, indent=2)))
        
        return state


class AnyToolEvaluationRunner:
    """Runner class for AnyTool schema mapping evaluation."""
    
    def __init__(self, 
                 hierarchy_file: str = "../data/hierarchy.json",
                 schema_dir: str = "../data/graphql"):
        """
        Initialize the AnyTool evaluation runner.
        
        Args:
            hierarchy_file: Path to hierarchy JSON file
            schema_dir: Directory containing GraphQL schemas
        """
        self.hierarchy_file = Path(hierarchy_file)
        self.schema_dir = Path(schema_dir)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize the shared dataset loader
        data_dir = Path(__file__).parent.parent / "data"
        self.dataset_loader = DatasetLoader(str(data_dir))
        
        logger.info(f"AnyTool runner initialized")
        logger.info(f"   Hierarchy file: {self.hierarchy_file}")
        logger.info(f"   Schema directory: {self.schema_dir}")
    
    def _get_agent_for_model(self, model: str) -> AnyToolMappingAgent:
        """Get or create an agent for the specified model."""
        return AnyToolMappingAgent(str(self.hierarchy_file), model)
    
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
            
            logger.info(f"💾 Results saved to {output_file}")
            
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
                logger.info(f"💾 Saved minimal results to {output_file}")
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
        # This is a placeholder implementation
        # In a real evaluation, you would compare with ground truth mappings
        
        if not predicted_mapping:
            return 0.0
        
        # For now, return a basic score based on whether we found any mappings
        # This should be replaced with actual ground truth comparison
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
                "hierarchy_file": str(self.hierarchy_file)
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
    parser.add_argument("--models", type=str, default="watsonx/mistralai/mistral-large",
                       help="Comma-separated list of models to use (default: watsonx/mistralai/mistral-large)")
    
    # File and directory arguments
    parser.add_argument("--hierarchy-file", type=str, default="../data/hierarchy.json",
                       help="Path to hierarchy.json file (default: ../data/hierarchy.json)")
    parser.add_argument("--schema-dir", type=str, default="../data/graphql",
                       help="Directory containing GraphQL schemas (default: ../data/graphql)")
    
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
    else:
        # Disable verbose logging from external libraries when not in verbose mode
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
    
    # Parse datasets and models
    datasets = [d.strip() for d in args.datasets.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    
    logger.info(f"Starting AnyTool evaluation")
    logger.info(f"   Datasets: {datasets}")
    logger.info(f"   Models: {models}")
    logger.info(f"   Hierarchy file: {args.hierarchy_file}")
    logger.info(f"   Schema directory: {args.schema_dir}")
    logger.info(f"   Quick mode: {args.quick}")
    
    # Initialize the evaluation runner
    try:
        runner = AnyToolEvaluationRunner(
            hierarchy_file=args.hierarchy_file,
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
            logger.info(f"🔄 Running evaluation: {dataset} with {model}")
            
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
