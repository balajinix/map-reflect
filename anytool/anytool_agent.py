import os
os.environ['LITELLM_LOG'] = 'ERROR'
os.environ['LITELLM_DROP_PARAMS'] = 'True'
import json
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from litellm import completion
import operator
import re
from dotenv import load_dotenv
load_dotenv()

# # Enable verbose logging for debugging
# litellm.set_verbose = True

# Tool definitions (from your provided code)
class ToolConverter:
    @staticmethod
    def to_langchain_tool(fn):
        langchain_tool = StructuredTool.from_function(func=fn)
        return langchain_tool

# Load hierarchy data (with error handling)

#with open("../data/hierarchy.json") as f:
#    hierarchy = json.load(f)
hierarchy = {}

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
        
        # Handle case where product_types might not be a dict
        if not isinstance(product_types, dict):
            print(f"Warning: Product '{product}' data is not a dictionary: {type(product_types)}")
            return []
        
        result = []
        for type_name, properties in product_types.items():
            # Ensure properties is a list
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
        
        products = [x for x in list(hierarchy.keys()) if not x.startswith("Mutation")]
        print(products)
        return products
    except Exception as e:
        print(f"Error in get_products: {str(e)}")
        return []

# Create LangChain tools
get_products_tool = ToolConverter.to_langchain_tool(get_products)
get_types_for_product_tool = ToolConverter.to_langchain_tool(get_types_for_product)
get_properties_for_type_tool = ToolConverter.to_langchain_tool(get_properties_for_type)

# Tool node (updated from deprecated ToolExecutor)
tools = [get_products_tool, get_types_for_product_tool, get_properties_for_type_tool]
tool_node = ToolNode(tools)

# State definition
class AgentState(TypedDict):
    messages: Annotated[List[Any], operator.add]
    user_query: str
    current_step: str
    available_products: List[str]
    selected_product: str
    available_types: List[Dict]
    selected_type: str
    final_properties: Dict
    reasoning: str

# LLM wrapper for IBM watsonx.ai via litellm
def call_llm(messages, model="watsonx/mistralai/mistral-large"):
    """Call IBM watsonx.ai via litellm"""
    try:
        response = completion(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {str(e)}")
        return f"Error calling LLM: {str(e)}"

# Node functions
def get_products_node(state: AgentState):
    """First node: Get all available products"""
    print("Step 1: Getting available products...")
    
    # Execute the get_products tool
    products = get_products()
    
    # Update state
    state["available_products"] = products
    state["current_step"] = "product_selection"
    
    # Add system message about available products
    products_msg = f"Available products: {', '.join(products)}"
    print(products_msg)
    state["messages"].append(AIMessage(content=f"Found {len(products)} products: {products_msg}"))
    
    return state

def select_product_node(state: AgentState):
    """Second node: Select the most relevant product using LLM"""
    print("🎯 Step 2: Selecting most relevant product...")
    
    user_query = state["user_query"]
    available_products = state["available_products"]
    
    # Create prompt for product selection
    system_prompt = """You are an intelligent assistant that helps users find relevant properties in a product hierarchy.
    Your task is to select the most relevant product based on the user's query.
    
    Analyze the user query and select the product that is most likely to contain the information they're looking for.
    Respond with ONLY the product name, nothing else."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User query: '{user_query}'\nAvailable products: {available_products}\n\nSelect the most relevant product:"}
    ]
    
    selected_product = call_llm(messages).strip()
    
    # Validate selection
    if selected_product not in available_products:
        # Fallback: select first product if LLM response is invalid
        selected_product = available_products[0]
    
    state["selected_product"] = selected_product
    state["current_step"] = "type_retrieval"
    
    reasoning = f"Selected product '{selected_product}' as most relevant to query: '{user_query}'"
    print(reasoning)
    state["messages"].append(AIMessage(content=reasoning))
    
    return state

def get_types_node(state: AgentState):
    """Third node: Get available types for selected product"""
    print("Step 3: Getting available types for selected product...")
    
    selected_product = state["selected_product"]
    
    try:
        # Execute the get_types_for_product tool
        types_data = get_types_for_product(selected_product)
        
        # Ensure types_data is a list
        if isinstance(types_data, str):
            print(f"Warning: get_types_for_product returned string: {types_data}")
            types_data = []
        elif not isinstance(types_data, list):
            print(f"Warning: get_types_for_product returned unexpected type: {type(types_data)}")
            types_data = []
        
        state["available_types"] = types_data
        state["current_step"] = "type_selection"
        
        if types_data:
            types_summary = f"Found {len(types_data)} types in product '{selected_product}'"
        else:
            types_summary = f"No types found in product '{selected_product}'"
        
        print(types_summary)
        state["messages"].append(AIMessage(content=types_summary))
        
    except Exception as e:
        print(f"Error in get_types_node: {str(e)}")
        state["available_types"] = []
        state["messages"].append(AIMessage(content=f"Error getting types: {str(e)}"))
    
    return state

def select_type_node(state: AgentState):
    """Fourth node: Select the most relevant type using LLM"""
    print("🎯 Step 4: Selecting most relevant type...")
    
    user_query = state["user_query"]
    available_types = state["available_types"]
    
    # Check if we have any types available
    if not available_types:
        print("No types available for selection")
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
        print("No valid type descriptions found")
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
    
    selected_type = call_llm(messages).strip()
    
    # Validate selection
    valid_types = [t.get('type_name', '') for t in available_types if isinstance(t, dict)]
    if selected_type not in valid_types and valid_types:
        # Fallback: select first type if LLM response is invalid
        selected_type = valid_types[0]
    
    state["selected_type"] = selected_type
    state["current_step"] = "property_retrieval"
    
    reasoning = f"Selected type '{selected_type}' as most relevant to query"
    print(reasoning)
    state["messages"].append(AIMessage(content=reasoning))
    
    return state

def get_properties_node(state: AgentState):
    """Fifth node: Get properties for selected type"""
    print("Step 5: Getting properties for selected type...")
    
    selected_product = state["selected_product"]
    selected_type = state["selected_type"]
    
    if not selected_type:
        print("No type selected, cannot get properties")
        state["final_properties"] = {"type_name": "", "properties": []}
        state["current_step"] = "property_analysis"
        return state
    
    try:
        # Execute the get_properties_for_type tool
        properties_data = get_properties_for_type(selected_product, selected_type)
        
        # Ensure properties_data is properly structured
        if not isinstance(properties_data, dict):
            properties_data = {"type_name": selected_type, "properties": []}
        
        state["final_properties"] = properties_data
        state["current_step"] = "property_analysis"
        
    except Exception as e:
        print(f"Error getting properties: {str(e)}")
        state["final_properties"] = {"type_name": selected_type, "properties": []}
        state["current_step"] = "property_analysis"
    
    return state

def analyze_properties_node(state: AgentState):
    """Final node: Analyze and present relevant properties"""
    print("🎯 Step 6: Analyzing and presenting relevant properties...")
    
    user_query = state["user_query"]
    properties_data = state["final_properties"]
    
    # Create detailed property descriptions for analysis
    property_descriptions = []
    for prop in properties_data['properties']:
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
    
    print(messages[-1]["content"])
    
    # Call LLM and get response
    raw_response = call_llm(messages)
    
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
        
        print(f"Successfully parsed {len(analysis_result)} property mappings")
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {raw_response}")
        
        # Fallback: create a basic structure if parsing fails
        analysis_result = [{
            "source_property": "unknown",
            "mapped_product": state['selected_product'],
            "mapped_type": state['selected_type'],
            "mapped_property": properties_data['properties'][0]['name'] if properties_data['properties'] else "unknown",
            "justification": "Failed to parse LLM response, using fallback mapping"
        }]
    
    # Store the structured result
    state["reasoning"] = analysis_result
    state["current_step"] = "complete"
    state["messages"].append(AIMessage(content=json.dumps(analysis_result, indent=2)))
    
    return state

# Build the graph
def create_property_search_graph():
    """Create and return the LangGraph workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("get_products", get_products_node)
    workflow.add_node("select_product", select_product_node)
    workflow.add_node("get_types", get_types_node)
    workflow.add_node("select_type", select_type_node)
    workflow.add_node("get_properties", get_properties_node)
    workflow.add_node("analyze_properties", analyze_properties_node)
    
    # Add edges (sequential flow)
    workflow.add_edge("get_products", "select_product")
    workflow.add_edge("select_product", "get_types")
    workflow.add_edge("get_types", "select_type")
    workflow.add_edge("select_type", "get_properties")
    workflow.add_edge("get_properties", "analyze_properties")
    workflow.add_edge("analyze_properties", END)
    
    # Set entry point
    workflow.set_entry_point("get_products")
    
    return workflow.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# Main execution function
def search_properties(user_query: str):
    """Main function to execute the property search"""
    
    print(f"Starting property search for query: '{user_query}'")
    print("=" * 60)
    
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
    
    # Create and run the graph
    graph = create_property_search_graph()
    final_state = graph.invoke(initial_state)
    # print_stream(graph.stream(initial_state))
    
    print("\n" + "=" * 60)
    print("Search completed!")
    print(f"Final reasoning: {final_state['reasoning']}")
    
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
    # Test queries
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
            # print(f"\nResult: {result['reasoning']}")
        except Exception as e:
            print(f"Error processing query: {e}")
        
        print("\n" + "="*80)
