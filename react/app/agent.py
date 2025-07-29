import os
import json
import requests
import re
import numpy as np
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
    source_schema = process(source_graphql_schema, source_name)
    target_schema = process(target_graphql_schema, target_name)
    config = {
        "credentials": {
            "apikey": os.getenv("WATSONX_APIKEY"),
            "project_id": os.getenv("WATSONX_PROJECT_ID"),
        },
        "model_name": "meta-llama/llama-3-3-70b-instruct",
        "max_new_tokens": 4096,
    }
    result = solve(
        source_schema_dict=target_schema,
        target_schema_dict=source_schema,
        pipeline_name="graphql_pipeline",
        **config,
    )

    trimmed_result = {"name": result["name"], "properties": {}}
    for x in result["properties"]:
        if result["properties"][x] != []:
            trimmed_result["properties"][x] = result["properties"][x]

    return trimmed_result

class GraphQLSchemaMappingAgent:
    """
    A GraphQL Mapping Agent using LangGraph for orchestration.
    """
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, base_schema_dir: str = "./data/graphql"):
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.schema_map = build_schema_index(base_schema_dir)
        self.graph = self._create_graph()

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
            
        return state

    def _select_schemas_node(self, state: AgentState) -> AgentState:
        """Node to select source and target schemas using LLM"""
        try:
            utterance = state["utterance"]
            schemas = state["schemas"]
            similarities = state.get("schema_similarities", [])
            
            # Use the LLM wrapper for schema selection with detailed reasoning
            source_schema, target_schema, reasoning = self._select_schemas_with_llm(utterance, schemas, similarities)
            
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
            error_msg = AIMessage(content=f"Error during schema selection: {str(e)}")
            state["messages"].append(error_msg)
            
        return state

    def _generate_mappings_node(self, state: AgentState) -> AgentState:
        """Node to generate mappings between schemas"""
        try:
            source_schema = state["source_schema"]
            target_schema = state["target_schema"]
            source_content = state["source_schema_content"]
            target_content = state["target_schema_content"]
            
            if source_content and target_content:
                mapping_result = find_mappings(
                    source_content, source_schema,
                    target_content, target_schema
                )
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
            else:
                error_content = "Missing schema content for mapping generation"
                state["error"] = error_content
                error_msg = AIMessage(content=error_content)
                state["messages"].append(error_msg)
                
        except Exception as e:
            error_content = f"Error generating mappings: {str(e)}"
            state["error"] = error_content
            error_msg = AIMessage(content=error_content)
            state["messages"].append(error_msg)
            
        return state

    def _find_relevant_schema(self, user_question: str, recall_limit=10) -> tuple[list, dict, list]:
        """Find relevant schemas using embeddings and return similarities"""
        if not self.schema_map:
            return [], {}, []
        
        # Prioritize 'solis' keys
        solis_keys = [k for k in self.schema_map.keys() if 'solis' in k.lower()]
        
        user_embedding = self.sentence_transformer.encode([user_question], show_progress_bar=False)[0]
        key_scores = []

        for k in solis_keys:
            doc = k.split("/")[1].replace("_", " ").replace(".graphql", "")
            doc = split_camel_case(doc)
            
            key_embedding = self.sentence_transformer.encode([doc], show_progress_bar=False)[0]
            cosine_sim = np.dot(user_embedding, key_embedding) / (
                np.linalg.norm(user_embedding) * np.linalg.norm(key_embedding)
            )
            schema_name = k.split("/")[1].replace(".graphql", "")
            key_scores.append((schema_name, cosine_sim))

        key_scores_sorted = sorted(key_scores, key=lambda x: x[1], reverse=True)
        best_keys_with_scores = key_scores_sorted[:recall_limit]
        best_keys = [x[0] for x in best_keys_with_scores]
        
        best_schema_map = {}
        for k in solis_keys:
            schema_name = k.split("/")[1].replace(".graphql", "")
            if schema_name in best_keys:
                best_schema_map[schema_name] = self.schema_map[k]
        
        return best_keys, best_schema_map, key_scores_sorted

    def _select_schemas_with_llm(self, utterance: str, schemas: List[str], similarities: List[tuple]) -> tuple[str, str, str]:
        """Select schemas using the LLM wrapper with detailed reasoning"""
        
        # Prepare schema information with similarities
        schema_info = []
        similarity_dict = {schema: score for schema, score in similarities}
        
        for schema in schemas[:10]:  # Limit to top 10 for context
            score = similarity_dict.get(schema, 0.0)
            schema_info.append(f"- {schema} (similarity: {score:.3f})")
        
        schema_selection_prompt = f"""
        Given a user utterance and a list of schemas for mapping, choose a source and a target schema.
        
        User utterance: "{utterance}"
        
        Available schemas with similarity scores:
        {chr(10).join(schema_info)}
        
        Please analyze the utterance and select:
        1. A source schema (what you're mapping FROM)
        2. A target schema (what you're mapping TO)
        
        Consider:
        - Keywords in the utterance that match schema names
        - Semantic similarity scores provided
        - Logical mapping direction based on the request
        
        Your response should be a JSON with the following fields:
        1. "source_schema": the name of the source schema
        2. "target_schema": the name of the target schema  
        3. "reasoning": detailed explanation of why these schemas were chosen, including:
           - How the utterance keywords relate to chosen schemas
           - Why this mapping direction makes sense
           - Similarity scores that influenced the decision
           - Any domain knowledge applied
        """
        
        messages = [
            {"role": "system", "content": "You are a schema selection assistant. Analyze the user request carefully and provide detailed reasoning for your schema selection choices. Respond only with valid JSON."},
            {"role": "user", "content": schema_selection_prompt}
        ]
        
        response = call_llm(messages)
        
        try:
            # Clean the response
            response = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(response)
            return data["source_schema"], data["target_schema"], data.get("reasoning", "No reasoning provided")
        except Exception as e:
            # Fallback with basic reasoning
            if len(schemas) >= 2:
                fallback_reasoning = f"""Fallback selection due to JSON parsing error: {str(e)}
                
Selected based on highest similarity scores:
- Source: {schemas[0]} (score: {similarity_dict.get(schemas[0], 0.0):.3f})
- Target: {schemas[1]} (score: {similarity_dict.get(schemas[1], 0.0):.3f})

This is an automatic fallback when LLM response cannot be parsed."""
                return schemas[0], schemas[1], fallback_reasoning
            elif schemas:
                return schemas[0], schemas[0], "Only one schema available - using as both source and target"
            else:
                return "", "", "No schemas available for selection"

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

    def print_conversation(self, result: Dict[str, Any]):
        """Print the conversation in a readable format"""
        print("\n" + "="*80)
        print("CONVERSATION FLOW")
        print("="*80)
        
        for i, message in enumerate(result.get("messages", [])):
            if isinstance(message, HumanMessage):
                print(f"\n🧑 USER MESSAGE {i+1}:")
                print("-" * 40)
                print(message.content)
            elif isinstance(message, SystemMessage):
                print(f"\n🔧 SYSTEM MESSAGE {i+1}:")
                print("-" * 40)
                print(message.content)
            elif isinstance(message, AIMessage):
                print(f"\n🤖 AI MESSAGE {i+1}:")
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
