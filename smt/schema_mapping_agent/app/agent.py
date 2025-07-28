import os
import json
import requests
import re
import numpy as np
import asyncio
import time
from typing import Dict, Any, List, Optional, TypedDict, Annotated
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

# MCP imports
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools


load_dotenv()
logger = logging.getLogger("solis_ai_api")

# Updated MCP server URL to match your server
MCP_SERVER_URL = "http://127.0.0.1:3001/sse"  # Updated to match your server port

class MCPServer:
    def __init__(self):
        self.stream_contexts = []
        self.session_contexts = []
        self.sessions = []
    
    async def connect_to_sse_server(self, url):
        stream_context = sse_client(url=url)
        streams = await stream_context.__aenter__()
        session_context = ClientSession(*streams)
        session = await session_context.__aenter__()
        await session.initialize()
        
        self.stream_contexts.append(stream_context)
        self.session_contexts.append(session_context)
        self.sessions.append(session)
        return session
    
    async def cleanup(self):
        for session_context in self.session_contexts:
            if session_context:
                await session_context.__aexit__(None, None, None)
        for stream_context in self.stream_contexts:
            if stream_context:
                await stream_context.__aexit__(None, None, None)

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
    original_utterance: str  # Keep original for storage routing
    schemas: List[str]
    schema_map: Dict[str, str]
    schema_similarities: Optional[List[tuple]]
    source_schema: Optional[str]
    target_schema: Optional[str]
    source_schema_content: Optional[str]
    target_schema_content: Optional[str]
    schema_selection_reasoning: Optional[str]
    mapping_result: Optional[Dict[str, Any]]
    store_where: Optional[str]  # "neo4j", "wiki", "both", "none"
    storage_results: Optional[Dict[str, Any]]  # Results from storage operations
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
    A GraphQL Mapping Agent using LangGraph for orchestration with MCP tool integration.
    """
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, base_schema_dir: str = "/home/gunjan/mydata/graphql"):
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.schema_map = build_schema_index(base_schema_dir)
        self.mcp_tools = {} 
        self.mcp_server = MCPServer()  # Add this line
        self.graph = None

    async def async_init(self):
        """Async initialization for MCP tools and graph"""
        if self.graph:
            logger.info("Agent already initialized.")
            return
        
        logger.info("Initializing Agent: loading MCP tools...")
        
        try:
            # Connect to MCP server and maintain connection
            session = await self.mcp_server.connect_to_sse_server(MCP_SERVER_URL)
            
            # Load MCP tools
            from langchain_mcp_adapters.tools import load_mcp_tools
            mcp_tools = await load_mcp_tools(session)
            
            # Convert to simple dict
            self.mcp_tools = {tool.name: tool for tool in mcp_tools}
            logger.info(f"Loaded MCP tools: {list(self.mcp_tools.keys())}")
            print(f"[DEBUG] Loaded MCP tools: {list(self.mcp_tools.keys())}")  # TEMP

            
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")
            self.mcp_tools = {}
        
        self.graph = self._create_graph()
    
    async def cleanup(self):
        """Cleanup MCP connections"""
        if self.mcp_server:
            await self.mcp_server.cleanup()
        
    def _create_graph(self):
        """Create the LangGraph workflow with integrated tool functions."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("find_schemas", self._find_relevant_schemas_node)
        workflow.add_node("select_schemas", self._select_schemas_node)
        workflow.add_node("generate_mappings", self._generate_mappings_node)
        workflow.add_node("check_store_destination", self._check_store_destination)
        workflow.add_node("store_in_neo4j", self._store_in_neo4j_node)
        workflow.add_node("store_in_wiki", self._store_in_wiki_node)
        workflow.add_node("store_in_neo4j_and_wiki", self._store_in_neo4j_and_wiki_node)

        # Add edges
        workflow.set_entry_point("find_schemas")
        workflow.add_edge("find_schemas", "select_schemas")
        workflow.add_edge("select_schemas", "generate_mappings")
        workflow.add_edge("generate_mappings", "check_store_destination")

        # Conditional routing based on detected storage intent
        conditional_mapping = {
            "none": END,
            "neo4j": "store_in_neo4j",
            "wiki": "store_in_wiki",
            "both": "store_in_neo4j_and_wiki"
        }

        workflow.add_conditional_edges(
            "check_store_destination",
            self._route_storage_path,
            conditional_mapping
        )

        # Define edges for individual storage options
        workflow.add_edge("store_in_neo4j", END)
        workflow.add_edge("store_in_wiki", END)
        workflow.add_edge("store_in_neo4j_and_wiki", END)

        return workflow.compile()
    
    def _sanitize_utterance_for_mapping(self, utterance: str) -> str:
        """Remove storage instructions from utterance to avoid affecting mapping logic"""
        # Pattern to match storage-related instructions
        storage_patterns = [
            r"(and\s+)?(store|save|push|populate|update|put)\s+(in|into|to|the)?\s*(neo4j|graph|database|wiki|documentation)",
            r"(and\s+)?(store|save|populate)\s+(the\s+)?(result|mapping|data)\s+(in|into|to)\s*(neo4j|graph|database|wiki)",
            r"(and\s+)?(update|populate)\s+(the\s+)?(wiki|documentation|neo4j|graph)",
            r"store\s+(it|this|the\s+result)\s+(in|into|to)\s*(neo4j|wiki|graph)",
        ]
        
        cleaned_utterance = utterance
        for pattern in storage_patterns:
            cleaned_utterance = re.sub(pattern, "", cleaned_utterance, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and conjunctions
        cleaned_utterance = re.sub(r'\s+', ' ', cleaned_utterance)
        cleaned_utterance = re.sub(r'\s+(and|,)\s*$', '', cleaned_utterance, flags=re.IGNORECASE)
        
        return cleaned_utterance.strip()

    async def _check_store_destination(self, state: AgentState) -> AgentState:
        """Check where to store the mapping results based on utterance"""
        original_utterance = state.get("original_utterance", "").lower()
        actions = []

        # Check for Neo4j/graph database keywords
        neo4j_keywords = ["neo4j", "graph database", "graph db", "store in graph", "populate neo4j"]
        wiki_keywords = ["wiki", "documentation", "update wiki", "populate wiki", "generate wiki"]
        
        for keyword in neo4j_keywords:
            if keyword in original_utterance:
                actions.append("neo4j")
                break
                
        for keyword in wiki_keywords:
            if keyword in original_utterance:
                actions.append("wiki")
                break

        # Determine storage destination
        if "neo4j" in actions and "wiki" in actions:
            state["store_where"] = "both"
        elif "neo4j" in actions:
            state["store_where"] = "neo4j"
        elif "wiki" in actions:
            state["store_where"] = "wiki"
        else:
            state["store_where"] = "none"
        
        # Add system message about storage decision
        storage_decision = state["store_where"]
        if storage_decision != "none":
            system_content = f"Storage Decision: Will store mapping results in {storage_decision}"
            system_msg = SystemMessage(content=system_content)
            state["messages"].append(system_msg)
        
        return state

    def _route_storage_path(self, state: AgentState) -> str:
        """Route to appropriate storage path"""
        return state.get("store_where", "none")

    async def _store_in_neo4j_node(self, state: AgentState) -> AgentState:
        """Store mapping results in Neo4j using MCP tool"""
        try:
            mapping_result = state.get("mapping_result")
            source_schema = state.get("source_schema")
            target_schema = state.get("target_schema")
            
            if not mapping_result:
                state["error"] = "No mapping result to store in Neo4j"
                return state
            
            neo4j_tool = self.mcp_tools.get("populate_neo4j")
            if not neo4j_tool:
                state["error"] = "Neo4j tool not available"
                return state
            
            # Prepare data for Neo4j storage
            neo4j_data = {
                "mapping_name": mapping_result.get("name", f"{source_schema}_to_{target_schema}"),
                "source_schema": source_schema,
                "target_schema": target_schema,
                "mapping_properties": mapping_result.get("properties", {}),
            }
            
            # Call Neo4j tool
            neo4j_result = await neo4j_tool.ainvoke({"data": mapping_result})
            
            if not state.get("storage_results"):
                state["storage_results"] = {}
            state["storage_results"]["neo4j"] = neo4j_result
            
            # Add AI message about Neo4j storage
            ai_content = f"""✅ **Neo4j Storage Complete**

**Stored Mapping:**
- Name: {neo4j_data['mapping_name']}
- Source: {source_schema}
- Target: {target_schema}
- Properties: {len(neo4j_data['mapping_properties'])} mappings

**Storage Result:**
{neo4j_result}"""
            
            ai_msg = AIMessage(content=ai_content)
            state["messages"].append(ai_msg)
            
        except Exception as e:
            error_msg = f"Error storing in Neo4j: {str(e)}"
            state["error"] = error_msg
            error_ai_msg = AIMessage(content=f"❌ {error_msg}")
            state["messages"].append(error_ai_msg)
        
        return state

    async def _store_in_wiki_node(self, state: AgentState) -> AgentState:
        """Generate wiki documentation using MCP tool"""
        try:
            mapping_result = state.get("mapping_result")
            source_schema = state.get("source_schema")
            target_schema = state.get("target_schema")
            
            if not mapping_result:
                state["error"] = "No mapping result to document in wiki"
                return state
                
            wiki_tool = self.mcp_tools.get("generate_wiki")
            if not wiki_tool:
                state["error"] = "Wiki tool not available"
                return state
                    
            # Call Wiki tool
            wiki_result = await wiki_tool.ainvoke({"content": mapping_result})
            
            if not state.get("storage_results"):
                state["storage_results"] = {}
            state["storage_results"]["wiki"] = wiki_result
            
            # Add AI message about wiki generation
            ai_content = f"""📖 **Wiki Documentation Generated**
GraphQL Schema Mapping: {source_schema} to {target_schema}
**Generation Result:**
{wiki_result}"""
            
            ai_msg = AIMessage(content=ai_content)
            state["messages"].append(ai_msg)
            
        except Exception as e:
            error_msg = f"Error generating wiki documentation: {str(e)}"
            state["error"] = error_msg
            error_ai_msg = AIMessage(content=f"❌ {error_msg}")
            state["messages"].append(error_ai_msg)
        
        return state

    
    async def _store_in_neo4j_and_wiki_node(self, state: AgentState) -> AgentState:
        """Store mapping results in both Neo4j and Wiki using integrated functions"""
        try:
            # Store in Neo4j
            state = await self._store_in_neo4j_node(state)  # Add await

            # Store in Wiki
            state = await self._store_in_wiki_node(state)   # Add await

        except Exception as e:
            error_msg = f"Error storing in Neo4j and Wiki: {str(e)}"
            state["error"] = error_msg
            error_ai_msg = AIMessage(content=f"❌ {error_msg}")
            state["messages"].append(error_ai_msg)

        return state

    async def _find_relevant_schemas_node(self, state: AgentState) -> AgentState:
        """Node to find relevant schemas based on user utterance"""
        try:
            # Store original utterance for storage routing
            original_utterance = state["utterance"]
            state["original_utterance"] = original_utterance
            
            # Clean utterance for mapping logic
            cleaned_utterance = self._sanitize_utterance_for_mapping(original_utterance)
            state["utterance"] = cleaned_utterance
            
            schemas, schema_map, similarities = self._find_relevant_schema(cleaned_utterance)
            
            # Add system message about schema finding with detailed analysis
            similarity_info = "\n".join([f"  - {schema}: {score:.3f}" for schema, score in similarities[:5]])
            system_content = f"""Found {len(schemas)} relevant schemas for utterance: '{cleaned_utterance}'
{'(Original: "' + original_utterance + '")' if cleaned_utterance != original_utterance else ''}

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

    async def _select_schemas_node(self, state: AgentState) -> AgentState:
        """Node to select source and target schemas using LLM"""
        try:
            utterance = state["utterance"]  # Use cleaned utterance
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

    async def _generate_mappings_node(self, state: AgentState) -> AgentState:
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
        
        user_embedding = self.sentence_transformer.encode([user_question])[0]
        key_scores = []

        for k in solis_keys:
            doc = k.split("/")[1].replace("_", " ").replace(".graphql", "")
            doc = split_camel_case(doc)
            
            key_embedding = self.sentence_transformer.encode([doc])[0]
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

    async def run_agent(self, utterance: str) -> Dict[str, Any]:
        """Run the agent with a user utterance"""
        if not self.graph:
            raise RuntimeError("Agent not initialized. Call async_init() first.")
            
        initial_state = {
            "messages": [HumanMessage(content=utterance)],
            "utterance": utterance,
            "original_utterance": utterance,
            "schemas": [],
            "schema_map": {},
            "schema_similarities": None,
            "source_schema": None,
            "target_schema": None,
            "source_schema_content": None,
            "target_schema_content": None,
            "schema_selection_reasoning": None,
            "mapping_result": None,
            "store_where": None,
            "storage_results": None,
            "error": None
        }
        
        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        return {
            "mapping_result": final_state.get("mapping_result"),
            "source_schema": final_state.get("source_schema"),
            "target_schema": final_state.get("target_schema"),
            "schema_selection_reasoning": final_state.get("schema_selection_reasoning"),
            "store_where": final_state.get("store_where"),
            "storage_results": final_state.get("storage_results"),
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
        
        # Print storage results if available
        storage_results = result.get("storage_results")
        if storage_results:
            print("\nSTORAGE RESULTS:")
            print("-" * 40)
            for storage_type, result_data in storage_results.items():
                print(f"{storage_type.upper()}: {result_data}")

# Tool wrapper for LangChain integration
class GraphQLMappingTool(BaseModel):
    """Tool for GraphQL schema mapping"""
    utterance: str = Field(description="The user's natural language query for schema mapping")

def create_graphql_mapping_tool(agent: GraphQLSchemaMappingAgent):
    """Create a LangChain tool from the agent"""
    async def mapping_function(utterance: str) -> str:
        result = await agent.run_agent(utterance)
        if result["error"]:
            return f"Error: {result['error']}"
        
        # Include reasoning and storage results in the tool output
        output = {
            "mapping_result": result["mapping_result"],
            "source_schema": result["source_schema"],
            "target_schema": result["target_schema"],
            "selection_reasoning": result["schema_selection_reasoning"],
            "storage_location": result["store_where"],
            "storage_results": result["storage_results"]
        }
        return json.dumps(output, indent=2)
    
    return StructuredTool.from_function(
        func=mapping_function,
        name="graphql_mapping",
        description="Map between GraphQL schemas based on natural language description and optionally store results in Neo4j or Wiki",
        args_schema=GraphQLMappingTool
    )


### UNCOMMENT IF YOU WANT TO TEST INDIVIDUALLY
import asyncio

async def main():
    agent = GraphQLSchemaMappingAgent()
    try:
        await agent.async_init()
        
        # Test query
        test_query = "Map concert applications to instana applications and store into wiki"
        print(f"Testing with: {test_query}")
        
        result = await agent.run_agent(test_query)
        agent.print_conversation(result)
        
        print("\nFinal Result:")
        print(json.dumps(result["mapping_result"], indent=2) if result["mapping_result"] else "No mapping result")
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())


