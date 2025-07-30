import json
import networkx as nx
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, TypedDict
import re
from dataclasses import dataclass
from collections import defaultdict
import logging
from litellm import completion
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
logger = logging.getLogger("solis_ai_api")

# Define the state structure
class MappingState(TypedDict):
    user_utterance: str
    property_data: Dict[str, Any]
    source_properties: List[str]
    destination_properties: List[str]
    candidate_mappings: Dict[str, List[Dict[str, str]]]
    final_mappings: Dict[str, List[Dict[str, str]]]
    reflection_feedback: str
    graph: nx.Graph
    chroma_collection: Any
    source_context: str  # Added missing field
    destination_context: str  # Added missing field

@dataclass
class PropertyMapping:
    source_property: str
    destination_property: str
    justification: str
    confidence_score: float = 0.0

class MapReflectAgent:
    def __init__(self, property_lookup_file: str, rebuild_index=False, model: str = "watsonx/mistralai/mistral-large"):
        """
        Initialize MapReflect agent.
        
        Args:
            property_lookup_file: Path to the property lookup JSON file
            rebuild_index: Whether to rebuild the ChromaDB index
            model: Model to use for LLM calls
        """
        # Initialize ChromaDB
        persist_directory = "chroma_dir"
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rebuild_index = rebuild_index
        self.model = model  # Store model for LLM calls
        
        # Load property data
        with open(property_lookup_file, 'r') as f:
            self.property_data = json.load(f)
        
        # Initialize the graph
        self.graph = self._build_graph()
    
    def call_llm(self, messages):
        """Call LLM with the configured model"""
        try:
            response = completion(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=5000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM Error: {str(e)}")
            return f"Error calling LLM: {str(e)}"
        
    def _build_graph(self):
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(MappingState)
        if self.rebuild_index:
            workflow.add_node("initialize_index", self._initialize_index)
        workflow.add_node("extract_contexts", self._extract_contexts)  # Added new node
        workflow.add_node("create_graph", self._create_property_graph)
        workflow.add_node("select_source_properties", self._select_source_properties)
        workflow.add_node("select_destination_properties", self._select_destination_properties)
        workflow.add_node("find_mappings", self._find_mappings)
        workflow.add_node("reflect_and_prune", self._reflect_and_prune)
        workflow.add_node("format_output", self._format_output)

        if self.rebuild_index:
            workflow.add_edge("initialize_index", "extract_contexts")
            workflow.set_entry_point("initialize_index")
        else:
            workflow.set_entry_point("extract_contexts")  # Updated entry point
        
        workflow.add_edge("extract_contexts", "create_graph")
        workflow.add_edge("create_graph", "select_source_properties")
        workflow.add_edge("select_source_properties", "select_destination_properties")
        workflow.add_edge("select_destination_properties", "find_mappings")
        workflow.add_edge("find_mappings", "reflect_and_prune")
        workflow.add_edge("reflect_and_prune", "format_output")
        workflow.add_edge("format_output", END)

        return workflow.compile()
    
    def _initialize_index(self, state: MappingState) -> MappingState:
        """Initialize ChromaDB index with all properties"""
        collection = self._create_collection_from_state(state)
        state["chroma_collection"] = collection
        return state

    def _extract_contexts(self, state: MappingState) -> MappingState:
        """Use LLM to extract both source and destination contexts from user utterance."""
        user_utterance = state["user_utterance"]

        prompt = f"""
        Analyze the following user request and extract the source context and destination context.

        User request: "{user_utterance}"

        Return your answer strictly in JSON format:
        {{
            "source_context": "<short phrase describing the source>",
            "destination_context": "<short phrase describing the destination>"
        }}

        - "source_context": the data domain or entity being mapped FROM
        - "destination_context": the data domain or entity being mapped TO
        """

        messages = [
            {"role": "system", "content": "You are an expert in schema mapping. Extract mapping contexts precisely."},
            {"role": "user", "content": prompt}
        ]

        response = self.call_llm(messages)
        response = response.replace("```json","").replace("```","").strip()

        try:
            contexts = json.loads(response)
            state["source_context"] = contexts.get("source_context", user_utterance)
            state["destination_context"] = contexts.get("destination_context", user_utterance)
        except json.JSONDecodeError:
            # Fallback if LLM fails to return valid JSON
            state["source_context"] = user_utterance
            state["destination_context"] = user_utterance

        return state
    
    def _create_property_graph(self, state: MappingState) -> MappingState:
        print("\nCreating property graph..")
        G = nx.Graph()
        for prop_key, prop_data in state["property_data"].items():
            G.add_node(prop_key, **prop_data)
        
        properties = list(state["property_data"].keys())
        for i, prop1 in enumerate(properties):
            for prop2 in properties[i+1:]:
                data1 = state["property_data"][prop1]
                data2 = state["property_data"][prop2]
                
                weight = 0
                if data1["product_name"] == data2["product_name"]:
                    weight += 0.5
                if data1["type_name"] == data2["type_name"]:
                    weight += 0.3
                
                field1_words = set(data1["field_name"].lower().split('.'))
                field2_words = set(data2["field_name"].lower().split('.'))
                similarity = len(field1_words & field2_words) / max(len(field1_words | field2_words), 1)
                weight += similarity * 0.2
                
                if weight > 0.1:
                    G.add_edge(prop1, prop2, weight=weight)
        
        state["graph"] = G
        return state
    
    def _select_source_properties(self, state: MappingState) -> MappingState:
        print("\nSelecting source properties with neighbors..")

        if state["chroma_collection"] is None:
            try:
                collection = self.chroma_client.get_collection(name="property_index")
            except:
                collection = self._create_collection_from_state(state)
            state["chroma_collection"] = collection
        else:
            collection = state["chroma_collection"]

        source_context = state["source_context"]
        query_embedding = self.embedding_model.encode(source_context).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(10, len(state["property_data"]))
        )

        base_results = set(results["ids"][0]) if results["ids"] else set()

        # Expand with 1-hop neighbors from the property graph
        neighbors = set()
        for prop_id in base_results:
            if prop_id in state["graph"]:
                neighbors.update(state["graph"].neighbors(prop_id))
        
        neighbors = set(list(neighbors)[:5])
        combined_results = list(base_results | neighbors)

        state["source_properties"] = combined_results
        print("Selected source properties:", state["source_properties"])
        return state

    def _create_collection_from_state(self, state: MappingState):
        print("\nCreating collection..")
        """Helper method to create ChromaDB collection from current state"""
        collection = self.chroma_client.get_or_create_collection(
            name="property_index",
            embedding_function=None
        )
        try:
            collection.delete(where={})
        except:
            pass

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for prop_key, prop_data in tqdm(state["property_data"].items()):
            ids.append(prop_key)
            doc_text = f"{prop_data['product_name']} {prop_data['type_name']} {prop_data['field_name']} {prop_data['description']}"
            documents.append(doc_text)
            metadatas.append({
                "product_name": prop_data["product_name"],
                "type_name": prop_data["type_name"],
                "field_name": prop_data["field_name"]
            })
            embeddings.append(self.embedding_model.encode(doc_text))

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        return collection
    
    def _select_destination_properties(self, state: MappingState) -> MappingState:
        print("\nSelecting destination properties with neighbors..")

        collection = state["chroma_collection"]
        destination_context = state["destination_context"]
        query_embedding = self.embedding_model.encode(destination_context).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(10, len(state["property_data"]))
        )

        base_results = set(results["ids"][0]) if results["ids"] else set()

        # Expand with 1-hop neighbors from the property graph
        neighbors = set()
        for prop_id in base_results:
            if prop_id in state["graph"]:
                neighbors.update(state["graph"].neighbors(prop_id))
        
        neighbors = set(list(neighbors)[:5])
        combined_results = list(base_results | neighbors)

        # state["destination_properties"] = combined_results
        state["destination_properties"] = [x for x in combined_results if x not in state["source_properties"]]
        print("Selected destination properties:", state["destination_properties"])
        return state

    
    def _find_mappings(self, state: MappingState) -> MappingState:
        print("\nFinding mappings..")
        source_props = state["source_properties"]
        dest_props = state["destination_properties"]
        property_data = state["property_data"]

        # Build a single prompt containing all source/destination pairs
        prompt = "Evaluate the following possible property mappings.\n"
        prompt += "For each candidate mapping, respond in JSON array format with:\n"
        prompt += """
        [
        {
            "source_property": "<source_prop_id>",
            "destination_property": "<dest_prop_id>",
            "valid": true/false,
            "confidence": 0.0-1.0,
            "justification": "short explanation"
        },
        ...
        ]
        Only include entries where valid=true.
        """

        prompt += "\n\nCandidate Pairs:\n"
        for source_prop in source_props:
            source_info = property_data[source_prop]
            prompt += f"\nSource: {source_prop}\n"
            prompt += f" - Product: {source_info['product_name']}\n"
            prompt += f" - Type: {source_info['type_name']}\n"
            prompt += f" - Field: {source_info['field_name']}\n"
            prompt += f" - Description: {source_info['description']}\n"
            for dest_prop in dest_props:
                dest_info = property_data[dest_prop]
                prompt += f"  Candidate Destination: {dest_prop}\n"
                prompt += f"   - Product: {dest_info['product_name']}\n"
                prompt += f"   - Type: {dest_info['type_name']}\n"
                prompt += f"   - Field: {dest_info['field_name']}\n"
                prompt += f"   - Description: {dest_info['description']}\n"

        messages = [
            {"role": "system", "content": "You are an expert in data mapping and schema alignment."},
            {"role": "user", "content": prompt}
        ]

        response = self.call_llm(messages)
        response = response.replace("```json","").replace("```","").strip()
        try:
            parsed_results = json.loads(response)
        except json.JSONDecodeError:
            print("LLM response not valid JSON, fallback: ", response)
            parsed_results = []

        candidate_mappings = defaultdict(list)
        for mapping in parsed_results:
            if mapping.get("valid") is True:
                candidate_mappings[mapping["source_property"]].append({
                    "mapProperty": mapping["destination_property"],
                    "justification": mapping.get("justification", ""),
                    "confidence": mapping.get("confidence", 0.5)
                })

        state["candidate_mappings"] = dict(candidate_mappings)
        print(f"Candidate mappings: {state['candidate_mappings']}")
        return state

    
    def _reflect_and_prune(self, state: MappingState) -> MappingState:
        print("\nPerforming self-reflection..")
        candidate_mappings = state["candidate_mappings"]
        property_data = state["property_data"]
        reflection_prompt = self._create_reflection_prompt(candidate_mappings, property_data)
        
        messages = [
            {"role": "system", "content": "You are an expert data mapper tasked with validating property mappings. Review each mapping critically and provide feedback."},
            {"role": "user", "content": reflection_prompt}
        ]
        
        reflection_feedback = self.call_llm(messages)
        print(f"Reflection Feedback: {reflection_feedback}")
        final_mappings = self._prune_mappings_based_on_reflection(candidate_mappings, reflection_feedback)
        
        state["final_mappings"] = final_mappings
        state["reflection_feedback"] = reflection_feedback
        return state
    
    def _format_output(self, state: MappingState) -> MappingState:
        return state
    

    def _create_reflection_prompt(self, candidate_mappings: Dict, property_data: Dict) -> str:
        prompt = "Review the following property mappings and identify any that should be removed:\n"
        for source_prop, mappings in candidate_mappings.items():
            source_info = property_data[source_prop]
            prompt += f"\nSource: {source_prop} ({source_info['description'][:100]}...)\n"
            for mapping in mappings:
                dest_prop = mapping["mapProperty"]
                dest_info = property_data[dest_prop]
                prompt += f"  -> {dest_prop} ({dest_info['description'][:100]}...)\n"
                prompt += f"     Justification: {mapping['justification']}\n"
                prompt += f"     Confidence: {mapping['confidence']}\n"
        prompt += "\nConsider: type compatibility, purpose similarity, business value, inconsistencies."
        return prompt
    
    def _prune_mappings_based_on_reflection(
        self, candidate_mappings: Dict[str, List[Dict]], reflection_feedback: str
    ) -> Dict:
        """Use LLM with reflection feedback to prune mappings."""
        print("\nPruning..")

        prompt = f"""
        You previously provided the following reflection feedback on candidate mappings:
        ---
        {reflection_feedback}
        ---

        Now, based on that feedback, remove any mappings that should not be included.

        Candidate mappings are below. 
        Return ONLY the pruned mappings in the following strict JSON format:

        {{
          "<source_property_id>": [
            {{
              "mapProperty": "<destination_property_id>",
              "justification": "<short reason>"
            }},
            ...
          ],
          ...
        }}
        """

        for source_prop, mappings in candidate_mappings.items():
            prompt += f"\nSource: {source_prop}\n"
            for mapping in mappings:
                prompt += f"  -> {mapping['mapProperty']} (Confidence: {mapping['confidence']})\n"
                prompt += f"     Justification: {mapping['justification']}\n"

        messages = [
            {"role": "system", "content": "You are an expert in schema mapping. Output only the final valid mappings in JSON."},
            {"role": "user", "content": prompt}
        ]

        response = self.call_llm(messages)
        response = response.replace("```json","").replace("```","").strip()

        try:
            final_mappings = json.loads(response)
        except json.JSONDecodeError:
            print("⚠️ Warning: LLM response not valid JSON. Falling back to reflection heuristics.")
            # fallback: drop low-confidence mappings
            final_mappings = {
                k: [
                    {"mapProperty": m["mapProperty"], "justification": m["justification"]}
                    for m in v if m["confidence"] >= 0.7
                ]
                for k, v in candidate_mappings.items()
            }

        return final_mappings

    def run(self, user_utterance: str) -> Dict[str, Any]:
        initial_state = MappingState(
            user_utterance=user_utterance,
            property_data=self.property_data,
            source_properties=[],
            destination_properties=[],
            candidate_mappings={},
            final_mappings={},
            reflection_feedback="",
            graph=nx.Graph(),
            chroma_collection=None,
            source_context="",  # Initialize empty contexts
            destination_context=""  # Initialize empty contexts
        )
        final_state = self.graph.invoke(initial_state)
        return {
            "mappings": final_state["final_mappings"],
            "source_properties": final_state["source_properties"],
            "destination_properties": final_state["destination_properties"],
            "reflection_feedback": final_state["reflection_feedback"]
        }

# Example usage
def main():
    agent = MapReflectAgent(property_lookup_file="api_lookup.json")
    user_input = "map concert vulnerabilities to turbo entities"
    result = agent.run(user_input)
    
    print("\nFinal Mappings:")
    print(json.dumps(result["mappings"], indent=2))

if __name__ == "__main__":
    main()
