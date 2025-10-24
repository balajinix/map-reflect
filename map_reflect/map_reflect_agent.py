# schema_mapping/map_reflect_agent.py
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

# Stable logger namespace
logger = logging.getLogger("solis_ai_api.map_reflect")

# ---------- helpers ----------

def _strip_code_fences(s: str) -> str:
    """Remove Markdown code fences from model output before json.loads."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.replace("```json", "").replace("```", "").strip()

def _safe_json_loads(payload: str):
    """Try json.loads; return (obj_or_none, error_str_or_none)."""
    try:
        return json.loads(payload), None
    except Exception as e:
        return None, str(e)

# ---------- types ----------

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
    source_context: str
    destination_context: str

@dataclass
class PropertyMapping:
    source_property: str
    destination_property: str
    justification: str
    confidence_score: float = 0.0

# ---------- agent ----------

class MapReflectAgent:
    def __init__(
        self,
        property_lookup_file: str,
        rebuild_index: bool = False,
        model: str = None,
        dataset: str | None = None,
        chroma_dir: str = "../data/chroma_db",
        collection_name: str | None = None,
        top_k: int = 10,
        neighbor_k: int = 5,
    ):
        """
        Initialize MapReflect agent.
        
        Args:
            property_lookup_file: Path to property lookup JSON file (required)
            rebuild_index: Whether to rebuild the ChromaDB index
            model: LLM model to use (if None, uses default from environment or fallback)
            dataset: Dataset name for namespaced collection (optional)
            chroma_dir: Directory for ChromaDB storage
            collection_name: Override collection name (optional)
            top_k: Number of top results to retrieve
            neighbor_k: Number of neighbors to expand
        """
        self.model = model
        self.dataset = dataset
        self.top_k = max(1, int(top_k))
        self.neighbor_k = max(0, int(neighbor_k))
        self.rebuild_index = rebuild_index

        # Chroma client & embeddings
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load properties
        with open(property_lookup_file, "r") as f:
            self.property_data = json.load(f)

        # Namespace collection per dataset unless overridden
        self.collection_name = (
            collection_name
            if collection_name
            else (f"property_index_{self.dataset.lower()}" if self.dataset else "property_index")
        )

        logger.info(
            "MapReflect init | dataset=%s model=%s chroma_dir=%s collection=%s props=%d",
            self.dataset or "default", self.model, chroma_dir, self.collection_name, len(self.property_data)
        )

        # Build workflow
        self.graph = self._build_graph()

    # ---- LLM wrapper ----
    def call_llm(self, messages):
        """Call LLM with the configured model."""
        try:
            resp = completion(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=5000
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error("LLM error with model %s: %s", self.model, e, exc_info=True)
            return f"Error calling LLM: {e}"

    # ---- LangGraph DAG ----
    def _build_graph(self):
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(MappingState)
        if self.rebuild_index:
            workflow.add_node("initialize_index", self._initialize_index)
        workflow.add_node("extract_contexts", self._extract_contexts)
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
            workflow.set_entry_point("extract_contexts")

        workflow.add_edge("extract_contexts", "create_graph")
        workflow.add_edge("create_graph", "select_source_properties")
        workflow.add_edge("select_source_properties", "select_destination_properties")
        workflow.add_edge("select_destination_properties", "find_mappings")
        workflow.add_edge("find_mappings", "reflect_and_prune")
        workflow.add_edge("reflect_and_prune", "format_output")
        workflow.add_edge("format_output", END)
        return workflow.compile()

    # ---- Nodes ----
    def _initialize_index(self, state: MappingState) -> MappingState:
        state["chroma_collection"] = self._create_collection_from_state(state)
        return state

    def _extract_contexts(self, state: MappingState) -> MappingState:
        user_utterance = state["user_utterance"]
        logger.info("Extracting contexts from utterance")

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
""".strip()

        messages = [
            {"role": "system", "content": "You are an expert in schema mapping. Extract mapping contexts precisely."},
            {"role": "user", "content": prompt},
        ]
        raw = self.call_llm(messages)
        raw = _strip_code_fences(raw)

        ctx, err = _safe_json_loads(raw)
        if err or not isinstance(ctx, dict):
            logger.warning("Context JSON parse failed; using utterance for both contexts. err=%s raw=%s", err, raw[:200])
            state["source_context"] = user_utterance
            state["destination_context"] = user_utterance
        else:
            state["source_context"] = ctx.get("source_context", user_utterance)
            state["destination_context"] = ctx.get("destination_context", user_utterance)
        logger.info("Contexts | source=%r dest=%r", state["source_context"], state["destination_context"])
        return state

    def _create_property_graph(self, state: MappingState) -> MappingState:
        logger.info("Creating property graph")
        G = nx.Graph()
        props = state["property_data"]

        for k, v in props.items():
            G.add_node(k, **v)

        edge_count = 0
        keys = list(props.keys())
        for i, p1 in enumerate(keys):
            d1 = props[p1]
            for p2 in keys[i + 1:]:
                d2 = props[p2]
                weight = 0.0
                if d1.get("product_name") == d2.get("product_name"):
                    weight += 0.5
                if d1.get("type_name") == d2.get("type_name"):
                    weight += 0.3
                w1 = set(str(d1.get("field_name", "")).lower().split("."))
                w2 = set(str(d2.get("field_name", "")).lower().split("."))
                sim = len(w1 & w2) / max(len(w1 | w2), 1)
                weight += 0.2 * sim
                if weight > 0.1:
                    G.add_edge(p1, p2, weight=weight)
                    edge_count += 1

        state["graph"] = G
        logger.info("Graph built | nodes=%d edges=%d", G.number_of_nodes(), edge_count)
        return state

    def _get_or_create_collection(self, state: MappingState):
        try:
            col = self.chroma_client.get_collection(name=self.collection_name)
            logger.info("Using existing Chroma collection: %s", self.collection_name)
            return col
        except Exception:
            return self._create_collection_from_state(state)

    def _select_source_properties(self, state: MappingState) -> MappingState:
        collection = state["chroma_collection"] or self._get_or_create_collection(state)
        state["chroma_collection"] = collection

        q = state["source_context"]
        q_emb = self.embedding_model.encode(q).tolist()
        res = collection.query(query_embeddings=[q_emb], n_results=min(self.top_k, len(state["property_data"])))
        base_ids = set(res["ids"][0]) if res.get("ids") else set()

        # 1-hop neighbors (deterministic cap)
        nbrs = set()
        for pid in base_ids:
            if pid in state["graph"]:
                nbrs.update(state["graph"].neighbors(pid))
        nbrs = set(sorted(nbrs)[: self.neighbor_k])

        combined = sorted(base_ids | nbrs)
        state["source_properties"] = combined
        logger.info("Selected source properties | base=%d +nbr=%d -> %d",
                    len(base_ids), len(nbrs), len(combined))
        return state

    def _create_collection_from_state(self, state: MappingState):
        logger.info("Building Chroma collection: %s", self.collection_name)
        col = self.chroma_client.get_or_create_collection(name=self.collection_name, embedding_function=None)
        try:
            col.delete(where={})
        except Exception:
            pass

        ids, docs, metas, embs = [], [], [], []
        props = state["property_data"]

        for pid, pdata in tqdm(
            props.items(),
            desc=f"Embedding properties -> {self.collection_name}",
            disable=logger.level > logging.INFO,
        ):
            ids.append(pid)
            doc = f"{pdata.get('product_name','')} {pdata.get('type_name','')} {pdata.get('field_name','')} {pdata.get('description','')}"
            docs.append(doc)
            metas.append({
                "product_name": pdata.get("product_name", ""),
                "type_name": pdata.get("type_name", ""),
                "field_name": pdata.get("field_name", "")
            })
            embs.append(self.embedding_model.encode(doc))

        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        logger.info("Chroma collection populated | %d items", len(ids))
        return col

    def _select_destination_properties(self, state: MappingState) -> MappingState:
        collection = state["chroma_collection"]
        q = state["destination_context"]
        q_emb = self.embedding_model.encode(q).tolist()
        res = collection.query(query_embeddings=[q_emb], n_results=min(self.top_k, len(state["property_data"])))
        base_ids = set(res["ids"][0]) if res.get("ids") else set()

        nbrs = set()
        for pid in base_ids:
            if pid in state["graph"]:
                nbrs.update(state["graph"].neighbors(pid))
        nbrs = set(sorted(nbrs)[: self.neighbor_k])

        combined = sorted(base_ids | nbrs)
        filtered = [x for x in combined if x not in state["source_properties"]]
        state["destination_properties"] = filtered
        logger.info("Selected destination properties | base=%d +nbr=%d -> %d (filtered from %d)",
                    len(base_ids), len(nbrs), len(filtered), len(combined))
        return state

    def _find_mappings(self, state: MappingState) -> MappingState:
        srcs = state["source_properties"]
        dsts = state["destination_properties"]
        pdata = state["property_data"]

        if not srcs or not dsts:
            logger.warning("Insufficient properties for mapping | src=%d dst=%d", len(srcs), len(dsts))
            state["candidate_mappings"] = {}
            return state
    
        logger.info("Finding mappings | evaluating %d source × %d dest = %d pairs", 
                    len(srcs), len(dsts), len(srcs) * len(dsts))
    
        prompt = [
            "Evaluate the following possible property mappings.",
            "For each candidate mapping, respond in JSON array format with:",
            """
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
CRITICAL: Return ONLY the JSON array with no additional text before or after.
""".strip(),
            "\nCandidate Pairs:\n"
        ]

        for sp in srcs:
            si = pdata[sp]
            prompt.append(f"\nSource: {sp}\n - Product: {si['product_name']}\n - Type: {si['type_name']}\n - Field: {si['field_name']}\n - Description: {si['description']}")
            for dp in dsts:
                di = pdata[dp]
                prompt.append(
                    f"  Candidate Destination: {dp}\n"
                    f"   - Product: {di['product_name']}\n"
                    f"   - Type: {di['type_name']}\n"
                    f"   - Field: {di['field_name']}\n"
                    f"   - Description: {di['description']}"
                )

        messages = [
            {"role": "system", "content": "You are an expert in data mapping and schema alignment. Return only valid JSON with no explanatory text."},
            {"role": "user", "content": "\n".join(prompt)}
        ]

        raw = self.call_llm(messages)
        raw = _strip_code_fences(raw)
        parsed, err = _safe_json_loads(raw)
        
        if err or not isinstance(parsed, list):
            logger.warning("Mapping JSON parse failed; err=%s", err)
            logger.info("Response length: %d characters", len(raw))
            logger.info("Response preview (first 500 chars):\n%s", raw[:500])
            logger.info("Response end (last 200 chars):\n%s", raw[-200:] if len(raw) > 200 else raw)
            
            # Try to extract JSON from verbose response
            # Look for JSON array anywhere in the response
            json_match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', raw)
            if json_match:
                try:
                    extracted_json = json_match.group()
                    logger.info("Found JSON array in response, attempting to parse...")
                    parsed = json.loads(extracted_json)
                    logger.info("Successfully recovered JSON array from verbose response (%d items)", len(parsed))
                except Exception as e2:
                    logger.warning("Failed to parse extracted JSON: %s", e2)
                    logger.debug("Extracted text was: %s", extracted_json[:500])
                    parsed = []
            else:
                logger.warning("No JSON array pattern found in response")
                parsed = []
    
        cand = defaultdict(list)
        valid_count = 0
        invalid_count = 0
        
        for m in parsed:
            if isinstance(m, dict):
                if m.get("valid") is True:
                    valid_count += 1
                    src = m.get("source_property")
                    dst = m.get("destination_property")
                    if not src or not dst:
                        logger.warning("Skipping mapping with missing src or dst: %s", m)
                        continue
                    cand[src].append({
                        "mapProperty": dst,
                        "justification": m.get("justification", ""),
                        "confidence": float(m.get("confidence", 0.5))
                    })
                else:
                    invalid_count += 1
    
        state["candidate_mappings"] = dict(cand)
        total_mappings = sum(len(v) for v in cand.values())
        
        logger.info("Parsed response: %d valid, %d invalid mappings", valid_count, invalid_count)
        
        if cand:
            logger.info("Candidate mappings found | %d sources with %d total mappings", 
                        len(cand), total_mappings)
        else:
            logger.warning("No candidate mappings found")
        
        return state

    def _create_reflection_prompt(self, candidate_mappings: Dict, property_data: Dict) -> str:
        lines = ["Review the following property mappings and identify any that should be removed:"]
        for sp, maps in candidate_mappings.items():
            sdesc = property_data.get(sp, {}).get("description", "")
            lines.append(f"\nSource: {sp} ({sdesc[:100]}...)")
            for m in maps:
                dp = m["mapProperty"]
                ddesc = property_data.get(dp, {}).get("description", "")
                lines.append(f"  -> {dp} ({ddesc[:100]}...)\n     Justification: {m['justification']}\n     Confidence: {m['confidence']}")
        lines.append("\nConsider: type compatibility, purpose similarity, business value, inconsistencies.")
        return "\n".join(lines)

    def _reflect_and_prune(self, state: MappingState) -> MappingState:
        cand = state["candidate_mappings"]
        if not cand:
            state["final_mappings"] = {}
            state["reflection_feedback"] = "No candidate mappings were generated"
            logger.info("No candidates to reflect on")
            return state

        prompt = self._create_reflection_prompt(cand, state["property_data"])
        messages = [
            {"role": "system", "content": "You are an expert data mapper tasked with validating property mappings. Review each mapping critically and provide feedback."},
            {"role": "user", "content": prompt}
        ]
        feedback = self.call_llm(messages)

        prune_prompt = [
            "You previously provided the following reflection feedback on candidate mappings:",
            "---",
            str(feedback),
            "---",
            "Now, based on that feedback, remove any mappings that should not be included.",
            "Candidate mappings are below.",
            "Return ONLY the pruned mappings in the following strict JSON format:",
            """
{
  "<source_property_id>": [
    {"mapProperty": "<destination_property_id>", "justification": "<short reason>"},
    ...
  ],
  ...
}
""".strip()
        ]
        for sp, maps in cand.items():
            prune_prompt.append(f"\nSource: {sp}")
            for m in maps:
                prune_prompt.append(f"  -> {m['mapProperty']} (Confidence: {m['confidence']})\n     Justification: {m['justification']}")

        messages = [
            {"role": "system", "content": "You are an expert in schema mapping. Output only the final valid mappings in JSON."},
            {"role": "user", "content": "\n".join(prune_prompt)}
        ]

        raw = self.call_llm(messages)
        raw = _strip_code_fences(raw)
        final, err = _safe_json_loads(raw)
        if err or not isinstance(final, dict):
            logger.warning("Pruned JSON parse failed; falling back to confidence filter; err=%s raw=%s", err, raw[:200])
            final = {
                s: [{"mapProperty": m["mapProperty"], "justification": m.get("justification", "")}
                    for m in maps if float(m.get("confidence", 0.0)) >= 0.7]
                for s, maps in cand.items()
            }

        state["final_mappings"] = final
        state["reflection_feedback"] = str(feedback)
        logger.info("Final mappings after reflection | sources=%d", len(final))
        return state

    def _format_output(self, state: MappingState) -> MappingState:
        return state

    # ---- public ----
    def run(self, user_utterance: str) -> Dict[str, Any]:
        logger.info("MapReflect run | utterance=%r model=%s", user_utterance, self.model)
        initial = MappingState(
            user_utterance=user_utterance,
            property_data=self.property_data,
            source_properties=[],
            destination_properties=[],
            candidate_mappings={},
            final_mappings={},
            reflection_feedback="",
            graph=nx.Graph(),
            chroma_collection=None,
            source_context="",
            destination_context=""
        )
        final_state = self.graph.invoke(initial)
        out = {
            "mappings": final_state["final_mappings"],
            "source_properties": final_state["source_properties"],
            "destination_properties": final_state["destination_properties"],
            "reflection_feedback": final_state["reflection_feedback"]
        }
        logger.info("MapReflect done | final_sources=%d", len(out["mappings"]))
        return out

# Example usage
def main():
    logging.basicConfig(level=logging.WARNING)
    agent = MapReflectAgent(
        property_lookup_file="../data/itops_api_lookup.json",
        dataset="test",
        model="watsonx/mistralai/mistral-large"  # Example model specification
    )
    user_input = "map concert vulnerabilities to turbo entities"
    result = agent.run(user_input)
    print("\nFinal Mappings:")
    print(json.dumps(result["mappings"], indent=2))

if __name__ == "__main__":
    main()
