"""
ReMatch: Retrieval Enhanced Schema Matching with LLMs
Faithful implementation of the paper: "ReMatch: Retrieval Enhanced Schema Matching with LLMs"
by Sheetrit et al., 2024
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re

from litellm import completion
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger("rematch")


# ---------- Helper Functions ----------

def _strip_code_fences(s: str) -> str:
    """Remove Markdown code fences from model output."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.replace("```json", "").replace("```", "").strip()


def _safe_json_loads(payload: str) -> Tuple[Any, Optional[str]]:
    """Try json.loads; return (obj_or_none, error_str_or_none)."""
    try:
        return json.loads(payload), None
    except Exception as e:
        return None, str(e)


# ---------- Data Classes ----------

@dataclass
class MatchCandidate:
    """Represents a candidate target attribute match."""
    target_table: str
    target_attribute: str
    rank: int
    justification: str


@dataclass
class MatchResult:
    """Result of matching a source attribute."""
    source_table: str
    source_attribute: str
    candidates: List[MatchCandidate]
    top_k_matches: List[str]  # List of target attribute IDs


# ---------- ReMatch Agent ----------

class ReMatchAgent:
    """
    ReMatch: Retrieval Enhanced Schema Matching with LLMs.
    
    Implements the approach from the paper:
    1. Convert schemas to structured documents (target tables, source attributes)
    2. For each source attribute, retrieve top-J candidate target tables
    3. Use LLM to rank and select top-K matching attributes from candidates
    """
    
    def __init__(
        self,
        property_lookup_file: str,
        model: str,
        dataset: str,
        top_j: int = 2,
        top_k: int = 5,
        use_guidance: bool = False,
        guidance_mappings: Optional[Dict[str, Tuple[str, str]]] = None
    ):
        """
        Initialize ReMatch agent.
        
        Args:
            property_lookup_file: Path to property lookup JSON
            model: LLM model identifier
            dataset: Dataset name for namespacing
            top_j: Number of candidate target tables to retrieve per source attribute
            top_k: Number of top matches to return
            use_guidance: Whether to use human guidance (relevance feedback)
            guidance_mappings: Dict mapping source_table -> (target_table, attribute_pair)
        """
        self.model = model
        self.dataset = dataset
        self.top_j = top_j
        self.top_k = top_k
        self.use_guidance = use_guidance
        self.guidance_mappings = guidance_mappings or {}
        
        # Load property data
        with open(property_lookup_file, 'r') as f:
            self.property_data = json.load(f)
        
        # Separate source and target properties
        self.source_properties = {}
        self.target_properties = {}
        self._parse_properties()
        
        # Group properties by table
        self.source_tables = self._group_by_table(self.source_properties)
        self.target_tables = self._group_by_table(self.target_properties)
        
        # Initialize embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Using sentence-transformers for embeddings")
        except ImportError:
            logger.error("sentence-transformers not available. Please install it.")
            raise
        
        # Pre-compute target table documents and embeddings
        self._build_document_corpus()
        
        logger.info(
            f"ReMatch initialized | dataset={dataset} model={model} "
            f"top_j={top_j} top_k={top_k} guidance={use_guidance}"
        )
    
    def _parse_properties(self):
        """Parse properties into source and target based on naming convention."""
        for prop_id, prop_data in self.property_data.items():
            # Adjust based on your data format
            # For MIMIC-OMOP: properties with 'source' category are source
            category = prop_data.get('category', '').lower()
            if 'source' in category or 'mimic' in category.lower():
                self.source_properties[prop_id] = prop_data
            else:
                self.target_properties[prop_id] = prop_data
    
    def _group_by_table(self, properties: Dict) -> Dict[str, List[str]]:
        """Group property IDs by their table name."""
        tables = defaultdict(list)
        for prop_id, prop_data in properties.items():
            table_name = prop_data.get('type_name', 'unknown')
            tables[table_name].append(prop_id)
        return dict(tables)
    
    # ========== Step 1: Document Creation ==========
    
    def _create_table_document(self, table_name: str, attribute_ids: List[str]) -> str:
        """
        Create structured document for a target table.
        
        Paper Format:
        - Title: table name
        - Opening paragraph: overview of table
        - Paragraph 2: Primary key attributes
        - Paragraph 3: Foreign key attributes
        - Paragraph 4: Other attributes
        
        Each attribute includes: name, data type, and description.
        """
        doc_parts = []
        
        # Title
        doc_parts.append(f"Table: {table_name}")
        
        # Opening paragraph - table overview
        doc_parts.append(f"This table represents {table_name} and contains the following attributes:")
        
        # Categorize attributes
        primary_keys = []
        foreign_keys = []
        other_attrs = []
        
        for attr_id in attribute_ids:
            attr_data = self.target_properties.get(attr_id, {})
            field_name = attr_data.get('field_name', attr_id)
            data_type = attr_data.get('data_type', 'unknown')
            description = attr_data.get('description', '')
            
            attr_text = f"{field_name} ({data_type}): {description}"
            
            # Categorize based on field name or metadata
            if 'id' in field_name.lower() and 'primary' in description.lower():
                primary_keys.append(attr_text)
            elif 'id' in field_name.lower() and 'foreign' in description.lower():
                foreign_keys.append(attr_text)
            else:
                other_attrs.append(attr_text)
        
        # Primary keys paragraph
        if primary_keys:
            doc_parts.append("\nPrimary Key Attributes:")
            doc_parts.extend([f"- {pk}" for pk in primary_keys])
        
        # Foreign keys paragraph
        if foreign_keys:
            doc_parts.append("\nForeign Key Attributes:")
            doc_parts.extend([f"- {fk}" for fk in foreign_keys])
        
        # Other attributes paragraph
        if other_attrs:
            doc_parts.append("\nOther Attributes:")
            doc_parts.extend([f"- {attr}" for attr in other_attrs])
        
        return "\n".join(doc_parts)
    
    def _create_attribute_document(
        self, 
        attr_id: str, 
        attr_data: Dict,
        table_name: str
    ) -> str:
        """
        Create structured document for a source attribute.
        
        Paper Format:
        - Highlight the specific attribute above the title
        - Title: table name
        - Include table context and attribute details
        """
        field_name = attr_data.get('field_name', attr_id)
        data_type = attr_data.get('data_type', 'unknown')
        description = attr_data.get('description', '')
        product_name = attr_data.get('product_name', '')
        
        doc_parts = [
            f">>> Attribute: {field_name} <<<",
            f"Table: {table_name}",
            f"Product: {product_name}",
            f"Data Type: {data_type}",
            f"Description: {description}"
        ]
        
        return "\n".join(doc_parts)
    
    def _build_document_corpus(self):
        """
        Pre-compute target table documents and embeddings.
        Paper: Step 1 (lines 2-4 in Algorithm 1).
        """
        logger.info("Building document corpus for target tables...")
        
        self.target_documents = {}
        self.target_embeddings = {}
        
        for table_name, attr_ids in tqdm(
            self.target_tables.items(),
            desc="Creating target table documents",
            disable=logger.level > logging.INFO
        ):
            # Create document
            doc = self._create_table_document(table_name, attr_ids)
            self.target_documents[table_name] = doc
            
            # Compute embedding
            embedding = self.embedding_model.encode(doc, convert_to_tensor=False)
            self.target_embeddings[table_name] = embedding
        
        logger.info(f"Built corpus with {len(self.target_documents)} target tables")
    
    # ========== Step 2: Candidate Table Retrieval ==========
    
    def _retrieve_candidate_tables(
        self,
        source_attr_id: str,
        source_attr_data: Dict,
        source_table: str
    ) -> List[str]:
        """
        Retrieve top-J candidate target tables for a source attribute.
        Paper: Step 2 (lines 7-11 in Algorithm 1).
        
        Uses embedding similarity between source attribute document
        and target table documents.
        """
        # Create source attribute document
        source_doc = self._create_attribute_document(
            source_attr_id, 
            source_attr_data,
            source_table
        )
        
        # Compute embedding
        source_embedding = self.embedding_model.encode(source_doc, convert_to_tensor=False)
        
        # Compute similarities with all target tables
        similarities = {}
        for table_name, target_embedding in self.target_embeddings.items():
            # Cosine similarity
            similarity = self._cosine_similarity(source_embedding, target_embedding)
            similarities[table_name] = similarity
        
        # Get top-J tables
        sorted_tables = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_j_tables = [table for table, score in sorted_tables[:self.top_j]]
        
        # Apply guidance if available
        if self.use_guidance and source_table in self.guidance_mappings:
            target_table, _ = self.guidance_mappings[source_table]
            if target_table not in top_j_tables:
                # Add guided table to candidates
                top_j_tables.append(target_table)
        
        logger.debug(f"Retrieved {len(top_j_tables)} candidate tables for {source_attr_id}")
        return top_j_tables
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    # ========== Step 3: LLM-Based Attribute Matching ==========
    
    def _match_attributes_with_llm(
        self,
        source_table: str,
        source_attr_id: str,
        source_attr_data: Dict,
        candidate_tables: List[str]
    ) -> List[str]:
        """
        Use LLM to select top-K matching target attributes from candidate tables.
        Paper: Step 3 (line 13 in Algorithm 1).
        
        The LLM receives:
        - Source attribute details
        - Candidate target tables with their attributes
        - Task to rank and select top-K matches
        """
        # Prepare source attribute description
        source_field = source_attr_data.get('field_name', source_attr_id)
        source_type = source_attr_data.get('data_type', 'unknown')
        source_desc = source_attr_data.get('description', '')
        
        # Prepare candidate target attributes
        candidate_attrs = []
        for table_name in candidate_tables:
            if table_name not in self.target_tables:
                continue
            
            for attr_id in self.target_tables[table_name]:
                attr_data = self.target_properties.get(attr_id, {})
                candidate_attrs.append({
                    'id': attr_id,
                    'table': table_name,
                    'field': attr_data.get('field_name', attr_id),
                    'type': attr_data.get('data_type', 'unknown'),
                    'description': attr_data.get('description', '')[:200]  # Truncate
                })
        
        # Add guidance if available
        guidance_text = ""
        if self.use_guidance and source_table in self.guidance_mappings:
            target_table, attr_mapping = self.guidance_mappings[source_table]
            guidance_text = f"\n\nGuidance: A known mapping exists - {attr_mapping} maps to table {target_table}. Consider this when making your selections."
        
        # Construct prompt
        prompt = f"""You are an expert at schema matching. Your task is to identify the top {self.top_k} most likely matching attributes from the target schema for the given source attribute.

Source Attribute:
- Table: {source_table}
- Field: {source_field}
- Type: {source_type}
- Description: {source_desc}

Candidate Target Attributes:
{json.dumps(candidate_attrs, indent=2)}
{guidance_text}

Task: Analyze the semantic similarity, data type compatibility, and contextual alignment between the source attribute and each candidate. Select the top {self.top_k} most likely matches.

Consider:
1. Semantic similarity of field names and descriptions
2. Data type compatibility
3. Table context and relationships
4. Domain knowledge

Return ONLY a JSON array of the top {self.top_k} target attribute IDs in order of match quality (best first):
["target_id_1", "target_id_2", "target_id_3", ...]

If fewer than {self.top_k} good matches exist, return only the valid matches."""

        messages = [
            {
                "role": "system", 
                "content": "You are an expert schema matching assistant. Respond only with valid JSON array of attribute IDs."
            },
            {"role": "user", "content": prompt}
        ]
        
        # Call LLM
        response = self._call_llm(messages, temperature=0.1)
        response = _strip_code_fences(response)
        
        # Parse response
        matches, err = _safe_json_loads(response)
        if err or not isinstance(matches, list):
            logger.warning(f"Failed to parse LLM matches for {source_attr_id}: {err}")
            return []
        
        # Validate matches are real attribute IDs
        valid_matches = [
            m for m in matches 
            if m in self.target_properties
        ][:self.top_k]
        
        logger.debug(f"LLM matched {source_attr_id} to {len(valid_matches)} attributes")
        return valid_matches
    
    # ========== Main Matching Pipeline ==========
    
    def match_attribute(
        self,
        source_attr_id: str,
        source_table: str
    ) -> MatchResult:
        """
        Match a single source attribute to target schema.
        
        Implements the full ReMatch pipeline:
        1. Retrieve top-J candidate target tables
        2. Use LLM to rank and select top-K matching attributes
        
        Returns:
            MatchResult with ranked candidate matches
        """
        if source_attr_id not in self.source_properties:
            logger.warning(f"Source attribute not found: {source_attr_id}")
            return MatchResult(
                source_table=source_table,
                source_attribute=source_attr_id,
                candidates=[],
                top_k_matches=[]
            )
        
        source_attr_data = self.source_properties[source_attr_id]
        
        # Step 2: Retrieve candidate tables
        candidate_tables = self._retrieve_candidate_tables(
            source_attr_id,
            source_attr_data,
            source_table
        )
        
        if not candidate_tables:
            logger.warning(f"No candidate tables found for {source_attr_id}")
            return MatchResult(
                source_table=source_table,
                source_attribute=source_attr_id,
                candidates=[],
                top_k_matches=[]
            )
        
        # Step 3: Match attributes using LLM
        top_k_matches = self._match_attributes_with_llm(
            source_table,
            source_attr_id,
            source_attr_data,
            candidate_tables
        )
        
        # Create candidate match objects
        candidates = []
        for rank, target_attr_id in enumerate(top_k_matches, start=1):
            target_data = self.target_properties.get(target_attr_id, {})
            target_table = target_data.get('type_name', 'unknown')
            
            candidates.append(MatchCandidate(
                target_table=target_table,
                target_attribute=target_attr_id,
                rank=rank,
                justification=f"Ranked #{rank} by LLM semantic matching"
            ))
        
        return MatchResult(
            source_table=source_table,
            source_attribute=source_attr_id,
            candidates=candidates,
            top_k_matches=top_k_matches
        )
    
    def run(
        self, 
        source_tables: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, MatchResult]]:
        """
        Run ReMatch on source schema.
        
        Args:
            source_tables: List of source table names to match. 
                          If None, matches all source tables.
        
        Returns:
            Nested dictionary: {table_name: {attr_id: MatchResult}}
        """
        if source_tables is None:
            source_tables = list(self.source_tables.keys())
        
        results = {}
        
        for table_name in tqdm(
            source_tables,
            desc="Matching tables",
            disable=logger.level > logging.INFO
        ):
            if table_name not in self.source_tables:
                logger.warning(f"Source table not found: {table_name}")
                continue
            
            table_results = {}
            attr_ids = self.source_tables[table_name]
            
            for attr_id in tqdm(
                attr_ids,
                desc=f"Matching attributes in {table_name}",
                leave=False,
                disable=logger.level > logging.INFO
            ):
                result = self.match_attribute(attr_id, table_name)
                table_results[attr_id] = result
            
            results[table_name] = table_results
        
        return results
    
    def evaluate(
        self,
        results: Dict[str, Dict[str, MatchResult]],
        ground_truth: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Evaluate matching results against ground truth.
        
        Args:
            results: Output from run()
            ground_truth: Dict mapping source_attr_id -> target_attr_id
        
        Returns:
            Dictionary with accuracy@K metrics
        """
        metrics = {}
        
        for k in range(1, self.top_k + 1):
            correct = 0
            total = 0
            
            for table_results in results.values():
                for attr_id, result in table_results.items():
                    if attr_id not in ground_truth:
                        continue
                    
                    total += 1
                    true_match = ground_truth[attr_id]
                    
                    # Check if true match is in top-k
                    top_k = result.top_k_matches[:k]
                    if true_match in top_k:
                        correct += 1
            
            accuracy = correct / total if total > 0 else 0.0
            metrics[f'accuracy@{k}'] = accuracy
        
        return metrics
    
    # ========== LLM Helper ==========
    
    def _call_llm(self, messages: List[Dict], temperature: float = 0.1) -> str:
        """Call LLM with error handling."""
        try:
            response = completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""


# ========== Example Usage ==========

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Basic usage without guidance
    agent = ReMatchAgent(
        property_lookup_file="../data/mimic_omop_lookup.json",
        model="watsonx/mistralai/mistral-large",
        dataset="mimic_omop",
        top_j=2,
        top_k=5,
        use_guidance=False
    )
    
    # Match all source attributes
    results = agent.run()
    
    # Print sample results
    for table_name, table_results in list(results.items())[:2]:
        print(f"\n{'='*60}")
        print(f"Table: {table_name}")
        print(f"{'='*60}")
        
        for attr_id, result in list(table_results.items())[:3]:
            print(f"\nSource Attribute: {attr_id}")
            print(f"Top {len(result.top_k_matches)} Matches:")
            for cand in result.candidates:
                print(f"  {cand.rank}. {cand.target_attribute} (table: {cand.target_table})")
    
    # Example 2: With guidance (relevance feedback)
    print("\n\n" + "="*60)
    print("Running with Guidance")
    print("="*60)
    
    guidance = {
        'admissions': ('person', 'subject_id'),
        'patients': ('person', 'subject_id')
    }
    
    agent_guided = ReMatchAgent(
        property_lookup_file="../data/mimic_omop_lookup.json",
        model="gpt-4",
        dataset="mimic_omop",
        top_j=1,
        top_k=5,
        use_guidance=True,
        guidance_mappings=guidance
    )
    
    results_guided = agent_guided.run(source_tables=['admissions'])
    
    # Evaluate if ground truth is available
    # ground_truth = load_ground_truth()  # Your ground truth loader
    # metrics = agent.evaluate(results, ground_truth)
    # print(f"\nEvaluation Metrics: {metrics}")


if __name__ == "__main__":
    main()