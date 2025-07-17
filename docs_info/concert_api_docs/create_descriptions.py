"""
Functions to create descriptions for each endpoint in Concert API. Can be called to populate the API lookup etc. 
Use the main function for testing with individual JSON files. 
"""

import os
import chromadb
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
import re
from typing import List, Dict, Optional


class ConcertAPIDescriptionGenerator:
    def __init__(self, chroma_path: str = "/data/wikidata/chromadb", collection_name: str = "concert_api"):
        """Initialize the RAG system for Concert API description generation."""
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(name=collection_name)
        
        # Load environment variables
        load_dotenv()
        
        # Initialize Watson AI credentials
        self.credentials = Credentials(
            url="https://us-south.ml.cloud.ibm.com",
            api_key=os.environ["WATSONX_APIKEY"]
        )
        self.project_id = os.environ["WATSONX_PROJECT_ID"]
    
    def parse_input(self, input_string: str) -> Dict[str, str]:
        """
        Parse input string in format: productname_typename.fields.fields...
        
        Args:
            input_string: String in format "productname_typename.field1.field2..."
            
        Returns:
            Dict with parsed components
        """
        parts = input_string.split('.')
        
        if not parts:
            raise ValueError("Invalid input format. Expected: productname_typename.fields...")
        
        # Split the first part to get product name and type name
        first_part = parts[0]
        if '_' in first_part:
            product_name, type_name = first_part.rsplit('_', 1)
        else:
            product_name = first_part
            type_name = ""
        
        fields = parts[1:] if len(parts) > 1 else []
        
        return {
            'product_name': product_name,
            'type_name': type_name,
            'fields': fields,
            'full_path': input_string
        }
    
    # def build_search_queries(self, parsed_input: Dict[str, str]) -> List[str]:
    #     """
    #     Build search queries to find relevant documents in ChromaDB.
        
    #     Args:
    #         parsed_input: Parsed input dictionary
            
    #     Returns:
    #         List of search queries
    #     """
    #     queries = []
        
    #     # Primary query with product and type name
    #     primary_query = f"{parsed_input['product_name']} {parsed_input['type_name']}"
    #     queries.append(primary_query)
        
    #     # Query with fields if available
    #     if parsed_input['fields']:
    #         fields_query = f"{parsed_input['product_name']} {parsed_input['type_name']} {' '.join(parsed_input['fields'])}"
    #         queries.append(fields_query)
        
    #     # Just the type name for broader context
    #     if parsed_input['type_name']:
    #         queries.append(parsed_input['type_name'])
        
    #     return queries

    def build_search_queries(self, parsed_input: Dict[str, str]) -> List[str]:
        queries = []

        # Only use type-level context for now
        if parsed_input["type_name"]:
            queries.append(parsed_input["type_name"])
            # queries.append(f"{parsed_input['product_name']} {parsed_input['type_name']}")

        return queries

    
    def retrieve_relevant_docs(self, parsed_input: Dict[str, str], top_k: int = 2) -> List[Dict]:
        """
        Retrieve relevant documents from ChromaDB.
        
        Args:
            parsed_input: Parsed input dictionary
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        search_queries = self.build_search_queries(parsed_input)
        all_results = []
        
        for query in search_queries:
            print(f"Using query: {query}")
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k
                )
                
                # Process results
                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                        distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                        
                        all_results.append({
                            'document': doc,
                            'metadata': metadata,
                            'distance': distance,
                            'query': query
                        })
            except Exception as e:
                print(f"Error querying ChromaDB with query '{query}': {e}")
                continue
        
        # Remove duplicates and sort by distance (lower is better)
        unique_results = []
        seen_docs = set()
        
        for result in sorted(all_results, key=lambda x: x['distance']):
            doc_hash = hash(result['document'])
            if doc_hash not in seen_docs:
                unique_results.append(result)
                seen_docs.add(doc_hash)
                
            if len(unique_results) >= top_k:
                break
        
        return unique_results[:top_k]
    
    def create_rag_prompt(self, original_desc: str, parsed_input: Dict[str, str], relevant_docs: List[Dict]) -> str:
        """
        Create the RAG prompt for the LLM.
        
        Args:
            parsed_input: Parsed input dictionary
            relevant_docs: List of relevant documents
            
        Returns:
            Formatted prompt string
        """
        # Format the retrieved documents
        context_docs = ""
        for i, doc in enumerate(relevant_docs, 1):
            context_docs += f"Document {i}:\n{doc['document']}\n\n"
        
        # Build field path string
        field_path = ""
        if parsed_input['fields']:
            field_path = f"Fields: {' -> '.join(parsed_input['fields'])}"
        
        rag_prompt = f"""You are an expert API documentation assistant. Your task is to generate a **clear and concise 2-line description** for a specific field in an API type, based only on the provided documentation. The original description of the field is given to help ground your response — expand on it with accurate and contextual details.

**Original Description**: {original_desc}  
**Field**: `{parsed_input['fields']}`  
**Type**: `{parsed_input['type_name']}`  
**Product**: `{parsed_input['product_name']}`  
**Full Path**: `{parsed_input['full_path']}`

**Documentation Snippets**:
{context_docs}

**Instructions**:
- Explain the type ({parsed_input['type_name']}) in one short sentence first if needed.
- Then describe what the {parsed_input['fields'][-1]} field means, using the original description as a starting point.
- Do **not** restate the full path or hierarchy.
- Be clear and factual. Avoid technical padding, generalizations, or filler phrases.

**Output Format**:  
A plain, 2-3 line paragraph that is simple and directly informative.

**Example**:  
Input: concert_PackageRecommendation.action
Output: PackageRecommendation provides recommendations to resolve package vulnerabilities and compliance issues. The action field indicates the specific remediation step to apply for the identified package issue.

Begin now:
"""
# ApplicationAssociations describes the collection of components linked to an application. The version field under build_artifacts specifies the exact versiond of a library associated with the application. Helps track and manage the precise artifact instance involved in the application's build.

        # print(rag_prompt)
        return rag_prompt
    
    def get_wxai_response(self, prompt: str, model_id: str = "mistralai/mistral-large", 
                         max_new_tokens: int = 1000, stop_sequences: List[str] = ["\n\n\n", "<|endoftext|>"]) -> str:
        """
        Get response from Watson AI model.
        
        Args:
            prompt: Input prompt
            model_id: Model identifier
            max_new_tokens: Maximum tokens to generate
            stop_sequences: Sequences to stop generation
            
        Returns:
            Generated response
        """
        parameters = {
            GenParams.MIN_NEW_TOKENS: 0,
            GenParams.MAX_NEW_TOKENS: max_new_tokens,
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.REPETITION_PENALTY: 1,
            GenParams.STOP_SEQUENCES: stop_sequences
        }

        model = ModelInference(
            model_id=model_id, 
            params=parameters, 
            credentials=self.credentials,
            project_id=self.project_id
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_text(prompt=[prompt])
                return response[0]
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise e
                continue
    
    def generate_description(self, input_string: str, original_desc: str = "", top_k: int = 2, model_id: str = "mistralai/mistral-large") -> Dict[str, any]:

        """
        Main method to generate API description.
        
        Args:
            input_string: Input in format "productname_typename.fields..."
            top_k: Number of documents to retrieve
            model_id: Model to use for generation
            
        Returns:
            Dictionary with results
        """
        try:
            # Parse input
            parsed_input = self.parse_input(input_string)
            print(f"Parsed input: {parsed_input}")
            
            # Retrieve relevant documents
            relevant_docs = self.retrieve_relevant_docs(parsed_input, top_k)
            print(f"Retrieved {len(relevant_docs)} relevant documents")
            
            if not relevant_docs:
                return {
                    'success': False,
                    'error': 'No relevant documents found in ChromaDB',
                    'input': input_string,
                    'parsed_input': parsed_input
                }
            
            # Create RAG prompt
            rag_prompt = self.create_rag_prompt(original_desc, parsed_input, relevant_docs)
            
            # Generate description
            description = self.get_wxai_response(rag_prompt, model_id)
            description = description.replace("**Output**:\n", "").replace("**Output**:", "").strip()

            return {
                'success': True,
                'input': input_string,
                'parsed_input': parsed_input,
                'relevant_docs': relevant_docs,
                'description': description,
                'prompt_used': rag_prompt
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'input': input_string
            }

    def get_description_only(self, input_string: str, original_desc: str = "", top_k: int = 2, model_id: str = "mistralai/mistral-large") -> str:

        """
        Convenience method to get only the description string.
        
        Args:
            input_string: Input in format "productname_typename.fields..."
            top_k: Number of documents to retrieve
            model_id: Model to use for generation
            
        Returns:
            Generated description string or error message
        """
        result = self.generate_description(input_string, original_desc, top_k, model_id)
        if result['success']:
            return result['description'].strip()
        else:
            return f"Error generating description: {result.get('error', 'Unknown error')}"



def main():
    """Example usage of the Concert API Description Generator."""
    
    # Initialize the generator
    generator = ConcertAPIDescriptionGenerator()
    
    # Example inputs
    test_inputs = [
        # "concert_PackageRecommendation.action",
        # "concert_PackageDetailsCollection.packages.recommendation.action",
        # "concert_PackageRecommendation.risk_type",
        # "concert_ApplicationAssociationsInfo.build_artifacts.version"
        "concert_BuildArtifact.associations.access_points.id"
    ]
    
    for input_string in test_inputs:
        print(f"\n{'='*60}")
        print(f"Processing: {input_string}")
        print(f"{'='*60}")
        
        result = generator.generate_description(
            input_string,
            original_desc="Associations of the build artifact -> Associated access points -> ID of associated component"  # works for only one example in the test_input list
        )

        if result['success']:
            print("✅ SUCCESS!")
            print(f"\nGenerated Description:")
            print("-" * 40)
            print(result['description'])
            print("-" * 40)
            print(f"Documents used: {len(result['relevant_docs'])}")
            
            # print("\nContext Documents Used:")
            # print("-" * 40)
            # for i, doc in enumerate(result["relevant_docs"], 1):
            #     print(f"[{i}] From: {doc['metadata'].get('endpoint', 'N/A')}")
            #     print(doc["document"])  # limit to 500 characters
            #     print("-" * 40)
            print("\nMetadata of Documents Used:")
            print("-" * 40)
            for i, doc in enumerate(result["relevant_docs"], 1):
                print(f"[{i}] Metadata: {doc['metadata']}")
                print("-" * 40)

if __name__ == "__main__":
    main()