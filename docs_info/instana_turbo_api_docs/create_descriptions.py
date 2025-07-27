"""
Functions to create descriptions for each endpoint in Instana and Turbo API (routes the endpoint based on the product from file name). Can be called to populate the API lookup etc. 
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


class InstanaAPIDescriptionGenerator:
    def __init__(self, chroma_path: str = "/data/wikidata/chromadb", 
                 instana_collection_name: str = "instana_api",
                 turbo_collection_name: str = "turbo_api"):
        """Initialize the RAG system for API description generation with multiple collections."""
        self.chroma_path = chroma_path
        self.instana_collection_name = instana_collection_name
        self.turbo_collection_name = turbo_collection_name
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        # Initialize collections
        self.instana_collection = None
        self.turbo_collection = None
        
        # Try to get collections, create if they don't exist
        try:
            self.instana_collection = self.client.get_collection(name=instana_collection_name)
            print(f"Successfully loaded collection: {instana_collection_name}")
        except Exception as e:
            print(f"Warning: Could not load collection {instana_collection_name}: {e}")
        
        try:
            self.turbo_collection = self.client.get_collection(name=turbo_collection_name)
            print(f"Successfully loaded collection: {turbo_collection_name}")
        except Exception as e:
            print(f"Warning: Could not load collection {turbo_collection_name}: {e}")
        
        # Load environment variables
        load_dotenv()
        
        # Initialize Watson AI credentials
        self.credentials = Credentials(
            url="https://us-south.ml.cloud.ibm.com",
            api_key=os.environ["WATSONX_APIKEY"]
        )
        self.project_id = os.environ["WATSONX_PROJECT_ID"]
    
    def get_collection_for_product(self, product_name: str):
        """
        Get the appropriate ChromaDB collection based on product name.
        
        Args:
            product_name: The product name (e.g., 'instana', 'turbo')
            
        Returns:
            ChromaDB collection object or None if not found
        """
        product_name_lower = product_name.lower()
        
        if product_name_lower == "instana":
            if self.instana_collection is None:
                print(f"Warning: Instana collection '{self.instana_collection_name}' not available")
                return None
            return self.instana_collection
        elif product_name_lower == "turbo":
            if self.turbo_collection is None:
                print(f"Warning: Turbo collection '{self.turbo_collection_name}' not available")
                return None
            return self.turbo_collection
        else:
            print(f"Warning: Unknown product name '{product_name}'. Supported products: instana, turbo")
            return None
    
    def parse_input(self, input_string: str) -> Dict[str, str]:
        """
        Parse input string in format: productname_typename_typename.field1.field2...
        
        Args:
            input_string: String in format "productname_typename_typename.field1.field2..."
            
        Returns:
            Dict with parsed components
        """
        # Split at the first dot into head and fields
        if '.' not in input_string:
            raise ValueError("Input must contain at least one field separated by a dot.")
        
        head, *field_parts = input_string.split('.')
        fields = field_parts

        # Now split head only at the first underscore
        if '_' not in head:
            raise ValueError("Input must contain an underscore between product name and type name.")
        
        product_name, type_name = head.split('_', 1)

        return {
            'product_name': product_name,
            'type_name': type_name,
            'fields': fields,
            'full_path': input_string
        }

    def build_search_queries(self, parsed_input: Dict[str, str]) -> List[str]:
        queries = []
        if parsed_input["type_name"]:
            polished = parsed_input["type_name"].replace("_200_response", "").replace("_", " ")
            split_camel = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', polished)
            queries.append(split_camel.lower())

        return queries

    def retrieve_relevant_docs(self, parsed_input: Dict[str, str], top_k: int = 2) -> List[Dict]:
        """
        Retrieve relevant documents from the appropriate ChromaDB collection.
        
        Args:
            parsed_input: Parsed input dictionary
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        # Get the appropriate collection based on product name
        collection = self.get_collection_for_product(parsed_input['product_name'])
        
        if collection is None:
            print(f"No collection available for product: {parsed_input['product_name']}")
            return []
        
        search_queries = self.build_search_queries(parsed_input)
        all_results = []
        
        print(f"Using collection for product '{parsed_input['product_name']}'")
        
        for query in search_queries:
            print(f"Using query: {query}")
            try:
                results = collection.query(
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
        
        polished_type = parsed_input["type_name"].replace("_200_response", "").replace("_", " ")

        # Build field path string
        field_path = ""
        if parsed_input['fields']:
            field_path = f"Fields: {' -> '.join(parsed_input['fields'])}"
        
        rag_prompt = f"""You are an expert API documentation assistant. Your task is to generate a **clear and concise 2-line description** for a specific field in an API type, based only on the provided documentation. The original description of the field is given to help ground your response — expand on it with accurate and contextual details.

**Original Description**: {original_desc}  
**Field**: `{parsed_input['fields']}`  
**Type**: `{polished_type}`
**Product**: `{parsed_input['product_name']}`  
**Full Path**: `{parsed_input['full_path']}`

**Documentation Snippets**:
{context_docs}

**Instructions**:
- Explain the type ({polished_type}) in one short sentence first if needed.
- Then describe what the {parsed_input['fields'][-1]} field means, using the original description as a starting point.
- Do not restate the full path or hierarchy.
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
            
            # Retrieve relevant documents from appropriate collection
            relevant_docs = self.retrieve_relevant_docs(parsed_input, top_k)
            print(f"Retrieved {len(relevant_docs)} relevant documents from {parsed_input['product_name']} collection")
            
            if not relevant_docs:
                return {
                    'success': False,
                    'error': f'No relevant documents found in ChromaDB for product: {parsed_input["product_name"]}',
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
    """Example usage of the API Description Generator."""
    
    # Initialize the generator
    generator = InstanaAPIDescriptionGenerator()
    
    # Example inputs - now supporting both instana and turbo
    test_inputs = [
        # "instana_mutation_manuallyCloseEvent_problem.id",
        # "instana__entities_200_response.items.entityHealthInfo.openIssues.metricAccessId"
        # "turbo_Action.compoundActions.actionState",
        "turbo_Action.actionID"
    ]
    
    for input_string in test_inputs:
        print(f"\n{'='*60}")
        print(f"Processing: {input_string}")
        print(f"{'='*60}")
        
        result = generator.generate_description(
            input_string,
            original_desc="None"  # works for only one example in the test_input list
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
        else:
            print("❌ FAILED!")
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()