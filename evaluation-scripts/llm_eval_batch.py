""" 
Uses an LLM as a judge to evaluate all json files in a directory. 
Returns an evaluation of joinability and equivalence scores, thresholds the scores and reports the positive rate. 
"""


import argparse, json, os, re, glob
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials

load_dotenv() 

EMBED_MODEL  = "paraphrase-MiniLM-L6-v2"
JOIN_HEUR_THRESHOLD = 0.15

_embedder = SentenceTransformer(EMBED_MODEL)

class WatsonXJudge:
    def __init__(self):
        self.credentials = Credentials(
            url="https://us-south.ml.cloud.ibm.com",
            api_key=os.environ["WATSONX_APIKEY"]
        )
        self.project_id = os.environ["WATSONX_PROJECT_ID"]

    def get_response(self, api1: dict, api2: dict, reason: str) -> Tuple[float, float, str]:
        # print(api1.get("type_name", "")).  # for DEBUGGING
        # print(api1.get("field_name", ""))
        # print(api1.get("description", ""))
        # print("--- & ---")
        # print(api2.get("type_name", ""))
        # print(api2.get("field_name", ""))
        # print(api2.get("description", ""))

        prompt = f"""
You are an API-mapping evaluator.

Your job is to judge whether two API schema fields are functionally equivalent and/or joinable based on their type names and descriptions.

---
Definitions:
- equivalence_score: a float between 0 and 1 indicating whether the two fields represent the same core function.
It measures how much the two fields serve similar purposes *within the broader product schemas they belong to*, and if both fields would serve a given user query. 
- joinability_score: a float between 0 and 1 indicating if the output of one field can be used to populate or enrich the the other.
Represents whether one field's output can help populate or enrich the other. Joinability reflects data flow produced in one schema might feed into the second schema.
- explanation: clearly explain the reasoning for both the equivalence and joinability scores in two short clauses.

Instructions:
1. First evaluate the **type-level intent** of each schema. Assess if the two fields come from  compatible or complementary type names — that is, do their parent types indicate roles that interact logically (e.g. Action and Recommendation).
2. Then examine the **field-level semantics and descriptions** assess equivalence and joinability. Do the fields describe the same concept, or do they feed into one another?
3. If type names are unrelated or incompatible, joinability_score should be low unless clearly overridden by field semantics.

Rules:
- Output must be a single valid JSON object with exactly these keys: \"equivalence_score\", \"joinability_score\", \"explanation\".
- Do not add any commentary, formatting, markdown, or prose outside the JSON.
- Only output the JSON object. No prefix or suffix.

---
Examples:
### API-A: turbo_Action.actionType (type: turbo_Action)
### API-B: concert_PackageRecommendation.action (type: concert_PackageRecommendation)
{{
  /"equivalence_score/": 0.95,
  /"joinability_score/": 0.5,
  /"explanation/": "Both describe the nature of an action, hence equivalent. Less joinability as one does not feed into another."
}}

### API-A: turbo_Action.createTime (type: turbo_Action)
### API-B: concert_PackageDetailsCollection.packages.recommendation.created_date (type: concert_PackageDetailsCollection)
{{
  /"equivalence_score/": 0.4,
  /"joinability_score/": 0.85,
  /"explanation/": "Different names but same meaning for creation timestamps. High joinability for tracking created actions."
}}

---
### Endpoint A
Name: {api1.get("field_name")} (type: {api1.get("type_name")})
Description: {api1.get("description", "")}

### Endpoint B
Name: {api2.get("field_name")} (type: {api2.get("type_name")})
Description: {api2.get("description", "")}

User justification: {reason}

Now return the JSON object:
"""

        parameters = {
            GenParams.MIN_NEW_TOKENS: 0,
            GenParams.MAX_NEW_TOKENS: 1000,
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.REPETITION_PENALTY: 1,
            GenParams.STOP_SEQUENCES: ["\n\n\n", "<|endoftext|>"]
        }

        model = ModelInference(
            model_id="mistralai/mistral-large",
            params=parameters,
            credentials=self.credentials,
            project_id=self.project_id
        )

        try:
            response = model.generate_text(prompt=[prompt])[0].strip()
            if response.startswith("```"):
                response = re.sub(r"```json|```", "", response).strip()
            data = json.loads(response)

            # print("WatsonX raw response:", response)  # for DEBUGGING
            data = json.loads(response)
            eq = float(data["equivalence_score"])
            jo = float(data["joinability_score"])
            expl = data.get("explanation", "").strip()
        except Exception as e:
            eq, jo, expl = 0.0, 0.0, f"parse error: {str(e)}"

        return eq, jo, expl

def compute_api_similarity(api1: dict, api2: dict) -> float:
    def flatten(api: dict) -> str:
        parts = [
            # api.get("product_name", ""),   # Remove noise while computing similarity. This improved the scores. 
            # api.get("type_name", ""),      # Description covers the type name already :)
            api.get("field_name", ""),
            api.get("description", "")
        ]
        return " ".join(str(x) for x in parts if x)

    emb1, emb2 = _embedder.encode([flatten(api1), flatten(api2)])
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

class APIMappingEvaluator:
    def __init__(self, mapping_file: str, api_lookup_file: str):
        self.mapping = json.load(open(mapping_file))
        self.lookup = json.load(open(api_lookup_file))
        self.judge = WatsonXJudge()

    def run(self):
        enriched = []
        for field, mappings in self.mapping.get("properties", {}).items():
            for m in mappings:
                api1 = self.lookup.get(field, {})
                api2 = self.lookup.get(m["mapProperty"], {})
                sim = compute_api_similarity(api1, api2)
                eq, jo, expl = self.judge.get_response(api1, api2, m.get("justification", ""))
                enriched.append({
                    "source": field,
                    "target": m["mapProperty"],
                    "score": f"{sim:.4f}",
                    "equivalence_score": f"{eq:.4f}",
                    "joinability_score": f"{jo:.4f}",
                    "judge_explanation": expl,
                    "justification": m.get("justification")
                })
        return enriched

class BatchAPIMappingEvaluator:
    def __init__(self, mapping_dir: str, api_lookup_file: str, output_file: str):
        self.mapping_dir = mapping_dir
        self.api_lookup_file = api_lookup_file
        self.output_file = output_file
        
        # Load API lookup once
        self.lookup = json.load(open(api_lookup_file))
        self.judge = WatsonXJudge()
        
    def process_single_mapping(self, mapping_file: str) -> List[Dict]:
        """Process a single mapping file"""
        print(f"Processing: {mapping_file}")
        
        mapping = json.load(open(mapping_file))
        enriched = []
        
        for field, mappings in mapping.get("properties", {}).items():
            for m in mappings:
                api1 = self.lookup.get(field, {})
                api2 = self.lookup.get(m["mapProperty"], {})
                sim = compute_api_similarity(api1, api2)
                eq, jo, expl = self.judge.get_response(api1, api2, m.get("justification", ""))
                
                result = {
                    "mapping_file": os.path.basename(mapping_file),
                    "source": field,
                    "target": m["mapProperty"],
                    "score": f"{sim:.4f}",
                    "equivalence_score": f"{eq:.4f}",
                    "joinability_score": f"{jo:.4f}",
                    "judge_explanation": expl,
                    "justification": m.get("justification")
                }
                enriched.append(result)
        
        return enriched
    
    def calculate_positive_rate(self, results: List[Dict], score_type: str, threshold: float = 0.5) -> Dict:
        """Calculate positive rate for a given score type"""
        total_predictions = len(results)
        positive_predictions = sum(1 for result in results if float(result[score_type]) > threshold)
        
        return {
            "total_predictions": total_predictions,
            "positive_predictions": positive_predictions,
            "negative_predictions": total_predictions - positive_predictions,
            "positive_rate": positive_predictions / total_predictions if total_predictions > 0 else 0
        }
    
    def run_batch_evaluation(self):
        """Run evaluation on all mapping files in the directory"""
        # Get all JSON files in the mapping directory
        mapping_files = glob.glob(os.path.join(self.mapping_dir, "*.json"))
        
        if not mapping_files:
            print(f"No JSON files found in {self.mapping_dir}")
            return
        
        print(f"Found {len(mapping_files)} mapping files")
        
        all_results = []
        
        for mapping_file in mapping_files:
            try:
                results = self.process_single_mapping(mapping_file)
                all_results.extend(results)
                
            except Exception as e:
                print(f"Error processing {mapping_file}: {str(e)}")
        
        # Save all results
        with open(self.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Saved all results to: {self.output_file}")
        
        # Calculate and print metrics
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        
        # Equivalence metrics
        equivalence_metrics = self.calculate_positive_rate(all_results, "equivalence_score", 0.5)
        print(f"\nEQUIVALENCE (threshold > 0.5):")
        print(f"  Total predictions: {equivalence_metrics['total_predictions']}")
        print(f"  Positive predictions: {equivalence_metrics['positive_predictions']}")
        print(f"  Negative predictions: {equivalence_metrics['negative_predictions']}")
        print(f"  Positive rate: {equivalence_metrics['positive_rate']:.4f}")
        
        # Joinability metrics
        joinability_metrics = self.calculate_positive_rate(all_results, "joinability_score", 0.5)
        print(f"\nJOINABILITY (threshold > 0.5):")
        print(f"  Total predictions: {joinability_metrics['total_predictions']}")
        print(f"  Positive predictions: {joinability_metrics['positive_predictions']}")
        print(f"  Negative predictions: {joinability_metrics['negative_predictions']}")
        print(f"  Positive rate: {joinability_metrics['positive_rate']:.4f}")
        
        return all_results

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch API Mapping Evaluator")
    ap.add_argument("--mapping_dir", required=True, help="Directory containing JSON mapping files")
    ap.add_argument("--api_lookup_file", required=True, help="JSON file containing API lookup data")
    ap.add_argument("--output", required=True, help="Output JSON file for results")
    args = ap.parse_args()
    
    # Create batch evaluator
    batch_evaluator = BatchAPIMappingEvaluator(
        mapping_dir=args.mapping_dir,
        api_lookup_file=args.api_lookup_file,
        output_file=args.output
    )
    
    # Run batch evaluation
    results = batch_evaluator.run_batch_evaluation()
    
    print(f"\nBatch evaluation completed!")
    print(f"Results saved to: {args.output}")