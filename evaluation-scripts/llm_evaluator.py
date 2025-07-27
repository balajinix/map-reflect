""" 
Uses an LLM as a judge to evaluate a single json file of mappings, just for testing. 
Returns an evaluation of joinability and equivalence scores. 
"""

import argparse, json, os, re
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
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
        print(api1.get("type_name", ""))
        print(api1.get("field_name", ""))
        print(api1.get("description", ""))
        print("--- & ---")
        print(api2.get("type_name", ""))
        print(api2.get("field_name", ""))
        print(api2.get("description", ""))


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

            print("WatsonX raw response:", response)  
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
            # api.get("product_name", ""),
            # api.get("type_name", ""),
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping_file", required=True)
    ap.add_argument("--api_lookup_file", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    evaluator = APIMappingEvaluator(args.mapping_file, args.api_lookup_file)
    result = evaluator.run()
    json.dump(result, open(args.output, "w"), indent=2)
    print("Saved", args.output)
