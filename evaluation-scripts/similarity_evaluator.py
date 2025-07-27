#!/usr/bin/env python3
import json
import argparse
import os
from sentence_transformers import SentenceTransformer
import numpy as np

def compute_api_similarity(api1_details: dict, api2_details: dict) -> float:
    """
    Compute cosine similarity between two API field descriptions using embeddings.
    """
    def get_text(details: dict) -> str:
        parts = []
        for key in ['field_name', 'description']:   #'product_name', 'type_name' -> excluding these improved the cosine sim score calculation
            value = details.get(key, "")
            if value:
                parts.append(value)
        return " ".join(parts)

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    text1 = get_text(api1_details)
    text2 = get_text(api2_details)
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return max(0.0, min(1.0, cosine_sim))


class APIMappingEvaluator:
    def __init__(self, mapping_file: str, api_lookup_file: str):
        with open(mapping_file, 'r') as f:
            self.mapping_json = json.load(f)

        with open(api_lookup_file, 'r') as f:
            self.api_lookup = json.load(f)

    def parse_mappings(self) -> list:
        """
        Convert nested mapping format to flat list:
        {"api1": ..., "api2": ..., "reason": ...}
        """
        mapping_pairs = []
        properties = self.mapping_json.get("properties", {})

        for source_api, targets in properties.items():
            for mapping in targets:
                target_api = mapping.get("mapProperty")
                justification = mapping.get("justification", "")
                mapping_pairs.append({
                    "api1": source_api,
                    "api2": target_api,
                    "reason": justification
                })
        return mapping_pairs

    def compute_scores(self, mapping_pairs: list) -> list:
        for pair in mapping_pairs:
            api1 = pair["api1"]
            api2 = pair["api2"]
            api1_details = self.api_lookup.get(api1, {})
            api2_details = self.api_lookup.get(api2, {})
            score = compute_api_similarity(api1_details, api2_details)
            pair["score"] = f"{score:.4f}"
            pair["api1_details"] = api1_details
            
            pair["api2_details"] = api2_details
        return mapping_pairs

    def evaluate_mapping_scores(self, mapping_pairs, threshold=0.3):  #threshold of 0.3 
        total_mappings = len(mapping_pairs)
        correct_mappings = sum(
            1 for pair in mapping_pairs if float(pair.get("score", 0)) > threshold
        )
        accuracy = correct_mappings / total_mappings if total_mappings > 0 else 0.0
        return {
            "total_mappings": total_mappings,
            "correct_mappings": correct_mappings,
            "accuracy": round(accuracy, 4)
        }

    def run(self) -> list:
        pairs = self.parse_mappings()
        pairs_with_scores = self.compute_scores(pairs)

        try:
            metrics = self.evaluate_mapping_scores(pairs_with_scores)
            print(metrics)
        except Exception as e:
            print(str(e))
        return pairs_with_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate API mappings using embeddings.")
    parser.add_argument("--mapping_file", required=True, help="Mapping file (nested JSON format)")
    parser.add_argument("--api_lookup_file", required=True, help="Flattened API schema lookup JSON file")
    parser.add_argument("--output", required=True, help="Output file to save results")
    args = parser.parse_args()

    evaluator = APIMappingEvaluator(args.mapping_file, args.api_lookup_file)
    results = evaluator.run()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Output saved to {args.output}")
