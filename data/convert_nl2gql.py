#!/usr/bin/env python3
import json

input_file = "nl2gql_dataset_raw.jsonl"
output_file = "nl2gql_dataset.jsonl"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for idx, line in enumerate(infile):
        data = json.loads(line)
        result = data.get("result", {})
        properties = result.get("properties", {})

        # Flatten ground truth mapping
        ground_truth_mapping = {}
        expected_target = ""
        for src_field, mappings in properties.items():
            if mappings:
                ground_truth_mapping[src_field] = mappings[0]["mapProperty"]
                expected_target = mappings[0]["mapTable"]

        converted = {
            "id": f"nl2gql_{idx:03d}",
            "utterance": data.get("mapping_utterance", data.get("question", "")),
            "source_schema": result.get("name", ""),
            "target_schema": expected_target,
            "mapping": ground_truth_mapping,
            "metadata": {
                "original_question": data.get("question", ""),
                "query": data.get("query", "")
            }
        }
        outfile.write(json.dumps(converted) + "\n")

print(f"✅ Converted NL2GQL dataset saved to {output_file}")
