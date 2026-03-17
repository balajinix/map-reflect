#!/usr/bin/env python3
"""
Simplified AnyTool-style runner for SchemaNet.

AnyTool approach:
1. Hierarchical navigation: Category -> Tool -> API
2. LLM-guided selection at each level
3. Self-reflection to verify selection

Simplified version:
- Uses WatsonX instead of OpenAI (compatible with Map-Reflect setup)
- Single-stage selection with all context
- Supports both hierarchical and flat modes
"""

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import time

# Try to import WatsonX client
try:
    from ibm_watsonx_ai.foundation_models import Model
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False
    print("WatsonX not available, using mock mode")


def get_watsonx_client(model_id="meta-llama/llama-3-3-70b-instruct"):
    """Initialize WatsonX client."""
    if not WATSONX_AVAILABLE:
        return None

    credentials = {
        "url": os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        "apikey": os.environ.get("WATSONX_APIKEY", "")
    }
    project_id = os.environ.get("WATSONX_PROJECT_ID", "")

    if not credentials["apikey"] or not project_id:
        print("Warning: WatsonX credentials not set")
        return None

    parameters = {
        GenParams.MAX_NEW_TOKENS: 512,
        GenParams.TEMPERATURE: 0.1,
        GenParams.TOP_P: 0.9,
    }

    return Model(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )


def load_anytool_data(data_dir):
    """Load AnyTool-format data files."""
    with open(os.path.join(data_dir, 'tool_data.json')) as f:
        tool_data = json.load(f)

    with open(os.path.join(data_dir, 'api_details.json')) as f:
        api_details = json.load(f)

    with open(os.path.join(data_dir, 'category_tool_details.json')) as f:
        category_tool_details = json.load(f)

    return tool_data, api_details, category_tool_details


def format_schema_context(tool_data, api_details, category_tool_details):
    """Format schema as context for LLM."""
    context_parts = []

    for category, tools in tool_data.items():
        cat_desc = category_tool_details.get(category, {}).get('description', '')
        context_parts.append(f"Schema: {category}")
        context_parts.append(f"Description: {cat_desc}")
        context_parts.append("Tables and Fields:")

        for tool_name, tool_info in tools.items():
            tool_desc = category_tool_details.get(category, {}).get('tools', {}).get(tool_name, {}).get('description', '')
            fields = tool_info.get('api_list_names', [])

            # Get field descriptions
            field_details = []
            for api in api_details.get(category, {}).get(tool_name, {}).get('api_list', []):
                field_details.append(f"  - {api['name']}: {api['description']}")

            context_parts.append(f"\n  Table: {tool_name}")
            context_parts.append(f"  Description: {tool_desc}")
            context_parts.append("  Fields:")
            context_parts.extend(field_details)

    return "\n".join(context_parts)


def create_mapping_prompt(query, schema_context, target_schema, mode="flat"):
    """Create prompt for field mapping."""
    if mode == "flat":
        prompt = f"""You are a schema mapping expert. Given a source field, find the best matching target field in the {target_schema} schema.

TARGET SCHEMA ({target_schema}):
{schema_context}

TASK:
{query}

Instructions:
1. Analyze the source field name and its semantic meaning
2. Find the most semantically similar field in the {target_schema} schema
3. Return ONLY the answer in this exact JSON format:
{{"schema": "{target_schema}", "table": "TABLE_NAME", "field": "FIELD_NAME"}}

Replace TABLE_NAME and FIELD_NAME with the actual table and field names from the schema above.

Answer:"""
    else:  # hierarchical
        prompt = f"""You are a schema mapping expert using hierarchical search.

TARGET SCHEMA ({target_schema}):
{schema_context}

TASK:
{query}

Step 1: The target schema is {target_schema}
Step 2: Which table in {target_schema} contains the matching field?
Step 3: Which field in that table is the best match?

Return ONLY the answer in this exact JSON format:
{{"schema": "{target_schema}", "table": "TABLE_NAME", "field": "FIELD_NAME"}}

Answer:"""

    return prompt


def parse_llm_response(response, target_schema=None):
    """Parse LLM response to extract category, table, field."""
    import re

    result = {
        'category': target_schema,  # Default to target schema
        'table': None,
        'field': None
    }

    # Try to parse as JSON first
    try:
        # Find JSON in response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            parsed = json.loads(json_match.group())
            result['category'] = parsed.get('schema', target_schema)
            result['table'] = parsed.get('table')
            result['field'] = parsed.get('field')
            return result
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback to line-by-line parsing
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.lower().startswith('category:') or line.lower().startswith('schema:'):
            result['category'] = line.split(':', 1)[1].strip()
        elif line.lower().startswith('table:'):
            result['table'] = line.split(':', 1)[1].strip()
        elif line.lower().startswith('field:'):
            result['field'] = line.split(':', 1)[1].strip()

    return result


def evaluate_prediction(prediction, ground_truth):
    """Evaluate prediction against ground truth."""
    gt = ground_truth[0]  # First (only) ground truth

    # Field-level match
    field_match = (
        prediction['category'] == gt['category_name'] and
        prediction['table'] == gt['tool_name'] and
        prediction['field'] == gt['api_name']
    )

    # Type-level match (table match)
    type_match = (
        prediction['category'] == gt['category_name'] and
        prediction['table'] == gt['tool_name']
    )

    return {
        'field_correct': field_match,
        'type_correct': type_match
    }


def run_anytool_schemanet(dataset_name, data_dir, queries_file, model=None, mode="flat", max_samples=None):
    """Run AnyTool-style evaluation on SchemaNet."""

    # Load data
    tool_data, api_details, category_tool_details = load_anytool_data(data_dir)

    # Load queries
    with open(queries_file) as f:
        queries = json.load(f)

    if max_samples:
        queries = queries[:max_samples]

    # Get target schema name (first category in tool_data)
    target_schema = list(tool_data.keys())[0]

    # Format schema context
    schema_context = format_schema_context(tool_data, api_details, category_tool_details)

    results = []
    field_correct = 0
    type_correct = 0

    for query_data in tqdm(queries, desc=f"Processing {dataset_name}"):
        query = query_data['query']
        gt_api_list = query_data['gt_api_list']

        # Create prompt
        prompt = create_mapping_prompt(query, schema_context, target_schema, mode)

        # Get LLM response
        if model:
            try:
                response = model.generate_text(prompt)
            except Exception as e:
                print(f"Error: {e}")
                response = ""
        else:
            # Mock response for testing
            response = f"Category: {gt_api_list[0]['category_name']}\nTable: {gt_api_list[0]['tool_name']}\nField: unknown"

        # Parse response
        prediction = parse_llm_response(response, target_schema)

        # Evaluate
        eval_result = evaluate_prediction(prediction, gt_api_list)

        if eval_result['field_correct']:
            field_correct += 1
        if eval_result['type_correct']:
            type_correct += 1

        results.append({
            'query_id': query_data['query_id'],
            'source_field': query_data.get('source_field'),
            'expected': query_data.get('expected_target_field'),
            'predicted': f"{prediction['category']}_{prediction['table']}.{prediction['field']}",
            'field_correct': eval_result['field_correct'],
            'type_correct': eval_result['type_correct'],
            'raw_response': response[:200]
        })

    total = len(queries)
    return {
        'dataset': dataset_name,
        'total': total,
        'field_correct': field_correct,
        'type_correct': type_correct,
        'field_accuracy': field_correct / total if total > 0 else 0,
        'type_accuracy': type_correct / total if total > 0 else 0,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Run AnyTool on SchemaNet')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., schemanet_bank1_bank2)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory with AnyTool-format data')
    parser.add_argument('--queries-file', type=str, required=True,
                        help='Path to queries JSON file')
    parser.add_argument('--model', type=str, default='meta-llama/llama-3-3-70b-instruct',
                        help='WatsonX model ID')
    parser.add_argument('--mode', type=str, choices=['flat', 'hierarchical'], default='flat',
                        help='Selection mode')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to process')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock responses (for testing)')

    args = parser.parse_args()

    # Initialize model
    if args.mock:
        model = None
        print("Running in mock mode")
    else:
        model = get_watsonx_client(args.model)
        if model is None:
            print("No model available, running in mock mode")

    # Run evaluation
    results = run_anytool_schemanet(
        args.dataset,
        args.data_dir,
        args.queries_file,
        model=model,
        mode=args.mode,
        max_samples=args.max_samples
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"AnyTool Results: {args.dataset}")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Total samples: {results['total']}")
    print(f"Field Accuracy: {results['field_accuracy']:.1%} ({results['field_correct']}/{results['total']})")
    print(f"Type Accuracy: {results['type_accuracy']:.1%} ({results['type_correct']}/{results['total']})")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
