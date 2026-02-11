#!/usr/bin/env python3
"""
Batch Coverage Validation - Fast type mapping discovery.

Redesigned approach:
1. Filter to high-similarity candidates (>= 0.7)
2. One LLM call per type pair (batch all fields)
3. Extract validated mappings

Expected: ~108 candidates, ~30 minutes total.
"""

import json
import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

# Configuration
DEFAULT_SIMILARITY_THRESHOLD = 0.7
MAX_RETRIES = 3

# WatsonX imports
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
    from ibm_watsonx_ai.foundation_models import ModelInference
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False


class WatsonXClient:
    """Simple WatsonX client for batch validation."""

    def __init__(self, model_id: str = "meta-llama/llama-3-3-70b-instruct"):
        if not WATSONX_AVAILABLE:
            raise ImportError("ibm_watsonx_ai not installed")

        api_key = os.getenv("WATSONX_API_KEY") or os.getenv("WATSONX_APIKEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

        if not api_key or not project_id:
            raise RuntimeError("WATSONX_API_KEY and WATSONX_PROJECT_ID required")

        credentials = Credentials(url=url, api_key=api_key)

        # Llama 3 stop sequences
        stop_sequences = ["<|eot_id|>", "<|end_of_text|>"]

        parameters = {
            GenParams.MAX_NEW_TOKENS: 2000,
            GenParams.TEMPERATURE: 0.1,
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.STOP_SEQUENCES: stop_sequences,
        }

        self.model = ModelInference(
            model_id=model_id,
            params=parameters,
            credentials=credentials,
            project_id=project_id
        )
        self.model_id = model_id

    def generate(self, prompt: str, max_retries: int = MAX_RETRIES) -> str:
        """Generate response with retry logic."""
        # Format for Llama 3
        formatted = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        for attempt in range(max_retries):
            try:
                response = self.model.generate_text(prompt=formatted)
                return response[0] if isinstance(response, list) else response
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(x in error_str for x in [
                    "500", "502", "503", "504", "rate limit", "timeout", "connection"
                ])
                if is_retryable and attempt < max_retries - 1:
                    wait_time = 2 ** attempt * 5
                    print(f"  Retry {attempt+1}/{max_retries} in {wait_time}s: {str(e)[:50]}")
                    time.sleep(wait_time)
                else:
                    raise


def load_coverage_dataset(dataset_file: Path) -> List[Dict]:
    """Load the coverage dataset."""
    samples = []
    with open(dataset_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def load_api_lookup(lookup_file: Path) -> Dict:
    """Load the API lookup for field descriptions."""
    with open(lookup_file, 'r') as f:
        return json.load(f)


def filter_high_similarity(samples: List[Dict], threshold: float) -> List[Dict]:
    """Filter to only high-similarity candidates."""
    filtered = []
    for sample in samples:
        similarity = sample.get('metadata', {}).get('similarity', 0)
        if similarity >= threshold:
            filtered.append(sample)
    return filtered


def get_field_info(api_lookup: Dict, field_names: List[str]) -> List[Dict]:
    """Get field info from API lookup."""
    fields = []
    for name in field_names:
        info = api_lookup.get(name, {})
        fields.append({
            'name': name,
            'short_name': name.split('.')[-1] if '.' in name else name,
            'description': info.get('description', '') if isinstance(info, dict) else '',
            'type': info.get('field_type', '') if isinstance(info, dict) else ''
        })
    return fields


def create_batch_prompt(
    source_type: str,
    target_type: str,
    source_fields: List[Dict],
    target_fields: List[Dict],
    source_product: str,
    target_product: str
) -> str:
    """Create a single prompt to map all fields at once."""

    # Format source fields
    source_list = []
    for f in source_fields[:30]:  # Limit to 30 fields
        desc = f" ({f['description'][:50]})" if f['description'] else ""
        source_list.append(f"  - {f['short_name']}{desc}")

    # Format target fields
    target_list = []
    for f in target_fields[:30]:  # Limit to 30 fields
        desc = f" ({f['description'][:50]})" if f['description'] else ""
        target_list.append(f"  - {f['short_name']}{desc}")

    source_str = "\n".join(source_list)
    target_str = "\n".join(target_list)

    prompt = f"""You are a schema mapping expert. Determine if fields from the source type can be mapped to the target type.

SOURCE: {source_product} / {source_type}
Fields:
{source_str}

TARGET: {target_product} / {target_type}
Fields:
{target_str}

TASK: Find semantically equivalent field mappings between source and target.
- Only map fields that represent the SAME data concept
- A type mapping is VALID if at least one field can be confidently mapped

Return a JSON object with this exact structure:
{{
  "valid": true/false,
  "confidence": 0.0-1.0,
  "mappings": {{
    "source_field": "target_field",
    ...
  }},
  "reason": "brief explanation"
}}

Return ONLY the JSON object, no other text."""

    return prompt


def parse_llm_response(response: str) -> Optional[Dict]:
    """Parse the LLM response to extract JSON."""
    # Try to find JSON object
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.S)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try to parse the whole response
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    return None


def validate_type_pair(
    client: WatsonXClient,
    sample: Dict,
    api_lookup: Dict
) -> Dict:
    """Validate a single type pair with batch field mapping."""

    source_type = sample.get('source_type', '')
    target_type = sample.get('target_type', '')
    source_product = sample.get('source_schema', '')
    target_product = sample.get('target_schema', '')

    # Get field info
    source_fields = get_field_info(api_lookup, sample.get('source_fields', []))
    target_fields = get_field_info(api_lookup, sample.get('target_fields', []))

    # Create prompt
    prompt = create_batch_prompt(
        source_type=source_type,
        target_type=target_type,
        source_fields=source_fields,
        target_fields=target_fields,
        source_product=source_product,
        target_product=target_product
    )

    # Call LLM
    try:
        response = client.generate(prompt)
        result = parse_llm_response(response)

        if result:
            return {
                'sample_id': sample.get('id', ''),
                'source_type': source_type,
                'target_type': target_type,
                'source_product': source_product,
                'target_product': target_product,
                'similarity': sample.get('metadata', {}).get('similarity', 0),
                'valid': result.get('valid', False),
                'confidence': result.get('confidence', 0),
                'mappings': result.get('mappings', {}),
                'reason': result.get('reason', ''),
                'raw_response': response[:500]
            }
        else:
            return {
                'sample_id': sample.get('id', ''),
                'source_type': source_type,
                'target_type': target_type,
                'valid': False,
                'confidence': 0,
                'mappings': {},
                'reason': 'Failed to parse response',
                'raw_response': response[:500]
            }
    except Exception as e:
        return {
            'sample_id': sample.get('id', ''),
            'source_type': source_type,
            'target_type': target_type,
            'valid': False,
            'confidence': 0,
            'mappings': {},
            'reason': f'Error: {str(e)[:100]}',
            'error': True
        }


def update_type_mapping_lookup(
    results: List[Dict],
    existing_lookup: Dict,
    min_confidence: float = 0.6
) -> Dict:
    """Update the type mapping lookup with validated mappings."""
    new_lookup = existing_lookup.copy()

    for result in results:
        if not result.get('valid', False):
            continue
        if result.get('confidence', 0) < min_confidence:
            continue
        if not result.get('mappings'):
            continue

        source_type = result.get('source_type', '')
        target_type = result.get('target_type', '')

        if not source_type or not target_type:
            continue

        # Add to lookup
        if source_type not in new_lookup:
            new_lookup[source_type] = {}

        # Extract target type name (without product prefix for lookup format)
        target_type_short = target_type.split('_', 1)[-1] if '_' in target_type else target_type

        if target_type_short not in new_lookup[source_type]:
            new_lookup[source_type][target_type_short] = ['Batch-Validation (Llama)']
        elif 'Batch-Validation (Llama)' not in new_lookup[source_type][target_type_short]:
            new_lookup[source_type][target_type_short].append('Batch-Validation (Llama)')

    return new_lookup


def main():
    parser = argparse.ArgumentParser(description='Batch coverage validation')
    parser.add_argument('--dataset', type=str,
                        default=str(DATA_DIR / 'itopsgraph_coverage_dataset.jsonl'),
                        help='Coverage dataset file')
    parser.add_argument('--api-lookup', type=str,
                        default=str(DATA_DIR / 'itopsgraph_api_lookup.json'),
                        help='API lookup file')
    parser.add_argument('--output', type=str,
                        default=str(DATA_DIR / 'batch_coverage_results.json'),
                        help='Output results file')
    parser.add_argument('--lookup-file', type=str,
                        default=str(DATA_DIR / 'itops_type_mapping_lookup.json'),
                        help='Existing type mapping lookup')
    parser.add_argument('--output-lookup', type=str,
                        default=str(DATA_DIR / 'itops_type_mapping_lookup_expanded.json'),
                        help='Output expanded lookup file')
    parser.add_argument('--similarity-threshold', type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
                        help='Minimum similarity threshold')
    parser.add_argument('--model', type=str, default='meta-llama/llama-3-3-70b-instruct',
                        help='WatsonX model ID')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without calling LLM')
    args = parser.parse_args()

    print("=" * 70)
    print("Batch Coverage Validation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Similarity threshold: {args.similarity_threshold}")
    print(f"  Model: {args.model}")
    print(f"  Dry run: {args.dry_run}")

    # Load data
    print("\nLoading data...")
    samples = load_coverage_dataset(Path(args.dataset))
    api_lookup = load_api_lookup(Path(args.api_lookup))
    print(f"  Total samples: {len(samples)}")

    # Filter to high similarity
    filtered = filter_high_similarity(samples, args.similarity_threshold)
    print(f"  After filtering (>= {args.similarity_threshold}): {len(filtered)}")

    if not filtered:
        print("\nNo samples meet the similarity threshold!")
        return

    # Show distribution
    print(f"\n  Similarity distribution of filtered samples:")
    sims = [s.get('metadata', {}).get('similarity', 0) for s in filtered]
    print(f"    Min: {min(sims):.3f}")
    print(f"    Max: {max(sims):.3f}")
    print(f"    Mean: {sum(sims)/len(sims):.3f}")

    if args.dry_run:
        print("\n[DRY RUN] Would process these samples:")
        for i, s in enumerate(filtered[:10]):
            print(f"  {i+1}. {s.get('source_type', '')} -> {s.get('target_type', '')} (sim={s.get('metadata', {}).get('similarity', 0):.3f})")
        if len(filtered) > 10:
            print(f"  ... and {len(filtered) - 10} more")
        return

    # Initialize client
    print("\nInitializing WatsonX client...")
    client = WatsonXClient(model_id=args.model)

    # Process samples
    print(f"\nProcessing {len(filtered)} samples...")
    results = []
    valid_count = 0

    for sample in tqdm(filtered, desc="Validating"):
        result = validate_type_pair(client, sample, api_lookup)
        results.append(result)

        if result.get('valid', False):
            valid_count += 1

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    # Summary
    print(f"\n{'=' * 70}")
    print("Results Summary")
    print(f"{'=' * 70}")
    print(f"  Total processed: {len(results)}")
    print(f"  Valid mappings: {valid_count}")
    print(f"  Invalid/failed: {len(results) - valid_count}")

    # Save results
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model': args.model,
            'similarity_threshold': args.similarity_threshold,
            'total_samples': len(samples),
            'filtered_samples': len(filtered),
            'valid_mappings': valid_count
        },
        'results': results
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Update lookup
    print("\nUpdating type mapping lookup...")
    existing_lookup = {}
    lookup_path = Path(args.lookup_file)
    if lookup_path.exists():
        with open(lookup_path, 'r') as f:
            existing_lookup = json.load(f)

    new_lookup = update_type_mapping_lookup(results, existing_lookup)
    new_entries = len(new_lookup) - len(existing_lookup)

    output_lookup_path = Path(args.output_lookup)
    with open(output_lookup_path, 'w') as f:
        json.dump(new_lookup, f, indent=2)

    print(f"  Existing entries: {len(existing_lookup)}")
    print(f"  New entries added: {new_entries}")
    print(f"  Total entries: {len(new_lookup)}")
    print(f"  Saved to: {output_lookup_path}")

    # Show some valid mappings
    valid_results = [r for r in results if r.get('valid', False)]
    if valid_results:
        print(f"\nSample valid mappings:")
        for r in valid_results[:5]:
            print(f"  {r['source_type']} -> {r['target_type']}")
            print(f"    Confidence: {r.get('confidence', 0):.2f}")
            print(f"    Mappings: {list(r.get('mappings', {}).items())[:3]}")


if __name__ == '__main__':
    main()
