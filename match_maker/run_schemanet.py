#!/usr/bin/env python3
"""
Run MatchMaker on SchemaNet datasets.

SchemaNet datasets have a different format where the ground truth mapping
is provided directly. We iterate through source fields and find target matches.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Override sqlite3 with pysqlite3 on systems with old sqlite3 (< 3.35.0)
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
import logging
from datetime import datetime

os.environ['LITELLM_LOG'] = 'ERROR'
os.environ['LITELLM_DROP_PARAMS'] = 'True'

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

for lib in ['litellm', 'LiteLLM', 'httpx', 'httpcore', 'urllib3', 'requests']:
    logging.getLogger(lib).setLevel(logging.ERROR)


def load_dataset(data_dir: Path, dataset_name: str) -> List[Dict]:
    """Load SchemaNet dataset."""
    dataset_file = data_dir / f"{dataset_name}_dataset.jsonl"
    samples = []
    with open(dataset_file) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def load_api_lookup(data_dir: Path, dataset_name: str) -> Dict:
    """Load API lookup file."""
    lookup_file = data_dir / f"{dataset_name}_api_lookup.json"
    with open(lookup_file) as f:
        return json.load(f)


def run_matchmaker_on_schemanet(
    dataset_name: str,
    data_dir: Path,
    model: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """Run MatchMaker on a SchemaNet dataset."""
    from match_maker_agent import MatchMakerAgent

    # Load dataset and API lookup
    samples = load_dataset(data_dir, dataset_name)
    api_lookup = load_api_lookup(data_dir, dataset_name)

    print(f"Loaded {len(samples)} samples from {dataset_name}")
    print(f"API lookup has {len(api_lookup)} types")

    # Create MatchMaker agent
    lookup_file = data_dir / f"{dataset_name}_api_lookup.json"
    chroma_dir = data_dir / f"chroma_db_{dataset_name}"

    agent = MatchMakerAgent(
        property_lookup_file=str(lookup_file),
        model=model,
        dataset=dataset_name,
        chroma_dir=str(chroma_dir),
        top_k=top_k,
        rebuild_index=True,  # Force rebuild for SchemaNet
        use_optimization=False,  # Skip optimization for SchemaNet (no source/target separation)
        abstention_mode='lenient'
    )

    # For SchemaNet, build index with individual FIELDS (not types)
    # The API lookup has types with nested fields - we need to index each field separately
    print("Building field-level index for SchemaNet...")
    from tqdm import tqdm
    documents = []
    doc_ids = []

    for type_id, type_data in api_lookup.items():
        type_name = type_data.get('name', type_id)
        type_desc = type_data.get('description', '')
        fields = type_data.get('fields', {})

        for field_name, field_data in fields.items():
            # Create field-level ID: type_field (e.g., bank2_branch_branch_id)
            field_id = f"{type_id}_{field_name}"
            field_type = field_data.get('type', 'unknown')
            field_desc = field_data.get('description', f'{field_name} field')

            # Create document text for embedding
            doc = f"{type_name}.{field_name}: {field_desc} (type: {field_type})"
            documents.append(doc)
            doc_ids.append(field_id)

    print(f"Indexing {len(documents)} fields from {len(api_lookup)} types")

    embeddings = [agent.embedding_model.encode(doc) for doc in tqdm(documents, desc="Embedding")]

    import chromadb
    if not hasattr(agent, 'chroma_client') or agent.chroma_client is None:
        agent.chroma_client = chromadb.PersistentClient(path=agent.index_path)
    try:
        agent.chroma_client.delete_collection(name=agent.collection_name)
    except:
        pass
    agent.collection = agent.chroma_client.get_or_create_collection(name=agent.collection_name)
    agent.collection.add(ids=doc_ids, documents=documents, embeddings=embeddings)
    print(f"Indexed {len(documents)} fields")

    results = []
    total_correct = 0
    total_fields = 0

    from tqdm import tqdm

    for sample in tqdm(samples, desc=f"Processing {dataset_name}"):
        sample_id = sample['id']
        mapping = sample.get('mapping', {})
        expected_target_type = sample.get('expected_target_type', '')

        sample_results = {
            'sample_id': sample_id,
            'utterance': sample.get('utterance', ''),
            'expected_target_type': expected_target_type,
            'predictions': {},
            'ground_truth': mapping,
            'correct': 0,
            'total': len(mapping)
        }

        # Process each source field
        for source_field, expected_target in mapping.items():
            # Extract field info
            # source_field format: "bank1_branch.branch_id"
            # expected_target format: "bank2_branch.branch_id"

            total_fields += 1

            source_parts = source_field.split('.')
            if len(source_parts) == 2:
                source_type = source_parts[0]
                source_field_name = source_parts[1]
            else:
                source_type = source_field
                source_field_name = source_field

            # Create a query description for the source field
            query_text = f"{source_field_name} field from {source_type.replace('_', ' ')} table"

            try:
                # Query the ChromaDB collection directly
                query_embedding = agent.embedding_model.encode(query_text)
                results_query = agent.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k
                )
                candidates = results_query['ids'][0] if results_query.get('ids') else []

                # Check if expected target is in candidates
                predicted = candidates[0] if candidates else None

                # Compare with expected
                # expected_target format: "bank2_branch.branch_id"
                target_parts = expected_target.split('.')
                expected_type = target_parts[0] if len(target_parts) == 2 else expected_target
                expected_field = target_parts[1] if len(target_parts) == 2 else expected_target

                # Candidates are in format "bank2_branch_branch_id" (type + field)
                expected_full = f"{expected_type}_{expected_field}"

                is_correct = (
                    predicted and
                    (predicted.lower() == expected_full.lower() or
                     predicted.lower().endswith(f"_{expected_field}".lower()))
                )

                if is_correct:
                    total_correct += 1
                    sample_results['correct'] += 1

                sample_results['predictions'][source_field] = {
                    'expected': expected_target,
                    'expected_id': expected_full,
                    'predicted': predicted,
                    'candidates': candidates[:5],
                    'correct': is_correct
                }

            except Exception as e:
                logger.warning(f"Error processing {source_field}: {e}")
                sample_results['predictions'][source_field] = {
                    'expected': expected_target,
                    'predicted': None,
                    'error': str(e),
                    'correct': False
                }

        results.append(sample_results)
        time.sleep(0.1)  # Rate limiting

    accuracy = total_correct / total_fields if total_fields > 0 else 0

    return {
        'metadata': {
            'dataset': dataset_name,
            'model': model,
            'total_samples': len(samples),
            'total_fields': total_fields,
            'correct': total_correct,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat()
        },
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description="Run MatchMaker on SchemaNet")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., schemanet_bank1_bank2)")
    parser.add_argument("--data-dir", type=str, default="data/schemanet", help="Data directory")
    parser.add_argument("--model", type=str, default="watsonx/meta-llama/llama-3-3-70b-instruct")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="match_maker/results")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Add script dir to path for imports
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    print(f"\n{'='*60}")
    print(f"MatchMaker on SchemaNet: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")

    try:
        results = run_matchmaker_on_schemanet(
            dataset_name=args.dataset,
            data_dir=data_dir,
            model=args.model,
            top_k=args.top_k
        )

        # Save results
        model_lower = args.model.lower()
        if 'llama' in model_lower:
            model_short = 'llama'
        elif 'mistral' in model_lower:
            model_short = 'mistral'
        elif 'granite' in model_lower:
            model_short = 'granite'
        else:
            # Extract last segment of model ID as short name
            model_short = args.model.split('/')[-1].split('-')[0]
        output_file = output_dir / f"matchmaker_{args.dataset}_{model_short}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_file}")
        print(f"\nAccuracy: {results['metadata']['accuracy']*100:.1f}%")
        print(f"Correct: {results['metadata']['correct']}/{results['metadata']['total_fields']}")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
