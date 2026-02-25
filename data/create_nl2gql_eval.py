#!/usr/bin/env python3
"""
Create nl2gql_eval.json from nl2gql_dataset.jsonl.

Extracts evaluation utterances in the format expected by MapReflect.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def extract_eval_samples(dataset_file: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """
    Extract evaluation samples from nl2gql_dataset.jsonl.
    
    Args:
        dataset_file: Path to nl2gql_dataset.jsonl
        max_samples: Maximum number of samples to extract (None = all)
    
    Returns:
        List of evaluation samples in MapReflect format
    """
    eval_samples = []
    seen_utterances = set()  # Track unique utterances
    
    with open(dataset_file, 'r') as f:
        for idx, line in enumerate(f):
            if max_samples and len(eval_samples) >= max_samples:
                break
            
            try:
                data = json.loads(line.strip())
                
                # Extract information
                result = data.get('result', {})
                source_table = result.get('name', '')

                # Create retrieval-style utterance - model must find the target
                utterance = f"find tables that can be joined with {source_table}"
                properties = result.get('properties', {})
                
                # Get target table from first mapping
                target_table = ''
                ground_truth_mapping = {}

                for src_field, mappings in properties.items():
                    if mappings and isinstance(mappings, list):
                        target_table = mappings[0].get('mapTable', '')
                        target_field = mappings[0].get('mapProperty', '')

                        # Build ground truth mapping with nl2gql_Table.field format
                        # to match property lookup naming convention
                        # Format: nl2gql_Singer_in_concert.concert_id (first letter capitalized, rest lowercase)
                        if target_field:
                            # Convert source field: concert_stadium_id -> nl2gql_Concert.stadium_id
                            # Strip table prefix from field name
                            src_field_name = src_field
                            if src_field.startswith(source_table + '_'):
                                src_field_name = src_field[len(source_table) + 1:]
                            # First letter capitalized: singer_in_concert -> Singer_in_concert
                            src_table_fmt = source_table[0].upper() + source_table[1:] if source_table else ''
                            src_key = f"nl2gql_{src_table_fmt}.{src_field_name}"

                            # Convert target field: stadium_stadium_id -> nl2gql_Stadium.stadium_id
                            tgt_field_name = target_field
                            if target_field.startswith(target_table + '_'):
                                tgt_field_name = target_field[len(target_table) + 1:]
                            tgt_table_fmt = target_table[0].upper() + target_table[1:] if target_table else ''
                            tgt_key = f"nl2gql_{tgt_table_fmt}.{tgt_field_name}"

                            ground_truth_mapping[src_key] = [tgt_key]
                
                # Skip if missing required fields or duplicate utterance
                if not utterance or not source_table or not target_table:
                    continue
                
                if utterance in seen_utterances:
                    continue
                
                seen_utterances.add(utterance)
                
                # Convert table names to PascalCase for expected_source/target
                source_pascal = source_table.replace('_', ' ').title().replace(' ', '')
                target_pascal = target_table.replace('_', ' ').title().replace(' ', '')
                
                eval_sample = {
                    'sample_id': f'nl2gql_{len(eval_samples)+1:03d}',
                    'utterance': utterance,
                    'expected_source': source_pascal,
                    'expected_target': target_pascal,
                    'description': data.get('question', ''),
                    'ground_truth_mapping': ground_truth_mapping,
                    'metadata': {
                        'original_question': data.get('question', ''),
                        'query': data.get('query', '')
                    }
                }
                
                eval_samples.append(eval_sample)
            
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {idx+1}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {idx+1}: {e}")
                continue
    
    return eval_samples


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create nl2gql_eval.json from nl2gql_dataset.jsonl'
    )
    parser.add_argument('--input', type=str, default='nl2gql_dataset.jsonl',
                       help='Input dataset file (default: nl2gql_dataset.jsonl)')
    parser.add_argument('--output', type=str, default='nl2gql_eval.json',
                       help='Output evaluation file (default: nl2gql_eval.json)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to extract (default: all unique)')
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        print(f"\nExpected format: nl2gql_dataset.jsonl")
        print(f"Each line should be a JSON object with:")
        print(f"  - mapping_utterance: The mapping query")
        print(f"  - result.name: Source table")
        print(f"  - result.properties: Mapping properties")
        return 1
    
    print("="*60)
    print("NL2GQL Evaluation File Generation")
    print("="*60)
    print()
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print()
    
    # Extract evaluation samples
    print("Extracting evaluation samples...")
    eval_samples = extract_eval_samples(str(input_file), args.max_samples)
    
    if not eval_samples:
        print("❌ Error: No valid evaluation samples extracted")
        return 1
    
    print(f"✓ Extracted {len(eval_samples)} unique evaluation samples")
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(eval_samples, f, indent=2)
    
    print(f"✓ Saved to: {output_file}")
    
    # Print statistics
    print()
    print("="*60)
    print("Generation Complete!")
    print("="*60)
    print()
    
    # Count unique source and target tables
    sources = set(s['expected_source'] for s in eval_samples)
    targets = set(s['expected_target'] for s in eval_samples)
    
    print(f"Statistics:")
    print(f"  Total samples: {len(eval_samples)}")
    print(f"  Unique source tables: {len(sources)}")
    print(f"  Unique target tables: {len(targets)}")
    print(f"  Samples with ground truth: {sum(1 for s in eval_samples if s['ground_truth_mapping'])}")
    
    # Show sample entries
    print(f"\nSample evaluation entries:")
    for i, sample in enumerate(eval_samples[:3]):
        print(f"\n  Sample {i+1}:")
        print(f"    ID: {sample['sample_id']}")
        print(f"    Utterance: {sample['utterance']}")
        print(f"    Source → Target: {sample['expected_source']} → {sample['expected_target']}")
        print(f"    Ground truth mappings: {len(sample['ground_truth_mapping'])}")
    
    print(f"\nFormat:")
    print(f"  ✓ Compatible with MapReflect dataset_loader")
    print(f"  ✓ Includes ground truth mappings for evaluation")
    print(f"  ✓ Includes original questions and queries as metadata")
    
    print(f"\nNext steps:")
    print(f"  1. Review {output_file}")
    print(f"  2. Optionally edit utterances for better clarity")
    print(f"  3. Use with MapReflect: place in data directory")
    
    return 0


if __name__ == '__main__':
    exit(main())
