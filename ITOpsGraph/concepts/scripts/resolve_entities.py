#!/usr/bin/env python3
"""
CLI script for entity resolution and deduplication.

Usage:
    # Resolve entities from NER output
    python -m concepts.scripts.resolve_entities \
        --input concepts/output/entities/entities.json \
        --output concepts/output/entities/resolved.json

    # With custom synonyms config
    python -m concepts.scripts.resolve_entities \
        --input concepts/output/entities/entities.json \
        --output concepts/output/entities/resolved.json \
        --synonyms concepts/config/synonyms.yaml

    # Disable embedding matching (faster)
    python -m concepts.scripts.resolve_entities \
        --input concepts/output/entities/entities.json \
        --output concepts/output/entities/resolved.json \
        --no-embeddings

    # Enable LLM verification for borderline cases
    python -m concepts.scripts.resolve_entities \
        --input concepts/output/entities/entities.json \
        --output concepts/output/entities/resolved.json \
        --use-llm
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from concepts.pipeline.resolve import (
    EntityResolver,
    ResolutionMetrics,
    resolve_entities_from_file,
)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(
        description="Resolve and deduplicate entities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input entities JSON file (from extract_entities.py)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output resolved entities JSON file"
    )
    parser.add_argument(
        "--synonyms", "-s",
        help="Synonyms YAML config file (default: concepts/config/synonyms.yaml)"
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.85,
        help="Threshold for fuzzy string matching (default: 0.85)"
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.80,
        help="Threshold for embedding similarity (default: 0.80)"
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable embedding-based matching"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM verification for borderline cases"
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="OpenAI model for LLM verification (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--metrics-output",
        help="Output JSON file for metrics"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate inputs
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Find synonyms file
    synonyms_path = None
    if args.synonyms:
        synonyms_path = Path(args.synonyms)
        if not synonyms_path.exists():
            logger.warning(f"Synonyms file not found: {synonyms_path}. Using defaults.")
            synonyms_path = None
    else:
        # Try default location
        default_synonyms = Path(__file__).parent.parent / "config" / "synonyms.yaml"
        if default_synonyms.exists():
            synonyms_path = default_synonyms
            logger.info(f"Using default synonyms file: {synonyms_path}")

    # Load entities
    logger.info(f"Loading entities from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    entities = data.get("entities", [])
    logger.info(f"Loaded {len(entities)} entities")

    # Initialize resolver
    logger.info("=" * 50)
    logger.info("STAGE: Entity Resolution")
    logger.info("=" * 50)

    resolver = EntityResolver(
        synonyms_path=synonyms_path,
        fuzzy_threshold=args.fuzzy_threshold,
        embedding_threshold=args.embedding_threshold,
        use_embeddings=not args.no_embeddings,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
    )

    # Resolve entities
    resolved = resolver.resolve(entities)

    # Save results
    output_data = {
        "created_at": datetime.now().isoformat(),
        "source_entities": str(input_file),
        "config": {
            "synonyms_path": str(synonyms_path) if synonyms_path else None,
            "fuzzy_threshold": args.fuzzy_threshold,
            "embedding_threshold": args.embedding_threshold,
            "use_embeddings": not args.no_embeddings,
            "use_llm": args.use_llm,
            "llm_model": args.llm_model if args.use_llm else None,
        },
        "metrics": resolver.metrics.to_dict(),
        "resolved_entities": [e.to_dict() for e in resolved],
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Resolved entities saved to: {output_file}")

    # Print summary
    metrics = resolver.metrics
    logger.info("\nResolution Summary:")
    logger.info(f"  Entities input: {metrics.entities_input}")
    logger.info(f"  Entities output: {metrics.entities_output}")
    logger.info(f"  Entities merged: {metrics.entities_merged}")
    logger.info(f"  Merge ratio: {metrics.merge_ratio:.1%}")
    logger.info(f"  Exact matches: {metrics.exact_matches}")
    logger.info(f"  Synonym matches: {metrics.synonym_matches}")
    logger.info(f"  Fuzzy matches: {metrics.fuzzy_matches}")
    logger.info(f"  Embedding matches: {metrics.embedding_matches}")
    if args.use_llm:
        logger.info(f"  LLM verified: {metrics.llm_verified}")
        logger.info(f"  LLM rejected: {metrics.llm_rejected}")
    logger.info(f"  Cross-product merges: {metrics.cross_product_merges}")
    logger.info(f"  Canonical coverage: {metrics.canonical_coverage:.1%}")
    logger.info(f"  Time: {metrics.resolution_time_seconds:.2f}s")

    # Entity type distribution
    type_dist = {}
    for entity in resolved:
        etype = entity.entity_type
        type_dist[etype] = type_dist.get(etype, 0) + 1

    if type_dist:
        logger.info("\nEntity Type Distribution:")
        for etype, count in sorted(type_dist.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {etype}: {count}")

    # Hierarchy distribution
    hier_dist = {}
    for entity in resolved:
        hier = entity.hierarchy
        hier_dist[hier] = hier_dist.get(hier, 0) + 1

    if hier_dist:
        logger.info("\nHierarchy Distribution:")
        for hier, count in sorted(hier_dist.items(), key=lambda x: -x[1]):
            logger.info(f"  {hier}: {count}")

    # Save metrics separately if requested
    if args.metrics_output:
        metrics_file = Path(args.metrics_output)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)

        metrics_data = {
            "created_at": datetime.now().isoformat(),
            "input": str(input_file),
            "output": str(output_file),
            "config": output_data["config"],
            "metrics": metrics.to_dict(),
        }

        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        logger.info(f"\nMetrics saved to: {metrics_file}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
