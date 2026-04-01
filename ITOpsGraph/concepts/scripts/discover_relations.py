#!/usr/bin/env python3
"""
CLI script for relation discovery between entities.

Usage:
    # Discover relations from resolved entities and chunks
    python -m concepts.scripts.discover_relations \
        --entities concepts/output/entities/resolved.json \
        --chunks concepts/output/entities/chunks.json \
        --output concepts/output/relations/relations.json

    # Disable specific methods
    python -m concepts.scripts.discover_relations \
        --entities concepts/output/entities/resolved.json \
        --chunks concepts/output/entities/chunks.json \
        --output concepts/output/relations/relations.json \
        --no-cooccurrence

    # Enable LLM classification
    python -m concepts.scripts.discover_relations \
        --entities concepts/output/entities/resolved.json \
        --chunks concepts/output/entities/chunks.json \
        --output concepts/output/relations/relations.json \
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

from concepts.pipeline.relations import (
    RelationDiscoverer,
    RelationMetrics,
    discover_relations_from_files,
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
        description="Discover relations between entities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--entities", "-e",
        required=True,
        help="Resolved entities JSON file (from resolve_entities.py)"
    )
    parser.add_argument(
        "--chunks", "-c",
        required=True,
        help="Chunks JSON file (from convert_docs.py)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output relations JSON file"
    )
    parser.add_argument(
        "--no-gliner2",
        action="store_true",
        help="Disable GLiNER2 relation extraction"
    )
    parser.add_argument(
        "--no-dep-parse",
        action="store_true",
        help="Disable dependency parsing"
    )
    parser.add_argument(
        "--no-cooccurrence",
        action="store_true",
        help="Disable co-occurrence analysis"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM relation classification"
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="OpenAI model for LLM classification (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--cooccurrence-threshold",
        type=float,
        default=0.3,
        help="Minimum weight for co-occurrence edges (default: 0.3)"
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
    entities_file = Path(args.entities)
    if not entities_file.exists():
        logger.error(f"Entities file not found: {entities_file}")
        sys.exit(1)

    chunks_file = Path(args.chunks)
    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        sys.exit(1)

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load entities
    logger.info(f"Loading entities from: {entities_file}")
    with open(entities_file, 'r') as f:
        entities_data = json.load(f)

    entities = entities_data.get("resolved_entities", entities_data.get("entities", []))
    logger.info(f"Loaded {len(entities)} entities")

    # Load chunks
    logger.info(f"Loading chunks from: {chunks_file}")
    with open(chunks_file, 'r') as f:
        chunks_data = json.load(f)

    chunks = chunks_data.get("chunks", [])
    logger.info(f"Loaded {len(chunks)} chunks")

    # Initialize discoverer
    logger.info("=" * 50)
    logger.info("STAGE: Relation Discovery")
    logger.info("=" * 50)

    discoverer = RelationDiscoverer(
        use_gliner2=not args.no_gliner2,
        use_dep_parse=not args.no_dep_parse,
        use_cooccurrence=not args.no_cooccurrence,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        cooccurrence_threshold=args.cooccurrence_threshold,
        min_confidence=args.min_confidence,
    )

    # Discover relations
    relations = discoverer.discover(entities, chunks)

    # Save results
    output_data = {
        "created_at": datetime.now().isoformat(),
        "source_entities": str(entities_file),
        "source_chunks": str(chunks_file),
        "config": {
            "use_gliner2": not args.no_gliner2,
            "use_dep_parse": not args.no_dep_parse,
            "use_cooccurrence": not args.no_cooccurrence,
            "use_llm": args.use_llm,
            "llm_model": args.llm_model if args.use_llm else None,
            "min_confidence": args.min_confidence,
            "cooccurrence_threshold": args.cooccurrence_threshold,
        },
        "metrics": discoverer.metrics.to_dict(),
        "relations": [r.to_dict() for r in relations],
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Relations saved to: {output_file}")

    # Print summary
    metrics = discoverer.metrics
    logger.info("\nDiscovery Summary:")
    logger.info(f"  Entities input: {metrics.entities_input}")
    logger.info(f"  Relations discovered: {metrics.relations_total}")
    logger.info(f"  - GLiNER2: {metrics.relations_gliner2}")
    logger.info(f"  - Dep parse: {metrics.relations_dep_parse}")
    logger.info(f"  - Co-occurrence: {metrics.relations_cooccurrence}")
    if args.use_llm:
        logger.info(f"  - LLM classified: {metrics.relations_llm}")
    logger.info(f"  Dep parse patterns: {metrics.dep_parse_patterns}")
    logger.info(f"  Avg confidence: {metrics.avg_confidence:.2f}")
    logger.info(f"  Time: {metrics.discovery_time_seconds:.2f}s")

    # Relation type distribution
    if metrics.relation_type_dist:
        logger.info("\nRelation Type Distribution:")
        for rtype, count in sorted(metrics.relation_type_dist.items(), key=lambda x: -x[1])[:15]:
            logger.info(f"  {rtype}: {count}")

    # Save metrics separately if requested
    if args.metrics_output:
        metrics_file = Path(args.metrics_output)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)

        metrics_data = {
            "created_at": datetime.now().isoformat(),
            "entities": str(entities_file),
            "chunks": str(chunks_file),
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
