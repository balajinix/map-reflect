#!/usr/bin/env python3
"""
CLI script for loading concepts and relations into Neo4j.

Usage:
    # Load concepts only
    python -m concepts.scripts.load_neo4j \
        --concepts concepts/output/entities/resolved.json

    # Load concepts and relations
    python -m concepts.scripts.load_neo4j \
        --concepts concepts/output/entities/resolved.json \
        --relations concepts/output/relations/relations.json

    # Load with traceability
    python -m concepts.scripts.load_neo4j \
        --concepts concepts/output/entities/resolved.json \
        --relations concepts/output/relations/relations.json \
        --chunks concepts/output/entities/chunks.json

    # Custom Neo4j connection
    python -m concepts.scripts.load_neo4j \
        --concepts concepts/output/entities/resolved.json \
        --neo4j-uri bolt://neo4j.example.com:7687 \
        --neo4j-user admin \
        --neo4j-password secret
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from concepts.pipeline.loader import (
    Neo4jLoader,
    LoaderMetrics,
    load_to_neo4j,
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
        description="Load concepts and relations into Neo4j",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--concepts", "-c",
        required=True,
        help="Resolved concepts JSON file (from resolve_entities.py)"
    )
    parser.add_argument(
        "--relations", "-r",
        help="Relations JSON file (from discover_relations.py)"
    )
    parser.add_argument(
        "--chunks",
        help="Chunks JSON file for traceability (from convert_docs.py)"
    )
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)"
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    parser.add_argument(
        "--neo4j-password",
        default="password",
        help="Neo4j password (default: password)"
    )
    parser.add_argument(
        "--no-vector-index",
        action="store_true",
        help="Skip vector index creation"
    )
    parser.add_argument(
        "--no-api-links",
        action="store_true",
        help="Skip linking to API graph (Product/Type/Field nodes)"
    )
    parser.add_argument(
        "--metrics-output",
        help="Output JSON file for metrics"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without loading to Neo4j"
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
    concepts_file = Path(args.concepts)
    if not concepts_file.exists():
        logger.error(f"Concepts file not found: {concepts_file}")
        sys.exit(1)

    relations_file = None
    if args.relations:
        relations_file = Path(args.relations)
        if not relations_file.exists():
            logger.error(f"Relations file not found: {relations_file}")
            sys.exit(1)

    chunks_file = None
    if args.chunks:
        chunks_file = Path(args.chunks)
        if not chunks_file.exists():
            logger.error(f"Chunks file not found: {chunks_file}")
            sys.exit(1)

    # Load and validate data
    logger.info(f"Loading concepts from: {concepts_file}")
    with open(concepts_file, 'r') as f:
        concepts_data = json.load(f)

    concepts = concepts_data.get("resolved_entities", concepts_data.get("entities", []))
    logger.info(f"Found {len(concepts)} concepts")

    relations = []
    if relations_file:
        logger.info(f"Loading relations from: {relations_file}")
        with open(relations_file, 'r') as f:
            relations_data = json.load(f)
        relations = relations_data.get("relations", [])
        logger.info(f"Found {len(relations)} relations")

    chunks = []
    if chunks_file:
        logger.info(f"Loading chunks from: {chunks_file}")
        with open(chunks_file, 'r') as f:
            chunks_data = json.load(f)
        chunks = chunks_data.get("chunks", [])
        logger.info(f"Found {len(chunks)} chunks")

    if args.dry_run:
        logger.info("\n=== DRY RUN ===")
        logger.info("Would load to Neo4j:")
        logger.info(f"  Concepts: {len(concepts)}")
        logger.info(f"  Relations: {len(relations)}")
        logger.info(f"  Documents: {len(set(c.get('source_path') for c in chunks if c.get('source_path')))}")
        logger.info(f"  Neo4j URI: {args.neo4j_uri}")
        logger.info(f"  Vector index: {not args.no_vector_index}")
        logger.info(f"  API links: {not args.no_api_links}")
        logger.info("\nDry run complete. Use without --dry-run to load.")
        return

    # Load to Neo4j
    logger.info("=" * 50)
    logger.info("STAGE: Neo4j Loading")
    logger.info("=" * 50)

    try:
        metrics = load_to_neo4j(
            concepts_file=concepts_file,
            relations_file=relations_file,
            chunks_file=chunks_file,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            create_vector_index=not args.no_vector_index,
            link_to_api_graph=not args.no_api_links,
        )
    except Exception as e:
        logger.error(f"Failed to load to Neo4j: {e}")
        sys.exit(1)

    # Print summary
    logger.info("\nLoading Summary:")
    logger.info(f"  Concepts created: {metrics.concepts_created}")
    logger.info(f"  Concepts updated: {metrics.concepts_updated}")
    logger.info(f"  Concepts skipped: {metrics.concepts_skipped}")
    logger.info(f"  Concept edges: {metrics.concept_edges}")
    logger.info(f"  Embeddings stored: {metrics.embeddings_stored}")
    logger.info(f"  Documents processed: {metrics.documents_processed}")
    logger.info(f"  MENTIONED_IN edges: {metrics.mentioned_in_edges}")
    if not args.no_api_links:
        logger.info(f"  API links (Product): {metrics.api_links_product}")
        logger.info(f"  API links (Type): {metrics.api_links_type}")
        logger.info(f"  API links (Field): {metrics.api_links_field}")
        logger.info(f"  Orphan concepts: {metrics.orphan_concepts}")
    logger.info(f"  Time: {metrics.load_time_seconds:.2f}s")

    # Save metrics if requested
    if args.metrics_output:
        metrics_file = Path(args.metrics_output)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)

        metrics_data = {
            "created_at": datetime.now().isoformat(),
            "concepts": str(concepts_file),
            "relations": str(relations_file) if relations_file else None,
            "chunks": str(chunks_file) if chunks_file else None,
            "neo4j_uri": args.neo4j_uri,
            "config": {
                "create_vector_index": not args.no_vector_index,
                "link_to_api_graph": not args.no_api_links,
            },
            "metrics": metrics.to_dict(),
        }

        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        logger.info(f"\nMetrics saved to: {metrics_file}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
