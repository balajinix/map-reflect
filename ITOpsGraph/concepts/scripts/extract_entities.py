#!/usr/bin/env python3
"""
CLI script for entity extraction from document chunks.

Usage:
    # Extract entities from chunks
    python -m concepts.scripts.extract_entities \
        --input concepts/output/entities/chunks.json \
        --output concepts/output/entities/entities.json

    # With custom entity types config
    python -m concepts.scripts.extract_entities \
        --input concepts/output/entities/chunks.json \
        --output concepts/output/entities/entities.json \
        --config concepts/config/entity_types.yaml

    # Disable LLM augmentation (faster, less accurate hierarchy)
    python -m concepts.scripts.extract_entities \
        --input concepts/output/entities/chunks.json \
        --output concepts/output/entities/entities.json \
        --no-llm
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from concepts.pipeline.ner import (
    EntityExtractor,
    NERMetrics,
    load_entity_types,
    extract_entities_from_chunks_file,
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
        description="Extract entities from document chunks using GLiNER2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input chunks JSON file (from convert_docs.py)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output entities JSON file"
    )
    parser.add_argument(
        "--config", "-c",
        help="Entity types YAML config file"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM augmentation for hierarchy classification"
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="OpenAI model for LLM augmentation (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for entity extraction (default: 0.3)"
    )
    parser.add_argument(
        "--gliner-model",
        default="gliner-community/gliner_large-v2.5",
        help="GLiNER model to use"
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
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of chunks to process (for testing)"
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

    # Load entity types config
    entity_types = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            logger.info(f"Loading entity types from: {config_path}")
            entity_types = load_entity_types(config_path)
        else:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")

    # Load chunks
    logger.info(f"Loading chunks from: {input_file}")
    with open(input_file, 'r') as f:
        chunks_data = json.load(f)

    chunks = chunks_data.get("chunks", [])
    if args.limit:
        chunks = chunks[:args.limit]
        logger.info(f"Limited to {len(chunks)} chunks")
    else:
        logger.info(f"Processing {len(chunks)} chunks")

    # Initialize extractor
    logger.info("=" * 50)
    logger.info("STAGE: Entity Extraction (NER)")
    logger.info("=" * 50)

    extractor = EntityExtractor(
        entity_types=entity_types,
        gliner_model=args.gliner_model,
        confidence_threshold=args.threshold,
        use_llm=not args.no_llm,
        llm_model=args.llm_model,
    )

    # Create chunk iterator
    from dataclasses import dataclass

    @dataclass
    class SimpleChunk:
        chunk_id: str
        text: str
        source_path: str
        product: str

    def chunk_generator():
        for c in chunks:
            yield SimpleChunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                source_path=c["source_path"],
                product=c["product"],
            )

    # Extract entities
    entities = []
    batch_size = 100

    for i, entity in enumerate(extractor.extract_from_chunks(chunk_generator(), batch_size=batch_size)):
        entities.append(entity)

    # Deduplicate within-document
    logger.info(f"Extracted {len(entities)} total entity mentions")

    # Save results
    output_data = {
        "created_at": datetime.now().isoformat(),
        "source_chunks": str(input_file),
        "config": {
            "gliner_model": args.gliner_model,
            "threshold": args.threshold,
            "use_llm": not args.no_llm,
            "llm_model": args.llm_model if not args.no_llm else None,
        },
        "metrics": extractor.metrics.to_dict(),
        "entities": [e.to_dict() for e in entities],
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Entities saved to: {output_file}")

    # Print summary
    metrics = extractor.metrics
    logger.info("\nExtraction Summary:")
    logger.info(f"  Chunks processed: {metrics.chunks_processed}")
    logger.info(f"  Entities extracted: {metrics.entities_total}")
    logger.info(f"  Entities per chunk: {metrics.entities_per_chunk:.1f}")
    logger.info(f"  Confidence avg/min/max: {metrics.confidence_avg:.2f}/{metrics.confidence_min:.2f}/{metrics.confidence_max:.2f}")
    logger.info(f"  Pass 1 (GLiNER): {metrics.pass1_count}")
    logger.info(f"  Pass 2 reclassified: {metrics.pass2_reclassified}")
    logger.info(f"  Time: {metrics.extraction_time_seconds:.2f}s")

    if metrics.entity_type_dist:
        logger.info("\nEntity Type Distribution:")
        for etype, count in sorted(metrics.entity_type_dist.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {etype}: {count}")

    if metrics.hierarchy_dist:
        logger.info("\nHierarchy Distribution:")
        for hier, count in sorted(metrics.hierarchy_dist.items(), key=lambda x: -x[1]):
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
