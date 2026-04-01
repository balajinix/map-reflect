#!/usr/bin/env python3
"""
Full concept extraction pipeline.

Runs all stages of the pipeline:
1. Document conversion (PDF/HTML/DOCX -> Markdown)
2. Section-aware chunking
3. Entity extraction (NER)
4. Entity resolution (deduplication)
5. Relation discovery
6. Neo4j loading

Usage:
    # Full pipeline
    python -m concepts.scripts.run_pipeline \
        --input docs/product_docs/raw/pdf/ \
        --output concepts/output/

    # With Neo4j loading
    python -m concepts.scripts.run_pipeline \
        --input docs/product_docs/raw/pdf/ \
        --output concepts/output/ \
        --neo4j-uri bolt://localhost:7687 \
        --neo4j-password password

    # Specific stages only
    python -m concepts.scripts.run_pipeline \
        --input docs/product_docs/raw/pdf/ \
        --output concepts/output/ \
        --stages convert,chunk,ner,resolve

    # Dry run (no Neo4j loading)
    python -m concepts.scripts.run_pipeline \
        --input docs/product_docs/raw/pdf/ \
        --output concepts/output/ \
        --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def run_convert(input_dir: Path, output_dir: Path, logger) -> Path:
    """Run document conversion stage."""
    from concepts.pipeline.convert import DocumentConverter

    logger.info("=" * 50)
    logger.info("STAGE 1: Document Conversion")
    logger.info("=" * 50)

    converter = DocumentConverter()
    chunks_file = output_dir / "entities" / "chunks.json"
    chunks_file.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    docs_processed = 0

    from concepts.pipeline.chunker import DocumentChunker
    chunker = DocumentChunker()

    for doc in converter.convert_directory(input_dir):
        docs_processed += 1
        for chunk in chunker.chunk_document(
            doc.markdown,
            doc.source_path,
            doc.product
        ):
            chunks.append(chunk.to_dict())

    # Save chunks
    output_data = {
        "created_at": datetime.now().isoformat(),
        "source_dir": str(input_dir),
        "docs_processed": docs_processed,
        "chunks": chunks,
    }

    with open(chunks_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Converted {docs_processed} documents -> {len(chunks)} chunks")
    logger.info(f"Saved to: {chunks_file}")

    return chunks_file


def run_ner(chunks_file: Path, output_dir: Path, config_path: Path | None, logger) -> Path:
    """Run entity extraction stage."""
    from concepts.pipeline.ner import EntityExtractor, load_entity_types

    logger.info("=" * 50)
    logger.info("STAGE 2: Entity Extraction (NER)")
    logger.info("=" * 50)

    # Load chunks
    with open(chunks_file, 'r') as f:
        chunks_data = json.load(f)

    chunks = chunks_data.get("chunks", [])

    # Load entity types config
    entity_types = None
    if config_path and config_path.exists():
        entity_types = load_entity_types(config_path)

    # Initialize extractor
    extractor = EntityExtractor(
        entity_types=entity_types,
        use_llm=False,  # Disable LLM for speed
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
    for entity in extractor.extract_from_chunks(chunk_generator()):
        entities.append(entity)

    # Save results
    entities_file = output_dir / "entities" / "entities.json"

    output_data = {
        "created_at": datetime.now().isoformat(),
        "source_chunks": str(chunks_file),
        "metrics": extractor.metrics.to_dict(),
        "entities": [e.to_dict() for e in entities],
    }

    with open(entities_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Extracted {len(entities)} entities")
    logger.info(f"Saved to: {entities_file}")

    return entities_file


def run_resolve(entities_file: Path, output_dir: Path, synonyms_path: Path | None, logger) -> Path:
    """Run entity resolution stage."""
    from concepts.pipeline.resolve import EntityResolver

    logger.info("=" * 50)
    logger.info("STAGE 3: Entity Resolution")
    logger.info("=" * 50)

    # Load entities
    with open(entities_file, 'r') as f:
        entities_data = json.load(f)

    entities = entities_data.get("entities", [])

    # Resolve
    resolver = EntityResolver(
        synonyms_path=synonyms_path,
        use_embeddings=True,
        use_llm=False,
    )

    resolved = resolver.resolve(entities)

    # Save results
    resolved_file = output_dir / "entities" / "resolved.json"

    output_data = {
        "created_at": datetime.now().isoformat(),
        "source_entities": str(entities_file),
        "metrics": resolver.metrics.to_dict(),
        "resolved_entities": [e.to_dict() for e in resolved],
    }

    with open(resolved_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Resolved {len(entities)} -> {len(resolved)} entities")
    logger.info(f"Saved to: {resolved_file}")

    return resolved_file


def run_relations(
    resolved_file: Path,
    chunks_file: Path,
    output_dir: Path,
    logger
) -> Path:
    """Run relation discovery stage."""
    from concepts.pipeline.relations import RelationDiscoverer

    logger.info("=" * 50)
    logger.info("STAGE 4: Relation Discovery")
    logger.info("=" * 50)

    # Load entities
    with open(resolved_file, 'r') as f:
        entities_data = json.load(f)

    entities = entities_data.get("resolved_entities", [])

    # Load chunks
    with open(chunks_file, 'r') as f:
        chunks_data = json.load(f)

    chunks = chunks_data.get("chunks", [])

    # Discover relations
    discoverer = RelationDiscoverer(
        use_gliner2=False,
        use_dep_parse=True,
        use_cooccurrence=True,
        use_llm=False,
    )

    relations = discoverer.discover(entities, chunks)

    # Save results
    relations_file = output_dir / "relations" / "relations.json"
    relations_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "created_at": datetime.now().isoformat(),
        "source_entities": str(resolved_file),
        "source_chunks": str(chunks_file),
        "metrics": discoverer.metrics.to_dict(),
        "relations": [r.to_dict() for r in relations],
    }

    with open(relations_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Discovered {len(relations)} relations")
    logger.info(f"Saved to: {relations_file}")

    return relations_file


def run_load(
    resolved_file: Path,
    relations_file: Path,
    chunks_file: Path,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    logger
):
    """Run Neo4j loading stage."""
    from concepts.pipeline.loader import load_to_neo4j

    logger.info("=" * 50)
    logger.info("STAGE 5: Neo4j Loading")
    logger.info("=" * 50)

    metrics = load_to_neo4j(
        concepts_file=resolved_file,
        relations_file=relations_file,
        chunks_file=chunks_file,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )

    logger.info(f"Loaded {metrics.concepts_created} concepts, {metrics.concept_edges} edges")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run the full concept extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory with documents (PDF, HTML, etc.)"
    )
    parser.add_argument(
        "--output", "-o",
        default="concepts/output",
        help="Output directory for pipeline results"
    )
    parser.add_argument(
        "--config",
        help="Entity types YAML config file"
    )
    parser.add_argument(
        "--synonyms",
        help="Synonyms YAML config file"
    )
    parser.add_argument(
        "--stages",
        default="convert,chunk,ner,resolve,relations,load",
        help="Comma-separated list of stages to run"
    )
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j connection URI"
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j username"
    )
    parser.add_argument(
        "--neo4j-password",
        default="password",
        help="Neo4j password"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without loading to Neo4j"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Parse stages
    stages = [s.strip().lower() for s in args.stages.split(",")]

    # Validate inputs
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config) if args.config else None
    synonyms_path = Path(args.synonyms) if args.synonyms else None

    # Default synonyms path
    if not synonyms_path:
        default_synonyms = Path(__file__).parent.parent / "config" / "synonyms.yaml"
        if default_synonyms.exists():
            synonyms_path = default_synonyms

    logger.info("=" * 60)
    logger.info("CONCEPT EXTRACTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Stages: {', '.join(stages)}")

    import time
    start_time = time.time()

    chunks_file = output_dir / "entities" / "chunks.json"
    entities_file = output_dir / "entities" / "entities.json"
    resolved_file = output_dir / "entities" / "resolved.json"
    relations_file = output_dir / "relations" / "relations.json"

    # Run stages
    if "convert" in stages or "chunk" in stages:
        chunks_file = run_convert(input_dir, output_dir, logger)

    if "ner" in stages:
        if not chunks_file.exists():
            logger.error(f"Chunks file not found: {chunks_file}")
            logger.error("Run 'convert' stage first")
            sys.exit(1)
        entities_file = run_ner(chunks_file, output_dir, config_path, logger)

    if "resolve" in stages:
        if not entities_file.exists():
            logger.error(f"Entities file not found: {entities_file}")
            logger.error("Run 'ner' stage first")
            sys.exit(1)
        resolved_file = run_resolve(entities_file, output_dir, synonyms_path, logger)

    if "relations" in stages:
        if not resolved_file.exists():
            logger.error(f"Resolved file not found: {resolved_file}")
            logger.error("Run 'resolve' stage first")
            sys.exit(1)
        if not chunks_file.exists():
            logger.error(f"Chunks file not found: {chunks_file}")
            sys.exit(1)
        relations_file = run_relations(resolved_file, chunks_file, output_dir, logger)

    if "load" in stages and not args.dry_run:
        if not resolved_file.exists():
            logger.error(f"Resolved file not found: {resolved_file}")
            sys.exit(1)
        run_load(
            resolved_file,
            relations_file if relations_file.exists() else None,
            chunks_file if chunks_file.exists() else None,
            args.neo4j_uri,
            args.neo4j_user,
            args.neo4j_password,
            logger
        )

    # Summary
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed:.2f}s")
    logger.info(f"Output directory: {output_dir}")

    # Save run summary
    summary_file = output_dir / "metrics" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_at": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "stages": stages,
        "elapsed_seconds": round(elapsed, 2),
        "files": {
            "chunks": str(chunks_file) if chunks_file.exists() else None,
            "entities": str(entities_file) if entities_file.exists() else None,
            "resolved": str(resolved_file) if resolved_file.exists() else None,
            "relations": str(relations_file) if relations_file.exists() else None,
        }
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
