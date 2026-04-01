#!/usr/bin/env python3
"""
CLI script for document conversion and chunking.

Usage:
    # Convert PDFs to markdown
    python -m concepts.scripts.convert_docs --input docs/product_docs/raw/pdf --output docs/product_docs/processed

    # Convert and chunk in one step
    python -m concepts.scripts.convert_docs --input docs/product_docs/raw/pdf --output docs/product_docs/processed --chunk --chunk-output concepts/output/chunks.json

    # Chunk existing markdown files
    python -m concepts.scripts.convert_docs --input docs/product_docs/processed --chunk-only --chunk-output concepts/output/chunks.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from concepts.pipeline.convert import DocumentConverter, convert_documents
from concepts.pipeline.chunker import DocumentChunker, chunk_markdown_files, chunk_from_converted_docs


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def run_conversion(args):
    """Run document conversion."""
    logger = logging.getLogger(__name__)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting documents from: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Determine extensions based on input type
    extensions = args.extensions
    if not extensions:
        # Auto-detect based on what's in the directory
        found_extensions = set()
        for ext in ['.pdf', '.html', '.htm', '.docx', '.md']:
            if list(input_dir.rglob(f'*{ext}')):
                found_extensions.add(ext)
        extensions = list(found_extensions) if found_extensions else ['.pdf', '.html', '.md']
        logger.info(f"Auto-detected extensions: {extensions}")

    # Run conversion
    converter = DocumentConverter(use_ocr=not args.no_ocr)
    converted_docs = []

    for doc in converter.convert_directory(input_dir, extensions):
        output_path = converter.save_markdown(doc, output_dir)
        logger.info(f"Converted: {doc.source_path.name} -> {output_path.name} ({doc.char_count:,} chars)")
        converted_docs.append(doc)

    return converter.metrics, converted_docs


def run_chunking(args, converted_docs=None):
    """Run document chunking."""
    logger = logging.getLogger(__name__)

    chunker = DocumentChunker(
        target_tokens=args.chunk_size,
        min_tokens=args.min_chunk,
        max_tokens=args.max_chunk,
    )

    chunks = []

    if converted_docs:
        # Chunk from converted documents
        logger.info(f"Chunking {len(converted_docs)} converted documents...")

        def doc_gen():
            for doc in converted_docs:
                yield doc.markdown, str(doc.source_path), doc.product

        for chunk in chunker.chunk_documents(doc_gen()):
            chunks.append(chunk)
    else:
        # Chunk from existing markdown files
        input_dir = Path(args.input)
        logger.info(f"Chunking markdown files from: {input_dir}")

        def doc_gen():
            for md_file in input_dir.rglob("*.md"):
                with open(md_file, 'r', encoding='utf-8') as f:
                    markdown = f.read()
                relative = md_file.relative_to(input_dir)
                product = relative.parts[0] if len(relative.parts) > 1 else "unknown"
                yield markdown, str(md_file), product

        for chunk in chunker.chunk_documents(doc_gen()):
            chunks.append(chunk)

    logger.info(f"Created {len(chunks)} chunks")

    # Save chunks if output specified
    if args.chunk_output:
        output_file = Path(args.chunk_output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "created_at": datetime.now().isoformat(),
            "metrics": chunker.metrics.to_dict(),
            "chunks": [c.to_dict() for c in chunks],
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Chunks saved to: {output_file}")

    return chunker.metrics, chunks


def main():
    parser = argparse.ArgumentParser(
        description="Convert documents to markdown and/or chunk for NER processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory with documents"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for converted markdown files"
    )
    parser.add_argument(
        "--extensions", "-e",
        nargs="+",
        help="File extensions to process (default: auto-detect)"
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR for scanned PDFs"
    )

    # Chunking options
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Also chunk the converted documents"
    )
    parser.add_argument(
        "--chunk-only",
        action="store_true",
        help="Only chunk existing markdown files (no conversion)"
    )
    parser.add_argument(
        "--chunk-output",
        help="Output JSON file for chunks"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=768,
        help="Target chunk size in tokens (default: 768)"
    )
    parser.add_argument(
        "--min-chunk",
        type=int,
        default=100,
        help="Minimum chunk size in tokens (default: 100)"
    )
    parser.add_argument(
        "--max-chunk",
        type=int,
        default=1024,
        help="Maximum chunk size in tokens (default: 1024)"
    )

    # Output options
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

    # Validate arguments
    if not args.chunk_only and not args.output:
        parser.error("--output is required unless --chunk-only is specified")

    if args.chunk_only and not args.chunk_output:
        parser.error("--chunk-output is required when using --chunk-only")

    all_metrics = {}

    # Run conversion if not chunk-only
    converted_docs = None
    if not args.chunk_only:
        logger.info("=" * 50)
        logger.info("STAGE: Document Conversion")
        logger.info("=" * 50)

        conv_metrics, converted_docs = run_conversion(args)
        if conv_metrics:
            all_metrics["conversion"] = conv_metrics.to_dict()
            logger.info("\nConversion Metrics:")
            logger.info(f"  Documents converted: {conv_metrics.docs_converted}")
            logger.info(f"  Documents failed: {conv_metrics.docs_failed}")
            logger.info(f"  Characters extracted: {conv_metrics.chars_extracted:,}")
            logger.info(f"  Headers found: {conv_metrics.headers_found}")
            logger.info(f"  Tables detected: {conv_metrics.tables_detected}")
            logger.info(f"  Time: {conv_metrics.conversion_time_seconds:.2f}s")

    # Run chunking if requested
    if args.chunk or args.chunk_only:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE: Document Chunking")
        logger.info("=" * 50)

        chunk_metrics, chunks = run_chunking(args, converted_docs)
        if chunk_metrics:
            all_metrics["chunking"] = chunk_metrics.to_dict()
            logger.info("\nChunking Metrics:")
            logger.info(f"  Documents processed: {chunk_metrics.docs_processed}")
            logger.info(f"  Chunks created: {chunk_metrics.chunks_created}")
            logger.info(f"  Total tokens: {chunk_metrics.total_tokens:,}")
            logger.info(f"  Avg chunk tokens: {chunk_metrics.avg_chunk_tokens:.1f}")
            logger.info(f"  Min/Max tokens: {chunk_metrics.min_chunk_tokens}/{chunk_metrics.max_chunk_tokens}")

    # Save metrics
    if args.metrics_output:
        metrics_file = Path(args.metrics_output)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "created_at": datetime.now().isoformat(),
            "input": str(args.input),
            "output": str(args.output) if args.output else None,
            **all_metrics
        }

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"\nMetrics saved to: {metrics_file}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
