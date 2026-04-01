"""
Section-aware document chunking module.

Splits documents into chunks based on document structure (headers)
rather than arbitrary token windows, preserving context for NER extraction.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetrics:
    """Metrics collected during chunking."""
    docs_processed: int = 0
    chunks_created: int = 0
    total_tokens: int = 0
    avg_chunk_tokens: float = 0.0
    min_chunk_tokens: int = 0
    max_chunk_tokens: int = 0
    chunks_with_headers: int = 0

    def to_dict(self) -> dict:
        return {
            "docs_processed": self.docs_processed,
            "chunks_created": self.chunks_created,
            "total_tokens": self.total_tokens,
            "avg_chunk_tokens": round(self.avg_chunk_tokens, 1),
            "min_chunk_tokens": self.min_chunk_tokens,
            "max_chunk_tokens": self.max_chunk_tokens,
            "chunks_with_headers": self.chunks_with_headers,
        }


@dataclass
class Chunk:
    """A chunk of text with metadata for NER processing."""
    chunk_id: str
    text: str
    token_count: int
    source_path: str
    product: str
    section_path: list[str]  # Hierarchy of section headers
    section_header: str  # Immediate parent header
    start_line: int
    end_line: int
    chunk_index: int  # Position in document

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "token_count": self.token_count,
            "source_path": str(self.source_path),
            "product": self.product,
            "section_path": self.section_path,
            "section_header": self.section_header,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_index": self.chunk_index,
        }


@dataclass
class Section:
    """A section of the document defined by a header."""
    header: str
    level: int
    content: str
    start_line: int
    end_line: int
    parent_headers: list[str] = field(default_factory=list)


class DocumentChunker:
    """
    Splits documents into chunks based on document structure.

    Features:
    - Splits at section boundaries (headers) rather than arbitrary positions
    - Preserves section hierarchy metadata
    - Includes parent header context in each chunk
    - Targets optimal chunk size for NER models (512-1024 tokens)

    Usage:
        chunker = DocumentChunker(target_tokens=768)
        for chunk in chunker.chunk_document(markdown_text, "doc.md", "turbonomic"):
            print(chunk.section_header, chunk.token_count)
    """

    def __init__(
        self,
        target_tokens: int = 768,
        min_tokens: int = 100,
        max_tokens: int = 1024,
        overlap_tokens: int = 50,
    ):
        """
        Initialize the chunker.

        Args:
            target_tokens: Target chunk size in tokens
            min_tokens: Minimum chunk size (smaller chunks merged with previous)
            max_tokens: Maximum chunk size (larger sections split)
            overlap_tokens: Token overlap between chunks for context
        """
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.metrics = ChunkMetrics()

        # Simple tokenizer: split on whitespace and punctuation
        # For production, use tiktoken or model-specific tokenizer
        self._token_pattern = re.compile(r'\b\w+\b|[^\w\s]')

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using simple word-based tokenization.

        For production, consider using tiktoken or the model's tokenizer.
        """
        if not text:
            return 0
        return len(self._token_pattern.findall(text))

    def _parse_sections(self, markdown: str) -> list[Section]:
        """
        Parse markdown into sections based on headers.

        Args:
            markdown: Markdown text

        Returns:
            List of Section objects
        """
        lines = markdown.split('\n')
        sections = []
        current_section = None
        header_stack = []  # Track header hierarchy

        # Pattern to match markdown headers
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')

        for line_num, line in enumerate(lines, 1):
            match = header_pattern.match(line)

            if match:
                # Close previous section
                if current_section:
                    current_section.end_line = line_num - 1
                    sections.append(current_section)

                level = len(match.group(1))
                header_text = match.group(2).strip()

                # Update header stack
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()

                parent_headers = [h[1] for h in header_stack]
                header_stack.append((level, header_text))

                current_section = Section(
                    header=header_text,
                    level=level,
                    content="",
                    start_line=line_num,
                    end_line=line_num,
                    parent_headers=parent_headers.copy(),
                )
            elif current_section:
                current_section.content += line + "\n"
            else:
                # Content before first header - create implicit section
                if line.strip():
                    current_section = Section(
                        header="Introduction",
                        level=0,
                        content=line + "\n",
                        start_line=line_num,
                        end_line=line_num,
                        parent_headers=[],
                    )

        # Close final section
        if current_section:
            current_section.end_line = len(lines)
            sections.append(current_section)

        return sections

    def _merge_small_sections(self, sections: list[Section]) -> list[Section]:
        """Merge sections that are too small."""
        if not sections:
            return sections

        merged = []
        current = None

        for section in sections:
            section_tokens = self.count_tokens(section.content)

            if current is None:
                current = section
            elif self.count_tokens(current.content) + section_tokens < self.min_tokens:
                # Merge with current
                current.content += f"\n## {section.header}\n{section.content}"
                current.end_line = section.end_line
            else:
                merged.append(current)
                current = section

        if current:
            merged.append(current)

        return merged

    def _split_large_section(self, section: Section, source_path: str, product: str, base_index: int) -> list[Chunk]:
        """
        Split a section that exceeds max_tokens into multiple chunks.

        Tries to split at paragraph boundaries.
        """
        chunks = []
        content = section.content
        paragraphs = re.split(r'\n\s*\n', content)

        current_text = ""
        current_start = section.start_line
        chunk_count = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.count_tokens(para)
            current_tokens = self.count_tokens(current_text)

            if current_tokens + para_tokens > self.max_tokens and current_text:
                # Create chunk from current content
                chunks.append(self._create_chunk(
                    text=current_text.strip(),
                    section=section,
                    source_path=source_path,
                    product=product,
                    start_line=current_start,
                    end_line=current_start + current_text.count('\n'),
                    chunk_index=base_index + chunk_count,
                ))
                chunk_count += 1

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_text)
                current_text = overlap_text + para + "\n\n"
                current_start = current_start + current_text.count('\n')
            else:
                current_text += para + "\n\n"

        # Final chunk
        if current_text.strip():
            chunks.append(self._create_chunk(
                text=current_text.strip(),
                section=section,
                source_path=source_path,
                product=product,
                start_line=current_start,
                end_line=section.end_line,
                chunk_index=base_index + chunk_count,
            ))

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get the last N tokens of text for overlap."""
        tokens = self._token_pattern.findall(text)
        if len(tokens) <= self.overlap_tokens:
            return text

        # Find where to cut
        overlap_tokens = tokens[-self.overlap_tokens:]
        # Reconstruct text (simplified - just use last part)
        words = text.split()
        overlap_words = words[-self.overlap_tokens:] if len(words) > self.overlap_tokens else words
        return " ".join(overlap_words) + "\n\n"

    def _create_chunk(
        self,
        text: str,
        section: Section,
        source_path: str,
        product: str,
        start_line: int,
        end_line: int,
        chunk_index: int,
    ) -> Chunk:
        """Create a Chunk object with full context."""
        # Build section path
        section_path = section.parent_headers + [section.header]

        # Add context header if section has a meaningful header
        if section.header and section.header != "Introduction":
            context_header = " > ".join(section_path)
            full_text = f"[Section: {context_header}]\n\n{text}"
        else:
            full_text = text

        token_count = self.count_tokens(full_text)

        # Generate chunk ID
        source_stem = Path(source_path).stem
        chunk_id = f"{product}_{source_stem}_chunk_{chunk_index:04d}"

        return Chunk(
            chunk_id=chunk_id,
            text=full_text,
            token_count=token_count,
            source_path=source_path,
            product=product,
            section_path=section_path,
            section_header=section.header,
            start_line=start_line,
            end_line=end_line,
            chunk_index=chunk_index,
        )

    def chunk_document(
        self,
        markdown: str,
        source_path: str,
        product: str = "unknown",
    ) -> Iterator[Chunk]:
        """
        Chunk a markdown document into sections.

        Args:
            markdown: Markdown text to chunk
            source_path: Path to source file (for metadata)
            product: Product name (for metadata)

        Yields:
            Chunk objects
        """
        # Parse into sections
        sections = self._parse_sections(markdown)

        if not sections:
            # No headers found - treat whole document as one section
            token_count = self.count_tokens(markdown)
            if token_count > 0:
                yield Chunk(
                    chunk_id=f"{product}_{Path(source_path).stem}_chunk_0000",
                    text=markdown,
                    token_count=token_count,
                    source_path=source_path,
                    product=product,
                    section_path=[],
                    section_header="",
                    start_line=1,
                    end_line=markdown.count('\n') + 1,
                    chunk_index=0,
                )
            return

        # Merge small sections
        sections = self._merge_small_sections(sections)

        chunk_index = 0
        for section in sections:
            section_tokens = self.count_tokens(section.content)

            if section_tokens > self.max_tokens:
                # Split large section
                sub_chunks = self._split_large_section(
                    section, source_path, product, chunk_index
                )
                for chunk in sub_chunks:
                    yield chunk
                    chunk_index += 1
            elif section_tokens >= self.min_tokens:
                # Create single chunk from section
                yield self._create_chunk(
                    text=section.content.strip(),
                    section=section,
                    source_path=source_path,
                    product=product,
                    start_line=section.start_line,
                    end_line=section.end_line,
                    chunk_index=chunk_index,
                )
                chunk_index += 1
            # Skip very small sections (will be handled by merge)

    def chunk_documents(
        self,
        documents: Iterator[tuple[str, str, str]],
    ) -> Iterator[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: Iterator of (markdown, source_path, product) tuples

        Yields:
            Chunk objects from all documents
        """
        chunk_sizes = []

        for markdown, source_path, product in documents:
            self.metrics.docs_processed += 1

            for chunk in self.chunk_document(markdown, source_path, product):
                self.metrics.chunks_created += 1
                self.metrics.total_tokens += chunk.token_count
                chunk_sizes.append(chunk.token_count)

                if chunk.section_header:
                    self.metrics.chunks_with_headers += 1

                yield chunk

        # Calculate statistics
        if chunk_sizes:
            self.metrics.avg_chunk_tokens = sum(chunk_sizes) / len(chunk_sizes)
            self.metrics.min_chunk_tokens = min(chunk_sizes)
            self.metrics.max_chunk_tokens = max(chunk_sizes)


def chunk_markdown_files(
    input_dir: Path,
    target_tokens: int = 768,
    min_tokens: int = 100,
    max_tokens: int = 1024,
) -> tuple[list[Chunk], ChunkMetrics]:
    """
    Convenience function to chunk all markdown files in a directory.

    Args:
        input_dir: Directory containing markdown files
        target_tokens: Target chunk size
        min_tokens: Minimum chunk size
        max_tokens: Maximum chunk size

    Returns:
        Tuple of (list of Chunks, ChunkMetrics)
    """
    chunker = DocumentChunker(
        target_tokens=target_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )

    input_dir = Path(input_dir)
    chunks = []

    def doc_generator():
        for md_file in input_dir.rglob("*.md"):
            with open(md_file, 'r', encoding='utf-8') as f:
                markdown = f.read()

            # Extract product from path
            relative = md_file.relative_to(input_dir)
            product = relative.parts[0] if len(relative.parts) > 1 else "unknown"

            yield markdown, str(md_file), product

    for chunk in chunker.chunk_documents(doc_generator()):
        chunks.append(chunk)
        logger.info(f"Created chunk: {chunk.chunk_id} ({chunk.token_count} tokens)")

    return chunks, chunker.metrics


def chunk_from_converted_docs(
    converted_docs,  # Iterator of ConvertedDocument
    target_tokens: int = 768,
    min_tokens: int = 100,
    max_tokens: int = 1024,
) -> Iterator[Chunk]:
    """
    Chunk documents from the convert module.

    Args:
        converted_docs: Iterator of ConvertedDocument objects
        target_tokens: Target chunk size
        min_tokens: Minimum chunk size
        max_tokens: Maximum chunk size

    Yields:
        Chunk objects
    """
    chunker = DocumentChunker(
        target_tokens=target_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )

    def doc_generator():
        for doc in converted_docs:
            yield doc.markdown, str(doc.source_path), doc.product

    yield from chunker.chunk_documents(doc_generator())


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="Chunk markdown documents")
    parser.add_argument("--input", "-i", required=True, help="Input directory with markdown files")
    parser.add_argument("--output", "-o", help="Output JSON file for chunks")
    parser.add_argument("--target-tokens", type=int, default=768, help="Target chunk size")
    parser.add_argument("--min-tokens", type=int, default=100, help="Minimum chunk size")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum chunk size")

    args = parser.parse_args()

    chunks, metrics = chunk_markdown_files(
        input_dir=Path(args.input),
        target_tokens=args.target_tokens,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    print("\nChunking Summary:")
    print(json.dumps(metrics.to_dict(), indent=2))

    if args.output:
        output_data = {
            "metrics": metrics.to_dict(),
            "chunks": [c.to_dict() for c in chunks],
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nChunks saved to: {args.output}")
    else:
        print(f"\nGenerated {len(chunks)} chunks")
        if chunks:
            print("\nSample chunks:")
            for chunk in chunks[:3]:
                print(f"  - {chunk.chunk_id}: {chunk.section_header} ({chunk.token_count} tokens)")
