"""
Concept extraction pipeline modules.

This package contains the core pipeline components for extracting
concepts from documentation and loading them into Neo4j.

Modules:
    convert: Document conversion (PDF, HTML, DOCX -> Markdown)
    chunker: Section-aware document chunking
    ner: Entity extraction using GLiNER2
    resolve: Entity resolution and deduplication
    relations: Relation discovery
    loader: Neo4j graph loading
    dedup: Document-hash idempotency
"""

from .convert import (
    DocumentConverter,
    ConvertedDocument,
    ConversionMetrics,
    convert_documents,
)

from .chunker import (
    DocumentChunker,
    Chunk,
    ChunkMetrics,
    chunk_markdown_files,
    chunk_from_converted_docs,
)

from .ner import (
    EntityExtractor,
    Entity,
    NERMetrics,
    extract_entities_from_chunks_file,
    load_entity_types,
)

from .resolve import (
    EntityResolver,
    ResolvedEntity,
    ResolutionMetrics,
    SynonymDictionary,
    resolve_entities_from_file,
    load_synonyms,
)

from .relations import (
    RelationDiscoverer,
    Relation,
    RelationMetrics,
    discover_relations_from_files,
)

from .loader import (
    Neo4jLoader,
    LoaderMetrics,
    load_to_neo4j,
    compute_doc_hash,
)

__all__ = [
    # convert
    "DocumentConverter",
    "ConvertedDocument",
    "ConversionMetrics",
    "convert_documents",
    # chunker
    "DocumentChunker",
    "Chunk",
    "ChunkMetrics",
    "chunk_markdown_files",
    "chunk_from_converted_docs",
    # ner
    "EntityExtractor",
    "Entity",
    "NERMetrics",
    "extract_entities_from_chunks_file",
    "load_entity_types",
    # resolve
    "EntityResolver",
    "ResolvedEntity",
    "ResolutionMetrics",
    "SynonymDictionary",
    "resolve_entities_from_file",
    "load_synonyms",
    # relations
    "RelationDiscoverer",
    "Relation",
    "RelationMetrics",
    "discover_relations_from_files",
    # loader
    "Neo4jLoader",
    "LoaderMetrics",
    "load_to_neo4j",
    "compute_doc_hash",
]
