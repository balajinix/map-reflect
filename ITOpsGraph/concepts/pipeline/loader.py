"""
Neo4j graph loader module.

Loads concepts and relations into Neo4j with:
1. Concept nodes with vector embeddings
2. Concept-to-Concept relationship edges
3. API graph integration (links to Product/Type/Field)
4. Source traceability (MENTIONED_IN edges)
5. Document-hash idempotency
"""

import logging
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LoaderMetrics:
    """Metrics collected during Neo4j loading."""
    concepts_created: int = 0
    concepts_updated: int = 0
    concepts_skipped: int = 0
    concept_edges: int = 0
    embeddings_stored: int = 0
    api_links_product: int = 0
    api_links_type: int = 0
    api_links_field: int = 0
    mentioned_in_edges: int = 0
    documents_processed: int = 0
    documents_skipped: int = 0
    orphan_concepts: int = 0
    load_time_seconds: float = 0.0
    # GraphRAG metrics
    chunks_created: int = 0
    chunk_embeddings_stored: int = 0
    has_chunk_edges: int = 0
    next_chunk_edges: int = 0
    mentions_edges: int = 0
    # Document-API links
    doc_product_edges: int = 0
    doc_type_edges: int = 0
    doc_field_edges: int = 0

    def to_dict(self) -> dict:
        return {
            "concepts_created": self.concepts_created,
            "concepts_updated": self.concepts_updated,
            "concepts_skipped": self.concepts_skipped,
            "concept_edges": self.concept_edges,
            "embeddings_stored": self.embeddings_stored,
            "api_links_product": self.api_links_product,
            "api_links_type": self.api_links_type,
            "api_links_field": self.api_links_field,
            "mentioned_in_edges": self.mentioned_in_edges,
            "documents_processed": self.documents_processed,
            "documents_skipped": self.documents_skipped,
            "orphan_concepts": self.orphan_concepts,
            "load_time_seconds": round(self.load_time_seconds, 2),
            # GraphRAG metrics
            "chunks_created": self.chunks_created,
            "chunk_embeddings_stored": self.chunk_embeddings_stored,
            "has_chunk_edges": self.has_chunk_edges,
            "next_chunk_edges": self.next_chunk_edges,
            "mentions_edges": self.mentions_edges,
            # Document-API links
            "doc_product_edges": self.doc_product_edges,
            "doc_type_edges": self.doc_type_edges,
            "doc_field_edges": self.doc_field_edges,
        }


def compute_doc_hash(content: str) -> str:
    """Compute SHA-256 hash of document content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


class Neo4jLoader:
    """
    Loads concepts and relations into Neo4j.

    Features:
    - Creates Concept nodes with vector embeddings
    - Stores embeddings in Neo4j vector index for similarity search
    - Creates typed relationship edges between concepts
    - Links concepts to existing API graph (Product, Type, Field)
    - Maintains source traceability with MENTIONED_IN edges
    - Idempotent: skips unchanged documents via content hash
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dimensions: int = 384,
        create_vector_index: bool = True,
        link_to_api_graph: bool = True,
        product_link_threshold: float = 0.7,
        type_link_threshold: float = 0.8,
        field_link_threshold: float = 0.85,
    ):
        """
        Initialize the Neo4j loader.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            embedding_model: Sentence transformer model for embeddings
            embedding_dimensions: Embedding vector dimensions
            create_vector_index: Create vector index for similarity search
            link_to_api_graph: Create links to existing API graph
            product_link_threshold: Confidence threshold for Product links
            type_link_threshold: Confidence threshold for Type links
            field_link_threshold: Confidence threshold for Field links
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.embedding_model_name = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.create_vector_index = create_vector_index
        self.link_to_api_graph = link_to_api_graph
        self.product_link_threshold = product_link_threshold
        self.type_link_threshold = type_link_threshold
        self.field_link_threshold = field_link_threshold

        self._driver = None
        self._embedding_model = None

        self.metrics = LoaderMetrics()

    def _get_driver(self):
        """Lazy load Neo4j driver."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password)
                )
                # Test connection
                with self._driver.session() as session:
                    session.run("RETURN 1")
                logger.info(f"Connected to Neo4j at {self.neo4j_uri}")
            except ImportError:
                logger.error("neo4j package not installed. Please install with: pip install neo4j")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise
        return self._driver

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Embedding model loaded")
            except ImportError:
                logger.warning("sentence-transformers not installed. Embeddings disabled.")
                self._embedding_model = "unavailable"
        return self._embedding_model

    def _compute_embedding(self, text: str) -> list[float] | None:
        """Compute embedding for text."""
        model = self._get_embedding_model()
        if model == "unavailable":
            return None
        try:
            embedding = model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.warning(f"Failed to compute embedding: {e}")
            return None

    def close(self):
        """Close Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def setup_schema(self):
        """Create Neo4j schema: constraints and indexes."""
        driver = self._get_driver()

        with driver.session() as session:
            # Concept node constraints
            session.run("""
                CREATE CONSTRAINT concept_name IF NOT EXISTS
                FOR (c:Concept) REQUIRE c.name IS UNIQUE
            """)

            # Document node constraint
            session.run("""
                CREATE CONSTRAINT document_path IF NOT EXISTS
                FOR (d:Document) REQUIRE d.path IS UNIQUE
            """)

            # Chunk node constraint (GraphRAG)
            session.run("""
                CREATE CONSTRAINT chunk_id IF NOT EXISTS
                FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE
            """)

            # Vector index for concept embeddings
            if self.create_vector_index:
                try:
                    session.run(f"""
                        CREATE VECTOR INDEX concept_embeddings IF NOT EXISTS
                        FOR (c:Concept) ON (c.embedding)
                        OPTIONS {{indexConfig: {{
                            `vector.dimensions`: {self.embedding_dimensions},
                            `vector.similarity_function`: 'cosine'
                        }}}}
                    """)
                    logger.info("Concept vector index created/verified")
                except Exception as e:
                    logger.warning(f"Concept vector index creation failed (may already exist): {e}")

                # Vector index for chunk embeddings (GraphRAG)
                try:
                    session.run(f"""
                        CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                        FOR (ch:Chunk) ON (ch.embedding)
                        OPTIONS {{indexConfig: {{
                            `vector.dimensions`: {self.embedding_dimensions},
                            `vector.similarity_function`: 'cosine'
                        }}}}
                    """)
                    logger.info("Chunk vector index created/verified")
                except Exception as e:
                    logger.warning(f"Chunk vector index creation failed (may already exist): {e}")

            # Full-text index for concept search
            try:
                session.run("""
                    CREATE FULLTEXT INDEX concept_search IF NOT EXISTS
                    FOR (c:Concept) ON EACH [c.name, c.description, c.aliases]
                """)
            except Exception as e:
                logger.debug(f"Full-text index creation: {e}")

            # Full-text index for chunk search (GraphRAG)
            try:
                session.run("""
                    CREATE FULLTEXT INDEX chunk_search IF NOT EXISTS
                    FOR (ch:Chunk) ON EACH [ch.text, ch.section_header]
                """)
            except Exception as e:
                logger.debug(f"Chunk full-text index creation: {e}")

        logger.info("Neo4j schema setup complete")

    def should_process_document(self, doc_path: str, content_hash: str) -> bool:
        """Check if document needs processing (idempotency)."""
        driver = self._get_driver()

        with driver.session() as session:
            result = session.run("""
                MATCH (d:Document {path: $path})
                RETURN d.content_hash AS hash
            """, path=doc_path)

            record = result.single()
            if record is None:
                return True  # New document
            return record["hash"] != content_hash  # Changed document

    def load_concepts(
        self,
        concepts: list[dict],
        documents: list[dict] | None = None,
    ) -> int:
        """
        Load concepts into Neo4j.

        Args:
            concepts: List of resolved concept dicts with keys:
                - canonical: Concept name
                - entity_type: Fine-grained type
                - hierarchy: Hierarchy category
                - aliases: List of alternative names
                - source_docs: List of source document paths
                - source_products: List of product names
                - confidence: Extraction confidence
            documents: Optional list of document dicts for traceability

        Returns:
            Number of concepts loaded
        """
        driver = self._get_driver()
        loaded = 0

        for concept in concepts:
            canonical = concept.get("canonical", "")
            if not canonical:
                continue

            # Compute embedding
            embedding = self._compute_embedding(canonical)

            # Prepare properties
            props = {
                "name": canonical,
                "canonical_name": canonical.lower().replace(" ", "_"),
                "entity_type": concept.get("entity_type", ""),
                "hierarchy": concept.get("hierarchy", ""),
                "aliases": concept.get("aliases", []),
                "source_docs": concept.get("source_docs", []),
                "source_products": concept.get("source_products", []),
                "confidence": concept.get("confidence", 0.0),
                "mentions": concept.get("mentions", 1),
                "resolution_method": concept.get("resolution_method", ""),
                "updated_at": datetime.now().isoformat(),
            }

            if embedding:
                props["embedding"] = embedding
                self.metrics.embeddings_stored += 1

            with driver.session() as session:
                result = session.run("""
                    MERGE (c:Concept {name: $name})
                    ON CREATE SET
                        c.canonical_name = $canonical_name,
                        c.entity_type = $entity_type,
                        c.hierarchy = $hierarchy,
                        c.aliases = $aliases,
                        c.source_docs = $source_docs,
                        c.source_products = $source_products,
                        c.confidence = $confidence,
                        c.mentions = $mentions,
                        c.resolution_method = $resolution_method,
                        c.created_at = datetime(),
                        c.updated_at = $updated_at,
                        c.embedding = $embedding
                    ON MATCH SET
                        c.entity_type = $entity_type,
                        c.hierarchy = $hierarchy,
                        c.aliases = $aliases,
                        c.source_docs = $source_docs,
                        c.source_products = $source_products,
                        c.confidence = $confidence,
                        c.mentions = $mentions,
                        c.resolution_method = $resolution_method,
                        c.updated_at = $updated_at,
                        c.embedding = $embedding
                    RETURN c.created_at = c.updated_at AS is_new
                """, **props)

                record = result.single()
                if record and record["is_new"]:
                    self.metrics.concepts_created += 1
                else:
                    self.metrics.concepts_updated += 1

                loaded += 1

        return loaded

    def load_relations(self, relations: list[dict]) -> int:
        """
        Load relations into Neo4j as edges between Concept nodes.

        Args:
            relations: List of relation dicts with keys:
                - head: Head concept name
                - tail: Tail concept name
                - relation_type: Relationship type
                - confidence: Relation confidence
                - method: Discovery method
                - source_sentence: Source sentence (optional)
                - source_chunk: Source chunk ID (optional)
                - source_doc: Source document path (optional)

        Returns:
            Number of relations loaded
        """
        driver = self._get_driver()
        loaded = 0

        for relation in relations:
            head = relation.get("head", "")
            tail = relation.get("tail", "")
            rel_type = relation.get("relation_type", "RELATED_TO")

            if not head or not tail:
                continue

            # Sanitize relation type for Neo4j
            rel_type_safe = rel_type.upper().replace(" ", "_").replace("-", "_")

            props = {
                "head": head,
                "tail": tail,
                "rel_type": rel_type_safe,
                "confidence": relation.get("confidence", 0.5),
                "method": relation.get("method", ""),
                "source_sentence": relation.get("source_sentence", "")[:500],  # Truncate
                "source_chunk": relation.get("source_chunk", ""),
                "source_doc": relation.get("source_doc", ""),
            }

            with driver.session() as session:
                # Use APOC or dynamic relationship type
                # Fallback to generic approach if APOC not available
                try:
                    session.run(f"""
                        MATCH (h:Concept {{name: $head}})
                        MATCH (t:Concept {{name: $tail}})
                        MERGE (h)-[r:{rel_type_safe}]->(t)
                        ON CREATE SET
                            r.confidence = $confidence,
                            r.method = $method,
                            r.source_sentence = $source_sentence,
                            r.source_chunk = $source_chunk,
                            r.source_doc = $source_doc,
                            r.created_at = datetime()
                        ON MATCH SET
                            r.confidence = CASE
                                WHEN r.confidence < $confidence THEN $confidence
                                ELSE r.confidence
                            END,
                            r.updated_at = datetime()
                    """, **props)
                    loaded += 1
                    self.metrics.concept_edges += 1
                except Exception as e:
                    logger.warning(f"Failed to create relation {head}-[{rel_type}]->{tail}: {e}")

        return loaded

    def load_documents(self, documents: list[dict]) -> int:
        """
        Load document nodes for traceability.

        Args:
            documents: List of document dicts with keys:
                - path: Document path
                - content_hash: SHA-256 hash of content
                - product: Product name
                - title: Document title (optional)

        Returns:
            Number of documents loaded
        """
        driver = self._get_driver()
        loaded = 0

        for doc in documents:
            path = doc.get("path", "")
            if not path:
                continue

            props = {
                "path": path,
                "content_hash": doc.get("content_hash", ""),
                "product": doc.get("product", ""),
                "title": doc.get("title", ""),
                "updated_at": datetime.now().isoformat(),
            }

            with driver.session() as session:
                session.run("""
                    MERGE (d:Document {path: $path})
                    SET d.content_hash = $content_hash,
                        d.product = $product,
                        d.title = $title,
                        d.updated_at = $updated_at
                """, **props)
                loaded += 1
                self.metrics.documents_processed += 1

        return loaded

    def link_documents_to_api_graph(self, documents: list[dict]) -> dict:
        """
        Create edges from Documents to API graph nodes.

        Links Documents to:
        - Product: DOCUMENTS edge (Document documents Product)
        - Type: DOCUMENTS edge (Document documents Type) - for types in same product
        - Field: DOCUMENTS edge (Document documents Field) - via concepts

        Args:
            documents: List of document dicts with product info

        Returns:
            Dict with edge counts
        """
        driver = self._get_driver()
        links = {"product": 0, "type": 0, "field": 0}

        for doc in documents:
            path = doc.get("path", "")
            product = doc.get("product", "")

            if not path or not product:
                continue

            with driver.session() as session:
                # Link Document to Product
                result = session.run("""
                    MATCH (d:Document {path: $path})
                    MATCH (p:Product {name: $product})
                    MERGE (d)-[r:DOCUMENTS]->(p)
                    ON CREATE SET r.created_at = datetime()
                    RETURN r
                """, path=path, product=product)

                if result.single():
                    links["product"] += 1
                    self.metrics.doc_product_edges += 1

                # Link Document to Types in the same product
                # (Document describes types from its product)
                result = session.run("""
                    MATCH (d:Document {path: $path})
                    MATCH (t:Type {product: $product})
                    MERGE (d)-[r:DOCUMENTS]->(t)
                    ON CREATE SET r.created_at = datetime(), r.match_type = 'product_scope'
                    RETURN count(r) AS cnt
                """, path=path, product=product)

                record = result.single()
                if record:
                    links["type"] += record["cnt"]
                    self.metrics.doc_type_edges += record["cnt"]

        return links

    def link_chunks_to_api_graph(self, chunks: list[dict], concepts: list[dict]) -> dict:
        """
        Create edges from Chunks to API graph via Concepts.

        If a Chunk mentions a Concept that DESCRIBES a Type/Field,
        create a transitive link from Chunk to that Type/Field.

        Args:
            chunks: List of chunk dicts
            concepts: List of concept dicts with source_chunks

        Returns:
            Dict with edge counts
        """
        driver = self._get_driver()
        links = {"type": 0, "field": 0}

        # Build chunk -> concepts mapping
        chunk_concepts = {}
        for concept in concepts:
            name = concept.get("canonical", "")
            for chunk_id in concept.get("source_chunks", []):
                if chunk_id not in chunk_concepts:
                    chunk_concepts[chunk_id] = []
                chunk_concepts[chunk_id].append(name)

        # For each chunk, link to Types/Fields via its concepts
        with driver.session() as session:
            for chunk_id, concept_names in chunk_concepts.items():
                for concept_name in concept_names:
                    # Link Chunk to Types that describe this concept
                    result = session.run("""
                        MATCH (ch:Chunk {chunk_id: $chunk_id})
                        MATCH (t:Type)-[:DESCRIBES]->(c:Concept {name: $concept_name})
                        MERGE (ch)-[r:REFERENCES]->(t)
                        ON CREATE SET r.created_at = datetime(), r.via_concept = $concept_name
                        RETURN count(r) AS cnt
                    """, chunk_id=chunk_id, concept_name=concept_name)

                    record = result.single()
                    if record:
                        links["type"] += record["cnt"]

                    # Link Chunk to Fields that describe this concept
                    result = session.run("""
                        MATCH (ch:Chunk {chunk_id: $chunk_id})
                        MATCH (f:Field)-[:DESCRIBES]->(c:Concept {name: $concept_name})
                        MERGE (ch)-[r:REFERENCES]->(f)
                        ON CREATE SET r.created_at = datetime(), r.via_concept = $concept_name
                        RETURN count(r) AS cnt
                    """, chunk_id=chunk_id, concept_name=concept_name)

                    record = result.single()
                    if record:
                        links["field"] += record["cnt"]

        return links

    def load_chunks(
        self,
        chunks: list[dict],
        compute_embeddings: bool = True,
    ) -> int:
        """
        Load chunk nodes for GraphRAG retrieval.

        Creates Chunk nodes with embeddings and establishes:
        - HAS_CHUNK edges from Document to Chunk
        - NEXT_CHUNK edges for sequential ordering

        Args:
            chunks: List of chunk dicts with keys:
                - chunk_id: Unique chunk identifier
                - text: Chunk content
                - token_count: Number of tokens
                - source_path: Document path
                - product: Product name
                - section_header: Section name
                - section_path: Section hierarchy
                - chunk_index: Position in document
            compute_embeddings: Whether to compute and store embeddings

        Returns:
            Number of chunks loaded
        """
        driver = self._get_driver()
        loaded = 0

        # Group chunks by document for efficient relationship creation
        chunks_by_doc = {}
        for chunk in chunks:
            doc_path = chunk.get("source_path", "")
            if doc_path not in chunks_by_doc:
                chunks_by_doc[doc_path] = []
            chunks_by_doc[doc_path].append(chunk)

        # Sort chunks within each document by index
        for doc_path in chunks_by_doc:
            chunks_by_doc[doc_path].sort(key=lambda c: c.get("chunk_index", 0))

        for doc_path, doc_chunks in chunks_by_doc.items():
            prev_chunk_id = None

            for chunk in doc_chunks:
                chunk_id = chunk.get("chunk_id", "")
                if not chunk_id:
                    continue

                text = chunk.get("text", "")

                # Compute embedding for chunk text
                embedding = None
                if compute_embeddings and text:
                    embedding = self._compute_embedding(text)
                    if embedding:
                        self.metrics.chunk_embeddings_stored += 1

                # Prepare properties
                props = {
                    "chunk_id": chunk_id,
                    "text": text,
                    "token_count": chunk.get("token_count", 0),
                    "source_path": doc_path,
                    "product": chunk.get("product", ""),
                    "section_header": chunk.get("section_header", ""),
                    "section_path": chunk.get("section_path", []),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0),
                    "updated_at": datetime.now().isoformat(),
                }

                if embedding:
                    props["embedding"] = embedding

                with driver.session() as session:
                    # Create or update Chunk node
                    session.run("""
                        MERGE (ch:Chunk {chunk_id: $chunk_id})
                        SET ch.text = $text,
                            ch.token_count = $token_count,
                            ch.source_path = $source_path,
                            ch.product = $product,
                            ch.section_header = $section_header,
                            ch.section_path = $section_path,
                            ch.chunk_index = $chunk_index,
                            ch.start_line = $start_line,
                            ch.end_line = $end_line,
                            ch.updated_at = $updated_at,
                            ch.embedding = $embedding
                    """, **props)

                    # Create HAS_CHUNK edge from Document
                    result = session.run("""
                        MATCH (d:Document {path: $doc_path})
                        MATCH (ch:Chunk {chunk_id: $chunk_id})
                        MERGE (d)-[r:HAS_CHUNK]->(ch)
                        ON CREATE SET r.created_at = datetime()
                        RETURN r
                    """, doc_path=doc_path, chunk_id=chunk_id)

                    if result.single():
                        self.metrics.has_chunk_edges += 1

                    # Create NEXT_CHUNK edge from previous chunk
                    if prev_chunk_id:
                        result = session.run("""
                            MATCH (prev:Chunk {chunk_id: $prev_id})
                            MATCH (curr:Chunk {chunk_id: $curr_id})
                            MERGE (prev)-[r:NEXT_CHUNK]->(curr)
                            ON CREATE SET r.created_at = datetime()
                            RETURN r
                        """, prev_id=prev_chunk_id, curr_id=chunk_id)

                        if result.single():
                            self.metrics.next_chunk_edges += 1

                    loaded += 1
                    self.metrics.chunks_created += 1
                    prev_chunk_id = chunk_id

        return loaded

    def create_mentioned_in_edges(
        self,
        concepts: list[dict],
        chunks: list[dict] | None = None,
    ) -> int:
        """
        Create MENTIONED_IN edges from Concepts to Documents.
        Also creates MENTIONS edges from Chunks to Concepts for provenance.

        Args:
            concepts: List of concept dicts with source_docs and source_chunks
            chunks: Optional chunks for section/chunk_id traceability

        Returns:
            Number of edges created
        """
        driver = self._get_driver()
        created = 0

        # Build chunk lookup
        chunk_lookup = {}
        if chunks:
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id:
                    chunk_lookup[chunk_id] = chunk

        for concept in concepts:
            name = concept.get("canonical", "")
            source_docs = concept.get("source_docs", [])
            source_chunks = concept.get("source_chunks", [])

            # Create MENTIONED_IN edges to Documents
            for doc_path in source_docs:
                with driver.session() as session:
                    result = session.run("""
                        MATCH (c:Concept {name: $name})
                        MATCH (d:Document {path: $doc_path})
                        MERGE (c)-[r:MENTIONED_IN]->(d)
                        ON CREATE SET r.created_at = datetime()
                        RETURN r
                    """, name=name, doc_path=doc_path)

                    if result.single():
                        created += 1
                        self.metrics.mentioned_in_edges += 1

            # Create MENTIONS edges from Chunks to Concepts (GraphRAG provenance)
            for chunk_id in source_chunks:
                with driver.session() as session:
                    result = session.run("""
                        MATCH (ch:Chunk {chunk_id: $chunk_id})
                        MATCH (c:Concept {name: $name})
                        MERGE (ch)-[r:MENTIONS]->(c)
                        ON CREATE SET r.created_at = datetime()
                        RETURN r
                    """, chunk_id=chunk_id, name=name)

                    if result.single():
                        self.metrics.mentions_edges += 1

        return created

    def link_to_api_graph(self, concepts: list[dict]) -> dict:
        """
        Create edges from API graph nodes (Product, Type, Field) to Concepts.

        Uses multiple matching strategies:
        1. Exact name matching
        2. Word-level matching (concept words appear in type/field names)
        3. Product-scoped matching (same product context)

        Relationships created:
        - Product -[:HAS_CONCEPT]-> Concept
        - Type -[:DESCRIBES]-> Concept (type relates to concept)
        - Field -[:DESCRIBES]-> Concept (field relates to concept)
        - Concept -[:MENTIONED_IN_TYPE]-> Type (concept mentioned in type's domain)

        Args:
            concepts: List of concept dicts

        Returns:
            Dict with counts of links created
        """
        if not self.link_to_api_graph:
            return {"product": 0, "type": 0, "field": 0}

        driver = self._get_driver()
        links = {"product": 0, "type": 0, "field": 0}

        for concept in concepts:
            name = concept.get("canonical", "")
            products = concept.get("source_products", [])
            confidence = concept.get("confidence", 0.5)
            entity_type = concept.get("entity_type", "")

            if not name or len(name) < 3:  # Skip very short names
                continue

            # Normalize concept name for matching
            name_lower = name.lower()
            name_words = set(name_lower.replace("-", " ").replace("_", " ").split())
            # Filter out common words
            stop_words = {"the", "a", "an", "and", "or", "for", "to", "in", "on", "of", "is", "it"}
            name_words = name_words - stop_words

            with driver.session() as session:
                # Link to Product nodes
                if confidence >= self.product_link_threshold:
                    for product in products:
                        result = session.run("""
                            MATCH (p:Product)
                            WHERE toLower(p.name) = toLower($product)
                            MATCH (c:Concept {name: $concept_name})
                            MERGE (p)-[r:HAS_CONCEPT]->(c)
                            ON CREATE SET
                                r.weight = $confidence,
                                r.match_type = 'product_source',
                                r.created_at = datetime()
                            RETURN r
                        """, product=product, concept_name=name, confidence=confidence)

                        if result.single():
                            links["product"] += 1
                            self.metrics.api_links_product += 1

                # Link to Type nodes - multiple strategies
                # Strategy 1: Exact or near-exact name match
                result = session.run("""
                    MATCH (t:Type)
                    WHERE toLower(t.name) = toLower($concept_name)
                       OR toLower(replace(t.name, '_', '')) = toLower(replace($concept_name, ' ', ''))
                    MATCH (c:Concept {name: $concept_name})
                    MERGE (t)-[r:DESCRIBES]->(c)
                    ON CREATE SET
                        r.weight = 1.0,
                        r.match_type = 'exact_name',
                        r.created_at = datetime()
                    RETURN count(r) AS count
                """, concept_name=name)

                record = result.single()
                if record and record["count"] > 0:
                    links["type"] += record["count"]
                    self.metrics.api_links_type += record["count"]

                # Strategy 2: Type name contains concept (for multi-word concepts)
                if len(name_words) >= 2:
                    result = session.run("""
                        MATCH (t:Type)
                        WHERE toLower(t.name) CONTAINS toLower($concept_name)
                        MATCH (c:Concept {name: $concept_name})
                        MERGE (t)-[r:DESCRIBES]->(c)
                        ON CREATE SET
                            r.weight = 0.9,
                            r.match_type = 'name_contains',
                            r.created_at = datetime()
                        RETURN count(r) AS count
                    """, concept_name=name)

                    record = result.single()
                    if record and record["count"] > 0:
                        links["type"] += record["count"]
                        self.metrics.api_links_type += record["count"]

                # Strategy 3: Product-scoped type matching
                # If concept is from product X, link to Types in product X with related names
                for product in products:
                    result = session.run("""
                        MATCH (t:Type)
                        WHERE t.product = $product
                          AND (
                              toLower(t.name) CONTAINS toLower($concept_name)
                              OR toLower(t.description) CONTAINS toLower($concept_name)
                          )
                        MATCH (c:Concept {name: $concept_name})
                        MERGE (t)-[r:DESCRIBES]->(c)
                        ON CREATE SET
                            r.weight = 0.85,
                            r.match_type = 'product_scoped',
                            r.created_at = datetime()
                        RETURN count(r) AS count
                    """, product=product, concept_name=name)

                    record = result.single()
                    if record and record["count"] > 0:
                        links["type"] += record["count"]
                        self.metrics.api_links_type += record["count"]

                # Link to Field nodes - multiple strategies
                # Strategy 1: Exact field name match
                result = session.run("""
                    MATCH (f:Field)
                    WHERE toLower(f.name) = toLower($concept_name)
                       OR toLower(replace(f.name, '_', ' ')) = toLower($concept_name)
                    MATCH (c:Concept {name: $concept_name})
                    MERGE (f)-[r:DESCRIBES]->(c)
                    ON CREATE SET
                        r.weight = 1.0,
                        r.match_type = 'exact_name',
                        r.created_at = datetime()
                    RETURN count(r) AS count
                """, concept_name=name)

                record = result.single()
                if record and record["count"] > 0:
                    links["field"] += record["count"]
                    self.metrics.api_links_field += record["count"]

                # Strategy 2: Field description contains concept
                result = session.run("""
                    MATCH (f:Field)
                    WHERE f.description IS NOT NULL
                      AND toLower(f.description) CONTAINS toLower($concept_name)
                    MATCH (c:Concept {name: $concept_name})
                    MERGE (f)-[r:DESCRIBES]->(c)
                    ON CREATE SET
                        r.weight = 0.8,
                        r.match_type = 'description_match',
                        r.created_at = datetime()
                    RETURN count(r) AS count
                """, concept_name=name)

                record = result.single()
                if record and record["count"] > 0:
                    links["field"] += record["count"]
                    self.metrics.api_links_field += record["count"]

                # Strategy 3: Product-scoped field matching
                for product in products:
                    result = session.run("""
                        MATCH (f:Field)
                        WHERE f.product = $product
                          AND (
                              toLower(f.name) CONTAINS toLower($concept_name)
                              OR (f.description IS NOT NULL AND toLower(f.description) CONTAINS toLower($concept_name))
                          )
                        MATCH (c:Concept {name: $concept_name})
                        MERGE (f)-[r:DESCRIBES]->(c)
                        ON CREATE SET
                            r.weight = 0.75,
                            r.match_type = 'product_scoped',
                            r.created_at = datetime()
                        RETURN count(r) AS count
                    """, product=product, concept_name=name)

                    record = result.single()
                    if record and record["count"] > 0:
                        links["field"] += record["count"]
                        self.metrics.api_links_field += record["count"]

        return links

    def link_types_to_concepts_by_embedding(
        self,
        concepts: list[dict],
        similarity_threshold: float = 0.75,
        limit_per_concept: int = 5,
    ) -> int:
        """
        Link Types to Concepts using embedding similarity.

        Computes embeddings for Type descriptions and matches them to
        concept embeddings using vector similarity.

        Args:
            concepts: List of concept dicts
            similarity_threshold: Minimum similarity score
            limit_per_concept: Max types to link per concept

        Returns:
            Number of links created
        """
        driver = self._get_driver()
        links_created = 0

        # First, ensure Types have embeddings
        logger.info("Computing embeddings for Types (this may take a while)...")

        with driver.session() as session:
            # Get Types without embeddings
            result = session.run("""
                MATCH (t:Type)
                WHERE t.embedding IS NULL AND t.description IS NOT NULL
                RETURN t.name AS name, t.description AS description
                LIMIT 1000
            """)

            types_to_embed = [(r["name"], r["description"]) for r in result]

        if types_to_embed:
            logger.info(f"Computing embeddings for {len(types_to_embed)} Types...")
            for name, description in types_to_embed:
                embedding = self._compute_embedding(description)
                if embedding:
                    with driver.session() as session:
                        session.run("""
                            MATCH (t:Type {name: $name})
                            SET t.embedding = $embedding
                        """, name=name, embedding=embedding)

        # Now link concepts to types by embedding similarity
        for concept in concepts:
            concept_name = concept.get("canonical", "")
            if not concept_name:
                continue

            # Compute concept embedding if not already done
            embedding = self._compute_embedding(concept_name)
            if not embedding:
                continue

            with driver.session() as session:
                # Find similar Types using vector search
                try:
                    result = session.run("""
                        CALL db.index.vector.queryNodes('type_embeddings', $limit, $embedding)
                        YIELD node, score
                        WHERE score >= $threshold
                        MATCH (c:Concept {name: $concept_name})
                        MERGE (node)-[r:DESCRIBES]->(c)
                        ON CREATE SET
                            r.weight = score,
                            r.match_type = 'embedding_similarity',
                            r.created_at = datetime()
                        RETURN count(r) AS count
                    """, limit=limit_per_concept, embedding=embedding,
                        threshold=similarity_threshold, concept_name=concept_name)

                    record = result.single()
                    if record:
                        links_created += record["count"]
                except Exception as e:
                    # Vector index may not exist for Types
                    logger.debug(f"Type embedding search failed: {e}")
                    break

        self.metrics.api_links_type += links_created
        return links_created

    def find_orphan_concepts(self) -> list[str]:
        """Find concepts with no API graph links (no HAS_CONCEPT or DESCRIBES edges)."""
        driver = self._get_driver()

        with driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)
                WHERE NOT (c)<-[:HAS_CONCEPT]-()
                  AND NOT (c)<-[:DESCRIBES]-()
                RETURN c.name AS name
            """)

            orphans = [record["name"] for record in result]
            self.metrics.orphan_concepts = len(orphans)
            return orphans

    def get_concept_api_links(self, concept_name: str) -> dict:
        """
        Get all API graph links for a concept.

        Returns:
            Dict with products, types, and fields linked to the concept
        """
        driver = self._get_driver()

        with driver.session() as session:
            # Get linked Products
            products_result = session.run("""
                MATCH (p:Product)-[:HAS_CONCEPT]->(c:Concept {name: $name})
                RETURN p.name AS name, p.product_name AS product_name
            """, name=concept_name)
            products = [dict(r) for r in products_result]

            # Get linked Types
            types_result = session.run("""
                MATCH (t:Type)-[r:DESCRIBES]->(c:Concept {name: $name})
                RETURN t.name AS name, t.product AS product,
                       t.description AS description, r.match_type AS match_type,
                       r.weight AS weight
                ORDER BY r.weight DESC
            """, name=concept_name)
            types = [dict(r) for r in types_result]

            # Get linked Fields
            fields_result = session.run("""
                MATCH (f:Field)-[r:DESCRIBES]->(c:Concept {name: $name})
                RETURN f.name AS name, f.full_name AS full_name,
                       f.product AS product, f.description AS description,
                       r.match_type AS match_type, r.weight AS weight
                ORDER BY r.weight DESC
            """, name=concept_name)
            fields = [dict(r) for r in fields_result]

            return {
                "concept": concept_name,
                "products": products,
                "types": types,
                "fields": fields,
                "total_links": len(products) + len(types) + len(fields),
            }

    def similarity_search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Search for similar concepts using vector similarity.

        Args:
            query: Search query text
            limit: Maximum number of results

        Returns:
            List of dicts with name, score, and concept properties
        """
        driver = self._get_driver()

        # Compute query embedding
        embedding = self._compute_embedding(query)
        if not embedding:
            logger.warning("Cannot perform similarity search without embeddings")
            return []

        with driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('concept_embeddings', $limit, $embedding)
                YIELD node, score
                RETURN node.name AS name,
                       node.entity_type AS entity_type,
                       node.hierarchy AS hierarchy,
                       score
                ORDER BY score DESC
            """, limit=limit, embedding=embedding)

            return [dict(record) for record in result]

    def chunk_similarity_search(
        self,
        query: str,
        limit: int = 10,
        product: str | None = None,
    ) -> list[dict]:
        """
        Search for similar chunks using vector similarity (GraphRAG retrieval).

        Args:
            query: Search query text
            limit: Maximum number of results
            product: Optional product filter

        Returns:
            List of dicts with chunk_id, text, score, and chunk properties
        """
        driver = self._get_driver()

        # Compute query embedding
        embedding = self._compute_embedding(query)
        if not embedding:
            logger.warning("Cannot perform similarity search without embeddings")
            return []

        with driver.session() as session:
            if product:
                # Filter by product
                result = session.run("""
                    CALL db.index.vector.queryNodes('chunk_embeddings', $limit * 2, $embedding)
                    YIELD node, score
                    WHERE node.product = $product
                    RETURN node.chunk_id AS chunk_id,
                           node.text AS text,
                           node.section_header AS section_header,
                           node.product AS product,
                           node.source_path AS source_path,
                           score
                    ORDER BY score DESC
                    LIMIT $limit
                """, limit=limit, embedding=embedding, product=product)
            else:
                result = session.run("""
                    CALL db.index.vector.queryNodes('chunk_embeddings', $limit, $embedding)
                    YIELD node, score
                    RETURN node.chunk_id AS chunk_id,
                           node.text AS text,
                           node.section_header AS section_header,
                           node.product AS product,
                           node.source_path AS source_path,
                           score
                    ORDER BY score DESC
                """, limit=limit, embedding=embedding)

            return [dict(record) for record in result]

    def get_chunk_context(
        self,
        chunk_id: str,
        window: int = 1,
    ) -> list[dict]:
        """
        Get surrounding chunks for context expansion (GraphRAG).

        Args:
            chunk_id: The chunk to get context for
            window: Number of chunks before/after to include

        Returns:
            List of chunk dicts in order (prev chunks, current, next chunks)
        """
        driver = self._get_driver()

        with driver.session() as session:
            # Get previous chunks
            prev_result = session.run("""
                MATCH path = (start:Chunk)-[:NEXT_CHUNK*1..$window]->(target:Chunk {chunk_id: $chunk_id})
                WITH nodes(path) AS chunks
                UNWIND chunks AS ch
                RETURN ch.chunk_id AS chunk_id,
                       ch.text AS text,
                       ch.section_header AS section_header,
                       ch.chunk_index AS chunk_index
                ORDER BY ch.chunk_index
            """, chunk_id=chunk_id, window=window)

            prev_chunks = [dict(r) for r in prev_result]

            # Get current chunk
            curr_result = session.run("""
                MATCH (ch:Chunk {chunk_id: $chunk_id})
                RETURN ch.chunk_id AS chunk_id,
                       ch.text AS text,
                       ch.section_header AS section_header,
                       ch.chunk_index AS chunk_index
            """, chunk_id=chunk_id)

            curr_chunk = curr_result.single()

            # Get next chunks
            next_result = session.run("""
                MATCH path = (start:Chunk {chunk_id: $chunk_id})-[:NEXT_CHUNK*1..$window]->(target:Chunk)
                WITH nodes(path) AS chunks
                UNWIND chunks AS ch
                WHERE ch.chunk_id <> $chunk_id
                RETURN ch.chunk_id AS chunk_id,
                       ch.text AS text,
                       ch.section_header AS section_header,
                       ch.chunk_index AS chunk_index
                ORDER BY ch.chunk_index
            """, chunk_id=chunk_id, window=window)

            next_chunks = [dict(r) for r in next_result]

            # Combine
            result = prev_chunks
            if curr_chunk:
                result.append(dict(curr_chunk))
            result.extend(next_chunks)

            return result


def load_to_neo4j(
    concepts_file: Path,
    relations_file: Path | None = None,
    chunks_file: Path | None = None,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    create_vector_index: bool = True,
    link_to_api_graph: bool = True,
) -> LoaderMetrics:
    """
    Load concepts and relations into Neo4j.

    Args:
        concepts_file: Path to resolved concepts JSON file
        relations_file: Path to relations JSON file (optional)
        chunks_file: Path to chunks JSON file for traceability (optional)
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        create_vector_index: Create vector index for embeddings
        link_to_api_graph: Create links to existing API graph

    Returns:
        LoaderMetrics with loading statistics
    """
    import time
    start_time = time.time()

    # Load concepts
    with open(concepts_file, 'r') as f:
        concepts_data = json.load(f)

    concepts = concepts_data.get("resolved_entities", concepts_data.get("entities", []))
    logger.info(f"Loaded {len(concepts)} concepts from {concepts_file}")

    # Load relations
    relations = []
    if relations_file and relations_file.exists():
        with open(relations_file, 'r') as f:
            relations_data = json.load(f)
        relations = relations_data.get("relations", [])
        logger.info(f"Loaded {len(relations)} relations from {relations_file}")

    # Load chunks for traceability
    chunks = []
    if chunks_file and chunks_file.exists():
        with open(chunks_file, 'r') as f:
            chunks_data = json.load(f)
        chunks = chunks_data.get("chunks", [])
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")

    # Initialize loader
    loader = Neo4jLoader(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        create_vector_index=create_vector_index,
        link_to_api_graph=link_to_api_graph,
    )

    try:
        # Setup schema
        loader.setup_schema()

        # Load concepts
        logger.info("Loading concepts...")
        loader.load_concepts(concepts)

        # Load relations
        if relations:
            logger.info("Loading relations...")
            loader.load_relations(relations)

        # Create document nodes and traceability
        if chunks:
            # Build document list from chunks
            docs = {}
            for chunk in chunks:
                path = chunk.get("source_path", "")
                if path and path not in docs:
                    docs[path] = {
                        "path": path,
                        "product": chunk.get("product", ""),
                        "content_hash": "",
                    }

            logger.info("Loading documents...")
            loader.load_documents(list(docs.values()))

            # GraphRAG: Load chunks with embeddings
            logger.info("Loading chunks with embeddings (GraphRAG)...")
            loader.load_chunks(chunks, compute_embeddings=True)

            logger.info("Creating traceability edges...")
            loader.create_mentioned_in_edges(concepts, chunks)

        # Link to API graph
        if link_to_api_graph:
            logger.info("Linking concepts to API graph...")
            loader.link_to_api_graph(concepts)
            loader.find_orphan_concepts()

            # Link documents to API graph
            if chunks:
                logger.info("Linking documents to API graph...")
                loader.link_documents_to_api_graph(list(docs.values()))

                logger.info("Linking chunks to API graph via concepts...")
                loader.link_chunks_to_api_graph(chunks, concepts)

        loader.metrics.load_time_seconds = time.time() - start_time

        logger.info(f"Loading complete in {loader.metrics.load_time_seconds:.2f}s")
        return loader.metrics

    finally:
        loader.close()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="Load concepts and relations into Neo4j")
    parser.add_argument("--concepts", "-c", required=True, help="Resolved concepts JSON file")
    parser.add_argument("--relations", "-r", help="Relations JSON file")
    parser.add_argument("--chunks", help="Chunks JSON file for traceability")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j user")
    parser.add_argument("--neo4j-password", default="password", help="Neo4j password")
    parser.add_argument("--no-vector-index", action="store_true", help="Skip vector index creation")
    parser.add_argument("--no-api-links", action="store_true", help="Skip API graph linking")

    args = parser.parse_args()

    relations_file = Path(args.relations) if args.relations else None
    chunks_file = Path(args.chunks) if args.chunks else None

    metrics = load_to_neo4j(
        concepts_file=Path(args.concepts),
        relations_file=relations_file,
        chunks_file=chunks_file,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        create_vector_index=not args.no_vector_index,
        link_to_api_graph=not args.no_api_links,
    )

    print("\nLoading Metrics:")
    print(json.dumps(metrics.to_dict(), indent=2))
