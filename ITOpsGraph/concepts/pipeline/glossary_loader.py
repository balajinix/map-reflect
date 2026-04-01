"""
Neo4j loader for glossary terms.

Loads curated glossary terms as high-quality Concept nodes with:
- Authoritative definitions from product documentation
- Product associations
- Cross-product term linking (same term in multiple products)
"""

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GlossaryLoaderMetrics:
    """Metrics from glossary loading."""
    terms_loaded: int = 0
    concepts_created: int = 0
    concepts_updated: int = 0
    product_links: int = 0
    type_links: int = 0
    field_links: int = 0
    cross_product_links: int = 0
    embeddings_stored: int = 0
    load_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "terms_loaded": self.terms_loaded,
            "concepts_created": self.concepts_created,
            "concepts_updated": self.concepts_updated,
            "product_links": self.product_links,
            "type_links": self.type_links,
            "field_links": self.field_links,
            "cross_product_links": self.cross_product_links,
            "embeddings_stored": self.embeddings_stored,
            "load_time_seconds": round(self.load_time_seconds, 2),
        }


class GlossaryLoader:
    """
    Loads glossary terms into Neo4j as Concept nodes.

    Creates high-quality Concept nodes from curated glossary definitions,
    with links to Products, Types, and Fields.
    """

    # Map glossary product names to Neo4j product names
    PRODUCT_MAP = {
        'concert': 'concert',
        'instana': 'instana',
        'turbonomic': 'turbonomic',
        'cloudability': 'cloudability',
        'ns1': 'ns1',
        'watsonx orchestrate': 'watsonx-orchestrate',
        'terraform': 'hcp_terraform',
        'packer': 'hcp_packer',
        'consul': 'hcp_terraform',  # Consul is part of HCP
        'boundary': 'hcp_terraform',  # Boundary is part of HCP
        'hashicorp': 'hcp_terraform',  # General HashiCorp terms
    }

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.embedding_model_name = embedding_model

        self._driver = None
        self._embedding_model = None
        self.metrics = GlossaryLoaderMetrics()

    def _get_driver(self):
        """Lazy load Neo4j driver."""
        if self._driver is None:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
        return self._driver

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed")
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
            logger.warning(f"Embedding failed: {e}")
            return None

    def close(self):
        """Close Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def load_terms(self, terms: list[dict], compute_embeddings: bool = True) -> int:
        """
        Load glossary terms as Concept nodes with product namespace.

        Terms are stored as "product.term" (e.g., "instana.alert") to:
        - Avoid collisions between products
        - Create clear product hierarchy
        - Allow same term with different meanings per product

        Args:
            terms: List of term dicts with keys: term, definition, product
            compute_embeddings: Whether to compute embeddings

        Returns:
            Number of terms loaded
        """
        driver = self._get_driver()
        loaded = 0

        for term_data in terms:
            term = term_data.get("term", "").strip()
            definition = term_data.get("definition", "").strip()
            product = term_data.get("product", "").lower()
            source_file = term_data.get("source_file", "")
            aliases = term_data.get("aliases", [])

            if not term or not definition:
                continue

            # Map product name
            neo4j_product = self.PRODUCT_MAP.get(product, product)

            # Create namespaced term: "instana.alert"
            namespaced_name = f"{neo4j_product}.{term}"

            # Compute embedding from term + definition
            embedding = None
            if compute_embeddings:
                embed_text = f"{term}: {definition}"
                embedding = self._compute_embedding(embed_text)
                if embedding:
                    self.metrics.embeddings_stored += 1

            props = {
                "name": namespaced_name,  # "instana.alert"
                "term": term,              # "alert" (original term)
                "canonical_name": term.lower().replace(" ", "_"),
                "description": definition,
                "entity_type": "GlossaryTerm",
                "hierarchy": "Concept",
                "product": neo4j_product,
                "source_file": source_file,
                "aliases": aliases,
                "confidence": 1.0,  # Glossary terms are authoritative
                "resolution_method": "glossary",
                "updated_at": datetime.now().isoformat(),
            }

            if embedding:
                props["embedding"] = embedding

            with driver.session() as session:
                # Create or update Concept node with namespaced name
                result = session.run("""
                    MERGE (c:Concept:GlossaryTerm {name: $name})
                    ON CREATE SET
                        c.term = $term,
                        c.canonical_name = $canonical_name,
                        c.description = $description,
                        c.entity_type = $entity_type,
                        c.hierarchy = $hierarchy,
                        c.product = $product,
                        c.source_file = $source_file,
                        c.aliases = $aliases,
                        c.confidence = $confidence,
                        c.resolution_method = $resolution_method,
                        c.created_at = datetime(),
                        c.updated_at = $updated_at,
                        c.embedding = $embedding
                    ON MATCH SET
                        c.term = $term,
                        c.description = CASE
                            WHEN c.description IS NULL OR size(c.description) < size($description)
                            THEN $description
                            ELSE c.description
                        END,
                        c.product = $product,
                        c.source_file = $source_file,
                        c.updated_at = $updated_at,
                        c.embedding = COALESCE($embedding, c.embedding)
                    RETURN c.created_at = c.updated_at AS is_new
                """, **props)

                record = result.single()
                if record and record["is_new"]:
                    self.metrics.concepts_created += 1
                else:
                    self.metrics.concepts_updated += 1

                loaded += 1
                self.metrics.terms_loaded += 1

        return loaded

    def link_to_products(self, terms: list[dict]) -> int:
        """Create HAS_GLOSSARY_TERM edges from Products to glossary Concepts."""
        driver = self._get_driver()
        links = 0

        for term_data in terms:
            term = term_data.get("term", "").strip()
            product = term_data.get("product", "").lower()

            if not term:
                continue

            neo4j_product = self.PRODUCT_MAP.get(product, product)
            namespaced_name = f"{neo4j_product}.{term}"

            with driver.session() as session:
                result = session.run("""
                    MATCH (p:Product {name: $product})
                    MATCH (c:GlossaryTerm {name: $namespaced_name})
                    MERGE (p)-[r:HAS_GLOSSARY_TERM]->(c)
                    ON CREATE SET
                        r.created_at = datetime()
                    RETURN r
                """, product=neo4j_product, namespaced_name=namespaced_name)

                if result.single():
                    links += 1
                    self.metrics.product_links += 1

        return links

    def link_to_types_and_fields(self, terms: list[dict]) -> dict:
        """
        Create DESCRIBES edges from Types/Fields to namespaced glossary terms.

        Matching rules:
        1. Same product only (instana.alert → Instana Types/Fields)
        2. Exact name match only (Type/Field name = term)

        No fuzzy matching - only link when Type/Field name exactly matches term.
        """
        driver = self._get_driver()
        links = {"type": 0, "field": 0}

        for term_data in terms:
            term = term_data.get("term", "").strip()
            product = term_data.get("product", "").lower()

            if not term or len(term) < 2:
                continue

            neo4j_product = self.PRODUCT_MAP.get(product, product)
            namespaced_name = f"{neo4j_product}.{term}"

            with driver.session() as session:
                # Link to Types: exact name match only
                result = session.run("""
                    MATCH (t:Type)
                    WHERE t.product = $product
                      AND toLower(t.name) = toLower($term)
                    MATCH (g:GlossaryTerm {name: $namespaced_name})
                    MERGE (t)-[r:HAS_GLOSSARY_TERM]->(g)
                    ON CREATE SET
                        r.match_type = 'exact_name',
                        r.created_at = datetime()
                    RETURN count(r) AS cnt
                """, product=neo4j_product, term=term, namespaced_name=namespaced_name)

                record = result.single()
                if record:
                    links["type"] += record["cnt"]
                    self.metrics.type_links += record["cnt"]

                # Link to Fields: exact name match only
                result = session.run("""
                    MATCH (f:Field)
                    WHERE f.product = $product
                      AND toLower(f.name) = toLower($term)
                    MATCH (g:GlossaryTerm {name: $namespaced_name})
                    MERGE (f)-[r:HAS_GLOSSARY_TERM]->(g)
                    ON CREATE SET
                        r.match_type = 'exact_name',
                        r.created_at = datetime()
                    RETURN count(r) AS cnt
                """, product=neo4j_product, term=term, namespaced_name=namespaced_name)

                record = result.single()
                if record:
                    links["field"] += record["cnt"]
                    self.metrics.field_links += record["cnt"]

        return links

    def link_cross_product_terms(self, terms: list[dict]) -> int:
        """
        Link same terms across different products with SAME_AS relationship.

        If "Application" exists in both Instana and Concert glossaries,
        create a SAME_AS edge between them.
        """
        driver = self._get_driver()
        links = 0

        # Group terms by normalized name
        term_groups = {}
        for t in terms:
            name = t.get("term", "").lower().strip()
            if name:
                if name not in term_groups:
                    term_groups[name] = []
                term_groups[name].append(t)

        # Create SAME_AS links for terms appearing in multiple products
        with driver.session() as session:
            for name, group in term_groups.items():
                if len(group) > 1:
                    # Get products for this term
                    products = list(set(t.get("product", "") for t in group))
                    if len(products) > 1:
                        # Create cross-product link using term property (not namespaced name)
                        result = session.run("""
                            MATCH (c:GlossaryTerm)
                            WHERE toLower(c.term) = $term
                            WITH c
                            MATCH (c2:GlossaryTerm)
                            WHERE toLower(c2.term) = $term AND c <> c2 AND c.product <> c2.product
                            MERGE (c)-[r:SAME_AS]->(c2)
                            ON CREATE SET r.created_at = datetime()
                            RETURN count(r) AS cnt
                        """, term=name)

                        record = result.single()
                        if record and record["cnt"] > 0:
                            links += record["cnt"]
                            self.metrics.cross_product_links += record["cnt"]

        return links


def load_glossary_to_neo4j(
    glossary_file: Path,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    compute_embeddings: bool = True,
    link_to_api: bool = True,
) -> GlossaryLoaderMetrics:
    """
    Load glossary terms from JSON file into Neo4j.

    Args:
        glossary_file: Path to glossary terms JSON file
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        compute_embeddings: Whether to compute embeddings
        link_to_api: Whether to link to API graph

    Returns:
        GlossaryLoaderMetrics with loading statistics
    """
    import time
    start_time = time.time()

    # Load terms
    with open(glossary_file, 'r') as f:
        data = json.load(f)

    terms = data.get("terms", [])
    logger.info(f"Loaded {len(terms)} glossary terms from {glossary_file}")

    # Initialize loader
    loader = GlossaryLoader(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )

    try:
        # Load terms as Concepts
        logger.info("Loading glossary terms as Concepts...")
        loader.load_terms(terms, compute_embeddings=compute_embeddings)

        # Link to Products
        logger.info("Linking to Products...")
        loader.link_to_products(terms)

        # Link to Types and Fields
        if link_to_api:
            logger.info("Linking to Types and Fields...")
            loader.link_to_types_and_fields(terms)

        # Link cross-product terms
        logger.info("Linking cross-product terms...")
        loader.link_cross_product_terms(terms)

        loader.metrics.load_time_seconds = time.time() - start_time
        return loader.metrics

    finally:
        loader.close()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="Load glossary terms into Neo4j")
    parser.add_argument("--glossary", "-g", required=True, help="Glossary terms JSON file")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j user")
    parser.add_argument("--neo4j-password", default="password", help="Neo4j password")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip embeddings")
    parser.add_argument("--no-api-links", action="store_true", help="Skip API graph linking")

    args = parser.parse_args()

    metrics = load_glossary_to_neo4j(
        glossary_file=Path(args.glossary),
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        compute_embeddings=not args.no_embeddings,
        link_to_api=not args.no_api_links,
    )

    print("\nGlossary Loading Summary:")
    print(f"  Terms loaded: {metrics.terms_loaded}")
    print(f"  Concepts created: {metrics.concepts_created}")
    print(f"  Concepts updated: {metrics.concepts_updated}")
    print(f"  Product links: {metrics.product_links}")
    print(f"  Type links: {metrics.type_links}")
    print(f"  Field links: {metrics.field_links}")
    print(f"  Cross-product links: {metrics.cross_product_links}")
    print(f"  Embeddings stored: {metrics.embeddings_stored}")
    print(f"  Time: {metrics.load_time_seconds:.2f}s")
