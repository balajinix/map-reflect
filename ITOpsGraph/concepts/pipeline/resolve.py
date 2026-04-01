"""
Entity resolution and deduplication module.

Deduplicates and canonicalizes entities using multiple strategies:
1. Exact matching (normalized case, whitespace, punctuation)
2. Domain synonym matching (curated ITOps dictionary)
3. Fuzzy matching (edit distance, Jaro-Winkler)
4. Embedding similarity (Sentence-BERT)
5. LLM verification (for borderline cases)
"""

import logging
import re
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class ResolvedEntity:
    """A resolved/canonical entity with all its mentions."""
    canonical: str
    canonical_normalized: str
    entity_type: str
    hierarchy: str
    aliases: list[str]
    mentions: int
    source_docs: list[str]
    source_products: list[str]
    source_chunks: list[str]  # GraphRAG: track source chunks for provenance
    confidence: float
    resolution_method: str
    original_entities: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "canonical": self.canonical,
            "canonical_normalized": self.canonical_normalized,
            "entity_type": self.entity_type,
            "hierarchy": self.hierarchy,
            "aliases": self.aliases,
            "mentions": self.mentions,
            "source_docs": self.source_docs,
            "source_products": self.source_products,
            "source_chunks": self.source_chunks,
            "confidence": self.confidence,
            "resolution_method": self.resolution_method,
        }


@dataclass
class ResolutionMetrics:
    """Metrics collected during entity resolution."""
    entities_input: int = 0
    entities_output: int = 0
    duplicates_found: int = 0
    entities_merged: int = 0
    merge_ratio: float = 0.0
    exact_matches: int = 0
    synonym_matches: int = 0
    fuzzy_matches: int = 0
    embedding_matches: int = 0
    llm_verified: int = 0
    llm_rejected: int = 0
    cross_product_merges: int = 0
    canonical_coverage: float = 0.0
    resolution_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "entities_input": self.entities_input,
            "entities_output": self.entities_output,
            "duplicates_found": self.duplicates_found,
            "entities_merged": self.entities_merged,
            "merge_ratio": round(self.merge_ratio, 3),
            "exact_matches": self.exact_matches,
            "synonym_matches": self.synonym_matches,
            "fuzzy_matches": self.fuzzy_matches,
            "embedding_matches": self.embedding_matches,
            "llm_verified": self.llm_verified,
            "llm_rejected": self.llm_rejected,
            "cross_product_merges": self.cross_product_merges,
            "canonical_coverage": round(self.canonical_coverage, 3),
            "resolution_time_seconds": round(self.resolution_time_seconds, 2),
        }


class SynonymDictionary:
    """
    Cross-product synonym dictionary for ITOps entities.

    Maps product-specific terminology to canonical forms.
    """

    def __init__(self, config_path: Path | None = None):
        """
        Initialize synonym dictionary.

        Args:
            config_path: Path to synonyms.yaml config file
        """
        self.synonyms = {}  # alias -> canonical
        self.canonical_to_aliases = {}  # canonical -> [aliases]

        if config_path and config_path.exists():
            self._load_config(config_path)
        else:
            self._load_defaults()

    def _load_config(self, config_path: Path):
        """Load synonyms from YAML config."""
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        for entry in config.get("cross_product_synonyms", {}).values():
            canonical = entry.get("canonical", "")
            aliases = entry.get("aliases", [])

            if canonical:
                canonical_lower = canonical.lower()
                self.canonical_to_aliases[canonical_lower] = [canonical] + aliases
                self.synonyms[canonical_lower] = canonical

                for alias in aliases:
                    self.synonyms[alias.lower()] = canonical

    def _load_defaults(self):
        """Load default ITOps synonyms."""
        defaults = {
            "virtual_machine": {
                "canonical": "Virtual Machine",
                "aliases": ["VM", "EC2 Instance", "Compute Instance", "Azure VM", "Virtual Server", "vServer"],
            },
            "container": {
                "canonical": "Container",
                "aliases": ["Docker Container", "Container Instance"],
            },
            "kubernetes": {
                "canonical": "Kubernetes",
                "aliases": ["K8s", "K8S", "Kube"],
            },
            "kubernetes_cluster": {
                "canonical": "Kubernetes Cluster",
                "aliases": ["K8s Cluster", "K8S Cluster", "Kube Cluster"],
            },
            "load_balancer": {
                "canonical": "Load Balancer",
                "aliases": ["ELB", "ALB", "NLB", "Azure Load Balancer", "Cloud Load Balancing"],
            },
            "database": {
                "canonical": "Database",
                "aliases": ["DB", "RDS", "Cloud SQL", "Azure SQL"],
            },
            "cpu": {
                "canonical": "CPU",
                "aliases": ["Processor", "vCPU", "CPU Usage", "CPU Utilization"],
            },
            "memory": {
                "canonical": "Memory",
                "aliases": ["RAM", "Memory Usage", "Memory Utilization"],
            },
            "storage": {
                "canonical": "Storage",
                "aliases": ["Disk", "Volume", "EBS", "Block Storage"],
            },
            "network": {
                "canonical": "Network",
                "aliases": ["VPC", "VNET", "Networking"],
            },
            "auto_scaling": {
                "canonical": "Auto Scaling",
                "aliases": ["Autoscaling", "Auto-scaling", "Auto Scale", "Autoscaler"],
            },
            "monitoring": {
                "canonical": "Monitoring",
                "aliases": ["Observability", "APM", "Application Monitoring"],
            },
            "alerting": {
                "canonical": "Alerting",
                "aliases": ["Alerts", "Alert Management", "Alarm"],
            },
            "dashboard": {
                "canonical": "Dashboard",
                "aliases": ["Dashboards", "Visualization", "Charts"],
            },
            "api": {
                "canonical": "API",
                "aliases": ["REST API", "API Endpoint", "Web API"],
            },
            "microservice": {
                "canonical": "Microservice",
                "aliases": ["Microservices", "Service", "Micro-service"],
            },
            "deployment": {
                "canonical": "Deployment",
                "aliases": ["Deploy", "Rollout", "Release"],
            },
            "configuration": {
                "canonical": "Configuration",
                "aliases": ["Config", "Settings", "Configs"],
            },
            "instance": {
                "canonical": "Instance",
                "aliases": ["EC2", "VM Instance", "Compute"],
            },
            "pod": {
                "canonical": "Pod",
                "aliases": ["Kubernetes Pod", "K8s Pod"],
            },
            "namespace": {
                "canonical": "Namespace",
                "aliases": ["K8s Namespace", "Kubernetes Namespace"],
            },
            "node": {
                "canonical": "Node",
                "aliases": ["Worker Node", "Kubernetes Node", "K8s Node"],
            },
            "cluster": {
                "canonical": "Cluster",
                "aliases": ["Compute Cluster", "Server Cluster"],
            },
            "service_mesh": {
                "canonical": "Service Mesh",
                "aliases": ["Istio", "Linkerd", "Envoy"],
            },
            "prometheus": {
                "canonical": "Prometheus",
                "aliases": ["Prom"],
            },
            "grafana": {
                "canonical": "Grafana",
                "aliases": [],
            },
        }

        for entry in defaults.values():
            canonical = entry["canonical"]
            aliases = entry["aliases"]

            canonical_lower = canonical.lower()
            self.canonical_to_aliases[canonical_lower] = [canonical] + aliases
            self.synonyms[canonical_lower] = canonical

            for alias in aliases:
                self.synonyms[alias.lower()] = canonical

    def get_canonical(self, text: str) -> str | None:
        """Get canonical form for a text if it's a known synonym."""
        return self.synonyms.get(text.lower())

    def get_aliases(self, canonical: str) -> list[str]:
        """Get all aliases for a canonical form."""
        return self.canonical_to_aliases.get(canonical.lower(), [])


class EntityResolver:
    """
    Resolves and deduplicates entities using multiple strategies.

    Resolution order:
    1. Exact matching (normalized)
    2. Synonym dictionary matching
    3. Fuzzy string matching
    4. Embedding similarity
    5. LLM verification

    Usage:
        resolver = EntityResolver()
        resolved = resolver.resolve(entities)
    """

    def __init__(
        self,
        synonyms_path: Path | None = None,
        fuzzy_threshold: float = 0.85,
        embedding_threshold: float = 0.80,
        use_embeddings: bool = True,
        use_llm: bool = False,
        llm_model: str = "gpt-4o-mini",
    ):
        """
        Initialize the entity resolver.

        Args:
            synonyms_path: Path to synonyms.yaml config
            fuzzy_threshold: Threshold for fuzzy matching (Jaro-Winkler)
            embedding_threshold: Threshold for embedding similarity
            use_embeddings: Whether to use embedding-based matching
            use_llm: Whether to use LLM for verification
            llm_model: OpenAI model for LLM verification
        """
        self.synonyms = SynonymDictionary(synonyms_path)
        self.fuzzy_threshold = fuzzy_threshold
        self.embedding_threshold = embedding_threshold
        self.use_embeddings = use_embeddings
        self.use_llm = use_llm
        self.llm_model = llm_model

        self._embedding_model = None
        self._openai_client = None

        self.metrics = ResolutionMetrics()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove punctuation except hyphens and underscores
        text = re.sub(r'[^\w\s\-_]', '', text)
        return text

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model...")
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded")
            except ImportError:
                logger.warning("sentence-transformers not installed. Embedding matching disabled.")
                self._embedding_model = "unavailable"
        return self._embedding_model

    def _get_openai_client(self):
        """Lazy load OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI()
            except ImportError:
                logger.warning("OpenAI not installed. LLM verification disabled.")
                self._openai_client = "unavailable"
            except Exception as e:
                logger.warning(f"OpenAI client init failed: {e}")
                self._openai_client = "unavailable"
        return self._openai_client

    def _compute_fuzzy_similarity(self, s1: str, s2: str) -> float:
        """Compute Jaro-Winkler similarity between two strings."""
        try:
            from rapidfuzz import fuzz
            return fuzz.WRatio(s1, s2) / 100.0
        except ImportError:
            # Fallback to simple implementation
            return self._jaro_winkler(s1, s2)

    def _jaro_winkler(self, s1: str, s2: str) -> float:
        """Simple Jaro-Winkler implementation."""
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        match_distance = max(len1, len2) // 2 - 1
        if match_distance < 0:
            match_distance = 0

        s1_matches = [False] * len1
        s2_matches = [False] * len2

        matches = 0
        transpositions = 0

        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)

            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3

        # Winkler modification
        prefix = 0
        for i in range(min(len1, len2, 4)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break

        return jaro + prefix * 0.1 * (1 - jaro)

    def _compute_embedding_similarity(self, texts: list[str]) -> dict[tuple[int, int], float]:
        """Compute pairwise embedding similarities."""
        model = self._get_embedding_model()

        if model == "unavailable":
            return {}

        try:
            import numpy as np

            embeddings = model.encode(texts)

            # Compute cosine similarities
            similarities = {}
            norms = np.linalg.norm(embeddings, axis=1)

            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    if norms[i] > 0 and norms[j] > 0:
                        sim = np.dot(embeddings[i], embeddings[j]) / (norms[i] * norms[j])
                        similarities[(i, j)] = float(sim)

            return similarities

        except Exception as e:
            logger.error(f"Embedding similarity computation failed: {e}")
            return {}

    def _verify_merge_llm(self, entity1: str, entity2: str, context: str = "") -> bool:
        """Use LLM to verify if two entities should be merged."""
        client = self._get_openai_client()

        if client == "unavailable":
            return True  # Default to merging if LLM unavailable

        try:
            prompt = f"""Should these two entity mentions be merged as referring to the same concept?

Entity 1: {entity1}
Entity 2: {entity2}
{f"Context: {context}" if context else ""}

Respond with only "yes" or "no"."""

            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an entity resolution expert for ITOps documentation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10,
            )

            answer = response.choices[0].message.content.strip().lower()
            return answer in ["yes", "y", "true"]

        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")
            return True

    def resolve(self, entities: list[dict]) -> list[ResolvedEntity]:
        """
        Resolve and deduplicate a list of entities.

        Args:
            entities: List of entity dicts with 'text', 'entity_type', 'hierarchy',
                     'confidence', 'source_path', 'product' keys

        Returns:
            List of ResolvedEntity objects
        """
        import time
        start_time = time.time()

        self.metrics.entities_input = len(entities)

        if not entities:
            return []

        # Group entities by normalized text
        normalized_groups = defaultdict(list)
        for entity in entities:
            normalized = self._normalize_text(entity.get("text", ""))
            if normalized:
                normalized_groups[normalized].append(entity)

        self.metrics.exact_matches = len(entities) - len(normalized_groups)

        # Build candidate clusters
        clusters = []
        processed = set()

        # First pass: exact + synonym matching
        for normalized, group in normalized_groups.items():
            if normalized in processed:
                continue

            cluster = list(group)
            processed.add(normalized)

            # Check for synonyms
            canonical = self.synonyms.get_canonical(normalized)
            if canonical:
                canonical_normalized = self._normalize_text(canonical)
                if canonical_normalized in normalized_groups and canonical_normalized not in processed:
                    cluster.extend(normalized_groups[canonical_normalized])
                    processed.add(canonical_normalized)
                    self.metrics.synonym_matches += len(normalized_groups[canonical_normalized])

                # Check other aliases
                for alias in self.synonyms.get_aliases(canonical):
                    alias_normalized = self._normalize_text(alias)
                    if alias_normalized in normalized_groups and alias_normalized not in processed:
                        cluster.extend(normalized_groups[alias_normalized])
                        processed.add(alias_normalized)
                        self.metrics.synonym_matches += len(normalized_groups[alias_normalized])

            clusters.append(cluster)

        # Second pass: fuzzy matching between cluster representatives
        if len(clusters) > 1:
            cluster_reps = [c[0].get("text", "") for c in clusters]
            merged_clusters = self._merge_by_fuzzy(clusters, cluster_reps)
            clusters = merged_clusters

        # Third pass: embedding similarity
        if self.use_embeddings and len(clusters) > 1:
            cluster_reps = [c[0].get("text", "") for c in clusters]
            merged_clusters = self._merge_by_embedding(clusters, cluster_reps)
            clusters = merged_clusters

        # Convert clusters to resolved entities
        resolved = []
        for cluster in clusters:
            resolved_entity = self._create_resolved_entity(cluster)
            resolved.append(resolved_entity)

            # Track cross-product merges
            products = set(e.get("product", "") for e in cluster)
            if len(products) > 1:
                self.metrics.cross_product_merges += 1

        # Update metrics
        self.metrics.entities_output = len(resolved)
        self.metrics.duplicates_found = len(clusters)
        self.metrics.entities_merged = self.metrics.entities_input - self.metrics.entities_output
        if self.metrics.entities_input > 0:
            self.metrics.merge_ratio = self.metrics.entities_merged / self.metrics.entities_input

        # Canonical coverage: entities with a known canonical form
        with_canonical = sum(1 for r in resolved if self.synonyms.get_canonical(r.canonical))
        if resolved:
            self.metrics.canonical_coverage = with_canonical / len(resolved)

        self.metrics.resolution_time_seconds = time.time() - start_time

        return resolved

    def _merge_by_fuzzy(
        self,
        clusters: list[list[dict]],
        representatives: list[str],
    ) -> list[list[dict]]:
        """Merge clusters by fuzzy string matching."""
        n = len(clusters)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                if find(i) != find(j):
                    sim = self._compute_fuzzy_similarity(
                        self._normalize_text(representatives[i]),
                        self._normalize_text(representatives[j])
                    )
                    if sim >= self.fuzzy_threshold:
                        # Optionally verify with LLM
                        should_merge = True
                        if self.use_llm:
                            should_merge = self._verify_merge_llm(
                                representatives[i], representatives[j]
                            )
                            if should_merge:
                                self.metrics.llm_verified += 1
                            else:
                                self.metrics.llm_rejected += 1

                        if should_merge:
                            union(i, j)
                            self.metrics.fuzzy_matches += 1

        # Group by parent
        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].extend(clusters[i])

        return list(groups.values())

    def _merge_by_embedding(
        self,
        clusters: list[list[dict]],
        representatives: list[str],
    ) -> list[list[dict]]:
        """Merge clusters by embedding similarity."""
        similarities = self._compute_embedding_similarity(representatives)

        if not similarities:
            return clusters

        n = len(clusters)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        # Merge similar clusters
        for (i, j), sim in similarities.items():
            if sim >= self.embedding_threshold and find(i) != find(j):
                # Optionally verify with LLM
                should_merge = True
                if self.use_llm:
                    should_merge = self._verify_merge_llm(
                        representatives[i], representatives[j]
                    )
                    if should_merge:
                        self.metrics.llm_verified += 1
                    else:
                        self.metrics.llm_rejected += 1

                if should_merge:
                    union(i, j)
                    self.metrics.embedding_matches += 1

        # Group by parent
        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].extend(clusters[i])

        return list(groups.values())

    def _create_resolved_entity(self, cluster: list[dict]) -> ResolvedEntity:
        """Create a ResolvedEntity from a cluster of entity mentions."""
        # Collect all unique text variants
        texts = list(set(e.get("text", "") for e in cluster))

        # Determine canonical form
        # Priority: synonym dictionary > most frequent > highest confidence
        canonical = None
        for text in texts:
            syn_canonical = self.synonyms.get_canonical(text)
            if syn_canonical:
                canonical = syn_canonical
                break

        if not canonical:
            # Use most frequent text
            text_counts = defaultdict(int)
            for e in cluster:
                text_counts[e.get("text", "")] += 1
            canonical = max(text_counts.keys(), key=lambda t: text_counts[t])

        # Collect metadata
        aliases = [t for t in texts if t != canonical]
        source_docs = list(set(e.get("source_path", "") for e in cluster if e.get("source_path")))
        source_products = list(set(e.get("product", "") for e in cluster if e.get("product")))
        source_chunks = list(set(e.get("chunk_id", "") for e in cluster if e.get("chunk_id")))

        # Determine entity type (most common)
        type_counts = defaultdict(int)
        for e in cluster:
            etype = e.get("entity_type", "")
            if etype:
                type_counts[etype] += 1
        entity_type = max(type_counts.keys(), key=lambda t: type_counts[t]) if type_counts else ""

        # Determine hierarchy (most common)
        hier_counts = defaultdict(int)
        for e in cluster:
            hier = e.get("hierarchy", "")
            if hier:
                hier_counts[hier] += 1
        hierarchy = max(hier_counts.keys(), key=lambda h: hier_counts[h]) if hier_counts else ""

        # Average confidence
        confidences = [e.get("confidence", 0) for e in cluster if e.get("confidence")]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Determine resolution method
        methods = []
        if len(texts) > 1:
            methods.append("exact")
        if self.synonyms.get_canonical(canonical):
            methods.append("synonym")
        resolution_method = "+".join(methods) if methods else "single"

        return ResolvedEntity(
            canonical=canonical,
            canonical_normalized=self._normalize_text(canonical),
            entity_type=entity_type,
            hierarchy=hierarchy,
            aliases=aliases,
            mentions=len(cluster),
            source_docs=source_docs,
            source_products=source_products,
            source_chunks=source_chunks,
            confidence=avg_confidence,
            resolution_method=resolution_method,
            original_entities=cluster,
        )


def resolve_entities_from_file(
    entities_file: Path,
    output_file: Path,
    synonyms_path: Path | None = None,
    fuzzy_threshold: float = 0.85,
    embedding_threshold: float = 0.80,
    use_embeddings: bool = True,
    use_llm: bool = False,
) -> ResolutionMetrics:
    """
    Convenience function to resolve entities from a JSON file.

    Args:
        entities_file: Path to entities JSON file (from NER)
        output_file: Path to output resolved entities JSON file
        synonyms_path: Path to synonyms.yaml config
        fuzzy_threshold: Threshold for fuzzy matching
        embedding_threshold: Threshold for embedding similarity
        use_embeddings: Whether to use embedding matching
        use_llm: Whether to use LLM verification

    Returns:
        ResolutionMetrics with resolution statistics
    """
    # Load entities
    with open(entities_file, 'r') as f:
        data = json.load(f)

    entities = data.get("entities", [])
    logger.info(f"Loaded {len(entities)} entities from {entities_file}")

    # Resolve
    resolver = EntityResolver(
        synonyms_path=synonyms_path,
        fuzzy_threshold=fuzzy_threshold,
        embedding_threshold=embedding_threshold,
        use_embeddings=use_embeddings,
        use_llm=use_llm,
    )

    resolved = resolver.resolve(entities)
    logger.info(f"Resolved to {len(resolved)} unique entities")

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    output_data = {
        "created_at": datetime.now().isoformat(),
        "source_entities": str(entities_file),
        "config": {
            "fuzzy_threshold": fuzzy_threshold,
            "embedding_threshold": embedding_threshold,
            "use_embeddings": use_embeddings,
            "use_llm": use_llm,
        },
        "metrics": resolver.metrics.to_dict(),
        "resolved_entities": [e.to_dict() for e in resolved],
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Resolved entities saved to {output_file}")

    return resolver.metrics


def load_synonyms(config_path: Path) -> SynonymDictionary:
    """Load synonym dictionary from config file."""
    return SynonymDictionary(config_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="Resolve and deduplicate entities")
    parser.add_argument("--input", "-i", required=True, help="Input entities JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output resolved entities JSON file")
    parser.add_argument("--synonyms", "-s", help="Synonyms YAML config file")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.85, help="Fuzzy matching threshold")
    parser.add_argument("--embedding-threshold", type=float, default=0.80, help="Embedding similarity threshold")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable embedding matching")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM verification")

    args = parser.parse_args()

    synonyms_path = Path(args.synonyms) if args.synonyms else None

    metrics = resolve_entities_from_file(
        entities_file=Path(args.input),
        output_file=Path(args.output),
        synonyms_path=synonyms_path,
        fuzzy_threshold=args.fuzzy_threshold,
        embedding_threshold=args.embedding_threshold,
        use_embeddings=not args.no_embeddings,
        use_llm=args.use_llm,
    )

    print("\nResolution Metrics:")
    print(json.dumps(metrics.to_dict(), indent=2))
