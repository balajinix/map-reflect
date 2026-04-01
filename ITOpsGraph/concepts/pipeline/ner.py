"""
Named Entity Recognition module using GLiNER2.

Extracts domain-specific entities from text chunks using a two-pass strategy:
1. Fine-grained extraction using GLiNER2
2. Hierarchy classification using LLM

GLiNER2 is a unified multi-task model (205M params) that handles NER,
text classification, and relation extraction in a single model.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# Default entity types for ITOps domain
DEFAULT_ENTITY_TYPES = {
    "fine_grained": {
        "itops_resources": [
            "Metric",
            "Alert",
            "Policy",
            "Action",
            "Workload",
            "Cluster",
            "Container",
            "Instance",
            "Endpoint",
            "Dashboard",
            "Namespace",
        ],
        "technologies": [
            "Service",
            "Tool",
            "Platform",
            "Protocol",
            "API",
            "Framework",
        ],
        "operational": [
            "Cost",
            "SLA",
            "Threshold",
            "Configuration",
            "Deployment",
            "Pipeline",
        ],
    },
    "hierarchy": [
        "Technology",
        "Feature",
        "Capability",
        "Resource",
        "Component",
        "Monitoring",
        "Optimization",
        "Integration",
    ],
}


@dataclass
class Entity:
    """An extracted entity with metadata."""
    text: str
    entity_type: str
    confidence: float
    start_char: int
    end_char: int
    chunk_id: str
    source_path: str
    product: str
    hierarchy: str = ""  # Assigned in pass 2
    validated: bool = False  # LLM validated

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "product": self.product,
            "hierarchy": self.hierarchy,
            "validated": self.validated,
        }


@dataclass
class NERMetrics:
    """Metrics collected during NER extraction."""
    chunks_processed: int = 0
    entities_total: int = 0
    entities_per_chunk: float = 0.0
    entity_type_dist: dict = field(default_factory=dict)
    hierarchy_dist: dict = field(default_factory=dict)
    confidence_avg: float = 0.0
    confidence_min: float = 1.0
    confidence_max: float = 0.0
    llm_additions: int = 0
    llm_removals: int = 0
    pass1_count: int = 0
    pass2_reclassified: int = 0
    extraction_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "chunks_processed": self.chunks_processed,
            "entities_total": self.entities_total,
            "entities_per_chunk": round(self.entities_per_chunk, 2),
            "entity_type_dist": self.entity_type_dist,
            "hierarchy_dist": self.hierarchy_dist,
            "confidence_avg": round(self.confidence_avg, 3),
            "confidence_min": round(self.confidence_min, 3),
            "confidence_max": round(self.confidence_max, 3),
            "llm_additions": self.llm_additions,
            "llm_removals": self.llm_removals,
            "pass1_count": self.pass1_count,
            "pass2_reclassified": self.pass2_reclassified,
            "extraction_time_seconds": round(self.extraction_time_seconds, 2),
        }


class GLiNER2Extractor:
    """
    Entity extractor using GLiNER2.

    GLiNER2 is a unified multi-task model for NER, text classification,
    and relation extraction with 205M parameters.

    Usage:
        extractor = GLiNER2Extractor()
        entities = extractor.extract(text, ["Service", "Platform", "Metric"])
    """

    def __init__(self, model_name: str = "gliner-community/gliner_large-v2.5"):
        """
        Initialize GLiNER2 extractor.

        Args:
            model_name: HuggingFace model name or path
        """
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the GLiNER model."""
        if self._model is None:
            try:
                # Try GLiNER2 first
                from gliner import GLiNER
                logger.info(f"Loading GLiNER model: {self.model_name}")
                self._model = GLiNER.from_pretrained(self.model_name)
                logger.info("GLiNER model loaded successfully")
            except ImportError:
                logger.warning("GLiNER not installed. Using fallback extractor.")
                self._model = "fallback"
            except Exception as e:
                logger.error(f"Failed to load GLiNER model: {e}")
                self._model = "fallback"
        return self._model

    def extract(
        self,
        text: str,
        entity_types: list[str],
        threshold: float = 0.3,
    ) -> list[tuple[str, str, float, int, int]]:
        """
        Extract entities from text.

        Args:
            text: Text to extract entities from
            entity_types: List of entity type labels
            threshold: Minimum confidence threshold

        Returns:
            List of (text, entity_type, confidence, start, end) tuples
        """
        model = self._load_model()

        if model == "fallback":
            return self._extract_fallback(text, entity_types)

        try:
            # GLiNER expects labels as a list
            entities = model.predict_entities(
                text,
                entity_types,
                threshold=threshold,
            )

            results = []
            for ent in entities:
                results.append((
                    ent["text"],
                    ent["label"],
                    ent["score"],
                    ent["start"],
                    ent["end"],
                ))
            return results

        except Exception as e:
            logger.error(f"GLiNER extraction failed: {e}")
            return self._extract_fallback(text, entity_types)

    # Common words to filter out in fallback extraction
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
        'we', 'us', 'our', 'you', 'your', 'he', 'she', 'him', 'her',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too',
        'very', 'just', 'also', 'now', 'here', 'there', 'when', 'where',
        'why', 'how', 'what', 'which', 'who', 'whom', 'whose',
        'if', 'then', 'else', 'because', 'while', 'although', 'though',
        'after', 'before', 'during', 'since', 'until', 'unless',
        'about', 'above', 'below', 'between', 'under', 'over', 'through',
        'into', 'onto', 'upon', 'out', 'up', 'down', 'off', 'away',
        'see', 'use', 'using', 'used', 'uses', 'get', 'set', 'new', 'old',
        'first', 'last', 'next', 'previous', 'following', 'section',
        'chapter', 'figure', 'table', 'example', 'note', 'important',
    }

    def _extract_fallback(
        self,
        text: str,
        entity_types: list[str],
    ) -> list[tuple[str, str, float, int, int]]:
        """
        Fallback extraction using pattern matching.

        This is a simple heuristic-based extractor for when GLiNER is unavailable.
        """
        results = []

        # Simple pattern-based extraction
        # Look for capitalized phrases that might be entities
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'

        # Also look for technical terms
        tech_pattern = r'\b([A-Z][A-Z0-9]+(?:[-_][A-Z0-9]+)*)\b'

        seen = set()

        for match in re.finditer(pattern, text):
            entity_text = match.group(1)
            # Filter stopwords and short entities
            if (entity_text.lower() not in self.STOPWORDS and
                entity_text not in seen and
                len(entity_text) > 2):
                seen.add(entity_text)
                # Guess entity type based on keywords
                entity_type = self._guess_entity_type(entity_text, entity_types)
                if entity_type:
                    results.append((
                        entity_text,
                        entity_type,
                        0.5,  # Low confidence for fallback
                        match.start(),
                        match.end(),
                    ))

        for match in re.finditer(tech_pattern, text):
            entity_text = match.group(1)
            if (entity_text.lower() not in self.STOPWORDS and
                entity_text not in seen and
                len(entity_text) > 1):
                seen.add(entity_text)
                entity_type = self._guess_entity_type(entity_text, entity_types)
                if entity_type:
                    results.append((
                        entity_text,
                        entity_type,
                        0.4,
                        match.start(),
                        match.end(),
                    ))

        return results

    def _guess_entity_type(self, text: str, entity_types: list[str]) -> str | None:
        """Guess entity type based on keywords."""
        text_lower = text.lower()

        type_keywords = {
            "Service": ["service", "api", "endpoint"],
            "Platform": ["platform", "cloud", "aws", "azure", "gcp", "kubernetes", "k8s"],
            "Tool": ["tool", "cli", "sdk"],
            "Metric": ["metric", "cpu", "memory", "latency", "throughput"],
            "Alert": ["alert", "alarm", "notification"],
            "Cluster": ["cluster", "node"],
            "Container": ["container", "pod", "docker"],
            "Instance": ["instance", "vm", "server"],
            "Dashboard": ["dashboard", "grafana", "kibana"],
            "Policy": ["policy", "rule"],
            "Configuration": ["config", "setting"],
            "Deployment": ["deployment", "release"],
        }

        for etype in entity_types:
            if etype in type_keywords:
                for keyword in type_keywords[etype]:
                    if keyword in text_lower:
                        return etype

        # Default to first available type if it looks like an entity
        if entity_types and len(text) > 2:
            return entity_types[0] if re.match(r'^[A-Z]', text) else None

        return None


class EntityExtractor:
    """
    Main entity extraction class implementing two-pass strategy.

    Pass 1: Extract fine-grained entities using GLiNER2
    Pass 2: Classify entities into hierarchy using LLM

    Usage:
        extractor = EntityExtractor()
        for entity in extractor.extract_from_chunks(chunks):
            print(entity.text, entity.entity_type, entity.hierarchy)
    """

    def __init__(
        self,
        entity_types: dict | None = None,
        gliner_model: str = "gliner-community/gliner_large-v2.5",
        confidence_threshold: float = 0.3,
        use_llm: bool = True,
        llm_model: str = "gpt-4o-mini",
    ):
        """
        Initialize the entity extractor.

        Args:
            entity_types: Dictionary with 'fine_grained' and 'hierarchy' keys
            gliner_model: GLiNER model name
            confidence_threshold: Minimum confidence for extraction
            use_llm: Whether to use LLM for pass 2
            llm_model: OpenAI model for LLM augmentation
        """
        self.entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self.confidence_threshold = confidence_threshold
        self.use_llm = use_llm
        self.llm_model = llm_model

        # Flatten fine-grained types
        self.fine_grained_types = []
        for category_types in self.entity_types.get("fine_grained", {}).values():
            self.fine_grained_types.extend(category_types)

        self.hierarchy_types = self.entity_types.get("hierarchy", [])

        # Initialize extractors
        self.gliner = GLiNER2Extractor(model_name=gliner_model)
        self._openai_client = None

        self.metrics = NERMetrics()

    def _get_openai_client(self):
        """Lazy initialize OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI()
            except ImportError:
                logger.warning("OpenAI not installed. LLM augmentation disabled.")
                self._openai_client = "unavailable"
            except Exception as e:
                logger.warning(f"OpenAI client init failed: {e}")
                self._openai_client = "unavailable"
        return self._openai_client

    def extract_entities(
        self,
        text: str,
        chunk_id: str,
        source_path: str,
        product: str,
    ) -> list[Entity]:
        """
        Extract entities from a single text using two-pass strategy.

        Args:
            text: Text to extract entities from
            chunk_id: Chunk identifier
            source_path: Source file path
            product: Product name

        Returns:
            List of Entity objects
        """
        import time
        start_time = time.time()

        # Pass 1: GLiNER2 extraction
        raw_entities = self.gliner.extract(
            text,
            self.fine_grained_types,
            threshold=self.confidence_threshold,
        )

        entities = []
        for ent_text, ent_type, confidence, start, end in raw_entities:
            entity = Entity(
                text=ent_text,
                entity_type=ent_type,
                confidence=confidence,
                start_char=start,
                end_char=end,
                chunk_id=chunk_id,
                source_path=source_path,
                product=product,
            )
            entities.append(entity)

        self.metrics.pass1_count += len(entities)

        # Pass 2: Hierarchy classification
        if entities:
            if self.use_llm:
                entities = self._classify_hierarchy(entities, text)
            else:
                # Use rule-based fallback when LLM is disabled
                entities = self._classify_hierarchy_rules(entities)

        # Update metrics
        elapsed = time.time() - start_time
        self.metrics.extraction_time_seconds += elapsed

        for entity in entities:
            # Update type distribution
            etype = entity.entity_type
            self.metrics.entity_type_dist[etype] = self.metrics.entity_type_dist.get(etype, 0) + 1

            # Update hierarchy distribution
            if entity.hierarchy:
                self.metrics.hierarchy_dist[entity.hierarchy] = \
                    self.metrics.hierarchy_dist.get(entity.hierarchy, 0) + 1

            # Update confidence stats
            self.metrics.confidence_min = min(self.metrics.confidence_min, entity.confidence)
            self.metrics.confidence_max = max(self.metrics.confidence_max, entity.confidence)

        return entities

    def _classify_hierarchy(
        self,
        entities: list[Entity],
        context: str,
    ) -> list[Entity]:
        """
        Classify entities into hierarchy categories using LLM.

        Args:
            entities: List of entities to classify
            context: Original text context

        Returns:
            Entities with hierarchy field populated
        """
        client = self._get_openai_client()

        if client == "unavailable":
            # Fallback: use simple rules
            return self._classify_hierarchy_rules(entities)

        try:
            # Batch entities for efficiency
            entity_list = [f"- {e.text} ({e.entity_type})" for e in entities]
            entities_str = "\n".join(entity_list)

            hierarchy_str = ", ".join(self.hierarchy_types)

            prompt = f"""Classify each entity into one of these hierarchy categories: {hierarchy_str}

Entities to classify:
{entities_str}

Context:
{context[:1000]}

Respond with a JSON array where each object has "text" and "hierarchy" fields.
Only include the JSON array, no other text."""

            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an entity classifier for ITOps documentation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000,
            )

            # Parse response
            import json
            response_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            classifications = json.loads(response_text)

            # Map classifications back to entities
            class_map = {c["text"]: c["hierarchy"] for c in classifications}

            for entity in entities:
                if entity.text in class_map:
                    entity.hierarchy = class_map[entity.text]
                    entity.validated = True
                    self.metrics.pass2_reclassified += 1

            return entities

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}. Using rule-based fallback.")
            return self._classify_hierarchy_rules(entities)

    def _classify_hierarchy_rules(self, entities: list[Entity]) -> list[Entity]:
        """Rule-based hierarchy classification fallback."""
        type_to_hierarchy = {
            # ITOps resources -> Resource
            "Metric": "Monitoring",
            "Alert": "Monitoring",
            "Dashboard": "Monitoring",
            "Policy": "Resource",
            "Action": "Capability",
            "Workload": "Resource",
            "Cluster": "Resource",
            "Container": "Resource",
            "Instance": "Resource",
            "Endpoint": "Resource",
            "Namespace": "Resource",
            # Technologies
            "Service": "Technology",
            "Tool": "Technology",
            "Platform": "Technology",
            "Protocol": "Technology",
            "API": "Integration",
            "Framework": "Technology",
            # Operational
            "Cost": "Optimization",
            "SLA": "Monitoring",
            "Threshold": "Monitoring",
            "Configuration": "Resource",
            "Deployment": "Capability",
            "Pipeline": "Integration",
        }

        for entity in entities:
            entity.hierarchy = type_to_hierarchy.get(entity.entity_type, "Component")

        return entities

    def extract_from_chunks(
        self,
        chunks: Iterator,  # Iterator of Chunk objects
        batch_size: int = 10,
    ) -> Iterator[Entity]:
        """
        Extract entities from multiple chunks.

        Args:
            chunks: Iterator of Chunk objects (from chunker module)
            batch_size: Number of chunks to process before updating metrics

        Yields:
            Entity objects
        """
        all_confidences = []
        chunk_count = 0
        entity_count = 0

        for chunk in chunks:
            chunk_count += 1
            self.metrics.chunks_processed += 1

            # Extract entities from chunk
            entities = self.extract_entities(
                text=chunk.text,
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                product=chunk.product,
            )

            for entity in entities:
                entity_count += 1
                all_confidences.append(entity.confidence)
                yield entity

            if chunk_count % batch_size == 0:
                logger.info(f"Processed {chunk_count} chunks, extracted {entity_count} entities")

        # Final metrics
        self.metrics.entities_total = entity_count
        if chunk_count > 0:
            self.metrics.entities_per_chunk = entity_count / chunk_count
        if all_confidences:
            self.metrics.confidence_avg = sum(all_confidences) / len(all_confidences)

    def deduplicate_entities(
        self,
        entities: list[Entity],
        similarity_threshold: float = 0.9,
    ) -> list[Entity]:
        """
        Deduplicate entities within a chunk based on text similarity.

        Args:
            entities: List of entities
            similarity_threshold: Threshold for considering entities as duplicates

        Returns:
            Deduplicated list of entities
        """
        if not entities:
            return entities

        # Group by normalized text
        seen = {}
        result = []

        for entity in entities:
            # Normalize text for comparison
            normalized = entity.text.lower().strip()
            normalized = re.sub(r'\s+', ' ', normalized)

            if normalized in seen:
                # Keep the one with higher confidence
                existing = seen[normalized]
                if entity.confidence > existing.confidence:
                    result.remove(existing)
                    result.append(entity)
                    seen[normalized] = entity
            else:
                seen[normalized] = entity
                result.append(entity)

        return result


def load_entity_types(config_path: Path) -> dict:
    """Load entity types from YAML config file."""
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def extract_entities_from_chunks_file(
    chunks_file: Path,
    output_file: Path,
    entity_types: dict | None = None,
    use_llm: bool = True,
    confidence_threshold: float = 0.3,
) -> NERMetrics:
    """
    Convenience function to extract entities from a chunks JSON file.

    Args:
        chunks_file: Path to chunks JSON file (from chunker)
        output_file: Path to output entities JSON file
        entity_types: Entity type configuration
        use_llm: Whether to use LLM for hierarchy classification
        confidence_threshold: Minimum confidence threshold

    Returns:
        NERMetrics with extraction statistics
    """
    import json
    from dataclasses import dataclass

    # Load chunks
    with open(chunks_file, 'r') as f:
        data = json.load(f)

    chunks_data = data.get("chunks", [])

    # Create simple chunk objects
    @dataclass
    class SimpleChunk:
        chunk_id: str
        text: str
        source_path: str
        product: str

    def chunk_generator():
        for c in chunks_data:
            yield SimpleChunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                source_path=c["source_path"],
                product=c["product"],
            )

    # Extract entities
    extractor = EntityExtractor(
        entity_types=entity_types,
        use_llm=use_llm,
        confidence_threshold=confidence_threshold,
    )

    entities = []
    for entity in extractor.extract_from_chunks(chunk_generator()):
        entities.append(entity)

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    output_data = {
        "created_at": datetime.now().isoformat(),
        "source_chunks": str(chunks_file),
        "metrics": extractor.metrics.to_dict(),
        "entities": [e.to_dict() for e in entities],
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Extracted {len(entities)} entities to {output_file}")

    return extractor.metrics


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="Extract entities from text chunks")
    parser.add_argument("--input", "-i", required=True, help="Input chunks JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output entities JSON file")
    parser.add_argument("--config", "-c", help="Entity types YAML config file")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM augmentation")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold")

    args = parser.parse_args()

    entity_types = None
    if args.config:
        entity_types = load_entity_types(Path(args.config))

    metrics = extract_entities_from_chunks_file(
        chunks_file=Path(args.input),
        output_file=Path(args.output),
        entity_types=entity_types,
        use_llm=not args.no_llm,
        confidence_threshold=args.threshold,
    )

    print("\nNER Metrics:")
    print(json.dumps(metrics.to_dict(), indent=2))
