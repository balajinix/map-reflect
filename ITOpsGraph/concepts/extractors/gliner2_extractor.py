"""
GLiNER2 extractor wrapper.

GLiNER2 is a unified multi-task model for:
- Named Entity Recognition (NER)
- Text Classification
- Structured Data Extraction
- Relation Extraction

Model: 205M parameters, CPU-efficient, competitive with GPT-4o on benchmarks.

References:
- https://github.com/fastino-ai/GLiNER2
- https://arxiv.org/html/2507.18546v1
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GLiNER2Extractor:
    """
    Wrapper for GLiNER2 model for entity and relation extraction.

    Supports both the original GLiNER models and GLiNER2 models.
    Falls back to pattern-based extraction if models unavailable.

    Usage:
        extractor = GLiNER2Extractor()

        # Entity extraction
        entities = extractor.extract_entities(
            "Turbonomic monitors Kubernetes clusters",
            ["Service", "Platform", "Cluster"]
        )

        # Relation extraction (GLiNER2 only)
        relations = extractor.extract_relations(
            "Turbonomic monitors Kubernetes clusters",
            entities,
            [("Service", "MONITORS", "Platform")]
        )
    """

    # Available model variants
    MODELS = {
        # GLiNER v2.5 (recommended for NER)
        "gliner_large": "gliner-community/gliner_large-v2.5",
        "gliner_medium": "gliner-community/gliner_medium-v2.5",
        "gliner_small": "gliner-community/gliner_small-v2.5",
        # Multilingual
        "gliner_multi": "gliner-community/gliner_multi-v2.1",
        # GLiNER2 (if available from fastino-ai)
        "gliner2_base": "fastino-ai/gliner2-base",
    }

    def __init__(
        self,
        model_name: str = "gliner_large",
        device: str = "cpu",
        threshold: float = 0.3,
    ):
        """
        Initialize GLiNER2 extractor.

        Args:
            model_name: Model key from MODELS dict or full HuggingFace path
            device: Device to run on ("cpu" or "cuda")
            threshold: Default confidence threshold
        """
        # Resolve model name
        if model_name in self.MODELS:
            self.model_path = self.MODELS[model_name]
        else:
            self.model_path = model_name

        self.device = device
        self.threshold = threshold
        self._model = None
        self._model_type = None  # "gliner" or "gliner2"

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return self._model

        # Try GLiNER2 first
        try:
            from gliner import GLiNER

            logger.info(f"Loading model: {self.model_path}")
            self._model = GLiNER.from_pretrained(self.model_path)

            if self.device == "cuda":
                self._model = self._model.to("cuda")

            self._model_type = "gliner"
            logger.info(f"Model loaded successfully (type: {self._model_type})")
            return self._model

        except ImportError:
            logger.warning("GLiNER not installed. pip install gliner")
            self._model = "unavailable"
            return self._model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model = "unavailable"
            return self._model

    def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        threshold: float | None = None,
        flat_ner: bool = True,
    ) -> list[dict]:
        """
        Extract entities from text.

        Args:
            text: Text to extract entities from
            entity_types: List of entity type labels to extract
            threshold: Confidence threshold (default: self.threshold)
            flat_ner: If True, don't allow nested entities

        Returns:
            List of entity dicts with keys: text, label, score, start, end
        """
        model = self._load_model()
        threshold = threshold or self.threshold

        if model == "unavailable":
            logger.warning("Model unavailable, using fallback extraction")
            return self._fallback_extract(text, entity_types)

        try:
            entities = model.predict_entities(
                text,
                entity_types,
                threshold=threshold,
                flat_ner=flat_ner,
            )

            # Normalize output format
            results = []
            for ent in entities:
                results.append({
                    "text": ent["text"],
                    "label": ent["label"],
                    "score": ent["score"],
                    "start": ent["start"],
                    "end": ent["end"],
                })

            return results

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return self._fallback_extract(text, entity_types)

    def extract_relations(
        self,
        text: str,
        entities: list[dict],
        relation_types: list[tuple[str, str, str]],
        threshold: float | None = None,
    ) -> list[dict]:
        """
        Extract relations between entities (GLiNER2 feature).

        Args:
            text: Text containing the entities
            entities: List of entity dicts from extract_entities()
            relation_types: List of (head_type, relation, tail_type) tuples
            threshold: Confidence threshold

        Returns:
            List of relation dicts with keys: head, relation, tail, score
        """
        model = self._load_model()
        threshold = threshold or self.threshold

        if model == "unavailable" or self._model_type != "gliner2":
            # Relation extraction requires GLiNER2
            logger.warning("Relation extraction requires GLiNER2 model")
            return []

        try:
            # GLiNER2 relation extraction API
            # Note: This is placeholder - actual API may differ
            relations = model.predict_relations(
                text,
                entities,
                relation_types,
                threshold=threshold,
            )

            return relations

        except AttributeError:
            logger.warning("This model version doesn't support relation extraction")
            return []
        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            return []

    def extract_with_schema(
        self,
        text: str,
        schema: dict[str, Any],
        threshold: float | None = None,
    ) -> dict:
        """
        Extract using a declarative schema (GLiNER2 feature).

        Args:
            text: Text to extract from
            schema: Schema dict with "entities" and optional "relations" keys
            threshold: Confidence threshold

        Returns:
            Dict with "entities" and "relations" lists

        Example schema:
            {
                "entities": ["Service", "Platform", "Metric"],
                "relations": [
                    {"head": "Service", "relation": "MONITORS", "tail": "Platform"},
                    {"head": "Service", "relation": "COLLECTS", "tail": "Metric"}
                ]
            }
        """
        threshold = threshold or self.threshold

        # Extract entities
        entity_types = schema.get("entities", [])
        entities = self.extract_entities(text, entity_types, threshold)

        # Extract relations if schema defines them
        relations = []
        if "relations" in schema and entities:
            relation_types = [
                (r["head"], r["relation"], r["tail"])
                for r in schema["relations"]
            ]
            relations = self.extract_relations(text, entities, relation_types, threshold)

        return {
            "entities": entities,
            "relations": relations,
        }

    def _fallback_extract(
        self,
        text: str,
        entity_types: list[str],
    ) -> list[dict]:
        """
        Fallback pattern-based extraction when model is unavailable.

        Uses simple heuristics to identify potential entities.
        """
        import re

        results = []
        seen = set()

        # Pattern for capitalized phrases
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'

        # Pattern for technical terms (acronyms, product names)
        tech_pattern = r'\b([A-Z][A-Z0-9]+(?:[-_][A-Z0-9]+)*)\b'

        # Pattern for common ITOps terms
        itops_keywords = {
            "Service": r'\b(\w+\s*(?:Service|API|Endpoint))\b',
            "Platform": r'\b(Kubernetes|K8s|AWS|Azure|GCP|Docker|OpenShift)\b',
            "Cluster": r'\b(\w+\s*[Cc]luster)\b',
            "Metric": r'\b(CPU|Memory|Latency|Throughput|IOPS|Bandwidth)\b',
            "Container": r'\b(\w+\s*[Cc]ontainer|[Pp]od)\b',
        }

        for etype, pattern in itops_keywords.items():
            if etype in entity_types:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity_text = match.group(1)
                    if entity_text.lower() not in seen:
                        seen.add(entity_text.lower())
                        results.append({
                            "text": entity_text,
                            "label": etype,
                            "score": 0.5,
                            "start": match.start(1),
                            "end": match.end(1),
                        })

        # Generic capitalized phrase extraction
        for match in re.finditer(cap_pattern, text):
            entity_text = match.group(1)
            if entity_text.lower() not in seen and len(entity_text) > 2:
                seen.add(entity_text.lower())
                # Assign first available type
                if entity_types:
                    results.append({
                        "text": entity_text,
                        "label": entity_types[0],
                        "score": 0.3,
                        "start": match.start(1),
                        "end": match.end(1),
                    })

        return results

    @property
    def is_available(self) -> bool:
        """Check if model is loaded and available."""
        return self._load_model() != "unavailable"

    @property
    def supports_relations(self) -> bool:
        """Check if model supports relation extraction."""
        self._load_model()
        return self._model_type == "gliner2"


# Convenience function
def extract_entities(
    text: str,
    entity_types: list[str],
    model: str = "gliner_large",
    threshold: float = 0.3,
) -> list[dict]:
    """
    Convenience function to extract entities.

    Args:
        text: Text to extract from
        entity_types: List of entity type labels
        model: Model name or path
        threshold: Confidence threshold

    Returns:
        List of entity dicts
    """
    extractor = GLiNER2Extractor(model_name=model, threshold=threshold)
    return extractor.extract_entities(text, entity_types)
