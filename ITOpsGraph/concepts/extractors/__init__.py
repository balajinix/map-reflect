"""
Entity extractors for the concept extraction pipeline.

Available extractors:
- GLiNER2Extractor: Primary extractor using GLiNER2 model
- SpaCyExtractor: Fallback using spaCy NER
- LLMExtractor: LLM-based entity extraction
"""

from .gliner2_extractor import (
    GLiNER2Extractor,
    extract_entities,
)

__all__ = [
    "GLiNER2Extractor",
    "extract_entities",
]
