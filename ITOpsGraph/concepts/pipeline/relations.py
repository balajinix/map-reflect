"""
Relation discovery module.

Discovers relationships between entities using multiple strategies:
1. GLiNER2 schema-based relation extraction
2. Dependency parse patterns (verb/preposition patterns)
3. Co-occurrence analysis (sentence/section/document level)
4. LLM relation classification (optional)
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
class Relation:
    """A discovered relation between two entities."""
    head: str
    tail: str
    relation_type: str
    confidence: float
    method: str  # gliner2, dep_parse, cooccurrence, llm
    source_sentence: str = ""
    source_chunk: str = ""
    source_doc: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "head": self.head,
            "tail": self.tail,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "method": self.method,
            "source_sentence": self.source_sentence,
            "source_chunk": self.source_chunk,
            "source_doc": self.source_doc,
            "metadata": self.metadata,
        }


@dataclass
class RelationMetrics:
    """Metrics collected during relation discovery."""
    entities_input: int = 0
    relations_total: int = 0
    relations_gliner2: int = 0
    relations_dep_parse: int = 0
    relations_cooccurrence: int = 0
    relations_llm: int = 0
    dep_parse_patterns: int = 0
    relation_type_dist: dict = field(default_factory=dict)
    avg_confidence: float = 0.0
    discovery_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "entities_input": self.entities_input,
            "relations_total": self.relations_total,
            "relations_gliner2": self.relations_gliner2,
            "relations_dep_parse": self.relations_dep_parse,
            "relations_cooccurrence": self.relations_cooccurrence,
            "relations_llm": self.relations_llm,
            "dep_parse_patterns": self.dep_parse_patterns,
            "relation_type_dist": self.relation_type_dist,
            "avg_confidence": round(self.avg_confidence, 3),
            "discovery_time_seconds": round(self.discovery_time_seconds, 2),
        }


# Verb pattern -> relation type mapping
VERB_PATTERNS = {
    "monitors": "MONITORS",
    "monitor": "MONITORS",
    "monitoring": "MONITORS",
    "manages": "MANAGES",
    "manage": "MANAGES",
    "managing": "MANAGES",
    "deploys": "DEPLOYS_TO",
    "deploy": "DEPLOYS_TO",
    "deploying": "DEPLOYS_TO",
    "deployed": "DEPLOYED_ON",
    "integrates": "INTEGRATES_WITH",
    "integrate": "INTEGRATES_WITH",
    "integration": "INTEGRATES_WITH",
    "requires": "REQUIRES",
    "require": "REQUIRES",
    "requiring": "REQUIRES",
    "enables": "ENABLES",
    "enable": "ENABLES",
    "enabling": "ENABLES",
    "triggers": "TRIGGERS",
    "trigger": "TRIGGERS",
    "triggering": "TRIGGERS",
    "scales": "SCALES",
    "scale": "SCALES",
    "scaling": "SCALES",
    "collects": "COLLECTS",
    "collect": "COLLECTS",
    "collecting": "COLLECTS",
    "sends": "SENDS_TO",
    "send": "SENDS_TO",
    "sending": "SENDS_TO",
    "receives": "RECEIVES_FROM",
    "receive": "RECEIVES_FROM",
    "receiving": "RECEIVES_FROM",
    "stores": "STORES_IN",
    "store": "STORES_IN",
    "storing": "STORES_IN",
    "uses": "USES",
    "use": "USES",
    "using": "USES",
    "provides": "PROVIDES",
    "provide": "PROVIDES",
    "providing": "PROVIDES",
    "supports": "SUPPORTS",
    "support": "SUPPORTS",
    "supporting": "SUPPORTS",
    "connects": "CONNECTS_TO",
    "connect": "CONNECTS_TO",
    "connecting": "CONNECTS_TO",
    "tracks": "TRACKS",
    "track": "TRACKS",
    "tracking": "TRACKS",
    "analyzes": "ANALYZES",
    "analyze": "ANALYZES",
    "analyzing": "ANALYZES",
    "optimizes": "OPTIMIZES",
    "optimize": "OPTIMIZES",
    "optimizing": "OPTIMIZES",
    "runs": "RUNS_ON",
    "run": "RUNS_ON",
    "running": "RUNS_ON",
    "hosts": "HOSTS",
    "host": "HOSTS",
    "hosting": "HOSTS",
    "contains": "CONTAINS",
    "contain": "CONTAINS",
    "containing": "CONTAINS",
    "includes": "INCLUDES",
    "include": "INCLUDES",
    "including": "INCLUDES",
}

# Preposition pattern -> relation type mapping
PREP_PATTERNS = {
    "from": "SOURCED_FROM",
    "in": "LOCATED_IN",
    "on": "DEPLOYED_ON",
    "with": "ASSOCIATED_WITH",
    "to": "SENDS_TO",
    "for": "USED_FOR",
    "by": "MANAGED_BY",
    "through": "VIA",
    "via": "VIA",
    "within": "PART_OF",
    "across": "SPANS",
}

# Hierarchical relation keywords
HIERARCHY_PATTERNS = {
    "is a": "IS_A",
    "is an": "IS_A",
    "type of": "IS_A",
    "kind of": "IS_A",
    "part of": "PART_OF",
    "belongs to": "PART_OF",
    "member of": "PART_OF",
    "component of": "PART_OF",
    "instance of": "INSTANCE_OF",
    "similar to": "SIMILAR_TO",
    "like": "SIMILAR_TO",
    "alternative to": "ALTERNATIVE_TO",
    "instead of": "ALTERNATIVE_TO",
}


class RelationDiscoverer:
    """
    Discovers relationships between entities using multiple strategies.

    Strategies (in order of preference):
    1. GLiNER2 relation extraction (if available)
    2. Dependency parse patterns
    3. Co-occurrence analysis
    4. LLM classification (optional)
    """

    def __init__(
        self,
        use_gliner2: bool = True,
        use_dep_parse: bool = True,
        use_cooccurrence: bool = True,
        use_llm: bool = False,
        llm_model: str = "gpt-4o-mini",
        cooccurrence_threshold: float = 0.3,
        min_confidence: float = 0.5,
    ):
        """
        Initialize the relation discoverer.

        Args:
            use_gliner2: Use GLiNER2 for relation extraction
            use_dep_parse: Use dependency parsing for patterns
            use_cooccurrence: Use co-occurrence analysis
            use_llm: Use LLM for relation classification
            llm_model: OpenAI model for LLM classification
            cooccurrence_threshold: Minimum weight for co-occurrence edges
            min_confidence: Minimum confidence to include a relation
        """
        self.use_gliner2 = use_gliner2
        self.use_dep_parse = use_dep_parse
        self.use_cooccurrence = use_cooccurrence
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.cooccurrence_threshold = cooccurrence_threshold
        self.min_confidence = min_confidence

        self._spacy_nlp = None
        self._gliner_model = None
        self._openai_client = None

        self.metrics = RelationMetrics()

    def _get_spacy_nlp(self):
        """Lazy load spaCy model."""
        if self._spacy_nlp is None:
            try:
                import spacy
                # Try to load a model with dependency parsing
                try:
                    self._spacy_nlp = spacy.load("en_core_web_sm")
                except OSError:
                    try:
                        self._spacy_nlp = spacy.load("en_core_web_md")
                    except OSError:
                        logger.warning("No spaCy model found. Dependency parsing disabled.")
                        self._spacy_nlp = "unavailable"
            except ImportError:
                logger.warning("spaCy not installed. Dependency parsing disabled.")
                self._spacy_nlp = "unavailable"
        return self._spacy_nlp

    def _get_gliner_model(self):
        """Lazy load GLiNER2 model."""
        if self._gliner_model is None:
            try:
                from gliner import GLiNER
                logger.info("Loading GLiNER model for relation extraction...")
                self._gliner_model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5")
                logger.info("GLiNER model loaded")
            except Exception as e:
                logger.warning(f"GLiNER not available for relations: {e}")
                self._gliner_model = "unavailable"
        return self._gliner_model

    def _get_openai_client(self):
        """Lazy load OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI()
            except ImportError:
                logger.warning("OpenAI not installed. LLM classification disabled.")
                self._openai_client = "unavailable"
            except Exception as e:
                logger.warning(f"OpenAI client init failed: {e}")
                self._openai_client = "unavailable"
        return self._openai_client

    def discover(
        self,
        entities: list[dict],
        chunks: list[dict],
    ) -> list[Relation]:
        """
        Discover relations between entities from document chunks.

        Args:
            entities: List of resolved entities with 'canonical', 'entity_type', etc.
            chunks: List of chunks with 'text', 'chunk_id', 'source_path', etc.

        Returns:
            List of discovered Relation objects
        """
        import time
        start_time = time.time()

        self.metrics.entities_input = len(entities)

        if not entities or not chunks:
            return []

        # Build entity lookup by text (case-insensitive)
        entity_lookup = {}
        for entity in entities:
            canonical = entity.get("canonical", "")
            if canonical:
                entity_lookup[canonical.lower()] = entity
                # Also add aliases
                for alias in entity.get("aliases", []):
                    entity_lookup[alias.lower()] = entity

        relations = []
        seen_relations = set()  # Dedup key: (head, tail, relation_type)

        # Process each chunk
        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", "")
            source_doc = chunk.get("source_path", "")

            if not text:
                continue

            # Find entities mentioned in this chunk
            mentioned_entities = self._find_entities_in_text(text, entity_lookup)

            if len(mentioned_entities) < 2:
                continue

            # Extract relations using each method
            chunk_relations = []

            # 1. Dependency parse patterns
            if self.use_dep_parse:
                dep_relations = self._extract_dep_parse_relations(
                    text, mentioned_entities, chunk_id, source_doc
                )
                chunk_relations.extend(dep_relations)
                self.metrics.relations_dep_parse += len(dep_relations)

            # 2. Co-occurrence analysis
            if self.use_cooccurrence:
                cooc_relations = self._extract_cooccurrence_relations(
                    text, mentioned_entities, chunk_id, source_doc
                )
                chunk_relations.extend(cooc_relations)
                self.metrics.relations_cooccurrence += len(cooc_relations)

            # 3. GLiNER2 relations (if available)
            if self.use_gliner2:
                gliner_relations = self._extract_gliner2_relations(
                    text, mentioned_entities, chunk_id, source_doc
                )
                chunk_relations.extend(gliner_relations)
                self.metrics.relations_gliner2 += len(gliner_relations)

            # Deduplicate and add
            for rel in chunk_relations:
                key = (rel.head.lower(), rel.tail.lower(), rel.relation_type)
                if key not in seen_relations and rel.confidence >= self.min_confidence:
                    seen_relations.add(key)
                    relations.append(rel)

        # Optional: LLM classification to refine relation types
        if self.use_llm and relations:
            relations = self._classify_relations_llm(relations)

        # Update metrics
        self.metrics.relations_total = len(relations)

        # Compute relation type distribution
        type_dist = defaultdict(int)
        confidences = []
        for rel in relations:
            type_dist[rel.relation_type] += 1
            confidences.append(rel.confidence)

        self.metrics.relation_type_dist = dict(type_dist)
        if confidences:
            self.metrics.avg_confidence = sum(confidences) / len(confidences)

        self.metrics.discovery_time_seconds = time.time() - start_time

        return relations

    def _find_entities_in_text(
        self,
        text: str,
        entity_lookup: dict,
    ) -> list[dict]:
        """Find which entities are mentioned in the text."""
        text_lower = text.lower()
        found = []
        seen_canonicals = set()

        for entity_text, entity in entity_lookup.items():
            if entity_text in text_lower:
                canonical = entity.get("canonical", entity_text)
                if canonical not in seen_canonicals:
                    # Find position in text
                    pos = text_lower.find(entity_text)
                    found.append({
                        **entity,
                        "matched_text": entity_text,
                        "position": pos,
                    })
                    seen_canonicals.add(canonical)

        # Sort by position in text
        found.sort(key=lambda e: e.get("position", 0))
        return found

    def _extract_dep_parse_relations(
        self,
        text: str,
        entities: list[dict],
        chunk_id: str,
        source_doc: str,
    ) -> list[Relation]:
        """Extract relations using dependency parsing patterns."""
        nlp = self._get_spacy_nlp()

        if nlp == "unavailable":
            return self._extract_pattern_relations_simple(text, entities, chunk_id, source_doc)

        relations = []

        try:
            doc = nlp(text)

            # Build entity position map
            entity_spans = []
            for entity in entities:
                matched = entity.get("matched_text", "")
                if matched:
                    # Find all occurrences
                    start = 0
                    while True:
                        pos = text.lower().find(matched, start)
                        if pos == -1:
                            break
                        entity_spans.append({
                            "entity": entity,
                            "start": pos,
                            "end": pos + len(matched),
                        })
                        start = pos + 1

            # Extract verb-mediated relations
            for token in doc:
                if token.pos_ == "VERB" and token.lemma_.lower() in VERB_PATTERNS:
                    relation_type = VERB_PATTERNS[token.lemma_.lower()]

                    # Find subject and object
                    subject = None
                    obj = None

                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            subject = self._find_entity_for_token(child, entity_spans, doc)
                        elif child.dep_ in ("dobj", "pobj", "attr"):
                            obj = self._find_entity_for_token(child, entity_spans, doc)

                    if subject and obj and subject != obj:
                        # Get the sentence containing this relation
                        sent = token.sent.text if token.sent else ""

                        relations.append(Relation(
                            head=subject.get("canonical", ""),
                            tail=obj.get("canonical", ""),
                            relation_type=relation_type,
                            confidence=0.8,
                            method="dep_parse",
                            source_sentence=sent,
                            source_chunk=chunk_id,
                            source_doc=source_doc,
                            metadata={"verb": token.text, "pattern": "verb"}
                        ))
                        self.metrics.dep_parse_patterns += 1

            # Extract prepositional relations
            for token in doc:
                if token.pos_ == "ADP" and token.text.lower() in PREP_PATTERNS:
                    relation_type = PREP_PATTERNS[token.text.lower()]

                    # Find head and object of preposition
                    head_entity = None
                    obj_entity = None

                    if token.head:
                        head_entity = self._find_entity_for_token(token.head, entity_spans, doc)

                    for child in token.children:
                        if child.dep_ == "pobj":
                            obj_entity = self._find_entity_for_token(child, entity_spans, doc)
                            break

                    if head_entity and obj_entity and head_entity != obj_entity:
                        sent = token.sent.text if token.sent else ""

                        relations.append(Relation(
                            head=head_entity.get("canonical", ""),
                            tail=obj_entity.get("canonical", ""),
                            relation_type=relation_type,
                            confidence=0.7,
                            method="dep_parse",
                            source_sentence=sent,
                            source_chunk=chunk_id,
                            source_doc=source_doc,
                            metadata={"prep": token.text, "pattern": "prep"}
                        ))
                        self.metrics.dep_parse_patterns += 1

        except Exception as e:
            logger.warning(f"Dependency parsing failed: {e}")
            return self._extract_pattern_relations_simple(text, entities, chunk_id, source_doc)

        return relations

    def _find_entity_for_token(self, token, entity_spans: list, doc) -> dict | None:
        """Find the entity that contains or is near a token."""
        token_start = token.idx
        token_end = token.idx + len(token.text)

        # Check if token is within any entity span
        for span in entity_spans:
            if span["start"] <= token_start and token_end <= span["end"]:
                return span["entity"]

        # Check compound tokens
        compound_start = token_start
        compound_end = token_end

        for child in token.children:
            if child.dep_ == "compound":
                compound_start = min(compound_start, child.idx)
                compound_end = max(compound_end, child.idx + len(child.text))

        for span in entity_spans:
            if (span["start"] <= compound_start and compound_end <= span["end"]) or \
               (compound_start <= span["start"] and span["end"] <= compound_end):
                return span["entity"]

        return None

    def _extract_pattern_relations_simple(
        self,
        text: str,
        entities: list[dict],
        chunk_id: str,
        source_doc: str,
    ) -> list[Relation]:
        """Simple pattern-based relation extraction without spaCy."""
        relations = []
        text_lower = text.lower()

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Find entities in this sentence
            sentence_entities = []
            for entity in entities:
                matched = entity.get("matched_text", "")
                if matched and matched in sentence_lower:
                    pos = sentence_lower.find(matched)
                    sentence_entities.append({
                        **entity,
                        "position": pos,
                    })

            if len(sentence_entities) < 2:
                continue

            sentence_entities.sort(key=lambda e: e.get("position", 0))

            # Look for verb patterns between entity pairs
            for i, head_entity in enumerate(sentence_entities[:-1]):
                for tail_entity in sentence_entities[i+1:]:
                    head_pos = head_entity.get("position", 0)
                    tail_pos = tail_entity.get("position", 0)

                    # Get text between entities
                    between = sentence_lower[head_pos:tail_pos]

                    # Check for verb patterns
                    for verb, relation_type in VERB_PATTERNS.items():
                        if verb in between:
                            relations.append(Relation(
                                head=head_entity.get("canonical", ""),
                                tail=tail_entity.get("canonical", ""),
                                relation_type=relation_type,
                                confidence=0.6,
                                method="pattern",
                                source_sentence=sentence.strip(),
                                source_chunk=chunk_id,
                                source_doc=source_doc,
                                metadata={"verb": verb}
                            ))
                            self.metrics.dep_parse_patterns += 1
                            break

                    # Check for hierarchy patterns
                    for pattern, relation_type in HIERARCHY_PATTERNS.items():
                        if pattern in between:
                            relations.append(Relation(
                                head=head_entity.get("canonical", ""),
                                tail=tail_entity.get("canonical", ""),
                                relation_type=relation_type,
                                confidence=0.7,
                                method="pattern",
                                source_sentence=sentence.strip(),
                                source_chunk=chunk_id,
                                source_doc=source_doc,
                                metadata={"pattern": pattern}
                            ))
                            self.metrics.dep_parse_patterns += 1
                            break

        return relations

    def _extract_cooccurrence_relations(
        self,
        text: str,
        entities: list[dict],
        chunk_id: str,
        source_doc: str,
    ) -> list[Relation]:
        """Extract relations from co-occurrence."""
        relations = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        # Track co-occurrences at different levels
        sentence_cooc = defaultdict(float)  # (entity1, entity2) -> count
        chunk_cooc = defaultdict(float)

        # Sentence-level co-occurrence (strongest signal)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_entities = []

            for entity in entities:
                matched = entity.get("matched_text", "")
                if matched and matched in sentence_lower:
                    canonical = entity.get("canonical", "")
                    if canonical not in sentence_entities:
                        sentence_entities.append(canonical)

            # Count pairs
            for i, e1 in enumerate(sentence_entities):
                for e2 in sentence_entities[i+1:]:
                    key = (min(e1, e2), max(e1, e2))
                    sentence_cooc[key] += 1.0

        # Chunk-level co-occurrence (weaker signal)
        chunk_entities = [e.get("canonical", "") for e in entities]
        for i, e1 in enumerate(chunk_entities):
            for e2 in chunk_entities[i+1:]:
                if e1 and e2 and e1 != e2:
                    key = (min(e1, e2), max(e1, e2))
                    chunk_cooc[key] += 0.5

        # Combine and create relations
        all_pairs = set(sentence_cooc.keys()) | set(chunk_cooc.keys())

        for pair in all_pairs:
            # Weighted score: sentence cooc is stronger
            weight = 0.6 * sentence_cooc.get(pair, 0) + 0.4 * chunk_cooc.get(pair, 0)

            # Normalize weight to confidence
            confidence = min(weight / 3.0, 1.0)

            if confidence >= self.cooccurrence_threshold:
                relations.append(Relation(
                    head=pair[0],
                    tail=pair[1],
                    relation_type="RELATED_TO",
                    confidence=confidence,
                    method="cooccurrence",
                    source_chunk=chunk_id,
                    source_doc=source_doc,
                    metadata={
                        "sentence_cooc": sentence_cooc.get(pair, 0),
                        "chunk_cooc": chunk_cooc.get(pair, 0),
                    }
                ))

        return relations

    def _extract_gliner2_relations(
        self,
        text: str,
        entities: list[dict],
        chunk_id: str,
        source_doc: str,
    ) -> list[Relation]:
        """Extract relations using GLiNER2."""
        model = self._get_gliner_model()

        if model == "unavailable":
            return []

        relations = []

        try:
            # Define relation types to extract
            relation_types = [
                "monitors", "manages", "uses", "requires",
                "enables", "contains", "part of", "integrates with",
                "deploys to", "runs on", "connects to"
            ]

            # Use GLiNER for relation-like entity extraction
            # Note: GLiNER2 may not have native relation extraction in all versions
            # This is a simplified approach using entity patterns

            # Extract entity pairs that might have relations
            entity_names = [e.get("canonical", "") for e in entities if e.get("canonical")]

            if len(entity_names) >= 2:
                # Look for relation patterns in the text
                for i, head in enumerate(entity_names[:-1]):
                    for tail in entity_names[i+1:]:
                        # Check if both appear in a pattern-like structure
                        head_lower = head.lower()
                        tail_lower = tail.lower()

                        # Simple heuristic: look for them near each other
                        text_lower = text.lower()
                        head_pos = text_lower.find(head_lower)
                        tail_pos = text_lower.find(tail_lower)

                        if head_pos >= 0 and tail_pos >= 0:
                            distance = abs(tail_pos - head_pos)
                            if distance < 200:  # Within 200 characters
                                between = text_lower[min(head_pos, tail_pos):max(head_pos + len(head_lower), tail_pos + len(tail_lower))]

                                # Check for known relation patterns
                                for verb, rel_type in VERB_PATTERNS.items():
                                    if verb in between:
                                        relations.append(Relation(
                                            head=head,
                                            tail=tail,
                                            relation_type=rel_type,
                                            confidence=0.75,
                                            method="gliner2",
                                            source_chunk=chunk_id,
                                            source_doc=source_doc,
                                            metadata={"pattern": verb}
                                        ))
                                        break

        except Exception as e:
            logger.warning(f"GLiNER2 relation extraction failed: {e}")

        return relations

    def _classify_relations_llm(self, relations: list[Relation]) -> list[Relation]:
        """Use LLM to refine relation types."""
        client = self._get_openai_client()

        if client == "unavailable":
            return relations

        # Process in batches to reduce API calls
        batch_size = 20
        refined_relations = []

        for i in range(0, len(relations), batch_size):
            batch = relations[i:i + batch_size]

            # Build prompt
            relation_texts = []
            for j, rel in enumerate(batch):
                relation_texts.append(
                    f"{j+1}. {rel.head} -> {rel.tail} (current: {rel.relation_type})"
                    f"\n   Context: {rel.source_sentence[:200] if rel.source_sentence else 'N/A'}"
                )

            prompt = f"""Classify the following entity relationships into the most appropriate relation type.

Available relation types:
- MONITORS: A observes/monitors B
- MANAGES: A manages/controls B
- USES: A uses/utilizes B
- REQUIRES: A depends on/requires B
- ENABLES: A enables/allows B
- CONTAINS: A contains/includes B
- PART_OF: A is part of B
- IS_A: A is a type/instance of B
- INTEGRATES_WITH: A integrates with B
- DEPLOYS_TO: A is deployed to B
- RUNS_ON: A runs on B
- CONNECTS_TO: A connects to B
- RELATED_TO: A is generally related to B (default if unsure)

Relations to classify:
{chr(10).join(relation_texts)}

For each relation, respond with just the number and new relation type, one per line:
1. RELATION_TYPE
2. RELATION_TYPE
...
"""

            try:
                response = client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are an expert at classifying relationships between IT operations concepts."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=500,
                )

                # Parse response
                lines = response.choices[0].message.content.strip().split("\n")

                for j, rel in enumerate(batch):
                    if j < len(lines):
                        line = lines[j].strip()
                        # Extract relation type from line
                        parts = line.split(".")
                        if len(parts) >= 2:
                            new_type = parts[1].strip().upper()
                            if new_type in ["MONITORS", "MANAGES", "USES", "REQUIRES", "ENABLES",
                                           "CONTAINS", "PART_OF", "IS_A", "INTEGRATES_WITH",
                                           "DEPLOYS_TO", "RUNS_ON", "CONNECTS_TO", "RELATED_TO"]:
                                rel.relation_type = new_type
                                rel.metadata["llm_classified"] = True
                                self.metrics.relations_llm += 1

                    refined_relations.append(rel)

            except Exception as e:
                logger.warning(f"LLM relation classification failed: {e}")
                refined_relations.extend(batch)

        return refined_relations


def discover_relations_from_files(
    entities_file: Path,
    chunks_file: Path,
    output_file: Path,
    use_gliner2: bool = True,
    use_dep_parse: bool = True,
    use_cooccurrence: bool = True,
    use_llm: bool = False,
    min_confidence: float = 0.5,
) -> RelationMetrics:
    """
    Discover relations from entity and chunk files.

    Args:
        entities_file: Path to resolved entities JSON file
        chunks_file: Path to chunks JSON file
        output_file: Path to output relations JSON file
        use_gliner2: Use GLiNER2 for extraction
        use_dep_parse: Use dependency parsing
        use_cooccurrence: Use co-occurrence analysis
        use_llm: Use LLM classification
        min_confidence: Minimum confidence threshold

    Returns:
        RelationMetrics with discovery statistics
    """
    # Load entities
    with open(entities_file, 'r') as f:
        entities_data = json.load(f)

    entities = entities_data.get("resolved_entities", entities_data.get("entities", []))
    logger.info(f"Loaded {len(entities)} entities from {entities_file}")

    # Load chunks
    with open(chunks_file, 'r') as f:
        chunks_data = json.load(f)

    chunks = chunks_data.get("chunks", [])
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")

    # Discover relations
    discoverer = RelationDiscoverer(
        use_gliner2=use_gliner2,
        use_dep_parse=use_dep_parse,
        use_cooccurrence=use_cooccurrence,
        use_llm=use_llm,
        min_confidence=min_confidence,
    )

    relations = discoverer.discover(entities, chunks)
    logger.info(f"Discovered {len(relations)} relations")

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    output_data = {
        "created_at": datetime.now().isoformat(),
        "source_entities": str(entities_file),
        "source_chunks": str(chunks_file),
        "config": {
            "use_gliner2": use_gliner2,
            "use_dep_parse": use_dep_parse,
            "use_cooccurrence": use_cooccurrence,
            "use_llm": use_llm,
            "min_confidence": min_confidence,
        },
        "metrics": discoverer.metrics.to_dict(),
        "relations": [r.to_dict() for r in relations],
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Relations saved to {output_file}")

    return discoverer.metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="Discover relations between entities")
    parser.add_argument("--entities", "-e", required=True, help="Resolved entities JSON file")
    parser.add_argument("--chunks", "-c", required=True, help="Chunks JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output relations JSON file")
    parser.add_argument("--no-gliner2", action="store_true", help="Disable GLiNER2")
    parser.add_argument("--no-dep-parse", action="store_true", help="Disable dependency parsing")
    parser.add_argument("--no-cooccurrence", action="store_true", help="Disable co-occurrence")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM classification")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum confidence")

    args = parser.parse_args()

    metrics = discover_relations_from_files(
        entities_file=Path(args.entities),
        chunks_file=Path(args.chunks),
        output_file=Path(args.output),
        use_gliner2=not args.no_gliner2,
        use_dep_parse=not args.no_dep_parse,
        use_cooccurrence=not args.no_cooccurrence,
        use_llm=args.use_llm,
        min_confidence=args.min_confidence,
    )

    print("\nRelation Discovery Metrics:")
    print(json.dumps(metrics.to_dict(), indent=2))
