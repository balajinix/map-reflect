#!/usr/bin/env python3
"""
Unified Map-Reflect Runner

Consolidated implementation merging:
- batch_runner.py: Batch inference with 10-11x speedup
- ue_reflect.py: Uncertainty quantification and self-reflection
- map_reflect_runner.py: Original wrapper

Features:
- Batch inference by default (configurable batch size)
- Optional uncertainty quantification (--use-uncertainty)
- Optional self-reflection with adaptive depth (--use-reflection)
- Neighborhood expansion via property graph (--neighbor-k)
- Checkpoint/resume capability (--checkpoint-interval, --resume)
- Response caching for efficiency

Usage:
    # Default (fast, batch mode)
    python -m map_reflect.map_reflect_runner \
        --datasets ITOpsGraph \
        --models endpoint_b/meta-llama/llama-3-3-70b-instruct \
        --property-lookup-file data/itopsgraph_api_lookup.json \
        --data-dir data

    # Full features
    python -m map_reflect.map_reflect_runner \
        --datasets ITOpsGraph \
        --models endpoint_b/meta-llama/llama-3-3-70b-instruct \
        --property-lookup-file data/itopsgraph_api_lookup.json \
        --data-dir data \
        --use-uncertainty \
        --use-reflection \
        --neighbor-k 5
"""

import os
import sys

# SQLite workaround for older systems (must be before chromadb import)
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import json
import random
import time
import re
import argparse
import hashlib
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import networkx as nx
from tqdm import tqdm
from dotenv import load_dotenv

# Suppress verbose logging before imports
os.environ['LITELLM_LOG'] = 'ERROR'

script_dir = Path(__file__).parent
load_dotenv(script_dir / ".env")

try:  # per-call latency log, off unless MAPREFLECT_LATENCY_LOG is set (map_reflect/latency_log.py)
    from map_reflect import latency_log as LL
except ImportError:  # run as a script from inside map_reflect/
    import latency_log as LL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Per-call usage logging (protocol v2). When TOKEN_LOG names a jsonl file, every LLM call through
# EndpointBBatchClient, EndpointABatchClient or LiteLLMBatchClient appends one record: model, stage, prompt and
# completion tokens, seconds. The stage is set per thread with set_usage_stage() (falling back to the
# TOKEN_STAGE environment variable), so one process can tag ranking, reflection and verification calls.
# Without TOKEN_LOG nothing is written and behaviour is unchanged.
# -----------------------------------------------------------------------------
_usage_local = threading.local()
_usage_lock = threading.Lock()


def set_usage_stage(stage: str) -> None:
    _usage_local.stage = stage


def get_usage_stage() -> str:
    return getattr(_usage_local, "stage", None) or os.environ.get("TOKEN_STAGE", "")


def set_usage_context(ctx: Optional[dict]) -> None:
    """Extra fields for this thread's usage records (e.g. the field key and prompt md5, so actual prompt tokens join
    to each call). Unset (the default) adds nothing, and the records are as before."""
    _usage_local.context = dict(ctx) if ctx else None


def get_usage_context() -> Optional[dict]:
    return getattr(_usage_local, "context", None)


def log_usage(model: str, prompt_tokens: int, completion_tokens: int, seconds: float,
              stage: Optional[str] = None, **extra) -> None:
    path = os.environ.get("TOKEN_LOG")
    ctx = get_usage_context() or {}
    if not path and "_sink" not in ctx:
        return
    rec = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), model=model,
               stage=stage if stage is not None else get_usage_stage(),
               prompt_tokens=int(prompt_tokens or 0), completion_tokens=int(completion_tokens or 0),
               seconds=round(float(seconds), 3), run=os.environ.get("TOKEN_RUN", ""), **extra)
    for k, v in ctx.items():
        if not k.startswith("_"):
            rec.setdefault(k, v)
    if "_sink" in ctx:  # the caller collects this call's usage (a list shared with the context it set)
        ctx["_sink"].append(dict(rec))
    if not path:
        return
    with _usage_lock:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


_LLAMA_TMPL = re.compile(r"<\|start_header_id\|>(\w+)<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", re.S)
_GRANITE_TMPL = re.compile(r"<\|start_of_role\|>(\w+)<\|end_of_role\|>(.*?)<\|end_of_text\|>", re.S)


def chat_messages(prompt: str) -> List[Dict[str, str]]:
    """Messages of a Llama 3 or Granite 4 templated prompt (the trailing assistant header is dropped).

    format_prompt_for_model renders a model's chat template client side, which suits endpoint_b's text endpoint.
    A chat endpoint (ENDPOINT_A) applies the template itself, so with ENDPOINT_A_SPLIT_TEMPLATE=1 the ENDPOINT_A client sends
    these messages and the template is applied once. A prompt without either template is one user message.
    """
    for rx in (_LLAMA_TMPL, _GRANITE_TMPL):
        msgs = [{"role": r, "content": c} for r, c in rx.findall(prompt)]
        if msgs:
            return msgs
    return [{"role": "user", "content": prompt}]


def _field_name(x: str) -> str:
    x = str(x)
    for sep in ("__", "."):
        if sep in x:
            x = x.rsplit(sep, 1)[1]
    return x.lower()


def resolve_ranked_id(raw: Any, shown: List[str]) -> Optional[str]:
    """Protocol v2: the candidate a ranked id names, among the candidates the model was shown.

    Exact match, then a case-insensitive match, then a unique match on the field name (the part after the last
    '.' or '__'). An id that resolves to none of the shown candidates (or to several) names no candidate: None.
    """
    s = normalize_field_id(str(raw)).strip()
    if not s or not shown:
        return None
    if s in shown:
        return s
    low = [c for c in shown if c.lower() == s.lower()]
    if len(low) == 1:
        return low[0]
    by_name = [c for c in shown if _field_name(c) == _field_name(s)]
    return by_name[0] if len(by_name) == 1 else None


NONE_ID = "NONE"


class Ranking(list):
    """A parsed ranking; with the v5 NONE option it also carries NONE's score and rank (None when not ranked)."""
    none_score = None
    none_rank = None
    # ok (a ranking), none_only (only NONE ranked), empty (a JSON array with no usable entry), unparsed (no array)
    status = "ok"


def shown_ids(candidates: List[Dict], limit: Optional[int] = 20) -> List[str]:
    """The candidate ids a ranking prompt shows (_build_ranking_prompt lists the first 20, or all with full_list)."""
    return [str(c.get("id", c.get("name", ""))) for c in (candidates or [])[:limit]]


def resolve_ranking(ranking: List[Dict], shown: List[str]) -> List[Dict]:
    """A ranking with every id resolved against the shown candidates; unresolved and repeated ids are dropped."""
    out, seen = [], set()
    for item in ranking or []:
        if not isinstance(item, dict) or "id" not in item:
            continue
        rid = resolve_ranked_id(item["id"], shown)
        if rid is None or rid in seen:
            continue
        seen.add(rid)
        out.append({**item, "id": rid})
    return out


def ranking_margin(ranking: List[Dict]) -> Tuple[Optional[str], float, bool]:
    """Protocol v2 margin: (top-1 id, margin, valid).

    The candidates' scores are sorted in descending order (a stable sort, so listed order breaks ties) and the
    margin is s1 - s2. One candidate: s2 = 0. Equal top scores: 0. A ranking with no usable candidate or no
    numeric score is invalid: top-1 None, margin 0. The margin is never negative.
    """
    cands = []
    for i, c in enumerate(ranking or []):
        if not isinstance(c, dict) or not c.get("id"):
            continue
        try:
            sc = float(c.get("score"))
        except (TypeError, ValueError):
            continue
        if sc != sc:  # NaN
            continue
        cands.append((sc, i, str(c["id"])))
    if not cands:
        return None, 0.0, False
    cands.sort(key=lambda t: (-t[0], t[1]))
    s1 = cands[0][0]
    s2 = cands[1][0] if len(cands) > 1 else 0.0
    return cands[0][2], max(0.0, s1 - s2), True

# Try to import EndpointB SDK
try:
    from endpoint_b_sdk import Credentials
    from endpoint_b_sdk.metanames import GenTextParamsMetaNames as GenParams
    from endpoint_b_sdk.foundation_models.utils.enums import DecodingMethods
    from endpoint_b_sdk.foundation_models import ModelInference
    HAS_ENDPOINT_B = True
except ImportError:
    HAS_ENDPOINT_B = False
    logger.warning("endpoint_b_sdk not installed, batch mode unavailable")

# Try to import ChromaDB for retrieval
try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    logger.warning("chromadb not installed, retrieval-based candidate selection unavailable")

# Try to import BM25 for hybrid retrieval
try:
    from rank_bm25 import BM25Okapi
    import pickle
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("rank_bm25 not installed. Install with: pip install rank-bm25")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MapReflectConfig:
    """Configuration for the unified Map-Reflect pipeline."""
    # Batch processing
    batch_size: int = 50
    max_workers: int = 4

    # Candidate retrieval
    top_k: int = 10
    neighbor_k: int = 5  # 1-hop neighbors (0 to disable)
    # Retrieval mode: the target is never shown to the model. Candidates come from hybrid
    # search over the whole catalogue (the lookup), prompts name no target product, and OMOP
    # target-table filtering is off. The record's target is still written out for scoring.
    retrieval_mode: bool = False
    # Where checkpoints are written and resumed from. The CLI sets it to --output-dir. It used to be
    # the fixed map_reflect/results/, shared by every run of a (dataset, model), so --resume could
    # load another experiment's predictions (found 2026-10-01 when a retrieval rerun resumed an
    # earlier target-given run).
    checkpoint_dir: Optional[str] = None
    # ToolBench retrieval only: also drop every API of the source tool, not just the source API.
    # ToolBench never maps an API to another API of the same tool (0 of 3,190 pairs), yet those
    # APIs dominate the nearest neighbours and were picked for about 80% of fields (issue #7).
    exclude_source_tool: bool = False
    # Protocol v2 control: give the first pass the schema evidence reflection arm B adds (data types, sibling
    # fields of the source and of each candidate, a same-table marker), with no reflection.
    evidence_first: bool = False
    # Method v3 control: in the given-target setting the first pass sees every field of the named target type
    # (no truncation to top_k * 3 candidates, no limit of 20 in the prompt).
    full_list: bool = False
    # Method v4 controls: candidate_lists maps "<sample_id>||<field>" to the exact candidate ids the first-pass prompt
    # shows, in that order and all of them (the hybrid top-20 shortlist, or a widened pool for a fresh mapping call);
    # a field with no entry is skipped. only_keys restricts the run to the listed fields (the declined-field re-ask).
    candidate_lists: Optional[Dict[str, List[str]]] = None
    only_keys: Optional[set] = None
    # Method v5: the ranking prompt also lists NONE ("none of these candidates matches"); the model scores it like a
    # candidate, its score and rank are logged as a coverage signal, and it never becomes the answer.
    none_option: bool = False

    # OMOP-specific enhancements (auto-enabled for OMOP_* datasets)
    omop_table_filtering: bool = True  # Filter candidates to utterance-specified table
    omop_type_hints: bool = True       # Add field type hints to prompts

    # Reflection signals (independent flags for clean ablation)
    use_self_reflection: bool = False      # Always reflect (no external signal)
    use_error_reflection: bool = False     # Reflect on detected errors
    use_uq_reflection: bool = False        # Reflect on high uncertainty
    use_neighborhood_reflection: bool = False  # Reflect using neighborhood validation

    # Legacy flags (for backward compatibility)
    use_uncertainty: bool = False  # Enable UQ calculation
    use_reflection: bool = False   # Legacy: enables error + uq reflection
    n_uncertainty_samples: int = 3

    # UQ method: "sampling" (consistency), "heuristic" (response-based), "perplexity" (logprobs)
    uq_method: str = "sampling"

    # UQ threshold configuration
    uq_threshold: Optional[float] = None  # Fixed threshold (overrides adaptive)
    uq_threshold_percentile: Optional[float] = None  # Percentile-based threshold (e.g., 50 = median)

    # Checkpointing
    checkpoint_interval: int = 50
    resume: bool = False

    # Caching
    use_cache: bool = True
    cache_dir: Optional[str] = None

    # Reproducibility / variance experiments
    # seed feeds Python/numpy RNG and the endpoint_b RANDOM_SEED param. NOTE: it only
    # affects generation when decoding is SAMPLE (see _get_model); under GREEDY the
    # seed is inert. sampling enables the seed to actually move outputs so we can
    # measure run-to-run variance across seeds.
    seed: int = 42
    sample_main_decoding: bool = False  # if True, main MAP/field calls use SAMPLE+RANDOM_SEED

    # LLM settings
    max_tokens: int = 4000
    temperature: float = 0.1
    uq_sampling_temperature: float = 0.7  # temp for UQ resampling (sampling/lmp_sampling/simba)

    # Field-level prediction (default on for SchemaNet-style datasets)
    predict_fields: bool = True  # Enable field-level predictions after type prediction

    # Constrained assignment: enforce one-to-one field mapping (Phase 3.5)
    # Default OFF: many-to-one is legitimate in schema mapping
    use_constrained_assignment: bool = False

    # Documentation context: inject product doc chunks into MAP prompts
    doc_context_chromadb: Optional[str] = None  # Path to ChromaDB with doc chunks
    doc_context_collection: str = "itopsgraph_docs"  # Collection name
    doc_context_top_k: int = 3  # Number of doc chunks to retrieve per field


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# =============================================================================
# Shared Utilities
# =============================================================================

def format_prompt_for_model(messages: List[Dict], model_id: str) -> Tuple[str, List[str]]:
    """Format messages based on model-specific chat template."""
    model_lower = model_id.lower()

    # Granite 4 models
    if "granite-4" in model_lower or "granite4" in model_lower:
        prompt_parts = []
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')
            prompt_parts.append(f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n")
        prompt_parts.append("<|start_of_role|>assistant<|end_of_role|>")
        return "".join(prompt_parts), ["<|end_of_text|>"]

    # Llama 3 models
    elif "llama" in model_lower:
        prompt_parts = ["<|begin_of_text|>"]
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')
            prompt_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(prompt_parts), ["<|eot_id|>", "<|end_of_text|>"]

    # Granite 3.x models
    elif "granite-3" in model_lower or "granite3" in model_lower:
        prompt_parts = []
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')
            prompt_parts.append(f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n")
        prompt_parts.append("<|start_of_role|>assistant<|end_of_role|>")
        return "".join(prompt_parts), ["<|end_of_text|>"]

    # Default fallback
    else:
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(content)
        return "\n\n".join(prompt_parts), ["<|endoftext|>"]


def robust_json() -> bool:
    """Method v4.1 parsing (MAPREFLECT_ROBUST_JSON=1, or MAPREFLECT_V41=1 for every v4.1 repair). Off by default,
    so method v4 parses exactly as before."""
    return os.environ.get("MAPREFLECT_V41") == "1" or os.environ.get("MAPREFLECT_ROBUST_JSON") == "1"


_DECODER = json.JSONDecoder(strict=False)


def json_blocks(s: str):
    """Every complete JSON object or array in s, left to right, each block once (nested values are inside it).

    A decoder reads from each opening bracket (strict=False, so a raw newline or tab inside a string is accepted);
    a block that fails to decode is skipped one character on, so prose such as "scores in [0, 1]" or a truncated
    tail never swallows a valid block, unlike a greedy first-bracket-to-last-bracket match.
    """
    i, n = 0, len(s or "")
    while i < n:
        j = min([k for k in (s.find("[", i), s.find("{", i)) if k >= 0], default=-1)
        if j < 0:
            return
        try:
            v, end = _DECODER.raw_decode(s, j)
        except ValueError:
            i = j + 1
            continue
        yield v
        i = end


def first_json(s: str, kind: type, want=None):
    """(value, status): the first JSON block of the given kind (list or dict) that satisfies want, if given.

    status: "ok"; "quotes_fixed" when it decodes only after single quotes become double quotes; "unparsed" when no
    block qualifies (value None).
    """
    ok = lambda v: isinstance(v, kind) and (want is None or want(v))
    for v in json_blocks(s):
        if ok(v):
            return v, "ok"
    if s and "'" in s:
        for v in json_blocks(s.replace("'", '"')):
            if ok(v):
                return v, "quotes_fixed"
    return None, "unparsed"


def _ranking_like(v: List) -> bool:
    return any(isinstance(x, dict) for x in v) or not v


def safe_parse_json_array(s: str) -> Optional[List]:
    """Parse JSON array from text, handling common issues.

    Under method v4.1 parsing (robust_json) this is the first JSON array that holds an object or is empty, so an
    unrelated bracket in prose ("[0, 1]") is passed over; otherwise the v4 greedy match.
    """
    if not s:
        return None
    if robust_json():
        v, _ = first_json(s, list, _ranking_like)
        return v
    m = re.search(r"(\[.*\])", s, re.S)
    txt = m.group(1) if m else s
    try:
        return json.loads(txt)
    except Exception:
        try:
            return json.loads(txt.replace("'", '"'))
        except Exception:
            return None


# =============================================================================
# OMOP-Specific Enhancements
# =============================================================================

def is_omop_dataset(dataset: str) -> bool:
    """Check if dataset is an OMOP dataset."""
    return dataset.upper().startswith("OMOP")


def is_cdm_dataset(dataset: str) -> bool:
    """Check if dataset maps product schemas to the common entity model.

    CEM_CDM rides the existing multi-target (ITOpsGraph-style) candidate path:
    each sample carries target_schema="cem" + metadata.target_type="<CanonType>",
    which the candidate generator turns into the prefix "cem_<CanonType>." to
    restrict candidates to canonical fields. No bespoke filtering needed.
    """
    return dataset.upper() in ("CEM_CDM", "CDM")


def parse_omop_utterance(utterance: str) -> Dict[str, str]:
    """
    Parse OMOP-style utterance to extract source/target table information.

    Examples:
        "map omop person with cms beneficiarysummary"
            → {"source_schema": "omop", "source_table": "person",
               "target_schema": "cms", "target_table": "beneficiarysummary"}
        "map omop visit occurrence with cms inpatientclaims"
            → {"source_schema": "omop", "source_table": "visit_occurrence",
               "target_schema": "cms", "target_table": "inpatientclaims"}
    """
    utterance_lower = utterance.lower().strip()
    result = {
        "source_schema": "",
        "source_table": "",
        "target_schema": "",
        "target_table": ""
    }

    # Pattern: "map {source_schema} {source_table} with {target_schema} {target_table}"
    # Handle multi-word table names like "visit occurrence"
    pattern = r"map\s+(\w+)\s+(.+?)\s+with\s+(\w+)\s+(.+?)$"
    match = re.match(pattern, utterance_lower)

    if match:
        result["source_schema"] = match.group(1)
        result["source_table"] = match.group(2).replace(" ", "_")
        result["target_schema"] = match.group(3)
        result["target_table"] = match.group(4).replace(" ", "_")

    return result


def extract_omop_target_table(sample: Dict) -> Dict[str, str]:
    """
    Extract target table info from OMOP sample's target_schema field.

    For OMOP datasets, the target_schema is often a dict like:
        {"id": "cms_BENEFICIARYSUMMARY", "name": "CMS BENEFICIARYSUMMARY", ...}

    Returns dict with target_schema (e.g., "cms") and target_table (e.g., "BENEFICIARYSUMMARY")
    """
    result = {"target_schema": "", "target_table": ""}

    target_schema = sample.get("target_schema") or sample.get("expected_target") or ""

    if isinstance(target_schema, dict):
        # Extract from dict format
        target_id = target_schema.get("id", target_schema.get("name", ""))
    else:
        target_id = str(target_schema)

    if target_id and "_" in target_id:
        # Parse "cms_BENEFICIARYSUMMARY" → schema="cms", table="BENEFICIARYSUMMARY"
        parts = target_id.split("_", 1)
        result["target_schema"] = parts[0].lower()
        result["target_table"] = parts[1] if len(parts) > 1 else ""

    return result


def get_omop_field_type_hints(field_name: str) -> List[str]:
    """
    Get type hints for OMOP field to help with semantic matching.

    Returns hints about what kind of field this is (ID, date, code, etc.)
    """
    field_lower = field_name.lower()
    # Extract just the field name (after the last dot)
    field_only = field_lower.split(".")[-1] if "." in field_lower else field_lower
    hints = []

    # ID fields - check field_only for more precise matching
    if field_only.endswith("_id") or field_only == "id":
        # Check specific ID types in order of specificity
        if field_only.startswith("provider") or field_only == "provider_id":
            hints.append("This is a PROVIDER identifier field")
            hints.append("Should map to provider fields like: *_npi, provider_id, attending_*")
            hints.append("WARNING: Do NOT map to patient IDs (desynpuf_id, bene_id)")
        elif field_only.startswith("location") or field_only == "location_id":
            hints.append("This is a LOCATION identifier field")
            hints.append("Should map to geographic fields like: *_county_cd, *_state_cd, sp_state_code, zip_code")
            hints.append("WARNING: Do NOT map to provider NPIs or patient IDs")
        elif field_only.startswith("care_site") or field_only == "care_site_id":
            hints.append("This is a CARE SITE identifier field")
            hints.append("Should map to facility/site fields like: facility_id, hospital_id")
        elif "concept" in field_only:
            hints.append("This is a CONCEPT code field (categorical/coded value)")
            hints.append("Should map to code fields like: *_cd, *_code, *_type")
        elif field_only.startswith("person") or field_only in ["person_id", "person_source_value"]:
            hints.append("This is a PERSON/PATIENT identifier field")
            hints.append("Should map to patient ID fields like: desynpuf_id, bene_id, subject_id, patient_id")
            hints.append("WARNING: Do NOT map to provider NPIs (*_npi)")
        elif field_only.startswith("visit") or "occurrence" in field_only:
            hints.append("This is a VISIT/ENCOUNTER identifier field")
            hints.append("Should map to claim/encounter IDs like: clm_id, hadm_id, encounter_id")
        else:
            hints.append("This is an identifier field")

    # Date fields
    if any(kw in field_lower for kw in ["date", "datetime", "time", "_dt"]):
        hints.append("This is a DATE/TIME field")
        if "birth" in field_lower:
            hints.append("Birth date - should map to: bene_birth_dt, dob, date_of_birth")
        elif "death" in field_lower:
            hints.append("Death date - should map to: bene_death_dt, dod, date_of_death")
        elif "start" in field_lower or "admission" in field_lower:
            hints.append("Start/admission date - should map to: clm_admsn_dt, admittime, start_date")
        elif "end" in field_lower or "discharge" in field_lower:
            hints.append("End/discharge date - should map to: nch_bene_dschrg_dt, dischtime, end_date")
        else:
            hints.append("Should map to date fields like: *_dt, *_date, *time")

    # Demographics
    if any(kw in field_lower for kw in ["gender", "sex"]):
        hints.append("This is a GENDER/SEX field")
        hints.append("Should map to: bene_sex_ident_cd, gender, sex")

    if any(kw in field_lower for kw in ["race", "ethnicity"]):
        hints.append("This is a RACE/ETHNICITY field")
        hints.append("Should map to: bene_race_cd, race, ethnicity")

    if any(kw in field_lower for kw in ["year_of_birth", "month_of_birth", "day_of_birth"]):
        hints.append("This is a BIRTH DATE component")
        hints.append("Should map to birth date fields: bene_birth_dt, dob")

    return hints


def filter_candidates_by_omop_table(
    candidates: List[Dict],
    target_table: str,
    target_schema: str
) -> List[Dict]:
    """
    Filter candidates to only include those from the specified OMOP target table.

    Args:
        candidates: List of candidate dicts with 'id' field
        target_table: Target table name (e.g., "beneficiarysummary")
        target_schema: Target schema name (e.g., "cms")

    Returns:
        Filtered list of candidates from the target table only
    """
    if not target_table:
        return candidates

    # Build expected prefix: cms_BENEFICIARYSUMMARY, mimic_PATIENTS, synthea_patients
    target_prefix = f"{target_schema}_{target_table}".lower()

    filtered = []
    for cand in candidates:
        cand_id = cand.get("id", cand.get("name", "")).lower()
        # Check if candidate is from the target table
        if cand_id.startswith(target_prefix + ".") or cand_id.startswith(target_prefix + "_"):
            filtered.append(cand)

    if filtered:
        logger.debug(f"OMOP table filter: {len(candidates)} → {len(filtered)} candidates (table={target_table})")
        return filtered
    else:
        # If no matches found, try a looser match (table name anywhere in ID)
        for cand in candidates:
            cand_id = cand.get("id", cand.get("name", "")).lower()
            if target_table.lower() in cand_id:
                filtered.append(cand)

        if filtered:
            logger.debug(f"OMOP table filter (loose): {len(candidates)} → {len(filtered)} candidates")
            return filtered

    # Fallback: return original candidates with warning
    logger.warning(f"OMOP table filter found no matches for table '{target_table}', returning all candidates")
    return candidates


def safe_parse_json_object(s: str, want: Optional[str] = None) -> Optional[Dict]:
    """Parse JSON object from text.

    Under method v4.1 parsing (robust_json) this is the first JSON object, or the first one holding the key want
    when it is given (a reasoning preamble with its own braces is passed over); otherwise the v4 greedy match,
    which ignores want.
    """
    if not s:
        return None
    if robust_json():
        v, _ = first_json(s, dict, (lambda o: want in o) if want else None)
        return v
    m = re.search(r"(\{.*\})", s, re.S)
    txt = m.group(1) if m else s
    try:
        return json.loads(txt)
    except Exception:
        try:
            return json.loads(txt.replace("'", '"'))
        except Exception:
            return None


def normalize_field_id(field_id: str) -> str:
    """Normalize a field ID by stripping formatting prefixes.

    Removes:
    - Leading rank prefixes like "1. ", "2. "
    - Leading "id: " prefix
    - Extra whitespace

    Args:
        field_id: The field ID string to normalize

    Returns:
        Normalized field ID
    """
    if not field_id or not isinstance(field_id, str):
        return field_id
    # Strip leading rank prefix (e.g., "1. ", "2. ", "10. ")
    s = re.sub(r'^\d+\.\s*', '', field_id)
    # Strip leading "id: " prefix (case-insensitive)
    s = re.sub(r'^id:\s*', '', s, flags=re.IGNORECASE)
    return s.strip()


def build_property_graph(lookup: Dict[str, Any]) -> nx.Graph:
    """Build a property graph connecting semantically related properties.

    Properties are connected if they share:
    - Same product_name (+0.5 weight)
    - Same type_name (+0.3 weight)
    - Similar field names (+0.2 * Jaccard similarity)

    Edges are created for pairs with weight > 0.1.
    """
    t0 = time.time()
    G = nx.Graph()

    # Add all properties as nodes
    for key, value in lookup.items():
        node_data = value if isinstance(value, dict) else {"value": value}
        G.add_node(key, **node_data)

    # Create edges based on similarity
    keys = list(lookup.keys())
    edge_count = 0

    for i, p1 in enumerate(keys):
        d1 = lookup[p1] if isinstance(lookup[p1], dict) else {}
        for p2 in keys[i + 1:]:
            d2 = lookup[p2] if isinstance(lookup[p2], dict) else {}

            weight = 0.0

            # Same product
            if d1.get("product_name") == d2.get("product_name") and d1.get("product_name"):
                weight += 0.5

            # Same type
            if d1.get("type_name") == d2.get("type_name") and d1.get("type_name"):
                weight += 0.3

            # Field name similarity (Jaccard on words)
            w1 = set(str(d1.get("field_name", p1)).lower().replace("_", ".").split("."))
            w2 = set(str(d2.get("field_name", p2)).lower().replace("_", ".").split("."))
            if w1 and w2:
                sim = len(w1 & w2) / max(len(w1 | w2), 1)
                weight += 0.2 * sim

            if weight > 0.1:
                G.add_edge(p1, p2, weight=weight)
                edge_count += 1

    logger.info(f"Built property graph in {time.time() - t0:.2f}s | nodes={G.number_of_nodes()} edges={edge_count}")
    return G


def expand_candidates_with_neighbors(
    candidates: List[Dict],
    property_graph: nx.Graph,
    lookup: Dict[str, Any],
    neighbor_k: int = 5
) -> List[Dict]:
    """Expand candidate list with 1-hop neighbors from the property graph.

    Args:
        candidates: Initial candidate list from retrieval
        property_graph: NetworkX graph of property relationships
        lookup: Property lookup for descriptions
        neighbor_k: Maximum number of neighbors to add per candidate

    Returns:
        Expanded candidate list with neighbors appended
    """
    if neighbor_k <= 0 or property_graph.number_of_nodes() == 0:
        return candidates

    # Get IDs of current candidates
    candidate_ids = {c.get("id", c.get("name", "")) for c in candidates}

    # Collect neighbors
    neighbor_ids = set()
    for cand in candidates:
        cand_id = cand.get("id", cand.get("name", ""))
        if cand_id in property_graph:
            for nbr in property_graph.neighbors(cand_id):
                if nbr not in candidate_ids:
                    neighbor_ids.add(nbr)

    # Limit neighbors
    neighbor_ids = set(sorted(neighbor_ids)[:neighbor_k])

    # Build neighbor candidate entries
    neighbor_candidates = []
    for nbr_id in neighbor_ids:
        nbr_info = lookup.get(nbr_id, {})
        nbr_desc = nbr_info.get("description", nbr_id) if isinstance(nbr_info, dict) else str(nbr_info)
        neighbor_candidates.append({
            "id": nbr_id,
            "name": nbr_id,
            "doc": f"{nbr_id} -- {nbr_desc}",
            "description": nbr_desc,
            "distance": 0.6,  # Slightly worse than direct retrieval
            "type": nbr_info.get("type", "") if isinstance(nbr_info, dict) else "",
            "from_neighbor_expansion": True
        })

    if neighbor_candidates:
        logger.debug(f"Expanded candidates with {len(neighbor_candidates)} neighbors")

    return candidates + neighbor_candidates


# =============================================================================
# LLM Client
# =============================================================================

class EndpointBBatchClient:
    """EndpointB client optimized for batch inference with caching."""

    def __new__(cls, model_id: str = "", *args, **kwargs):
        # Callers that pick EndpointABatchClient for 'endpoint_a/' and this class otherwise (the frozen
        # scripts/naacl/reflect_pilot.py among them) reach the LiteLLM gateway for 'litellm/'
        # ids without any change of their own.
        if cls is EndpointBBatchClient and str(model_id).startswith("litellm/"):
            return LiteLLMBatchClient(model_id, *args, **kwargs)
        return super().__new__(cls)

    def __init__(self, model_id: str, config: MapReflectConfig):
        if not HAS_ENDPOINT_B:
            raise RuntimeError("endpoint_b_sdk required for batch processing")

        self.model_id = model_id.replace("endpoint_b/", "")
        self.config = config
        self._model = None
        self._lock = threading.Lock()

        # Response cache (optionally persisted to disk so the MAP phase of one
        # (dataset, seed) is computed once and reused across all reflection configs)
        self._cache: Dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_path: Optional[Path] = None
        if config.cache_dir:
            self._cache_path = Path(config.cache_dir) / "llm_response_cache.json"
            if self._cache_path.exists():
                try:
                    with open(self._cache_path) as f:
                        self._cache = json.load(f)
                    logger.info(f"Loaded {len(self._cache)} cached responses from {self._cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to load cache from {self._cache_path}: {e}")

    def save_cache(self):
        """Persist the response cache to disk (no-op if no cache_dir configured)."""
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f)
            logger.info(f"Saved {len(self._cache)} responses to {self._cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache to {self._cache_path}: {e}")

    def _get_model(self):
        """Get or create model instance (thread-safe)."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    api_key = os.getenv("ENDPOINT_B_API_KEY") or os.getenv("ENDPOINT_B_APIKEY")
                    project_id = os.getenv("ENDPOINT_B_PROJECT_ID")
                    url = os.getenv("ENDPOINT_B_URL", "https://ENDPOINT_B")

                    if not api_key or not project_id:
                        raise RuntimeError("ENDPOINT_B_API_KEY and ENDPOINT_B_PROJECT_ID required")

                    _, stop_sequences = format_prompt_for_model([], self.model_id)

                    if self.config.sample_main_decoding:
                        # Seeded sampling: lets the seed actually move outputs so
                        # cross-seed variance is measurable (greedy ignores the seed).
                        params = {
                            GenParams.MAX_NEW_TOKENS: self.config.max_tokens,
                            GenParams.TEMPERATURE: self.config.temperature,
                            GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
                            GenParams.RANDOM_SEED: self.config.seed,
                            GenParams.STOP_SEQUENCES: stop_sequences,
                        }
                        logger.info(f"Main decoding: SAMPLE (temp={self.config.temperature}, seed={self.config.seed})")
                    else:
                        params = {
                            GenParams.MAX_NEW_TOKENS: self.config.max_tokens,
                            GenParams.TEMPERATURE: self.config.temperature,
                            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
                            GenParams.STOP_SEQUENCES: stop_sequences,
                        }

                    self._model = ModelInference(
                        model_id=self.model_id,
                        params=params,
                        credentials=Credentials(url=url, api_key=api_key),
                        project_id=project_id
                    )
                    logger.info(f"Initialized EndpointB model: {self.model_id}")

        return self._model

    def _cache_key(self, prompt: str) -> str:
        """Generate cache key for a prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def _endpoint(self) -> str:
        return os.getenv("ENDPOINT_B_URL", "https://ENDPOINT_B").rstrip("/") + "/ml/v1/text/generation"

    def generate_batch(
        self,
        prompts: List[str],
        use_cache: bool = True,
        batch_size: int = 50
    ) -> List[str]:
        """
        Generate responses for multiple prompts using batch inference.

        Args:
            prompts: List of prompts to process
            use_cache: Whether to use response caching
            batch_size: Max prompts per API call

        Returns:
            List of responses in same order as prompts
        """
        if not prompts:
            return []

        model = self._get_model()
        results = [None] * len(prompts)
        uncached_indices = []
        uncached_prompts = []

        # Check cache
        if use_cache and self.config.use_cache:
            for i, prompt in enumerate(prompts):
                key = self._cache_key(prompt)
                if key in self._cache:
                    results[i] = self._cache[key]
                    self._cache_hits += 1
                else:
                    uncached_indices.append(i)
                    uncached_prompts.append(prompt)
                    self._cache_misses += 1

            if self._cache_hits > 0:
                logger.debug(f"Cache: {self._cache_hits} hits, {self._cache_misses} misses")
        else:
            uncached_indices = list(range(len(prompts)))
            uncached_prompts = prompts

        if not uncached_prompts:
            return results

        # Process in batches
        for batch_start in range(0, len(uncached_prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(uncached_prompts))
            batch_prompts = uncached_prompts[batch_start:batch_end]
            batch_indices = uncached_indices[batch_start:batch_end]

            t0 = time.time()
            if LL.enabled():
                LL.take_throttle_wait()
            try:
                # Use max concurrency (10) for parallel requests
                if os.environ.get("TOKEN_LOG") or LL.enabled():
                    # Raw responses carry token counts; the text is the same generated_text.
                    raw = model.generate(prompt=batch_prompts, concurrency_limit=10)
                    raw = raw if isinstance(raw, list) else [raw]
                    responses = []
                    dt_each = (time.time() - t0) / max(len(batch_prompts), 1)
                    t_end = time.time()
                    wait = LL.take_throttle_wait() if LL.enabled() else 0.0
                    for i_rr, rr in enumerate(raw):
                        res = (rr.get("results") or [{}])[0] if isinstance(rr, dict) else {}
                        responses.append(res.get("generated_text", ""))
                        if os.environ.get("TOKEN_LOG"):
                            log_usage(self.model_id, res.get("input_token_count", 0),
                                      res.get("generated_token_count", 0), dt_each,
                                      finish_reason=res.get("stop_reason"), latency_basis="batch_mean")
                        if LL.enabled():
                            LL.log_call(t_start=t0, t_end=t_end, endpoint=self._endpoint(), model=self.model_id,
                                        prompt=batch_prompts[i_rr] if i_rr < len(batch_prompts) else "",
                                        prompt_tokens=res.get("input_token_count"),
                                        completion_tokens=res.get("generated_token_count"),
                                        stage=get_usage_stage(), context=get_usage_context(),
                                        basis="call" if len(batch_prompts) == 1 else "batch",
                                        n_batch=len(batch_prompts), throttle_wait_s=wait)
                else:
                    responses = model.generate_text(prompt=batch_prompts, concurrency_limit=10)

                if isinstance(responses, str):
                    responses = [responses]
                elif not isinstance(responses, list):
                    responses = list(responses)

                t1 = time.time()
                logger.info(f"Batch {len(batch_prompts)} prompts in {t1-t0:.2f}s ({len(batch_prompts)/(t1-t0):.1f} p/s)")

                # Store results and cache
                for idx, (orig_idx, response) in enumerate(zip(batch_indices, responses)):
                    results[orig_idx] = response
                    if use_cache and self.config.use_cache:
                        key = self._cache_key(batch_prompts[idx])
                        self._cache[key] = response

            except Exception as e:
                logger.error(f"Batch generation failed: {e}")
                if LL.enabled():
                    for prompt in batch_prompts:  # the failed batch request, then each sequential retry below
                        LL.log_call(t_start=t0, t_end=time.time(), endpoint=self._endpoint(), model=self.model_id,
                                    prompt=prompt, stage=get_usage_stage(), context=get_usage_context(),
                                    basis="call" if len(batch_prompts) == 1 else "batch",
                                    n_batch=len(batch_prompts), error=True, throttle_wait_s=LL.take_throttle_wait())
                # Fallback to sequential
                for idx, (orig_idx, prompt) in enumerate(zip(batch_indices, batch_prompts)):
                    try:
                        t_f = time.time()
                        response = model.generate_text(prompt=prompt)
                        if LL.enabled():
                            LL.log_call(t_start=t_f, t_end=time.time(), endpoint=self._endpoint(),
                                        model=self.model_id, prompt=prompt, stage=get_usage_stage(),
                                        context=get_usage_context(), throttle_wait_s=LL.take_throttle_wait())
                        if isinstance(response, list):
                            response = response[0]
                        results[orig_idx] = response
                        if use_cache and self.config.use_cache:
                            self._cache[self._cache_key(prompt)] = response
                    except Exception as e2:
                        logger.error(f"Fallback failed: {e2}")
                        results[orig_idx] = f"Error: {e2}"

        return results

    def generate_single(self, prompt: str, temperature: float = None) -> str:
        """Generate a single response (for reflection step)."""
        model = self._get_model()

        # Check cache first
        if self.config.use_cache:
            key = self._cache_key(prompt)
            if key in self._cache:
                self._cache_hits += 1
                return self._cache[key]
            self._cache_misses += 1

        try:
            response = model.generate_text(prompt=prompt)
            if isinstance(response, list):
                response = response[0]

            if self.config.use_cache:
                self._cache[self._cache_key(prompt)] = response

            return response
        except Exception as e:
            logger.error(f"Single generation failed: {e}")
            return f"Error: {e}"

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache)
        }

    def generate_with_logprobs(self, prompt: str, max_tokens: int = None) -> Dict[str, Any]:
        """
        Generate response with token-level log probabilities.

        Args:
            prompt: The prompt to generate from
            max_tokens: Override max tokens (useful for short verification prompts)

        Returns:
            Dict with 'text', 'generated_tokens' (list of {text, logprob}), and 'perplexity'
        """
        api_key = os.getenv("ENDPOINT_B_API_KEY") or os.getenv("ENDPOINT_B_APIKEY")
        project_id = os.getenv("ENDPOINT_B_PROJECT_ID")
        url = os.getenv("ENDPOINT_B_URL", "https://ENDPOINT_B")

        if not api_key or not project_id:
            raise RuntimeError("ENDPOINT_B_API_KEY and ENDPOINT_B_PROJECT_ID required")

        _, stop_sequences = format_prompt_for_model([], self.model_id)

        # Create model with return_options for logprobs
        try:
            from endpoint_b_sdk.foundation_models.schema import TextGenParameters, ReturnOptionProperties
        except ImportError:
            # Fallback for older SDK versions
            logger.warning("ReturnOptionProperties not available, using heuristic UQ instead")
            return {
                "text": self.generate_single(prompt),
                "generated_tokens": [],
                "perplexity": None
            }

        params = {
            GenParams.MAX_NEW_TOKENS: max_tokens if max_tokens else self.config.max_tokens,
            GenParams.TEMPERATURE: self.config.temperature,
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.STOP_SEQUENCES: stop_sequences,
            GenParams.RETURN_OPTIONS: {
                "input_text": True,  # Required when requesting logprobs
                "generated_tokens": True,
                "token_logprobs": True,
                "top_n_tokens": 5,  # Get top alternatives for P(True) analysis (EndpointB max is 5)
            }
        }

        model = ModelInference(
            model_id=self.model_id,
            params=params,
            credentials=Credentials(url=url, api_key=api_key),
            project_id=project_id
        )

        try:
            result = model.generate(prompt=prompt)

            # Extract text and tokens
            if isinstance(result, dict):
                text = result.get("results", [{}])[0].get("generated_text", "")
                tokens = result.get("results", [{}])[0].get("generated_tokens", [])
            else:
                # Handle list response
                text = result[0].get("generated_text", "") if result else ""
                tokens = result[0].get("generated_tokens", []) if result else []

            # Calculate perplexity from logprobs
            logprobs = []
            for token in tokens:
                if isinstance(token, dict) and "logprob" in token:
                    lp = token.get("logprob")
                    if lp is not None and lp != 0:  # Skip padding tokens
                        logprobs.append(lp)

            if logprobs:
                # Perplexity = exp(-mean(log_probs))
                perplexity = float(np.exp(-np.mean(logprobs)))
            else:
                perplexity = None

            return {
                "text": text,
                "generated_tokens": tokens,
                "perplexity": perplexity,
                "logprobs": logprobs
            }

        except Exception as e:
            logger.error(f"Generate with logprobs failed: {e}")
            return {
                "text": self.generate_single(prompt),
                "generated_tokens": [],
                "perplexity": None
            }


class EndpointABatchClient:
    """ENDPOINT_A (OpenAI-compatible gateway) client mirroring EndpointBBatchClient's interface
    for MAP-phase runs. Routes generate_batch through the ENDPOINT_A chat endpoint, so backbones
    endpoint_b cannot serve (e.g. gpt-oss-120b) run via the same runner and in parallel with
    endpoint_b (separate endpoint, separate rate limit).

    model_id form: 'endpoint_a/openai/gpt-oss-120b-a100' -> body model 'openai/gpt-oss-120b-a100',
    endpoint suffix 'gpt-oss-120b-a100' (overridable via ENDPOINT_A_ENDPOINT). Prompts arrive already
    plain-formatted (format_prompt_for_model default fallback for non-granite/llama ids); we send
    each as a single user chat message. safe_parse_json_array tolerates any reasoning preamble.
    """

    def __init__(self, model_id: str, config: 'MapReflectConfig'):
        import openai
        self.config = config
        self.body_model = model_id.replace("endpoint_a/", "", 1)
        self.model_id = self.body_model  # drives format_prompt_for_model (-> default fallback)
        suffix = os.environ.get("ENDPOINT_A_ENDPOINT") or self.body_model.split("/")[-1]
        key = os.environ["ENDPOINT_A_API_KEY"]
        base = f'{os.environ["ENDPOINT_A_URL"].rstrip("/")}/{suffix}/v1'
        self._base_url = base
        # ENDPOINT_A_TIMEOUT (seconds) bounds each call, so one hung request cannot stall a stream; the retry loop
        # below retries timeouts. Unset keeps the client's default.
        tmo = os.environ.get("ENDPOINT_A_TIMEOUT")
        self._client = openai.OpenAI(api_key=key, base_url=base, default_headers={"ENDPOINT_A_API_KEY": key},
                                     **({"timeout": float(tmo), "max_retries": 0} if tmo else {}))
        # Qwen3.x thinks by default; I turn it off so it answers like the other backbones (one
        # ranking, no reasoning preamble). ENDPOINT_A_ENABLE_THINKING=1 keeps the model's default.
        self._extra_body = None
        if "qwen3" in self.body_model.lower() and os.environ.get("ENDPOINT_A_ENABLE_THINKING") != "1":
            self._extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        self._cache: Dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_path: Optional[Path] = None
        if config.cache_dir:
            self._cache_path = Path(config.cache_dir) / "llm_response_cache.json"
            if self._cache_path.exists():
                try:
                    with open(self._cache_path) as f:
                        self._cache = json.load(f)
                except Exception:
                    pass
        logger.info(f"Initialized ENDPOINT_A model: {self.body_model} (endpoint {suffix})")

    def _cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    def _one(self, prompt: str) -> str:
        t_first = time.time()
        for attempt in range(5):
            try:
                t0 = time.time()
                msgs = (chat_messages(prompt) if os.environ.get("ENDPOINT_A_SPLIT_TEMPLATE") == "1"
                        else [{"role": "user", "content": prompt}])
                r = self._client.chat.completions.create(
                    model=self.body_model,
                    messages=msgs,
                    temperature=self.config.temperature,
                    max_tokens=max(self.config.max_tokens, 3000),
                    extra_body=self._extra_body)
                u = getattr(r, "usage", None)
                log_usage(self.body_model, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0),
                          time.time() - t0, finish_reason=(r.choices[0].finish_reason if r.choices else None))
                if LL.enabled():
                    LL.log_call(t_start=t_first, t_end=time.time(), endpoint=self._base_url, model=self.body_model,
                                prompt=prompt, prompt_tokens=getattr(u, "prompt_tokens", None),
                                completion_tokens=getattr(u, "completion_tokens", None), stage=get_usage_stage(),
                                context=get_usage_context(), attempts=attempt + 1)
                return r.choices[0].message.content or ""
            except Exception as e:
                s = str(e).lower()
                if any(p in s for p in ("rate", "429", "timeout", "timed out", "500", "502", "503", "504",
                                        "overloaded")):
                    time.sleep(min(2 ** attempt + 0.3 * attempt, 20))
                    continue
                logger.error(f"ENDPOINT_A call failed: {str(e)[:150]}")
                if LL.enabled():
                    LL.log_call(t_start=t_first, t_end=time.time(), endpoint=self._base_url, model=self.body_model,
                                prompt=prompt, stage=get_usage_stage(), context=get_usage_context(),
                                attempts=attempt + 1, error=True)
                return f"Error: {e}"
        if LL.enabled():
            LL.log_call(t_start=t_first, t_end=time.time(), endpoint=self._base_url, model=self.body_model,
                        prompt=prompt, stage=get_usage_stage(), context=get_usage_context(), attempts=5, error=True)
        return "Error: retries exhausted"

    def generate_batch(self, prompts: List[str], use_cache: bool = True,
                       batch_size: int = 50) -> List[str]:
        if not prompts:
            return []
        results: List[Optional[str]] = [None] * len(prompts)
        todo = []
        for i, p in enumerate(prompts):
            if use_cache and self.config.use_cache and self._cache_key(p) in self._cache:
                results[i] = self._cache[self._cache_key(p)]
                self._cache_hits += 1
            else:
                todo.append(i)
                self._cache_misses += 1
        if todo:
            from concurrent.futures import ThreadPoolExecutor
            t0 = time.time()
            stage, ctx = get_usage_stage(), get_usage_context()

            def run(i):
                set_usage_stage(stage)
                set_usage_context(ctx)
                return self._one(prompts[i])
            # ENDPOINT_A_BATCH_WORKERS bounds the requests in flight (the latency measurement sends one at a time);
            # unset keeps 32. The responses do not depend on it.
            with ThreadPoolExecutor(max_workers=int(os.environ.get("ENDPOINT_A_BATCH_WORKERS") or 32)) as ex:
                out = list(ex.map(run, todo))
            for i, resp in zip(todo, out):
                results[i] = resp
                if use_cache and self.config.use_cache:
                    self._cache[self._cache_key(prompts[i])] = resp
            dt = time.time() - t0
            logger.info(f"ENDPOINT_A batch {len(todo)} prompts in {dt:.1f}s ({len(todo)/max(dt,1e-6):.1f} p/s)")
        return results

    def generate_single(self, prompt: str, temperature: float = None) -> str:
        return self._one(prompt)

    def save_cache(self):
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def get_cache_stats(self) -> Dict[str, int]:
        return {"hits": self._cache_hits, "misses": self._cache_misses, "size": len(self._cache)}


class LiteLLMBudgetExceeded(RuntimeError):
    pass


class LiteLLMBatchClient(EndpointABatchClient):
    """LiteLLM gateway (ENDPOINT_C) (OpenAI-compatible) client, for frontier models used sparingly.

    model_id form: 'litellm/azure/gpt-5.6-luna' -> body model 'azure/gpt-5.6-luna', sent verbatim.
    Base URL from LITELLM_BASE_URL, key from LITELLM_API_KEY (Authorization: Bearer). Only the
    transport differs from EndpointABatchClient; prompts and parsing are the method's own.

    Every call appends its usage to LITELLM_TOKEN_LOG (jsonl: prompt, completion and reasoning
    tokens, finish_reason, priced cost). The gateway returns no cost, so I price tokens from
    GET /v1/model/info, falling back to the card below. The spend over the whole log is checked
    before each call against LITELLM_COST_CAP_USD (default 60) and the run stops once it is
    reached. finish_reason == 'length' is logged as a truncation. Error text never carries the key.
    """

    FALLBACK_CARDS = {
        "azure/gpt-5.6-luna": (1e-6, 6e-6),
        "aws/claude-opus-4-7": (3.8e-6, 1.9e-5),
    }

    def __init__(self, model_id: str, config: 'MapReflectConfig'):
        import openai
        import threading
        self.config = config
        self.body_model = model_id.replace("litellm/", "", 1)
        self.model_id = self.body_model  # drives format_prompt_for_model (-> default fallback)
        self._key = os.environ["LITELLM_API_KEY"]
        self._base = os.environ.get("LITELLM_BASE_URL",
                                    "https://ENDPOINT_C").rstrip("/")
        # Calls take about 20 s; a request the gateway leaves hanging is abandoned and retried.
        self._client = openai.OpenAI(api_key=self._key, base_url=f"{self._base}/v1",
                                     timeout=float(os.environ.get("LITELLM_TIMEOUT", "180")), max_retries=0)
        self._max_tokens = int(os.environ.get("LITELLM_MAX_TOKENS", "8192"))
        self._cap = float(os.environ.get("LITELLM_COST_CAP_USD", "60"))
        self._log_path = Path(os.environ.get("LITELLM_TOKEN_LOG", "results/naacl/frontier_luna/tokens.jsonl"))
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._send_temperature = True
        self._rates = self._fetch_rates()
        self._spent = 0.0
        if self._log_path.exists():
            for line in open(self._log_path):
                try:
                    self._spent += float(json.loads(line).get("cost_usd", 0.0))
                except Exception:
                    pass
        self._cache: Dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_path: Optional[Path] = None
        logger.info(f"Initialized LiteLLM model: {self.body_model} (spent so far ${self._spent:.2f} "
                    f"of ${self._cap:.2f}; rates in/out per token {self._rates})")

    def _sanitize(self, text: str) -> str:
        return str(text).replace(self._key, "<key>")[:300]

    def _fetch_rates(self):
        import urllib.request
        try:
            req = urllib.request.Request(f"{self._base}/v1/model/info",
                                         headers={"authorization": f"Bearer {self._key}"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            for m in (data.get("data", data) if isinstance(data, dict) else data):
                info = m.get("model_info") or {}
                if self.body_model in (m.get("model_name"), info.get("key")):
                    pin, pout = info.get("input_cost_per_token"), info.get("output_cost_per_token")
                    if isinstance(pin, (int, float)) and isinstance(pout, (int, float)):
                        return (float(pin), float(pout))
        except Exception as e:
            logger.warning(f"LiteLLM /v1/model/info unavailable ({self._sanitize(e)}); using the fallback card")
        return self.FALLBACK_CARDS.get(self.body_model, (1e-5, 3e-5))

    def _record(self, r, dt: float):
        u = getattr(r, "usage", None)
        pt = int(getattr(u, "prompt_tokens", 0) or 0)
        ct = int(getattr(u, "completion_tokens", 0) or 0)
        det = getattr(u, "completion_tokens_details", None)
        rt = int(getattr(det, "reasoning_tokens", 0) or 0) if det else 0
        pdet = getattr(u, "prompt_tokens_details", None)
        cached = int(getattr(pdet, "cached_tokens", 0) or 0) if pdet else 0
        fin = r.choices[0].finish_reason if r.choices else None
        cost = pt * self._rates[0] + ct * self._rates[1]
        rec = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), model=self.body_model, prompt_tokens=pt,
                   completion_tokens=ct, reasoning_tokens=rt, cached_tokens=cached, finish_reason=fin,
                   truncated=fin == "length", seconds=round(dt, 2), cost_usd=cost,
                   run=os.environ.get("LITELLM_RUN_TAG", ""))
        with self._lock:
            self._spent += cost
            with open(self._log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        log_usage(self.body_model, pt, ct, dt, cost_usd=cost, finish_reason=fin)
        if fin == "length":
            logger.warning(f"LiteLLM truncation (finish_reason=length, {ct} completion tokens)")

    def _one(self, prompt: str) -> str:
        if self._spent >= self._cap:
            raise LiteLLMBudgetExceeded(f"LiteLLM cost cap reached: ${self._spent:.2f} >= ${self._cap:.2f}")
        for attempt in range(5):
            try:
                kw = dict(model=self.body_model, messages=[{"role": "user", "content": prompt}],
                          max_completion_tokens=self._max_tokens)
                if self._send_temperature:
                    kw["temperature"] = self.config.temperature
                t0 = time.time()
                r = self._client.chat.completions.create(**kw)
                self._record(r, time.time() - t0)
                return r.choices[0].message.content or ""
            except LiteLLMBudgetExceeded:
                raise
            except Exception as e:
                s = str(e).lower()
                if "temperature" in s and ("unsupported" in s or "400" in s):
                    # Reasoning models accept only the default temperature. Other threads may have
                    # sent theirs before the first rejection, so every such call retries without it.
                    if self._send_temperature:
                        self._send_temperature = False
                        logger.warning("LiteLLM: model rejects temperature; sending the default from now on")
                    continue
                if any(p in s for p in ("rate", "429", "timeout", "timed out", "500", "502", "503", "overloaded")):
                    logger.warning(f"LiteLLM retry {attempt + 1}: {self._sanitize(e)[:120]}")
                    time.sleep(min(2 ** attempt + 0.3 * attempt, 30))
                    continue
                logger.error(f"LiteLLM call failed: {self._sanitize(e)}")
                return f"Error: {self._sanitize(e)}"
        return "Error: retries exhausted"

    def generate_batch(self, prompts: List[str], use_cache: bool = True,
                       batch_size: int = 50) -> List[str]:
        # Fewer threads than ENDPOINT_A: the gateway is shared and the budget is small.
        if not prompts:
            return []
        from concurrent.futures import ThreadPoolExecutor
        workers = int(os.environ.get("LITELLM_THREADS", "8"))
        stage = get_usage_stage()

        def run(p):
            set_usage_stage(stage)
            return self._one(p)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(run, prompts))


# =============================================================================
# Uncertainty Estimation
# =============================================================================

class PerplexityEstimator:
    """Perplexity-based uncertainty estimation using token logprobs."""

    def __init__(self, client: 'EndpointBBatchClient'):
        self.client = client

    def estimate_uncertainty(self, prompt: str, response: str = None) -> Dict[str, Any]:
        """
        Estimate uncertainty using perplexity from token log probabilities.

        Lower perplexity = more confident model = lower uncertainty
        Higher perplexity = less confident model = higher uncertainty

        Returns:
            Dict with uncertainty score (0-1) and perplexity value
        """
        result = self.client.generate_with_logprobs(prompt)
        perplexity = result.get("perplexity")

        if perplexity is None:
            # Fallback to heuristic if logprobs unavailable
            return {
                "uncertainty": 0.5,
                "perplexity": None,
                "method": "fallback_heuristic"
            }

        # Convert perplexity to uncertainty score (0-1)
        # Perplexity ranges: ~1 (very confident) to potentially >100 (very uncertain)
        # Using sigmoid-like mapping: uncertainty = 1 - 1/(1 + log(perplexity)/k)
        # where k controls the sensitivity
        k = 3.0  # Tunable parameter

        if perplexity <= 1.0:
            uncertainty = 0.0
        else:
            # log(perplexity) maps perplexity to a more linear scale
            log_ppl = np.log(perplexity)
            uncertainty = log_ppl / (log_ppl + k)
            uncertainty = float(min(1.0, max(0.0, uncertainty)))

        return {
            "uncertainty": uncertainty,
            "perplexity": float(perplexity),
            "method": "perplexity",
            "n_tokens": len(result.get("logprobs", []))
        }


class PTrueEstimator:
    """
    P(True) uncertainty estimation.

    Asks the model to verify its own prediction:
    "Is this mapping correct? (A) True (B) False"

    Uses the probability of "True" vs "False" to estimate confidence.
    Based on: https://github.com/IINemo/lm-polygraph
    """

    def __init__(self, client: 'EndpointBBatchClient'):
        self.client = client

    def _build_verification_prompt(
        self,
        source_field: str,
        source_desc: str,
        predicted_target: str,
        target_desc: str,
        source_product: str,
        target_product: str
    ) -> str:
        """Build a verification prompt for P(True) estimation."""
        # Use a direct, constrained format that forces immediate answer
        prompt = f"""Schema mapping verification task.

Source: {source_field} ({source_desc}) from {source_product}
Target: {predicted_target} ({target_desc}) from {target_product}

Is this a correct semantic mapping? Answer only A or B.
A = Yes, correct mapping
B = No, incorrect mapping

Answer:"""
        return prompt

    def estimate_uncertainty(
        self,
        source_field: str,
        source_desc: str,
        predicted_target: str,
        target_desc: str,
        source_product: str = "",
        target_product: str = ""
    ) -> Dict[str, Any]:
        """
        Estimate uncertainty using P(True) method.

        Asks the model if its prediction is correct and uses the
        probability of "True" as confidence.

        Returns:
            Dict with uncertainty score (0-1) and p_true value
        """
        verification_prompt = self._build_verification_prompt(
            source_field, source_desc,
            predicted_target, target_desc,
            source_product, target_product
        )

        # Format for the model
        messages = [{"role": "user", "content": verification_prompt}]
        formatted_prompt, _ = format_prompt_for_model(messages, self.client.model_id)

        # Get response with logprobs (limited tokens for faster verification)
        result = self.client.generate_with_logprobs(formatted_prompt, max_tokens=5)

        generated_text = result.get("text", "").strip().upper()
        generated_tokens = result.get("generated_tokens", [])

        # Try to extract P(A) vs P(B) from first token logprobs
        p_true = None  # P(A) = correct
        p_false = None  # P(B) = incorrect

        # Look through first few tokens for A/B answers
        for token_idx, token_data in enumerate(generated_tokens[:3]):
            if not isinstance(token_data, dict):
                continue

            top_tokens = token_data.get("top_tokens", [])

            # Search top_tokens for A or B
            for tok in top_tokens:
                tok_text = tok.get("text", "").strip().upper()
                # Remove special characters like Ġ (space marker)
                tok_text_clean = tok_text.replace("Ġ", "").replace("▁", "").strip()
                tok_logprob = tok.get("logprob", -10)

                if tok_text_clean in ["A", "(A)", "A)", "YES", "CORRECT", "TRUE"]:
                    if p_true is None or np.exp(tok_logprob) > p_true:
                        p_true = np.exp(tok_logprob)
                elif tok_text_clean in ["B", "(B)", "B)", "NO", "INCORRECT", "FALSE"]:
                    if p_false is None or np.exp(tok_logprob) > p_false:
                        p_false = np.exp(tok_logprob)

            # If we found both, we can compute relative probabilities
            if p_true is not None and p_false is not None:
                break

        # If we found probabilities, normalize them
        if p_true is not None and p_false is not None:
            total = p_true + p_false
            if total > 0:
                p_true = p_true / total
                p_false = p_false / total

        # Fallback: parse the text response if no logprobs found
        if p_true is None and p_false is None:
            # Check the actual generated text
            if "A" in generated_text[:10] or "YES" in generated_text[:10] or "CORRECT" in generated_text[:20]:
                p_true = 0.7  # Default confidence for text-based
            elif "B" in generated_text[:10] or "NO" in generated_text[:10] or "INCORRECT" in generated_text[:20]:
                p_false = 0.7
            else:
                # Couldn't determine - return moderate uncertainty
                return {
                    "uncertainty": 0.5,
                    "p_true": None,
                    "p_false": None,
                    "response": generated_text[:100],
                    "method": "p_true_fallback"
                }

        # Calculate uncertainty from P(True)
        if p_true is not None:
            # High P(True) = low uncertainty
            uncertainty = 1.0 - p_true
        elif p_false is not None:
            # High P(False) = high uncertainty (model thinks it's wrong)
            uncertainty = p_false
        else:
            uncertainty = 0.5

        uncertainty = float(min(1.0, max(0.0, uncertainty)))

        return {
            "uncertainty": uncertainty,
            "p_true": float(p_true) if p_true is not None else None,
            "p_false": float(p_false) if p_false is not None else None,
            "response": generated_text,
            "method": "p_true"
        }

    def estimate_uncertainty_batch(
        self,
        fields: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Batch estimate uncertainty for multiple fields.

        Uses batch generation for massive speedup (10-50x faster).

        Args:
            fields: List of dicts with keys:
                - source_field, source_desc, predicted_target, target_desc
                - source_product, target_product

        Returns:
            List of uncertainty info dicts in same order
        """
        if not fields:
            return []

        # Build all verification prompts
        prompts = []
        for field in fields:
            prompt = self._build_verification_prompt(
                source_field=field.get("source_field", ""),
                source_desc=field.get("source_desc", ""),
                predicted_target=field.get("predicted_target", ""),
                target_desc=field.get("target_desc", ""),
                source_product=field.get("source_product", ""),
                target_product=field.get("target_product", "")
            )
            # Format for model
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt, _ = format_prompt_for_model(messages, self.client.model_id)
            prompts.append(formatted_prompt)

        # Batch generate all responses
        logger.info(f"P(True) batch: verifying {len(prompts)} predictions...")
        responses = self.client.generate_batch(prompts, use_cache=True, batch_size=50)

        # Parse all responses
        results = []
        for response in responses:
            response_text = (response or "").strip().upper()

            # Parse A/B answer
            if "A" in response_text[:10] or "YES" in response_text[:15] or "CORRECT" in response_text[:20]:
                p_true = 0.9  # High confidence for clear A answer
                uncertainty = 0.1
            elif "B" in response_text[:10] or "NO" in response_text[:15] or "INCORRECT" in response_text[:20]:
                p_true = 0.1  # Low confidence - model says it's wrong
                uncertainty = 0.9
            else:
                p_true = 0.5  # Unclear response
                uncertainty = 0.5

            results.append({
                "uncertainty": uncertainty,
                "p_true": p_true,
                "p_false": 1.0 - p_true,
                "response": response_text[:50],
                "method": "p_true_batch"
            })

        return results


class SemanticEntropyEstimator:
    """
    Semantic Entropy uncertainty estimation for schema mapping.

    For discrete field mappings, we simplify semantic entropy to exact-match clustering:
    1. Generate N samples of rankings (with temperature > 0)
    2. Extract top-1 prediction from each sample
    3. Cluster by exact match (field IDs are discrete)
    4. Compute entropy: H = -Σ p(class) * log(p(class))

    High entropy = samples disagree = uncertain
    Low entropy = samples agree = confident

    Based on: Kuhn et al. "Semantic Uncertainty" (2023)
    """

    def __init__(self, client: 'EndpointBBatchClient', n_samples: int = 5):
        self.client = client
        self.n_samples = n_samples
        self.sampling_temperature = 0.7  # Higher temp for diverse samples

    def estimate_uncertainty(
        self,
        ranking_prompt: str,
        initial_response: str,
        candidates: List[Dict]
    ) -> Dict[str, Any]:
        """
        Estimate uncertainty using semantic entropy over multiple samples.

        Args:
            ranking_prompt: The ranking prompt to sample from
            initial_response: The initial (greedy) response
            candidates: List of candidate fields

        Returns:
            Dict with uncertainty score (0-1), entropy, and sample distribution
        """
        # Parse initial response to get first prediction
        initial_ranking = safe_parse_json_array(initial_response)
        if not initial_ranking or not isinstance(initial_ranking[0], dict):
            return {
                "uncertainty": 0.8,
                "entropy": None,
                "method": "semantic_entropy_parse_error"
            }

        initial_top1 = initial_ranking[0].get("id", "")

        # Collect predictions: start with the initial one
        predictions = [initial_top1]

        # Generate additional samples with temperature
        for _ in range(self.n_samples - 1):
            try:
                sample_response = self._generate_sample(ranking_prompt)
                sample_ranking = safe_parse_json_array(sample_response)

                if sample_ranking and isinstance(sample_ranking[0], dict):
                    sample_top1 = sample_ranking[0].get("id", "")
                    predictions.append(sample_top1)
                else:
                    # Parse failed - add empty prediction
                    predictions.append("")

            except Exception as e:
                logger.warning(f"Semantic entropy sampling error: {e}")
                predictions.append("")

        # Cluster by exact match and compute distribution
        cluster_counts = {}
        for pred in predictions:
            cluster_counts[pred] = cluster_counts.get(pred, 0) + 1

        # Compute entropy
        n_total = len(predictions)
        entropy = 0.0
        distribution = {}

        for pred, count in cluster_counts.items():
            p = count / n_total
            distribution[pred] = p
            if p > 0:
                entropy -= p * np.log(p)

        # Normalize entropy to [0, 1] range
        # Max entropy = log(n_samples) when all samples are different
        max_entropy = np.log(self.n_samples)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Convert to uncertainty score
        # High entropy = high uncertainty
        uncertainty = float(min(1.0, max(0.0, normalized_entropy)))

        # Calculate agreement rate (what fraction agree with top prediction)
        top_pred = max(cluster_counts.keys(), key=lambda k: cluster_counts[k])
        agreement_rate = cluster_counts[top_pred] / n_total

        return {
            "uncertainty": uncertainty,
            "entropy": float(entropy),
            "normalized_entropy": float(normalized_entropy),
            "agreement_rate": float(agreement_rate),
            "n_samples": self.n_samples,
            "n_unique": len(cluster_counts),
            "top_prediction": top_pred,
            "distribution": distribution,
            "method": "semantic_entropy"
        }

    def _generate_sample(self, prompt: str) -> str:
        """Generate a single sample with temperature for diversity."""
        # Use the client's model but with higher temperature
        api_key = os.getenv("ENDPOINT_B_API_KEY") or os.getenv("ENDPOINT_B_APIKEY")
        project_id = os.getenv("ENDPOINT_B_PROJECT_ID")
        url = os.getenv("ENDPOINT_B_URL", "https://ENDPOINT_B")

        if not api_key or not project_id:
            raise RuntimeError("ENDPOINT_B_API_KEY and ENDPOINT_B_PROJECT_ID required")

        _, stop_sequences = format_prompt_for_model([], self.client.model_id)

        # Create model with sampling temperature
        model = ModelInference(
            model_id=self.client.model_id,
            params={
                GenParams.MAX_NEW_TOKENS: self.client.config.max_tokens,
                GenParams.TEMPERATURE: self.sampling_temperature,
                GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,  # Use sampling, not greedy
                GenParams.STOP_SEQUENCES: stop_sequences,
                GenParams.TOP_K: 50,
                GenParams.TOP_P: 0.95,
            },
            credentials=Credentials(url=url, api_key=api_key),
            project_id=project_id
        )

        response = model.generate_text(prompt=prompt)
        if isinstance(response, list):
            response = response[0]
        return response

    def _generate_samples_batch(self, prompts: List[str]) -> List[str]:
        """Generate samples with temperature in batch for massive speedup."""
        if not prompts:
            return []

        api_key = os.getenv("ENDPOINT_B_API_KEY") or os.getenv("ENDPOINT_B_APIKEY")
        project_id = os.getenv("ENDPOINT_B_PROJECT_ID")
        url = os.getenv("ENDPOINT_B_URL", "https://ENDPOINT_B")

        if not api_key or not project_id:
            raise RuntimeError("ENDPOINT_B_API_KEY and ENDPOINT_B_PROJECT_ID required")

        _, stop_sequences = format_prompt_for_model([], self.client.model_id)

        # Create model with sampling temperature
        model = ModelInference(
            model_id=self.client.model_id,
            params={
                GenParams.MAX_NEW_TOKENS: self.client.config.max_tokens,
                GenParams.TEMPERATURE: self.sampling_temperature,
                GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
                GenParams.STOP_SEQUENCES: stop_sequences,
                GenParams.TOP_K: 50,
                GenParams.TOP_P: 0.95,
            },
            credentials=Credentials(url=url, api_key=api_key),
            project_id=project_id
        )

        # Batch generate with concurrency
        try:
            responses = model.generate_text(prompt=prompts, concurrency_limit=10)
            if isinstance(responses, str):
                responses = [responses]
            elif not isinstance(responses, list):
                responses = list(responses)
            return responses
        except Exception as e:
            logger.error(f"Batch sampling failed: {e}")
            # Fallback to sequential
            return [self._generate_sample(p) for p in prompts]

    def estimate_uncertainty_batch(
        self,
        fields: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Batch estimate uncertainty using semantic entropy.

        For each field, generates N samples and computes entropy.
        Uses batch generation for massive speedup.

        Args:
            fields: List of dicts with keys:
                - ranking_prompt: The formatted ranking prompt
                - initial_response: The greedy response
                - candidates: List of candidate fields

        Returns:
            List of uncertainty info dicts in same order
        """
        if not fields:
            return []

        n_fields = len(fields)
        logger.info(f"Semantic entropy batch: sampling {n_fields} fields x {self.n_samples} samples...")

        # Parse all initial responses first
        initial_top1s = []
        for field in fields:
            initial_response = field.get("initial_response", "")
            initial_ranking = safe_parse_json_array(initial_response)
            if initial_ranking and isinstance(initial_ranking[0], dict):
                initial_top1s.append(initial_ranking[0].get("id", ""))
            else:
                initial_top1s.append("")

        # Build all sampling prompts: (n_samples - 1) samples per field
        # (we already have the initial greedy response)
        all_prompts = []
        prompt_to_field_idx = []  # Track which field each prompt belongs to

        for field_idx, field in enumerate(fields):
            prompt = field.get("ranking_prompt", "")
            for _ in range(self.n_samples - 1):
                all_prompts.append(prompt)
                prompt_to_field_idx.append(field_idx)

        # Batch generate all samples
        if all_prompts:
            all_responses = self._generate_samples_batch(all_prompts)
        else:
            all_responses = []

        # Organize responses by field
        field_samples = {i: [initial_top1s[i]] for i in range(n_fields)}

        for resp_idx, response in enumerate(all_responses):
            field_idx = prompt_to_field_idx[resp_idx]
            sample_ranking = safe_parse_json_array(response)
            if sample_ranking and isinstance(sample_ranking[0], dict):
                field_samples[field_idx].append(sample_ranking[0].get("id", ""))
            else:
                field_samples[field_idx].append("")

        # Compute entropy for each field
        results = []
        for field_idx in range(n_fields):
            predictions = field_samples[field_idx]

            # Cluster by exact match
            cluster_counts = {}
            for pred in predictions:
                cluster_counts[pred] = cluster_counts.get(pred, 0) + 1

            # Compute entropy
            n_total = len(predictions)
            entropy = 0.0
            distribution = {}

            for pred, count in cluster_counts.items():
                p = count / n_total
                distribution[pred] = p
                if p > 0:
                    entropy -= p * np.log(p)

            # Normalize
            max_entropy = np.log(self.n_samples)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            uncertainty = float(min(1.0, max(0.0, normalized_entropy)))

            # Agreement rate
            top_pred = max(cluster_counts.keys(), key=lambda k: cluster_counts[k])
            agreement_rate = cluster_counts[top_pred] / n_total

            results.append({
                "uncertainty": uncertainty,
                "entropy": float(entropy),
                "normalized_entropy": float(normalized_entropy),
                "agreement_rate": float(agreement_rate),
                "n_samples": self.n_samples,
                "n_unique": len(cluster_counts),
                "top_prediction": top_pred,
                "distribution": distribution,
                "method": "semantic_entropy_batch"
            })

        return results


class SIMBASamplingEstimator:
    """
    SIMBA-style sampling consistency UQ for schema mapping.

    Implements the Consistency Hypothesis (Xiao et al., 2025): run the MAP
    phase k times at temperature > 0, measure how often the top-1 prediction
    changes across runs. High inconsistency = genuine uncertainty.

    Disagreement rate = 1 - pairwise_consistency, where:
        consistency = |{(i,j): i<j, pred_i == pred_j}| / C(k,2)

    This differs from the heuristic entropy trigger (single-pass, uncalibrated
    scores) by actually sampling the model's distribution, detecting cases
    where the model is confidently wrong at temperature=0 but inconsistent
    when sampled.
    """

    def __init__(self, client: 'EndpointBBatchClient', n_samples: int = 3):
        self.client = client
        self.n_samples = n_samples
        self.sampling_temperature = 0.7

    def _generate_samples_batch(self, prompts: List[str]) -> List[str]:
        """Generate samples with temperature in batch."""
        if not prompts:
            return []

        api_key = os.getenv("ENDPOINT_B_API_KEY") or os.getenv("ENDPOINT_B_APIKEY")
        project_id = os.getenv("ENDPOINT_B_PROJECT_ID")
        url = os.getenv("ENDPOINT_B_URL", "https://ENDPOINT_B")

        if not api_key or not project_id:
            raise RuntimeError("ENDPOINT_B_API_KEY and ENDPOINT_B_PROJECT_ID required")

        _, stop_sequences = format_prompt_for_model([], self.client.model_id)

        model = ModelInference(
            model_id=self.client.model_id,
            params={
                GenParams.MAX_NEW_TOKENS: self.client.config.max_tokens,
                GenParams.TEMPERATURE: self.sampling_temperature,
                GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
                GenParams.STOP_SEQUENCES: stop_sequences,
                GenParams.TOP_K: 50,
                GenParams.TOP_P: 0.95,
            },
            credentials=Credentials(url=url, api_key=api_key),
            project_id=project_id
        )

        try:
            responses = model.generate_text(prompt=prompts, concurrency_limit=10)
            if isinstance(responses, str):
                responses = [responses]
            elif not isinstance(responses, list):
                responses = list(responses)
            return responses
        except Exception as e:
            logger.error(f"SIMBA batch sampling failed: {e}")
            return [""] * len(prompts)

    def estimate_uncertainty_batch(
        self,
        fields: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Batch estimate uncertainty using SIMBA pairwise consistency.

        Args:
            fields: List of dicts with keys:
                - ranking_prompt: The formatted ranking prompt
                - initial_response: The greedy response
                - candidates: List of candidate fields

        Returns:
            List of uncertainty info dicts with disagreement_rate as UQ signal.
        """
        if not fields:
            return []

        n_fields = len(fields)
        logger.info(f"SIMBA sampling: {n_fields} fields x {self.n_samples} samples...")

        # Parse initial (greedy) responses
        initial_top1s = []
        for field in fields:
            initial_response = field.get("initial_response", "")
            initial_ranking = safe_parse_json_array(initial_response)
            if initial_ranking and isinstance(initial_ranking[0], dict):
                initial_top1s.append(initial_ranking[0].get("id", ""))
            else:
                initial_top1s.append("")

        # Build sampling prompts: (n_samples - 1) per field
        all_prompts = []
        prompt_to_field_idx = []
        for field_idx, field in enumerate(fields):
            prompt = field.get("ranking_prompt", "")
            for _ in range(self.n_samples - 1):
                all_prompts.append(prompt)
                prompt_to_field_idx.append(field_idx)

        # Batch generate
        if all_prompts:
            all_responses = self._generate_samples_batch(all_prompts)
        else:
            all_responses = []

        # Organize: field_idx -> list of top-1 predictions (including greedy)
        field_samples = {i: [initial_top1s[i]] for i in range(n_fields)}
        for resp_idx, response in enumerate(all_responses):
            field_idx = prompt_to_field_idx[resp_idx]
            sample_ranking = safe_parse_json_array(response)
            if sample_ranking and isinstance(sample_ranking[0], dict):
                field_samples[field_idx].append(sample_ranking[0].get("id", ""))
            else:
                field_samples[field_idx].append("")

        # Compute pairwise consistency for each field
        from itertools import combinations
        results = []
        for field_idx in range(n_fields):
            predictions = field_samples[field_idx]
            n = len(predictions)

            # Pairwise consistency: fraction of pairs that agree
            pairs = list(combinations(range(n), 2))
            if pairs:
                agreeing = sum(1 for i, j in pairs if predictions[i] == predictions[j])
                consistency = agreeing / len(pairs)
            else:
                consistency = 1.0

            disagreement_rate = 1.0 - consistency

            # Majority vote prediction
            from collections import Counter
            vote_counts = Counter(p for p in predictions if p)
            majority_pred = vote_counts.most_common(1)[0][0] if vote_counts else initial_top1s[field_idx]

            # Distribution for diagnostics
            distribution = {}
            for pred in predictions:
                distribution[pred] = distribution.get(pred, 0) + 1
            for k in distribution:
                distribution[k] = distribution[k] / n

            results.append({
                "uncertainty": disagreement_rate,
                "consistency": float(consistency),
                "disagreement_rate": float(disagreement_rate),
                "n_samples": self.n_samples,
                "n_unique": len(set(predictions)),
                "majority_prediction": majority_pred,
                "greedy_prediction": initial_top1s[field_idx],
                "greedy_matches_majority": initial_top1s[field_idx] == majority_pred,
                "distribution": distribution,
                "predictions": predictions,
                "method": "simba_sampling"
            })

        return results


class LMPolygraphSamplingEstimator:
    """Black-box sampling UQ using the *real* lm-polygraph estimator classes.

    Draws K samples of the ranking (SAMPLE decoding), extracts the top-1 target
    id from each, and computes the sampling/consistency family from lm-polygraph:
    LexicalSimilarity, NumSemSets, DegMat, Eccentricity, EigValLaplacian. Because
    the mapping answer is a discrete target id, "semantic equivalence" is exact-id
    equality, so the semantic similarity matrix lm-polygraph expects is the K x K
    id-agreement matrix. One K-sample pass yields every signal, each stored under
    ``lmp_*`` so the offline scorer can evaluate them independently.

    endpoint_b exposes no token logprobs, so info-based methods (perplexity, semantic
    entropy w/ likelihoods) are out; these text-only methods are API-feasible.
    """

    def __init__(self, client: 'EndpointBBatchClient', n_samples: int = 5):
        self.client = client
        self.n_samples = n_samples
        self.sampling_temperature = 0.7
        self._estimators = None

    def _load_estimators(self):
        if self._estimators is not None:
            return self._estimators
        import lm_polygraph.estimators as E
        self._estimators = {
            "lmp_lexsim": E.LexicalSimilarity(),
            "lmp_numsemsets": E.NumSemSets(),
            "lmp_degmat": E.DegMat(affinity="entail"),
            "lmp_ecc": E.Eccentricity(affinity="entail"),
            "lmp_eigval": E.EigValLaplacian(affinity="entail"),
        }
        return self._estimators

    def _generate_samples_batch(self, prompts: List[str]) -> List[str]:
        """Generate temperature-sampled completions in one batched call."""
        if not prompts:
            return []
        api_key = os.getenv("ENDPOINT_B_API_KEY") or os.getenv("ENDPOINT_B_APIKEY")
        project_id = os.getenv("ENDPOINT_B_PROJECT_ID")
        url = os.getenv("ENDPOINT_B_URL", "https://ENDPOINT_B")
        if not api_key or not project_id:
            raise RuntimeError("ENDPOINT_B_API_KEY and ENDPOINT_B_PROJECT_ID required")
        _, stop_sequences = format_prompt_for_model([], self.client.model_id)
        model = ModelInference(
            model_id=self.client.model_id,
            params={
                GenParams.MAX_NEW_TOKENS: self.client.config.max_tokens,
                GenParams.TEMPERATURE: self.sampling_temperature,
                GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
                GenParams.STOP_SEQUENCES: stop_sequences,
                GenParams.TOP_K: 50,
                GenParams.TOP_P: 0.95,
            },
            credentials=Credentials(url=url, api_key=api_key),
            project_id=project_id,
        )
        try:
            responses = model.generate_text(prompt=prompts, concurrency_limit=10)
            if isinstance(responses, str):
                responses = [responses]
            return list(responses)
        except Exception as e:
            logger.error(f"lm-polygraph sampling failed: {e}")
            return [""] * len(prompts)

    def estimate_uncertainty_batch(self, fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """fields: dicts with ranking_prompt, initial_response, candidates."""
        if not fields:
            return []
        n_fields = len(fields)
        logger.info(f"lm-polygraph sampling: {n_fields} fields x {self.n_samples} samples...")

        # greedy top-1 is sample 0
        initial_top1s = []
        for field in fields:
            r = safe_parse_json_array(field.get("initial_response", ""))
            initial_top1s.append(r[0].get("id", "") if r and isinstance(r[0], dict) else "")

        all_prompts, prompt_to_field = [], []
        for fi, field in enumerate(fields):
            for _ in range(self.n_samples - 1):
                all_prompts.append(field.get("ranking_prompt", ""))
                prompt_to_field.append(fi)
        all_responses = self._generate_samples_batch(all_prompts) if all_prompts else []

        field_samples = {i: [initial_top1s[i]] for i in range(n_fields)}
        for ri, resp in enumerate(all_responses):
            fi = prompt_to_field[ri]
            r = safe_parse_json_array(resp)
            field_samples[fi].append(r[0].get("id", "") if r and isinstance(r[0], dict) else "")

        # Build lm-polygraph stats: sample_texts + exact-id agreement matrices.
        sample_texts, ent, con = [], [], []
        for fi in range(n_fields):
            preds = field_samples[fi]
            K = len(preds)
            M = np.array([[1.0 if preds[i] == preds[j] else 0.0 for j in range(K)] for i in range(K)])
            sample_texts.append(preds); ent.append(M); con.append(1.0 - M)
        stats = {
            "sample_texts": sample_texts,
            "semantic_matrix_entail": np.array(ent),
            "semantic_matrix_contra": np.array(con),
            "semantic_matrix_classes": None,
        }

        # Run the real lm-polygraph estimators (batched over all fields).
        est_vals = {}
        try:
            for name, est in self._load_estimators().items():
                try:
                    est_vals[name] = np.asarray(est(stats), dtype=float)
                except Exception as e:
                    logger.warning(f"lm-polygraph estimator {name} failed: {e}")
                    est_vals[name] = np.full(n_fields, np.nan)
        except Exception as e:
            logger.error(f"lm-polygraph estimators unavailable ({e}); falling back to consistency only")

        results = []
        for fi in range(n_fields):
            preds = field_samples[fi]
            n = len(preds)
            counts = Counter(p for p in preds if p)
            modal, modal_n = (counts.most_common(1)[0] if counts else (initial_top1s[fi], 0))
            disagreement = 1.0 - (modal_n / n if n else 1.0)
            # discrete semantic entropy over the id distribution (no logprobs needed)
            probs = np.array([c / n for c in counts.values()]) if counts else np.array([1.0])
            sem_entropy = float(-(probs * np.log(probs + 1e-12)).sum())
            info = {
                "uncertainty": float(disagreement),   # primary: gate/mean-compatible
                "method": "lmp_sampling",
                "n_samples": self.n_samples,
                "n_unique": len(set(preds)),
                "majority_prediction": modal,
                "greedy_prediction": initial_top1s[fi],
                "lmp_sem_entropy": sem_entropy,
                "lmp_disagreement": float(disagreement),
                "distribution": {k: v / n for k, v in counts.items()},
            }
            for name, arr in est_vals.items():
                info[name] = float(arr[fi]) if fi < len(arr) else float("nan")
            results.append(info)
        return results


class ArithmeticAggregationEstimator:
    """
    Arithmetic aggregation (arith-agg) uncertainty estimation.

    Based on SIMBA framework (Xiao et al., 2025):
    - Computes arithmetic mean of top-k similarity/confidence scores
    - High scores = high confidence = low uncertainty
    - Does not require multiple LLM samples (fast)
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def estimate_uncertainty(
        self,
        response: str,
        candidates: List[Dict]
    ) -> Dict[str, Any]:
        """
        Estimate uncertainty using arithmetic mean of top-k scores.

        Args:
            response: The ranking response from LLM
            candidates: List of candidate fields

        Returns:
            Dict with uncertainty score (0-1) and metrics
        """
        ranking = safe_parse_json_array(response)
        if not ranking:
            return {"uncertainty": 0.8, "method": "arith_agg_parse_error"}

        # Extract scores from ranking
        scores = []
        for item in ranking[:self.top_k]:
            if isinstance(item, dict):
                score = item.get("score", 0.5)
                try:
                    scores.append(float(score))
                except (ValueError, TypeError):
                    scores.append(0.5)

        if not scores:
            return {"uncertainty": 0.8, "method": "arith_agg_no_scores"}

        # Compute arithmetic mean
        arith_mean = np.mean(scores)

        # Convert to uncertainty: high score = low uncertainty
        # Scores are typically in [0, 1] range where higher = better match
        # But some models use distance scores where lower = better
        # We'll assume higher = better confidence, so uncertainty = 1 - score

        # Normalize scores to [0, 1] if needed
        if max(scores) > 1.0:
            # Likely distance scores, normalize and invert
            max_score = max(scores)
            normalized_scores = [1.0 - (s / max_score) for s in scores]
            arith_mean = np.mean(normalized_scores)

        # Uncertainty is inverse of confidence
        uncertainty = float(max(0.0, min(1.0, 1.0 - arith_mean)))

        # Also compute score spread (std) as secondary metric
        score_std = float(np.std(scores)) if len(scores) > 1 else 0.0

        return {
            "uncertainty": uncertainty,
            "arith_mean": float(arith_mean),
            "score_std": score_std,
            "n_scores": len(scores),
            "top_k": self.top_k,
            "method": "arith_agg"
        }


class HarmonicAggregationEstimator:
    """
    Harmonic aggregation (harm-agg) uncertainty estimation.

    Based on SIMBA framework (Xiao et al., 2025):
    - Computes harmonic mean of top-k similarity/confidence scores
    - Penalizes low scores more heavily than arithmetic mean
    - More sensitive to outliers/low-confidence predictions
    - Does not require multiple LLM samples (fast)
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def estimate_uncertainty(
        self,
        response: str,
        candidates: List[Dict]
    ) -> Dict[str, Any]:
        """
        Estimate uncertainty using harmonic mean of top-k scores.

        Args:
            response: The ranking response from LLM
            candidates: List of candidate fields

        Returns:
            Dict with uncertainty score (0-1) and metrics
        """
        ranking = safe_parse_json_array(response)
        if not ranking:
            return {"uncertainty": 0.8, "method": "harm_agg_parse_error"}

        # Extract scores from ranking
        scores = []
        for item in ranking[:self.top_k]:
            if isinstance(item, dict):
                score = item.get("score", 0.5)
                try:
                    scores.append(float(score))
                except (ValueError, TypeError):
                    scores.append(0.5)

        if not scores:
            return {"uncertainty": 0.8, "method": "harm_agg_no_scores"}

        # Normalize scores to [0, 1] if needed
        if max(scores) > 1.0:
            # Likely distance scores, normalize and invert
            max_score = max(scores)
            scores = [1.0 - (s / max_score) for s in scores]

        # Ensure no zero scores for harmonic mean (add small epsilon)
        epsilon = 1e-6
        scores = [max(s, epsilon) for s in scores]

        # Compute harmonic mean: n / sum(1/x_i)
        harm_mean = len(scores) / sum(1.0 / s for s in scores)

        # Also compute arithmetic mean for comparison
        arith_mean = np.mean(scores)

        # Uncertainty is inverse of confidence
        uncertainty = float(max(0.0, min(1.0, 1.0 - harm_mean)))

        # Score spread
        score_std = float(np.std(scores)) if len(scores) > 1 else 0.0

        return {
            "uncertainty": uncertainty,
            "harm_mean": float(harm_mean),
            "arith_mean": float(arith_mean),
            "harm_arith_gap": float(arith_mean - harm_mean),  # Larger gap = more variance
            "score_std": score_std,
            "n_scores": len(scores),
            "top_k": self.top_k,
            "method": "harm_agg"
        }


class UncertaintyEstimator:
    """Sampling-based uncertainty estimation for rankings."""

    def __init__(self, client: EndpointBBatchClient, n_samples: int = 3):
        self.client = client
        self.n_samples = n_samples

    def estimate_ranking_uncertainty(
        self,
        prompt: str,
        primary_response: str,
        candidates: List[Dict]
    ) -> Dict[str, Any]:
        """
        Estimate uncertainty for a ranking task by checking consistency across samples.

        Returns:
            Dict with uncertainty score and metrics
        """
        # Parse primary response to get top prediction
        primary_ranking = safe_parse_json_array(primary_response)
        if not primary_ranking:
            return {"uncertainty": 0.8, "method": "parse_failure"}

        # Validate that ranking contains dicts with 'id' field
        if not all(isinstance(r, dict) and 'id' in r for r in primary_ranking):
            return {"uncertainty": 0.8, "method": "invalid_format"}

        primary_top = primary_ranking[0].get("id", "") if primary_ranking else ""

        def _safe_score(v, default=0.5):
            try:
                return float(v) if v is not None else default
            except (ValueError, TypeError):
                return default

        primary_scores = {r.get("id", ""): _safe_score(r.get("score", 0.5)) for r in primary_ranking}

        # Generate additional samples
        agreement_count = 1  # Primary response agrees with itself
        score_variances = []

        for _ in range(self.n_samples - 1):
            try:
                alt_prompt = prompt + "\n\nProvide an alternative ranking considering different interpretations."
                sample_response = self.client.generate_single(alt_prompt)
                sample_ranking = safe_parse_json_array(sample_response)

                if sample_ranking and all(isinstance(r, dict) and 'id' in r for r in sample_ranking):
                    sample_top = sample_ranking[0].get("id", "") if sample_ranking else ""

                    # Check top-1 agreement
                    if sample_top == primary_top:
                        agreement_count += 1

                    # Calculate score variance for overlapping candidates
                    sample_scores = {r.get("id", ""): _safe_score(r.get("score", 0.5)) for r in sample_ranking}
                    for cid in primary_scores:
                        if cid in sample_scores:
                            score_variances.append(abs(primary_scores[cid] - sample_scores[cid]))

            except Exception as e:
                logger.warning(f"Sampling error: {e}")
                continue

        # Calculate uncertainty metrics
        top1_agreement_rate = agreement_count / self.n_samples
        avg_score_variance = np.mean(score_variances) if score_variances else 0.3

        # Combine into overall uncertainty score
        uncertainty = 1.0 - (0.6 * top1_agreement_rate + 0.4 * (1.0 - min(avg_score_variance * 2, 1.0)))
        uncertainty = max(0.0, min(1.0, uncertainty))

        return {
            "uncertainty": float(uncertainty),
            "top1_agreement": float(top1_agreement_rate),
            "score_variance": float(avg_score_variance),
            "n_samples": self.n_samples,
            "method": "sampling_consistency"
        }


def estimate_uncertainty_from_response(response: str, candidates: List[Dict]) -> float:
    """
    Quick uncertainty estimation from response characteristics (heuristics).
    """
    parsed = safe_parse_json_array(response)

    if not parsed:
        return 0.8  # High uncertainty - couldn't parse

    if len(parsed) == 0:
        return 0.9  # Very high uncertainty - empty response

    if not all(isinstance(r, dict) for r in parsed):
        return 0.8  # High uncertainty - invalid format

    # Check for low confidence in top prediction
    top_score = parsed[0].get("score", 0.5) if parsed else 0.5
    try:
        top_score = float(top_score) if top_score is not None else 0.5
    except (ValueError, TypeError):
        top_score = 0.5

    # Check if reasons are provided
    has_reasons = all(r.get("reason", "").strip() for r in parsed[:3] if isinstance(r, dict))

    # Check score spread
    def safe_float(v, default=0.5):
        try:
            return float(v) if v is not None else default
        except (ValueError, TypeError):
            return default

    scores = [safe_float(r.get("score", 0.5), 0.5) for r in parsed if isinstance(r, dict)]
    score_range = max(scores) - min(scores) if len(scores) > 1 else 0.5

    # Combine heuristics
    uncertainty = 0.0
    uncertainty += 0.3 * (1.0 - top_score)
    uncertainty += 0.2 * (0.0 if has_reasons else 1.0)
    uncertainty += 0.3 * (1.0 - min(score_range * 1.5, 1.0))
    uncertainty += 0.2 * 0.3  # Base uncertainty

    return max(0.0, min(1.0, uncertainty))


# =============================================================================
# Error Detection for Reflection
# =============================================================================

def detect_mapping_errors(
    source_field: str,
    source_product: str,
    target_product: str,
    initial_ranking: List[Dict],
    candidates: List[Dict]
) -> List[str]:
    """
    Analyze the initial ranking to detect specific error patterns.

    Returns a list of error messages describing potential problems.
    """
    errors = []

    if not initial_ranking or not isinstance(initial_ranking[0], dict):
        return ["Could not parse initial ranking"]

    top_pred = initial_ranking[0].get("id", "")
    top_reason = initial_ranking[0].get("reason", "")

    # Extract field name components
    source_parts = source_field.lower().replace(".", "_").split("_")
    top_pred_parts = top_pred.lower().replace(".", "_").split("_")

    # Error 1: Predicted source-side field instead of target-side
    if top_pred.lower().startswith(source_product.lower() + "_"):
        errors.append(
            f"ERROR: Top prediction '{top_pred}' is from SOURCE product '{source_product}', "
            f"but should be from TARGET product '{target_product}'. "
            "The model matched by name similarity instead of finding the semantic equivalent in the target schema."
        )

    # Error 2: Foreign key mapped to wrong table
    # Foreign keys typically end in _id and reference another table's primary key
    if source_field.endswith("_id") or ".id" in source_field.lower():
        # Check if this looks like a foreign key (contains another table name)
        # e.g., concert_stadium_id should map to stadium's primary key, not concert's
        for part in source_parts:
            if part not in ["id", source_product.lower()] and len(part) > 2:
                # This might be a referenced table name
                referenced_table = part
                # Check if top prediction is in the wrong table
                if top_pred_parts and top_pred_parts[0] != referenced_table:
                    if referenced_table in source_field.lower() and referenced_table not in top_pred.lower():
                        errors.append(
                            f"ERROR: Source field '{source_field}' appears to be a foreign key referencing '{referenced_table}'. "
                            f"The top prediction '{top_pred}' may be mapping to the wrong table. "
                            f"Foreign keys should map to the PRIMARY KEY of the referenced table (e.g., {referenced_table}_id or {referenced_table}.id)."
                        )
                        break

    # Error 3: Name similarity without semantic match
    # Check if top prediction has high lexical overlap but might be semantically wrong
    source_words = set(source_parts) - {"id", "input", "type", "filter", source_product.lower()}
    pred_words = set(top_pred_parts) - {"id", "input", "type", "filter", target_product.lower()}

    overlap = source_words & pred_words
    if overlap and len(overlap) >= len(source_words) * 0.5:
        # High name overlap - check if the reason mentions "name" or "similar"
        reason_lower = top_reason.lower()
        if any(word in reason_lower for word in ["name", "similar", "same", "match", "identical"]):
            errors.append(
                f"WARNING: Top prediction may be based on NAME SIMILARITY rather than semantic meaning. "
                f"Overlapping terms: {overlap}. "
                "Verify that the fields represent the SAME DATA, not just similar names."
            )

    # Error 4: Type mismatch indicators
    # Check for type indicators in field names that don't match
    source_type_hints = []
    pred_type_hints = []

    type_keywords = ["string", "int", "bool", "float", "date", "time", "enum", "list", "array"]
    for kw in type_keywords:
        if kw in source_field.lower():
            source_type_hints.append(kw)
        if kw in top_pred.lower():
            pred_type_hints.append(kw)

    if source_type_hints and pred_type_hints and set(source_type_hints) != set(pred_type_hints):
        errors.append(
            f"WARNING: Possible type mismatch. Source field suggests type '{source_type_hints}', "
            f"but prediction suggests type '{pred_type_hints}'. Verify type compatibility."
        )

    # Error 5: Score-based warning
    top_score = initial_ranking[0].get("score", 0.5)
    try:
        top_score = float(top_score)
    except (ValueError, TypeError):
        top_score = 0.5

    if top_score < 0.6:
        errors.append(
            f"WARNING: Low confidence score ({top_score:.2f}) on top prediction. "
            "The model is uncertain - consider alternative candidates."
        )

    # Error 6: Check if top predictions are too similar (ambiguous)
    if len(initial_ranking) >= 2:
        second_pred = initial_ranking[1]
        if isinstance(second_pred, dict):
            second_score = second_pred.get("score", 0)
            try:
                second_score = float(second_score)
            except (ValueError, TypeError):
                second_score = 0

            if abs(top_score - second_score) < 0.1:
                errors.append(
                    f"WARNING: Top two predictions have similar scores ({top_score:.2f} vs {second_score:.2f}). "
                    f"Ambiguous choice between '{top_pred}' and '{second_pred.get('id', 'unknown')}'. "
                    "Carefully compare semantic meanings."
                )

    return errors


# =============================================================================
# Data Loading
# =============================================================================

def load_property_lookup(lookup_file: str) -> Dict[str, Any]:
    """Load property lookup file."""
    with open(lookup_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_dataset(data_dir: Path, dataset: str, max_samples: Optional[int] = None,
                 eval_file: Optional[Path] = None) -> List[Dict]:
    """Load evaluation dataset."""
    # Use custom eval file if provided
    if eval_file:
        filepath = Path(eval_file)
        if not filepath.exists():
            raise FileNotFoundError(f"Eval file not found: {eval_file}")
    else:
        dataset_lower = dataset.lower()

        # Try different file patterns
        patterns = [
            f"{dataset_lower}_eval_dataset.jsonl",
            f"{dataset_lower}_dataset.jsonl",
            f"{dataset_lower}_coverage_eval_dataset.jsonl",
            f"{dataset}_dataset.jsonl",
        ]

        filepath = None
        for pattern in patterns:
            candidate = data_dir / pattern
            if candidate.exists():
                filepath = candidate
                break

        if not filepath:
            raise FileNotFoundError(f"No dataset file found for {dataset} in {data_dir}")

    samples = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if line.strip():
                data = json.loads(line)
                sample = extract_sample_data(data, dataset, i)
                if sample:
                    samples.append(sample)

    logger.info(f"Loaded {len(samples)} samples from {filepath}")
    return samples


def extract_sample_data(data: Dict[str, Any], dataset: str, index: int) -> Optional[Dict[str, Any]]:
    """Extract sample data from JSON object."""
    # Handle unified format with result.properties
    if "result" in data and "properties" in data["result"]:
        result = data["result"]
        source_schema = result.get("name", "")
        properties = result.get("properties", {})

        ground_truth_mapping = {}
        expected_target = ""
        for src_field, mappings in properties.items():
            if mappings and len(mappings) > 0:
                ground_truth_mapping[src_field] = mappings[0]["mapProperty"]
                expected_target = mappings[0].get("mapTable", "")

        sample = {
            "sample_id": data.get("id", f"{dataset.lower()}_{index:03d}"),
            "utterance": data.get("mapping_utterance", data.get("question", "")),
            "expected_source": source_schema,
            "expected_target": expected_target,
            "ground_truth_mapping": ground_truth_mapping,
        }
        # Carry through database field for multi-database datasets (Spider)
        if "database" in data:
            sample["database"] = data["database"]
            sample.setdefault("metadata", {})["database"] = data["database"]
        return sample

    # Handle ToolBench and similar formats
    if "ground_truth_mapping" in data and "expected_target" in data:
        return {
            "sample_id": data.get("sample_id", f"{dataset.lower()}_{index:03d}"),
            "utterance": data.get("utterance", ""),
            "expected_source": data.get("expected_source", ""),
            "expected_target": data.get("expected_target"),
            "ground_truth_mapping": data.get("ground_truth_mapping", {}),
        }

    # Handle coverage dataset format
    if "source_fields" in data and "target_fields" in data:
        source_fields = data.get("source_fields", [])
        target_fields = data.get("target_fields", [])
        ground_truth = {sf: "" for sf in source_fields}

        return {
            "sample_id": data.get("id", f"{dataset.lower()}_{index:03d}"),
            "utterance": data.get("utterance", ""),
            "expected_source": data.get("source_schema", data.get("source_type", "")),
            "expected_target": data.get("target_schema", data.get("target_type", "")),
            "ground_truth_mapping": ground_truth,
            "source_fields": source_fields,
            "target_fields": target_fields,
            "metadata": data.get("metadata", {}),
        }

    # Handle ITOpsGraph format
    if "mapping" in data or "source_schema" in data:
        # Extract target_fields from target_schema.properties if available (OMOP datasets)
        target_fields = []
        target_schema = data.get("target_schema", {})
        if isinstance(target_schema, dict) and "properties" in target_schema:
            props = target_schema.get("properties", [])
            if isinstance(props, list):
                target_fields = [p.get("name", "") for p in props if p.get("name")]

        return {
            "sample_id": data.get("id", f"{dataset.lower()}_{index:03d}"),
            "utterance": data.get("utterance", data.get("mapping_utterance", "")),
            "expected_source": data.get("source_schema", ""),
            "expected_target": target_schema,
            "ground_truth_mapping": data.get("mapping", {}),
            "target_fields": target_fields,
            "metadata": data.get("metadata", {}),
        }

    return None


# =============================================================================
# ChromaDB Indexing
# =============================================================================

def index_chroma(collection_name: str, data: Dict[str, Any], chroma_dir: str):
    """Index documents in ChromaDB."""
    if not HAS_CHROMADB:
        logger.warning("ChromaDB not available, skipping indexing")
        return None

    client = chromadb.PersistentClient(path=chroma_dir)
    documents_list, metadatas_list, ids_list = [], [], []

    for i, (key, value) in enumerate(data.items()):
        desc = value.get("description", "") if isinstance(value, dict) else str(value)
        doc = f"{key}: {desc}"
        metadata = {"source": collection_name, "class": key}

        documents_list.append(doc)
        metadatas_list.append(metadata)
        ids_list.append(f"doc_id_{i}")

    embed_model_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete existing collection if it exists, then create fresh
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    # Use get_or_create to avoid race conditions
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_model_function,
        metadata={"hnsw:space": "cosine"}
    )

    # Add documents in batches
    batch_size = 5000
    for i in range(0, len(documents_list), batch_size):
        end_idx = min(i + batch_size, len(documents_list))
        collection.add(
            documents=documents_list[i:end_idx],
            metadatas=metadatas_list[i:end_idx],
            ids=ids_list[i:end_idx]
        )

    logger.info(f"Indexed {len(documents_list)} documents in collection '{collection_name}'")
    return collection


# =============================================================================
# BM25 Hybrid Retrieval
# =============================================================================

def tokenize_field_name(field_name: str) -> List[str]:
    """
    Tokenize a field name by splitting on _, ., and camelCase boundaries.
    Example: prod_f_Kind.KIND_PROD_G_SBOM -> ["prod_f", "kind", "kind", "prod_g", "sbom"]
    """
    # First split on . and _
    parts = re.split(r'[._]', field_name)

    tokens = []
    for part in parts:
        # Split camelCase: insert space before uppercase letters that follow lowercase
        camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
        # Also split on consecutive uppercase followed by lowercase (e.g., HTTPServer -> HTTP Server)
        camel_split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', camel_split)
        # Split and lowercase
        for token in camel_split.split():
            token_lower = token.lower().strip()
            if token_lower:
                tokens.append(token_lower)

    return tokens


class BM25Retriever:
    """BM25 retriever for lexical matching of field names."""

    def __init__(self, collection_name: str, chroma_dir: str):
        self.collection_name = collection_name
        self.chroma_dir = chroma_dir
        self.index_path = os.path.join(chroma_dir, f"bm25_index_{collection_name}.pkl")
        self.bm25 = None
        self.doc_ids = []
        self.doc_texts = []
        self.tokenized_corpus = []

    def build_index(self, documents: List[Dict[str, Any]]):
        """Build BM25 index from documents."""
        self.doc_ids = []
        self.doc_texts = []
        self.tokenized_corpus = []

        for doc in documents:
            doc_id = doc.get("id", doc.get("doc_id", ""))
            doc_text = doc.get("doc", doc.get("text", doc_id))

            self.doc_ids.append(doc_id)
            self.doc_texts.append(doc_text)
            # Tokenize using field name tokenizer
            tokens = tokenize_field_name(doc_id) + tokenize_field_name(doc_text)
            self.tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save_index()
        logger.info(f"Built BM25 index with {len(self.doc_ids)} documents")

    def _save_index(self):
        """Persist BM25 index to disk."""
        data = {
            "doc_ids": self.doc_ids,
            "doc_texts": self.doc_texts,
            "tokenized_corpus": self.tokenized_corpus
        }
        os.makedirs(os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else ".", exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(data, f)
        logger.debug(f"Saved BM25 index to {self.index_path}")

    def load_index(self) -> bool:
        """Load BM25 index from disk. Returns True if loaded successfully."""
        if not os.path.exists(self.index_path):
            return False

        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
            self.doc_ids = data["doc_ids"]
            self.doc_texts = data["doc_texts"]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.debug(f"Loaded BM25 index from {self.index_path} with {len(self.doc_ids)} documents")
            return True
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for documents matching the query."""
        if self.bm25 is None:
            return []

        # Tokenize query the same way as documents
        query_tokens = tokenize_field_name(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some match
                results.append({
                    "doc_id": self.doc_ids[idx],
                    "id": self.doc_ids[idx],
                    "name": self.doc_ids[idx],
                    "doc": self.doc_texts[idx],
                    "bm25_score": float(scores[idx]),
                    "distance": 1.0 - min(float(scores[idx]) / 10.0, 1.0),  # Convert score to distance-like
                    "meta": {}
                })

        return results


def reciprocal_rank_fusion(semantic_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
    """
    Merge results using Reciprocal Rank Fusion.
    score(doc) = 1/(k + rank_semantic) + 1/(k + rank_bm25)
    """
    # Build score dict
    scores = {}
    doc_info = {}

    # Add semantic scores
    for rank, doc in enumerate(semantic_results):
        doc_id = doc.get("id", doc.get("doc_id", ""))
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        if doc_id not in doc_info:
            doc_info[doc_id] = doc

    # Add BM25 scores
    for rank, doc in enumerate(bm25_results):
        doc_id = doc.get("id", doc.get("doc_id", ""))
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        if doc_id not in doc_info:
            doc_info[doc_id] = doc

    # Sort by fused score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, fused_score in sorted_docs:
        doc = doc_info[doc_id].copy()
        doc["fused_score"] = fused_score
        # Ensure 'distance' field exists (use 1 - fused_score as proxy if missing)
        if "distance" not in doc:
            doc["distance"] = 1.0 - fused_score
        results.append(doc)

    return results


# Global BM25 retriever cache
_bm25_retrievers: Dict[str, BM25Retriever] = {}


def retrieve_candidates(query_text: str, collection_name: str, chroma_dir: str, top_k: int = 10) -> List[Dict]:
    """
    Retrieve candidate matches using hybrid search (ChromaDB + BM25 with Reciprocal Rank Fusion).

    Uses both semantic (ChromaDB with sentence-transformers) and lexical (BM25) retrieval,
    then merges results using Reciprocal Rank Fusion for better coverage.
    """
    if not HAS_CHROMADB:
        return []

    # --- Semantic retrieval via ChromaDB ---
    embed_model = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=chroma_dir)

    try:
        collection = client.get_collection(name=collection_name, embedding_function=embed_model)
    except Exception:
        return []

    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    semantic_candidates = []
    for _id, doc, dist, meta in zip(ids, docs, dists, metas):
        full_name = meta.get('class') if isinstance(meta, dict) else _id
        semantic_candidates.append({
            "doc_id": _id,
            "id": full_name,
            "name": full_name,
            "doc": doc,
            "distance": dist,
            "meta": meta
        })

    # --- BM25 retrieval (if available) ---
    bm25_candidates = []
    if HAS_BM25:
        # Get or create BM25 retriever for this collection
        if collection_name not in _bm25_retrievers:
            bm25_retriever = BM25Retriever(collection_name, chroma_dir)
            if not bm25_retriever.load_index():
                # Build index from ChromaDB collection
                logger.info(f"Building BM25 index for collection '{collection_name}'...")
                all_docs = collection.get(include=["documents", "metadatas"])
                documents = []
                for _id, doc, meta in zip(all_docs["ids"], all_docs["documents"], all_docs["metadatas"]):
                    full_name = meta.get('class') if isinstance(meta, dict) else _id
                    documents.append({
                        "id": full_name,
                        "doc_id": _id,
                        "doc": doc,
                        "meta": meta
                    })
                bm25_retriever.build_index(documents)
            _bm25_retrievers[collection_name] = bm25_retriever

        bm25_retriever = _bm25_retrievers[collection_name]
        bm25_candidates = bm25_retriever.search(query_text, top_k=top_k)

    # --- Merge with Reciprocal Rank Fusion ---
    if bm25_candidates:
        fused_candidates = reciprocal_rank_fusion(semantic_candidates, bm25_candidates)
        # Return top_k after fusion
        return fused_candidates[:top_k]
    else:
        # Fall back to semantic-only if BM25 not available
        return semantic_candidates


# =============================================================================
# Documentation Context Retriever
# =============================================================================

class DocContextRetriever:
    """Retrieves relevant product documentation chunks from ChromaDB.

    Used to enrich MAP phase prompts with documentation context so the LLM
    has more information about what each field/type represents beyond just
    the field name and sparse description.
    """

    def __init__(self, chromadb_path: str, collection_name: str = "itopsgraph_docs",
                 top_k: int = 3):
        self.top_k = top_k
        self._collection = None
        try:
            client = chromadb.PersistentClient(path=chromadb_path)
            self._collection = client.get_collection(name=collection_name)
            logger.info(f"DocContext: loaded collection '{collection_name}' "
                        f"({self._collection.count()} chunks) from {chromadb_path}")
        except Exception as e:
            logger.warning(f"DocContext: failed to load collection '{collection_name}': {e}")

    @property
    def available(self) -> bool:
        return self._collection is not None

    def retrieve(self, query: str, product: str = None) -> List[str]:
        """Retrieve top-K doc chunks relevant to a query.

        Args:
            query: field name, description, or type name to search for
            product: optional product name to filter results

        Returns:
            List of document chunk strings
        """
        if not self.available:
            return []

        try:
            where_filter = {"product": product} if product else None
            results = self._collection.query(
                query_texts=[query],
                n_results=self.top_k,
                where=where_filter,
                include=["documents", "metadatas"],
            )
            chunks = []
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                source = meta.get("source_file", "")
                section = meta.get("section_title", "")
                header = f"[{source} / {section}]" if section else f"[{source}]"
                # Truncate long chunks
                text = doc[:500] if len(doc) > 500 else doc
                chunks.append(f"{header}\n{text}")
            return chunks
        except Exception as e:
            logger.debug(f"DocContext retrieval failed for '{query[:50]}': {e}")
            return []

    def get_field_context(self, field_key: str, product: str = None) -> str:
        """Get documentation context for a specific field.

        Args:
            field_key: e.g., 'prod_c_Application.id' or 'Application.id'
            product: product name for filtering

        Returns:
            Formatted doc context string (empty if nothing found)
        """
        # Build a meaningful query from the field key
        parts = field_key.replace("_", " ").split(".")
        if len(parts) >= 2:
            query = f"{parts[0]} {parts[1]} field"
        else:
            query = field_key.replace("_", " ")

        chunks = self.retrieve(query, product=product)
        if not chunks:
            return ""

        return "\n".join(chunks)

    def get_type_context(self, type_name: str, product: str = None) -> str:
        """Get documentation context for a type/table.

        Args:
            type_name: e.g., 'ApplicationPerspective' or 'AwsEc2Instance'
            product: product name for filtering

        Returns:
            Formatted doc context string
        """
        # Convert CamelCase to readable query
        import re
        readable = re.sub(r'([A-Z])', r' \1', type_name).strip()
        query = f"{readable} type schema"

        chunks = self.retrieve(query, product=product)
        if not chunks:
            return ""

        return "\n".join(chunks)


# =============================================================================
# Main Pipeline
# =============================================================================

class MapReflectRunner:
    """
    Unified Map-Reflect runner with batch processing, uncertainty quantification,
    and self-reflection.
    """

    def __init__(
        self,
        property_lookup_file: str,
        data_dir: str,
        model: str,
        dataset: str,
        config: MapReflectConfig = None
    ):
        self.property_lookup_file = Path(property_lookup_file)
        self.data_dir = Path(data_dir)
        self.model = model
        self.dataset = dataset
        self.config = config or MapReflectConfig()

        # Initialize EndpointB client
        self.client = (EndpointABatchClient(model, self.config) if str(model).startswith("endpoint_a/")
                       else LiteLLMBatchClient(model, self.config) if str(model).startswith("litellm/")
                       else EndpointBBatchClient(model, self.config))

        # Load property lookup
        with open(self.property_lookup_file) as f:
            self.property_lookup = json.load(f)

        # Initialize documentation context retriever
        self.doc_retriever = None
        if self.config.doc_context_chromadb:
            self.doc_retriever = DocContextRetriever(
                chromadb_path=self.config.doc_context_chromadb,
                collection_name=self.config.doc_context_collection,
                top_k=self.config.doc_context_top_k,
            )

        # Build property graph for neighbor expansion
        self.property_graph = None
        if self.config.neighbor_k > 0:
            self.property_graph = build_property_graph(self.property_lookup)

        # Initialize uncertainty estimator based on method
        self.uncertainty_estimator = None
        self.perplexity_estimator = None
        self.p_true_estimator = None
        self.semantic_entropy_estimator = None
        self.arith_agg_estimator = None
        self.harm_agg_estimator = None
        self.simba_estimator = None
        self.lmp_sampling_estimator = None
        if self.config.use_uncertainty:
            if self.config.uq_method == "lmp_sampling":
                self.lmp_sampling_estimator = LMPolygraphSamplingEstimator(
                    self.client, self.config.n_uncertainty_samples
                )
                self.lmp_sampling_estimator.sampling_temperature = self.config.uq_sampling_temperature
                logger.info(f"  UQ Method: lmp_sampling (lm-polygraph LexSim/NumSemSets/DegMat/Ecc/EigVal, n={self.config.n_uncertainty_samples}, temp={self.config.uq_sampling_temperature})")
            elif self.config.uq_method == "perplexity":
                self.perplexity_estimator = PerplexityEstimator(self.client)
                logger.info(f"  UQ Method: perplexity (logprobs-based)")
            elif self.config.uq_method == "p_true":
                self.p_true_estimator = PTrueEstimator(self.client)
                logger.info(f"  UQ Method: p_true (self-verification)")
            elif self.config.uq_method == "semantic_entropy":
                self.semantic_entropy_estimator = SemanticEntropyEstimator(
                    self.client, self.config.n_uncertainty_samples
                )
                logger.info(f"  UQ Method: semantic_entropy (n={self.config.n_uncertainty_samples} samples)")
            elif self.config.uq_method == "arith_agg":
                self.arith_agg_estimator = ArithmeticAggregationEstimator(top_k=self.config.top_k)
                logger.info(f"  UQ Method: arith_agg (arithmetic aggregation, top_k={self.config.top_k})")
            elif self.config.uq_method == "harm_agg":
                self.harm_agg_estimator = HarmonicAggregationEstimator(top_k=self.config.top_k)
                logger.info(f"  UQ Method: harm_agg (harmonic aggregation, top_k={self.config.top_k})")
            elif self.config.uq_method == "simba":
                self.simba_estimator = SIMBASamplingEstimator(
                    self.client, self.config.n_uncertainty_samples
                )
                logger.info(f"  UQ Method: simba (pairwise consistency, n={self.config.n_uncertainty_samples} samples)")
            elif self.config.uq_method == "sampling":
                self.uncertainty_estimator = UncertaintyEstimator(
                    self.client, self.config.n_uncertainty_samples
                )
                logger.info(f"  UQ Method: sampling (consistency-based, n={self.config.n_uncertainty_samples})")
            else:  # heuristic
                logger.info(f"  UQ Method: heuristic (response-based)")

        # Results directory
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Chroma directory
        self.chroma_dir = str(self.data_dir / "chroma_db")

        # OMOP-specific settings
        self.is_omop = is_omop_dataset(dataset)
        if self.is_omop:
            logger.info(f"OMOP dataset detected - enabling OMOP-specific enhancements")
            logger.info(f"  Table filtering: {self.config.omop_table_filtering}")
            logger.info(f"  Type hints: {self.config.omop_type_hints}")

        # common entity model target (product schema -> canonical CEM)
        self.is_cdm = is_cdm_dataset(dataset)
        if self.is_cdm:
            logger.info(f"CEM_CDM dataset detected - mapping product schemas to the "
                        f"common entity model (canonical target prefix 'cem_<Type>.')")

        logger.info(f"MapReflectRunner initialized")
        logger.info(f"  Model: {model}")
        logger.info(f"  Dataset: {dataset}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Neighbor-K: {self.config.neighbor_k}")
        logger.info(f"  Uncertainty: {self.config.use_uncertainty}")
        logger.info(f"  Reflection: {self.config.use_reflection}")

    def _format_prompt(self, messages: List[Dict]) -> str:
        """Format messages into a prompt string using model-specific template."""
        formatted, _ = format_prompt_for_model(messages, self.model)
        return formatted

    def _run_constrained_assignment(self, prompt_tasks: List[Dict]):
        """Phase 3.5: Enforce one-to-one field mapping within each sample.

        For each sample, run greedy assignment with preemption so that each
        target field is claimed by at most one source field.  Evicted and
        unmatched fields get boosted uncertainty to trigger reflection.
        """
        logger.info("Phase 3.5: Constrained assignment...")

        sample_task_groups: Dict[int, List[int]] = defaultdict(list)
        for tidx, task in enumerate(prompt_tasks):
            sample_task_groups[task["sample_idx"]].append(tidx)

        total_evictions = 0
        total_unmatched = 0

        for sample_idx, task_indices in sample_task_groups.items():
            task_entries = []
            for tidx in task_indices:
                ranking = prompt_tasks[tidx].get("ranking", [])
                top_score = ranking[0].get("score", 0) if (ranking and isinstance(ranking[0], dict)) else 0
                task_entries.append((tidx, top_score, ranking))
            task_entries.sort(key=lambda x: -x[1])

            assigned_targets: Dict[str, tuple] = {}
            assignments: Dict[int, str] = {}
            eviction_queue = []

            def try_assign(tidx, ranking):
                for cand in ranking:
                    if not isinstance(cand, dict):
                        continue
                    cand_id = cand.get("id", "")
                    cand_score = cand.get("score", 0)
                    if not cand_id:
                        continue
                    if cand_id not in assigned_targets:
                        assigned_targets[cand_id] = (tidx, cand_score)
                        assignments[tidx] = cand_id
                        return True
                    else:
                        existing_tidx, existing_score = assigned_targets[cand_id]
                        if cand_score > existing_score:
                            eviction_queue.append((existing_tidx, prompt_tasks[existing_tidx].get("ranking", [])))
                            del assignments[existing_tidx]
                            assigned_targets[cand_id] = (tidx, cand_score)
                            assignments[tidx] = cand_id
                            return True
                return False

            for tidx, top_score, ranking in task_entries:
                if not try_assign(tidx, ranking):
                    prompt_tasks[tidx]["assignment_status"] = "unmatched"
                    total_unmatched += 1

            max_rounds = len(task_indices)
            round_num = 0
            while eviction_queue and round_num < max_rounds:
                round_num += 1
                current_queue = eviction_queue[:]
                eviction_queue = []
                for evicted_tidx, evicted_ranking in current_queue:
                    available_ranking = [
                        c for c in evicted_ranking
                        if isinstance(c, dict) and c.get("id", "") not in assigned_targets
                    ]
                    if available_ranking:
                        cand = available_ranking[0]
                        assigned_targets[cand["id"]] = (evicted_tidx, cand.get("score", 0))
                        assignments[evicted_tidx] = cand["id"]
                        prompt_tasks[evicted_tidx]["assignment_status"] = "reassigned"
                        prompt_tasks[evicted_tidx]["assignment_penalty"] = 0.8
                        total_evictions += 1
                    else:
                        prompt_tasks[evicted_tidx]["assignment_status"] = "unmatched"
                        total_unmatched += 1

            for tidx in task_indices:
                task = prompt_tasks[tidx]
                status = task.get("assignment_status", "")
                ranking = task.get("ranking", [])

                if tidx in assignments:
                    assigned_id = assignments[tidx]
                    assigned_cand = None
                    rest = []
                    for cand in ranking:
                        if isinstance(cand, dict) and cand.get("id") == assigned_id:
                            assigned_cand = dict(cand)
                            if status == "reassigned":
                                assigned_cand["score"] = assigned_cand.get("score", 0) * task.get("assignment_penalty", 1.0)
                                assigned_cand["reassigned"] = True
                        else:
                            rest.append(cand)
                    if assigned_cand:
                        task["ranking"] = [assigned_cand] + rest
                    task["assigned_target"] = assigned_id
                else:
                    task["ranking"] = [{"id": "NO_MATCH", "score": 0.0,
                                        "reason": "All candidate targets assigned to higher-confidence sources"}] + ranking
                    task["uncertainty"] = max(task.get("uncertainty", 0.5), 0.9)
                    task["assigned_target"] = None
                    if not status:
                        task["assignment_status"] = "unmatched"

        logger.info(f"Phase 3.5: {total_evictions} evictions, {total_unmatched} unmatched across {len(sample_task_groups)} samples")

    def _schema_evidence(self, source_prop: str, cand_ids: List[str], n_sib: int = 6) -> str:
        """The evidence reflection arm B shows (scripts/naacl/reflect_v2.py), for the first pass: data types,
        up to n_sib sibling fields of the source and of each candidate, and a same-table marker."""
        if not hasattr(self, "_ev_by_type"):
            self._ev_by_type = {}
            for k, v in self.property_lookup.items():
                if isinstance(v, dict) and v.get("type_name"):
                    self._ev_by_type.setdefault((v.get("product_name"), v.get("type_name")), []).append(k)

        def meta(k):
            v = self.property_lookup.get(k, {})
            return v if isinstance(v, dict) else {}

        def dtype(k):
            m = meta(k)
            t = m.get("field_type") or m.get("type") or "unknown"
            flags = [f for f, on in (("list", m.get("is_list")), ("required", m.get("is_required")))
                     if str(on).lower() == "true"]
            return t + (f" ({', '.join(flags)})" if flags else "")

        def siblings(k):
            m = meta(k)
            group = self._ev_by_type.get((m.get("product_name"), m.get("type_name")), [])
            return m.get("type_name") or "?", [meta(g).get("field_name") or g for g in group if g != k][:n_sib]

        st, ss = siblings(source_prop)
        lines = [f"Source {source_prop}: type {dtype(source_prop)}; belongs to {st}"
                 + (f" with fields {', '.join(ss)}" if ss else "")]
        src_group = (meta(source_prop).get("product_name"), meta(source_prop).get("type_name"))
        for c in cand_ids:
            ct, cs = siblings(c)
            same = (meta(c).get("product_name"), meta(c).get("type_name")) == src_group and src_group[1]
            lines.append(f"Candidate {c}: type {dtype(c)}; belongs to {ct}"
                         + (" (the same table/type as the source)" if same else "")
                         + (f" with fields {', '.join(cs)}" if cs else ""))
        return "\n".join(lines)

    def _build_ranking_prompt(
        self,
        source_prop: str,
        source_desc: str,
        candidates: List[Dict],
        source_product: str,
        target_product: str,
        omop_type_hints: List[str] = None
    ) -> str:
        """Build prompt for ranking candidates."""
        cand_block = "\n".join([
            f"{i+1}. id: {c.get('id', c.get('name', ''))}\n   text: {c.get('doc', c.get('description', ''))}\n   distance: {c.get('distance', 0.5):.4f}"
            for i, c in enumerate(candidates[:self._shown_limit()])  # Limit for prompt size
        ])
        none_task = ""
        if self.config.none_option:
            n_shown = len(candidates[:self._shown_limit()])
            cand_block += (f"\n{n_shown + 1}. id: {NONE_ID}\n   text: none of the candidates above is a correct match "
                           f"for the {source_product} property")
            none_task = (f"\nInclude {NONE_ID} in the ranking like any candidate: give it a high score when no listed "
                         "candidate is a correct match, and a low score when one is.")

        # Build OMOP-specific type hints section
        type_hints_block = ""
        if omop_type_hints:
            type_hints_block = "\n\nFIELD TYPE HINTS (use these to guide matching):\n" + "\n".join(f"- {h}" for h in omop_type_hints)

        # OMOP-specific instructions for healthcare schema mapping
        omop_instructions = ""
        if self.is_omop and self.config.omop_type_hints:
            omop_instructions = """

IMPORTANT: This is a healthcare schema mapping task. Follow these rules:
1. Match by SEMANTIC PURPOSE, not just name similarity
2. Patient/person IDs (person_id, person_source_value) should map to patient identifiers (desynpuf_id, bene_id, subject_id)
3. Provider IDs should map to provider fields (*_npi, provider_id) - NOT to patient IDs
4. Location IDs should map to geographic fields (*_county_cd, *_state_cd, zip) - NOT to provider NPIs
5. Date fields should map to date fields (*_dt, *_date, *time)
6. Concept/code fields should map to code fields (*_cd, *_code, *_type)
7. DO NOT confuse patient identifiers with provider identifiers (NPIs)"""

        # Documentation context (from product docs ChromaDB)
        doc_context_block = ""
        if self.doc_retriever and self.doc_retriever.available:
            # Retrieve docs for source field
            src_context = self.doc_retriever.get_field_context(source_prop, product=source_product)
            if src_context:
                doc_context_block += f"\n\nSOURCE DOCUMENTATION CONTEXT ({source_product}):\n{src_context}"
            # Retrieve docs for target product (using top candidate as query hint)
            if candidates:
                top_cand_id = candidates[0].get('id', candidates[0].get('name', ''))
                tgt_context = self.doc_retriever.get_field_context(top_cand_id, product=target_product)
                if tgt_context:
                    doc_context_block += f"\n\nTARGET DOCUMENTATION CONTEXT ({target_product}):\n{tgt_context}"

        evidence_block = ""
        if self.config.evidence_first:
            evidence_block = "\n\nSchema evidence:\n" + self._schema_evidence(
                source_prop, [c.get("id", c.get("name", "")) for c in candidates[:self._shown_limit()]])

        prompt = f"""You are given a {source_product} property and a list of candidate {target_product} properties.
Rank the candidates by how well they semantically match the {source_product} property.

{source_product} property: {source_prop}
Description: {source_desc}{type_hints_block}{omop_instructions}{doc_context_block}{evidence_block}

Candidates:
{cand_block}

Task: Produce a JSON array of objects, ordered from best match to worst, where each object has:
- "id": candidate id (string, exactly as shown above)
- "score": a float between 0.0 and 1.0 (higher is better match)
- "reason": one concise sentence explaining why you assigned that score{none_task}

Return only the JSON array and nothing else."""

        messages = [
            {"role": "system", "content": "You are an expert in API schema mapping."},
            {"role": "user", "content": prompt}
        ]
        return self._format_prompt(messages)

    def _build_field_ranking_prompt(
        self,
        source_field: str,
        source_desc: str,
        target_type: str,
        target_fields: List[Dict],
        source_product: str,
        target_product: str
    ) -> str:
        """Build prompt for ranking fields within a predicted target type."""
        # Format field candidates
        field_block = "\n".join([
            f"{i+1}. field: {f.get('name', '')}\n   type: {f.get('type', 'unknown')}\n   description: {f.get('description', '')}"
            for i, f in enumerate(target_fields)
        ])

        # Documentation context
        doc_context_block = ""
        if self.doc_retriever and self.doc_retriever.available:
            src_context = self.doc_retriever.get_field_context(source_field, product=source_product)
            if src_context:
                doc_context_block += f"\n\nSOURCE DOCUMENTATION ({source_product}):\n{src_context}"
            tgt_context = self.doc_retriever.get_type_context(target_type, product=target_product)
            if tgt_context:
                doc_context_block += f"\n\nTARGET TYPE DOCUMENTATION ({target_product} {target_type}):\n{tgt_context}"

        prompt = f"""You are given a {source_product} field and a list of candidate fields from the {target_product} table "{target_type}".
Select the field that best matches the semantic meaning of the source field.

{source_product} field: {source_field}
Description: {source_desc}{doc_context_block}

Target table: {target_type}
Candidate fields:
{field_block}

Task: Rank the candidate fields by how well they semantically match the source field.
Produce a JSON array of objects, ordered from best match to worst, where each object has:
- "field": field name (string, exactly as shown above)
- "score": a float between 0.0 and 1.0 (higher is better match)
- "reason": one concise sentence explaining why you assigned that score

Return only the JSON array and nothing else."""

        messages = [
            {"role": "system", "content": "You are an expert in schema field matching."},
            {"role": "user", "content": prompt}
        ]
        return self._format_prompt(messages)

    def _predict_field_in_type(
        self,
        source_field: str,
        source_desc: str,
        target_type: str,
        source_product: str,
        target_product: str
    ) -> Optional[Dict]:
        """Predict the best matching field within a target type.

        Returns:
            Dict with 'field', 'full_field', 'score', 'reason' or None if no fields found
        """
        # Get fields for this type from property lookup
        type_info = self.property_lookup.get(target_type, {})
        fields_dict = type_info.get('fields', {})

        if not fields_dict:
            logger.debug(f"No fields found for type {target_type}")
            return None

        # Convert to list of field dicts
        target_fields = [
            {
                'name': field_name,
                'type': field_info.get('type', 'unknown'),
                'description': field_info.get('description', f'Field {field_name}')
            }
            for field_name, field_info in fields_dict.items()
        ]

        # Build and send prompt
        prompt = self._build_field_ranking_prompt(
            source_field, source_desc, target_type, target_fields,
            source_product, target_product
        )

        response = self.client.generate_single(prompt)

        # Parse response
        parsed = safe_parse_json_array(response)

        if parsed and len(parsed) > 0 and isinstance(parsed[0], dict):
            top_field = parsed[0]
            field_name = top_field.get('field', '')

            # Handle numeric ID references (e.g., "1" -> first field)
            if field_name.isdigit():
                idx = int(field_name) - 1
                if 0 <= idx < len(target_fields):
                    field_name = target_fields[idx]['name']

            return {
                'field': field_name,
                'full_field': f"{target_type}.{field_name}",
                'score': top_field.get('score', 1.0),
                'reason': top_field.get('reason', ''),
                'ranking': parsed
            }

        # Fallback: match by name similarity
        source_field_name = source_field.split('.')[-1] if '.' in source_field else source_field.split('_')[-1]
        for tf in target_fields:
            if source_field_name.lower() in tf['name'].lower() or tf['name'].lower() in source_field_name.lower():
                return {
                    'field': tf['name'],
                    'full_field': f"{target_type}.{tf['name']}",
                    'score': 0.5,
                    'reason': 'fallback name similarity match',
                    'ranking': []
                }

        # Last resort: first field
        if target_fields:
            return {
                'field': target_fields[0]['name'],
                'full_field': f"{target_type}.{target_fields[0]['name']}",
                'score': 0.3,
                'reason': 'fallback first field',
                'ranking': []
            }

        return None

    def _build_reflection_prompt(
        self,
        source_prop: str,
        source_desc: str,
        initial_ranking: List[Dict],
        candidates: List[Dict],
        uncertainty_score: float,
        high_uncertainty: bool,
        source_product: str,
        target_product: str,
        detected_errors: List[str] = None,
        reflection_triggers: List[str] = None,
        sibling_mappings: Dict[str, str] = None
    ) -> str:
        """Build prompt for reflection with uncertainty-adaptive depth.

        Follows the proven two-tier approach from ue_reflect.py:
        - HIGH uncertainty: aggressive prompt urging the model to change its answer
        - LOW uncertainty: mild prompt that preserves correct answers
        """
        reflection_triggers = reflection_triggers or []
        n_candidates = 8 if high_uncertainty else 5
        initial_block = "\n".join([
            f"{i+1}. {r.get('id', 'unknown')} (score={r.get('score')}) - {r.get('reason', '')}"
            for i, r in enumerate(initial_ranking[:n_candidates]) if isinstance(r, dict)
        ])
        cand_block = "\n".join([
            f"- {c.get('id', 'unknown')}: {c.get('doc', '')[:100]}"
            for c in candidates[:n_candidates] if isinstance(c, dict)
        ])

        # Format detected errors (only included when errors are present)
        error_block = ""
        if detected_errors:
            error_block = "\n\n=== DETECTED ERRORS ===\n" + "\n".join(detected_errors)

        # Format sibling mappings (constraint context)
        sibling_block = ""
        if sibling_mappings:
            other_mappings = {k: v for k, v in sibling_mappings.items() if k != source_prop}
            if other_mappings:
                mapping_lines = "\n".join(f"  {src} → {tgt}" for src, tgt in other_mappings.items())
                sibling_block = f"""

=== OTHER FIELD MAPPINGS IN THIS SAMPLE ===
{mapping_lines}

Use these to ensure CONSISTENCY. If a target field is already mapped, prefer a DIFFERENT target field."""

        # Documentation context for high-uncertainty reflection (DocRAG-enriched reflection)
        doc_reflection_block = ""
        if high_uncertainty and hasattr(self, 'doc_retriever') and self.doc_retriever and self.doc_retriever.available:
            # Retrieve documentation for top-2 candidates to give reflection NEW information
            doc_chunks = []
            for cand in initial_ranking[:2]:
                if not isinstance(cand, dict):
                    continue
                cand_id = cand.get("id", "")
                if cand_id and cand_id != "NO_MATCH":
                    ctx = self.doc_retriever.get_field_context(cand_id, product=target_product)
                    if ctx:
                        doc_chunks.append(f"Documentation for {cand_id}:\n{ctx}")
            # Also get docs for source field
            src_ctx = self.doc_retriever.get_field_context(source_prop, product=source_product)
            if src_ctx:
                doc_chunks.insert(0, f"Documentation for {source_prop}:\n{src_ctx}")
            if doc_chunks:
                doc_reflection_block = "\n\n=== DOCUMENTATION CONTEXT (use this to resolve uncertainty) ===\n" + "\n\n".join(doc_chunks)

        # Two-tier reflection instruction based on uncertainty (proven approach from ue_reflect.py)
        if high_uncertainty and doc_reflection_block:
            # Doc-enriched reflection: use documentation to verify/correct, not blind rejection
            reflection_instruction = """The initial ranking shows UNCERTAINTY. Use the DOCUMENTATION CONTEXT below to verify or correct your ranking.

INSTRUCTIONS:
1. Read the documentation for the source field and the top candidates carefully
2. Based on the documentation, determine which candidate best matches the SOURCE field's semantic purpose
3. If the documentation confirms the top candidate, keep it. If it suggests a different candidate, change your ranking.
4. Match by SEMANTIC PURPOSE described in the docs, not by name similarity

Use the documentation as ground truth — it describes what each field actually means."""
        elif high_uncertainty:
            reflection_instruction = """CRITICAL: The initial ranking shows HIGH UNCERTAINTY, which strongly suggests the top-ranked candidate is WRONG.

SEMANTIC MATCHING RULES - Apply these carefully:
1. Foreign keys (fields ending in _id that reference another table) should map to the PRIMARY KEY of the referenced table, NOT to another foreign key
   - Example: concert_stadium_id (FK to stadium) -> stadium_stadium_id (PK of stadium), NOT concert_concert_id
2. Match the TARGET entity, not the SOURCE entity
   - If source is "concert_stadium_id", the target should be in the STADIUM table, not CONCERT
3. Field names often follow pattern: {table}_{field}. Match the semantic purpose of {field}.

You MUST:
1. Assume the #1 candidate is incorrect - it's probably matching by name similarity instead of semantic purpose
2. Look for the candidate that represents the SAME DATA in the TARGET schema
3. For foreign keys, find the primary key they reference
4. CHANGE your top prediction based on semantic meaning, not name similarity

Be bold - your initial answer is likely wrong."""
        else:
            reflection_instruction = """The initial ranking shows reasonable confidence.
Please review for any obvious errors or improvements."""

        prompt = f"""You previously ranked {target_product} candidates for this {source_product} property:

{source_product} property: {source_prop}
Description: {source_desc}

Initial ranking (best first):
{initial_block}

Candidates:
{cand_block}

Uncertainty Score: {uncertainty_score:.3f} {"(HIGH)" if high_uncertainty else "(LOW)"}{error_block}{sibling_block}{doc_reflection_block}

{reflection_instruction}

Task:
1) Point out possible errors in the initial ranking (1-2 sentences).
2) Produce a revised JSON array (id, score, reason) from best to worst match.

Return a JSON object:
{{"analysis": "your error analysis", "revised_ranking": [array of ranked objects]}}"""

        messages = [
            {"role": "system", "content": "You are an expert schema mapper performing self-reflection."},
            {"role": "user", "content": prompt}
        ]
        return self._format_prompt(messages)

    def _parse_ranking_response(self, response: str, candidates: List[Dict]) -> List[Dict]:
        """Parse a ranking response (protocol v2).

        Numeric ids map to the candidates by position; every other id is resolved against the candidates shown
        in the prompt (resolve_ranked_id), and ids that name no shown candidate are dropped. A response that
        cannot be read as a ranking gives an empty ranking, which protocol v2 counts as invalid (wrong, margin 0).
        """
        parsed = safe_parse_json_array(response)
        dropped = 0
        if robust_json() and parsed is not None:
            # v4.1: keep the well-formed entries and count the malformed ones, rather than discarding the ranking
            good = [item for item in parsed if isinstance(item, dict) and 'id' in item]
            dropped, parsed = len(parsed) - len(good), (good if good or not parsed else None)
        if parsed is None or not all(isinstance(item, dict) and 'id' in item for item in parsed):
            out = Ranking()
            out.status = "unparsed"
            return out
        shown = shown_ids(candidates, self._shown_limit())
        id_to_candidate = {str(i+1): c.get('id', c.get('name', '')) for i, c in enumerate(candidates)}
        if self.config.none_option:
            id_to_candidate[str(len(shown) + 1)] = NONE_ID
        for item in parsed:
            item_id = str(item['id'])
            if item_id in id_to_candidate:
                item['id'] = id_to_candidate[item_id]
        out = Ranking()
        if self.config.none_option:
            # the NONE entry is a signal, not a candidate: log its score and rank, then drop it
            for pos, item in enumerate(parsed):
                if str(item.get('id', '')).strip().upper() == NONE_ID:
                    out.none_score, out.none_rank = item.get('score'), pos
                    break
            parsed = [it for it in parsed if str(it.get('id', '')).strip().upper() != NONE_ID]
        out.extend(resolve_ranking(parsed, shown))
        if not out:
            out.status = "none_only" if out.none_rank is not None else "empty"
        elif dropped:
            out.status = "partial"  # v4.1: a ranking read with malformed entries dropped
        return out

    def _parse_reflection_response(self, response: str, candidates: List[Dict], initial_ranking: List[Dict]) -> Dict:
        """Parse reflection response."""
        parsed = safe_parse_json_object(response)

        if parsed and "revised_ranking" in parsed:
            result = parsed
        else:
            array_parsed = safe_parse_json_array(response)
            if array_parsed:
                result = {"analysis": "Parsing fallback", "revised_ranking": array_parsed}
            else:
                result = {"analysis": "Could not parse response", "revised_ranking": initial_ranking}

        # Map numeric IDs back to actual field IDs and normalize
        if result.get("revised_ranking"):
            id_to_candidate = {str(i+1): c.get('id', c.get('name', '')) for i, c in enumerate(candidates)}
            for item in result["revised_ranking"]:
                if isinstance(item, dict) and 'id' in item:
                    item_id = str(item['id'])
                    if item_id in id_to_candidate:
                        item['id'] = id_to_candidate[item_id]
                    else:
                        # Normalize any formatting prefixes (e.g., "1. ", "id: ")
                        item['id'] = normalize_field_id(item['id'])

        return result

    def _source_prefix(self, sample: Dict) -> Optional[str]:
        """Id prefix of the source schema, which the retrieval utterance names.

        Matches the AnyTool retrieval baseline, which excludes the source product from its search:
        ITOpsGraph excludes the source product, OMOP the OMOP side, ToolBench the source API, and
        NL2GQL and Spider the source table.
        """
        src = sample.get("expected_source", "")
        if isinstance(src, dict):
            src = src.get("id", src.get("name", ""))
        src = str(src or "").strip()
        if not src:
            return None
        ds = self.dataset.lower()
        if ds.startswith("omop"):
            return src.split("_")[0].lower() + "_"
        if ds == "toolbench":
            return src.lower() + "."
        return src.lower() + "_"

    def _retrieve_target_free(self, src_field: str, query_text: str,
                              collection_name: str, sample: Dict) -> List[Dict]:
        """Retrieval-mode candidates: hybrid search over the whole catalogue.

        Uses retrieve_candidates (dense + BM25, Reciprocal Rank Fusion) over every field in the
        lookup, drops the source schema's own fields (see _source_prefix) and the source field's own
        type, and keeps top_k * 3 candidates, the same budget the target-given path allows. A source
        field's nearest neighbours are mostly its own schema, so the pool is fetched wide (as the
        interactive agent does, capped at 5000 for the ChromaDB batch limit) before exclusion.
        """
        budget = self.config.top_k * 3
        pool = retrieve_candidates(query_text, collection_name, self.chroma_dir,
                                   min(len(self.property_lookup), 5000))
        return self._exclude_source(pool, src_field, sample)[:budget]

    def _toolbench_tool(self, api_id: str) -> Optional[str]:
        """Normalised tool name of a ToolBench API id ("Tool_Name_api_name").

        Tool names can prefix each other ("spotify" and "spotify_web"), so the longest matching tool
        name wins. Tool names come from both sides of data/toolbench_dataset.jsonl.
        """
        if not hasattr(self, "_tb_tools"):
            tools = set()
            # Retrieval runs use --data-dir data/retrieval_mode, which has only the eval file.
            paths = [p for p in (self.data_dir / "toolbench_dataset.jsonl",
                                 Path(__file__).resolve().parents[1] / "data" / "toolbench_dataset.jsonl")
                     if p.exists()]
            if not paths:
                raise FileNotFoundError("--exclude-source-tool needs data/toolbench_dataset.jsonl")
            for path in paths[:1]:
                for line in open(path):
                    r = json.loads(line)
                    for side in ("source_api", "target_api"):
                        name = (r.get(side) or {}).get("tool_name")
                        if name:
                            tools.add(name.replace(" ", "_").lower())
            self._tb_tools = tools
        a = api_id.lower()
        hits = [t for t in self._tb_tools if a.startswith(t + "_")]
        return max(hits, key=len) if hits else None

    def _exclude_source(self, candidates: List[Dict], src_field: str, sample: Dict) -> List[Dict]:
        """Drop the source field, its own type, and the source schema (see _source_prefix).

        With exclude_source_tool on ToolBench, also drop every API of the source tool.
        """
        sf = src_field.lower()
        own = sf.rsplit(".", 1)[0] + "." if "." in sf else (sf.rsplit("__", 1)[0] + "__" if "__" in sf else None)
        src_prefix = self._source_prefix(sample)
        src_tool = None
        if self.config.exclude_source_tool and self.dataset.lower() == "toolbench":
            # The records the runner passes here do not keep source_api, so fall back to the source API id.
            name = (sample.get("source_api") or {}).get("tool_name")
            src = sample.get("expected_source", "")
            src = src.get("id", "") if isinstance(src, dict) else str(src or "")
            src_tool = name.replace(" ", "_").lower() if name else self._toolbench_tool(src)
        out = []
        for c in candidates:
            cid = str(c.get("id", "")).lower()
            if cid == sf or (own and cid.startswith(own)) or (src_prefix and cid.startswith(src_prefix)):
                continue
            if src_tool and self._toolbench_tool(cid.split(".", 1)[0]) == src_tool:
                continue
            out.append(c)
        return out

    def _parsed_utterance(self, sample: Dict) -> Optional[Dict[str, str]]:
        """OMOP target table for the given-target setting (None elsewhere), as the batch path computes it."""
        if not (self.is_omop and not self.config.retrieval_mode):
            return None
        # First try to extract from sample's target_schema (more reliable)
        parsed_utterance = extract_omop_target_table(sample)
        # Fallback to utterance parsing if target_schema not available
        if not parsed_utterance.get("target_table"):
            parsed_utterance = parse_omop_utterance(sample.get("utterance", ""))
        return parsed_utterance

    def collection_name(self) -> str:
        """The index collection the batch path searches (see run_batch)."""
        if self.config.retrieval_mode:
            return f"mapreflect_{self.dataset.lower()}_catalogue"
        return f"mapreflect_{self.dataset.lower()}_target"

    def first_pass_prompt(self, sample: Dict, src_field: str) -> Tuple[str, List[str]]:
        """The ranking prompt the batch path builds for one field, and the candidate ids it shows (no model call)."""
        expected_source = sample.get("expected_source", "")
        expected_target = sample.get("expected_target", "")
        if isinstance(expected_source, dict):
            source_product = expected_source.get("name", "SOURCE").upper()
        else:
            source_product = str(expected_source).split("_")[0].upper() if expected_source else "SOURCE"
        if isinstance(expected_target, dict):
            target_product = expected_target.get("name", "TARGET").upper()
        else:
            target_product = str(expected_target).split("_")[0].upper() if expected_target else "TARGET"
        if self.config.retrieval_mode:
            target_product = "TARGET"
        candidates, hints = self._get_candidates_for_sample(src_field, sample, self.collection_name(),
                                                            self._parsed_utterance(sample))
        src_info = self.property_lookup.get(src_field, {})
        src_desc = src_info.get("description", src_field) if isinstance(src_info, dict) else str(src_info)
        prompt = self._build_ranking_prompt(src_field, src_desc, candidates, source_product, target_product,
                                            omop_type_hints=hints)
        return prompt, shown_ids(candidates, self._shown_limit())

    def _shown_limit(self) -> Optional[int]:
        """How many candidates a ranking prompt shows: 20, or all of them with full_list or candidate_lists."""
        return None if (self.config.full_list or self.config.candidate_lists is not None) else 20

    def shown_candidate_ids(self, sample: Dict, src_field: str) -> List[str]:
        """The candidate ids the first-pass prompt showed for one field (deterministic; no model call)."""
        cands, _ = self._get_candidates_for_sample(src_field, sample, self.collection_name(),
                                                   self._parsed_utterance(sample))
        return shown_ids(cands, self._shown_limit())

    def _get_candidates_for_sample(
        self,
        src_field: str,
        sample: Dict,
        collection_name: str,
        parsed_utterance: Dict[str, str] = None
    ) -> Tuple[List[Dict], List[str]]:
        """Get candidates for a source field.

        Returns:
            Tuple of (candidates list, omop_type_hints list)
        """
        src_info = self.property_lookup.get(src_field, {})
        src_desc = src_info.get("description", src_field) if isinstance(src_info, dict) else str(src_info)
        query_text = f"{src_field}: {src_desc}"

        if self.config.candidate_lists is not None:
            listed = self.config.candidate_lists.get(f"{sample.get('sample_id')}||{src_field}")
            hints = get_omop_field_type_hints(src_field) if (self.is_omop and self.config.omop_type_hints) else []
            out = []
            for prop_name in listed or []:
                info = self.property_lookup.get(prop_name, {})
                desc = info.get("description", prop_name) if isinstance(info, dict) else str(info)
                out.append({"id": prop_name, "name": prop_name, "doc": f"{prop_name} -- {desc}", "distance": 0.5,
                            "meta": info if isinstance(info, dict) else {}})
            return out, hints

        # Get OMOP type hints if applicable
        omop_type_hints = []
        if self.is_omop and self.config.omop_type_hints:
            omop_type_hints = get_omop_field_type_hints(src_field)

        # Check for explicit target fields in sample
        sample_target_fields = sample.get("target_fields", [])

        if self.config.retrieval_mode:
            candidates = self._retrieve_target_free(src_field, query_text, collection_name, sample)
        elif sample_target_fields:
            # Use explicit target fields
            candidates = []
            for prop_name in sample_target_fields:
                prop_lookup = self.property_lookup.get(prop_name, {})
                prop_desc = prop_lookup.get("description", prop_name) if isinstance(prop_lookup, dict) else str(prop_lookup)
                candidates.append({
                    "id": prop_name,
                    "name": prop_name,
                    "doc": f"{prop_name} -- {prop_desc}",
                    "distance": 0.5,
                    "meta": prop_lookup if isinstance(prop_lookup, dict) else {}
                })
        else:
            # Check for target schema filtering (multi-target datasets like ITOpsGraph)
            # Use 'or' to handle None values (not just missing keys)
            target_schema = sample.get("expected_target") or sample.get("target_schema") or ""
            sample_metadata = sample.get("metadata", {})
            target_type = sample_metadata.get("target_type", "")

            # Handle dict target_schema (OMOP datasets have nested schema objects)
            if isinstance(target_schema, dict):
                target_schema = target_schema.get("name", target_schema.get("id", ""))

            if target_schema and isinstance(target_schema, str):
                # Build target prefix for filtering
                # Handle multi-word schema names (e.g., "MIMIC PATIENTS" -> "mimic_")
                target_product = target_schema.split()[0].lower() if " " in target_schema else target_schema.lower()

                # NL2GQL uses special prefix format: "nl2gql_TableName." (e.g., "nl2gql_Stadium.")
                if self.dataset.lower() == "nl2gql":
                    target_prefix = f"nl2gql_{target_schema}."
                # ToolBench uses "APIName." prefix (e.g., "Cloud_Cast_Get_humidty.")
                elif self.dataset.lower() == "toolbench":
                    target_prefix = f"{target_schema}."
                elif target_type:
                    # ITOpsGraph style: product + type prefix (e.g., "prod_b_FilterInput.")
                    target_prefix = f"{target_product}_{target_type}."

                    # Fix: Handle case where target_type already contains part of target_product
                    # e.g., target_product="prod_h_tf", target_type="prod_h_ProdHProject"
                    # would create "prod_h_tf_prod_h_ProdHProject." (double prefix)
                    # Check if this prefix finds any matches; if not, try stripping overlap
                    if not any(k.lower().startswith(target_prefix.lower()) for k in self.property_lookup):
                        for part in target_product.split("_"):
                            if target_type.lower().startswith(part.lower() + "_"):
                                alt_type = target_type[len(part) + 1:]
                                alt_prefix = f"{target_product}_{alt_type}."
                                if any(k.lower().startswith(alt_prefix.lower()) for k in self.property_lookup):
                                    target_prefix = alt_prefix
                                    break
                else:
                    # Simple product prefix (e.g., "mimic_" from "MIMIC PATIENTS")
                    target_prefix = f"{target_product}_"

                # Filter candidates from lookup by target prefix
                candidates = []
                for prop_name, prop_info in self.property_lookup.items():
                    if prop_name.lower().startswith(target_prefix.lower()):
                        prop_desc = prop_info.get("description", prop_name) if isinstance(prop_info, dict) else str(prop_info)
                        candidates.append({
                            "id": prop_name,
                            "name": prop_name,
                            "doc": f"{prop_name} -- {prop_desc}",
                            "distance": 0.5,
                            "meta": prop_info if isinstance(prop_info, dict) else {}
                        })

                # Limit candidates to avoid token overflow
                max_candidates = self.config.top_k * 3
                if len(candidates) > max_candidates and not self.config.full_list:
                    logger.debug(f"Truncating {len(candidates)} candidates to {max_candidates}")
                    candidates = candidates[:max_candidates]
            else:
                # Fallback to retrieval-based candidates
                candidates = retrieve_candidates(query_text, collection_name, self.chroma_dir, self.config.top_k)

        # Expand with neighbors
        if self.property_graph is not None and self.config.neighbor_k > 0:
            candidates = expand_candidates_with_neighbors(
                candidates, self.property_graph, self.property_lookup, self.config.neighbor_k
            )
            if self.config.retrieval_mode:
                # graph neighbours can re-add the source schema's own fields
                candidates = self._exclude_source(candidates, src_field, sample)

        # OMOP-specific: Filter candidates to target table from utterance
        if (self.is_omop and self.config.omop_table_filtering and parsed_utterance
                and not self.config.retrieval_mode):
            target_table = parsed_utterance.get("target_table", "")
            target_schema = parsed_utterance.get("target_schema", "")
            if target_table and target_schema:
                candidates = filter_candidates_by_omop_table(
                    candidates, target_table, target_schema
                )

        return candidates, omop_type_hints

    def process_sample(
        self,
        sample: Dict,
        collection_name: str,
        all_uncertainties: List[float]
    ) -> Dict:
        """Process a single sample."""
        sample_id = sample.get("sample_id", "unknown")
        ground_truth = sample.get("ground_truth_mapping", {})

        # Determine source/target products
        expected_source = sample.get("expected_source", "")
        expected_target = sample.get("expected_target", "")

        if isinstance(expected_source, dict):
            source_product = expected_source.get("name", "SOURCE").upper()
        else:
            source_product = str(expected_source).split("_")[0].upper() if expected_source else "SOURCE"

        if isinstance(expected_target, dict):
            target_product = expected_target.get("name", "TARGET").upper()
        else:
            target_product = str(expected_target).split("_")[0].upper() if expected_target else "TARGET"
        if self.config.retrieval_mode:
            target_product = "TARGET"  # the prompt must not name the product being searched for

        sample_result = {
            "sample_id": sample_id,
            "utterance": sample.get("utterance", ""),
            "expected_source": expected_source,
            "expected_target": expected_target,
            "ground_truth_mapping": ground_truth,
            "predictions": {},
        }

        sample_uncertainties = []
        sibling_mappings: Dict[str, str] = {}  # Collect top-1 predictions for sibling context

        # Parse utterance for OMOP datasets
        parsed_utterance = None
        if self.is_omop and not self.config.retrieval_mode:
            # First try to extract from sample's target_schema (more reliable)
            parsed_utterance = extract_omop_target_table(sample)
            # Fallback to utterance parsing if target_schema not available
            if not parsed_utterance.get("target_table"):
                parsed_utterance = parse_omop_utterance(sample.get("utterance", ""))

        # Process each source field
        for src_field, gt_target in ground_truth.items():
            # Get candidates
            candidates, omop_type_hints = self._get_candidates_for_sample(
                src_field, sample, collection_name, parsed_utterance
            )

            if not candidates:
                logger.warning(f"No candidates for {src_field}, skipping")
                continue

            # Get source description
            src_info = self.property_lookup.get(src_field, {})
            src_desc = src_info.get("description", src_field) if isinstance(src_info, dict) else str(src_info)

            # Build ranking prompt
            ranking_prompt = self._build_ranking_prompt(
                src_field, src_desc, candidates, source_product, target_product,
                omop_type_hints=omop_type_hints
            )

            # Get ranking response
            response = self.client.generate_single(ranking_prompt)
            ranking = self._parse_ranking_response(response, candidates)

            # Track top-1 prediction for sibling context
            if ranking and isinstance(ranking[0], dict):
                sibling_mappings[src_field] = ranking[0].get("id", "")

            # Estimate uncertainty based on configured method
            if self.config.use_uncertainty:
                if self.config.uq_method == "perplexity" and self.perplexity_estimator:
                    # Perplexity-based UQ using token logprobs
                    uncertainty_info = self.perplexity_estimator.estimate_uncertainty(ranking_prompt)
                elif self.config.uq_method == "p_true" and self.p_true_estimator:
                    # P(True) self-verification UQ
                    if ranking and isinstance(ranking[0], dict):
                        top_pred = ranking[0]
                        top_pred_id = top_pred.get("id", "")
                        # Find the candidate info for the top prediction
                        top_pred_desc = ""
                        for cand in candidates:
                            if cand.get("id") == top_pred_id or cand.get("name") == top_pred_id:
                                top_pred_desc = cand.get("doc", "")
                                break
                        uncertainty_info = self.p_true_estimator.estimate_uncertainty(
                            source_field=src_field,
                            source_desc=src_desc,
                            predicted_target=top_pred_id,
                            target_desc=top_pred_desc,
                            source_product=source_product,
                            target_product=target_product
                        )
                    else:
                        uncertainty_info = {"uncertainty": 0.8, "method": "p_true_no_prediction"}
                elif self.config.uq_method == "semantic_entropy" and self.semantic_entropy_estimator:
                    # Semantic entropy UQ using multiple samples
                    uncertainty_info = self.semantic_entropy_estimator.estimate_uncertainty(
                        ranking_prompt, response, candidates
                    )
                elif self.config.uq_method == "sampling" and self.uncertainty_estimator:
                    # Sampling-based UQ using consistency
                    uncertainty_info = self.uncertainty_estimator.estimate_ranking_uncertainty(
                        ranking_prompt, response, candidates
                    )
                elif self.config.uq_method == "arith_agg" and self.arith_agg_estimator:
                    # Arithmetic aggregation UQ (SIMBA-style)
                    uncertainty_info = self.arith_agg_estimator.estimate_uncertainty(
                        response, candidates
                    )
                elif self.config.uq_method == "harm_agg" and self.harm_agg_estimator:
                    # Harmonic aggregation UQ (SIMBA-style)
                    uncertainty_info = self.harm_agg_estimator.estimate_uncertainty(
                        response, candidates
                    )
                else:
                    # Heuristic-based UQ from response characteristics
                    uncertainty_score = estimate_uncertainty_from_response(response, candidates)
                    uncertainty_info = {"uncertainty": uncertainty_score, "method": "heuristic"}
            else:
                # Default: quick heuristic
                uncertainty_score = estimate_uncertainty_from_response(response, candidates)
                uncertainty_info = {"uncertainty": uncertainty_score, "method": "heuristic"}

            uncertainty_score = uncertainty_info.get("uncertainty", 0.5)
            sample_uncertainties.append(uncertainty_score)
            all_uncertainties.append(uncertainty_score)

            # Detect mapping errors from initial ranking
            detected_errors = detect_mapping_errors(
                src_field, source_product, target_product, ranking, candidates
            )

            # Self-reflection (optional)
            reflection = None
            if self.config.use_reflection:
                # Compute dynamic threshold
                if len(all_uncertainties) > 1:
                    mean_u = np.mean(all_uncertainties)
                    std_u = np.std(all_uncertainties)
                    threshold = mean_u + 0.5 * std_u
                else:
                    threshold = 0.6

                high_uncertainty = uncertainty_score > threshold

                # Trigger reflection if high uncertainty OR errors detected
                should_reflect = high_uncertainty or len(detected_errors) > 0

                if should_reflect:
                    reflection_prompt = self._build_reflection_prompt(
                        src_field, src_desc, ranking, candidates,
                        uncertainty_score, high_uncertainty, source_product, target_product,
                        detected_errors=detected_errors,
                        sibling_mappings=sibling_mappings
                    )
                    reflection_response = self.client.generate_single(reflection_prompt)
                    reflection = self._parse_reflection_response(reflection_response, candidates, ranking)
                    reflection["high_uncertainty"] = high_uncertainty
                    reflection["detected_errors"] = detected_errors

            # Field-level prediction (optional, default on)
            field_prediction = None
            if self.config.predict_fields:
                # Get top-1 type prediction (use reflection if available)
                if reflection and reflection.get("revised_ranking"):
                    final_ranking = reflection["revised_ranking"]
                else:
                    final_ranking = ranking

                if final_ranking and isinstance(final_ranking[0], dict):
                    top_type = normalize_field_id(final_ranking[0].get("id", ""))
                    if top_type:
                        field_prediction = self._predict_field_in_type(
                            src_field, src_desc, top_type,
                            source_product, target_product
                        )

            # Store prediction
            sample_result["predictions"][src_field] = {
                "initial_ranking": ranking,
                "none_score": getattr(ranking, "none_score", None),
                "none_rank": getattr(ranking, "none_rank", None),
                "parse_status": getattr(ranking, "status", None),
                "reflection": reflection,
                "field_prediction": field_prediction,
                "uncertainty": float(uncertainty_score),
                "uncertainty_info": uncertainty_info,
                "detected_errors": detected_errors,
            }

        # Evaluate sample (both type-level and field-level)
        type_correct_count = 0
        field_correct_count = 0
        total_count = 0
        for src_field, gt_target in ground_truth.items():
            if src_field in sample_result["predictions"]:
                pred = sample_result["predictions"][src_field]
                any_reflection = (
                    self.config.use_reflection or self.config.use_self_reflection or
                    self.config.use_error_reflection or self.config.use_uq_reflection or
                    self.config.use_neighborhood_reflection
                )
                if any_reflection and pred.get("reflection"):
                    revised = pred["reflection"].get("revised_ranking", [])
                else:
                    revised = pred.get("initial_ranking", [])

                # Extract ground truth type (e.g., "bank2_branch" from "bank2_branch.branch_id")
                gt_target_str = gt_target[0] if isinstance(gt_target, list) else gt_target
                gt_type = gt_target_str.split('.')[0] if '.' in gt_target_str else gt_target_str

                if revised and isinstance(revised[0], dict):
                    top_pred = normalize_field_id(revised[0].get("id", ""))
                    # Type-level accuracy
                    if isinstance(gt_target, list):
                        gt_types = [t.split('.')[0] for t in gt_target]
                        if top_pred in gt_types:
                            type_correct_count += 1
                    elif top_pred == gt_type:
                        type_correct_count += 1

                # Field-level accuracy (if field prediction available)
                field_pred = pred.get("field_prediction")
                if field_pred:
                    pred_full_field = field_pred.get("full_field", "")
                    if isinstance(gt_target, list):
                        if pred_full_field in gt_target:
                            field_correct_count += 1
                    elif pred_full_field == gt_target_str:
                        field_correct_count += 1

                total_count += 1

        sample_result["type_accuracy"] = type_correct_count / total_count if total_count > 0 else 0.0
        sample_result["field_accuracy"] = field_correct_count / total_count if total_count > 0 else 0.0
        sample_result["accuracy"] = sample_result["type_accuracy"]  # Legacy: type accuracy
        sample_result["type_correct_count"] = type_correct_count
        sample_result["field_correct_count"] = field_correct_count
        sample_result["correct_count"] = type_correct_count  # Legacy
        sample_result["total_count"] = total_count
        sample_result["mean_uncertainty"] = float(np.mean(sample_uncertainties)) if sample_uncertainties else 0.5

        return sample_result

    def run(self, samples: List[Dict]) -> Dict:
        """
        Run the Map-Reflect pipeline on samples using batch inference.

        This method batches all ranking prompts together for efficient API usage,
        then processes reflection in a second batch pass if enabled.

        Args:
            samples: List of sample dicts

        Returns:
            Results dict with metrics and per-sample results
        """
        total_start = time.time()
        results = []
        all_uncertainties = []

        # Setup checkpoint file
        model_name = self.model.replace("/", "_").replace("endpoint_b_", "")
        ckpt_dir = Path(self.config.checkpoint_dir) if self.config.checkpoint_dir else self.results_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = ckpt_dir / f"checkpoint_{self.dataset.lower()}_{model_name}.json"

        # Load checkpoint if resuming
        processed_ids = set()
        if self.config.resume and checkpoint_file.exists():
            try:
                with open(checkpoint_file) as f:
                    checkpoint_data = json.load(f)
                results = checkpoint_data.get("results", [])
                all_uncertainties = checkpoint_data.get("all_uncertainties", [])
                processed_ids = {r["sample_id"] for r in results}
                logger.info(f"Resumed from checkpoint: {len(processed_ids)} samples already processed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

        # Index target properties for retrieval
        collection_name = f"mapreflect_{self.dataset.lower()}_target"
        if self.config.retrieval_mode:
            # distinct name: index_chroma rebuilds the dense index each run, but BM25 loads a
            # cached pickle by collection name, which could come from a different catalogue
            collection_name = f"mapreflect_{self.dataset.lower()}_catalogue"
        if HAS_CHROMADB:
            index_chroma(collection_name, self.property_lookup, self.chroma_dir)

        # Filter samples to process
        samples_to_process = []
        for i, sample in enumerate(samples):
            sample_id = sample.get("sample_id", f"sample_{i}")
            if sample_id in processed_ids:
                continue
            # Skip samples with no utterance (but allow empty ground_truth_mapping
            # for type-level-only datasets like ITOM_ProductH)
            if not sample.get("utterance") and not sample.get("ground_truth_mapping"):
                continue
            samples_to_process.append((i, sample))

        if not samples_to_process:
            logger.info("No samples to process")
            return self._build_final_results(results, all_uncertainties, time.time() - total_start)

        logger.info(f"Processing {len(samples_to_process)} samples in batch mode (batch_size={self.config.batch_size})")

        # =====================================================================
        # PHASE 1: Collect all ranking prompts
        # =====================================================================
        logger.info("Phase 1: Collecting ranking prompts...")
        prompt_tasks = []  # List of (sample_idx, sample, src_field, gt_target, candidates, prompt, source_product, target_product)

        for sample_idx, sample in tqdm(samples_to_process, desc="Collecting prompts"):
            ground_truth = sample.get("ground_truth_mapping", {})
            expected_source = sample.get("expected_source", "")
            expected_target = sample.get("expected_target", "")

            if isinstance(expected_source, dict):
                source_product = expected_source.get("name", "SOURCE").upper()
            else:
                source_product = str(expected_source).split("_")[0].upper() if expected_source else "SOURCE"

            if isinstance(expected_target, dict):
                target_product = expected_target.get("name", "TARGET").upper()
            else:
                target_product = str(expected_target).split("_")[0].upper() if expected_target else "TARGET"
            if self.config.retrieval_mode:
                target_product = "TARGET"  # the prompt must not name the product being searched for

            parsed_utterance = self._parsed_utterance(sample)

            for src_field, gt_target in ground_truth.items():
                if (self.config.only_keys is not None
                        and f"{sample.get('sample_id')}||{src_field}" not in self.config.only_keys):
                    continue
                candidates, omop_type_hints = self._get_candidates_for_sample(
                    src_field, sample, collection_name, parsed_utterance
                )
                if not candidates:
                    continue

                src_info = self.property_lookup.get(src_field, {})
                src_desc = src_info.get("description", src_field) if isinstance(src_info, dict) else str(src_info)

                ranking_prompt = self._build_ranking_prompt(
                    src_field, src_desc, candidates, source_product, target_product,
                    omop_type_hints=omop_type_hints
                )
                if LL.enabled():  # the latency log names the field each batched ranking call belongs to
                    LL.register_prompt(ranking_prompt, f"{sample.get('sample_id')}||{src_field}")

                prompt_tasks.append({
                    "sample_idx": sample_idx,
                    "sample": sample,
                    "src_field": src_field,
                    "gt_target": gt_target,
                    "candidates": candidates,
                    "prompt": ranking_prompt,
                    "src_desc": src_desc,
                    "source_product": source_product,
                    "target_product": target_product,
                    "omop_type_hints": omop_type_hints,
                })

        logger.info(f"Collected {len(prompt_tasks)} ranking prompts from {len(samples_to_process)} samples")

        # =====================================================================
        # PHASE 2: Batch inference for initial rankings
        # =====================================================================
        logger.info("Phase 2: Batch inference for rankings...")
        ranking_prompts = [t["prompt"] for t in prompt_tasks]
        ranking_responses = self.client.generate_batch(
            ranking_prompts,
            use_cache=True,
            batch_size=self.config.batch_size
        )

        # =====================================================================
        # PHASE 3: Process responses and collect reflection prompts
        # =====================================================================
        logger.info("Phase 3: Processing responses...")
        reflection_tasks = []  # Tasks that need reflection

        # First pass: parse all responses
        for task_idx, (task, response) in enumerate(zip(prompt_tasks, ranking_responses)):
            ranking = self._parse_ranking_response(response, task["candidates"])
            task["ranking"] = ranking
            task["response"] = response
            if not ranking and os.environ.get("UNPARSED_DUMP"):
                # Opt-in diagnostics: keep the raw response of every ranking that could not be parsed.
                with _usage_lock, open(os.environ["UNPARSED_DUMP"], "a") as f:
                    f.write(json.dumps(dict(sample_id=task["sample"].get("sample_id"), field=task["src_field"],
                                            n_candidates=len(task["candidates"]), response=response)) + "\n")

        # BATCH UQ: Estimate uncertainty for all tasks at once (massive speedup)
        if self.config.use_uncertainty and self.config.uq_method == "p_true" and self.p_true_estimator:
            # P(True) batch estimation
            uq_fields = []
            for task in prompt_tasks:
                ranking = task.get("ranking", [])
                if ranking and isinstance(ranking[0], dict):
                    top_pred = ranking[0]
                    top_pred_id = top_pred.get("id", "")
                    top_pred_desc = ""
                    for cand in task["candidates"]:
                        if cand.get("id") == top_pred_id or cand.get("name") == top_pred_id:
                            top_pred_desc = cand.get("doc", "")
                            break
                    uq_fields.append({
                        "source_field": task["src_field"],
                        "source_desc": task["src_desc"],
                        "predicted_target": top_pred_id,
                        "target_desc": top_pred_desc,
                        "source_product": task["source_product"],
                        "target_product": task["target_product"]
                    })
                else:
                    uq_fields.append(None)  # No prediction

            # Batch estimate for valid fields
            valid_indices = [i for i, f in enumerate(uq_fields) if f is not None]
            valid_fields = [uq_fields[i] for i in valid_indices]

            if valid_fields:
                batch_results = self.p_true_estimator.estimate_uncertainty_batch(valid_fields)

                # Assign results back to tasks
                result_idx = 0
                for task_idx, task in enumerate(prompt_tasks):
                    if uq_fields[task_idx] is not None:
                        task["uncertainty_info"] = batch_results[result_idx]
                        task["uncertainty"] = batch_results[result_idx]["uncertainty"]
                        result_idx += 1
                    else:
                        task["uncertainty_info"] = {"uncertainty": 0.8, "method": "p_true_no_prediction"}
                        task["uncertainty"] = 0.8
                    all_uncertainties.append(task["uncertainty"])

        elif self.config.use_uncertainty and self.config.uq_method == "semantic_entropy" and self.semantic_entropy_estimator:
            # Semantic entropy batch estimation
            uq_fields = []
            for task in prompt_tasks:
                uq_fields.append({
                    "ranking_prompt": task["prompt"],
                    "initial_response": task["response"],
                    "candidates": task["candidates"]
                })

            batch_results = self.semantic_entropy_estimator.estimate_uncertainty_batch(uq_fields)

            for task_idx, task in enumerate(prompt_tasks):
                task["uncertainty_info"] = batch_results[task_idx]
                task["uncertainty"] = batch_results[task_idx]["uncertainty"]
                all_uncertainties.append(task["uncertainty"])

        elif self.config.use_uncertainty and self.config.uq_method == "lmp_sampling" and self.lmp_sampling_estimator:
            # lm-polygraph sampling-consistency family (LexSim/NumSemSets/DegMat/Ecc/EigVal)
            uq_fields = [{
                "ranking_prompt": task["prompt"],
                "initial_response": task["response"],
                "candidates": task["candidates"],
            } for task in prompt_tasks]

            batch_results = self.lmp_sampling_estimator.estimate_uncertainty_batch(uq_fields)

            for task_idx, task in enumerate(prompt_tasks):
                task["uncertainty_info"] = batch_results[task_idx]
                task["uncertainty"] = batch_results[task_idx]["uncertainty"]
                all_uncertainties.append(task["uncertainty"])

        elif self.config.use_uncertainty and self.config.uq_method == "simba" and self.simba_estimator:
            # SIMBA pairwise consistency batch estimation
            uq_fields = []
            for task in prompt_tasks:
                uq_fields.append({
                    "ranking_prompt": task["prompt"],
                    "initial_response": task["response"],
                    "candidates": task["candidates"]
                })

            batch_results = self.simba_estimator.estimate_uncertainty_batch(uq_fields)

            for task_idx, task in enumerate(prompt_tasks):
                task["uncertainty_info"] = batch_results[task_idx]
                task["uncertainty"] = batch_results[task_idx]["uncertainty"]
                # Store majority vote for potential use in reflection
                task["simba_majority_prediction"] = batch_results[task_idx].get("majority_prediction", "")
                all_uncertainties.append(task["uncertainty"])

        elif self.config.use_uncertainty and self.config.uq_method == "perplexity" and self.perplexity_estimator:
            # Perplexity-based UQ (still sequential - requires logprobs per call)
            for task in prompt_tasks:
                uncertainty_info = self.perplexity_estimator.estimate_uncertainty(task["prompt"])
                task["uncertainty_info"] = uncertainty_info
                task["uncertainty"] = uncertainty_info.get("uncertainty", 0.5)
                all_uncertainties.append(task["uncertainty"])

        else:
            # Fast heuristic-based UQ from response (default for batch mode)
            for task in prompt_tasks:
                uncertainty_score = estimate_uncertainty_from_response(task["response"], task["candidates"])
                task["uncertainty_info"] = {"uncertainty": uncertainty_score, "method": "heuristic"}
                task["uncertainty"] = uncertainty_score
                all_uncertainties.append(uncertainty_score)

        # =====================================================================
        # PHASE 3.5: Constrained assignment (one-to-one field mapping)
        # =====================================================================
        # Group tasks by sample, then within each sample enforce that each
        # target field is assigned to at most one source field.
        if self.config.use_constrained_assignment:
            self._run_constrained_assignment(prompt_tasks)
        else:
            logger.info("Phase 3.5: Skipped (constrained assignment disabled)")

        # Second pass: detect errors and reflection triggers
        for task_idx, task in enumerate(prompt_tasks):
            ranking = task.get("ranking", [])
            response = task.get("response", "")
            uncertainty_score = task.get("uncertainty", 0.5)

            # Detect mapping errors
            detected_errors = detect_mapping_errors(
                task["src_field"], task["source_product"], task["target_product"],
                ranking, task["candidates"]
            )
            task["detected_errors"] = detected_errors

            # Check if reflection is needed based on enabled signals
            any_reflection_enabled = (
                self.config.use_self_reflection or
                self.config.use_error_reflection or
                self.config.use_uq_reflection or
                self.config.use_neighborhood_reflection or
                self.config.use_reflection  # Legacy flag
            )

            if any_reflection_enabled:
                # Calculate uncertainty threshold
                if self.config.uq_threshold is not None:
                    # Use fixed threshold
                    threshold = self.config.uq_threshold
                elif self.config.uq_threshold_percentile is not None and len(all_uncertainties) > 1:
                    # Use percentile-based threshold
                    threshold = np.percentile(all_uncertainties, self.config.uq_threshold_percentile)
                elif len(all_uncertainties) > 1:
                    # Default: adaptive threshold based on mean + std
                    mean_u = np.mean(all_uncertainties)
                    std_u = np.std(all_uncertainties)
                    threshold = max(0.25, mean_u + 1.5 * std_u)
                else:
                    threshold = 0.6

                high_uncertainty = uncertainty_score > threshold
                task["high_uncertainty"] = high_uncertainty

                # Determine which signals trigger reflection
                triggers = []

                # Self-reflection: always triggers (no external signal needed)
                if self.config.use_self_reflection:
                    triggers.append("self")

                # Error-based reflection
                if (self.config.use_error_reflection or self.config.use_reflection) and len(detected_errors) > 0:
                    triggers.append("error")

                # UQ-based reflection
                if (self.config.use_uq_reflection or self.config.use_reflection) and high_uncertainty:
                    triggers.append("uq")

                # Assignment conflict reflection: evicted or unmatched fields
                assignment_status = task.get("assignment_status", "")
                if assignment_status in ("reassigned", "unmatched"):
                    triggers.append("assignment_conflict")

                # Neighborhood-based reflection (TODO: implement neighborhood validation signal)
                if self.config.use_neighborhood_reflection:
                    # For now, trigger if using neighborhood expansion and multiple close candidates
                    if self.config.neighbor_k > 0 and len(ranking) >= 2:
                        score_gap = ranking[0].get("score", 1) - ranking[1].get("score", 0)
                        if score_gap < 0.2:  # Close competition suggests neighborhood check needed
                            triggers.append("neighborhood")

                task["reflection_triggers"] = triggers
                should_reflect = len(triggers) > 0

                if should_reflect:
                    task["needs_reflection"] = True
                    reflection_tasks.append(task_idx)

        # Build sibling mappings (initial top-1 predictions grouped by sample)
        sample_sibling_mappings: Dict[int, Dict[str, str]] = {}
        for task in prompt_tasks:
            sample_idx = task["sample_idx"]
            if sample_idx not in sample_sibling_mappings:
                sample_sibling_mappings[sample_idx] = {}
            ranking = task.get("ranking", [])
            if ranking and isinstance(ranking[0], dict):
                sample_sibling_mappings[sample_idx][task["src_field"]] = ranking[0].get("id", "")

        # Build reflection prompts with sibling context
        for task_idx in reflection_tasks:
            task = prompt_tasks[task_idx]
            triggers = task.get("reflection_triggers", [])
            sibling_mappings = sample_sibling_mappings.get(task["sample_idx"], {})
            reflection_prompt = self._build_reflection_prompt(
                task["src_field"], task["src_desc"], task["ranking"], task["candidates"],
                task["uncertainty"], task.get("high_uncertainty", False),
                task["source_product"], task["target_product"],
                detected_errors=task.get("detected_errors", []) if "error" in triggers else [],
                reflection_triggers=triggers,
                sibling_mappings=sibling_mappings
            )
            task["reflection_prompt"] = reflection_prompt

        # =====================================================================
        # PHASE 4: Batch inference for reflections (if any)
        # =====================================================================
        any_reflection_enabled = (
            self.config.use_self_reflection or
            self.config.use_error_reflection or
            self.config.use_uq_reflection or
            self.config.use_neighborhood_reflection or
            self.config.use_reflection
        )
        if reflection_tasks and any_reflection_enabled:
            logger.info(f"Phase 4: Batch inference for {len(reflection_tasks)} reflections...")
            reflection_prompts = [prompt_tasks[idx]["reflection_prompt"] for idx in reflection_tasks]
            reflection_responses = self.client.generate_batch(
                reflection_prompts,
                use_cache=True,
                batch_size=self.config.batch_size
            )

            for task_idx, response in zip(reflection_tasks, reflection_responses):
                task = prompt_tasks[task_idx]
                reflection = self._parse_reflection_response(response, task["candidates"], task["ranking"])
                reflection["high_uncertainty"] = task.get("high_uncertainty", False)
                reflection["detected_errors"] = task.get("detected_errors", [])
                task["reflection"] = reflection
        else:
            logger.info("Phase 4: Skipped (no reflections needed)")

        # =====================================================================
        # PHASE 5: Assemble results by sample
        # =====================================================================
        logger.info("Phase 5: Assembling results...")

        # Group tasks by sample
        sample_tasks = {}
        for task in prompt_tasks:
            sample_idx = task["sample_idx"]
            if sample_idx not in sample_tasks:
                sample_tasks[sample_idx] = []
            sample_tasks[sample_idx].append(task)

        # Build results for each sample
        for sample_idx, sample in samples_to_process:
            tasks = sample_tasks.get(sample_idx, [])
            if not tasks:
                continue

            sample = tasks[0]["sample"]
            ground_truth = sample.get("ground_truth_mapping", {})

            sample_result = {
                "sample_id": sample.get("sample_id", f"sample_{sample_idx}"),
                "utterance": sample.get("utterance", ""),
                "expected_source": sample.get("expected_source", ""),
                "expected_target": sample.get("expected_target", ""),
                "ground_truth_mapping": ground_truth,
                "predictions": {},
            }

            # Determine source/target products for field prediction
            expected_source = sample.get("expected_source", "")
            expected_target = sample.get("expected_target", "")
            if isinstance(expected_source, dict):
                source_product = expected_source.get("name", "SOURCE").upper()
            else:
                source_product = str(expected_source).split("_")[0].upper() if expected_source else "SOURCE"
            if isinstance(expected_target, dict):
                target_product = expected_target.get("name", "TARGET").upper()
            else:
                target_product = str(expected_target).split("_")[0].upper() if expected_target else "TARGET"
            if self.config.retrieval_mode:
                target_product = "TARGET"  # the prompt must not name the product being searched for

            sample_uncertainties = []
            for task in tasks:
                src_field = task["src_field"]
                sample_uncertainties.append(task["uncertainty"])

                # Get final ranking (reflection if available)
                ranking = task["ranking"]
                reflection = task.get("reflection")
                any_reflection = (
                    self.config.use_reflection or self.config.use_self_reflection or
                    self.config.use_error_reflection or self.config.use_uq_reflection or
                    self.config.use_neighborhood_reflection
                )
                if any_reflection and reflection and reflection.get("revised_ranking"):
                    final_ranking = reflection["revised_ranking"]
                else:
                    final_ranking = ranking

                # Field-level prediction
                field_prediction = None
                if self.config.predict_fields and final_ranking and isinstance(final_ranking[0], dict):
                    top_type = normalize_field_id(final_ranking[0].get("id", ""))
                    if top_type:
                        src_info = self.property_lookup.get(src_field, {})
                        src_desc = src_info.get("description", src_field) if isinstance(src_info, dict) else str(src_info)
                        field_prediction = self._predict_field_in_type(
                            src_field, src_desc, top_type,
                            source_product, target_product
                        )

                sample_result["predictions"][src_field] = {
                    "initial_ranking": ranking,
                    "none_score": getattr(task.get("ranking"), "none_score", None),
                    "none_rank": getattr(task.get("ranking"), "none_rank", None),
                    "parse_status": getattr(task.get("ranking"), "status", None),
                    "reflection": reflection,
                    "field_prediction": field_prediction,
                    "uncertainty": float(task["uncertainty"]),
                    "uncertainty_info": task["uncertainty_info"],
                    "detected_errors": task.get("detected_errors", []),
                }

            # Evaluate sample (both type-level and field-level)
            type_correct_count = 0
            field_correct_count = 0
            total_count = 0
            for src_field, gt_target in ground_truth.items():
                if src_field in sample_result["predictions"]:
                    pred = sample_result["predictions"][src_field]
                    if any_reflection and pred.get("reflection"):
                        revised = pred["reflection"].get("revised_ranking", [])
                    else:
                        revised = pred.get("initial_ranking", [])

                    # Extract ground truth type
                    gt_target_str = gt_target[0] if isinstance(gt_target, list) else gt_target
                    gt_type = gt_target_str.split('.')[0] if '.' in gt_target_str else gt_target_str

                    if revised and isinstance(revised[0], dict):
                        top_pred = normalize_field_id(revised[0].get("id", ""))
                        # Type-level accuracy
                        if isinstance(gt_target, list):
                            gt_types = [t.split('.')[0] for t in gt_target]
                            if top_pred in gt_types:
                                type_correct_count += 1
                        elif top_pred == gt_type:
                            type_correct_count += 1

                    # Field-level accuracy
                    field_pred = pred.get("field_prediction")
                    if field_pred:
                        pred_full_field = field_pred.get("full_field", "")
                        if isinstance(gt_target, list):
                            if pred_full_field in gt_target:
                                field_correct_count += 1
                        elif pred_full_field == gt_target_str:
                            field_correct_count += 1

                    total_count += 1

            sample_result["type_accuracy"] = type_correct_count / total_count if total_count > 0 else 0.0
            sample_result["field_accuracy"] = field_correct_count / total_count if total_count > 0 else 0.0
            sample_result["accuracy"] = sample_result["type_accuracy"]  # Legacy
            sample_result["type_correct_count"] = type_correct_count
            sample_result["field_correct_count"] = field_correct_count
            sample_result["correct_count"] = type_correct_count  # Legacy
            sample_result["total_count"] = total_count
            sample_result["mean_uncertainty"] = float(np.mean(sample_uncertainties)) if sample_uncertainties else 0.5

            results.append(sample_result)

            # Checkpoint periodically
            if len(results) % self.config.checkpoint_interval == 0:
                checkpoint_data = {
                    "dataset": self.dataset,
                    "model": self.model,
                    "results": results,
                    "all_uncertainties": all_uncertainties,
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, cls=NumpyEncoder)
                logger.info(f"Checkpoint saved: {len(results)} samples")

        return self._build_final_results(results, all_uncertainties, time.time() - total_start)

    def _build_final_results(self, results: List[Dict], all_uncertainties: List[float], total_time: float) -> Dict:
        """Build the final results dictionary."""
        total_type_correct = sum(r.get("type_correct_count", r.get("correct_count", 0)) for r in results)
        total_field_correct = sum(r.get("field_correct_count", 0) for r in results)
        total_fields = sum(r["total_count"] for r in results)
        type_accuracy = total_type_correct / total_fields if total_fields > 0 else 0.0
        field_accuracy = total_field_correct / total_fields if total_fields > 0 else 0.0

        cache_stats = self.client.get_cache_stats()

        return {
            "dataset": self.dataset,
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "batch_size": self.config.batch_size,
                "neighbor_k": self.config.neighbor_k,
                "retrieval_mode": self.config.retrieval_mode,
                "top_k": self.config.top_k,
                "predict_fields": self.config.predict_fields,
                # Reflection signals
                "use_self_reflection": self.config.use_self_reflection,
                "use_error_reflection": self.config.use_error_reflection,
                "use_uq_reflection": self.config.use_uq_reflection,
                "use_neighborhood_reflection": self.config.use_neighborhood_reflection,
                # Legacy
                "use_uncertainty": self.config.use_uncertainty,
                "use_reflection": self.config.use_reflection,
                "n_uncertainty_samples": self.config.n_uncertainty_samples,
                "uq_method": self.config.uq_method,
                # Reproducibility
                "seed": self.config.seed,
                "sample_main_decoding": self.config.sample_main_decoding,
            },
            "metrics": {
                "total_samples": len(results),
                "total_fields": total_fields,
                "type_correct": total_type_correct,
                "field_correct": total_field_correct,
                "type_accuracy": type_accuracy,
                "field_accuracy": field_accuracy,
                "correct_predictions": total_type_correct,  # Legacy
                "accuracy": type_accuracy,  # Legacy: type accuracy
                "mean_uncertainty": float(np.mean(all_uncertainties)) if all_uncertainties else 0.5,
                "total_time_seconds": total_time,
                "samples_per_second": len(results) / total_time if total_time > 0 else 0,
            },
            "cache_stats": cache_stats,
            "results": results
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Map-Reflect Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument("--datasets", required=True, help="Comma-separated datasets")
    parser.add_argument("--models", required=True, help="Model ID")
    parser.add_argument("--property-lookup-file", required=True, help="Property lookup JSON")
    parser.add_argument("--data-dir", required=True, help="Data directory")

    # Batch processing
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size (default: 50)")
    parser.add_argument("--no-cache", action="store_true", help="Disable response caching")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature (default: 0.1, use 0 for deterministic)")

    # Candidate retrieval
    parser.add_argument("--top-k", type=int, default=10, help="Top-K candidates (default: 10)")
    parser.add_argument("--neighbor-k", type=int, default=5, help="1-hop neighbors (default: 5, 0 to disable)")
    parser.add_argument("--retrieval-mode", action="store_true",
                        help="Target-free: candidates from hybrid search over the whole catalogue; "
                             "the record's target is used only for scoring")
    parser.add_argument("--exclude-source-tool", action="store_true",
                        help="ToolBench retrieval: also exclude every API of the source tool")

    # Field-level prediction (default on)
    parser.add_argument("--no-field-prediction", action="store_true",
                        help="Disable field-level prediction (only predict target type)")

    # Reflection signals (use these for clean ablation)
    parser.add_argument("--use-self-reflection", action="store_true",
                        help="Enable self-reflection (no external signals)")
    parser.add_argument("--use-error-reflection", action="store_true",
                        help="Enable error-triggered reflection")
    parser.add_argument("--use-uq-reflection", action="store_true",
                        help="Enable uncertainty-triggered reflection")
    parser.add_argument("--use-neighborhood-reflection", action="store_true",
                        help="Enable neighborhood-triggered reflection")

    # Legacy flags (for backward compatibility)
    parser.add_argument("--use-uncertainty", action="store_true", help="Enable uncertainty quantification")
    parser.add_argument("--use-reflection", action="store_true", help="Legacy: enable error+uq reflection")
    parser.add_argument("--n-uncertainty-samples", type=int, default=3, help="Samples for UQ (default: 3)")
    parser.add_argument("--uq-sampling-temperature", type=float, default=0.7,
                        help="Temperature for UQ resampling (sampling/lmp_sampling/simba); default 0.7")
    parser.add_argument("--uq-method", type=str, default="sampling",
                        choices=["sampling", "heuristic", "perplexity", "p_true", "semantic_entropy", "arith_agg", "harm_agg", "simba", "lmp_sampling"],
                        help="UQ method: sampling, heuristic, perplexity, p_true, semantic_entropy, arith_agg (SIMBA arithmetic), harm_agg (SIMBA harmonic), lmp_sampling (lm-polygraph sampling family)")

    # UQ threshold configuration
    parser.add_argument("--uq-threshold", type=float, help="Fixed UQ threshold (overrides adaptive)")
    parser.add_argument("--uq-threshold-percentile", type=float, help="Percentile-based UQ threshold (e.g., 50 = median)")

    # OMOP-specific enhancements (auto-enabled for OMOP_* datasets)
    parser.add_argument("--no-omop-table-filtering", action="store_true",
                        help="Disable OMOP table filtering from utterance")
    parser.add_argument("--no-omop-type-hints", action="store_true",
                        help="Disable OMOP field type hints in prompts")

    # Checkpointing
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Save every N samples (default: 50)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    # Limiting
    parser.add_argument("--max-samples", type=int, help="Limit samples for testing")
    parser.add_argument("--no-constrained-assignment", action="store_true",
                        help="Disable one-to-one field assignment (Phase 3.5)")
    parser.add_argument("--full-list", action="store_true",
                        help="Method v3 control: the given-target first pass sees every field of the named target type")
    parser.add_argument("--candidates-file",
                        help="Method v4 control: JSON {\"<sample_id>||<field>\": [candidate ids]}; the first pass shows "
                             "exactly these, in order; fields without an entry are skipped")
    parser.add_argument("--none-option", action="store_true",
                        help="Method v5: the ranking prompt also lists NONE; its score and rank are logged, never the answer")
    parser.add_argument("--only-keys", help="Method v4 control: run only the \"<sample_id>||<field>\" keys listed")
    parser.add_argument("--evidence-first", action="store_true",
                        help="Protocol v2 control: the first pass sees the schema evidence reflection arm B adds")
    parser.add_argument("--no-assignment", action="store_true",
                        help="Protocol v2 name for --no-constrained-assignment: each source field keeps its own top-1")
    parser.add_argument("--split-manifest", type=str,
                        help="Protocol v2 split manifest (JSON with 'dev' and 'test' sample-id lists); only listed samples run")
    parser.add_argument("--split", choices=["dev", "test", "test_grouped", "all"], default="all",
                        help="With --split-manifest: which part of the manifest to run (default: both)")

    # Custom eval file (for running on subset of samples)
    parser.add_argument("--eval-file", type=str, help="Custom JSONL eval file (overrides default dataset file)")

    # Documentation context (RAG from product docs)
    parser.add_argument("--doc-context", type=str,
                        help="Path to ChromaDB with product doc chunks (e.g., /path/to/ITOpsGraph/wiki/docs/chromadb)")
    parser.add_argument("--doc-context-collection", type=str, default="itopsgraph_docs",
                        help="ChromaDB collection name (default: itopsgraph_docs)")
    parser.add_argument("--doc-context-top-k", type=int, default=3,
                        help="Number of doc chunks to retrieve per field (default: 3)")

    # Output
    parser.add_argument("--output-dir", type=str, default="map_reflect/results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (Python/numpy + endpoint_b RANDOM_SEED). Only moves outputs under --sample-decoding.")
    parser.add_argument("--map-cache-dir", type=str, default=None,
                        help="Persist LLM response cache here so MAP is computed once per (dataset, seed) and reused across configs.")
    parser.add_argument("--sample-decoding", action="store_true",
                        help="Use SAMPLE+RANDOM_SEED for main MAP/field calls (needed for seed variance). Default GREEDY preserves the paper pipeline.")

    args = parser.parse_args()

    # Seed Python/numpy RNG (endpoint_b seed flows through config -> RANDOM_SEED).
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = MapReflectConfig(
        batch_size=args.batch_size,
        use_cache=not args.no_cache,
        cache_dir=args.map_cache_dir,
        seed=args.seed,
        sample_main_decoding=args.sample_decoding,
        temperature=args.temperature,
        top_k=args.top_k,
        neighbor_k=args.neighbor_k,
        retrieval_mode=args.retrieval_mode,
        exclude_source_tool=args.exclude_source_tool,
        evidence_first=args.evidence_first,
        full_list=args.full_list,
        candidate_lists=json.load(open(args.candidates_file)) if args.candidates_file else None,
        only_keys={l.strip() for l in open(args.only_keys) if l.strip()} if args.only_keys else None,
        none_option=args.none_option,
        checkpoint_dir=args.output_dir,
        predict_fields=not args.no_field_prediction,
        # New reflection signal flags
        use_self_reflection=args.use_self_reflection,
        use_error_reflection=args.use_error_reflection,
        use_uq_reflection=args.use_uq_reflection,
        use_neighborhood_reflection=args.use_neighborhood_reflection,
        # Legacy flags
        use_uncertainty=args.use_uncertainty,
        use_reflection=args.use_reflection,
        n_uncertainty_samples=args.n_uncertainty_samples,
        uq_sampling_temperature=args.uq_sampling_temperature,
        uq_method=args.uq_method,
        # UQ threshold configuration
        uq_threshold=args.uq_threshold,
        uq_threshold_percentile=args.uq_threshold_percentile,
        # OMOP-specific (enabled by default, use --no-* to disable)
        omop_table_filtering=not args.no_omop_table_filtering,
        omop_type_hints=not args.no_omop_type_hints,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        use_constrained_assignment=not (args.no_constrained_assignment or args.no_assignment),
        doc_context_chromadb=args.doc_context,
        doc_context_collection=args.doc_context_collection,
        doc_context_top_k=args.doc_context_top_k,
    )

    for dataset in args.datasets.split(","):
        dataset = dataset.strip()
        logger.info(f"Processing dataset: {dataset}")

        # Load data
        eval_file = Path(args.eval_file) if args.eval_file else None
        if args.split_manifest:
            # Filter to the manifest first, then apply --max-samples, so smoke tests stay inside the split.
            man = json.load(open(args.split_manifest))
            # "all" also covers test_grouped, so first passes exist for every grouped-test sample whatever
            # exclusion the analysis applies (protocol v2 excludes the earlier development split from test).
            keep = (set(man.get("dev", [])) | set(man.get("test", [])) | set(man.get("test_grouped", []))
                    if args.split == "all" else set(man.get(args.split, [])))
            samples = [x for x in load_dataset(Path(args.data_dir), dataset, None, eval_file)
                       if str(x.get("sample_id", x.get("id", ""))) in keep]
            if args.max_samples:
                samples = samples[:args.max_samples]
            logger.info(f"Split manifest {args.split_manifest} ({args.split}): {len(samples)} samples")
        else:
            samples = load_dataset(Path(args.data_dir), dataset, args.max_samples, eval_file)

        # Initialize runner
        runner = MapReflectRunner(
            property_lookup_file=args.property_lookup_file,
            data_dir=args.data_dir,
            model=args.models,
            dataset=dataset,
            config=config
        )

        # Run evaluation
        results = runner.run(samples)

        # Persist the response cache so sibling configs in this (dataset, seed) reuse MAP
        runner.client.save_cache()

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = args.models.replace("/", "_").replace("endpoint_b_", "")
        output_file = output_dir / f"mapreflect_{dataset.lower()}_{model_name}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Results saved to {output_file}")
        logger.info(f"Metrics: {results['metrics']}")


if __name__ == "__main__":
    main()
