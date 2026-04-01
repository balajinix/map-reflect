#!/usr/bin/env python3
"""
Generate annotation candidates for ITOM-Hashicorp field mappings.

Uses multiple signals to suggest the best target field for each source field:
  1. Map-Reflect predictions (Llama-3.3-70B)
  2. Name matching (exact / fuzzy)
  3. Type matching (same field type)
  4. Embedding similarity (sentence-transformers)

Outputs:
  - data/itom_hashicorp_annotation_v2.md  (human-editable annotation template)
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MR_RESULTS_DIR = BASE_DIR / "map_reflect" / "results"

# Cross-product filter (same as demo)
ITOM_ONLY = {"instana", "concert", "turbonomic"}
HASHI_ONLY = {"infragraph", "packer", "aws", "azure", "gcp", "hcp"}
SHARED = {"hcp_terraform"}


def load_dataset():
    samples = []
    with open(DATA_DIR / "itom_hashicorp_dataset.jsonl") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def load_api_lookup():
    # Prefer v2 if available
    v2_path = DATA_DIR / "itom_hashicorp_api_lookup_v2.json"
    v1_path = DATA_DIR / "itom_hashicorp_api_lookup.json"
    path = v2_path if v2_path.exists() else v1_path
    with open(path) as f:
        return json.load(f)


def load_mr_results():
    """Load Map-Reflect results, preferring Llama."""
    results = {}
    for model_tag in ["ibm_granite-4-h-small", "meta-llama_llama-3-3-70b-instruct"]:
        fpath = MR_RESULTS_DIR / f"mapreflect_itom_hashicorp_{model_tag}.json"
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            for r in data.get("results", []):
                results[r["sample_id"]] = r
    return results


def get_type_fields(api_lookup, product, type_name):
    """Get all fields for a given product.TypeName.

    Excludes embedded message references (e.g., 'base', 'virtual_machine')
    that are composition artifacts, not real fields.
    """
    # Known embedded message reference field names (proto composition)
    EMBEDDED_REFS = {
        "base", "virtual_machine", "aws_resource", "azure_resource",
        "gcp_resource", "hcp_resource", "hcp_terraform_resource",
    }

    fields = {}
    prefix = f"{product}_{type_name}."
    for key, meta in api_lookup.items():
        if key.startswith(prefix):
            field_name = key[len(prefix):]
            # Skip embedded message references
            if field_name in EMBEDDED_REFS:
                continue
            fields[key] = {
                "field_name": field_name,
                "description": meta.get("description", ""),
                "field_type": meta.get("field_type", ""),
            }
    return fields


def normalize_name(name):
    """Normalize a field name for comparison."""
    # Remove product_Type. prefix
    if "." in name:
        name = name.split(".")[-1]
    # Split camelCase and snake_case, lowercase
    parts = re.findall(r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)", name)
    parts.extend(p.lower() for p in name.split("_") if p)
    return set(p.lower() for p in parts)


def name_similarity(src_name, tgt_name):
    """Compute name similarity between two field names."""
    src_base = src_name.split(".")[-1].lower() if "." in src_name else src_name.lower()
    tgt_base = tgt_name.split(".")[-1].lower() if "." in tgt_name else tgt_name.lower()

    # Exact match
    if src_base == tgt_base:
        return 1.0, "exact name match"

    # Normalized match
    src_parts = normalize_name(src_name)
    tgt_parts = normalize_name(tgt_name)
    if src_parts and tgt_parts:
        overlap = len(src_parts & tgt_parts)
        union = len(src_parts | tgt_parts)
        jaccard = overlap / union if union > 0 else 0
        if jaccard >= 0.5:
            return jaccard, f"name overlap: {src_parts & tgt_parts}"

    return 0.0, ""


def type_similarity(src_type, tgt_type):
    """Simple type compatibility score."""
    if not src_type or not tgt_type:
        return 0.0

    src_t = src_type.lower()
    tgt_t = tgt_type.lower()

    # Exact type match
    if src_t == tgt_t:
        return 0.3  # modest bonus

    # Compatible types
    string_types = {"string", "str", "text", "id"}
    num_types = {"int", "int32", "int64", "float", "double", "number", "long"}
    time_types = {"timestamp", "datetime", "date", "time"}
    bool_types = {"boolean", "bool"}

    for group in [string_types, num_types, time_types, bool_types]:
        if src_t in group and tgt_t in group:
            return 0.2

    return 0.0


_embedding_model = None

def get_embedding_model(model_name="all-MiniLM-L6-v2"):
    """Load embedding model once and cache it."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"  Warning: Could not load embedding model: {e}")
    return _embedding_model


def compute_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """Compute embeddings for a list of texts."""
    model = get_embedding_model(model_name)
    if model is None:
        return None
    try:
        return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    except Exception as e:
        print(f"  Warning: Could not compute embeddings: {e}")
        return None


def generate_candidates(src_field_key, src_meta, tgt_fields, mr_prediction, src_embeddings=None, tgt_embeddings=None, tgt_keys=None, src_idx=None):
    """Generate ranked candidates for a source field.

    Returns list of {id, score, source, reason}.
    """
    candidates = {}  # id -> {score, source, reason}

    src_name = src_meta["field_name"]
    src_desc = src_meta.get("description", "")
    src_type = src_meta.get("field_type", "")

    # Valid target field keys (only from the actual target type)
    valid_targets = set(tgt_fields.keys())

    # Signal 1: Map-Reflect predictions (filter to target type only)
    if mr_prediction:
        ranking = mr_prediction.get("initial_ranking", [])
        for i, r in enumerate(ranking[:5]):
            rid = r.get("id", "")
            if rid and rid != "NO_MATCH" and rid in valid_targets:
                score = r.get("score", 0)
                reason = r.get("reason", "")[:80]
                if rid not in candidates or score > candidates[rid]["score"]:
                    candidates[rid] = {
                        "score": score,
                        "source": "map-reflect",
                        "reason": f"MR rank#{i+1}: {reason}",
                    }

    # Signal 2: Name matching
    for tgt_key, tgt_meta in tgt_fields.items():
        name_sim, name_reason = name_similarity(src_name, tgt_meta["field_name"])
        if name_sim > 0:
            # Scale name match score
            score = 0.5 + 0.45 * name_sim  # 0.5 to 0.95
            if tgt_key not in candidates or score > candidates[tgt_key]["score"]:
                candidates[tgt_key] = {
                    "score": round(score, 2),
                    "source": "name-match",
                    "reason": name_reason,
                }

    # Signal 3: Type matching (small bonus, doesn't override)
    for tgt_key, tgt_meta in tgt_fields.items():
        type_sim = type_similarity(src_type, tgt_meta.get("field_type", ""))
        if type_sim > 0 and tgt_key in candidates:
            # Add type bonus to existing candidate
            candidates[tgt_key]["score"] = min(1.0, candidates[tgt_key]["score"] + type_sim)
            candidates[tgt_key]["reason"] += f" +type({src_type})"

    # Signal 4: Embedding similarity
    if src_embeddings is not None and tgt_embeddings is not None and src_idx is not None:
        src_emb = src_embeddings[src_idx]
        sims = np.dot(tgt_embeddings, src_emb)
        top_indices = np.argsort(sims)[::-1][:5]
        for rank, idx in enumerate(top_indices):
            sim = float(sims[idx])
            if sim < 0.3:
                continue
            tgt_key = tgt_keys[idx]
            # Scale to 0.3 - 0.9
            score = round(0.3 + 0.6 * sim, 2)
            if tgt_key not in candidates or score > candidates[tgt_key]["score"]:
                candidates[tgt_key] = {
                    "score": score,
                    "source": "embedding",
                    "reason": f"cosine={sim:.2f}",
                }

    # Sort by score, deduplicate
    sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1]["score"])
    result = []
    for cid, cdata in sorted_candidates[:5]:
        result.append({
            "id": cid,
            "score": cdata["score"],
            "source": cdata["source"],
            "reason": cdata["reason"],
        })

    return result


def build_field_text(field_key, meta):
    """Build a text description for embedding."""
    name = meta.get("field_name", field_key.split(".")[-1])
    desc = meta.get("description", "")
    ftype = meta.get("field_type", "")
    parts = [name]
    if desc:
        parts.append(desc)
    if ftype:
        parts.append(f"type: {ftype}")
    return " — ".join(parts)


def generate_annotation_template(samples, api_lookup, mr_results):
    """Generate the annotation markdown template with candidates."""

    # Filter to cross-product only
    cross_samples = []
    for s in samples:
        src = s["source_schema"]
        tgt = s["target_schema"]
        is_cross = (
            (src in ITOM_ONLY and tgt not in ITOM_ONLY) or
            (tgt in ITOM_ONLY and src not in ITOM_ONLY)
        )
        if is_cross:
            cross_samples.append(s)

    print(f"\n{len(cross_samples)} cross-product samples to annotate")

    lines = []
    lines.append("# ITOM-Hashicorp Field Annotation (v2)")
    lines.append("")
    lines.append("Instructions:")
    lines.append("- For each source field, choose the best target field from the candidates")
    lines.append("- Write the target field key in the 'Your Choice' column")
    lines.append("- Write `--` for fields with no reasonable match")
    lines.append("- Each target field should be used AT MOST ONCE per type pair (bipartite constraint)")
    lines.append("")
    lines.append(f"Total: {len(cross_samples)} type pairs to annotate")
    lines.append("")

    all_annotations = {}  # sample_id -> data for JSON output

    for si, sample in enumerate(cross_samples):
        sid = sample["id"]
        src_product = sample["source_schema"]
        tgt_product = sample["target_schema"]
        meta = sample["metadata"]
        src_type = meta["source_type"]
        tgt_type = meta["target_type"]
        relationship = meta["relationship"]
        description = meta["description"]

        print(f"  [{si+1}/{len(cross_samples)}] {src_product}.{src_type} -> {tgt_product}.{tgt_type}")

        # Get fields
        src_fields = get_type_fields(api_lookup, src_product, src_type)
        tgt_fields = get_type_fields(api_lookup, tgt_product, tgt_type)

        if not src_fields:
            print(f"    WARNING: No source fields found for {src_product}.{src_type}")
            continue
        if not tgt_fields:
            print(f"    WARNING: No target fields found for {tgt_product}.{tgt_type}")
            continue

        # Get MR predictions
        mr_result = mr_results.get(sid, {})
        mr_preds = mr_result.get("predictions", {})

        # Compute embeddings for this type pair
        src_texts = [build_field_text(k, v) for k, v in src_fields.items()]
        tgt_texts = [build_field_text(k, v) for k, v in tgt_fields.items()]
        src_keys = list(src_fields.keys())
        tgt_keys = list(tgt_fields.keys())

        src_embs = compute_embeddings(src_texts) if src_texts and tgt_texts else None
        tgt_embs = compute_embeddings(tgt_texts) if src_texts and tgt_texts else None

        # Generate section header
        lines.append("---")
        lines.append("")
        lines.append(f"## {sid}: {src_product}.{src_type} → {tgt_product}.{tgt_type}")
        lines.append(f"**Relationship:** {relationship}")
        lines.append(f"**Description:** {description}")
        lines.append(f"**Source fields:** {len(src_fields)} | **Target fields:** {len(tgt_fields)}")
        lines.append("")

        # List target fields for reference
        tgt_field_names = [f"`{m['field_name']}`" for m in tgt_fields.values()]
        lines.append(f"Target fields: {', '.join(tgt_field_names[:30])}")
        if len(tgt_field_names) > 30:
            lines.append(f"  ... and {len(tgt_field_names) - 30} more")
        lines.append("")

        # Table header
        lines.append("| Source Field | Type | Candidates (ranked) | Your Choice |")
        lines.append("|-------------|------|---------------------|-------------|")

        sample_annotations = {
            "source": f"{src_product}.{src_type}",
            "target": f"{tgt_product}.{tgt_type}",
            "relationship": relationship,
            "description": description,
            "annotated": False,
            "mappings": {},
        }

        for src_idx, (src_key, src_meta) in enumerate(src_fields.items()):
            src_name = src_meta["field_name"]
            src_type_str = src_meta.get("field_type", "?")

            mr_pred = mr_preds.get(src_key, {})
            candidates = generate_candidates(
                src_key, src_meta, tgt_fields, mr_pred,
                src_embeddings=src_embs, tgt_embeddings=tgt_embs,
                tgt_keys=tgt_keys, src_idx=src_idx,
            )

            # Format candidates
            cand_strs = []
            for c in candidates[:3]:
                cid_short = c["id"].split(".")[-1] if "." in c["id"] else c["id"]
                source_tag = c["source"][:2].upper()  # MR, NA, EM, TY
                cand_strs.append(f"{cid_short} ({c['score']:.2f}, {source_tag})")

            cand_display = " / ".join(cand_strs) if cand_strs else "—"

            # Pre-fill best candidate
            best = candidates[0]["id"] if candidates else "--"
            best_short = best.split(".")[-1] if "." in best else best

            lines.append(f"| `{src_name}` | {src_type_str} | {cand_display} | {best_short} |")

            # Store for JSON
            sample_annotations["mappings"][src_key] = {
                "candidates": candidates[:5],
                "default_choice": best,
            }

        lines.append("")
        all_annotations[sid] = sample_annotations

    return "\n".join(lines), all_annotations


def main():
    print("=== Generating ITOM-Hashicorp Annotation Candidates ===")

    print("\nLoading data...")
    samples = load_dataset()
    api_lookup = load_api_lookup()
    mr_results = load_mr_results()
    print(f"  {len(samples)} samples, {len(api_lookup)} lookup entries, {len(mr_results)} MR results")

    print("\nGenerating candidates...")
    template, annotations = generate_annotation_template(samples, api_lookup, mr_results)

    # Save markdown template
    md_path = DATA_DIR / "itom_hashicorp_annotation_v2.md"
    with open(md_path, "w") as f:
        f.write(template)
    print(f"\nSaved annotation template: {md_path}")

    # Save JSON candidates (for programmatic use)
    json_path = DATA_DIR / "itom_hashicorp_annotation_candidates.json"
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"Saved annotation candidates: {json_path}")

    # Summary
    total_fields = sum(len(a["mappings"]) for a in annotations.values())
    print(f"\nSummary: {len(annotations)} type pairs, {total_fields} fields to annotate")


if __name__ == "__main__":
    main()
