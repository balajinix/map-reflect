"""
ITOM-Hashicorp Schema Mapping Demo + Annotation Tool

Streamlit app showing schema mappings between ITOM products (Instana,
Concert, Turbonomic, HCP Terraform) and Hashicorp products (Infragraph,
Packer, AWS, Azure, GCP, HCP).

Features:
  - Type pair cards with 3-column layout
  - Pre-computed candidate field mappings (Map-Reflect + embedding + name match)
  - Interactive annotation: accept/reject/correct each field mapping
  - Bipartite validation (each target field used at most once)
  - Save/load annotations to JSON
  - Export annotated dataset
"""

import streamlit as st
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Page config
st.set_page_config(
    page_title="ITOM–Hashicorp Schema Mapping",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRODUCT_COLORS = {
    "instana": "#1E88E5",
    "concert": "#7B1FA2",
    "turbonomic": "#00897B",
    "hcp_terraform": "#F4511E",
    "infragraph": "#6D4C41",
    "packer": "#3949AB",
    "aws": "#FF8F00",
    "azure": "#0078D4",
    "gcp": "#4285F4",
    "hcp": "#000000",
    "core_infrastructure": "#546E7A",
}

ITOM_ONLY = {"instana", "concert", "turbonomic"}
HASHI_ONLY = {"infragraph", "packer", "aws", "azure", "gcp", "hcp"}
SHARED = {"hcp_terraform"}
ITOM_PRODUCTS = ITOM_ONLY | SHARED
HASHI_PRODUCTS = HASHI_ONLY | SHARED

DATA_DIR = Path(__file__).parent / "data"
ANNOTATIONS_PATH = DATA_DIR / "itom_hashicorp_annotations.json"


@st.cache_data
def load_data():
    base = DATA_DIR
    mr_results_dir = Path(__file__).parent / "map_reflect" / "results"

    with open(base / "itom_hashicorp_entity_model.json") as f:
        entity_model = json.load(f)

    with open(base / "itom_hashicorp_entity_mapping.json") as f:
        entity_mapping = json.load(f)

    samples = []
    with open(base / "itom_hashicorp_dataset.jsonl") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Prefer v2 lookup
    lookup_path = base / "itom_hashicorp_api_lookup_v2.json"
    if not lookup_path.exists():
        lookup_path = base / "itom_hashicorp_api_lookup.json"
    with open(lookup_path) as f:
        api_lookup = json.load(f)

    # Map-Reflect results (prefer Llama, backfill with Granite)
    results = {}
    for model_tag in ["ibm_granite-4-h-small", "meta-llama_llama-3-3-70b-instruct"]:
        fpath = mr_results_dir / f"mapreflect_itom_hashicorp_{model_tag}.json"
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            for r in data.get("results", []):
                sid = r["sample_id"]
                results[sid] = r
                results[sid]["_model"] = model_tag

    # Pre-computed annotation candidates
    candidates = {}
    cand_path = base / "itom_hashicorp_annotation_candidates.json"
    if cand_path.exists():
        with open(cand_path) as f:
            candidates = json.load(f)

    # ITOpsGraph type mapping lookup (which methods suggested each type mapping)
    type_mapping_lookup = {}
    tml_path = base / "itopsgraph_type_mapping_lookup.json"
    if tml_path.exists():
        with open(tml_path) as f:
            type_mapping_lookup = json.load(f)

    # Build type_name -> product mapping from api_lookup
    type_to_product = {}
    for key, meta in api_lookup.items():
        p = meta.get("product_name", "")
        t = meta.get("type_name", "")
        if p and t and t not in type_to_product:
            type_to_product[t] = p

    # Generate additional samples from ITOpsGraph type mapping lookup
    existing_pairs = set()
    for s in samples:
        existing_pairs.add((s["metadata"]["source_type"], s["metadata"]["target_type"]))

    tml_sample_id = 1000
    for src_type, targets in type_mapping_lookup.items():
        if src_type not in type_to_product:
            continue
        src_product = type_to_product[src_type]

        for tgt_type, methods in targets.items():
            if tgt_type not in type_to_product:
                continue
            if src_type == tgt_type:
                continue
            if (src_type, tgt_type) in existing_pairs:
                continue

            tgt_product = type_to_product[tgt_type]
            existing_pairs.add((src_type, tgt_type))

            # Build field-level mapping from api_lookup
            src_prefix = f"{src_product}_{src_type}."
            tgt_prefix = f"{tgt_product}_{tgt_type}."
            src_fields = [k for k in api_lookup if k.startswith(src_prefix)]
            tgt_fields = [k for k in api_lookup if k.startswith(tgt_prefix)]

            mapping = {}
            if tgt_fields:
                tgt_field_names = {k.split(".")[-1].lower(): k for k in tgt_fields}
                for sf in src_fields:
                    sf_name = sf.split(".")[-1].lower()
                    if sf_name in tgt_field_names:
                        mapping[sf] = tgt_field_names[sf_name]
                    else:
                        mapping[sf] = tgt_fields[0]  # placeholder

            sample = {
                "id": f"itom_hashi_{tml_sample_id:04d}",
                "utterance": f"Map {src_product} {src_type} to {tgt_product} schema",
                "source_schema": src_product,
                "target_schema": tgt_product,
                "expected_target_type": f"{tgt_product}_{tgt_type}",
                "mapping": mapping,
                "metadata": {
                    "source_type": src_type,
                    "target_type": tgt_type,
                    "relationship": "maps_to",
                    "description": f"Mapping suggested by {', '.join(methods)}",
                    "source_entity": src_type,
                    "target_entity": tgt_type,
                    "itopsgraph_methods": methods,
                },
            }
            samples.append(sample)
            tml_sample_id += 1

    return entity_model, entity_mapping, samples, api_lookup, results, candidates, type_mapping_lookup


def load_annotations():
    """Load saved annotations from disk."""
    if ANNOTATIONS_PATH.exists():
        with open(ANNOTATIONS_PATH) as f:
            return json.load(f)
    return {}


def save_annotations(annotations):
    """Save annotations to disk."""
    with open(ANNOTATIONS_PATH, "w") as f:
        json.dump(annotations, f, indent=2)


def product_badge(product):
    color = PRODUCT_COLORS.get(product, "#999")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;font-weight:bold">{product}</span>'


def get_target_fields(api_lookup, product, type_name):
    """Get all target field names for a type (excluding embedded refs)."""
    EMBEDDED_REFS = {
        "base", "virtual_machine", "aws_resource", "azure_resource",
        "gcp_resource", "hcp_resource", "hcp_terraform_resource",
    }
    prefix = f"{product}_{type_name}."
    fields = []
    for key in api_lookup:
        if key.startswith(prefix):
            field_name = key[len(prefix):]
            if field_name not in EMBEDDED_REFS:
                fields.append(field_name)
    return sorted(fields)


def render_annotation_ui(sample, candidates_data, api_lookup):
    """Render interactive annotation controls for a type pair.

    Returns dict of {src_field_short: chosen_target_short_or_None} if user
    clicks Save, else None.
    """
    sid = sample["id"]
    meta = sample["metadata"]
    src_product = sample["source_schema"]
    tgt_product = sample["target_schema"]
    src_type = meta["source_type"]
    tgt_type = meta["target_type"]

    # Get candidate data for this sample
    cand_info = candidates_data.get(sid, {})
    field_candidates = cand_info.get("mappings", {})

    # Get all target fields for the dropdown
    all_target_fields = get_target_fields(api_lookup, tgt_product, tgt_type)

    # Load existing annotations
    if "annotations" not in st.session_state:
        st.session_state.annotations = load_annotations()
    saved = st.session_state.annotations.get(sid, {})
    saved_mappings = saved.get("mappings", {})

    # Count annotations
    annotated_count = sum(
        1 for v in saved_mappings.values() if v.get("target") is not None or v.get("no_match")
    )
    total_fields = len(field_candidates)

    # Expander with progress
    progress_label = f"{annotated_count}/{total_fields}" if total_fields > 0 else "0/0"
    is_complete = annotated_count == total_fields and total_fields > 0

    expander_label = f"Field Annotations ({total_fields} fields) — {progress_label} annotated"
    if is_complete:
        expander_label += " ✓"

    with st.expander(expander_label, expanded=False):
        if not field_candidates:
            st.info("No candidates available. Run `scripts/generate_annotation_candidates.py` first.")
            return

        # Bipartite tracking: which targets are claimed
        claimed_targets = defaultdict(list)
        # Store per-field options for use in save handler
        field_options = {}  # src_short -> options list

        for src_key, cand_data in field_candidates.items():
            src_short = src_key.split(".")[-1] if "." in src_key else src_key
            src_field_meta = api_lookup.get(src_key, {})
            src_desc = src_field_meta.get("description", "")
            src_ftype = src_field_meta.get("field_type", "")

            candidates = cand_data.get("candidates", [])

            # Build options for selectbox
            options = []
            option_labels = []

            # Add candidates
            for c in candidates[:5]:
                cid = c["id"]
                c_short = cid.split(".")[-1] if "." in cid else cid
                score = c.get("score", 0)
                source = c.get("source", "?")
                tag = source[:2].upper()
                options.append(c_short)
                option_labels.append(f"{c_short} ({score:.2f}, {tag})")

            # Add remaining target fields not in candidates
            candidate_shorts = {(c["id"].split(".")[-1] if "." in c["id"] else c["id"]) for c in candidates[:5]}
            for tf in all_target_fields:
                if tf not in candidate_shorts:
                    options.append(tf)
                    option_labels.append(tf)

            # Add special options
            options.append("__NO_MATCH__")
            option_labels.append("No Match")

            # Store for save handler
            field_options[src_short] = options

            # Determine default selection
            saved_field = saved_mappings.get(src_short, {})
            if saved_field:
                if saved_field.get("no_match"):
                    default_idx = len(options) - 1  # No Match
                elif saved_field.get("target"):
                    tgt = saved_field["target"]
                    try:
                        default_idx = options.index(tgt)
                    except ValueError:
                        default_idx = 0
                else:
                    default_idx = 0
            else:
                default_idx = 0  # First candidate

            # Render row
            cols = st.columns([2, 1, 4])
            with cols[0]:
                type_tag = f" `{src_ftype}`" if src_ftype else ""
                st.markdown(f"**{src_short}**{type_tag}")
                if src_desc:
                    st.caption(src_desc[:100])
            with cols[1]:
                pass  # spacer
            with cols[2]:
                choice = st.selectbox(
                    f"Target for {src_short}",
                    options=range(len(options)),
                    format_func=lambda i, labels=option_labels: labels[i],
                    index=default_idx,
                    key=f"ann_{sid}_{src_short}",
                    label_visibility="collapsed",
                )
                chosen = options[choice]
                if chosen != "__NO_MATCH__":
                    claimed_targets[chosen].append(src_short)

        # Bipartite violations
        violations = {t: srcs for t, srcs in claimed_targets.items() if len(srcs) > 1}
        if violations:
            violation_strs = [
                f"**{t}** claimed by: {', '.join(srcs)}"
                for t, srcs in violations.items()
            ]
            st.warning(
                f"Bipartite violation: {len(violations)} target field(s) used more than once:\n\n"
                + "\n".join(f"- {v}" for v in violation_strs)
            )

        # Save button
        if st.button(f"Save annotations for {sid}", key=f"save_{sid}"):
            mappings = {}
            for src_key in field_candidates:
                src_short = src_key.split(".")[-1] if "." in src_key else src_key
                widget_key = f"ann_{sid}_{src_short}"
                choice_idx = st.session_state.get(widget_key, 0)
                opts = field_options.get(src_short, ["__NO_MATCH__"])
                chosen = opts[choice_idx] if choice_idx < len(opts) else "__NO_MATCH__"

                if chosen == "__NO_MATCH__":
                    mappings[src_short] = {"target": None, "no_match": True}
                else:
                    mappings[src_short] = {"target": chosen, "no_match": False}

            st.session_state.annotations[sid] = {
                "source": f"{src_product}.{src_type}",
                "target": f"{tgt_product}.{tgt_type}",
                "annotated": True,
                "timestamp": datetime.now().isoformat(),
                "mappings": mappings,
            }
            save_annotations(st.session_state.annotations)
            st.success(f"Saved {len(mappings)} field annotations for {sid}")
            if violations:
                st.warning(f"Note: {len(violations)} bipartite violation(s) remain")


def render_mapping_card(sample, result, entity_model, entity_mapping, api_lookup, candidates_data, type_mapping_lookup=None):
    """Render a single mapping card with annotation controls."""
    meta = sample["metadata"]
    src_product = sample["source_schema"]
    tgt_product = sample["target_schema"]
    src_type = meta["source_type"]
    tgt_type = meta["target_type"]
    relationship = meta["relationship"]
    description = meta["description"]

    # Concept layer: source and target entities from the data model
    src_entity = meta.get("source_entity", src_type)
    tgt_entity = meta.get("target_entity", tgt_type)

    # Look up concept descriptions and match methods from entity mapping
    src_entity_key = f"{src_product}.{src_entity}"
    tgt_entity_key = f"{tgt_product}.{tgt_entity}"
    src_em_info = entity_mapping.get(src_entity_key, {})
    tgt_em_info = entity_mapping.get(tgt_entity_key, {})
    src_entity_desc = src_em_info.get("description", "")
    tgt_entity_desc = tgt_em_info.get("description", "")

    # Find match method for concept -> API mapping
    def get_match_method(em_info, api_type):
        for t in em_info.get("matched_types", []):
            if t.get("type_name") == api_type:
                return t.get("match_method", "")
        return ""

    src_match_method = get_match_method(src_em_info, src_type)
    tgt_match_method = get_match_method(tgt_em_info, tgt_type)

    # Check if concept name differs from API type name
    src_concept_differs = src_entity != src_type
    tgt_concept_differs = tgt_entity != tgt_type

    # Entity model relationships
    em_product = entity_model.get(src_product, {}).get("entities", {})
    entity_data = em_product.get(src_entity, {})
    entity_rels = entity_data.get("relationships", [])

    predictions = result.get("predictions", {}) if result else {}
    gt = sample.get("mapping", {})
    model_tag = result.get("_model", "unknown") if result else "no result"
    model_short = "Llama" if "llama" in model_tag else "Granite" if "granite" in model_tag else model_tag

    # Count correct predictions
    correct = 0
    total = len(gt)
    for src_f, gt_t in gt.items():
        pred = predictions.get(src_f, {})
        ranking = pred.get("initial_ranking", [])
        if ranking and ranking[0].get("id", "").lower() == gt_t.lower():
            correct += 1

    # Check annotation status
    saved = st.session_state.get("annotations", {}).get(sample["id"], {})
    is_annotated = saved.get("annotated", False)

    with st.container():
        # Header row: Source API  --relationship-->  Target API
        cols = st.columns([1, 1, 1])
        with cols[0]:
            st.markdown(f"{product_badge(src_product)} **{src_type}**", unsafe_allow_html=True)
        with cols[1]:
            rel_labels = {
                "semantic_equivalent": ("same concept", "#4CAF50"),
                "monitors": ("monitors", "#1E88E5"),
                "optimizes": ("optimizes", "#7B1FA2"),
                "governs": ("governs", "#FF9800"),
                "maps_to": ("maps to", "#3F51B5"),
                "triggers": ("triggers", "#F44336"),
                "produces": ("produces", "#8BC34A"),
                "projection": ("maps to", "#3F51B5"),
                "supply_chain": ("produces", "#8BC34A"),
                "composed_of": ("maps to", "#3F51B5"),
                "derived_from": ("monitors", "#1E88E5"),
                "references": ("maps to", "#3F51B5"),
                "acts_on": ("optimizes", "#7B1FA2"),
                "schema_mapping": ("maps to", "#3F51B5"),
            }
            label, rel_color = rel_labels.get(relationship, (relationship.replace("_", " "), "#999"))
            ann_badge = ' <span style="background:#4CAF50;color:white;padding:1px 6px;border-radius:8px;font-size:0.7em">annotated</span>' if is_annotated else ""
            st.markdown(
                f'<div style="text-align:center">'
                f'<span style="background:{rel_color};color:white;padding:2px 8px;border-radius:12px;font-size:0.8em">'
                f'{label}</span>{ann_badge}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                f'<div style="text-align:right">{product_badge(tgt_product)} **{tgt_type}**</div>',
                unsafe_allow_html=True,
            )

        st.caption(f"_{description}_")

        # Concept row: show the data model concepts that motivated this API pairing
        concept_cols = st.columns([1, 1, 1])
        with concept_cols[0]:
            st.markdown(f"**Concept:** `{src_entity}`")
            if src_concept_differs:
                method_tag = f' <span style="background:#78909C;color:white;padding:1px 5px;border-radius:4px;font-size:0.7em">{src_match_method}</span>' if src_match_method else ""
                st.markdown(
                    f'<span style="font-size:0.85em">Mapped to API: <code>{src_type}</code>{method_tag}</span>',
                    unsafe_allow_html=True,
                )
            if src_entity_desc:
                st.caption(src_entity_desc[:120])

        with concept_cols[1]:
            # Show concept-level relationship context
            if entity_rels:
                relevant_rels = [
                    r for r in entity_rels
                    if tgt_entity in r.get("target", "")
                ]
                other_rels = [
                    r for r in entity_rels
                    if tgt_entity not in r.get("target", "")
                ]
                if relevant_rels:
                    for r in relevant_rels[:2]:
                        st.markdown(
                            f'<span style="color:#666;font-size:0.85em">'
                            f'{r["type"]} -> {r["target"]}</span>',
                            unsafe_allow_html=True,
                        )
                        if r.get("description"):
                            st.caption(r["description"][:100])
                if other_rels:
                    other_strs = [f'{r["target"]}' for r in other_rels[:4]]
                    st.caption(f"Other relations: {', '.join(other_strs)}")
            else:
                st.caption("No concept relationships defined")

        with concept_cols[2]:
            st.markdown(f"**Concept:** `{tgt_entity}`")
            if tgt_concept_differs:
                method_tag = f' <span style="background:#78909C;color:white;padding:1px 5px;border-radius:4px;font-size:0.7em">{tgt_match_method}</span>' if tgt_match_method else ""
                st.markdown(
                    f'<span style="font-size:0.85em">Mapped to API: <code>{tgt_type}</code>{method_tag}</span>',
                    unsafe_allow_html=True,
                )
            if tgt_entity_desc:
                st.caption(tgt_entity_desc[:120])

        # API detail row
        detail_cols = st.columns([1, 1, 1])
        with detail_cols[0]:
            src_fields = [
                (k, v)
                for k, v in api_lookup.items()
                if v.get("product_name") == src_product and v.get("type_name") == src_type
            ]
            st.markdown(f"`{src_product}.{src_type}` — {len(src_fields)} fields")

        with detail_cols[1]:
            pass

        with detail_cols[2]:
            tgt_fields = [
                (k, v)
                for k, v in api_lookup.items()
                if v.get("product_name") == tgt_product and v.get("type_name") == tgt_type
            ]
            st.markdown(f"`{tgt_product}.{tgt_type}` — {len(tgt_fields)} fields")

        # Accuracy badge (from MR predictions vs synthetic GT)
        if total > 0:
            pct = correct / total * 100
            badge_color = "#4CAF50" if pct >= 70 else "#FF9800" if pct >= 40 else "#F44336"
            st.markdown(
                f'<span style="background:{badge_color};color:white;padding:2px 10px;border-radius:12px;font-size:0.85em">'
                f'MR accuracy: {correct}/{total} ({pct:.0f}%)</span>'
                f' &nbsp; <span style="color:#888;font-size:0.8em">Model: {model_short}</span>',
                unsafe_allow_html=True,
            )

        # ITOpsGraph type mapping: which methods suggested this type pair
        if type_mapping_lookup or meta.get("itopsgraph_methods"):
            # Look up src_type -> tgt_type (the lookup uses bare type names)
            src_mappings = type_mapping_lookup.get(src_type, {})
            methods_for_pair = src_mappings.get(tgt_type, [])
            # Also check sample metadata (for ITOpsGraph-sourced samples)
            if not methods_for_pair and meta.get("itopsgraph_methods"):
                methods_for_pair = meta["itopsgraph_methods"]

            # Also check reverse direction
            tgt_mappings = type_mapping_lookup.get(tgt_type, {})
            reverse_methods = tgt_mappings.get(src_type, [])

            # Collect all other targets that src_type was mapped to by any method
            other_targets = {
                t: m for t, m in src_mappings.items()
                if t != tgt_type
            }

            # Method color coding
            METHOD_COLORS = {
                "Map-Reflect": "#1565C0",
                "ReAct": "#6A1B9A",
                "UE-Reflect": "#00838F",
                "MatchMaker": "#2E7D32",
                "ReMatch": "#E65100",
                "AnyTool": "#4527A0",
                "SMT": "#AD1457",
                "GNN": "#37474F",
            }

            def method_badge(method_name):
                # Extract method family (before the parenthesized model)
                family = method_name.split("(")[0].strip()
                color = METHOD_COLORS.get(family, "#757575")
                return (
                    f'<span style="background:{color};color:white;padding:1px 6px;'
                    f'border-radius:4px;font-size:0.75em;margin-right:3px">'
                    f'{method_name}</span>'
                )

            if methods_for_pair:
                badges = " ".join(method_badge(m) for m in methods_for_pair)
                st.markdown(
                    f'<div style="margin:4px 0">'
                    f'<span style="font-size:0.85em;color:#555">Methods that suggested this mapping:</span> '
                    f'{badges}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span style="font-size:0.85em;color:#999">No ITOpsGraph methods suggested this type mapping</span>',
                    unsafe_allow_html=True,
                )

            if reverse_methods:
                rev_badges = " ".join(method_badge(m) for m in reverse_methods)
                st.markdown(
                    f'<div style="margin:2px 0">'
                    f'<span style="font-size:0.85em;color:#555">Reverse ({tgt_type} -> {src_type}):</span> '
                    f'{rev_badges}</div>',
                    unsafe_allow_html=True,
                )

            # Show other targets this source was mapped to (top 5 by method count)
            if other_targets:
                top_others = sorted(other_targets.items(), key=lambda x: -len(x[1]))[:5]
                other_strs = [
                    f"`{t}` ({len(m)} methods)"
                    for t, m in top_others
                ]
                st.caption(f"Other targets for {src_type}: {', '.join(other_strs)}")

        # Annotation UI (replaces old read-only field table)
        render_annotation_ui(sample, candidates_data, api_lookup)

        st.divider()


def export_annotations(annotations, samples, api_lookup):
    """Export annotations as a dataset JSONL."""
    sample_lookup = {s["id"]: s for s in samples}
    new_samples = []
    stats = {"annotated": 0, "matched": 0, "unmatched": 0}

    for sample in samples:
        sid = sample["id"]
        ann = annotations.get(sid)
        if not ann or not ann.get("annotated"):
            new_samples.append(sample)
            continue

        stats["annotated"] += 1
        src_product = sample["source_schema"]
        tgt_product = sample["target_schema"]
        src_type = sample["metadata"]["source_type"]
        tgt_type = sample["metadata"]["target_type"]

        clean_mapping = {}
        unmatched = []
        for src_short, mapping_info in ann.get("mappings", {}).items():
            src_key = f"{src_product}_{src_type}.{src_short}"
            if mapping_info.get("no_match") or not mapping_info.get("target"):
                unmatched.append(src_key)
                stats["unmatched"] += 1
            else:
                tgt_key = f"{tgt_product}_{tgt_type}.{mapping_info['target']}"
                clean_mapping[src_key] = tgt_key
                stats["matched"] += 1

        new_sample = dict(sample)
        new_sample["mapping"] = clean_mapping
        new_sample["metadata"] = dict(sample["metadata"])
        new_sample["metadata"]["annotation_source"] = "human_v2"
        new_sample["metadata"]["annotation_timestamp"] = ann.get("timestamp", "")
        new_sample["metadata"]["unmatched_fields"] = unmatched
        new_samples.append(new_sample)

    out_path = DATA_DIR / "itom_hashicorp_dataset_annotated.jsonl"
    with open(out_path, "w") as f:
        for s in new_samples:
            f.write(json.dumps(s) + "\n")

    return out_path, stats


def main():
    entity_model, entity_mapping, samples, api_lookup, results, candidates_data, type_mapping_lookup = load_data()

    # Init session state
    if "annotations" not in st.session_state:
        st.session_state.annotations = load_annotations()

    st.title("ITOM – Hashicorp Schema Mapping")
    st.markdown(
        "Mappings between **ITOM** and **Hashicorp** schema types. "
        "Use the annotation controls to accept/reject/correct field mappings."
    )

    # Sidebar
    with st.sidebar:
        st.header("Filters")

        all_rels = sorted(set(s["metadata"]["relationship"] for s in samples))
        selected_rels = st.multiselect("Relationship type", all_rels, default=all_rels)

        all_src = sorted(set(s["source_schema"] for s in samples))
        selected_src = st.multiselect("Source product", all_src, default=all_src)

        all_tgt = sorted(set(s["target_schema"] for s in samples))
        selected_tgt = st.multiselect("Target product", all_tgt, default=all_tgt)

        cross_only = st.checkbox("Cross-product only (ITOM <-> Hashi)", value=True)

        # Annotation filter
        ann_filter = st.radio(
            "Annotation status",
            ["All", "Annotated only", "Unannotated only"],
            index=0,
        )

        st.divider()

        # Annotation progress
        st.header("Annotation Progress")
        annotations = st.session_state.annotations
        annotated_ids = {sid for sid, a in annotations.items() if a.get("annotated")}

        # Count cross-product samples
        cross_ids = set()
        for s in samples:
            src = s["source_schema"]
            tgt = s["target_schema"]
            is_cross = (
                (src in ITOM_ONLY and tgt not in ITOM_ONLY) or
                (tgt in ITOM_ONLY and src not in ITOM_ONLY)
            )
            if is_cross:
                cross_ids.add(s["id"])

        annotated_cross = len(annotated_ids & cross_ids)
        total_cross = len(cross_ids)
        st.metric("Annotated (cross-product)", f"{annotated_cross}/{total_cross}")
        if total_cross > 0:
            st.progress(annotated_cross / total_cross)

        st.divider()

        # Save All
        st.header("Actions")
        if st.button("Save All Annotations"):
            save_annotations(st.session_state.annotations)
            st.success("All annotations saved to disk")

        # Export
        if st.button("Export Annotated Dataset"):
            out_path, stats = export_annotations(
                st.session_state.annotations, samples, api_lookup
            )
            st.success(
                f"Exported to `{out_path.name}`\n\n"
                f"- Annotated: {stats['annotated']} type pairs\n"
                f"- Matched fields: {stats['matched']}\n"
                f"- Unmatched fields: {stats['unmatched']}"
            )

        st.divider()
        st.header("Stats")
        st.metric("Total mappings", len(samples))
        with_results = sum(1 for s in samples if s["id"] in results)
        st.metric("With MR predictions", with_results)
        st.metric("With candidates", len(candidates_data))

    # Filter samples
    filtered = []
    for s in samples:
        if s["metadata"]["relationship"] not in selected_rels:
            continue
        if s["source_schema"] not in selected_src:
            continue
        if s["target_schema"] not in selected_tgt:
            continue
        if cross_only:
            src = s["source_schema"]
            tgt = s["target_schema"]
            is_cross = (
                (src in ITOM_ONLY and tgt not in ITOM_ONLY) or
                (tgt in ITOM_ONLY and src not in ITOM_ONLY)
            )
            if not is_cross:
                continue
        # Annotation filter
        if ann_filter == "Annotated only" and s["id"] not in annotated_ids:
            continue
        if ann_filter == "Unannotated only" and s["id"] in annotated_ids:
            continue
        filtered.append(s)

    st.markdown(f"**Showing {len(filtered)} of {len(samples)} mappings**")

    for s in filtered:
        result = results.get(s["id"])
        render_mapping_card(s, result, entity_model, entity_mapping, api_lookup, candidates_data, type_mapping_lookup)


if __name__ == "__main__":
    main()
