"""
ITGraph Schema Mapping Annotation Tool

Streamlit app for annotating schema mappings in the ITGraph dataset.
Each sample contains a type mapping predicted by multiple schema mapping
methods (Map-Reflect, MatchMaker, ReMatch, GNN, SMT, ReAct, UE-Reflect)
across multiple LLMs (Granite, Llama, Mistral).

The annotator reviews field-level mappings, informed by:
  - Method consensus scores (how many methods agree)
  - Per-field confidence from each method
  - API field metadata (type, description) when available

Features:
  - Filter by product, relationship, confidence, annotation status
  - Per-field accept/reject/correct with target field dropdown
  - Bipartite validation (each target field used at most once)
  - Save/load annotations, export annotated dataset
"""

import streamlit as st
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

st.set_page_config(
    page_title="ITGraph Annotation",
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
    "cloudability": "#E91E63",
    "ns1": "#00BCD4",
    "watsonx-orchestrate": "#6200EA",
}

METHOD_COLORS = {
    "Map-Reflect": "#1565C0",
    "ReAct": "#6A1B9A",
    "UE-Reflect": "#00838F",
    "MatchMaker": "#2E7D32",
    "ReMatch": "#E65100",
    "SMT": "#AD1457",
    "GNN": "#37474F",
}

DATA_DIR = Path(__file__).parent / "data"
ANNOTATIONS_PATH = DATA_DIR / "itgraph_annotations.json"


@st.cache_data
def load_data():
    """Load ITGraph dataset and supporting files."""
    # Prefer MB dataset (cross-product, Llama-only, annotation-aware)
    dataset_path = DATA_DIR / "itgraph_mb_dataset.jsonl"
    if not dataset_path.exists():
        dataset_path = DATA_DIR / "itgraph_dataset_v2.jsonl"
    if not dataset_path.exists():
        dataset_path = DATA_DIR / "itgraph_dataset.jsonl"
    with open(dataset_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    # High-confidence subset IDs for filtering
    hc_ids = set()
    hc_path = DATA_DIR / "itgraph_mb_dataset_high_confidence.jsonl"
    if not hc_path.exists():
        hc_path = DATA_DIR / "itgraph_dataset_v2_high_confidence.jsonl"
    if not hc_path.exists():
        hc_path = DATA_DIR / "itgraph_dataset_high_confidence.jsonl"
    if hc_path.exists():
        with open(hc_path) as f:
            for line in f:
                if line.strip():
                    hc_ids.add(json.loads(line)["id"])

    # Type mapping summary (which methods suggested each type pair)
    type_summary = {}
    tms_path = DATA_DIR / "itgraph_type_mapping_summary.json"
    if tms_path.exists():
        with open(tms_path) as f:
            type_summary = json.load(f)

    # API field lookup (for field descriptions/types)
    api_lookup = {}
    lookup_path = DATA_DIR / "itom_hashicorp_api_lookup_v2.json"
    if not lookup_path.exists():
        lookup_path = DATA_DIR / "itom_hashicorp_api_lookup.json"
    if lookup_path.exists():
        with open(lookup_path) as f:
            api_lookup = json.load(f)

    # Schema descriptions (type and field level, merged from all schema sources)
    schema_descs = {"types": {}, "fields": {}}
    desc_path = DATA_DIR / "schema_descriptions.json"
    if desc_path.exists():
        with open(desc_path) as f:
            schema_descs = json.load(f)

    # Expert entity relationship lookup (from ITOM workshop data model)
    expert_lookup = {"api_type_to_entity": {}, "type_pair_expert": {}}
    expert_path = DATA_DIR / "expert_entity_lookup.json"
    if expert_path.exists():
        with open(expert_path) as f:
            expert_lookup = json.load(f)

    return samples, hc_ids, type_summary, api_lookup, schema_descs, expert_lookup


def load_annotations():
    if ANNOTATIONS_PATH.exists():
        with open(ANNOTATIONS_PATH) as f:
            return json.load(f)
    return {}


def save_annotations(annotations):
    with open(ANNOTATIONS_PATH, "w") as f:
        json.dump(annotations, f, indent=2)


def product_badge(product):
    color = PRODUCT_COLORS.get(product, "#999")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em;font-weight:bold">{product}</span>'
    )


def method_badge(method_name):
    family = method_name.split("(")[0].strip()
    color = METHOD_COLORS.get(family, "#757575")
    return (
        f'<span style="background:{color};color:white;padding:1px 6px;'
        f'border-radius:4px;font-size:0.75em;margin-right:3px">'
        f'{method_name}</span>'
    )


def consensus_badge(consensus, n_methods):
    """Color-coded badge for consensus score."""
    if consensus >= 0.7:
        color = "#4CAF50"
    elif consensus >= 0.4:
        color = "#FF9800"
    else:
        color = "#F44336"
    return (
        f'<span style="background:{color};color:white;padding:1px 6px;'
        f'border-radius:8px;font-size:0.75em">'
        f'{consensus:.2f} ({n_methods}m)</span>'
    )


def get_target_fields_from_lookup(api_lookup, product, type_name):
    """Get target field names from API lookup (when available)."""
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


def get_target_fields_from_dataset(samples, product, type_name):
    """Extract target field names from dataset mappings (fallback when no API lookup)."""
    prefix = f"{product}_{type_name}."
    fields = set()
    for s in samples:
        # Check mapping values
        for tgt_key in s.get("mapping", {}).values():
            if tgt_key.startswith(prefix):
                fields.add(tgt_key[len(prefix):])
        # Check field_assignments targets
        for fa in s["metadata"].get("field_assignments", {}).values():
            tgt = fa.get("target", "")
            if tgt.startswith(prefix):
                fields.add(tgt[len(prefix):])
    return sorted(fields)


def render_type_annotation_ui(sample, all_samples, schema_descs):
    """Render type-level annotation: is this type mapping correct?"""
    sid = sample["id"]
    meta = sample["metadata"]
    src_product = sample["source_schema"]
    tgt_product = sample["target_schema"]
    src_type = meta["source_type"]
    tgt_type = meta["target_type"]

    if "annotations" not in st.session_state:
        st.session_state.annotations = load_annotations()
    saved = st.session_state.annotations.get(sid, {})
    saved_type = saved.get("type_annotation", {})

    # Current judgment
    saved_judgment = saved_type.get("judgment", "")
    judgment_options = ["Unannotated", "Correct", "Plausible", "Incorrect", "Uncertain"]
    try:
        default_judgment = judgment_options.index(saved_judgment) if saved_judgment else 0
    except ValueError:
        default_judgment = 0

    TYPE_JUDGMENT_COLORS = {
        "Correct": "#4CAF50",
        "Plausible": "#2196F3",
        "Incorrect": "#F44336",
        "Uncertain": "#FF9800",
    }

    # Show saved judgment as a persistent badge + toast notification
    if saved_judgment and saved_judgment != "Unannotated":
        jcolor = TYPE_JUDGMENT_COLORS.get(saved_judgment, "#999")
        notes_text = f' — {saved_type.get("notes", "")}' if saved_type.get("notes") else ""
        corrected_text = f' (suggested: {saved_type["corrected_target_type"]})' if saved_type.get("corrected_target_type") else ""
        st.markdown(
            f'<span style="background:{jcolor};color:white;padding:2px 8px;'
            f'border-radius:8px;font-size:0.8em">'
            f'Type mapping: {saved_judgment}{corrected_text}</span>'
            f'<span style="font-size:0.8em;color:#666">{notes_text}</span>',
            unsafe_allow_html=True,
        )

    # Check if we just saved (via session state flag)
    just_saved_key = f"_type_saved_{sid}"
    if st.session_state.get(just_saved_key):
        st.success(f"Saved type annotation: {saved_judgment}")
        st.session_state[just_saved_key] = False

    tcols = st.columns([3, 2, 2, 1])

    with tcols[0]:
        judgment = st.radio(
            "Type mapping judgment",
            judgment_options,
            index=default_judgment,
            key=f"type_judgment_{sid}",
            horizontal=True,
        )

    with tcols[1]:
        # If incorrect, let annotator suggest a better target type
        saved_correction = saved_type.get("corrected_target_type", "")

        # Build list of all target types for this product
        tgt_types_in_product = sorted(set(
            s["metadata"]["target_type"]
            for s in all_samples
            if s["target_schema"] == tgt_product
        ))
        correction_options = ["(none)"] + tgt_types_in_product
        default_correction = 0
        if saved_correction:
            try:
                default_correction = correction_options.index(saved_correction)
            except ValueError:
                default_correction = 0

        corrected_type = st.selectbox(
            "Better target type",
            correction_options,
            index=default_correction,
            key=f"type_correction_{sid}",
        )

    with tcols[2]:
        saved_notes = saved_type.get("notes", "")
        notes = st.text_input(
            "Notes",
            value=saved_notes,
            key=f"type_notes_{sid}",
            placeholder="e.g. partial overlap...",
        )

    with tcols[3]:
        st.markdown("")  # spacer for vertical alignment
        if st.button("Save", key=f"save_type_{sid}"):
            type_ann = {
                "judgment": judgment,
                "corrected_target_type": corrected_type if corrected_type != "(none)" else "",
                "notes": notes,
                "timestamp": datetime.now().isoformat(),
            }
            if sid not in st.session_state.annotations:
                st.session_state.annotations[sid] = {
                    "source": f"{src_product}.{src_type}",
                    "target": f"{tgt_product}.{tgt_type}",
                }
            st.session_state.annotations[sid]["type_annotation"] = type_ann
            st.session_state.annotations[sid]["annotated"] = True
            save_annotations(st.session_state.annotations)
            st.session_state[just_saved_key] = True
            st.rerun()


def render_annotation_ui(sample, api_lookup, all_samples, schema_descs):
    """Render interactive annotation controls for field mappings."""
    sid = sample["id"]
    meta = sample["metadata"]
    src_product = sample["source_schema"]
    tgt_product = sample["target_schema"]
    src_type = meta["source_type"]
    tgt_type = meta["target_type"]
    field_assignments = meta.get("field_assignments", {})

    if not field_assignments:
        st.info("No field-level predictions for this type pair.")
        return

    # Get all possible target fields
    all_target_fields = get_target_fields_from_lookup(api_lookup, tgt_product, tgt_type)
    if not all_target_fields:
        all_target_fields = get_target_fields_from_dataset(all_samples, tgt_product, tgt_type)

    # Load existing annotations
    if "annotations" not in st.session_state:
        st.session_state.annotations = load_annotations()
    saved = st.session_state.annotations.get(sid, {})
    saved_mappings = saved.get("mappings", {})

    annotated_count = sum(
        1 for v in saved_mappings.values()
        if v.get("target") is not None or v.get("no_match")
    )
    total_fields = len(field_assignments)
    is_complete = annotated_count == total_fields and total_fields > 0

    expander_label = (
        f"Field Annotations ({total_fields} fields) — "
        f"{annotated_count}/{total_fields} annotated"
    )
    if is_complete:
        expander_label += " ✓"

    with st.expander(expander_label, expanded=False):
        # Header row
        hdr = st.columns([2, 2, 2, 1])
        hdr[0].markdown("**Source Field**")
        hdr[1].markdown("**Predicted Target**")
        hdr[2].markdown("**Your Annotation**")
        hdr[3].markdown("**Consensus**")

        claimed_targets = defaultdict(list)
        field_options = {}

        for src_key, fa_data in field_assignments.items():
            src_short = src_key.split(".")[-1] if "." in src_key else src_key
            # Use full key as widget identifier to avoid duplicates
            src_widget_id = src_key.replace(".", "_")
            predicted_target = fa_data.get("target", "")
            pred_short = predicted_target.split(".")[-1] if "." in predicted_target else predicted_target
            fa_consensus = fa_data.get("consensus", 0)
            fa_n_methods = fa_data.get("n_methods", 0)

            # Source field metadata (api_lookup first, then schema_descs)
            src_meta = api_lookup.get(src_key, {})
            src_desc = src_meta.get("description", "") or schema_descs["fields"].get(src_key, "")
            src_ftype = src_meta.get("field_type", "")

            # Build options: predicted target first, then all others, then No Match
            options = []
            option_labels = []

            if pred_short:
                options.append(pred_short)
                option_labels.append(
                    f"{pred_short} (predicted, {fa_consensus:.2f}, {fa_n_methods}m)"
                )

            for tf in all_target_fields:
                if tf != pred_short:
                    options.append(tf)
                    tf_key = f"{tgt_product}_{tgt_type}.{tf}"
                    tf_meta = api_lookup.get(tf_key, {})
                    tf_desc = tf_meta.get("description", "") or schema_descs["fields"].get(tf_key, "")
                    lbl = f"{tf} — {tf_desc[:50]}" if tf_desc else tf
                    option_labels.append(lbl)

            options.append("__NO_MATCH__")
            option_labels.append("No Match")

            field_options[src_widget_id] = options

            # Determine default from saved annotation
            saved_field = saved_mappings.get(src_widget_id, {})
            if not saved_field:
                saved_field = saved_mappings.get(src_short, {})
            if saved_field:
                if saved_field.get("no_match"):
                    default_idx = len(options) - 1
                elif saved_field.get("target"):
                    try:
                        default_idx = options.index(saved_field["target"])
                    except ValueError:
                        default_idx = 0
                else:
                    default_idx = 0
            else:
                default_idx = 0  # predicted target

            # Target field description
            pred_desc = (
                api_lookup.get(predicted_target, {}).get("description", "")
                or schema_descs["fields"].get(predicted_target, "")
            )

            # Render row
            cols = st.columns([2, 2, 2, 1])
            with cols[0]:
                type_tag = f" `{src_ftype}`" if src_ftype else ""
                st.markdown(f"**{src_short}**{type_tag}")
                if src_desc:
                    st.caption(src_desc[:100])
            with cols[1]:
                st.markdown(f"`{pred_short}`")
                if pred_desc:
                    st.caption(f"{pred_desc[:80]}")
                st.caption(f"consensus {fa_consensus:.2f}, {fa_n_methods} methods")
            with cols[2]:
                choice = st.selectbox(
                    f"Target for {src_short}",
                    options=range(len(options)),
                    format_func=lambda i, labels=option_labels: labels[i],
                    index=default_idx,
                    key=f"ann_{sid}_{src_widget_id}",
                    label_visibility="collapsed",
                )
                chosen = options[choice]
                if chosen != "__NO_MATCH__":
                    claimed_targets[chosen].append(src_short)
            with cols[3]:
                st.markdown(
                    consensus_badge(fa_consensus, fa_n_methods),
                    unsafe_allow_html=True,
                )

        # Bipartite violations
        violations = {t: srcs for t, srcs in claimed_targets.items() if len(srcs) > 1}
        if violations:
            violation_strs = [
                f"**{t}** claimed by: {', '.join(srcs)}"
                for t, srcs in violations.items()
            ]
            st.warning(
                f"Bipartite violation: {len(violations)} target(s) used more than once:\n\n"
                + "\n".join(f"- {v}" for v in violation_strs)
            )

        # Save button
        if st.button(f"Save annotations for {sid}", key=f"save_{sid}"):
            mappings = {}
            for src_key in field_assignments:
                src_widget_id = src_key.replace(".", "_")
                widget_key = f"ann_{sid}_{src_widget_id}"
                choice_idx = st.session_state.get(widget_key, 0)
                opts = field_options.get(src_widget_id, ["__NO_MATCH__"])
                chosen = opts[choice_idx] if choice_idx < len(opts) else "__NO_MATCH__"

                if chosen == "__NO_MATCH__":
                    mappings[src_widget_id] = {"target": None, "no_match": True}
                else:
                    mappings[src_widget_id] = {"target": chosen, "no_match": False}

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


def render_mapping_card(sample, api_lookup, all_samples, type_summary, schema_descs, expert_lookup):
    """Render a single mapping card."""
    meta = sample["metadata"]
    sid = sample["id"]
    src_product = sample["source_schema"]
    tgt_product = sample["target_schema"]
    src_type = meta["source_type"]
    tgt_type = meta["target_type"]
    relationship = meta["relationship"]
    description = meta["description"]
    n_methods = meta.get("n_methods", 0)
    methods = meta.get("methods", [])
    type_methods = meta.get("type_methods", [])
    field_assignments = meta.get("field_assignments", {})
    unmatched_fields = meta.get("unmatched_fields", [])

    # Type descriptions from schema
    src_type_key = f"{src_product}.{src_type}"
    tgt_type_key = f"{tgt_product}.{tgt_type}"
    src_type_desc = schema_descs["types"].get(src_type_key, "")
    tgt_type_desc = schema_descs["types"].get(tgt_type_key, "")

    # Annotation status
    saved = st.session_state.get("annotations", {}).get(sid, {})
    is_annotated = saved.get("annotated", False)

    with st.container():
        # Header: Source -> Target
        cols = st.columns([1, 1, 1])
        with cols[0]:
            st.markdown(
                f'{product_badge(src_product)} <b>{src_type}</b>',
                unsafe_allow_html=True,
            )
            if src_type_desc:
                st.caption(src_type_desc[:200])
        with cols[1]:
            rel_color = "#3F51B5"
            ann_badge = (
                ' <span style="background:#4CAF50;color:white;padding:1px 6px;'
                'border-radius:8px;font-size:0.7em">annotated</span>'
                if is_annotated else ""
            )
            st.markdown(
                f'<div style="text-align:center">'
                f'<span style="background:{rel_color};color:white;padding:2px 8px;'
                f'border-radius:12px;font-size:0.8em">{relationship.replace("_", " ")}</span>'
                f'{ann_badge}</div>',
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                f'<div style="text-align:right">'
                f'{product_badge(tgt_product)} <b>{tgt_type}</b></div>',
                unsafe_allow_html=True,
            )
            if tgt_type_desc:
                st.caption(tgt_type_desc[:200])

        st.caption(f"_{description}_")

        # Expert entity context
        api_to_entity = expert_lookup.get("api_type_to_entity", {})
        pair_expert = expert_lookup.get("type_pair_expert", {})

        src_api_key = f"{src_product}.{src_type}"
        tgt_api_key = f"{tgt_product}.{tgt_type}"
        src_entities = api_to_entity.get(src_api_key, [])
        tgt_entities = api_to_entity.get(tgt_api_key, [])
        pair_key = f"{src_api_key} -> {tgt_api_key}"
        expert_rel = pair_expert.get(pair_key)

        has_expert = src_entities or tgt_entities or expert_rel
        if has_expert:
            expert_parts = []
            if src_entities:
                ent_labels = ", ".join(
                    f'<b>{e["entity_label"]}</b> <span style="color:#888">({e["domain"]})</span>'
                    for e in src_entities
                )
                expert_parts.append(f'Source entity: {ent_labels}')
            if tgt_entities:
                ent_labels = ", ".join(
                    f'<b>{e["entity_label"]}</b> <span style="color:#888">({e["domain"]})</span>'
                    for e in tgt_entities
                )
                expert_parts.append(f'Target entity: {ent_labels}')

            if expert_rel:
                rel_labels = ", ".join(expert_rel["relationships"])
                expert_parts.append(
                    f'Expert relationship: <b>{expert_rel["src_entity"]}</b> '
                    f'&mdash;[{rel_labels}]&rarr; '
                    f'<b>{expert_rel["tgt_entity"]}</b>'
                )
                if expert_rel.get("notes"):
                    expert_parts.append(f'<span style="color:#666">{expert_rel["notes"]}</span>')

            border_color = "#4CAF50" if expert_rel else "#2196F3"
            st.markdown(
                f'<div style="background:#f0f7f0;padding:6px 12px;border-radius:6px;'
                f'border-left:3px solid {border_color};font-size:0.85em;margin:4px 0">'
                f'<span style="color:#999;font-size:0.8em">EXPERT ENTITY GRAPH</span><br>'
                + "<br>".join(expert_parts)
                + '</div>',
                unsafe_allow_html=True,
            )

        # Field reference lists for source and target types
        src_prefix = f"{src_product}_{src_type}."
        tgt_prefix = f"{tgt_product}_{tgt_type}."

        # Gather source fields from field_assignments keys
        src_fields_info = []
        for src_key in field_assignments:
            fname = src_key.split(".")[-1] if "." in src_key else src_key
            fdesc = (
                api_lookup.get(src_key, {}).get("description", "")
                or schema_descs["fields"].get(src_key, "")
            )
            ftype = api_lookup.get(src_key, {}).get("field_type", "")
            src_fields_info.append((fname, ftype, fdesc))

        # Gather all target fields
        tgt_fields_info = []
        all_tgt_fields = get_target_fields_from_lookup(api_lookup, tgt_product, tgt_type)
        if not all_tgt_fields:
            all_tgt_fields = get_target_fields_from_dataset(all_samples, tgt_product, tgt_type)
        for tf in all_tgt_fields:
            tf_key = f"{tgt_prefix}{tf}"
            fdesc = (
                api_lookup.get(tf_key, {}).get("description", "")
                or schema_descs["fields"].get(tf_key, "")
            )
            ftype = api_lookup.get(tf_key, {}).get("field_type", "")
            tgt_fields_info.append((tf, ftype, fdesc))

        ref_cols = st.columns(2)
        with ref_cols[0]:
            src_rows = []
            for fname, ftype, fdesc in src_fields_info:
                type_tag = f' <code>{ftype}</code>' if ftype else ""
                desc_tag = f' <span style="color:#666">— {fdesc[:60]}</span>' if fdesc else ""
                src_rows.append(
                    f'<span style="font-size:0.82em">'
                    f'<b>{fname}</b>{type_tag}{desc_tag}</span>'
                )
            st.markdown(
                f'<div style="background:#f8f9fa;padding:8px 12px;border-radius:6px;'
                f'border-left:3px solid {PRODUCT_COLORS.get(src_product, "#999")}">'
                f'<span style="font-size:0.75em;color:#999">SOURCE FIELDS '
                f'({len(src_fields_info)})</span><br>'
                + "<br>".join(src_rows)
                + "</div>",
                unsafe_allow_html=True,
            )
        with ref_cols[1]:
            tgt_rows = []
            for fname, ftype, fdesc in tgt_fields_info:
                type_tag = f' <code>{ftype}</code>' if ftype else ""
                desc_tag = f' <span style="color:#666">— {fdesc[:60]}</span>' if fdesc else ""
                tgt_rows.append(
                    f'<span style="font-size:0.82em">'
                    f'<b>{fname}</b>{type_tag}{desc_tag}</span>'
                )
            st.markdown(
                f'<div style="background:#f8f9fa;padding:8px 12px;border-radius:6px;'
                f'border-left:3px solid {PRODUCT_COLORS.get(tgt_product, "#999")}">'
                f'<span style="font-size:0.75em;color:#999">TARGET FIELDS '
                f'({len(tgt_fields_info)})</span><br>'
                + "<br>".join(tgt_rows)
                + "</div>",
                unsafe_allow_html=True,
            )

        # Method badges row
        info_cols = st.columns([2, 1])
        with info_cols[0]:
            # Field-level methods (which methods predicted the field mappings)
            if methods:
                badges = " ".join(method_badge(m) for m in methods)
                st.markdown(
                    f'<span style="font-size:0.85em;color:#555">'
                    f'Field mapping methods:</span> {badges}',
                    unsafe_allow_html=True,
                )
            # Type-level methods (which methods suggested this type pair)
            if type_methods:
                type_badges = " ".join(method_badge(m) for m in type_methods)
                st.markdown(
                    f'<span style="font-size:0.85em;color:#555">'
                    f'Type discovery methods:</span> {type_badges}',
                    unsafe_allow_html=True,
                )
        with info_cols[1]:
            n_fields = len(field_assignments)
            n_unmatched = len(unmatched_fields)
            st.markdown(
                f"**{n_fields}** field mappings, "
                f"**{n_unmatched}** unmatched, "
                f"**{n_methods}** methods agreed"
            )

        # Field-level consensus summary (compact)
        if field_assignments:
            consensuses = [
                fa.get("consensus", 0) for fa in field_assignments.values()
            ]
            avg_consensus = sum(consensuses) / len(consensuses)
            high = sum(1 for c in consensuses if c >= 0.7)
            med = sum(1 for c in consensuses if 0.4 <= c < 0.7)
            low = sum(1 for c in consensuses if c < 0.4)
            st.markdown(
                f'<span style="font-size:0.85em">'
                f'Avg field consensus: <b>{avg_consensus:.2f}</b> &mdash; '
                f'<span style="color:#4CAF50">high:{high}</span> '
                f'<span style="color:#FF9800">med:{med}</span> '
                f'<span style="color:#F44336">low:{low}</span></span>',
                unsafe_allow_html=True,
            )

        # Type-level annotation
        render_type_annotation_ui(sample, all_samples, schema_descs)

        # Field-level annotation
        render_annotation_ui(sample, api_lookup, all_samples, schema_descs)

        st.divider()


def export_annotations(annotations, samples):
    """Export annotations as an updated dataset JSONL."""
    new_samples = []
    stats = {
        "annotated": 0, "matched": 0, "unmatched": 0,
        "type_correct": 0, "type_plausible": 0, "type_incorrect": 0, "type_uncertain": 0,
    }

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

        new_sample = dict(sample)
        new_sample["metadata"] = dict(sample["metadata"])
        new_sample["metadata"]["annotation_source"] = "human"

        # Type-level annotation
        type_ann = ann.get("type_annotation", {})
        judgment = type_ann.get("judgment", "")
        if judgment and judgment != "Unannotated":
            new_sample["metadata"]["type_judgment"] = judgment
            new_sample["metadata"]["type_annotation_notes"] = type_ann.get("notes", "")
            corrected = type_ann.get("corrected_target_type", "")
            if corrected:
                new_sample["metadata"]["corrected_target_type"] = corrected
            stats_key = f"type_{judgment.lower()}"
            if stats_key in stats:
                stats[stats_key] += 1

        # Field-level annotation
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

        if ann.get("mappings"):
            new_sample["mapping"] = clean_mapping
            new_sample["metadata"]["unmatched_fields"] = unmatched
            new_sample["metadata"]["annotation_timestamp"] = ann.get("type_annotation", {}).get("timestamp", "")

        new_samples.append(new_sample)

    out_path = DATA_DIR / "itgraph_dataset_annotated.jsonl"
    with open(out_path, "w") as f:
        for s in new_samples:
            f.write(json.dumps(s) + "\n")

    return out_path, stats


def main():
    samples, hc_ids, type_summary, api_lookup, schema_descs, expert_lookup = load_data()

    if "annotations" not in st.session_state:
        st.session_state.annotations = load_annotations()

    st.title("ITGraph Schema Mapping Annotation")
    st.markdown(
        "Review and annotate field-level schema mappings predicted by "
        "multiple methods (Map-Reflect, MatchMaker, ReMatch, GNN, SMT, ReAct)."
    )

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")

        all_src = sorted(set(s["source_schema"] for s in samples))
        selected_src = st.multiselect("Source product", all_src, default=all_src)

        all_tgt = sorted(set(s["target_schema"] for s in samples))
        selected_tgt = st.multiselect("Target product", all_tgt, default=all_tgt)

        # Confidence filter
        confidence_filter = st.radio(
            "Confidence",
            ["All", "High confidence only"],
            index=0,
        )

        # Min methods filter
        min_methods = st.slider("Min methods agreeing", 1, 9, 1)

        # Hide pre-marked incorrect
        hide_incorrect = st.checkbox(
            "Hide already-rejected type mappings", value=True
        )

        # Annotation status
        ann_filter = st.radio(
            "Annotation status",
            ["All", "Annotated only", "Unannotated only",
             "Type-annotated only", "Type-unannotated only"],
            index=0,
        )

        st.divider()

        # Progress
        st.header("Annotation Progress")
        annotations = st.session_state.annotations
        annotated_ids = {sid for sid, a in annotations.items() if a.get("annotated")}
        type_annotated_ids = {
            sid for sid, a in annotations.items()
            if a.get("type_annotation", {}).get("judgment", "") not in ("", "Unannotated")
        }
        field_annotated_ids = {
            sid for sid, a in annotations.items()
            if a.get("mappings")
        }

        st.metric("Type-level annotated", f"{len(type_annotated_ids)} / {len(samples)}")
        if samples:
            st.progress(len(type_annotated_ids) / len(samples))
        st.metric("Field-level annotated", f"{len(field_annotated_ids)} / {len(samples)}")
        if samples:
            st.progress(len(field_annotated_ids) / len(samples))

        hc_annotated = len(type_annotated_ids & hc_ids)
        st.metric(
            "High-confidence (type-annotated)",
            f"{hc_annotated} / {len(hc_ids)}",
        )
        if hc_ids:
            st.progress(hc_annotated / len(hc_ids))

        # Type judgment breakdown
        judgment_counts = defaultdict(int)
        for sid, a in annotations.items():
            j = a.get("type_annotation", {}).get("judgment", "")
            if j and j != "Unannotated":
                judgment_counts[j] += 1
        if judgment_counts:
            st.markdown("**Type judgments:**")
            for j in ["Correct", "Plausible", "Incorrect", "Uncertain"]:
                if judgment_counts.get(j, 0) > 0:
                    st.markdown(f"- {j}: {judgment_counts[j]}")

        st.divider()

        # Actions
        st.header("Actions")
        if st.button("Save All Annotations"):
            save_annotations(st.session_state.annotations)
            st.success("All annotations saved to disk")

        if st.button("Export Annotated Dataset"):
            out_path, stats = export_annotations(
                st.session_state.annotations, samples
            )
            st.success(
                f"Exported to `{out_path.name}`\n\n"
                f"- Annotated: {stats['annotated']} type pairs\n"
                f"- Type judgments: {stats['type_correct']} correct, "
                f"{stats['type_plausible']} plausible, "
                f"{stats['type_incorrect']} incorrect, "
                f"{stats['type_uncertain']} uncertain\n"
                f"- Matched fields: {stats['matched']}\n"
                f"- Unmatched fields: {stats['unmatched']}"
            )

        st.divider()

        st.header("Dataset Stats")
        st.metric("Total type pairs", len(samples))
        st.metric("High confidence", len(hc_ids))

        # Method coverage
        all_methods = set()
        for s in samples:
            for m in s["metadata"].get("methods", []):
                all_methods.add(m.split("(")[0].strip())
        st.metric("Distinct method families", len(all_methods))

    # Filter
    filtered = []
    for s in samples:
        if s["source_schema"] not in selected_src:
            continue
        if s["target_schema"] not in selected_tgt:
            continue
        if confidence_filter == "High confidence only" and s["id"] not in hc_ids:
            continue
        if s["metadata"].get("n_methods", 0) < min_methods:
            continue
        # Hide pre-marked incorrect (from dataset or from annotations)
        if hide_incorrect:
            judgment = s["metadata"].get("human_judgment", "")
            ann_judgment = st.session_state.annotations.get(
                s["id"], {}
            ).get("type_annotation", {}).get("judgment", "")
            if judgment == "incorrect" or ann_judgment == "Incorrect":
                continue
        if ann_filter == "Annotated only" and s["id"] not in annotated_ids:
            continue
        if ann_filter == "Unannotated only" and s["id"] in annotated_ids:
            continue
        if ann_filter == "Type-annotated only" and s["id"] not in type_annotated_ids:
            continue
        if ann_filter == "Type-unannotated only" and s["id"] in type_annotated_ids:
            continue
        filtered.append(s)

    st.markdown(f"**{len(filtered)}** type pairs match filters (of {len(samples)} total)")

    if not filtered:
        st.info("No type pairs match the current filters.")
        return

    # Single-card navigation
    if "card_idx" not in st.session_state:
        st.session_state.card_idx = 0

    # Clamp index
    idx = st.session_state.card_idx
    if idx >= len(filtered):
        idx = len(filtered) - 1
    if idx < 0:
        idx = 0
    st.session_state.card_idx = idx

    # Navigation bar
    nav_cols = st.columns([1, 1, 3, 1, 1])
    with nav_cols[0]:
        if st.button("Prev", disabled=(idx == 0), use_container_width=True):
            st.session_state.card_idx = idx - 1
            st.rerun()
    with nav_cols[1]:
        if st.button("Next", disabled=(idx >= len(filtered) - 1), use_container_width=True):
            st.session_state.card_idx = idx + 1
            st.rerun()
    with nav_cols[2]:
        st.markdown(
            f"**{idx + 1}** of **{len(filtered)}** &nbsp; "
            f"(`{filtered[idx]['id']}`)",
            unsafe_allow_html=True,
        )
    with nav_cols[3]:
        jump = st.number_input(
            "Go to #", min_value=1, max_value=len(filtered),
            value=idx + 1, key="jump_to", label_visibility="collapsed",
        )
    with nav_cols[4]:
        if st.button("Go", use_container_width=True):
            st.session_state.card_idx = jump - 1
            st.rerun()

    st.divider()

    # Render single card
    s = filtered[idx]
    render_mapping_card(s, api_lookup, samples, type_summary, schema_descs, expert_lookup)

    # Bottom navigation with skip-to-next-unannotated
    bot_cols = st.columns([1, 1, 2])
    with bot_cols[0]:
        if st.button("Prev", key="bot_prev", disabled=(idx == 0), use_container_width=True):
            st.session_state.card_idx = idx - 1
            st.rerun()
    with bot_cols[1]:
        if st.button("Next", key="bot_next", disabled=(idx >= len(filtered) - 1), use_container_width=True):
            st.session_state.card_idx = idx + 1
            st.rerun()
    with bot_cols[2]:
        if st.button("Skip to next unannotated", key="skip_unann", use_container_width=True):
            found = False
            for i in range(idx + 1, len(filtered)):
                fid = filtered[i]["id"]
                # Check dataset-level judgment (pre-baked from previous annotations)
                dataset_judgment = filtered[i]["metadata"].get("human_judgment", "")
                if dataset_judgment in ("incorrect", "correct", "plausible"):
                    continue
                # Check session-level annotation
                ann = st.session_state.annotations.get(fid, {})
                ta = ann.get("type_annotation", {})
                if ta.get("judgment", "") not in ("", "Unannotated"):
                    continue
                st.session_state.card_idx = i
                found = True
                st.rerun()
            if not found:
                st.info("No more unannotated type pairs in the current filter.")


if __name__ == "__main__":
    main()
