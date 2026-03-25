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

    return entity_model, entity_mapping, samples, api_lookup, results, candidates


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


def render_mapping_card(sample, result, entity_model, entity_mapping, api_lookup, candidates_data):
    """Render a single mapping card with annotation controls."""
    meta = sample["metadata"]
    src_product = sample["source_schema"]
    tgt_product = sample["target_schema"]
    src_type = meta["source_type"]
    tgt_type = meta["target_type"]
    relationship = meta["relationship"]
    description = meta["description"]

    entity_key = f"{src_product}.{meta.get('source_entity', src_type)}"
    em_info = entity_mapping.get(entity_key, {})
    entity_desc = em_info.get("description", "")

    em_product = entity_model.get(src_product, {}).get("entities", {})
    entity_data = em_product.get(meta.get("source_entity", src_type), {})
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
        # Header
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

        # 3-column detail
        detail_cols = st.columns([1, 1, 1])
        with detail_cols[0]:
            st.markdown("**Source API Type**")
            src_fields = [
                (k, v)
                for k, v in api_lookup.items()
                if v.get("product_name") == src_product and v.get("type_name") == src_type
            ]
            st.markdown(f"`{src_product}.{src_type}` — {len(src_fields)} fields")

        with detail_cols[1]:
            st.markdown("**Entity Model**")
            if entity_desc:
                st.markdown(f"_{entity_desc[:100]}_")
            if entity_rels:
                rel_strs = [f"-> {r['target']}" for r in entity_rels[:4]]
                st.markdown("Relationships: " + ", ".join(rel_strs))

        with detail_cols[2]:
            st.markdown("**Target API Type**")
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
    entity_model, entity_mapping, samples, api_lookup, results, candidates_data = load_data()

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
        render_mapping_card(s, result, entity_model, entity_mapping, api_lookup, candidates_data)


if __name__ == "__main__":
    main()
