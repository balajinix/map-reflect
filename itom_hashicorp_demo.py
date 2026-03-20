"""
ITOM-Hashicorp Schema Mapping Demo

Streamlit app showing schema mappings between ITOM products (Instana,
Concert, Turbonomic, HCP Terraform) and Hashicorp products (Infragraph,
Packer, AWS, Azure, GCP, HCP).

Layout: ITOM API (left) | Entity Model (center) | Hashicorp API (right)
Field-level mappings are shown in expandable sections.
"""

import streamlit as st
import json
from pathlib import Path
from collections import defaultdict

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

# hcp_terraform appears in both ITOM and Hashicorp; for cross-product
# classification, a mapping is cross-product only when one side is
# exclusively ITOM and the other exclusively Hashicorp.
ITOM_ONLY = {"instana", "concert", "turbonomic"}
HASHI_ONLY = {"infragraph", "packer", "aws", "azure", "gcp", "hcp"}
SHARED = {"hcp_terraform"}  # belongs to both
ITOM_PRODUCTS = ITOM_ONLY | SHARED
HASHI_PRODUCTS = HASHI_ONLY | SHARED


@st.cache_data
def load_data():
    base = Path(__file__).parent / "data"
    mr_results_dir = Path(__file__).parent / "map_reflect" / "results"

    # Entity model (conceptual)
    with open(base / "itom_hashicorp_entity_model.json") as f:
        entity_model = json.load(f)

    # Entity mapping (conceptual -> actual types)
    with open(base / "itom_hashicorp_entity_mapping.json") as f:
        entity_mapping = json.load(f)

    # Dataset (ground truth)
    samples = []
    with open(base / "itom_hashicorp_dataset.jsonl") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # API lookup
    with open(base / "itom_hashicorp_api_lookup.json") as f:
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
                # Llama overwrites Granite (loaded second)
                results[sid] = r
                results[sid]["_model"] = model_tag

    return entity_model, entity_mapping, samples, api_lookup, results


def product_badge(product):
    color = PRODUCT_COLORS.get(product, "#999")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;font-weight:bold">{product}</span>'


def render_field_table(predictions, ground_truth, api_lookup):
    """Render field-level mapping table inside an expander."""
    rows = []
    for src_field, gt_target in ground_truth.items():
        pred_data = predictions.get(src_field, {})
        ranking = pred_data.get("initial_ranking", [])
        top_pred = ranking[0] if ranking else {}
        pred_target = top_pred.get("id", "—")
        score = top_pred.get("score", 0)
        reason = top_pred.get("reason", "")

        # Check correctness
        correct = pred_target.lower() == gt_target.lower() if gt_target else False
        icon = "✅" if correct else "❌"

        # Get descriptions
        src_desc = api_lookup.get(src_field, {}).get("description", "")
        tgt_desc = api_lookup.get(pred_target, {}).get("description", "")

        src_short = src_field.split(".")[-1] if "." in src_field else src_field
        tgt_short = pred_target.split(".")[-1] if "." in pred_target else pred_target

        rows.append({
            "": icon,
            "Source Field": src_short,
            "Predicted Target": tgt_short,
            "Score": f"{score:.2f}",
            "Reason": reason[:80],
        })

    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No field predictions available")


def render_mapping_card(sample, result, entity_model, entity_mapping, api_lookup):
    """Render a single mapping card with 3-column layout."""
    meta = sample["metadata"]
    src_product = sample["source_schema"]
    tgt_product = sample["target_schema"]
    src_type = meta["source_type"]
    tgt_type = meta["target_type"]
    relationship = meta["relationship"]
    description = meta["description"]

    # Get entity info
    entity_key = f"{src_product}.{meta.get('source_entity', src_type)}"
    em_info = entity_mapping.get(entity_key, {})
    entity_desc = em_info.get("description", "")

    # Get entity relationships from model
    em_product = entity_model.get(src_product, {}).get("entities", {})
    entity_data = em_product.get(meta.get("source_entity", src_type), {})
    entity_rels = entity_data.get("relationships", [])

    # Determine accuracy from result
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

    # Card container
    with st.container():
        # Header
        cols = st.columns([1, 1, 1])
        with cols[0]:
            st.markdown(f"{product_badge(src_product)} **{src_type}**", unsafe_allow_html=True)
        with cols[1]:
            rel_color = {
                "semantic_equivalent": "#4CAF50",
                "projection": "#2196F3",
                "governs": "#FF9800",
                "acts_on": "#F44336",
                "supply_chain": "#9C27B0",
                "composed_of": "#00BCD4",
                "derived_from": "#795548",
                "references": "#607D8B",
                "schema_mapping": "#3F51B5",
                "produces": "#8BC34A",
            }.get(relationship, "#999")
            st.markdown(
                f'<div style="text-align:center">'
                f'<span style="background:{rel_color};color:white;padding:2px 8px;border-radius:12px;font-size:0.8em">'
                f'{relationship.replace("_"," ")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                f'<div style="text-align:right">{product_badge(tgt_product)} **{tgt_type}**</div>',
                unsafe_allow_html=True,
            )

        # Description row
        st.caption(f"_{description}_")

        # 3-column detail
        detail_cols = st.columns([1, 1, 1])

        with detail_cols[0]:
            st.markdown("**ITOM API Type**")
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
                rel_strs = [f"→ {r['target']}" for r in entity_rels[:4]]
                st.markdown("Relationships: " + ", ".join(rel_strs))

        with detail_cols[2]:
            st.markdown("**Hashicorp/Target API Type**")
            tgt_fields = [
                (k, v)
                for k, v in api_lookup.items()
                if v.get("product_name") == tgt_product and v.get("type_name") == tgt_type
            ]
            st.markdown(f"`{tgt_product}.{tgt_type}` — {len(tgt_fields)} fields")

        # Accuracy badge
        if total > 0:
            pct = correct / total * 100
            badge_color = "#4CAF50" if pct >= 70 else "#FF9800" if pct >= 40 else "#F44336"
            st.markdown(
                f'<span style="background:{badge_color};color:white;padding:2px 10px;border-radius:12px;font-size:0.85em">'
                f'Field accuracy: {correct}/{total} ({pct:.0f}%)</span>'
                f' &nbsp; <span style="color:#888;font-size:0.8em">Model: {model_short}</span>',
                unsafe_allow_html=True,
            )

        # Expandable field mappings
        if predictions and gt:
            with st.expander(f"Field mappings ({len(gt)} fields)", expanded=False):
                render_field_table(predictions, gt, api_lookup)

        st.divider()


def main():
    entity_model, entity_mapping, samples, api_lookup, results = load_data()

    st.title("ITOM – Hashicorp Schema Mapping")
    st.markdown(
        "Mappings between **ITOM** (Instana, Concert, Turbonomic, Terraform) "
        "and **Hashicorp** (Infragraph, Packer, AWS, Azure, GCP, HCP) schema types. "
        "Predictions from Map-Reflect (Llama-3.3-70B, backfilled with Granite-4)."
    )

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")

        # Relationship type filter
        all_rels = sorted(set(s["metadata"]["relationship"] for s in samples))
        selected_rels = st.multiselect("Relationship type", all_rels, default=all_rels)

        # Source product filter
        all_src = sorted(set(s["source_schema"] for s in samples))
        selected_src = st.multiselect("Source product", all_src, default=all_src)

        # Target product filter
        all_tgt = sorted(set(s["target_schema"] for s in samples))
        selected_tgt = st.multiselect("Target product", all_tgt, default=all_tgt)

        # Cross-product only toggle
        cross_only = st.checkbox("Cross-product only (ITOM ↔ Hashi)", value=True)

        st.divider()
        st.header("Stats")
        st.metric("Total mappings", len(samples))
        with_results = sum(1 for s in samples if s["id"] in results)
        st.metric("With predictions", with_results)

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
            # Cross-product = at least one side is an ITOM-only product
            # (instana, concert, turbonomic) and the other is different.
            # This excludes Hashicorp-internal (hcp↔hcp_terraform, infragraph↔aws, packer↔infragraph)
            # but includes hcp_terraform→infragraph (terraform provisioning infra)
            has_itom = src in ITOM_ONLY or tgt in ITOM_ONLY
            if not has_itom:
                continue
        filtered.append(s)

    st.markdown(f"**Showing {len(filtered)} of {len(samples)} mappings**")

    for s in filtered:
        result = results.get(s["id"])
        render_mapping_card(s, result, entity_model, entity_mapping, api_lookup)


if __name__ == "__main__":
    main()
