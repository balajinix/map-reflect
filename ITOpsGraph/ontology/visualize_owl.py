"""
Visualize an OWL ontology as an interactive HTML network graph.

Usage:
    python visualize_owl.py <path_to_owl_file> [--output output.html]

Requirements:
    pip install owlready2 pyvis
"""

import argparse
import sys
from owlready2 import get_ontology, Thing
from pyvis.network import Network


def load_ontology(owl_path):
    """Load an OWL file and return the ontology object."""
    onto = get_ontology(owl_path).load()
    return onto


def build_graph(onto, output_path="ontology_graph.html"):
    """Build an interactive pyvis network from the ontology."""
    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#333333",
        notebook=False,
    )

    net.barnes_hut(
        gravity=-5000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.05,
    )

    added_nodes = set()

    # --- Add classes and subclass relationships ---
    for cls in onto.classes():
        cls_name = cls.name
        if cls_name not in added_nodes:
            net.add_node(
                cls_name,
                label=cls_name,
                color="#4a90d9",
                shape="ellipse",
                size=25,
                title=f"Class: {cls.iri}",
            )
            added_nodes.add(cls_name)

        for parent in cls.is_a:
            if hasattr(parent, "name") and parent is not Thing:
                parent_name = parent.name
                if parent_name not in added_nodes:
                    net.add_node(
                        parent_name,
                        label=parent_name,
                        color="#4a90d9",
                        shape="ellipse",
                        size=25,
                        title=f"Class: {parent.iri}",
                    )
                    added_nodes.add(parent_name)
                net.add_edge(
                    cls_name,
                    parent_name,
                    label="subClassOf",
                    color="#888888",
                    arrows="to",
                    font={"size": 10, "color": "#888888"},
                )

    # --- Add object properties ---
    for prop in onto.object_properties():
        prop_name = prop.name
        domains = list(prop.domain)
        ranges = list(prop.range)

        for d in domains:
            if not hasattr(d, "name"):
                continue
            d_name = d.name
            if d_name not in added_nodes:
                net.add_node(
                    d_name,
                    label=d_name,
                    color="#4a90d9",
                    shape="ellipse",
                    size=25,
                    title=f"Class: {d.iri}",
                )
                added_nodes.add(d_name)

            for r in ranges:
                if not hasattr(r, "name"):
                    continue
                r_name = r.name
                if r_name not in added_nodes:
                    net.add_node(
                        r_name,
                        label=r_name,
                        color="#4a90d9",
                        shape="ellipse",
                        size=25,
                        title=f"Class: {r.iri}",
                    )
                    added_nodes.add(r_name)
                net.add_edge(
                    d_name,
                    r_name,
                    label=prop_name,
                    color="#e07b39",
                    arrows="to",
                    font={"size": 10, "color": "#e07b39"},
                )

    # --- Add data properties as attribute nodes ---
    for prop in onto.data_properties():
        prop_name = prop.name
        domains = list(prop.domain)

        for d in domains:
            if not hasattr(d, "name"):
                continue
            d_name = d.name
            dp_node = f"dp_{prop_name}"
            if dp_node not in added_nodes:
                net.add_node(
                    dp_node,
                    label=prop_name,
                    color="#50b86c",
                    shape="box",
                    size=15,
                    title=f"Data Property: {prop.iri}",
                )
                added_nodes.add(dp_node)
            net.add_edge(
                d_name,
                dp_node,
                label="hasProperty",
                color="#50b86c",
                arrows="to",
                font={"size": 10, "color": "#50b86c"},
            )

    # --- Add individuals ---
    for ind in onto.individuals():
        ind_name = ind.name
        if ind_name not in added_nodes:
            net.add_node(
                ind_name,
                label=ind_name,
                color="#d94a7a",
                shape="diamond",
                size=20,
                title=f"Individual: {ind.iri}",
            )
            added_nodes.add(ind_name)

        for cls in ind.is_a:
            if hasattr(cls, "name"):
                cls_name = cls.name
                if cls_name not in added_nodes:
                    net.add_node(
                        cls_name,
                        label=cls_name,
                        color="#4a90d9",
                        shape="ellipse",
                        size=25,
                        title=f"Class: {cls.iri}",
                    )
                    added_nodes.add(cls_name)
                net.add_edge(
                    ind_name,
                    cls_name,
                    label="instanceOf",
                    color="#d94a7a",
                    arrows="to",
                    font={"size": 10, "color": "#d94a7a"},
                )

    # --- Summary ---
    node_count = len(net.nodes)
    edge_count = len(net.edges)
    print(f"Graph built: {node_count} nodes, {edge_count} edges")

    if node_count == 0:
        print("Warning: No classes, properties, or individuals found in the ontology.")
        print("The OWL file might use imports or complex constructs not captured here.")
        return None

    net.set_options("""
    {
      "interaction": {
        "navigationButtons": true,
        "keyboard": true,
        "hover": true
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -5000,
          "springLength": 200
        },
        "stabilization": {
          "iterations": 200
        }
      }
    }
    """)

    net.save_graph(output_path)

    # Add title and legend by modifying the saved HTML
    header_html = """
    <div style="font-family: sans-serif; padding: 10px; position: fixed; top: 10px; left: 10px; background: white; border: 1px solid #ccc; border-radius: 5px; z-index: 1000;">
        <div style="margin: 0 0 8px 0; color: #333; font-size: 14px; font-weight: bold;">ITOpsGraph Ontology</div>
        <div style="font-size: 11px;">
            <b>Legend:</b>
            <span style="color:#4a90d9;">&#9679; Class</span> |
            <span style="color:#e07b39;">&#9679; Object Property</span> |
            <span style="color:#50b86c;">&#9632; Data Property</span> |
            <span style="color:#d94a7a;">&#9670; Individual</span> |
            <span style="color:#888888;">-- subClassOf</span>
        </div>
    </div>
    """
    with open(output_path, 'r') as f:
        html_content = f.read()
    html_content = html_content.replace('<title>Network</title>', '<title>ITOpsGraph Ontology</title>', 1)
    html_content = html_content.replace('<body>', f'<body>{header_html}', 1)
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Saved interactive graph to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize an OWL ontology as an interactive HTML graph."
    )
    parser.add_argument("owl_file", help="Path to the .owl file")
    parser.add_argument(
        "--output", "-o",
        default="ontology_graph.html",
        help="Output HTML file (default: ontology_graph.html)",
    )
    args = parser.parse_args()

    owl_path = args.owl_file
    # owlready2 expects a file URI for local files
    if not owl_path.startswith("http") and not owl_path.startswith("file://"):
        import os
        owl_path = "file://" + os.path.abspath(owl_path)

    print(f"Loading ontology from: {owl_path}")
    onto = load_ontology(owl_path)
    print(f"Ontology loaded: {onto.base_iri}")
    print(f"  Classes: {len(list(onto.classes()))}")
    print(f"  Object Properties: {len(list(onto.object_properties()))}")
    print(f"  Data Properties: {len(list(onto.data_properties()))}")
    print(f"  Individuals: {len(list(onto.individuals()))}")

    build_graph(onto, args.output)


if __name__ == "__main__":
    main()
