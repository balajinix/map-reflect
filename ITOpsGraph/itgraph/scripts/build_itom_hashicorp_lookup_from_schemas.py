#!/usr/bin/env python3
"""
Build enriched API lookup from live GraphQL introspection + Hashicorp proto schemas.

Sources:
  - ITOM: data/itom_graphql_schema.json (GraphQL introspection result)
  - Hashicorp: /Users/balaji/code/infragraph-nl2q/schemas/proto/*.proto

Output:
  - data/itom_hashicorp_api_lookup_v2.json

Improvements over v1:
  - Proper field descriptions from GraphQL introspection and proto comments
  - Expanded nested fields (BaseVirtualMachine.base.id -> BaseVirtualMachine.id)
  - Field type information
  - Cross-references to embedded Base fields
"""

import json
import re
from pathlib import Path
from collections import OrderedDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROTO_DIR = Path("/Users/balaji/code/infragraph-nl2q/schemas/proto")

# Product classification for GraphQL types
# The ITOM GraphQL schema exposes types from multiple products through a unified API
GRAPHQL_PRODUCT_MAP = {
    # Instana types
    "Entity": "instana",
    "ExternalIds": "instana",
    "ExternalIdEntry": "instana",
    "DiscoveredBy": "instana",
    "EntityAspects": "instana",
    "VirtualMachineAspect": "instana",
    "StorageAspect": "instana",
    "ApplicationServiceAspect": "instana",
    "EntityReference": "instana",
    "EntityDetails": "instana",
    "Tag": "instana",
    "Property": "instana",
    "Resource": "instana",
    "TopologyNode": "instana",
    "Topology": "instana",
    "HealthSummary": "instana",
    "TypeHealth": "instana",
    "Event": "instana",
    "ApplicationPerspective": "instana",
    "ApplicationService": "instana",
    # Turbonomic types
    "Action": "turbonomic",
    "Risk": "turbonomic",
    "Savings": "turbonomic",
    "Statistic": "turbonomic",
    "StatValue": "turbonomic",
    "StatFilter": "turbonomic",
    "CVE": "instana",
}

# Product source mapping from proto file names
PROTO_PRODUCT_MAP = {
    "aws": "aws",
    "azure": "azure",
    "gcp": "gcp",
    "hcp": "hcp",
    "hcp_terraform": "hcp_terraform",
    "packer": "packer",
    "base": "infragraph",
}


def resolve_graphql_type(type_obj):
    """Resolve a GraphQL type to its name and modifiers."""
    is_list = False
    is_required = False
    t = type_obj
    while t:
        kind = t.get("kind", "")
        if kind == "NON_NULL":
            is_required = True
        elif kind == "LIST":
            is_list = True
        if t.get("name"):
            return t["name"], is_list, is_required
        t = t.get("ofType")
    return "Unknown", is_list, is_required


def parse_graphql_schema(schema_path):
    """Parse GraphQL introspection result into api_lookup entries."""
    with open(schema_path) as f:
        data = json.load(f)

    types = data["data"]["__schema"]["types"]
    entries = {}

    for t in types:
        if t["kind"] != "OBJECT" or t["name"].startswith("__"):
            continue
        type_name = t["name"]
        product = GRAPHQL_PRODUCT_MAP.get(type_name)
        if not product:
            continue

        type_desc = t.get("description") or ""
        fields = t.get("fields") or []

        for f in fields:
            field_name = f["name"]
            field_desc = f.get("description") or ""
            field_type, is_list, is_required = resolve_graphql_type(f["type"])

            key = f"{product}_{type_name}.{field_name}"
            entries[key] = {
                "product_name": product,
                "type_name": type_name,
                "field_name": field_name,
                "full_name": f"{type_name}.{field_name}",
                "description": field_desc,
                "field_type": field_type,
                "is_list": is_list,
                "is_required": is_required,
                "type_description": type_desc,
                "schema_source": "graphql",
            }

    return entries


def parse_proto_file(proto_path):
    """Parse a .proto file and extract message types with fields.

    Returns dict of message_name -> {fields: [...], comments: str}
    """
    with open(proto_path) as f:
        content = f.read()

    messages = {}
    # Match message blocks (handles nested messages by tracking braces)
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match message declaration
        m = re.match(r"^message\s+(\w+)\s*\{", line)
        if m:
            msg_name = m.group(1)
            # Collect comment above message
            msg_comment = ""
            j = i - 1
            while j >= 0 and lines[j].strip().startswith("//"):
                msg_comment = lines[j].strip().lstrip("/ ") + " " + msg_comment
                j -= 1

            # Parse fields inside the message
            fields = []
            brace_depth = 1
            i += 1
            while i < len(lines) and brace_depth > 0:
                fline = lines[i].strip()
                if "{" in fline:
                    brace_depth += fline.count("{")
                if "}" in fline:
                    brace_depth -= fline.count("}")

                if brace_depth == 1:  # Only parse top-level fields
                    # Match field declarations
                    fm = re.match(
                        r"^(?:optional\s+|repeated\s+|)?"
                        r"([\w.]+)\s+"  # type
                        r"(\w+)\s*=\s*(\d+)",  # name = number
                        fline,
                    )
                    if fm:
                        field_type = fm.group(1)
                        field_name = fm.group(2)
                        is_list = fline.startswith("repeated")
                        is_optional = fline.startswith("optional")

                        # Check for inline comment
                        field_comment = ""
                        cm = re.search(r"//\s*(.*)", fline)
                        if cm:
                            field_comment = cm.group(1).strip()
                        # Check comment on line above
                        if not field_comment and i > 0:
                            prev = lines[i - 1].strip()
                            if prev.startswith("//"):
                                field_comment = prev.lstrip("/ ").strip()

                        # Check for edge metadata options
                        edge_label = ""
                        node_label = ""
                        el = re.search(r'edge_label\)\s*=\s*"(\w+)"', fline)
                        if el:
                            edge_label = el.group(1)
                        nl = re.search(r'node_label\)\s*=\s*"(\w+)"', fline)
                        if nl:
                            node_label = nl.group(1)

                        # Check for map type
                        map_m = re.match(r"^map<(\w+),\s*(\w+)>", fline)
                        if map_m:
                            field_type = f"map<{map_m.group(1)}, {map_m.group(2)}>"
                            is_list = False

                        fields.append({
                            "name": field_name,
                            "type": field_type,
                            "is_list": is_list,
                            "is_optional": is_optional,
                            "comment": field_comment,
                            "edge_label": edge_label,
                            "node_label": node_label,
                        })
                i += 1

            messages[msg_name] = {
                "fields": fields,
                "comment": msg_comment.strip(),
            }
        else:
            i += 1

    return messages


def expand_base_fields(messages):
    """Expand embedded Base fields into parent messages.

    E.g., BaseVirtualMachine has `Base base = 1`, so we expand all Base fields
    into BaseVirtualMachine with prefix notation.

    Returns a dict of message_name -> list of expanded (field_name, field_type,
    description, is_list, embed_path).
    """
    expanded = {}

    for msg_name, msg_data in messages.items():
        flat_fields = []
        for field in msg_data["fields"]:
            embedded_msg = messages.get(field["type"])
            if embedded_msg and field["type"] != msg_name:
                # This field is an embedded message - expand its fields
                for sub_field in embedded_msg["fields"]:
                    # Recursively expand (one more level for Base inside BaseXxx)
                    sub_embedded = messages.get(sub_field["type"])
                    if sub_embedded and sub_field["type"] != field["type"]:
                        for sub_sub in sub_embedded["fields"]:
                            desc = sub_sub["comment"]
                            if sub_sub["edge_label"]:
                                desc += f" (edge: {sub_sub['edge_label']} -> {sub_sub['node_label']})"
                            flat_fields.append({
                                "name": sub_sub["name"],
                                "type": sub_sub["type"],
                                "description": desc.strip(),
                                "is_list": sub_sub["is_list"],
                                "embed_path": f"{field['name']}.{sub_field['name']}.{sub_sub['name']}",
                                "source_level": "base",
                            })
                    else:
                        desc = sub_field["comment"]
                        if sub_field["edge_label"]:
                            desc += f" (edge: {sub_field['edge_label']} -> {sub_field['node_label']})"
                        flat_fields.append({
                            "name": sub_field["name"],
                            "type": sub_field["type"],
                            "description": desc.strip(),
                            "is_list": sub_field["is_list"],
                            "embed_path": f"{field['name']}.{sub_field['name']}",
                            "source_level": "domain",
                        })
            else:
                desc = field["comment"]
                if field["edge_label"]:
                    desc += f" (edge: {field['edge_label']} -> {field['node_label']})"
                flat_fields.append({
                    "name": field["name"],
                    "type": field["type"],
                    "description": desc.strip(),
                    "is_list": field["is_list"],
                    "embed_path": field["name"],
                    "source_level": "own",
                })

        expanded[msg_name] = flat_fields

    return expanded


def build_proto_lookup(proto_dir):
    """Parse all proto files and build api_lookup entries with expanded fields."""
    entries = {}

    # Skip non-domain protos
    skip_files = {"model.proto", "resource.proto", "edge.proto"}

    # First pass: collect all messages across all files
    all_messages = {}
    file_messages = {}  # track which file each message came from

    for proto_file in sorted(proto_dir.glob("*.proto")):
        if proto_file.name in skip_files:
            continue
        messages = parse_proto_file(proto_file)
        product = PROTO_PRODUCT_MAP.get(proto_file.stem, proto_file.stem)
        for msg_name, msg_data in messages.items():
            all_messages[msg_name] = msg_data
            file_messages[msg_name] = product

    # Expand embedded fields
    expanded = expand_base_fields(all_messages)

    # Build lookup entries
    for msg_name, flat_fields in expanded.items():
        product = file_messages.get(msg_name, "unknown")

        # Skip resource-only types (AwsResource, AzureResource, etc.)
        if msg_name.endswith("Resource") and not msg_name.startswith("Base"):
            continue

        msg_comment = all_messages[msg_name]["comment"]

        for field in flat_fields:
            key = f"{product}_{msg_name}.{field['name']}"

            # Skip if this is the embedded message reference itself (e.g., base=1, virtual_machine=1)
            if field["type"] in all_messages and field["source_level"] == "own":
                continue

            entries[key] = {
                "product_name": product,
                "type_name": msg_name,
                "field_name": field["name"],
                "full_name": f"{msg_name}.{field['name']}",
                "description": field["description"],
                "field_type": field["type"],
                "is_list": field["is_list"],
                "is_required": False,
                "type_description": msg_comment,
                "schema_source": "proto",
                "embed_path": field["embed_path"],
                "source_level": field["source_level"],
            }

    return entries


def merge_with_v1(v2_entries, v1_path):
    """Merge v2 entries with v1 lookup, preferring v2 where available but
    keeping v1 entries that don't exist in v2 (e.g., concert, turbonomic
    types that come from the old ITOpsGraph lookup, not from GraphQL
    introspection)."""
    if not v1_path.exists():
        return v2_entries

    with open(v1_path) as f:
        v1 = json.load(f)

    merged = dict(v2_entries)  # start with v2

    for key, v1_entry in v1.items():
        if key not in merged:
            # Keep v1 entry but mark its source
            v1_entry["schema_source"] = "itopsgraph_v1"
            merged[key] = v1_entry
        else:
            # v2 exists — backfill description if v2 is empty but v1 has one
            if not merged[key].get("description") and v1_entry.get("description"):
                merged[key]["description"] = v1_entry["description"]

    return merged


def main():
    print("=== Building Enriched ITOM-Hashicorp API Lookup (v2) ===\n")

    # Step 1: Parse GraphQL schema
    graphql_path = DATA_DIR / "itom_graphql_schema.json"
    if graphql_path.exists():
        print(f"Parsing GraphQL schema from {graphql_path}...")
        graphql_entries = parse_graphql_schema(graphql_path)
        print(f"  {len(graphql_entries)} fields from GraphQL introspection")

        # Count by product
        by_product = {}
        for e in graphql_entries.values():
            p = e["product_name"]
            by_product[p] = by_product.get(p, 0) + 1
        for p, c in sorted(by_product.items()):
            print(f"    {p}: {c} fields")
    else:
        print(f"WARNING: GraphQL schema not found at {graphql_path}")
        graphql_entries = {}

    # Step 2: Parse Proto schemas
    print(f"\nParsing proto schemas from {PROTO_DIR}...")
    proto_entries = build_proto_lookup(PROTO_DIR)
    print(f"  {len(proto_entries)} fields from proto schemas (with Base expansion)")

    by_product = {}
    by_type = {}
    for e in proto_entries.values():
        p = e["product_name"]
        t = e["type_name"]
        by_product[p] = by_product.get(p, 0) + 1
        by_type[t] = by_type.get(t, 0) + 1
    for p, c in sorted(by_product.items()):
        print(f"    {p}: {c} fields")
    print(f"  {len(by_type)} distinct message types")

    # Step 3: Merge GraphQL + Proto
    all_entries = {}
    all_entries.update(graphql_entries)
    all_entries.update(proto_entries)
    print(f"\nCombined: {len(all_entries)} fields")

    # Step 4: Merge with v1 (keep concert/turbonomic types from old lookup)
    v1_path = DATA_DIR / "itom_hashicorp_api_lookup.json"
    merged = merge_with_v1(all_entries, v1_path)
    print(f"After merging with v1: {len(merged)} fields")

    # Stats
    by_source = {}
    for e in merged.values():
        s = e.get("schema_source", "unknown")
        by_source[s] = by_source.get(s, 0) + 1
    for s, c in sorted(by_source.items()):
        print(f"  source={s}: {c}")

    # Count fields with descriptions
    with_desc = sum(1 for e in merged.values() if e.get("description"))
    print(f"  Fields with descriptions: {with_desc}/{len(merged)} ({100*with_desc/len(merged):.0f}%)")

    # Step 5: Save
    out_path = DATA_DIR / "itom_hashicorp_api_lookup_v2.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
