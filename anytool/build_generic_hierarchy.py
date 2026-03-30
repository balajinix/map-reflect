#!/usr/bin/env python3
"""Build AnyTool hierarchy from any API lookup file."""
import json
import sys
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent


def build_hierarchy(lookup_path, output_prefix):
    with open(lookup_path) as f:
        api_lookup = json.load(f)

    tool_data = {}
    api_details = {}

    for field_key, meta in api_lookup.items():
        # Handle both ITOpsGraph format (product_name/type_name/field_name)
        # and ToolBench format (category/tool/api + name)
        product = meta.get("product_name", meta.get("category", ""))
        type_name = meta.get("type_name", meta.get("tool", ""))
        # For ToolBench, field_name is the parameter name (from key), not the API name
        # key format: Tool_API.parameter -> extract parameter after last dot
        field_name = meta.get("field_name")
        if not field_name:
            field_name = field_key.split(".")[-1] if "." in field_key else field_key
        description = meta.get("description", "")

        if not product or not type_name:
            continue

        category = product
        # Use the actual tool/type as the tool within the category
        tool_key = type_name
        if category not in tool_data:
            tool_data[category] = {}
            api_details[category] = {}

        if tool_key not in tool_data[category]:
            tool_data[category][tool_key] = {"api_list_names": [], "tool_description": tool_key + " schema"}
            api_details[category][tool_key] = {"tool_name": tool_key, "tool_description": tool_key, "api_list": []}

        api_name = type_name + "." + field_name
        tool_data[category][tool_key]["api_list_names"].append(api_name)
        api_details[category][tool_key]["api_list"].append({
            "name": api_name,
            "description": description[:100] if description else field_name + " in " + type_name,
        })

    out_dir = Path(__file__).parent
    with open(out_dir / ("tool_data_" + output_prefix + ".json"), "w") as f:
        json.dump(tool_data, f, indent=2)
    with open(out_dir / ("api_details_" + output_prefix + ".json"), "w") as f:
        json.dump(api_details, f, indent=2)
    with open(out_dir / ("category_tool_details_" + output_prefix + ".json"), "w") as f:
        ctd = {cat: {"description": cat, "tools": {t: t for t in tool_data[cat]}} for cat in tool_data}
        json.dump(ctd, f, indent=2)

    total = sum(len(tool_data[c][t]["api_list_names"]) for c in tool_data for t in tool_data[c])
    print(output_prefix + ": " + str(len(tool_data)) + " categories, " + str(total) + " fields")


datasets = [
    ("nl2gql", "data/nl2gql_api_lookup.json"),
    ("toolbench", "data/toolbench_api_lookup.json"),
]

for prefix, lookup in datasets:
    path = BASE / lookup
    if path.exists():
        build_hierarchy(path, prefix)
    else:
        print(prefix + ": lookup not found at " + str(path))
