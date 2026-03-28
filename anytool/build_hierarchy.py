#!/usr/bin/env python3
"""
Build AnyTool 3-tier hierarchy from ITOpsGraph API lookup.

Creates:
  - tool_data.json: {category: {tool/product: {api_list_names: [fields], description: ...}}}
  - api_details.json: {category: {tool: {api_list: [{name, description, parameters}]}}}
  - category_tool_details.json: {category: {description, tools: {tool: description}}}

Mapping:
  Category = domain (observability, governance, infrastructure, cloud, data)
  Tool = product (instana, concert, turbonomic, aws, infragraph, etc.)
  API = type.field (Entity.uuid, Application.id, AwsEc2Instance.state, etc.)
"""
import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent

# Product-to-category mapping
PRODUCT_CATEGORIES = {
    "instana": "Observability",
    "turbonomic": "Optimization",
    "concert": "Governance",
    "cloudability": "FinOps",
    "ns1": "Networking",
    "watsonx-orchestrate": "AI_Orchestration",
    "aws": "Cloud_Infrastructure",
    "azure": "Cloud_Infrastructure",
    "gcp": "Cloud_Infrastructure",
    "infragraph": "Infrastructure_Graph",
    "packer": "Image_Management",
    "hcp": "Cloud_Platform",
    "hcp_terraform": "Infrastructure_as_Code",
    "base": "Infrastructure_Graph",
    "model": "Infrastructure_Graph",
    "edge": "Infrastructure_Graph",
    "resource": "Infrastructure_Graph",
}

CATEGORY_DESCRIPTIONS = {
    "Observability": "Monitoring, tracing, and alerting for application and infrastructure health",
    "Optimization": "Resource optimization, cost management, and performance tuning",
    "Governance": "Application lifecycle governance, compliance, and vulnerability management",
    "FinOps": "Cloud financial operations, cost reporting, and utilization analytics",
    "Networking": "DNS management, traffic steering, and network monitoring",
    "AI_Orchestration": "AI workflow orchestration, skill management, and automation",
    "Cloud_Infrastructure": "Cloud provider resources (compute, storage, networking, IAM)",
    "Infrastructure_Graph": "Unified infrastructure graph model and resource relationships",
    "Image_Management": "Machine image building, versioning, and artifact management",
    "Cloud_Platform": "HashiCorp Cloud Platform services and management",
    "Infrastructure_as_Code": "Terraform workspaces, runs, and infrastructure provisioning",
}


def build_hierarchy(api_lookup_path):
    """Build 3-tier hierarchy from API lookup."""
    with open(api_lookup_path) as f:
        api_lookup = json.load(f)

    # Group fields by product and type
    product_types = defaultdict(lambda: defaultdict(list))
    for field_key, meta in api_lookup.items():
        product = meta.get("product_name", "")
        type_name = meta.get("type_name", "")
        field_name = meta.get("field_name", "")
        description = meta.get("description", "")
        field_type = meta.get("field_type", "")

        if not product or not type_name:
            continue

        product_types[product][type_name].append({
            "name": f"{type_name}.{field_name}",
            "field_name": field_name,
            "description": description or f"{field_name} field in {type_name}",
            "type": field_type,
            "full_key": field_key,
        })

    # Build tool_data.json (3-tier)
    tool_data = defaultdict(dict)
    api_details = defaultdict(lambda: defaultdict(dict))
    category_tool_details = {}

    for product, types in product_types.items():
        category = PRODUCT_CATEGORIES.get(product, "Other")

        # Tool = product
        tool_name = product
        api_list_names = []
        api_list_detailed = []

        for type_name, fields in types.items():
            for field in fields:
                api_list_names.append(field["name"])
                api_list_detailed.append({
                    "name": field["name"],
                    "description": field["description"],
                    "method": "GET",
                    "required_parameters": [],
                    "optional_parameters": [
                        {"name": "field_type", "type": "STRING", "description": field["type"]}
                    ],
                })

        tool_data[category][tool_name] = {
            "api_list_names": api_list_names,
            "tool_description": f"{product} API - {len(types)} types, {len(api_list_names)} fields",
        }

        api_details[category][tool_name] = {
            "tool_name": tool_name,
            "tool_description": f"{product} API schema with {len(types)} types",
            "api_list": api_list_detailed,
        }

    # Build category_tool_details
    for category in tool_data:
        tools_in_cat = {}
        for tool_name in tool_data[category]:
            n_apis = len(tool_data[category][tool_name]["api_list_names"])
            tools_in_cat[tool_name] = f"{tool_name} product with {n_apis} API fields"

        category_tool_details[category] = {
            "description": CATEGORY_DESCRIPTIONS.get(category, f"{category} domain"),
            "tools": tools_in_cat,
        }

    return dict(tool_data), dict(api_details), category_tool_details


def main():
    api_lookup_path = BASE / "data" / "itopsgraph_api_lookup.json"
    if not api_lookup_path.exists():
        api_lookup_path = BASE / "data" / "itom_hashicorp_api_lookup_v2.json"

    print(f"Building hierarchy from {api_lookup_path}...")

    tool_data, api_details, category_tool_details = build_hierarchy(api_lookup_path)

    # Save
    out_dir = Path(__file__).parent
    with open(out_dir / "tool_data.json", "w") as f:
        json.dump(tool_data, f, indent=2)
    with open(out_dir / "api_details.json", "w") as f:
        json.dump(api_details, f, indent=2)
    with open(out_dir / "category_tool_details.json", "w") as f:
        json.dump(category_tool_details, f, indent=2)

    # Stats
    total_categories = len(tool_data)
    total_tools = sum(len(v) for v in tool_data.values())
    total_apis = sum(
        len(tool_data[c][t]["api_list_names"])
        for c in tool_data for t in tool_data[c]
    )
    print(f"\nHierarchy built:")
    print(f"  Categories: {total_categories}")
    print(f"  Tools (products): {total_tools}")
    print(f"  APIs (fields): {total_apis}")
    print(f"\nCategories:")
    for cat, tools in sorted(tool_data.items()):
        n_tools = len(tools)
        n_apis = sum(len(t["api_list_names"]) for t in tools.values())
        print(f"  {cat}: {n_tools} tools, {n_apis} APIs")

    print(f"\nSaved: tool_data.json, api_details.json, category_tool_details.json")


if __name__ == "__main__":
    main()
