#!/usr/bin/env python3
"""Debug AnyTool on OMOP-CMS: trace one sample end-to-end."""
import os
import json
import sys
import logging

os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_DROP_PARAMS"] = "True"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "anytool"))
import anytool_faithful as at

logging.basicConfig(level=logging.INFO)

# Load OMOP hierarchy
at.load_database(hierarchy_prefix="omop_cms")
print("Categories:", list(at.TOOL_DATA.keys()))
for cat in at.TOOL_DATA:
    for tool in at.TOOL_DATA[cat]:
        names = at.TOOL_DATA[cat][tool]["api_list_names"]
        print(f"  {cat}/{tool}: {len(names)} fields")
        print(f"    First 5: {names[:5]}")

# Load one sample
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
with open(os.path.join(base, "data/omop_cms_dataset.jsonl")) as f:
    s = json.loads(f.readline())

query = s["utterance"]
gt = s["mapping"]
source_fields = list(gt.keys())

print(f"\nQuery: {query}")
print(f"Source fields ({len(source_fields)}): {source_fields[:5]}")
print(f"GT targets: {list(gt.values())[:5]}")

# Extract products
src, tgt = at.extract_products_from_query(query)
print(f"\nExtracted products: src={src}, tgt={tgt}")

# Determine categories
src_cat = at.find_product_category(src) if src else ""
print(f"Source category: {src_cat}")

cats = set()
if src:
    c = at.find_product_category(src)
    if c:
        cats.add(c)
if tgt:
    c = at.find_product_category(tgt)
    if c:
        cats.add(c)
if len(cats) <= 1:
    cats = set(at.TOOL_DATA.keys())
print(f"Categories to explore: {cats}")

# Which tools would be searched?
for cat in cats:
    tools = list(at.TOOL_DATA[cat].keys())
    if src:
        tools = [t for t in tools if t != src]
    print(f"  {cat}: search tools = {tools}")

# Run step by step
print("\n" + "=" * 50)
print("PHASE 2: Tool Agent on CMS")
print("=" * 50)
api_pool = []
tools_to_search = ["CMS"]
agent = at.ToolAgent(query, "watsonx/meta-llama/llama-3-3-70b-instruct",
                     "CMS", tools_to_search, api_pool)
agent.run()
print(f"\nAPI pool after tool agent: {len(api_pool)} items")
for api in api_pool[:10]:
    print(f"  {api}")

print("\n" + "=" * 50)
print("PHASE 3: Solver")
print("=" * 50)

if not api_pool:
    # Force add some CMS fields to test the solver
    print("Pool empty! Forcing CMS fields into pool...")
    for field in at.TOOL_DATA["CMS"]["CMS"]["api_list_names"][:20]:
        api_pool.append({"category_name": "CMS", "tool_name": "CMS", "api_name": field})
    print(f"Forced pool size: {len(api_pool)}")

result = at.solve_mapping(query, source_fields, api_pool, "watsonx/meta-llama/llama-3-3-70b-instruct")
print(f"\nMappings: {json.dumps(result, indent=2)}")
print(f"\nGT (first 3):")
for k, v in list(gt.items())[:3]:
    print(f"  {k} -> {v}")
