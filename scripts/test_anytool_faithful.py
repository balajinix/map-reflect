#!/usr/bin/env python3
"""Test the faithful AnyTool implementation step by step."""
import os
import sys
import json

os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_DROP_PARAMS"] = "True"

sys.path.insert(0, str(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "anytool")))

import anytool_faithful as at
import logging
logging.basicConfig(level=logging.INFO)

td, ad, ctd = at.load_database()
categories = list(td.keys())
# Re-export for convenience
TOOL_DATA = at.TOOL_DATA
load_database = at.load_database
MetaAgent = at.MetaAgent
ToolAgent = at.ToolAgent
solve_mapping = at.solve_mapping
get_tools_in_category = at.get_tools_in_category
print("Categories:", categories)

query = "Map concert Application to instana schema"
source_fields = ["concert_Application.id", "concert_Application.name",
                 "concert_Application.status", "concert_Application.tags"]

model = "watsonx/meta-llama/llama-3-3-70b-instruct"

# Phase 1: Meta-agent
print("\n" + "=" * 50)
print("PHASE 1: Meta-Agent")
print("=" * 50)
meta = MetaAgent(query, model, categories)
selected = meta.run()
print(f"Selected categories: {selected}")
print(f"Tokens used: {meta.total_tokens}")

if not selected:
    print("No categories selected, using fallback")
    selected = ["Governance", "Observability"]

# Phase 2: Tool agents
print("\n" + "=" * 50)
print("PHASE 2: Tool Agents")
print("=" * 50)
api_pool = []
for cat in selected:
    if cat not in TOOL_DATA:
        print(f"  Skipping unknown category: {cat}")
        continue
    tools = list(TOOL_DATA[cat].keys())
    print(f"\n  Category: {cat}, Tools: {tools}")
    agent = ToolAgent(query, model, cat, tools, api_pool)
    agent.run()
    print(f"  Pool size after {cat}: {len(api_pool)}")
    print(f"  Tokens: {agent.total_tokens}")

print(f"\nTotal API pool: {len(api_pool)}")
for api in api_pool[:10]:
    print(f"  {api['tool_name']}.{api['api_name']}")

# Phase 3: Solver
print("\n" + "=" * 50)
print("PHASE 3: Solver")
print("=" * 50)
mappings = solve_mapping(query, source_fields, api_pool, model)
print(f"Mappings: {json.dumps(mappings, indent=2)}")
