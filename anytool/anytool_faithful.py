#!/usr/bin/env python3
"""
Faithful AnyTool implementation for schema mapping.

Implements the 3-tier hierarchical agent architecture from the AnyTool paper:
  Meta-agent → Category agents → Tool agents → API candidate pool → Solver

Uses function calling (via LiteLLM) to navigate the hierarchy.
Self-reflection re-activates agents when the solver fails.
"""
import json
import os
import sys
import time
import logging
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
load_dotenv()

os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_DROP_PARAMS"] = "True"

import litellm

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
TOOL_DATA = None
API_DETAILS = None
CATEGORY_TOOL_DETAILS = None


def load_database(hierarchy_prefix=None):
    """Load the 3-tier hierarchy.

    Args:
        hierarchy_prefix: if set, load tool_data_{prefix}.json etc.
                          Otherwise load tool_data.json (default ITOpsGraph).
    """
    global TOOL_DATA, API_DETAILS, CATEGORY_TOOL_DETAILS
    if TOOL_DATA is None or hierarchy_prefix is not None:
        suffix = "_" + hierarchy_prefix if hierarchy_prefix else ""
        td_path = BASE / ("tool_data" + suffix + ".json")
        ad_path = BASE / ("api_details" + suffix + ".json")
        ctd_path = BASE / ("category_tool_details" + suffix + ".json")
        with open(td_path) as f:
            TOOL_DATA = json.load(f)
        with open(ad_path) as f:
            API_DETAILS = json.load(f)
        if ctd_path.exists():
            with open(ctd_path) as f:
                CATEGORY_TOOL_DETAILS = json.load(f)
        else:
            CATEGORY_TOOL_DETAILS = {}
    return TOOL_DATA, API_DETAILS, CATEGORY_TOOL_DETAILS


# ============================================================================
# Database query functions (called by agents via function calling)
# ============================================================================

def query_all_categories() -> list:
    """Query all categories in the database."""
    return list(TOOL_DATA.keys())


def get_tools_in_category(category_name: str) -> list:
    """Get all tools (products) in a specific category."""
    if category_name not in TOOL_DATA:
        return f"Category '{category_name}' not found. Available: {list(TOOL_DATA.keys())}"
    return list(TOOL_DATA[category_name].keys())


def get_tool_descriptions(tool_list: list) -> dict:
    """Get descriptions for a list of tools."""
    result = {}
    for category in TOOL_DATA:
        for tool_name, tool_data in TOOL_DATA[category].items():
            if tool_name in tool_list:
                result[tool_name] = {
                    "description": tool_data.get("tool_description", ""),
                    "num_apis": len(tool_data.get("api_list_names", [])),
                    "category": category,
                }
    return result


def find_product_category(product_name: str) -> str:
    """Find which category a product belongs to."""
    for category, tools in TOOL_DATA.items():
        if product_name in tools:
            return category
    return ""


def get_apis_in_tool(category_name: str, tool_name: str) -> list:
    """Get all API names in a specific tool, grouped by type."""
    if category_name not in TOOL_DATA:
        return f"Category '{category_name}' not found"
    if tool_name not in TOOL_DATA[category_name]:
        return f"Tool '{tool_name}' not found in '{category_name}'"
    apis = TOOL_DATA[category_name][tool_name]["api_list_names"]
    # Group by type for readability
    types = {}
    for api in apis:
        parts = api.split(".", 1)
        type_name = parts[0] if len(parts) == 2 else "other"
        if type_name not in types:
            types[type_name] = []
        types[type_name].append(api)
    # Return type summary (avoids context overflow)
    summary = []
    for type_name, fields in sorted(types.items()):
        if len(fields) <= 5:
            summary.append(f"{type_name}: {fields}")
        else:
            summary.append(f"{type_name}: {fields[:5]} ... ({len(fields)} total)")
    if len(summary) > 30:
        return summary[:30] + [f"... and {len(summary) - 30} more types"]
    return summary


def get_api_details(category_name: str, tool_name: str, api_name: str) -> dict:
    """Get details of a specific API."""
    if category_name in API_DETAILS and tool_name in API_DETAILS[category_name]:
        for api in API_DETAILS[category_name][tool_name].get("api_list", []):
            if api["name"] == api_name:
                return api
    return f"API '{api_name}' not found in {category_name}/{tool_name}"


# ============================================================================
# Function schemas for LLM function calling
# ============================================================================

META_AGENT_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_tools_in_category",
            "description": "Get all tools/products in a specific category",
            "parameters": {
                "type": "object",
                "properties": {
                    "category_name": {"type": "string", "description": "Category name"}
                },
                "required": ["category_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_tool_descriptions",
            "description": "Get descriptions for a list of tools",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tool names"
                    }
                },
                "required": ["tool_list"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "select_category",
            "description": "Select a category to explore for relevant APIs",
            "parameters": {
                "type": "object",
                "properties": {
                    "category_name": {"type": "string"}
                },
                "required": ["category_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish_search",
            "description": "Finish the category search",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]

TOOL_AGENT_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_apis_in_tool",
            "description": "Get all API field names in a specific tool/product",
            "parameters": {
                "type": "object",
                "properties": {
                    "category_name": {"type": "string"},
                    "tool_name": {"type": "string"}
                },
                "required": ["category_name", "tool_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_api_details",
            "description": "Get details of a specific API field",
            "parameters": {
                "type": "object",
                "properties": {
                    "category_name": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "api_name": {"type": "string"}
                },
                "required": ["category_name", "tool_name", "api_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_api_to_pool",
            "description": "Add an API to the candidate pool for mapping",
            "parameters": {
                "type": "object",
                "properties": {
                    "category_name": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "api_name": {"type": "string"}
                },
                "required": ["category_name", "tool_name", "api_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish_search",
            "description": "Finish exploring this tool",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]


# ============================================================================
# LLM calling with function support
# ============================================================================

def call_llm(messages, tools=None, model="watsonx/meta-llama/llama-3-3-70b-instruct",
             max_tokens=2048, temperature=0.0, max_retries=3):
    """Call LLM with optional function calling and retry logic."""
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = litellm.completion(**kwargs)
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(2 * (attempt + 1))  # backoff
            else:
                logger.error(f"LLM call failed after {max_retries} retries: {e}")
                return None


# ============================================================================
# Agent classes
# ============================================================================

class AnyToolAgent:
    """Base agent with function-calling loop."""

    def __init__(self, model, system_prompt, tools_schema, max_steps=8):
        self.model = model
        self.messages = [{"role": "system", "content": system_prompt}]
        self.tools_schema = tools_schema
        self.max_steps = max_steps
        self.finished = False
        self.total_tokens = 0

    def step(self):
        """Execute one agent step with function calling."""
        response = call_llm(self.messages, tools=self.tools_schema, model=self.model)
        if response is None:
            self.finished = True
            return None

        self.total_tokens += response.usage.total_tokens if response.usage else 0
        msg = response.choices[0].message

        # Check for tool calls
        if msg.tool_calls:
            # Add assistant message with tool calls
            self.messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            })
            return msg.tool_calls
        else:
            # No tool calls — add response and finish
            self.messages.append({"role": "assistant", "content": msg.content or ""})
            self.finished = True
            return None

    def add_tool_result(self, tool_call_id, name, result):
        """Add a tool execution result."""
        self.messages.append({
            "tool_call_id": tool_call_id,
            "role": "tool",
            "name": name,
            "content": json.dumps(result) if not isinstance(result, str) else result,
        })


class MetaAgent(AnyToolAgent):
    """Top-level agent that selects categories."""

    def __init__(self, query, model, categories):
        system_prompt = (
            f"You are a schema mapping assistant. Given a mapping query, you need to find "
            f"ALL relevant API categories. The database has these categories: {categories}. "
            f"The query asks to map from one product to another. You MUST select the categories "
            f"for BOTH the source AND target products. For example, if the query says "
            f"'Map concert Application to instana schema', select BOTH the category containing "
            f"concert AND the category containing instana. "
            f"Use get_tools_in_category to see which products are in each category. "
            f"Use select_category to choose categories. Select at least 2 categories. "
            f"Call finish_search when done."
        )
        super().__init__(model, system_prompt, META_AGENT_FUNCTIONS)
        self.messages.append({"role": "user", "content": query})
        self.selected_categories = []

    def run(self):
        """Run the meta-agent to select categories."""
        for _ in range(self.max_steps):
            if self.finished:
                break
            tool_calls = self.step()
            if tool_calls is None:
                break
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except:
                    args = {}

                if name == "get_tools_in_category":
                    result = get_tools_in_category(args.get("category_name", ""))
                elif name == "get_tool_descriptions":
                    result = get_tool_descriptions(args.get("tool_list", []))
                elif name == "select_category":
                    cat = args.get("category_name", "")
                    if cat and cat not in self.selected_categories:
                        self.selected_categories.append(cat)
                    result = f"Category '{cat}' selected"
                elif name == "finish_search":
                    self.finished = True
                    result = "Search finished"
                else:
                    result = f"Unknown function: {name}"

                self.add_tool_result(tc.id, name, result)

        return self.selected_categories


class ToolAgent(AnyToolAgent):
    """Agent that explores tools within a category and adds APIs to the pool."""

    def __init__(self, query, model, category, tools, api_pool):
        system_prompt = (
            f"You are a schema mapping assistant exploring the '{category}' category. "
            f"Available tools (products): {tools}. "
            f"Step 1: Use get_apis_in_tool to list fields for each tool. "
            f"Step 2: Identify fields that could be mapping targets (e.g., id, name, status, "
            f"tags, description, type, label, uuid — common schema fields). "
            f"Step 3: Use add_api_to_pool to add each relevant field. Add at least 5-10 fields. "
            f"Step 4: Call finish_search when done. "
            f"Be thorough — add any field that might semantically match a source field."
        )
        super().__init__(model, system_prompt, TOOL_AGENT_FUNCTIONS)
        self.messages.append({"role": "user", "content": query})
        self.category = category
        self.api_pool = api_pool

    def run(self):
        """Run the tool agent to find relevant APIs."""
        for _ in range(self.max_steps):
            if self.finished:
                break
            tool_calls = self.step()
            if tool_calls is None:
                break
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except:
                    args = {}

                if name == "get_apis_in_tool":
                    result = get_apis_in_tool(
                        args.get("category_name", ""),
                        args.get("tool_name", "")
                    )
                elif name == "get_api_details":
                    result = get_api_details(
                        args.get("category_name", ""),
                        args.get("tool_name", ""),
                        args.get("api_name", "")
                    )
                elif name == "add_api_to_pool":
                    api_entry = {
                        "category_name": args.get("category_name", ""),
                        "tool_name": args.get("tool_name", ""),
                        "api_name": args.get("api_name", ""),
                    }
                    self.api_pool.append(api_entry)
                    result = f"Added {api_entry['api_name']} to pool. Pool size: {len(self.api_pool)}"
                elif name == "finish_search":
                    self.finished = True
                    result = "Search finished"
                else:
                    result = f"Unknown function: {name}"

                self.add_tool_result(tc.id, name, result)

        return self.api_pool


# ============================================================================
# Solver: maps source fields to API candidate pool
# ============================================================================

def solve_mapping(query, source_fields, api_pool, model):
    """Given source fields and an API candidate pool, produce field mappings.

    Uses the API pool to identify target products, then queries the full
    database for candidate fields within those products.
    """
    if not api_pool:
        return {}

    # Identify target products from the pool
    target_products = set()
    for api in api_pool:
        target_products.add(api["tool_name"])

    # Build comprehensive candidate list from target products
    candidates = []
    for category in TOOL_DATA:
        for tool_name in TOOL_DATA[category]:
            if tool_name not in target_products:
                continue
            if tool_name in API_DETAILS.get(category, {}):
                for api in API_DETAILS[category][tool_name].get("api_list", []):
                    desc = api.get("description", "")
                    candidates.append(f"{tool_name}_{api['name']}: {desc[:80]}")

    # Limit candidates for context
    if len(candidates) > 100:
        candidates = candidates[:100]

    pool_text = "\n".join(f"- {c}" for c in candidates)

    prompt = f"""You are a schema mapping expert. Map each source field to the best matching target field.

Query: {query}

Source fields:
{json.dumps(source_fields[:30], indent=2)}

Available target fields (format: product_TypeName.fieldName):
{pool_text}

For each source field, produce a JSON object with:
- "source": source field name (exact as shown above)
- "target": target field name (exact format: product_TypeName.fieldName as shown above)
- "reason": one sentence explanation

Return a JSON array. If no good match, set target to null.
Return ONLY the JSON array."""

    messages = [
        {"role": "system", "content": "You are a schema mapping expert. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ]

    response = call_llm(messages, model=model, max_tokens=4000)
    if response is None:
        return {}

    content = response.choices[0].message.content or ""

    # Parse JSON from response
    try:
        import re
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            mappings = json.loads(match.group())
            result = {}
            for m in mappings:
                if isinstance(m, dict) and m.get("target"):
                    src = m.get("source", "")
                    tgt = m.get("target", "")
                    result[src] = {
                        "target_property": tgt,
                        "justification": m.get("reason", ""),
                    }
            return result
    except:
        pass

    return {}


# ============================================================================
# Main AnyTool pipeline
# ============================================================================

def extract_products_from_query(query):
    """Extract source and target product names from a mapping query."""
    all_products = set()
    for cat_tools in TOOL_DATA.values():
        all_products.update(cat_tools.keys())

    found = []
    query_lower = query.lower()
    for prod in sorted(all_products, key=len, reverse=True):
        if prod.lower() in query_lower:
            found.append(prod)

    if len(found) >= 2:
        return found[0], found[1]
    elif len(found) == 1:
        return found[0], None
    return None, None


def run_anytool(query, source_fields, model="watsonx/meta-llama/llama-3-3-70b-instruct",
                max_reflection=1):
    """Run the full AnyTool pipeline on a schema mapping query.

    Args:
        query: mapping utterance (e.g., "Map concert Application to instana schema")
        source_fields: list of source field names to map
        model: LLM model ID
        max_reflection: max self-reflection rounds

    Returns:
        dict of {source_field: {target_property, justification}}
    """
    load_database()

    # Extract source/target products from query
    src_prod, tgt_prod = extract_products_from_query(query)

    for reflection_round in range(max_reflection + 1):
        # Phase 1: Determine categories to explore
        categories_to_explore = set()

        if src_prod:
            cat = find_product_category(src_prod)
            if cat:
                categories_to_explore.add(cat)
        if tgt_prod:
            cat = find_product_category(tgt_prod)
            if cat:
                categories_to_explore.add(cat)

        # If we only found one product, use meta-agent to find target categories
        if len(categories_to_explore) <= 1:
            meta = MetaAgent(query, model, list(TOOL_DATA.keys()))
            selected = meta.run()
            if selected:
                categories_to_explore.update(selected)

        # If still insufficient, add all categories
        if len(categories_to_explore) <= 1:
            categories_to_explore = set(TOOL_DATA.keys())

        # Phase 2: Tool agents explore target products (NOT source)
        api_pool = []
        src_category = find_product_category(src_prod) if src_prod else ""
        for category in categories_to_explore:
            if category not in TOOL_DATA:
                continue
            tools = list(TOOL_DATA[category].keys())
            # Exclude source product from target search
            if src_prod:
                tools = [t for t in tools if t != src_prod]
            if tgt_prod and tgt_prod in tools:
                tools = [tgt_prod]  # Focus on target product
            if not tools:
                continue

            # For small hierarchies (<=2 tools), skip agent and add all fields directly
            total_fields = sum(
                len(TOOL_DATA[category][t]["api_list_names"]) for t in tools
            )
            if total_fields <= 500:
                for t in tools:
                    for api_name in TOOL_DATA[category][t]["api_list_names"]:
                        api_pool.append({
                            "category_name": category,
                            "tool_name": t,
                            "api_name": api_name,
                        })
            else:
                agent = ToolAgent(query, model, category, tools, api_pool)
                agent.run()

        if not api_pool:
            # Reflection: broaden search
            logger.info(f"Reflection {reflection_round + 1}: empty pool, broadening")
            categories_to_explore = set(TOOL_DATA.keys()) - categories_to_explore
            for category in list(categories_to_explore)[:3]:
                tools = list(TOOL_DATA[category].keys())
                agent = ToolAgent(query, model, category, tools, api_pool)
                agent.run()

        if not api_pool:
            continue

        # Phase 3: Solver maps source fields to API pool
        mappings = solve_mapping(query, source_fields, api_pool, model)

        if mappings:
            return mappings

        logger.info(f"Reflection {reflection_round + 1}: solver produced no mappings")

    return {}


# ============================================================================
# Batch runner for evaluation
# ============================================================================

def run_anytool_on_dataset(dataset_path, output_path, model="watsonx/meta-llama/llama-3-3-70b-instruct",
                           max_samples=None):
    """Run AnyTool on an evaluation dataset and save results."""
    load_database()

    with open(dataset_path) as f:
        samples = [json.loads(l) for l in f if l.strip()]

    if max_samples:
        samples = samples[:max_samples]

    print(f"Running AnyTool on {len(samples)} samples with {model}")
    results = []

    for i, sample in enumerate(samples):
        sid = sample.get("id", f"sample_{i}")
        utterance = sample.get("utterance", sample.get("mapping_utterance", ""))
        gt = sample.get("mapping", sample.get("ground_truth", {}))
        source_fields = list(gt.keys()) if gt else []

        if not utterance:
            # Build utterance from metadata
            meta = sample.get("metadata", {})
            src_schema = sample.get("source_schema", "")
            tgt_schema = sample.get("target_schema", "")
            src_type = meta.get("source_type", "")
            utterance = f"Map {src_schema} {src_type} to {tgt_schema} schema"

        print(f"  [{i+1}/{len(samples)}] {sid}: {utterance[:60]}...", end=" ", flush=True)

        try:
            mappings = run_anytool(utterance, source_fields, model=model, max_reflection=1)
        except Exception as e:
            logger.error(f"Error on {sid}: {e}")
            mappings = {}

        print(f"→ {len(mappings)} mappings")

        result = {
            "sample_id": sid,
            "utterance": utterance,
            "predicted_mapping": mappings,
            "ground_truth_mapping": gt,
        }
        results.append(result)

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {output_path}")

    # Quick accuracy
    correct = total = 0
    for r in results:
        pm = r["predicted_mapping"]
        gt = r["ground_truth_mapping"]
        for src, pred_info in pm.items():
            tgt = pred_info.get("target_property", "") if isinstance(pred_info, dict) else str(pred_info)
            if tgt:
                total += 1
                gt_t = gt.get(src, "")
                if gt_t and tgt.lower() == gt_t.lower():
                    correct += 1
    acc = correct / total * 100 if total else 0
    print(f"Accuracy: {correct}/{total} ({acc:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset JSONL file")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--model", type=str, default="watsonx/meta-llama/llama-3-3-70b-instruct")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--test", action="store_true", help="Run single test query")
    parser.add_argument("--hierarchy", type=str, default=None,
                        help="Hierarchy prefix (e.g., 'omop_cms' loads tool_data_omop_cms.json)")
    args = parser.parse_args()

    load_database(hierarchy_prefix=args.hierarchy)
    print(f"Database: {len(TOOL_DATA)} categories, "
          f"{sum(len(v) for v in TOOL_DATA.values())} tools, "
          f"{sum(len(TOOL_DATA[c][t]['api_list_names']) for c in TOOL_DATA for t in TOOL_DATA[c])} APIs")

    if args.test:
        query = "Map concert Application to instana schema"
        source_fields = ["concert_Application.id", "concert_Application.name",
                         "concert_Application.status", "concert_Application.tags"]
        print(f"\nQuery: {query}")
        result = run_anytool(query, source_fields, model=args.model)
        print(f"Mappings: {json.dumps(result, indent=2)}")
    elif args.dataset:
        output = args.output or "anytool/results/anytool_faithful_results.json"
        run_anytool_on_dataset(args.dataset, output, model=args.model,
                               max_samples=args.max_samples)
    else:
        print("Use --test for single query or --dataset <file> for batch evaluation")
