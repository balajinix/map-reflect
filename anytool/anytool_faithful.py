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


def load_database():
    """Load the 3-tier hierarchy."""
    global TOOL_DATA, API_DETAILS, CATEGORY_TOOL_DETAILS
    if TOOL_DATA is None:
        with open(BASE / "tool_data.json") as f:
            TOOL_DATA = json.load(f)
        with open(BASE / "api_details.json") as f:
            API_DETAILS = json.load(f)
        with open(BASE / "category_tool_details.json") as f:
            CATEGORY_TOOL_DETAILS = json.load(f)
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


def get_apis_in_tool(category_name: str, tool_name: str) -> list:
    """Get all API names in a specific tool."""
    if category_name not in TOOL_DATA:
        return f"Category '{category_name}' not found"
    if tool_name not in TOOL_DATA[category_name]:
        return f"Tool '{tool_name}' not found in '{category_name}'"
    apis = TOOL_DATA[category_name][tool_name]["api_list_names"]
    # Return first 50 to avoid context overflow
    if len(apis) > 50:
        return apis[:50] + [f"... and {len(apis) - 50} more"]
    return apis


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
             max_tokens=2048, temperature=0.0):
    """Call LLM with optional function calling."""
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
        logger.error(f"LLM call failed: {e}")
        return None


# ============================================================================
# Agent classes
# ============================================================================

class AnyToolAgent:
    """Base agent with function-calling loop."""

    def __init__(self, model, system_prompt, tools_schema, max_steps=15):
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
            f"the relevant API categories to search. The database has these categories: {categories}. "
            f"Use get_tools_in_category to explore what each category contains. "
            f"Use select_category to choose categories that might contain relevant APIs. "
            f"Explore broadly — the mapping target may be in an unexpected category. "
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
            f"Available tools: {tools}. "
            f"Use get_apis_in_tool to see what fields each tool has. "
            f"Use get_api_details for more info on specific fields. "
            f"Use add_api_to_pool to add relevant fields to the candidate pool for mapping. "
            f"Focus on fields that could map to the source fields in the query. "
            f"Call finish_search when done."
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
    """Given source fields and an API candidate pool, produce field mappings."""
    if not api_pool:
        return {}

    # Build candidate description
    pool_desc = []
    for api in api_pool:
        details = get_api_details(api["category_name"], api["tool_name"], api["api_name"])
        if isinstance(details, dict):
            pool_desc.append(f"- {api['tool_name']}.{api['api_name']}: {details.get('description', '')}")
        else:
            pool_desc.append(f"- {api['tool_name']}.{api['api_name']}")

    pool_text = "\n".join(pool_desc[:50])  # Limit for context

    prompt = f"""You are a schema mapping expert. Map each source field to the best matching target field from the candidate pool.

Query: {query}

Source fields to map:
{json.dumps(source_fields[:30], indent=2)}

Candidate target fields:
{pool_text}

For each source field, produce a JSON object with:
- "source": source field name
- "target": best matching target field (format: "tool_name.TypeName.fieldName")
- "reason": one sentence explanation

Return a JSON array. If no good match exists for a field, set target to null.
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
        # Try to extract JSON array
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

def run_anytool(query, source_fields, model="watsonx/meta-llama/llama-3-3-70b-instruct",
                max_reflection=2):
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
    categories = list(TOOL_DATA.keys())

    for reflection_round in range(max_reflection + 1):
        # Phase 1: Meta-agent selects categories
        meta = MetaAgent(query, model, categories)
        selected = meta.run()

        if not selected:
            # Fallback: use all categories
            selected = categories[:5]

        # Phase 2: Tool agents explore each category
        api_pool = []
        for category in selected:
            if category not in TOOL_DATA:
                continue
            tools = list(TOOL_DATA[category].keys())
            agent = ToolAgent(query, model, category, tools, api_pool)
            agent.run()

        if not api_pool:
            continue

        # Phase 3: Solver maps source fields to API pool
        mappings = solve_mapping(query, source_fields, api_pool, model)

        if mappings:
            return mappings

        # Self-reflection: if solver failed, try again with different categories
        logger.info(f"Reflection round {reflection_round + 1}: solver produced no mappings")

    return {}


if __name__ == "__main__":
    load_database()
    print(f"Database: {len(TOOL_DATA)} categories, "
          f"{sum(len(v) for v in TOOL_DATA.values())} tools, "
          f"{sum(len(TOOL_DATA[c][t]['api_list_names']) for c in TOOL_DATA for t in TOOL_DATA[c])} APIs")

    # Test with a sample query
    query = "Map concert Application to instana schema"
    source_fields = ["concert_Application.id", "concert_Application.name",
                     "concert_Application.status", "concert_Application.tags"]

    print(f"\nQuery: {query}")
    print(f"Source fields: {source_fields}")

    result = run_anytool(query, source_fields)
    print(f"\nMappings: {json.dumps(result, indent=2)}")
