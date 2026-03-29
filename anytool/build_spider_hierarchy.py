#!/usr/bin/env python3
"""Build per-database AnyTool hierarchy for Spider.

Spider has 164 databases. Each sample maps within one database.
We build one hierarchy per database: database → tables → columns.
"""
import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent


def main():
    with open(BASE / "data/spider_api_lookup.json") as f:
        api_lookup = json.load(f)

    # Group by database
    db_fields = defaultdict(lambda: defaultdict(list))
    for field_key, meta in api_lookup.items():
        db = meta.get("database", meta.get("product_name", ""))
        table = meta.get("type_name", "")
        field = meta.get("field_name", "")
        desc = meta.get("description", "")
        if db and table and field:
            db_fields[db][table].append({
                "name": table + "." + field,
                "field_name": field,
                "description": desc or field + " in " + table,
            })

    # Build combined hierarchy (all databases as categories, tables as tools)
    tool_data = {}
    api_details = {}

    for db, tables in db_fields.items():
        tool_data[db] = {}
        api_details[db] = {}
        for table, fields in tables.items():
            tool_data[db][table] = {
                "api_list_names": [f["name"] for f in fields],
                "tool_description": table + " table in " + db,
            }
            api_details[db][table] = {
                "tool_name": table,
                "tool_description": table + " table in " + db,
                "api_list": [{"name": f["name"], "description": f["description"]} for f in fields],
            }

    out_dir = Path(__file__).parent
    with open(out_dir / "tool_data_spider.json", "w") as f:
        json.dump(tool_data, f, indent=2)
    with open(out_dir / "api_details_spider.json", "w") as f:
        json.dump(api_details, f, indent=2)
    with open(out_dir / "category_tool_details_spider.json", "w") as f:
        ctd = {}
        for db in tool_data:
            ctd[db] = {
                "description": db + " database",
                "tools": {t: t + " table" for t in tool_data[db]},
            }
        json.dump(ctd, f, indent=2)

    total_fields = sum(
        len(tool_data[db][t]["api_list_names"])
        for db in tool_data for t in tool_data[db]
    )
    print(f"Spider hierarchy: {len(tool_data)} databases, "
          f"{sum(len(v) for v in tool_data.values())} tables, "
          f"{total_fields} fields")


if __name__ == "__main__":
    main()
