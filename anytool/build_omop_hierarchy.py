#!/usr/bin/env python3
"""Build AnyTool hierarchy for OMOP datasets."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

def build_omop_hierarchy(lookup_path):
    with open(lookup_path) as f:
        api_lookup = json.load(f)

    tool_data = {}
    api_details = {}

    for field_key, meta in api_lookup.items():
        product = meta.get("product_name", "")
        type_name = meta.get("type_name", "")
        field_name = meta.get("field_name", "")
        description = meta.get("description", "")

        if not product or not type_name:
            continue

        category = product
        if category not in tool_data:
            tool_data[category] = {}
            api_details[category] = {}

        if product not in tool_data[category]:
            tool_data[category][product] = {"api_list_names": [], "tool_description": product + " schema"}
            api_details[category][product] = {"tool_name": product, "tool_description": product + " schema", "api_list": []}

        api_name = type_name + "." + field_name
        tool_data[category][product]["api_list_names"].append(api_name)
        api_details[category][product]["api_list"].append({
            "name": api_name,
            "description": description[:100] if description else field_name + " in " + type_name,
        })

    return tool_data, api_details


datasets = [
    ("omop_cms", "data/omop_cms_api_lookup.json"),
    ("omop_mimic", "data/omop_mimic_api_lookup.json"),
    ("omop_synthea", "data/omop_synthea_api_lookup.json"),
]

for ds, lookup_file in datasets:
    path = BASE / lookup_file
    if not path.exists():
        print(ds + ": lookup not found")
        continue

    td, ad = build_omop_hierarchy(path)

    out_dir = Path(__file__).parent
    out_td = out_dir / ("tool_data_" + ds + ".json")
    out_ad = out_dir / ("api_details_" + ds + ".json")
    out_ctd = out_dir / ("category_tool_details_" + ds + ".json")

    with open(out_td, "w") as f:
        json.dump(td, f, indent=2)
    with open(out_ad, "w") as f:
        json.dump(ad, f, indent=2)
    with open(out_ctd, "w") as f:
        ctd = {}
        for cat in td:
            ctd[cat] = {"description": cat + " schema", "tools": {t: t + " product" for t in td[cat]}}
        json.dump(ctd, f, indent=2)

    total = sum(len(td[c][t]["api_list_names"]) for c in td for t in td[c])
    print(ds + ": " + str(len(td)) + " categories, " + str(total) + " fields")
    for cat in td:
        for tool in td[cat]:
            n = len(td[cat][tool]["api_list_names"])
            print("  " + cat + "/" + tool + ": " + str(n) + " fields")
