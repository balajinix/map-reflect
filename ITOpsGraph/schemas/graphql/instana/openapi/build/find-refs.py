import json
import re

# Load OpenAPI JSON
with open("openapi.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract all schemas and references
schemas = data.get("components", {}).get("schemas", {})
references = re.findall(r'"[$]ref": *"#/components/schemas/(\w+)"', json.dumps(data))

# Check if any reference is missing
missing_refs = [ref for ref in references if ref not in schemas]

if missing_refs:
    print("🚨 Missing $ref schemas:", missing_refs)
else:
    print("✅ All $ref schemas are valid.")

