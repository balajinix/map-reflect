import json
from collections import defaultdict

# Load OpenAPI JSON
with open("openapi.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Dictionary to track normalized path templates
path_templates = defaultdict(list)

# Normalize paths (replace parameters with placeholders)
def normalize_path(path):
    return "/".join(["{var}" if part.startswith("{") and part.endswith("}") else part for part in path.split("/")])

# Check all paths in OpenAPI file
for path in data.get("paths", {}):
    normalized = normalize_path(path)
    path_templates[normalized].append(path)

# Find and print duplicate path templates
duplicates = {k: v for k, v in path_templates.items() if len(v) > 1}

if duplicates:
    print("🚨 Identical Path Templates Detected:")
    for normalized, variants in duplicates.items():
        print(f"- {normalized}: {variants}")
else:
    print("✅ No conflicting path templates found.")

