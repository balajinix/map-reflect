import json

# Load OpenAPI JSON
with open("openapi.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def find_invalid_refs(obj, path="root"):
    """ Recursively search for misplaced $ref values """
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}"
            if key == "$ref" and not isinstance(value, str):
                print(f"🚨 Invalid $ref at: {new_path} → {value}")
            elif key == "$ref" and path.split(".")[-1] not in ["schema", "$ref"]:
                print(f"⚠️ Possible misplaced $ref at: {new_path} → {value}")
            find_invalid_refs(value, new_path)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            find_invalid_refs(item, f"{path}[{index}]")

print("🔍 Scanning OpenAPI for misplaced $ref values...")
find_invalid_refs(data)
print("✅ Scan complete.")

