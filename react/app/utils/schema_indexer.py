# utils/schema_indexer.py

import os
import glob
from typing import Dict

def build_schema_index(base_dir: str) -> Dict[str, str]:
    """
    Scans the given base_dir for .graphql files and returns a dictionary
    mapping filename to file content, e.g.:

    {
      "instana_demo.graphql": "<content>",
      "concert_schema.graphql": "<content>",
      ...
    }

    All .graphql files are expected to be directly in base_dir.
    """
    schema_map = {}

    # Look for .graphql files directly in base_dir
    if not os.path.isdir(base_dir):
        print(f"Warning: Schema directory does not exist: {base_dir}")
        return schema_map

    graphql_files = glob.glob(os.path.join(base_dir, "*.graphql"))
    print(f"Found {len(graphql_files)} .graphql files in {base_dir}")

    for fpath in graphql_files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            schema_map[fname] = content
            print(f"  Loaded: {fname}")
        except Exception as e:
            print(f"  Failed to load {fname}: {e}")

    print(f"Successfully indexed {len(schema_map)} schemas")
    return schema_map
