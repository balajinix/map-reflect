# utils/schema_indexer.py

import os
import glob
from typing import Dict

def build_schema_index(base_dir: str) -> Dict[str, str]:
    """
    Scans the given base_dir for subdirectories (instana, concert, turbo),
    reads all .graphql files in each subdirectory, and returns a dictionary
    mapping a 'key' to the file content, e.g.:

    {
      "Instana/instana_demo.graphql": "<content>",
      "Concert/music_schema.graphql": "<content>",
      ...
    }

    Then your agent can pick from among these entries based on user query.
    """
    schema_map = {}

    # For each subdir in e.g. [instana, concert, turbo, solis]
    subdirs = ["instana", "concert", "turbo", "solis"]
    for sub in subdirs:
        sub_path = os.path.join(base_dir, sub)
        if not os.path.isdir(sub_path):
            continue

        # gather all .graphql files in that sub
        graphql_files = glob.glob(os.path.join(sub_path, "*.graphql"))
        for fpath in graphql_files:
            # e.g. "instana_demo.graphql"
            fname = os.path.basename(fpath)
            # We'll store key = "Instana/instana_demo.graphql" or "Concert/music_schema.graphql"
            # Capitalize sub for the user-friendly name, or keep it as-lower
            sub_key = sub.capitalize()
            map_key = f"{sub_key}/{fname}"

            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            schema_map[map_key] = content

    return schema_map
