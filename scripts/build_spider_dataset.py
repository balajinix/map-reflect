#!/usr/bin/env python3
"""
Build Spider schema mapping dataset from Spider 1.0 tables.json and
Gunjan's JOIN-parsed ground truth.

Creates:
  data/spider_api_lookup.json    - Field-level property lookup
  data/spider_dataset.jsonl      - Evaluation utterances (NL2GQL format)

Also creates a ChromaDB collection for schema retrieval.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Source files
TABLES_JSON = Path("/Users/balaji/code/ai-c-7149/sql-to-graphql/Queries/spider/tables.json")
GROUND_TRUTH = Path("/Users/balaji/code/gunjan-schema-mapping/spider_workspace/output_ground_truth.jsonl")


def load_spider_schemas(tables_file: Path):
    """Load Spider tables.json and build schema index."""
    with open(tables_file) as f:
        databases = json.load(f)

    # Index: db_id -> {tables, columns, foreign_keys}
    schema_index = {}
    for db in databases:
        db_id = db["db_id"]
        tables = db["table_names_original"]
        columns = db["column_names_original"]  # [table_idx, column_name]
        col_types = db.get("column_types", [])
        foreign_keys = db["foreign_keys"]  # [[col_idx, col_idx], ...]
        primary_keys = db["primary_keys"]  # [col_idx, ...]

        schema_index[db_id] = {
            "tables": tables,
            "columns": columns,
            "column_types": col_types,
            "foreign_keys": foreign_keys,
            "primary_keys": primary_keys,
        }

    return schema_index


def load_ground_truth(gt_file: Path):
    """Load Gunjan's parsed ground truth."""
    samples = []
    with open(gt_file) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def build_api_lookup(schema_index, relevant_dbs):
    """Build field-level API lookup from Spider schemas.

    Only includes databases that appear in the ground truth.
    Key format: <table>.<column> (e.g., concert.stadium_id)
    """
    lookup = {}

    for db_id in relevant_dbs:
        if db_id not in schema_index:
            continue
        schema = schema_index[db_id]
        tables = schema["tables"]
        columns = schema["columns"]
        col_types = schema["column_types"]

        for col_idx, (table_idx, col_name) in enumerate(columns):
            if table_idx < 0:  # Skip the * column
                continue
            table_name = tables[table_idx].lower()
            col_name_lower = col_name.lower()

            # Key: table_column (matching ground truth format)
            field_key = f"{table_name}_{col_name_lower}"

            # Determine if this is a primary key or foreign key
            is_pk = col_idx in schema["primary_keys"]
            is_fk = any(col_idx in fk for fk in schema["foreign_keys"])

            col_type = col_types[col_idx] if col_idx < len(col_types) else "text"

            description = f"{col_name} column in {table_name} table"
            if is_pk:
                description += " (primary key)"
            if is_fk:
                description += " (foreign key)"

            lookup[field_key] = {
                "product_name": db_id,
                "type_name": table_name,
                "field_name": col_name_lower,
                "description": description,
                "type": col_type,
                "database": db_id,
            }

    return lookup


def build_dataset(ground_truth_samples, schema_index):
    """Build deduplicated dataset in NL2GQL format.

    Deduplicates by (source_table, target_table) pair, keeping
    the sample with the most natural question. Aggregates all
    field mappings for each pair.
    """
    # Group by table pair, aggregate field mappings
    pair_data = defaultdict(lambda: {
        "questions": [],
        "queries": [],
        "field_mappings": {},  # src_field -> [{mapTable, mapProperty}]
        "db_id": None,
    })

    for s in ground_truth_samples:
        src_table = s["result"]["name"]
        properties = s["result"]["properties"]

        for src_field, mappings in properties.items():
            for m in mappings:
                tgt_table = m["mapTable"]
                pair_key = (src_table, tgt_table)
                pair_data[pair_key]["questions"].append(s["question"])
                pair_data[pair_key]["queries"].append(s["query"])

                if src_field not in pair_data[pair_key]["field_mappings"]:
                    pair_data[pair_key]["field_mappings"][src_field] = []
                # Avoid duplicate mappings
                existing = [
                    em for em in pair_data[pair_key]["field_mappings"][src_field]
                    if em["mapProperty"] == m["mapProperty"]
                ]
                if not existing:
                    pair_data[pair_key]["field_mappings"][src_field].append(m)

    # Find db_id for each table pair from schema_index
    table_to_db = {}
    for db_id, schema in schema_index.items():
        for table in schema["tables"]:
            table_to_db[table.lower()] = db_id

    # Build samples
    samples = []
    for (src_table, tgt_table), data in sorted(pair_data.items()):
        # Pick the longest/most descriptive question
        questions = list(set(data["questions"]))
        best_question = max(questions, key=len) if questions else ""

        db_id = table_to_db.get(src_table, table_to_db.get(tgt_table, "unknown"))

        # Build result in NL2GQL format
        result = {
            "name": src_table,
            "properties": data["field_mappings"],
        }

        sample = {
            "id": f"spider_{len(samples):04d}",
            "question": best_question,
            "mapping_utterance": f"map {src_table} data to {tgt_table} data",
            "result": result,
            "database": db_id,
        }
        samples.append(sample)

    return samples


def create_chromadb_collection(api_lookup, collection_name="spider_schemas"):
    """Create ChromaDB collection from API lookup for schema retrieval."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ChromaDB or sentence-transformers not available, skipping collection creation")
        return

    chroma_dir = DATA_DIR / "chroma_db_spider"
    chroma_dir.mkdir(exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=collection_name)

    # Build documents
    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = []
    ids = []
    metadatas = []

    for field_key, meta in api_lookup.items():
        doc = f"{meta['type_name']}.{meta['field_name']}: {meta['description']}"
        docs.append(doc)
        ids.append(field_key)
        metadatas.append({
            "product_name": meta["product_name"],
            "type_name": meta["type_name"],
            "field_name": meta["field_name"],
        })

    print(f"Encoding {len(docs)} documents...")
    embeddings = model.encode(docs, show_progress_bar=True)

    # Add in batches (ChromaDB has batch size limits)
    batch_size = 5000
    for i in range(0, len(docs), batch_size):
        end = min(i + batch_size, len(docs))
        collection.add(
            ids=ids[i:end],
            documents=docs[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
        )
        print(f"  Indexed batch {i//batch_size + 1}: {end}/{len(docs)}")

    print(f"ChromaDB collection '{collection_name}' created at {chroma_dir}")
    print(f"  {collection.count()} documents indexed")


def main():
    print("Loading Spider schemas...")
    schema_index = load_spider_schemas(TABLES_JSON)
    print(f"  {len(schema_index)} databases")

    print("Loading ground truth...")
    gt_samples = load_ground_truth(GROUND_TRUTH)
    print(f"  {len(gt_samples)} raw samples")

    print("Building dataset...")
    dataset = build_dataset(gt_samples, schema_index)
    print(f"  {len(dataset)} unique table-pair samples")

    # Find relevant databases
    relevant_dbs = set()
    for s in dataset:
        relevant_dbs.add(s.get("database", "unknown"))
    print(f"  {len(relevant_dbs)} databases")

    print("Building API lookup...")
    api_lookup = build_api_lookup(schema_index, relevant_dbs)
    print(f"  {len(api_lookup)} fields")

    # Save dataset
    out_dataset = DATA_DIR / "spider_dataset.jsonl"
    with open(out_dataset, "w") as f:
        for s in dataset:
            f.write(json.dumps(s) + "\n")
    print(f"Saved: {out_dataset}")

    # Save API lookup
    out_lookup = DATA_DIR / "spider_api_lookup.json"
    with open(out_lookup, "w") as f:
        json.dump(api_lookup, f, indent=2)
    print(f"Saved: {out_lookup}")

    # Create ChromaDB collection
    print("\nCreating ChromaDB collection...")
    create_chromadb_collection(api_lookup)

    # Summary
    print(f"\n{'='*60}")
    print(f"Spider Schema Mapping Dataset")
    print(f"{'='*60}")
    print(f"  Samples:    {len(dataset)}")
    print(f"  Fields:     {len(api_lookup)}")
    print(f"  Databases:  {len(relevant_dbs)}")
    total_mappings = sum(
        len(props)
        for s in dataset
        for props in s["result"]["properties"].values()
    )
    print(f"  Field mappings: {total_mappings}")


if __name__ == "__main__":
    main()
