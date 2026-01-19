#!/usr/bin/env python3
"""
OMOP Schema Indexer
Indexes OMOP/MIMIC/CMS/Synthea schemas from API lookup JSON files into ChromaDB.
Creates separate collections for each dataset.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    logger.error("ChromaDB not found. Install with: pip install chromadb")
    exit(1)


class OMOPSchemaIndexer:
    """Index OMOP-style schemas into ChromaDB for schema discovery."""

    def __init__(self, chroma_db_path: str, collection_name: str):
        self.chroma_db_path = Path(chroma_db_path)
        self.collection_name = collection_name

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Delete existing collection and create fresh
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"description": f"Schemas for {collection_name}"}
        )
        logger.info(f"Created new collection: {collection_name}")

    def index_from_api_lookup(self, api_lookup_file: str, hierarchy_file: str = None) -> int:
        """
        Index schemas from API lookup JSON file.

        Args:
            api_lookup_file: Path to API lookup JSON (e.g., omop_mimic_api_lookup.json)
            hierarchy_file: Optional path to hierarchy JSON for additional context

        Returns:
            Number of schemas indexed
        """
        # Load API lookup
        with open(api_lookup_file) as f:
            api_lookup = json.load(f)

        logger.info(f"Loaded {len(api_lookup)} fields from {api_lookup_file}")

        # Load hierarchy if provided
        hierarchy = {}
        if hierarchy_file and Path(hierarchy_file).exists():
            with open(hierarchy_file) as f:
                hierarchy = json.load(f)
            logger.info(f"Loaded hierarchy from {hierarchy_file}")

        # Group fields by schema (product_type)
        schemas = defaultdict(lambda: {"fields": [], "descriptions": []})

        for field_key, field_meta in api_lookup.items():
            # field_key format: "TypeName.fieldName"
            # field_meta contains: product_name, type_name, field_name, description

            product = field_meta.get("product_name", "").lower()
            type_name = field_meta.get("type_name", "")
            field_name = field_meta.get("field_name", "")
            description = field_meta.get("description", "")

            if not product or not type_name:
                continue

            schema_id = f"{product}_{type_name}"
            schemas[schema_id]["product"] = product
            schemas[schema_id]["type_name"] = type_name
            schemas[schema_id]["fields"].append(field_name)
            if description:
                schemas[schema_id]["descriptions"].append(description)

        # Index each schema
        indexed_count = 0
        for schema_id, schema_data in schemas.items():
            try:
                # Create searchable content
                product = schema_data["product"]
                type_name = schema_data["type_name"]
                fields = schema_data["fields"]
                descriptions = schema_data["descriptions"]

                # Make type name readable (convert snake_case to spaces)
                readable_type = type_name.replace("_", " ").lower()

                # Create searchable text
                searchable_parts = [
                    f"Schema: {product} {readable_type}",
                    f"Product: {product}",
                    f"Type: {type_name}",
                    f"Fields: {', '.join(fields[:30])}",  # First 30 fields
                ]

                # Add descriptions if available
                if descriptions:
                    desc_text = " ".join(descriptions[:5])[:500]  # First 5 descriptions, max 500 chars
                    searchable_parts.append(f"Description: {desc_text}")

                # Add domain keywords based on type name
                domain_keywords = self._extract_domain_keywords(type_name, product)
                if domain_keywords:
                    searchable_parts.append(f"Domain: {', '.join(domain_keywords)}")

                searchable_content = "\n".join(searchable_parts)

                # Metadata
                metadata = {
                    "schema_id": schema_id,
                    "product": product,
                    "type_name": type_name,
                    "num_fields": len(fields),
                    "indexed_at": datetime.now().isoformat()
                }

                # Add to ChromaDB
                self.collection.add(
                    documents=[searchable_content],
                    metadatas=[metadata],
                    ids=[schema_id]
                )

                indexed_count += 1

            except Exception as e:
                logger.error(f"Error indexing {schema_id}: {e}")
                continue

        logger.info(f"Indexed {indexed_count} schemas into collection '{self.collection_name}'")
        return indexed_count

    def _extract_domain_keywords(self, type_name: str, product: str) -> List[str]:
        """Extract domain keywords based on type name and product."""
        keywords = set()
        type_lower = type_name.lower()

        # Medical/healthcare domain keywords
        medical_mappings = {
            "person": ["patient", "individual", "demographics"],
            "patient": ["person", "individual", "demographics"],
            "visit": ["encounter", "admission", "hospitalization"],
            "admission": ["visit", "encounter", "hospital"],
            "condition": ["diagnosis", "disease", "health"],
            "procedure": ["treatment", "operation", "surgery"],
            "drug": ["medication", "prescription", "pharmacy"],
            "observation": ["measurement", "result", "finding"],
            "measurement": ["observation", "lab", "vital"],
            "specimen": ["sample", "lab", "test"],
            "note": ["documentation", "clinical", "text"],
            "provider": ["doctor", "physician", "clinician"],
            "care_site": ["facility", "hospital", "location"],
        }

        for key, values in medical_mappings.items():
            if key in type_lower:
                keywords.update(values)
                keywords.add(key)

        # Add product as keyword
        keywords.add(product)

        return list(keywords)

    def search_schemas(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for schemas matching the query."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )

            matches = []
            for i in range(len(results.get("ids", [[]])[0])):
                match = {
                    "schema_id": results["ids"][0][i],
                    "similarity_score": 1 - results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "content": results["documents"][0][i]
                }
                matches.append(match)

            return matches

        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "chroma_db_path": str(self.chroma_db_path)
            }
        except Exception as e:
            return {"error": str(e)}


def index_omop_dataset(dataset_name: str, data_dir: str, chroma_db_path: str):
    """Index a specific OMOP dataset."""
    data_path = Path(data_dir)

    # Determine file paths based on dataset name
    dataset_lower = dataset_name.lower().replace("omop_", "omop_")
    api_lookup_file = data_path / f"{dataset_lower}_api_lookup.json"
    hierarchy_file = data_path / f"{dataset_lower}_hierarchy.json"

    if not api_lookup_file.exists():
        logger.error(f"API lookup file not found: {api_lookup_file}")
        return None

    # Create collection name
    collection_name = f"{dataset_lower}_schemas"

    logger.info(f"\n{'='*60}")
    logger.info(f"Indexing {dataset_name}")
    logger.info(f"{'='*60}")
    logger.info(f"API lookup: {api_lookup_file}")
    logger.info(f"Hierarchy: {hierarchy_file}")
    logger.info(f"Collection: {collection_name}")

    # Create indexer and index schemas
    indexer = OMOPSchemaIndexer(chroma_db_path, collection_name)
    count = indexer.index_from_api_lookup(
        str(api_lookup_file),
        str(hierarchy_file) if hierarchy_file.exists() else None
    )

    return indexer, count


def main():
    parser = argparse.ArgumentParser(description="Index OMOP schemas into ChromaDB")
    parser.add_argument("--dataset", default="all",
                       help="Dataset to index: omop_mimic, omop_cms, omop_synthea, itops, or all")
    parser.add_argument("--data-dir", default="../data",
                       help="Data directory containing API lookup files")
    parser.add_argument("--chroma-db-path", default="../data/chroma_db",
                       help="Path to ChromaDB database")
    parser.add_argument("--test-query", help="Test query after indexing")

    args = parser.parse_args()

    # Define datasets to index
    if args.dataset.lower() == "all":
        datasets = ["omop_mimic", "omop_cms", "omop_synthea", "itops"]
    else:
        datasets = [args.dataset.lower()]

    # Index each dataset
    results = {}
    for dataset in datasets:
        result = index_omop_dataset(dataset, args.data_dir, args.chroma_db_path)
        if result:
            indexer, count = result
            results[dataset] = {
                "count": count,
                "stats": indexer.get_collection_stats()
            }

            # Test search if query provided
            if args.test_query:
                print(f"\nTesting search for '{args.test_query}':")
                matches = indexer.search_schemas(args.test_query, n_results=3)
                for i, match in enumerate(matches, 1):
                    print(f"  {i}. {match['schema_id']} (score: {match['similarity_score']:.3f})")

    # Print summary
    print(f"\n{'='*60}")
    print("Indexing Summary")
    print(f"{'='*60}")
    for dataset, result in results.items():
        print(f"  {dataset}: {result['count']} schemas indexed")


if __name__ == "__main__":
    main()
