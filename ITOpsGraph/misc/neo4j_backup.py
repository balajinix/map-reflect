#!/usr/bin/env python3
"""
Neo4j Backup/Export Script
Exports all nodes and relationships to JSON files.
"""

import os
import json
import argparse
from datetime import datetime
from neo4j import GraphDatabase


def export_database(uri: str, user: str, password: str, output_dir: str):
    """Export the entire Neo4j database to JSON files."""
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        with driver.session() as session:
            # Get database stats
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            print(f"Database contains:")
            print(f"  - {node_count} nodes")
            print(f"  - {rel_count} relationships")
            
            # Get all labels
            labels = [r[0] for r in session.run("CALL db.labels()").values()]
            print(f"  - Labels: {labels}")
            
            # Get all relationship types
            rel_types = [r[0] for r in session.run("CALL db.relationshipTypes()").values()]
            print(f"  - Relationship types: {rel_types}")
            
            # Export nodes by label
            print("\nExporting nodes...")
            all_nodes = []
            for label in labels:
                result = session.run(f"MATCH (n:`{label}`) RETURN n, labels(n) as labels, elementId(n) as id")
                nodes = []
                for record in result:
                    node = dict(record["n"])
                    node["_labels"] = record["labels"]
                    node["_id"] = record["id"]
                    nodes.append(node)
                all_nodes.extend(nodes)
                print(f"  - {label}: {len(nodes)} nodes")
            
            # Export relationships
            print("\nExporting relationships...")
            all_rels = []
            for rel_type in rel_types:
                result = session.run(f"""
                    MATCH (a)-[r:`{rel_type}`]->(b) 
                    RETURN type(r) as type, properties(r) as props, 
                           elementId(a) as from_id, elementId(b) as to_id,
                           labels(a) as from_labels, labels(b) as to_labels
                """)
                rels = []
                for record in result:
                    rel = {
                        "_type": record["type"],
                        "_from_id": record["from_id"],
                        "_to_id": record["to_id"],
                        "_from_labels": record["from_labels"],
                        "_to_labels": record["to_labels"],
                        "properties": record["props"]
                    }
                    rels.append(rel)
                all_rels.extend(rels)
                print(f"  - {rel_type}: {len(rels)} relationships")
            
            # Save to files
            nodes_file = os.path.join(output_dir, f"neo4j_nodes_{timestamp}.json")
            rels_file = os.path.join(output_dir, f"neo4j_relationships_{timestamp}.json")
            schema_file = os.path.join(output_dir, f"neo4j_schema_{timestamp}.json")
            
            with open(nodes_file, "w") as f:
                json.dump(all_nodes, f, indent=2, default=str)
            print(f"\nSaved nodes to: {nodes_file}")
            
            with open(rels_file, "w") as f:
                json.dump(all_rels, f, indent=2, default=str)
            print(f"Saved relationships to: {rels_file}")
            
            # Save schema info
            schema = {
                "exported_at": timestamp,
                "node_count": node_count,
                "relationship_count": rel_count,
                "labels": labels,
                "relationship_types": rel_types
            }
            with open(schema_file, "w") as f:
                json.dump(schema, f, indent=2)
            print(f"Saved schema to: {schema_file}")
            
            print(f"\nBackup complete!")
            
    finally:
        driver.close()


def main():
    parser = argparse.ArgumentParser(description="Export Neo4j database to JSON")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default=None, help="Neo4j password (or set NEO4J_PASSWORD env var)")
    parser.add_argument("--output", "-o", default="./neo4j_backup", help="Output directory")
    
    args = parser.parse_args()
    
    password = args.password or os.environ.get("NEO4J_PASSWORD")
    if not password:
        password = input("Enter Neo4j password: ")
    
    export_database(args.uri, args.user, password, args.output)


if __name__ == "__main__":
    main()
