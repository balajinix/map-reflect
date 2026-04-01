#!/usr/bin/env python3
"""
Neo4j Cleanup Script
Removes all nodes, relationships, constraints, and indexes from the database.
"""

import os
import argparse
from neo4j import GraphDatabase


def cleanup_database(uri: str, user: str, password: str, confirm: bool = False):
    """Clean up the entire Neo4j database."""
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with driver.session() as session:
            # Get counts before cleanup
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            print(f"Current database state:")
            print(f"  - Nodes: {node_count}")
            print(f"  - Relationships: {rel_count}")
            
            # Get node labels
            labels = session.run("CALL db.labels()").values()
            if labels:
                print(f"  - Labels: {', '.join([l[0] for l in labels])}")
            
            # Get relationship types
            rel_types = session.run("CALL db.relationshipTypes()").values()
            if rel_types:
                print(f"  - Relationship types: {', '.join([r[0] for r in rel_types])}")
            
            if node_count == 0 and rel_count == 0:
                print("\nDatabase is already empty.")
                return
            
            if not confirm:
                response = input("\nAre you sure you want to delete everything? (yes/no): ")
                if response.lower() not in ['yes', 'y']:
                    print("Aborted.")
                    return
            
            print("\nCleaning up...")
            
            # Drop all constraints first (required before dropping indexes)
            print("  Dropping constraints...")
            constraints = session.run("SHOW CONSTRAINTS YIELD name").values()
            for constraint in constraints:
                constraint_name = constraint[0]
                try:
                    # Use backticks for names that might be numeric or have special chars
                    session.run(f"DROP CONSTRAINT `{constraint_name}`")
                    print(f"    Dropped constraint: {constraint_name}")
                except Exception as e:
                    print(f"    Warning: Could not drop constraint {constraint_name}: {e}")
            
            # Drop all indexes
            print("  Dropping indexes...")
            indexes = session.run("SHOW INDEXES YIELD name, type").values()
            for index in indexes:
                index_name = index[0]
                index_type = index[1]
                # Skip lookup indexes (they're system indexes)
                if 'lookup' in str(index_type).lower():
                    continue
                try:
                    session.run(f"DROP INDEX `{index_name}`")
                    print(f"    Dropped index: {index_name}")
                except Exception as e:
                    print(f"    Warning: Could not drop index {index_name}: {e}")
            
            # Delete all nodes and relationships in batches
            print("  Deleting nodes and relationships...")
            batch_size = 10000
            total_deleted = 0
            
            while True:
                result = session.run(f"""
                    MATCH (n)
                    WITH n LIMIT {batch_size}
                    DETACH DELETE n
                    RETURN count(*) as deleted
                """)
                deleted = result.single()["deleted"]
                total_deleted += deleted
                
                if deleted > 0:
                    print(f"    Deleted {total_deleted} nodes so far...")
                
                if deleted < batch_size:
                    break
            
            print(f"\nCleanup complete!")
            print(f"  - Deleted {total_deleted} nodes (and their relationships)")
            
            # Verify
            final_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            print(f"  - Remaining nodes: {final_count}")
            
    finally:
        driver.close()


def main():
    parser = argparse.ArgumentParser(description="Clean up Neo4j database")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default=None, help="Neo4j password (or set NEO4J_PASSWORD env var)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    password = args.password or os.environ.get("NEO4J_PASSWORD")
    if not password:
        password = input("Enter Neo4j password: ")
    
    cleanup_database(args.uri, args.user, password, confirm=args.yes)


if __name__ == "__main__":
    main()
