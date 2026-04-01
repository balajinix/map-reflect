#!/usr/bin/env python3
"""
Neo4j database inspection script.
"""

import os
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

if not PASSWORD:
    PASSWORD = input("Enter Neo4j password: ")

print(f"Connecting to {URI} as {USER}...")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
driver.verify_connectivity()
print("Connection successful!\n")

with driver.session() as s:
    # Node counts by label
    print("=== Node Counts ===")
    result = s.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count ORDER BY count DESC")
    for r in result:
        print(f"  {r['label']}: {r['count']}")
    
    # Relationship counts by type
    print("\n=== Relationship Counts ===")
    result = s.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC")
    for r in result:
        print(f"  {r['type']}: {r['count']}")
    
    # Products
    print("\n=== Products ===")
    result = s.run("MATCH (p:Product) RETURN p.name, p.description")
    for r in result:
        print(f"  {r['p.name']}: {r['p.description'] or '(no description)'}")
    
    # Types per product
    print("\n=== Types per Product ===")
    result = s.run("""
        MATCH (p:Product)-[:HAS_TYPE]->(t:Type)
        RETURN p.name as product, collect(t.name) as types
    """)
    for r in result:
        print(f"  {r['product']}: {', '.join(r['types'][:5])}{'...' if len(r['types']) > 5 else ''}")
    
    # Sample fields
    print("\n=== Sample Fields (first 10) ===")
    result = s.run("MATCH (f:Field) RETURN f.product, f.full_name LIMIT 10")
    for r in result:
        print(f"  {r['f.product']}.{r['f.full_name']}")
    
    # Field mappings
    print("\n=== Field Mappings (first 10) ===")
    result = s.run("""
        MATCH (sf:Field)-[m:FIELD_MAPPING]->(tf:Field)
        RETURN sf.product + '.' + sf.full_name as source, 
               tf.product + '.' + tf.full_name as target,
               m.score as score
        LIMIT 10
    """)
    rows = list(result)
    if rows:
        for r in rows:
            print(f"  {r['source']} -> {r['target']} (score: {r['score']})")
    else:
        print("  (none)")
    
    # Type mappings
    print("\n=== Type Mappings (first 10) ===")
    result = s.run("""
        MATCH (st:Type)-[m:TYPE_MAPPING]->(tt:Type)
        RETURN st.product + '.' + st.name as source,
               tt.product + '.' + tt.name as target
        LIMIT 10
    """)
    rows = list(result)
    if rows:
        for r in rows:
            print(f"  {r['source']} -> {r['target']}")
    else:
        print("  (none)")

driver.close()
print("\nDone.")
