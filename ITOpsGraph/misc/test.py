#!/usr/bin/env python3
"""
Simple Neo4j connection test.
"""

import os
from neo4j import GraphDatabase

# Connection settings
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

if not PASSWORD:
    PASSWORD = input("Enter Neo4j password: ")

print(f"Connecting to {URI} as {USER}...")

try:
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("Connection successful!")
    
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
        print(f"Node count: {count}")
    
    driver.close()
    
except Exception as e:
    print(f"Connection failed: {e}")
