from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from pymilvus import list_collections
import numpy as np
import json

class MilvusHelper:
    def __init__(self, model_name, collection_name, dim, host="127.0.0.1", port="19530"):
        try:
            self.model = SentenceTransformer(model_name)
            self.collection_name = collection_name
            self.dim = dim
            connections.connect("default", host=host, port=port)
        except Exception as e:
            print(f"Error initializing MilvusHelper: {e}")

    def create_collection(self):
        try:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            schema = CollectionSchema(fields, description=f"{self.collection_name} Collection")
            collection = Collection(name=self.collection_name, schema=schema)
            return collection
        except Exception as e:
            print(f"Error creating collection in Milvus: {e}")

    def insert_data(self, entities, embeddings):
        try:
            collection = self.create_collection()
            ids = [i for i in range(len(entities))]
            collection.insert([ids, embeddings.tolist()])
            collection.create_index(field_name="vector", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
        except Exception as e:
            print(f"Error inserting data into Milvus: {e}")

    def search(self, query, search_params, top_k=5):
        try:
            collection = Collection(name=self.collection_name)
            collection.load()
            query_embedding = self.model.encode([query])[0].tolist()
            results = collection.search([query_embedding], "vector", search_params, limit=top_k, expr=None)
            collection.release()
            return results
        except Exception as e:
            print(f"Error searching in Milvus: {e}")

    def disconnect(self):
        try:
            connections.disconnect("default")
        except Exception as e:
            print(f"Error disconnecting from Milvus: {e}")

    @staticmethod
    def save_data(file_path, data):
        try:
            np.save(file_path, data)
        except Exception as e:
            print(f"Error saving data to {file_path}: {e}")

    @staticmethod
    def load_data(file_path):
        try:
            return np.load(file_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")

    @staticmethod
    def save_json(file_path, data):
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving JSON to {file_path}: {e}")

    @staticmethod
    def load_json(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {file_path}: {e}")


if __name__ == "__main__":
    connections.connect("default", host="localhost", port="19530")
    collections = list_collections()
    for collection in collections:
        print(collection)
