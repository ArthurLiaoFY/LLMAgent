# %%
import json

from qdrant_client import QdrantClient, models
from qdrant_client.http import models
from qdrant_client.http.models import CollectionStatus
from qdrant_client.models import Distance, PointStruct, VectorParams
from qdrant_client.qdrant_fastembed import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_SPARSE_EMBEDDING_MODELS,
)

# %%
# %%
with open("../../secrets.json") as f:
    secrets = json.loads(f.read())
# %%
client = QdrantClient(**secrets.get("qdrant").get("local"))
client
# %%
client.set_model("sentence-transformers/all-MiniLM-L6-v2")
client.set_sparse_model("prithivida/Splade_PP_en_v1")

# %%
collection_name = "startups"
if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=client.get_fastembed_vector_params(),
        # comment this line to use dense vectors only
        sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
    )
# %%
with open("../startups_demo.json") as f:
    startups_demo = json.loads(f.read())

metadata = []
documents = []

for line in startups_demo.values():
    documents.append(line.get("description"))
    metadata.append(line)
# %%
# client.add(
#     collection_name="startups",
#     documents=documents,
#     metadata=metadata,
#     parallel=None,  # Use all available CPU cores to encode data.
#     # Requires wrapping code into if __name__ == '__main__' block
# )
# %%


class HybridSearcher:
    DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SPARSE_MODEL = "prithivida/Splade_PP_en_v1"

    def __init__(self, collection_name):
        self.collection_name = collection_name
        # initialize Qdrant client
        self.qdrant_client = QdrantClient(**secrets.get("qdrant").get("local"))
        self.qdrant_client.set_model(self.DENSE_MODEL)
        # comment this line to use dense vectors only
        self.qdrant_client.set_sparse_model(self.SPARSE_MODEL)

    def search(self, text: str):
        search_result = self.qdrant_client.query(
            collection_name=self.collection_name,
            query_text=text,
            query_filter=None,  # If you don't want any filters for now
            limit=5,  # 5 the closest results
        )
        # `search_result` contains found vector ids with similarity scores
        # along with the stored payload

        # Select and return metadata
        metadata = [hit.metadata for hit in search_result]
        return metadata

    def main(self, text: str):

        city_of_interest = "Berlin"

        # Define a filter for cities
        city_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="city", match=models.MatchValue(value=city_of_interest)
                )
            ]
        )

        search_result = self.qdrant_client.query(
            collection_name=self.collection_name,
            query_text=text,
            query_filter=city_filter,
            limit=5,
        )

        return search_result


# %%
hs = HybridSearcher(collection_name=collection_name)
hs.search(text="health")
# %%
