# %%
import json

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm

# %%
with open("../secrets.json") as f:
    secrets = json.loads(f.read())
with open("../startups_demo.json") as f:
    startups_demo = json.loads(f.read())
# %%
client = QdrantClient(**secrets.get("qdrant").get("local"))
client
# %%
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="mps",
)  # or device="cpu" if you don't have a GPU
# %%
df = pd.DataFrame.from_dict(data=startups_demo, orient="index")
# %%
vectors = model.encode(
    [row.alt + ". " + row.description for row in df.itertuples()],
    show_progress_bar=True,
)
# %%
vectors.shape
# %%
collection_name = "startups"
if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vectors.shape[1],
            distance=Distance.COSINE,
        ),
    )
collection = client.get_collection(collection_name=collection_name)

# %%
client.upload_collection(
    collection_name=collection_name,
    vectors=vectors,
    payload=list(startups_demo.values()),
    ids=None,  # Vector ids will be assigned automatically
    batch_size=256,  # How many vectors will be uploaded in a single request?
)
