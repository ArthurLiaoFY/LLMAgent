# %%
import json
import re
import uuid
from typing import Any, Dict

import inflection
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

with open("../../secrets.json") as f:
    secrets = json.loads(f.read())
# %%
structures = []
with open("../../code_structures.jsonl", "r") as fp:
    for i, row in enumerate(fp):
        entry = json.loads(row)
        structures.append(entry)


# %%


def textify(chunk: Dict[str, Any]) -> str:
    # Get rid of all the camel case / snake case
    # - inflection.underscore changes the camel case to snake case
    # - inflection.humanize converts the snake case to human readable form
    name = inflection.humanize(inflection.underscore(chunk["name"]))
    signature = inflection.humanize(inflection.underscore(chunk["signature"]))

    # Check if docstring is provided
    docstring = ""
    if chunk["docstring"]:
        docstring = f"that docs {chunk['docstring']} "

    # Extract the location of that snippet of code
    context = (
        f"module {chunk['context']['module']} " f"file {chunk['context']['file_name']}"
    )
    if chunk["context"]["struct_name"]:
        struct_name = inflection.humanize(
            inflection.underscore(chunk["context"]["struct_name"])
        )
        context = f"defined in struct {struct_name} {context}"

    # Combine all the bits and pieces together
    text_representation = (
        f"{chunk['code_type']} {name} "
        f"{docstring}"
        f"defined as {signature} "
        f"{context}"
    )

    # Remove any special characters and concatenate the tokens
    tokens = re.split(r"\W", text_representation)
    tokens = filter(lambda x: x, tokens)
    return " ".join(tokens)


# %%
text_representations = list(map(textify, structures))
# %%

nlp_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp_embeddings = nlp_model.encode(
    text_representations,
    show_progress_bar=True,
)

# %%
# Extract the code snippets from the structures to a separate list
code_snippets = [structure["context"]["snippet"] for structure in structures]

code_model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-code",
    token=secrets.get("hugging_face").get("token"),
    trust_remote_code=True,
)
code_model.max_seq_length = 8192  # increase the context length window
code_embeddings = code_model.encode(
    code_snippets,
    batch_size=4,
    show_progress_bar=True,
)

# %%

client = QdrantClient(**secrets.get("qdrant").get("cloud"))
collection_name = "code_base"
if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text": models.VectorParams(
                size=nlp_embeddings.shape[1],
                distance=models.Distance.COSINE,
            ),
            "code": models.VectorParams(
                size=code_embeddings.shape[1],
                distance=models.Distance.COSINE,
            ),
        },
    )

# %%

points = [
    models.PointStruct(
        id=uuid.uuid4().hex,
        vector={
            "text": text_embedding,
            "code": code_embedding,
        },
        payload=structure,
    )
    for text_embedding, code_embedding, structure in zip(
        nlp_embeddings, code_embeddings, structures
    )
]

client.upload_points(
    collection_name=collection_name,
    points=points,
    batch_size=64,
)

# %%
query = "How do I count points in a collection?"

hits = client.query_points(
    collection_name=collection_name,
    query=nlp_model.encode(query).tolist(),
    using="text",
    limit=5,
).points

# %%
hits = client.query_points(
    collection_name=collection_name,
    query=code_model.encode(query).tolist(),
    using="code",
    limit=5,
).points
# %%
[hits[i].score for i in range(len(hits))]
# %%
