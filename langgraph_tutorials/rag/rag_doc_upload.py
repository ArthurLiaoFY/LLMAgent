# %%
import json

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# %%
with open("../../secrets.json") as f:
    secrets = json.loads(f.read())
collection_name = "advance_rag"
embd_model_name = "all-MiniLM-L6-v2"
device = "mps"
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
# %%

embd_model = SentenceTransformer(
    embd_model_name,
    device=device,
)
# %%
docs_list = [
    item for sublist in [WebBaseLoader(url).load() for url in urls] for item in sublist
]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)
# %%
client = QdrantClient(**secrets.get("qdrant").get("cloud"))

if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embd_model.get_sentence_embedding_dimension(),
            distance=Distance.COSINE,
        ),
    )
# %%
client.upload_collection(
    collection_name=collection_name,
    vectors=[embd_model.encode(sentences=doc_s.page_content) for doc_s in doc_splits],
    payload=[
        {**doc_s.metadata, **{"content": doc_s.page_content}} for doc_s in doc_splits
    ],
    ids=[idx for idx in range(len(doc_splits))],
    # Vector ids will be assigned automatically,
    # if you dont specify the id, the collection may have duplicate data with different id
    batch_size=36,  # How many vectors will be uploaded in a single request?
)

# %%
