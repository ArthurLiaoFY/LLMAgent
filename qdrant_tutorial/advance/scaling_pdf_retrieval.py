# %%
import json
import random
import uuid

import numpy as np
import torch
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
from datasets import load_dataset
from qdrant_client import QdrantClient, models
from tqdm import tqdm

with open("../../secrets.json") as f:
    secrets = json.loads(f.read())


# %%
client = QdrantClient(**secrets.get("qdrant").get("cloud"))
# %%
colqwen_model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="mps",  # Use "cuda:0" for GPU, "cpu" for CPU, or "mps" for Apple Silicon
).eval()

colqwen_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

# %%
colpali_model = ColPali.from_pretrained(
    "vidore/colpali-v1.3",
    torch_dtype=torch.bfloat16,
    device_map="mps",  # Use "cuda:0" for GPU, "cpu" for CPU, or "mps" for Apple Silicon
).eval()
# %%
colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")

# %%
for collection_name in ["colpali_tutorial", "colqwen_tutorial"]:
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "original": models.VectorParams(  # switch off HNSW
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),  # switching off HNSW
                ),
                "mean_pooling_columns": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
                "mean_pooling_rows": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            },
        )

# %%
ufo_dataset = "davanstrien/ufo-ColPali"
dataset = load_dataset(ufo_dataset, split="train")


# %%
def get_patches(image_size, model_processor, model, model_name):
    if model_name == "colPali":
        return model_processor.get_n_patches(image_size, patch_size=model.patch_size)
    elif model_name == "colQwen":
        return model_processor.get_n_patches(
            image_size,
            patch_size=model.patch_size,
            spatial_merge_size=model.spatial_merge_size,
        )
    return None, None


# %%
image = colpali_processor.process_images([dataset[0]["image"]]).to("mps")
image_embd = colpali_model(**image)
row_patch, column_patch = get_patches(
    image_size=dataset[0]["image"].size,
    model_processor=colpali_processor,
    model=colpali_model,
    model_name="colPali",
)


# %%
def embed_and_mean_pool_batch(image_batch, model_processor, model, model_name):
    # embed
    with torch.no_grad():
        processed_images = model_processor.process_images(image_batch).to(model.device)
        image_embeddings = model(**processed_images)

    image_embeddings_batch = image_embeddings.cpu().float().numpy().tolist()

    # mean pooling
    pooled_by_rows_batch = []
    pooled_by_columns_batch = []

    for image_embedding, tokenized_image, image in zip(
        image_embeddings, processed_images.input_ids, image_batch
    ):
        x_patches, y_patches = get_patches(
            image.size, model_processor, model, model_name
        )

        image_tokens_mask = tokenized_image == model_processor.image_token_id

        image_tokens = image_embedding[image_tokens_mask].view(
            x_patches, y_patches, model.dim
        )
        # turns a (1024, 128) tenser to (32, 32, 128)

        pooled_by_rows = torch.mean(
            image_tokens, dim=0
        )  # (32, 32, 128) to (None, 32, 128)
        pooled_by_columns = torch.mean(
            image_tokens, dim=1
        )  # (32, 32, 128) to (32, None, 128)

        image_token_idxs = torch.nonzero(image_tokens_mask.int(), as_tuple=False)
        first_image_token_idx = image_token_idxs[0].cpu().item()
        last_image_token_idx = image_token_idxs[-1].cpu().item()

        prefix_tokens = image_embedding[:first_image_token_idx]
        postfix_tokens = image_embedding[last_image_token_idx + 1 :]

        # print(f"There are {len(prefix_tokens)} prefix tokens and {len(postfix_tokens)} in a {model_name} PDF page embedding")

        # adding back prefix and postfix special tokens
        pooled_by_rows = (
            torch.cat((prefix_tokens, pooled_by_rows, postfix_tokens), dim=0)
            .cpu()
            .float()
            .numpy()
            .tolist()
        )
        pooled_by_columns = (
            torch.cat((prefix_tokens, pooled_by_columns, postfix_tokens), dim=0)
            .cpu()
            .float()
            .numpy()
            .tolist()
        )

        pooled_by_rows_batch.append(pooled_by_rows)
        pooled_by_columns_batch.append(pooled_by_columns)

    return image_embeddings_batch, pooled_by_rows_batch, pooled_by_columns_batch


# %%
def upload_batch(
    original_batch,
    pooled_by_rows_batch,
    pooled_by_columns_batch,
    payload_batch,
    collection_name,
):
    try:
        client.upload_collection(
            collection_name=collection_name,
            vectors={
                "mean_pooling_rows": pooled_by_rows_batch,
                "mean_pooling_columns": pooled_by_columns_batch,
                "original": original_batch,
            },
            payload=payload_batch,
            ids=[str(uuid.uuid4()) for i in range(len(original_batch))],
        )
    except Exception as e:
        print(f"Error during upsert: {e}")


# %%
batch_size = 1  # based on available compute
dataset_source = ufo_dataset
collection_name = "colpali_tutorial"

with tqdm(
    total=len(dataset),
    desc=f'Uploading progress of "{dataset_source}" dataset to "{collection_name}" collection',
) as pbar:
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        image_batch = batch["image"]
        current_batch_size = len(image_batch)
        try:
            original_batch, pooled_by_rows_batch, pooled_by_columns_batch = (
                embed_and_mean_pool_batch(
                    image_batch, colpali_processor, colpali_model, "colPali"
                )
            )
        except Exception as e:
            print(f"Error during embed: {e}")
            continue
        try:
            upload_batch(
                np.asarray(original_batch, dtype=np.float32),
                np.asarray(pooled_by_rows_batch, dtype=np.float32),
                np.asarray(pooled_by_columns_batch, dtype=np.float32),
                [
                    {"source": dataset_source, "index": j}
                    for j in range(i, i + current_batch_size)
                ],
                collection_name,
            )
        except Exception as e:
            print(f"Error during upsert: {e}")
            continue
        # Update the progress bar
        pbar.update(current_batch_size)
print("Uploading complete!")
# %%
batch_size = 1  # based on available compute
dataset_source = ufo_dataset
collection_name = "colqwen_tutorial"

with tqdm(
    total=len(dataset),
    desc=f'Uploading progress of "{dataset_source}" dataset to "{collection_name}" collection',
) as pbar:
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        image_batch = batch["image"]
        current_batch_size = len(image_batch)
        try:
            original_batch, pooled_by_rows_batch, pooled_by_columns_batch = (
                embed_and_mean_pool_batch(
                    image_batch, colqwen_processor, colqwen_model, "colQwen"
                )
            )
        except Exception as e:
            print(f"Error during embed: {e}")
            continue
        try:
            upload_batch(
                np.asarray(original_batch, dtype=np.float32),
                np.asarray(pooled_by_rows_batch, dtype=np.float32),
                np.asarray(pooled_by_columns_batch, dtype=np.float32),
                [
                    {"source": dataset_source, "index": j}
                    for j in range(i, i + current_batch_size)
                ],
                collection_name,
            )
        except Exception as e:
            print(f"Error during upsert: {e}")
            continue
        # Update the progress bar
        pbar.update(current_batch_size)
print("Uploading complete!")


# %%
def batch_embed_query(query_batch, model_processor, model):
    with torch.no_grad():
        processed_queries = model_processor.process_queries(query_batch).to(
            model.device
        )
        query_embeddings_batch = model(**processed_queries)
    return query_embeddings_batch.cpu().float().numpy()


# %%
query = "Megalithic statues on Pasqua Island"
colpali_query = batch_embed_query(
    [query], colpali_processor, colpali_model
)  # (1, 22, 128)
colqwen_query = batch_embed_query([query], colqwen_processor, colqwen_model)

print(
    f'ColPali embedded query "{query}" with {len(colpali_query[0])} multivectors of dim {len(colpali_query[0][0])}'
)
print(
    f'ColQwen embedded query "{query}" with {len(colqwen_query[0])} multivectors of dim {len(colqwen_query[0][0])}'
)


# %%
def reranking_search_batch(
    query_batch, collection_name, search_limit=20, prefetch_limit=200
):
    search_queries = [
        models.QueryRequest(
            query=query,
            prefetch=[
                models.Prefetch(
                    query=query, limit=prefetch_limit, using="mean_pooling_columns"
                ),
                models.Prefetch(
                    query=query, limit=prefetch_limit, using="mean_pooling_rows"
                ),
            ],
            limit=search_limit,
            with_payload=True,
            with_vector=False,
            using="original",
        )
        for query in query_batch
    ]
    return client.query_batch_points(
        collection_name=collection_name, requests=search_queries
    )


# %%
answer_colpali = reranking_search_batch(colpali_query, "colpali_tutorial")
answer_colqwen = reranking_search_batch(colqwen_query, "colqwen_tutorial")
# %%
dataset[answer_colpali[0].points[0].payload["index"]]["image"]

# %%
