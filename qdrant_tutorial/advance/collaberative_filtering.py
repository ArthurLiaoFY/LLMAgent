# %%
import json
import os
from collections import defaultdict

import pandas as pd
import requests
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    NamedSparseVector,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

with open("../../secrets.json") as f:
    secrets = json.loads(f.read())


# Collection name
collection_name = "movies"

# Set Qdrant Client
qdrant_client = QdrantClient(**secrets.get("qdrant").get("cloud"))
if not qdrant_client.collection_exists(collection_name=collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={"ratings": SparseVectorParams()},
    )

# %%
# Load CSV files
ratings_df = pd.read_csv("../ml-latest/ratings.csv", low_memory=False)
movies_df = pd.read_csv("../ml-latest/movies.csv", low_memory=False)

# Convert movieId in ratings_df and movies_df to string
ratings_df["movieId"] = ratings_df["movieId"].astype(str)
movies_df["movieId"] = movies_df["movieId"].astype(str)
# %%
rating = ratings_df["rating"]

# Normalize ratings
ratings_df["rating"] = (rating - rating.mean()) / rating.std()

# Merge ratings with movie metadata to get movie titles
merged_df = ratings_df.merge(
    movies_df[["movieId", "title"]], left_on="movieId", right_on="movieId", how="inner"
)

# Aggregate ratings to handle duplicate (userId, title) pairs
ratings_agg_df = merged_df.groupby(["userId", "movieId"]).rating.mean().reset_index()

ratings_agg_df.head()

# %%
# Convert ratings to sparse vectors
user_sparse_vectors = defaultdict(lambda: {"values": [], "indices": []})
for row in ratings_agg_df.itertuples():
    user_sparse_vectors[row.userId]["values"].append(row.rating)
    user_sparse_vectors[row.userId]["indices"].append(int(row.movieId))


# %%
# Define a data generator
# can be change to by batch instead of by point
def data_generator():
    for user_id, sparse_vector in user_sparse_vectors.items():
        yield PointStruct(
            id=user_id,
            vector={
                "ratings": SparseVector(
                    indices=sparse_vector["indices"], values=sparse_vector["values"]
                )
            },
            payload={"user_id": user_id, "movie_id": sparse_vector["indices"]},
        )


# Upload points using the data generator
qdrant_client.upload_points(collection_name=collection_name, points=data_generator())
# %%
my_ratings = {
    603: 1,  # Matrix
    13475: 1,  # Star Trek
    11: 1,  # Star Wars
    1091: -1,  # The Thing
    862: 1,  # Toy Story
    597: -1,  # Titanic
    680: -1,  # Pulp Fiction
    13: 1,  # Forrest Jump
    120: 1,  # Lord of the Rings
    87: -1,  # Indiana Jones
    562: -1,  # Die Hard
}


# %%
# Perform the search
results = qdrant_client.query_points(
    collection_name=collection_name,
    query=SparseVector(
        indices=list(my_ratings.keys()),
        values=list(my_ratings.values()),
    ),
    using="ratings",  # you must name the vector name while using this
    limit=20,
).points


# %%
# Convert results to scores and sort by score
def results_to_scores(results):
    movie_scores = defaultdict(lambda: 0)
    for result in results:
        for movie_id in result.payload["movie_id"]:
            movie_scores[movie_id] += result.score
    return movie_scores


# Convert results to scores and sort by score
movie_scores = results_to_scores(results)
top_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

# %%
top_movies
# %%
