# %%
import json

from IPython.display import Image, display

from agent_framework.agent.pg_to_qdrant_agent import table_summary_upsert_agent

with open("secrets.json") as f:
    secrets = json.loads(f.read())

with open("config.json") as f:
    config = json.loads(f.read())
# %%
agent = table_summary_upsert_agent()
# %%
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))


# %%
a = agent.invoke(
    {
        "postgres_connection_info": secrets.get("postgres"),
        "qdrant_connection_info": secrets.get("qdrant"),
        "collection": config.get("vector_store").get("collection"),
        "recursion_limit": 4,
        "debug": False,
    }
)

# %%
a
# %%
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("att_rawdata_202503310947.csv")
# %%
df.select_dtypes(include="object").aggregate(
    func=lambda x: list(np.unique(x)), axis=0
).to_dict()
# %%
