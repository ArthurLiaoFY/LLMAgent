# %%
import json

from IPython.display import Image, display

from agent_framework.core.agent.pg_to_qdrant_agent import table_summary_upsert_agent

with open("secrets.json") as f:
    secrets = json.loads(f.read())

with open("config.json") as f:
    config = json.loads(f.read())
# %%
table_summary_upsert_agent().invoke(
    {
        "postgres_connection_info": secrets.get("postgres"),
        "qdrant_connection_info": secrets.get("qdrant"),
        "collection": config.get("vector_store").get("collection"),
        "recursion_limit": 4,
        "debug": False,
    }
)

# %%

display(Image(table_summary_upsert_agent().get_graph().draw_mermaid_png()))
# %%
