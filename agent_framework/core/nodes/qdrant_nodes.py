from qdrant_client.models import Distance, PointStruct, VectorParams
from typing_extensions import Dict

from agent_framework.core.model import llm_embd, llm_vector_size
from agent_framework.core.states.qdrant_states import (
    QdrantClientState,
    QdrantConnectionInfo,
)
from agent_framework.core.tools.qdrant_utils import (
    connect_collection,
    connect_qdrant_client,
    create_collection_vector_store,
)


def connect_qdrant_client_node(
    state: QdrantConnectionInfo,
) -> Dict:
    client = connect_qdrant_client.invoke(
        input={"qdrant_connection_info": state["qdrant_connection_info"]}
    )

    return {
        "qdrant_client": client,
        "recursion_time": 0 if client is not None else 1,
        "is_connected": True if client is not None else False,
    }


def reconnect_qdrant_client_node(
    state: QdrantConnectionInfo,
) -> Dict:
    client = connect_qdrant_client.invoke(
        input={"qdrant_connection_info": state["qdrant_connection_info"]}
    )

    return {
        "qdrant_client": client,
        "recursion_time": (
            state["recursion_time"]
            if client is not None
            else state["recursion_time"] + 1
        ),
        "is_connected": True if client is not None else False,
    }


def delete_connection_info_node(
    state: QdrantConnectionInfo,
):
    return {
        "qdrant_connection_info": {},
    }


def create_new_collection_node(
    state: QdrantClientState,
):
    create_collection_vector_store.invoke(
        {
            "qdrant_client": state["qdrant_client"],
            "collection": state["collection"],
            "llm_vector_size": llm_vector_size,
        }
    )


def connect_collection_node(
    state: QdrantClientState,
):
    vector_store = connect_collection.invoke(
        {
            "qdrant_client": state["qdrant_client"],
            "collection": state["collection"],
            "llm_embd": llm_embd,
        }
    )
    return {
        "vector_store": vector_store,
        "is_connected": True if vector_store is not None else False,
    }
