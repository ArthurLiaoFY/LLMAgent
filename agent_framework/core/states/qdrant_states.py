from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from typing_extensions import Dict, TypedDict, Union

from agent_framework.core.states.pg_states import TableState


class QdrantConnectionInfo(TypedDict):
    # ---------------------------
    qdrant_connection_info: Dict[str, str]
    recursion_limit: int
    # ---------------------------
    qdrant_client: Union[QdrantClient, None]
    recursion_time: int
    is_connected: bool
    # ---------------------------


class QdrantClientState(TypedDict):
    # ---------------------------
    qdrant_client: QdrantClient
    collection: str
    # ---------------------------
    vector_store: Union[QdrantVectorStore, None]
    is_connected: bool
    # ---------------------------


class QdrantRetrieveState(TypedDict):
    pass
