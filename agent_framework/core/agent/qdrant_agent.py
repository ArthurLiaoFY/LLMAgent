from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_framework.core.nodes.qdrant_nodes import (
    connect_collection_node,
    connect_qdrant_client_node,
    create_new_collection_node,
    delete_connection_info_node,
    reconnect_qdrant_client_node,
)
from agent_framework.core.routes.llm_routes import llm_inference_route
from agent_framework.core.routes.qdrant_routes import (
    client_connection_route,
    collection_connection_route,
)
from agent_framework.core.states.qdrant_states import (
    QdrantClientState,
    QdrantConnectionInfo,
)


def connect_qdrant_agent() -> CompiledStateGraph:
    graph = StateGraph(QdrantConnectionInfo)
    graph.add_node(node="connect_qdrant_client", action=connect_qdrant_client_node)
    graph.add_node(node="reconnect_qdrant_client", action=reconnect_qdrant_client_node)
    graph.add_node(node="delete_sensitive_info", action=delete_connection_info_node)

    graph.add_edge(start_key=START, end_key="connect_qdrant_client")
    graph.add_conditional_edges(
        source="connect_qdrant_client", path=client_connection_route
    )
    graph.add_conditional_edges(
        source="reconnect_qdrant_client", path=client_connection_route
    )
    # graph.add_edge(start_key="delete_sensitive_info", end_key=END)

    return graph.compile()


def connect_qdrant_collection_agent() -> CompiledStateGraph:
    graph = StateGraph(QdrantClientState)
    graph.add_node(node="connect_collection", action=connect_collection_node)
    graph.add_node(node="create_new_collection", action=create_new_collection_node)

    graph.add_edge(start_key=START, end_key="connect_collection")
    graph.add_conditional_edges(
        source="connect_collection", path=collection_connection_route
    )
    graph.add_edge(start_key="create_new_collection", end_key="connect_collection")

    return graph.compile()
