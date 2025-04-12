from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_framework.nodes.pg_nodes import (
    connect_database_node,
    delete_connection_info_node,
    get_database_common_info_node,
    reconnect_database_node,
)
from agent_framework.routes.llm_routes import llm_inference_route
from agent_framework.routes.pg_routes import database_connection_route
from agent_framework.states.pg_states import DatabaseState, PostgresConnectionInfo


def connect_postgres_agent() -> CompiledStateGraph:
    graph = StateGraph(PostgresConnectionInfo)
    graph.add_node(node="connect_db", action=connect_database_node)
    graph.add_node(node="reconnect_db", action=reconnect_database_node)

    graph.add_edge(start_key=START, end_key="connect_db")
    graph.add_conditional_edges(
        source="connect_db",
        path=database_connection_route,
    )
    graph.add_conditional_edges(
        source="reconnect_db",
        path=database_connection_route,
    )

    return graph.compile()


def get_postgres_table_info_agent() -> CompiledStateGraph:
    graph = StateGraph(DatabaseState)
    graph.add_node(
        node="get_database_common_info", action=get_database_common_info_node
    )
    graph.add_edge(start_key=START, end_key="get_database_common_info")
    graph.add_edge(start_key="get_database_common_info", end_key=END)
    return graph.compile()
