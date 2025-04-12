from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_framework.nodes.pg_to_qdrant_nodes import (
    check_point_exist_node,
    extract_table_summary_node,
    get_table_info_node,
    get_vector_store_info_node,
    remove_sensitive_info_node,
    upsert_to_vector_database_node,
)
from agent_framework.routes.pg_to_qdrant_routes import (
    database_connection_fail_route,
    vector_store_connection_fail_route,
)
from agent_framework.states.pg_to_qdrant_states import PostgresQdrantState


def table_summary_upsert_agent() -> CompiledStateGraph:
    graph = StateGraph(PostgresQdrantState)
    graph.add_node(node="get_tables_info", action=get_table_info_node)
    graph.add_node(node="get_vector_store_info", action=get_vector_store_info_node)
    graph.add_node(node="gather", action=RunnablePassthrough())
    graph.add_node(node="check_point_exist", action=check_point_exist_node)
    graph.add_node(node="extract_table_summary", action=extract_table_summary_node)
    graph.add_node(node="point_upsert", action=upsert_to_vector_database_node)
    graph.add_node(node="remove_sensitive_info", action=remove_sensitive_info_node)

    graph.add_edge(start_key=START, end_key="get_tables_info")
    graph.add_edge(start_key=START, end_key="get_vector_store_info")
    graph.add_conditional_edges(
        source="get_tables_info",
        path=database_connection_fail_route,
    )
    graph.add_conditional_edges(
        source="get_vector_store_info",
        path=vector_store_connection_fail_route,
    )
    graph.add_edge(
        start_key="gather",
        end_key="check_point_exist",
    )
    graph.add_edge(start_key="check_point_exist", end_key="extract_table_summary")
    graph.add_edge(start_key="extract_table_summary", end_key="point_upsert")
    graph.add_edge(start_key="point_upsert", end_key="remove_sensitive_info")
    graph.add_edge(start_key="remove_sensitive_info", end_key=END)

    return graph.compile()
