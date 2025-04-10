from langgraph.graph import END

from agent_framework.core.states.pg_to_qdrant_states import PostgresQdrantState


def database_connection_fail_route(state: PostgresQdrantState):
    """check connection of database"""
    if state["database_is_connected"]:
        return "gather"
    else:
        return END


def vector_store_connection_fail_route(state: PostgresQdrantState):
    """check connection of vector store"""
    if state["vector_store_is_connected"]:
        return "gather"
    else:
        return END
