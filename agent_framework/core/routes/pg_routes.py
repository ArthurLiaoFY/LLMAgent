from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent_framework.core.states.pg_states import PostgresConnectionInfo


def database_connection_route(state: PostgresConnectionInfo):
    """Reconnect to database"""
    if state["is_connected"]:
        return "delete_sensitive_info"
    else:
        if state["recursion_time"] < state["recursion_limit"]:
            return "reconnect_db"
        else:
            return "delete_sensitive_info"


# def inspect_table(state: DatabaseState):
#     # We will return a list of `Send` objects
#     # Each `Send` object consists of the name of a node in the graph
#     # as well as the state to send to that node
#     return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
