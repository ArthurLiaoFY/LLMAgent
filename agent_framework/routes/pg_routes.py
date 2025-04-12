from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent_framework.states.pg_states import PostgresConnectionInfo


def database_connection_route(state: PostgresConnectionInfo):
    """Reconnect to database"""
    if state["is_connected"]:
        return END
    else:
        if state["recursion_time"] < state["recursion_limit"]:
            return "reconnect_db"
        else:
            return END


# def inspect_table(state: DatabaseState):
#     # We will return a list of `Send` objects
#     # Each `Send` object consists of the name of a node in the graph
#     # as well as the state to send to that node
#     return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
