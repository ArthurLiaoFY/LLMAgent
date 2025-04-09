from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent_framework.core.states.pg_states import DatabaseState


def llm_inference_route(state: DatabaseState):
    """Reconnect to database"""
    if state["debug"]:
        return "extract_fake_summary"
    else:
        return "extract_table_summary"
