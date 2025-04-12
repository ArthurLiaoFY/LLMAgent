from typing import Annotated

import numpy as np
from IPython.display import Image, display
from langchain_core.messages import ToolMessage, convert_to_messages
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_ollama import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command
from langgraph_supervisor import create_supervisor
from typing_extensions import Literal


def pretty_print_messages(stream_chunk):
    if isinstance(stream_chunk, tuple):
        ns, stream_chunk = stream_chunk
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")

    for node_name, node_update in stream_chunk.items():
        print(f"Update from node {node_name}:")
        print("\n")

        for m in convert_to_messages(node_update["messages"]):
            m.pretty_print()
        print("\n")


def make_handoff_tool(*, agent_name: str):
    """Create a tool that can return handoff via a Command"""
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Ask another agent for help."""
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            # navigate to another agent node in the PARENT graph
            goto=agent_name,
            graph=Command.PARENT,
            # This is the state update that the agent `agent_name` will see when it is invoked.
            # We're passing agent's FULL internal message history AND adding a tool message to make sure
            # the resulting chat history is valid.
            update={"messages": state["messages"] + [tool_message]},
        )

    return handoff_to_agent


# %%


@tool
def add(a: list) -> float:
    """Sum up numbers provided."""
    return np.sum(a)


@tool
def multiply(a: list) -> float:
    """Multiply numbers provided."""
    return np.prod(a)


@tool
def mean(a: list) -> float:
    """Calculate mean value of numbers provided."""
    return np.mean(a)


addition_agent = create_react_agent(
    model=model,
    tools=[add, make_handoff_tool(agent_name="multiplication_agent")],
    name="addition_expert",
    prompt=(
        "You are an addition expert, you can ask the multiplication expert for help with multiplication. "
        "Always do your portion of calculation before the handoff."
    ),
)

multiplication_agent = create_react_agent(
    model=model,
    tools=[multiply, make_handoff_tool(agent_name="addition_agent")],
    name="multiplication_expert",
    prompt=(
        "You are an multiplication expert, you can ask the addition expert for help with addition. "
        "Always do your portion of calculation before the handoff."
    ),
)
supervisor = create_supervisor(
    agents=[addition_agent, multiplication_agent],
    model=model,
    tools=[
        make_handoff_tool(agent_name="addition_agent"),
        make_handoff_tool(agent_name="multiplication_agent"),
    ],
    prompt=(
        "You are a math team supervisor managing a addition expert and a multiplication expert. "
        "Your work is to distill the problem into addition part and multiplication part. "
        "For addition problems, call addition_agent for help. "
        "For multiplication problems, call multiplication_agent for help. "
        "Do not solve the problem by yourself."
    ),
)


# %%

graph = supervisor.compile()


# %%


# %%
for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "human",
                "content": "what's (3 + 5) * 12",
            }
        ]
    }
):
    pretty_print_messages(chunk)
# %%
for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "human",
                "content": "hi, my name is arthur",
            }
        ]
    }
):
    pretty_print_messages(chunk)
# %%
