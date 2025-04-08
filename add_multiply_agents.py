# %%

from typing import Annotated, Union

import numpy as np
from IPython.display import Image, display
from langchain.cache import InMemoryCache
from langchain.globals import set_debug, set_llm_cache
from langchain_core.messages import ToolMessage, convert_to_messages
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_ollama import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command
from langgraph_supervisor import create_handoff_tool, create_supervisor
from pydantic import BaseModel, Field
from typing_extensions import Literal

set_llm_cache(InMemoryCache())
set_debug(False)
# %%
model = ChatOllama(
    model="qwen2.5:14b",
    temperature=0,
)


class Response(BaseModel):
    mathematic_answer: Union[float, int] = Field(
        description="the answer of the math equation."
    )


class GraphState(AgentState):
    structured_response: Response


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
    tools=[
        add,
        create_handoff_tool(
            agent_name="multiplication_agent",
            name="assign to multiplication expert",
            description="Assign task to multiplication expert",
        ),
    ],
    name="addition_agent",
    prompt=(
        "You are an addition expert, you can ask the multiplication expert for help with multiplication. "
        "Always do your portion of calculation before the handoff."
    ),
    state_schema=GraphState,
)
multiplication_agent = create_react_agent(
    model=model,
    tools=[
        multiply,
        create_handoff_tool(
            agent_name="addition_agent",
            name="assign to addition expert",
            description="Assign task to addition expert",
        ),
    ],
    name="multiplication_agent",
    prompt=(
        "You are an multiplication expert, you can ask the addition expert for help with addition. "
        "Always do your portion of calculation before the handoff."
    ),
)
supervisor = create_supervisor(
    agents=[addition_agent, multiplication_agent],
    model=model,
    tools=[
        create_handoff_tool(
            agent_name="addition_agent",
            name="assign to addition expert",
            description="Assign task to addition expert",
        ),
        create_handoff_tool(
            agent_name="multiplication_agent",
            name="assign to multiplication expert",
            description="Assign task to multiplication expert",
        ),
    ],
    prompt=(
        "You are a math team supervisor managing a addition expert and a multiplication expert. "
        "Your work is to distill the problem into addition part and multiplication part. "
        "For addition problems, call addition_agent for help. "
        "For multiplication problems, call multiplication_agent for help. "
        "Do not solve the problem by yourself."
    ),
    state_schema=GraphState,
    response_format=Response,
)


# %%

graph = supervisor.compile()

# %%
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

# %%
result = graph.invoke(
    {
        "messages": [
            {
                "role": "human",
                "content": "what's (3 + 5) * 12",
            }
        ]
    }
)
# %%
result = graph.invoke(
    {
        "messages": [
            {
                "role": "human",
                "content": "hi, my name is arthur",
            }
        ]
    }
)
# %%
