# %%
import json

import numpy as np
from langchain.tools import tool
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing_extensions import List, TypedDict, Union

with open("../../config.json") as f:
    config = json.loads(f.read())


model = ChatOllama(model=config.get("llm_model").get("model_name"), temperature=0)


# %%
class GraphState(AgentState):
    query: str
    model_response: str
    a_for_addition: List[Union[float, int]]
    a_for_multiply: List[Union[float, int]]


chat_bot_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            template="""You are a helpful assistance"""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


@tool
def call_model(messages: list, query: str):
    """call llm model for chat"""
    return (chat_bot_prompt | model).invoke({"messages": messages, "query": query})


def call_model_node(state: GraphState):
    model_response = call_model.invoke(
        {
            "messages": state["messages"],
            "query": state["messages"][-1].content,
        }
    )
    return {
        "messages": model_response,
        "model_response": model_response.content,
    }


memory = MemorySaver()
graph = StateGraph(state_schema=GraphState)
graph.add_node("call model", call_model_node)
graph.add_edge(START, "call model")
graph.add_edge("call model", END)
app = graph.compile(checkpointer=memory)
result = app.invoke(
    input={"messages": ["hi my name is arthur"]},
    config={"configurable": {"thread_id": "test123"}},
)

# %%
app.invoke(
    input={"messages": ["what is my name"]},
    config={"configurable": {"thread_id": "test123"}},
)
# %%
app.invoke(
    input={"messages": ["what is the first alphabet of my name."]},
    config={"configurable": {"thread_id": "test123"}},
)
# %%
