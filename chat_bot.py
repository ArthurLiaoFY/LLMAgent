# %%
import json
import os
from typing import Annotated, Union

from IPython.display import Image, display
from langchain.cache import InMemoryCache
from langchain.globals import set_debug, set_llm_cache
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

set_llm_cache(InMemoryCache())
set_debug(True)
# %%

with open("secrets.json") as f:
    secrets = json.loads(f.read())

with open("config.json") as f:
    config = json.loads(f.read())

os.environ["LANGSMITH_API_KEY"] = secrets.get("langsmith").get("api_key")
os.environ["TAVILY_API_KEY"] = secrets.get("tavily").get("api_key")

tavily_tool = TavilySearchResults(max_results=2)
llm = ChatOllama(model=config.get("llm_model", {}).get("model_name"), temperature=0)

# %%


@tool
def fahrenheit_to_celsius(fahrenheit: Union[float, int]) -> Union[float, int]:
    """temperature fahrenheit convert to celsius"""
    return (fahrenheit - 32) * 5 / 9


@tool
def celsius_to_fahrenheit(celsius: Union[float, int]) -> Union[float, int]:
    """temperature celsius convert to fahrenheit"""
    return celsius * 9 / 5 + 32


# %%
class Response(BaseModel):
    Fahrenheit: Union[float, int] = Field(
        description="the temperature in respective location in Fahrenheit °F."
    )
    Celsius: Union[float, int] = Field(
        description="the temperature in respective location in Celsius °C."
    )


class GraphState(AgentState):
    structured_response: Response


# %%

agent = create_react_agent(
    model=llm,
    tools=[tavily_tool, fahrenheit_to_celsius, celsius_to_fahrenheit],
    state_schema=GraphState,
    response_format=(
        "Always call temperature convert tools while you only got celsius temperature or fahrenheit temperature, "
        "if you receive celsius temperature, use tool 'celsius_to_fahrenheit', "
        "if you receive fahrenheit temperature, use tool 'fahrenheit_to_celsius'.",
        Response,
    ),
    prompt="""
    You are a supervisor, 
    for searching task, use tavily tool, 
    for temperature transformation, 
    use fahrenheit_to_celsius or celsius_to_fahrenheit tool
    """,
)

# %%

try:
    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
result = agent.invoke(
    input={
        "messages": [
            {
                "role": "human",
                "content": "what is the celsius temperature 0 in fahrenheit?",
            }
        ],
    },
)
result

# %%
result = agent.invoke(
    input={
        "messages": [
            {
                "role": "human",
                "content": "search for the current temperature in japan tokyo?",
            }
        ],
    },
)
result
# %%
