# %%
import json
import os
import re
from typing import List

from IPython.display import Image, display
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

with open("../secrets.json") as f:
    secrets = json.loads(f.read())

with open("../config.json") as f:
    config = json.loads(f.read())


os.environ["LANGSMITH_API_KEY"] = secrets.get("langsmith").get("api_key")
os.environ["TAVILY_API_KEY"] = secrets.get("tavily").get("api_key")

online_search = TavilySearchResults(max_results=5)
model = ChatOllama(model=config.get("llm_model").get("model_name"), temperature=0)


@tool
def summarize_results(query: str):
    "A LLM model for summarizing the results"
    return model.invoke({"query": query})


# %%
tools = [online_search, summarize_results]
# %%


class ReWOO(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str


# %%


class Plan(BaseModel):
    order: int = Field(description="The execution order of the plan.")
    plan: str = Field(
        description="A brief description of the plan to solve the problem."
    )
    tool: str = Field(description="The tool used to solve the problem.")
    tool_input: str = Field(description="The input query to be sent to the tool.")


class Response(BaseModel):
    plans: List[Plan] = Field(description="Plan to solve the question.")


prompt = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": """
                For the following task, make plans that can solve the problem step by step. For each plan, indicate \
                which external tool together with tool input to retrieve evidence. 

                Tools can be one of the following:
                {tool_desc}

                For example,
                Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
                hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
                less than Toby. How many hours did Rebecca work?
                Plan: Given Thomas worked x hours, translate the problem into algebraic expressions and solve
                with Wolfram Alpha. evidence 1 = WolframAlpha[Solve x + (2x - 10) + ((2x - 10) - 8) = 157]
                Plan: Find out the number of hours Thomas worked. evidence 2 = LLM[What is x, given evidence 1]
                Plan: Calculate the number of hours Rebecca worked. evidence = Calculator[(2 * evidence 2 - 10) - 8]
            """,
        },
        {
            "role": "human",
            "content": """
                Begin! 
                Describe your plans with rich details. Each Plan should be followed by only one evidence.

                Task: {task}
            """,
        },
    ]
)
tool_desc = "\n".join(
    [
        """{tool_name}:  {tool_desc}""".format(
            tool_name=tool.name, tool_desc=tool.description
        )
        for tool in tools
    ]
)
task = "what is the exact hometown of the 2024 mens australian open winner"
model_with_structured_output = model.with_structured_output(Response)
chain = prompt | model_with_structured_output
result = chain.invoke({"task": task, "tool_desc": tool_desc})

# %%
# result.plans[0].plan
# result.plans[0].tool
result.plans[0].tool_input


# %%