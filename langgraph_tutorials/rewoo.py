# %%
import json
import os
import re
from typing import Dict, List

from IPython.display import Image, display
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from langchain_core.tools.structured import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

with open("../secrets.json") as f:
    secrets = json.loads(f.read())

with open("../config.json") as f:
    config = json.loads(f.read())


os.environ["LANGSMITH_API_KEY"] = secrets.get("langsmith").get("api_key")
os.environ["TAVILY_API_KEY"] = secrets.get("tavily").get("api_key")


model = ChatOllama(model=config.get("llm_model").get("model_name"), temperature=0)
travily_search = TavilySearchResults(max_results=3)


@tool
def chat_bot(query: str):
    """
    You are a smart and friendly AI assistant. You handle general conversations, tasks, and instructions normally and directly.
    However, when a user asks a factual or knowledge-based question — especially about current events, news, specific information, or anything that may require up-to-date facts — you must NOT answer it yourself.
    """


@tool
def travily_result_summarize(query: str):
    """
    A search engine optimized for comprehensive, accurate, and trusted results.
    Useful for when you need to answer questions about current events.
    Input should be a search query.
    And will summarizes and answers base on user questions.
    """
    travily_search = TavilySearchResults(max_results=5)
    search_result_prompt = ChatPromptTemplate.from_messages(
        [
            {
                "role": "human",
                "content": """
                    You are a smart assistant that answers questions based on search engine results.
                    Given a user question and a list of search results retrieved from Travily, 
                    extract the most relevant information to directly answer the question.
                    Keep your answer concise and factual. Include important numbers or descriptions if available. 
                    Do not add extra explanations or assumptions.
                    """,
            },
            {
                "role": "human",
                "content": """
                    User Question:
                    {query}

                    Travily Search Results (sorted by relevance):
                    {search_results}

                    Based on the above information, provide a direct answer to the question. 
                    If no relevant answer can be found, reply with: "The answer is not available based on the current search results."
                """,
            },
        ]
    )
    chain = search_result_prompt | model

    return chain.invoke(
        {"query": query, "search_results": travily_search.invoke({"query": query})}
    )


# %%
tools = [chat_bot, travily_result_summarize]

# %%


class Plan(BaseModel):
    order: int = Field(description="The execution order of the plan.")
    plan: str = Field(
        description="A brief description of the plan to solve the problem."
    )
    tool: str = Field(description="The tool used to solve the problem.")
    tool_input: str = Field(description="The input query to be sent to the tool.")


class PlanResponse(BaseModel):
    plans: List[Plan] = Field(description="Plan to solve the question.")


class PlanExecuted(TypedDict):
    plan: str
    tool_name: str
    tool_input: str
    tool_executed_result: str


class ReWOO(TypedDict):
    # -------------------------
    question: str
    # -------------------------
    tools: Dict[str, StructuredTool]
    tools_desc: str
    # -------------------------
    planner_chain: RunnableSequence
    # -------------------------
    plans_executed: Dict[int, PlanExecuted]
    # -------------------------
    question_answer: str
    # -------------------------
    current_step: int
    step_history: List[int]


planner_prompt = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": """
                For the following task, break it down into a sequence of minimal, logically necessary plan step by step, 
                each of which solves exactly one small sub-problem.
                For each sub-problem, indicate which external tool together with tool input to retrieve evidence. 

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
tool_desc = "\n                ".join(
    [
        """{tool_name}:  {tool_desc}""".format(
            tool_name=tool.name, tool_desc=tool.description
        )
        for tool in tools
    ]
)
# task = "hello what can you do for me?"


# %%
def get_plan(state: ReWOO):
    planned_results = state["planner_chain"].invoke(
        {"task": state["question"], "tool_desc": state["tools_desc"]}
    )
    return {
        "plans_executed": {
            plan.order: {
                "plan": plan.plan,
                "tool_name": plan.tool,
                "tool_input": plan.tool_input,
            }
            for plan in planned_results.plans
        }
    }


def tool_execution(state: ReWOO):
    return {
        "plans_executed": {
            order: {
                **plan_detail,
                **{
                    "tool_executed_result": state["tools"][
                        plan_detail["tool_name"]
                    ].invoke({"query": plan_detail["tool_input"]})
                },
            }
            for order, plan_detail in state["plans_executed"].items()
        }
    }


# %%

graph = StateGraph(ReWOO)
graph.add_node("plan", get_plan)
graph.add_node("tool execute", tool_execution)
# graph.add_node("solve", solve)
# graph.add_edge("solve", END)
# graph.add_conditional_edges("tool", _route)
graph.add_edge(START, "plan")
graph.add_edge("plan", "tool execute")

app = graph.compile()
result = app.invoke(
    {
        "question": "what is the exact hometown of the 2024 mens australian open winner",
        "tools": {tool.name: tool for tool in tools},
        "tools_desc": "\n                ".join(
            [
                """{tool_name}:  {tool_desc}""".format(
                    tool_name=tool.name, tool_desc=tool.description
                )
                for tool in tools
            ]
        ),
        "planner_chain": planner_prompt | model.with_structured_output(PlanResponse),
    }
)

# %%
result
# %%
