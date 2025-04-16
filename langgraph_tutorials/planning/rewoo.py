# %%
import json
import os
import re
from typing import Dict, List

from IPython.display import Image, display
from langchain.cache import InMemoryCache
from langchain.globals import set_debug, set_llm_cache
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables.base import RunnableSequence
from langchain_core.tools.structured import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

set_llm_cache(InMemoryCache())
set_debug(True)

with open("../secrets.json") as f:
    secrets = json.loads(f.read())

with open("../config.json") as f:
    config = json.loads(f.read())

os.environ["LANGSMITH_API_KEY"] = secrets.get("langsmith").get("api_key")
os.environ["TAVILY_API_KEY"] = secrets.get("tavily").get("api_key")


model = ChatOllama(model=config.get("llm_model").get("model_name"), temperature=0)
tavily_search = TavilySearchResults(max_results=3)


@tool
def chat_bot(query: str):
    """
    You are a smart and friendly AI assistant. You handle general conversations, tasks, and instructions normally and directly.
    However, when a user asks a factual or knowledge-based question — especially about current events, news, specific information, or anything that may require up-to-date facts — you must NOT answer it yourself.
    """
    return model.invoke({"query": query})


@tool
def redefine_tool_input(query: str, history_results: List[str]):
    """This tool rewrites a user's follow-up query into a self-contained, context-independent question by referencing the provided historical information."""
    redefine_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                template="""
                    You are an intelligent assistant that rewrites user questions based on the context provided.
                    Given a passage and a follow-up query that refers to the passage, rewrite the query into a self-contained question, replacing all references with specific information from the passage.
                """
            ),
            HumanMessagePromptTemplate.from_template(
                template="""
                    Passage:
                    {history}

                    Original Query:
                    {query}

                    Rewritten Query:
                """
            ),
        ]
    )
    chain = redefine_prompt | model
    return chain.invoke({"query": query, "history": ", ".join(history_results)})


@tool
def summarized_tavily_result(query: str):
    """
    A search engine optimized for comprehensive, accurate, and trusted results.
    Useful for when you need to answer questions about current events.
    Input should be a search query.
    And will summarizes and answers base on user questions.
    """
    tavily_search = TavilySearchResults(max_results=3)
    search_results = tavily_search.invoke({"query": query})
    search_result_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                template="""
                    You are a smart assistant that answers questions based on search engine results.
                    Given a user question and a list of search results retrieved from Tavily, 
                    extract the most relevant information to directly answer the question.
                    Keep your answer concise and factual. Include important numbers or descriptions if available. 
                    Do not add extra explanations or assumptions.
                    """
            ),
            HumanMessagePromptTemplate.from_template(
                template="""
                    User Question:
                    {query}

                    Tavily search results:
                    {search_results}
                    """
            ),
        ]
    )
    chain = search_result_prompt | model

    return chain.invoke({"query": query, "search_results": search_results})


# %%

# %%
tools = [chat_bot, summarized_tavily_result]

# %%


class Plan(BaseModel):
    order: int = Field(description="The execution order of the plan.")
    plan: str = Field(
        description="A brief description of the plan to solve the problem."
    )
    tool_name: str = Field(description="The tool used to solve the problem.")
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
    plans: Dict[str, PlanExecuted]
    # -------------------------
    question_answer: str
    # -------------------------
    current_step: int
    total_steps: int
    plan_history: List[str]


planner_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            template="""
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
            """
        ),
        HumanMessagePromptTemplate.from_template(
            template="""
                Begin! 
                Describe your plans with rich details. Each Plan should be followed by only one evidence.

                Task: {task}
            """
        ),
    ]
)


# %%
def get_plan(state: ReWOO):
    planned_results = state["planner_chain"].invoke(
        {
            "task": state["question"],
            "tool_desc": state["tools_desc"],
            "plan_history": state["plan_history"],
        }
    )
    return {
        "total_steps": len(planned_results.plans),
        "plans": {
            str(plan.order): {
                "plan": plan.plan,
                "tool_name": plan.tool_name,
                "tool_input": plan.tool_input,
            }
            for plan in planned_results.plans
        },
    }


def tool_execution(state: ReWOO):
    step = str(state["current_step"])
    plan_detail = state["plans"][step]
    tool_execute_result = state["tools"][plan_detail["tool_name"]].invoke(
        {
            "query": redefine_tool_input.invoke(
                {
                    "query": plan_detail["tool_input"],
                    "history_results": state["plan_history"],
                }
            ).content
        }
    )
    state["plans"][step].update({"tool_executed_result": tool_execute_result.content})
    state["plan_history"].append(tool_execute_result.content)

    return {"current_step": state["current_step"] + 1}


def continue_tool_execute(state: ReWOO):
    if state["current_step"] > state["total_steps"]:
        return END
    else:
        return "tool execute"


# %%

graph = StateGraph(ReWOO)
graph.add_node("plan", get_plan)
graph.add_node("tool execute", tool_execution)
graph.add_edge(start_key=START, end_key="plan")
graph.add_edge(start_key="plan", end_key="tool execute")
graph.add_conditional_edges(source="tool execute", path=continue_tool_execute)

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
        "current_step": 1,
        "plan_history": [],
    }
)

# %%
result
# %%
