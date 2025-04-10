# %%
import json

import numpy as np
import pandas as pd
import psycopg2
from IPython.display import Image, display
from langchain.cache import InMemoryCache
from langchain.globals import set_debug, set_llm_cache
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLCheckerTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph_supervisor import create_handoff_tool, create_supervisor
from langsmith import Client
from psycopg2.extras import DictCursor
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Dict, List, TypedDict, Union

with open("secrets.json") as f:
    secrets = json.loads(f.read())
set_llm_cache(InMemoryCache())
set_debug(False)
# %%
# Create a LANGSMITH_API_KEY in Settings > API Keys
prompt = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": """
                Given an input question, create a syntactically correct {dialect} query to run to help find the answer. 
                You can order the results by a relevant column to return the most interesting examples in the database.
                Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
                Pay attention to use only the column names that you can see in the schema description. 
                Be careful to not query for columns that do not exist. 
                Also, pay attention to which column is in which table.

                Only use the following tables:
                {table_info}
            """,
        },
        {
            "role": "human",
            "content": "Question: {input}",
        },
    ]
)

prompt2 = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": """
                Given an SQL query, please check the followings and fix it.
                    - Wrap each column name and table name in double quotes to denote them as delimited identifiers.
            """,
        },
        {
            "role": "human",
            "content": "SQL query: {query}",
        },
    ]
)


model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)
db = psycopg2.connect(**secrets.get("postgres"))
db_for_llm = SQLDatabase.from_uri(
    database_uri="postgresql://{user}:{password}@{host}:{port}/{dbname}".format(
        **secrets.get("postgres")
    )
)
db_for_llm._sample_rows_in_table_info = 5


@tool
def get_table_schema(db: SQLDatabase, tables: List[str]):
    """return the respective schema and sample rows for selected tables."""
    return db.get_table_info_no_throw(table_names=tables)


@tool
def get_tables_from_db(db: SQLDatabase):
    """return the tables from database."""
    return db.get_usable_table_names()


@tool
def run_query(db: SQLDatabase, query: str):
    """run query and return results"""
    return db.run_no_throw(query)


class SQLQueryOutput(BaseModel):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


# %%


@tool
def get_sql_query(db: SQLDatabase, question: str):
    """
    make llm return the answer of the question

    Example:
        get_sql_query.invoke(
            {
                "db": db_for_llm,
                "question": "what is the mean and std of sepal length for 'setosa'",
            }
        )
    """
    return (
        (prompt | model.with_structured_output(SQLQueryOutput))
        .invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "table_info": db.get_table_info_no_throw(
                    table_names=db.get_usable_table_names()
                ),
                "input": question,
            }
        )
        .query
    )


@tool
def basic_check_sql_query_without_execute(query: str):
    """
    check the query result without executing
    """
    return (
        (prompt2 | model.with_structured_output(SQLQueryOutput))
        .invoke(
            {
                "query": query,
            }
        )
        .query
    )


query = get_sql_query.invoke(
    {
        "db": db_for_llm,
        "question": "what is the mean and std of sepal length for 'setosa'",
    }
)
query2 = basic_check_sql_query_without_execute.invoke({"query": query})

# sql_query_checker = QuerySQLCheckerTool(db=db_for_llm, llm=model)
# sql_query_checker.invoke(
#     {
#         "query": "SELECT AVG(SepalLength) AS Mean, STDDEV(SepalLength) AS Std FROM Iris WHERE Species = 'Iris-setosa';"
#     }
# )

# %%


class Response(BaseModel):
    sql_query: str = Field(description="The postgreSQL query to answer the question.")


class SQLTableSchema(TypedDict):
    columns: List[str]
    schema: str


class GraphState(BaseModel):
    list_of_tables: List[str] = Field(description="list of tables in database")
    table_schemas: Dict[str, str] = Field(
        description="dict of table as key and schema as value"
    )


ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a concise summary of the following SQL information: \\n\\n"
            "{single_table_info}",
        )
    ]
)

sql_agent_prompt = """
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct postgresql query to run.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.

    You have access to tools for interacting with the database. 
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    Wrap each column name in double quotes to denote them as delimited identifiers.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables.
"""

# %%

sql_agent = create_react_agent(
    name="sql_agent",
    model=model,
    tools=sql_tools,
    prompt=sql_agent_prompt,
    state_schema=GraphState,
    response_format=Response,
)
# %%
result = sql_agent.invoke(
    input={
        "messages": [
            {
                "role": "human",
                "content": "what is the mean and std of sepal length of 'setosa'",
            }
        ]
    },
    config={"recursion_limit": 10},
)

# %%
result
# %%


@tool
def sql_query_to_dataframe(db: psycopg2.extensions.connection, sql_query: str) -> float:
    """Execute postgreSQL query and return as pd.DataFrame"""
    cur = db.cursor(cursor_factory=DictCursor)
    cur.execute(sql_query)
    return pd.DataFrame(
        data=cur.fetchall(),
        columns=[desc[0] for desc in cur.description],
    )


# %%

sql_query_to_dataframe.invoke(
    {"db": db, "sql_query": result["structured_response"].sql_query}
)


# %%


@tool
def add(a: list) -> float:
    """Transfer postgreSQL query to pd.DataFrame"""
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
