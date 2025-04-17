# %%
import json
import os
from typing import Annotated

from IPython.display import Image, display
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command, interrupt
from langgraph_supervisor import create_handoff_tool, create_supervisor
from typing_extensions import TypedDict

with open("../../secrets.json") as f:
    secrets = json.loads(f.read())

with open("../../config.json") as f:
    config = json.loads(f.read())

os.environ["LANGSMITH_API_KEY"] = secrets.get("langsmith").get("api_key")
os.environ["TAVILY_API_KEY"] = secrets.get("tavily").get("api_key")

search_tool = TavilySearchResults(max_results=2)
llm = ChatOllama(model="qwen2.5:14b", temperature=0)
db = SQLDatabase.from_uri(
    database_uri="postgresql://{user}:{password}@{host}:{port}/{dbname}".format(
        **secrets.get("postgres_work")
    )
)
sql_tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
memory = MemorySaver()
# %%
search_agent = create_react_agent(
    name="search agent",
    model=llm,
    tools=[
        *sql_tools,
        create_handoff_tool(
            agent_name="sql_agent",
            name="handoff_to_sql_agent",
            description="Assign task to sql agent for sql query generation.",
        ),
    ],
    prompt=f"""
        You are an search expert, you can ask the sql agent for help with sql query generation. 
        Always do your portion of work before the handoff.
    """,
    checkpointer=memory,
    state_schema=AgentState,
)
sql_agent = create_react_agent(
    name="sql agent",
    model=llm,
    tools=[
        *sql_tools,
        create_handoff_tool(
            agent_name="search_agent",
            name="handoff_to_search_agent",
            description="Assign task to search agent for online searching.",
        ),
    ],
    prompt=f"""
        You are an sql expert, you can ask the search agent for help with online information search. 

        Given an input question, create a syntactically correct {db.dialect} query to run to help find the answer.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
        For each table, only the listed columns are relevant for generating queries:
        Be careful to not query for columns that do not exist.

        Only use the following tables:
        1. Table: `productivity_raw`  
        Description: This table contains the **expected production quantity** information, organized by factory site and production line. Use this table when the user wants to know the planned or forecasted output for a specific line or site.
        Available columns:  
        line_name, model_name, line_rate, plan_qty 

        2. Table: `sn_raw`  
        Description: This table contains the **actual production quantity** information, organized by factory site and production line. Use this table when the user wants to know how many units were actually produced in a certain time frame.
        Available columns:  
        line_name, model_name, mo_number, station_name, sn, date_start, date_end  

        3. Table: `eqp_status_raw`  
        Description: This table contains information about the **equipment status** for each production line and factory site. Use this table when the user wants to understand the operational condition of equipment (e.g., running, idle, down).
        Available columns:  
        line_name, station_name, date_start, date_end, equipment_code, status 

        Table info:
        {db.get_table_info_no_throw(table_names=["sn_raw", "eqp_status_raw", "productivity_raw"])}

        Always do your portion of work before the handoff.
    """,
    checkpointer=memory,
    state_schema=AgentState,
)
supervisor = create_supervisor(
    supervisor_name="supervisor",
    agents=[sql_agent, search_agent],
    model=llm,
    tools=[
        create_handoff_tool(
            agent_name="search agent",
            name="handoff_to_search_agent",
            description="Assign task to search agent for online searching.",
        ),
        create_handoff_tool(
            agent_name="sql agent",
            name="handoff_to_sql_agent",
            description="Assign task to sql agent for sql query generation.",
        ),
    ],
    prompt=(
        "You are a team supervisor managing a sql expert and a search expert. "
        "Your work is to distill the problem into sql part and search part. "
        "For sql problems, call sql agent for help. "
        "For search problems, call search agent for help. "
        "Do not solve the problem by yourself."
    ),
    state_schema=AgentState,
)
app = supervisor.compile()
# %%
display(Image(app.get_graph(xray=True).draw_mermaid_png()))

# %%
thread_id = "test123"
result = app.invoke(
    input={"messages": ["my name is arthur."]},
    config={"configurable": {"thread_id": thread_id}},
)

# %%
app.invoke(
    input={"messages": ["what is the current weather in tokyo."]},
    config={"configurable": {"thread_id": thread_id}},
)
# %%
app.invoke(
    input={
        "messages": [
            "what is the sum of duration time group by equipment and status from 2024/09/01 00:00:00 to 2024/09/07 00:00:00",
        ]
    },
    config={"configurable": {"thread_id": thread_id}},
)

# %%
