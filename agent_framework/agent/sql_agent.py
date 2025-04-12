from langchain.agents import AgentType, create_sql_agent, create_tool_calling_agent
from langchain.agents.agent import AgentExecutor, RunnableMultiActionAgent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from agent_framework.model import llm_model
from agent_framework.tools.handoff_tools import make_handoff_tool


class SQLAgentExecutor:
    def __init__(
        self,
        connection_infos: dict,
        other_agents_names: list[str],
        agent_name: str = "SQL Agent Executor",
    ):
        self.agent_name = agent_name
        self.other_agents_names = other_agents_names
        self.connection_infos = connection_infos
        self.db = SQLDatabase.from_uri(
            database_uri="postgresql://{user}:{password}@{host}:{port}/{dbname}".format(
                **self.connection_infos
            )
        )
        sql_tools = SQLDatabaseToolkit(db=self.db, llm=llm_model).get_tools()
        self.sql_agent = AgentExecutor(
            name=self.agent_name,
            agent=RunnableMultiActionAgent(
                runnable=create_tool_calling_agent(
                    llm=llm_model,
                    tools=sql_tools
                    + [
                        make_handoff_tool(agent_name=agent_name)
                        for agent_name in self.other_agents_names
                    ],
                    prompt=ChatPromptTemplate.from_messages(
                        [
                            SystemMessage(
                                content="""
                                You are an agent designed to interact with a SQL database.
                                Given an input question, create a syntactically correct {dialect} query to run.
                                Do not do any calculation by yourself, Just return information for other experts to calculate.
                                Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.

                                You can order the results by a relevant column to return the most interesting examples in the database
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
                            ),
                            HumanMessagePromptTemplate.from_template("{input}"),
                            AIMessage(
                                content="""
                                Begin!

                                Question: {input}
                                Thought:{agent_scratchpad}
                                """
                            ),
                            MessagesPlaceholder(variable_name="agent_scratchpad"),
                        ]
                    ),
                ),
                input_keys_arg=["input"],
                return_keys_arg=["output"],
            ),
            tools=sql_tools,
            verbose=True,
        )
