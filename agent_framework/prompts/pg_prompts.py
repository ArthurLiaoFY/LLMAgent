from langchain_core.prompts import ChatPromptTemplate

pg_table_information_extractor = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": "You are an SQL expert specializing in analyzing Postgres schema structures, "
            "helping users understand what information this table contains."
            "consider foreign table information if this table contains foreign keys..",
        },
        {
            "role": "human",
            "content": """
            Given an input question, 
            first create a syntactically correct Postgres query to run, 
            then look at the results of the query and return the answer. 

            Use the following tables: 
            main table: {table} with columns: {columns}, 

            primary key information: {primary_key}, 
            foreign key information: {relationship_desc}, 

            related tables: 
            {related_tables_desc}

            Question: {question}.
            """,
        },
    ]
)
