from langchain_core.prompts import ChatPromptTemplate

sql_coder_from_rag = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ 
            Your task is to convert a question into a SQL query, given a Postgres database schema.
            Adhere to these rules:
            - Deliberately go through the question and database schema word by word to appropriately answer the question
            - Use Table Aliases to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
            - When creating a ratio, always cast the numerator as float""",
        ),
        (
            "human",
            """
            Generate a SQL query that answers the question `{question}`.
            This query will run on a database whose schema is represented in : {content}
            The following SQL query best answers the question `{question}`:
            ```sql
            """,
        ),
    ]
)
