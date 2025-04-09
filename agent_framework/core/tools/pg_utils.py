import pandas as pd
import psycopg2
from psycopg2.extensions import connection
from typing_extensions import Dict, List, Union

from agent_framework.core.model import llm_model
from agent_framework.core.prompts.pg_prompts import pg_table_information_extractor
from langchain.tools import tool


@tool
def database_connection(
    postgres_connection_info: Dict[str, Union[int, str]],
) -> Union[connection, None]:
    """Connect to Postgres database"""
    try:
        return psycopg2.connect(
            host=postgres_connection_info.get("host"),
            port=postgres_connection_info.get("port"),
            dbname=postgres_connection_info.get("dbname"),
            user=postgres_connection_info.get("user"),
            password=postgres_connection_info.get("password"),
        )

    except psycopg2.OperationalError:
        return None


@tool
def close_connection(database: connection) -> None:
    """Close connection to Postgres database"""
    database.close()


@tool
def get_table_list(database: connection) -> List[str]:
    """Get a list of tables in the database."""
    with database.cursor() as curs:
        curs.execute(
            """
            SELECT table_name 
            FROM information_schema.tables
            WHERE table_schema = 'public'
            """
        )
        return [row[0] for row in curs.fetchall()]


@tool
def get_table_columns(database: connection, table_name: str) -> List[str]:
    """Get whole columns from table."""
    with database.cursor() as curs:
        curs.execute(
            f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{table_name}';
            """
        )
        return [row[0] for row in curs.fetchall()]


@tool
def get_table_oid(database: connection, table_name: str) -> List[str]:
    """Get oid from table."""
    with database.cursor() as curs:
        curs.execute(
            f"""
            SELECT oid
            FROM pg_class 
            WHERE relname = '{table_name}';
            """
        )
        return [str(row[0]) for row in curs.fetchall()][0]


@tool
def get_related_tables(database: connection, table_name: str) -> List[str]:
    """Find all the table related with table input."""
    with database.cursor() as curs:
        curs.execute(
            f"""
            SELECT 
                tc.table_name AS table_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            WHERE 1=1
                AND tc.constraint_type = 'FOREIGN KEY' 
                AND kcu.table_name = '{table_name}';
            """
        )
        return [row[0] for row in curs.fetchall()]


@tool
def get_related_tables_desc(database: connection, table_name: str) -> List[str]:
    """Find all the table desc related with table input."""
    related_table_desc = [
        row
        + " with columns: "
        + ", ".join(get_table_columns.invoke({"database": database, "table_name": row}))
        for row in get_related_tables.invoke(
            {"database": database, "table_name": table_name}
        )
    ]
    return (
        " and ".join(related_table_desc) + ". " if len(related_table_desc) > 0 else ""
    )


@tool
def get_relationship_desc(database: connection, table_name: str) -> str:
    """Find all the table relationship desc."""
    with database.cursor() as curs:
        curs.execute(
            f"""
            SELECT 
                kcu.column_name AS fk_column,
                ccu.table_name AS referenced_table,
                ccu.column_name AS referenced_column
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE 1=1
                AND tc.constraint_type = 'FOREIGN KEY' 
                AND tc.table_name = '{table_name}';
            """
        )

        foreign_key_info = [
            f"foreign key {row[0]} references {row[2]} in table {row[1]}"
            for row in curs.fetchall()
        ]
        return (
            " and ".join(foreign_key_info) + ". " if len(foreign_key_info) > 0 else ""
        )


@tool
def get_table_primary_key(database: connection, table_name: str) -> List[str]:
    """Get primary key from table."""
    with database.cursor() as curs:
        curs.execute(
            f"""
            SELECT k.column_name
            FROM information_schema.table_constraints t
            JOIN information_schema.key_column_usage k
                ON t.table_name = k.table_name
                AND t.constraint_name = k.constraint_name
            WHERE t.constraint_type = 'PRIMARY KEY'
                AND t.table_name = '{table_name}';
            """
        )
        return [row[0] for row in curs.fetchall()]


@tool
def get_sample_data(
    database: connection, table_name: str, sample_size: int = 10
) -> pd.DataFrame:
    """Query sample from table with specified sample size."""
    with database.cursor() as curs:
        curs.execute(
            f"""
            SELECT *
            FROM public."{table_name}"
            LIMIT {sample_size};
            """
        )
        return pd.DataFrame(
            data=curs.fetchall(), columns=[desc[0] for desc in curs.description]
        )


@tool
def query(database: connection, query: str) -> pd.DataFrame:
    """Query database with specified query."""
    with database.cursor() as curs:
        curs.execute(query)
        return pd.DataFrame(
            data=curs.fetchall(), columns=[desc[0] for desc in curs.description]
        )


@tool
def table_summary_extract_from_llm(
    table_name: str,
    table_columns: List[str],
    primary_key: List[str],
    related_tables_desc: str,
    relationship_desc: str,
) -> str:
    """return LLM table summary"""
    return llm_model.invoke(
        input=pg_table_information_extractor.invoke(
            {
                # ---------------------------
                "table": table_name,
                "columns": ", ".join(table_columns),
                "primary_key": ", ".join(primary_key),
                "related_tables_desc": related_tables_desc,
                "relationship_desc": relationship_desc,
                # ---------------------------
                "question": f"What information does {table_name} table contains?",
            }
        )
    ).content
