from operator import add

from langchain_core.documents.base import Document
from langchain_ollama.chat_models import ChatOllama
from psycopg2.extensions import connection
from typing_extensions import Annotated, Dict, List, TypedDict, Union


class TableState(TypedDict):
    # ---------------------------
    table_oid: str
    table: str
    columns: List[str]
    primary_key: List[str]
    related_tables_desc: str
    relationship_desc: str
    # ---------------------------
    table_info_summary: Document


class PostgresConnectionInfo(TypedDict):
    # ---------------------------
    postgres_connection_info: Dict[str, Union[int, str]]
    recursion_limit: int
    # ---------------------------
    database: Union[connection, None]
    recursion_time: int
    is_connected: bool


class DatabaseState(TypedDict):
    # ---------------------------
    database: connection
    # ---------------------------
    tables: Dict[str, TableState]
