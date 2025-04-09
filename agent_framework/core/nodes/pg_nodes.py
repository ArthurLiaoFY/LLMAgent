from langchain_core.output_parsers import JsonOutputParser
from langgraph.types import Send
from qdrant_client.models import Distance, PointStruct, VectorParams
from typing_extensions import Dict, List, Union

from agent_framework.core.model import llm_model
from agent_framework.core.prompts.pg_prompts import pg_table_information_extractor
from agent_framework.core.states.pg_states import (
    DatabaseState,
    PostgresConnectionInfo,
    TableState,
)
from agent_framework.core.tools.doc_utils import str_to_doc
from agent_framework.core.tools.pg_utils import (
    close_connection,
    connection,
    database_connection,
    get_related_tables_desc,
    get_relationship_desc,
    get_sample_data,
    get_table_columns,
    get_table_list,
    get_table_oid,
    get_table_primary_key,
    query,
)
from agent_framework.core.tools.qdrant_utils import check_point_exist


def connect_database_node(
    state: PostgresConnectionInfo,
) -> Dict[str, Union[connection, None, bool]]:
    conn = database_connection.invoke(
        input={"postgres_connection_info": state["postgres_connection_info"]}
    )

    return {
        "database": conn,
        "recursion_time": 0 if conn is not None else 1,
        "is_connected": True if conn is not None else False,
    }


def reconnect_database_node(
    state: PostgresConnectionInfo,
) -> Dict[str, Union[connection, None, bool]]:
    conn = database_connection.invoke(
        input={"postgres_connection_info": state["postgres_connection_info"]}
    )

    return {
        "database": conn,
        "recursion_time": (
            state["recursion_time"] if conn is not None else state["recursion_time"] + 1
        ),
        "is_connected": True if conn is not None else False,
    }


def delete_connection_info_node(
    state: PostgresConnectionInfo,
):
    return {
        "postgres_connection_info": {},
    }


def get_database_common_info_node(state: DatabaseState):
    return {
        "tables": {
            table_name: {
                "table_oid": get_table_oid.invoke(
                    input={"database": state["database"], "table_name": table_name}
                ),
                "table": table_name,
                "columns": get_table_columns.invoke(
                    input={"database": state["database"], "table_name": table_name}
                ),
                "primary_key": get_table_primary_key.invoke(
                    input={"database": state["database"], "table_name": table_name}
                ),
                "related_tables_desc": get_related_tables_desc.invoke(
                    input={"database": state["database"], "table_name": table_name}
                ),
                "relationship_desc": get_relationship_desc.invoke(
                    input={"database": state["database"], "table_name": table_name}
                ),
                # "sample_data": get_sample_data.invoke(
                #     input={"database": state["database"], "table_name": table_name}
                # ),
            }
            for table_name, in zip(
                get_table_list.invoke(input={"database": state["database"]})
            )
        },
    }
