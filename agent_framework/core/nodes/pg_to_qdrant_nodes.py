from agent_framework.core.agent.pg_agent import (
    connect_postgres_agent,
    get_postgres_table_info_agent,
)
from agent_framework.core.agent.qdrant_agent import (
    connect_qdrant_agent,
    connect_qdrant_collection_agent,
)
from agent_framework.core.model import llm_model
from agent_framework.core.prompts.pg_prompts import pg_table_information_extractor
from agent_framework.core.prompts.sql_prompts import sql_coder_from_rag
from agent_framework.core.states.pg_to_qdrant_states import PostgresQdrantState
from agent_framework.core.tools.doc_utils import join_docs, str_to_doc
from agent_framework.core.tools.pg_utils import table_summary_extract_from_llm
from agent_framework.core.tools.qdrant_utils import check_point_exist, upsert_collection


def remove_sensitive_info_node(state: PostgresQdrantState):
    return {
        "postgres_connection_info": {},
        "qdrant_connection_info": {},
    }


def get_table_info_node(state: PostgresQdrantState):
    postgres = connect_postgres_agent().invoke(
        {
            "postgres_connection_info": state["postgres_connection_info"],
            "recursion_limit": state["recursion_limit"],
        }
    )
    table_summary = get_postgres_table_info_agent().invoke(
        {
            "database": postgres["database"],
        }
    )
    return {
        "database": postgres["database"],
        "database_is_connected": postgres["is_connected"],
        "tables": table_summary["tables"],
    }


def get_vector_store_info_node(state: PostgresQdrantState):
    qdrant = connect_qdrant_agent().invoke(
        {
            "qdrant_connection_info": state["qdrant_connection_info"],
            "recursion_limit": state["recursion_limit"],
        }
    )

    vector_store = connect_qdrant_collection_agent().invoke(
        {
            "qdrant_client": qdrant["qdrant_client"],
            "collection": state["collection"],
        }
    )
    return {
        "qdrant_client": qdrant["qdrant_client"],
        "vector_store_is_connected": qdrant["is_connected"],
        "vector_store": vector_store["vector_store"],
    }


def check_point_exist_node(state: PostgresQdrantState):
    return {
        "tables": {
            table_name: {**table_details}
            for table_name, table_details in state["tables"].items()
            if not check_point_exist(
                **{
                    "client": state["qdrant_client"],
                    "collection_name": state["collection"],
                    "table_name": table_details.get("table"),
                    "table_oid": table_details.get("table_oid"),
                }
            )
        }
    }


def extract_table_summary_node(state: PostgresQdrantState):
    return {
        "tables": {
            table_name: {
                **table_details,
                "table_info_summary": str_to_doc.invoke(
                    {
                        "content": (
                            "Hello World"
                            if state["debug"]
                            else table_summary_extract_from_llm.invoke(
                                {
                                    "table_name": table_name,
                                    "table_columns": table_details.get("columns"),
                                    "primary_key": table_details.get("primary_key"),
                                    "related_tables_desc": table_details.get(
                                        "related_tables_desc"
                                    ),
                                    "relationship_desc": table_details.get(
                                        "relationship_desc"
                                    ),
                                }
                            )
                        ),
                        "metadata": {
                            k: ", ".join(detail) if type(detail) == list else detail
                            for k, detail in table_details.items()
                        },
                    }
                ),
            }
            for table_name, table_details in state["tables"].items()
        }
    }


def upsert_to_vector_database_node(state: PostgresQdrantState):
    upsert_collection.invoke(
        {
            "vector_store": state["vector_store"],
            "docs": [
                table_details.get("table_info_summary")
                for table_name, table_details in state["tables"].items()
            ],
        }
    )


def get_related_documents_node(
    state: PostgresQdrantState,
):
    state["vector_store"].as_retriever()
    return {
        "joined_related_documents": join_docs.invoke(
            {
                "docs": state["vector_store"].similarity_search(
                    query=state["question"],
                    k=state["similarity_doc_number"],
                )
            }
        )
    }


def generate_respective_sql_code_node(
    state: PostgresQdrantState,
):
    print(
        llm_model.invoke(
            input=sql_coder_from_rag.invoke(
                {
                    "question": state["question"],
                    "content": state["joined_related_documents"],
                }
            )
        )
    )
    return {
        "sql_code": llm_model.invoke(
            input=sql_coder_from_rag.invoke(
                {
                    "question": state["question"],
                    "content": state["joined_related_documents"],
                }
            )
        ).content
    }
