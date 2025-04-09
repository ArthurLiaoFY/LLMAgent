import json

from langchain_ollama import ChatOllama, OllamaEmbeddings

with open("config.json") as f:
    config = json.loads(f.read())
llm_vector_size = config.get("llm_model", {}).get("vector_size", "5120")
llm_model = ChatOllama(
    model=config.get("llm_model", {}).get("model_name", "deepseek-r1:14b"),
    temperature=0,
)
llm_embd = OllamaEmbeddings(
    model=config.get("llm_model", {}).get("model_name", "deepseek-r1:14b"),
)
