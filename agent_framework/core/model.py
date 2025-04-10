import json

from langchain_ollama import ChatOllama, OllamaEmbeddings

with open("config.json") as f:
    config = json.loads(f.read())
llm_model = ChatOllama(
    model=config.get("llm_model", {}).get("model_name"),
    temperature=0,
)
llm_coder = ChatOllama(
    model=config.get("llm_coder", {}).get("model_name"),
    temperature=0,
)
embd_model = OllamaEmbeddings(
    model=config.get("embd_model", {}).get("model_name"),
)

embd_size = config.get("embd_model", {}).get("embd_size")