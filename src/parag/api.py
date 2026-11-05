"""HTTP API over the demo store. Run demo/run_demo.py first to load data.

    uvicorn parag.api:app

Serves POST /search and /ask, and the same answers to other agents over A2A
(agent card at /.well-known/agent-card.json, JSON-RPC at /a2a).
Set OLLAMA_MODEL=none to return the permitted passages instead of an LLM answer.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header
from pydantic import BaseModel, Field

from parag.a2a_agent import RagAnswerer, mount_a2a
from parag.authz import Authorizer
from parag.rag import Embedder
from parag.retriever import PermissionAwareSearcher, Strategy
from parag.store import ChunkStore

STATE = Path(os.getenv("PARAG_STATE", "data/demo_state.json"))
PUBLIC_URL = os.getenv("PARAG_PUBLIC_URL", "http://127.0.0.1:8000")


class SearchRequest(BaseModel):
    question: str
    k: int = Field(default=4, ge=1, le=50)
    strategy: Strategy = Strategy.FOLDER_PRE_FILTER


class AskRequest(BaseModel):
    question: str


components: dict = {}


def _llm():
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    if model.lower() == "none":
        return None
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = json.loads(STATE.read_text())
    embedder = Embedder()
    store = ChunkStore(state["milvus_uri"], collection=state["collection"], dim=embedder.dim)
    store.create()
    authz = Authorizer.connect(state["fga_url"], state["store_id"], state["model_id"])
    searcher = PermissionAwareSearcher(store, authz)
    components.update(embedder=embedder, searcher=searcher, answerer=RagAnswerer(searcher, embedder, _llm()))
    yield
    searcher.close()
    store.close()


app = FastAPI(title="permission-aware-rag", lifespan=lifespan)
mount_a2a(app, answer=lambda user, question: components["answerer"](user, question), base_url=PUBLIC_URL)


# Demo only: the caller is identified by an X-User header. A real deployment must
# take the user from a verified token (e.g. a JWT checked by the API gateway).
def _subject(x_user: str) -> str:
    return f"user:{x_user}"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search")
def search(body: SearchRequest, x_user: str = Header()) -> dict:
    vector = components["embedder"].embed_query(body.question)
    result = components["searcher"].search(_subject(x_user), vector, body.k, body.strategy)
    return {
        "hits": [{"doc_id": h.id, "score": round(h.score, 4), "text": h.text} for h in result.hits],
        "stats": {
            "strategy": result.stats.strategy,
            "authz_ms": round(result.stats.authz_ms, 2),
            "vector_ms": round(result.stats.vector_ms, 2),
        },
    }


@app.post("/ask")
def ask(body: AskRequest, x_user: str = Header()) -> dict:
    answer = components["answerer"](_subject(x_user), body.question)
    return {"answer": answer.text, "sources": answer.sources}
