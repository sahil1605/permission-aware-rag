"""HTTP API over the demo store. Run demo/run_demo.py first to load data.

    uvicorn parag.api:app
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from parag.authz import Authorizer
from parag.rag import PROMPT, Embedder, PermissionAwareRetriever, format_docs
from parag.retriever import PermissionAwareSearcher, Strategy
from parag.store import ChunkStore

STATE = Path(os.getenv("PARAG_STATE", "data/demo_state.json"))


class SearchRequest(BaseModel):
    question: str
    k: int = Field(default=4, ge=1, le=50)
    strategy: Strategy = Strategy.FOLDER_PRE_FILTER


class AskRequest(BaseModel):
    question: str


components: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = json.loads(STATE.read_text())
    embedder = Embedder()
    store = ChunkStore(state["milvus_uri"], collection=state["collection"], dim=embedder.dim)
    store.create()
    authz = Authorizer.connect(state["fga_url"], state["store_id"], state["model_id"])
    components.update(embedder=embedder, searcher=PermissionAwareSearcher(store, authz))
    yield
    components["searcher"].close()
    store.close()


app = FastAPI(title="permission-aware-rag", lifespan=lifespan)


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
    from langchain_ollama import ChatOllama

    retriever = PermissionAwareRetriever(
        searcher=components["searcher"], embedder=components["embedder"], user=_subject(x_user), k=3
    )
    docs = retriever.invoke(body.question)
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=0)
    answer = (PROMPT | llm | StrOutputParser()).invoke({"context": format_docs(docs), "question": body.question})
    return {"answer": answer.strip(), "sources": [d.metadata["doc_id"] for d in docs]}
