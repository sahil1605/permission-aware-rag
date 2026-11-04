"""Load the Acme Robotics demo corpus and show that each user only retrieves what they may see.

    python demo/run_demo.py            # retrieval only
    python demo/run_demo.py --ask      # also generate answers with a local Ollama model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parag import Authorizer, ChunkStore, PermissionAwareSearcher
from parag.rag import Embedder, PermissionAwareRetriever, build_rag_chain

CORPUS = Path(__file__).parent / "corpus.json"
STATE = Path("data/demo_state.json")
ORG = "organization:acme"


def corpus_tuples(corpus: dict) -> list[tuple[str, str, str]]:
    tuples = []
    for folder_id, folder in corpus["folders"].items():
        tuples += [(viewer, "viewer", f"folder:{folder_id}") for viewer in folder["viewers"]]
    for doc in corpus["documents"]:
        tuples.append((f"folder:{doc['folder']}", "parent", f"document:{doc['id']}"))
    for name, user in corpus["users"].items():
        tuples.append((f"user:{name}", "member", ORG))
        tuples += [(f"user:{name}", "member", f"team:{team}") for team in user["teams"]]
        tuples += [(f"user:{name}", "direct_viewer", f"document:{d}") for d in user.get("direct_documents", [])]
    return tuples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fga-url", default="http://127.0.0.1:8080")
    parser.add_argument("--milvus-uri", default="data/demo.db")
    parser.add_argument("--ask", action="store_true", help="generate answers with Ollama")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model for --ask")
    args = parser.parse_args()

    corpus = json.loads(CORPUS.read_text())
    embedder = Embedder()

    authz = Authorizer.bootstrap(args.fga_url, "acme-demo")
    authz.write_tuples(corpus_tuples(corpus))

    store = ChunkStore(args.milvus_uri, collection="demo", dim=embedder.dim)
    store.create(drop_existing=True)
    docs = corpus["documents"]
    vectors = embedder.embed_documents([d["text"] for d in docs])
    store.insert([{"id": d["id"], "folder_id": d["folder"], "text": d["text"], "vector": v} for d, v in zip(docs, vectors)])

    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(
        json.dumps(
            {
                "fga_url": args.fga_url,
                "store_id": authz.store_id,
                "model_id": authz.model_id,
                "milvus_uri": args.milvus_uri,
                "collection": "demo",
            },
            indent=2,
        )
    )

    searcher = PermissionAwareSearcher(store, authz)
    users = list(corpus["users"])
    for question in corpus["questions"]:
        print(f"\nQ: {question}")
        for user in users:
            hit = searcher.search(f"user:{user}", embedder.embed_query(question), k=1).hits[0]
            print(f"  {user:>6} top result: [doc {hit.id:>2}] {hit.text[:70]}...")

    if args.ask:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=args.model, temperature=0)
        question = corpus["questions"][0]
        print(f"\nAnswers to: {question}")
        for user in ["bob", "alice", "erin"]:
            retriever = PermissionAwareRetriever(searcher=searcher, embedder=embedder, user=f"user:{user}", k=3)
            answer = build_rag_chain(retriever, llm).invoke(question)
            print(f"  {user:>6}: {answer.strip()}")

    searcher.close()
    store.close()
    print(f"\nDemo state written to {STATE}; start the API with: uvicorn parag.api:app")


if __name__ == "__main__":
    main()
