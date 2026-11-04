"""LangChain integration: a permission-aware retriever and a RAG chain that uses it."""

from __future__ import annotations

from fastembed import TextEmbedding
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from pydantic import ConfigDict

from parag.retriever import PermissionAwareSearcher, Strategy


class Embedder:
    """Local embeddings via fastembed (ONNX, no GPU or API key needed)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model = TextEmbedding(model_name)
        self.dim = len(self.embed_query("probe"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class PermissionAwareRetriever(BaseRetriever):
    """Retriever bound to one user; it can only return chunks that user may view."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    searcher: PermissionAwareSearcher
    embedder: Embedder
    user: str
    k: int = 4
    strategy: Strategy = Strategy.FOLDER_PRE_FILTER

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        result = self.searcher.search(self.user, self.embedder.embed_query(query), self.k, self.strategy)
        return [
            Document(
                page_content=h.text,
                metadata={"doc_id": h.id, "folder_id": h.folder_id, "score": h.score},
            )
            for h in result.hits
        ]


# The fallback answer is deliberately vague: saying "you don't have access" would
# confirm that a matching document exists.
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the context below. Cite sources as [doc N]. "
            "If the context does not contain the answer, reply exactly: "
            '"I could not find that in the documents available to you."',
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "(no documents)"
    return "\n\n".join(f"[doc {d.metadata['doc_id']}] {d.page_content}" for d in docs)


def build_rag_chain(retriever: PermissionAwareRetriever, llm: BaseChatModel) -> Runnable:
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
