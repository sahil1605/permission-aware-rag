"""A2A (Agent2Agent protocol) front end for the permission-aware retriever.

Other agents send a question as an A2A message and get back an answer built only from
documents the *end user* behind the request may view, plus the IDs of those documents.

The end user is taken from the request's authenticated identity, never from the message
text or metadata, so a calling agent can't widen access by asking "as" someone else.
In this demo the identity is the X-User header, like the REST endpoints. A real
deployment must take it from a verified token instead (for example an OAuth access token
obtained through token exchange for the user the calling agent acts on behalf of).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from a2a.helpers.proto_helpers import new_data_part, new_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    APIKeySecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
    UnsupportedOperationError,
)
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol
from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from starlette.authentication import AuthCredentials, AuthenticationBackend, SimpleUser
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import HTTPConnection

from parag.rag import NOT_FOUND, PROMPT, Embedder, PermissionAwareRetriever, format_docs
from parag.retriever import PermissionAwareSearcher

USER_HEADER = "X-User"
# OpenFGA reads "user:x#member" as a userset, so only plain IDs are accepted.
_VALID_USER_ID = re.compile(r"[A-Za-z0-9_.@-]{1,64}")

MISSING_IDENTITY = (
    f"Who is this question for? Send the end user's ID in the {USER_HEADER} header; "
    "answers depend on what that user may see."
)


@dataclass
class Answer:
    text: str
    sources: list[int] = field(default_factory=list)


# (OpenFGA subject such as "user:bob", question) -> Answer
AnswerFn = Callable[[str, str], Answer]


class RagAnswerer:
    """Retrieves with the user's permissions, then answers with the LLM if one is given.

    Without an LLM it returns the permitted passages themselves, which is often what a
    calling agent wants: it can write the answer with its own model.
    """

    def __init__(
        self,
        searcher: PermissionAwareSearcher,
        embedder: Embedder,
        llm: BaseChatModel | None = None,
        k: int = 3,
    ):
        self._searcher = searcher
        self._embedder = embedder
        self._llm = llm
        self._k = k

    def __call__(self, user: str, question: str) -> Answer:
        retriever = PermissionAwareRetriever(searcher=self._searcher, embedder=self._embedder, user=user, k=self._k)
        docs = retriever.invoke(question)
        sources = [d.metadata["doc_id"] for d in docs]
        if self._llm is None:
            return Answer(format_docs(docs) if docs else NOT_FOUND, sources)
        chain = PROMPT | self._llm | StrOutputParser()
        return Answer(chain.invoke({"context": format_docs(docs), "question": question}).strip(), sources)


class PermissionAwareRagExecutor(AgentExecutor):
    """Answers each A2A message with one reply message (no long-running task)."""

    def __init__(self, answer: AnswerFn):
        self._answer = answer

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user = context.call_context.user
        if not user.is_authenticated:
            await event_queue.enqueue_event(self._reply(context, Answer(MISSING_IDENTITY)))
            return
        question = context.get_user_input().strip()
        # Retrieval and the LLM call are blocking, so keep them off the event loop.
        answer = await asyncio.to_thread(self._answer, f"user:{user.user_name}", question)
        await event_queue.enqueue_event(self._reply(context, answer))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Replies are a single message, so there is never a running task to cancel.
        raise UnsupportedOperationError()

    @staticmethod
    def _reply(context: RequestContext, answer: Answer):
        # OpenFGA object IDs as strings: A2A data parts would turn bare numbers into floats.
        sources = [f"document:{doc_id}" for doc_id in answer.sources]
        return new_message(
            parts=[new_text_part(answer.text), new_data_part({"sources": sources})],
            context_id=context.context_id,
            task_id=context.task_id,
        )


class HeaderUserBackend(AuthenticationBackend):
    """Demo authentication: trusts the X-User header, exactly like the REST endpoints."""

    async def authenticate(self, conn: HTTPConnection):
        user = conn.headers.get(USER_HEADER, "")
        if not _VALID_USER_ID.fullmatch(user):
            return None
        return AuthCredentials(["authenticated"]), SimpleUser(user)


def build_agent_card(rpc_url: str) -> AgentCard:
    return AgentCard(
        name="Permission-aware RAG agent",
        description=(
            "Answers questions from company documents, using only the documents the end user "
            "may view (OpenFGA permissions applied inside the Milvus vector search)."
        ),
        version="0.2.0",
        supported_interfaces=[
            AgentInterface(
                url=rpc_url,
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_1_0,
            )
        ],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/json"],
        security_schemes={
            "end_user": SecurityScheme(
                api_key_security_scheme=APIKeySecurityScheme(
                    location="header",
                    name=USER_HEADER,
                    description="Demo only: ID of the end user the calling agent acts for.",
                )
            )
        },
        security_requirements=[SecurityRequirement(schemes={"end_user": StringList()})],
        skills=[
            AgentSkill(
                id="ask_documents",
                name="Answer from permitted documents",
                description=(
                    "Answers a question from the documents the end user may view and returns "
                    'the documents used as a JSON data part, e.g. {"sources": ["document:21"]}.'
                ),
                tags=["rag", "authorization", "openfga"],
                examples=["What is the approved budget for Project Falcon?"],
            )
        ],
    )


def mount_a2a(app: FastAPI, answer: AnswerFn, base_url: str, rpc_path: str = "/a2a") -> AgentCard:
    """Serve the agent card at /.well-known/agent-card.json and JSON-RPC at `rpc_path`."""
    card = build_agent_card(base_url.rstrip("/") + rpc_path)
    handler = DefaultRequestHandler(
        agent_executor=PermissionAwareRagExecutor(answer),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url=rpc_path),
    )
    app.add_middleware(AuthenticationMiddleware, backend=HeaderUserBackend())
    return card
