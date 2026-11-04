"""Thin wrapper over the OpenFGA SDK exposing only the calls the retriever needs."""

from __future__ import annotations

from collections.abc import Iterable

from openfga_sdk import ClientConfiguration
from openfga_sdk.client.models import (
    ClientBatchCheckItem,
    ClientBatchCheckRequest,
    ClientCheckRequest,
    ClientListObjectsRequest,
    ClientTuple,
    ClientWriteRequest,
)
from openfga_sdk.client.models.write_transaction_opts import WriteTransactionOpts
from openfga_sdk.models.create_store_request import CreateStoreRequest
from openfga_sdk.sync import OpenFgaClient

# JSON form of model.fga, which is what the OpenFGA API accepts.
AUTHORIZATION_MODEL = {
    "schema_version": "1.1",
    "type_definitions": [
        {"type": "user"},
        {
            "type": "organization",
            "relations": {"member": {"this": {}}},
            "metadata": {"relations": {"member": {"directly_related_user_types": [{"type": "user"}]}}},
        },
        {
            "type": "team",
            "relations": {"member": {"this": {}}},
            "metadata": {"relations": {"member": {"directly_related_user_types": [{"type": "user"}]}}},
        },
        {
            "type": "folder",
            "relations": {"viewer": {"this": {}}},
            "metadata": {
                "relations": {
                    "viewer": {
                        "directly_related_user_types": [
                            {"type": "user"},
                            {"type": "team", "relation": "member"},
                            {"type": "organization", "relation": "member"},
                        ]
                    }
                }
            },
        },
        {
            "type": "document",
            "relations": {
                "parent": {"this": {}},
                "direct_viewer": {"this": {}},
                "viewer": {
                    "union": {
                        "child": [
                            {"computedUserset": {"relation": "direct_viewer"}},
                            {
                                "tupleToUserset": {
                                    "tupleset": {"relation": "parent"},
                                    "computedUserset": {"relation": "viewer"},
                                }
                            },
                        ]
                    }
                },
            },
            "metadata": {
                "relations": {
                    "parent": {"directly_related_user_types": [{"type": "folder"}]},
                    "direct_viewer": {
                        "directly_related_user_types": [
                            {"type": "user"},
                            {"type": "team", "relation": "member"},
                        ]
                    },
                    "viewer": {"directly_related_user_types": []},
                }
            },
        },
    ],
}

Tuple = tuple[str, str, str]  # (user, relation, object)

# OpenFGA rejects Write requests with more than 100 tuples by default.
_MAX_TUPLES_PER_WRITE = 100


class Authorizer:
    def __init__(self, client: OpenFgaClient):
        self._client = client

    @classmethod
    def bootstrap(cls, api_url: str, store_name: str) -> Authorizer:
        """Create a fresh store, write the authorization model, and return a client bound to both."""
        client = OpenFgaClient(ClientConfiguration(api_url=api_url))
        store = client.create_store(CreateStoreRequest(name=store_name))
        client.set_store_id(store.id)
        model = client.write_authorization_model(AUTHORIZATION_MODEL)
        client.set_authorization_model_id(model.authorization_model_id)
        return cls(client)

    @classmethod
    def connect(cls, api_url: str, store_id: str, model_id: str) -> Authorizer:
        client = OpenFgaClient(
            ClientConfiguration(api_url=api_url, store_id=store_id, authorization_model_id=model_id)
        )
        return cls(client)

    @property
    def store_id(self) -> str:
        return self._client.get_store_id()

    @property
    def model_id(self) -> str:
        return self._client.get_authorization_model_id()

    def write_tuples(self, tuples: Iterable[Tuple]) -> None:
        self._write(writes=[ClientTuple(user=u, relation=r, object=o) for u, r, o in tuples])

    def delete_tuples(self, tuples: Iterable[Tuple]) -> None:
        self._write(deletes=[ClientTuple(user=u, relation=r, object=o) for u, r, o in tuples])

    def _write(self, writes: list[ClientTuple] | None = None, deletes: list[ClientTuple] | None = None) -> None:
        # Non-transactional mode lets the SDK split large loads into parallel chunks.
        options = {
            "transaction": WriteTransactionOpts(
                disabled=True, max_per_chunk=_MAX_TUPLES_PER_WRITE, max_parallel_requests=10
            )
        }
        response = self._client.write(ClientWriteRequest(writes=writes, deletes=deletes), options)
        failed = [r for r in (response.writes or []) + (response.deletes or []) if not r.success]
        if failed:
            raise RuntimeError(f"{len(failed)} tuple writes failed, first error: {failed[0].error}")

    def check(self, user: str, relation: str, obj: str) -> bool:
        return self._client.check(ClientCheckRequest(user=user, relation=relation, object=obj)).allowed

    def batch_check(self, user: str, relation: str, objects: list[str]) -> set[str]:
        """Return the subset of `objects` that `user` has `relation` on."""
        if not objects:
            return set()
        request = ClientBatchCheckRequest(
            checks=[ClientBatchCheckItem(user=user, relation=relation, object=o) for o in objects]
        )
        response = self._client.batch_check(request)
        for r in response.result:
            if r.error is not None:
                raise RuntimeError(f"BatchCheck failed for {r.request.object}: {r.error}")
        return {r.request.object for r in response.result if r.allowed}

    def list_objects(self, user: str, relation: str, type_: str) -> list[str]:
        """List objects of `type_` the user has `relation` on.

        The server caps the result size (--listObjects-max-results, default 1000) and
        returns whatever it found when the deadline expires, so the list may be partial.
        """
        response = self._client.list_objects(ClientListObjectsRequest(user=user, relation=relation, type=type_))
        return response.objects

    def close(self) -> None:
        self._client.close()
