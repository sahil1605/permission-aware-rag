"""Milvus collection holding chunk embeddings, the chunk text, and the folder each chunk belongs to."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pymilvus import DataType, MilvusClient

# Milvus rejects searches with limit above 16384.
MAX_SEARCH_LIMIT = 16384


@dataclass(frozen=True)
class Hit:
    id: int
    folder_id: int
    score: float
    text: str = ""


class ChunkStore:
    def __init__(self, uri: str, collection: str = "chunks", dim: int = 384):
        self._client = MilvusClient(uri)
        self.collection = collection
        self.dim = dim

    def create(self, drop_existing: bool = False) -> None:
        if self._client.has_collection(self.collection):
            if not drop_existing:
                self._client.load_collection(self.collection)
                return
            self._client.drop_collection(self.collection)

        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        # Denormalized copy of the document's OpenFGA `parent` tuple, used by the
        # folder-level pre-filter. It must be updated whenever a document moves.
        schema.add_field("folder_id", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)

        index = self._client.prepare_index_params()
        # FLAT is exact search, so any recall loss in the benchmark comes from
        # the authorization strategy rather than from approximate search.
        index.add_index("vector", index_type="FLAT", metric_type="COSINE")
        index.add_index("folder_id", index_type="INVERTED")
        self._client.create_collection(self.collection, schema=schema, index_params=index)

    def insert(self, rows: Sequence[dict], batch_size: int = 5000) -> None:
        for start in range(0, len(rows), batch_size):
            self._client.insert(self.collection, list(rows[start : start + batch_size]))

    def search(self, vector: Sequence[float], limit: int, filter: str = "") -> list[Hit]:
        results = self._client.search(
            self.collection,
            data=[list(vector)],
            limit=min(limit, MAX_SEARCH_LIMIT),
            filter=filter,
            output_fields=["folder_id", "text"],
        )
        return [
            Hit(
                id=r["id"],
                folder_id=r["entity"]["folder_id"],
                score=r["distance"],
                text=r["entity"].get("text", ""),
            )
            for r in results[0]
        ]

    def close(self) -> None:
        self._client.close()


def in_filter(field: str, values: Sequence[int]) -> str:
    """Build a Milvus `field in [...]` expression from integer IDs."""
    return f"{field} in [{','.join(str(int(v)) for v in values)}]"
