"""Vector search that only returns chunks the requesting user is allowed to see.

Three strategies are implemented so they can be compared on the same data:

* post_filter: search first, then drop results the user can't see using BatchCheck.
  If too few survive, search deeper and check again.
* pre_filter: ask OpenFGA for every document the user can view (ListObjects) and
  restrict the vector search to those IDs.
* folder_pre_filter: ask OpenFGA for the folders the user can view plus the documents
  shared with them directly, and filter on the `folder_id` stored with each chunk.
  This relies on the model defining `viewer` as `direct_viewer or viewer from parent`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum

from parag.authz import Authorizer
from parag.store import ChunkStore, Hit, in_filter


class Strategy(StrEnum):
    POST_FILTER = "post_filter"
    PRE_FILTER = "pre_filter"
    FOLDER_PRE_FILTER = "folder_pre_filter"


@dataclass
class SearchStats:
    strategy: Strategy
    authz_ms: float = 0.0
    vector_ms: float = 0.0
    authz_calls: int = 0
    candidates_checked: int = 0
    # Number of IDs returned by ListObjects (pre-filter strategies only).
    listed_objects: int = 0

    @property
    def total_ms(self) -> float:
        return self.authz_ms + self.vector_ms


@dataclass
class SearchResult:
    hits: list[Hit]
    stats: SearchStats = field(repr=False)


class _Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._start) * 1000


class PermissionAwareSearcher:
    def __init__(
        self,
        store: ChunkStore,
        authz: Authorizer,
        *,
        overfetch: int = 4,
        max_candidates: int = 1000,
    ):
        """
        overfetch: post_filter starts by fetching k * overfetch candidates.
        max_candidates: post_filter stops expanding the search after this many candidates,
            so a user who can see very little doesn't trigger thousands of checks.
        """
        self._store = store
        self._authz = authz
        self._overfetch = overfetch
        self._max_candidates = max_candidates
        self._pool = ThreadPoolExecutor(max_workers=2)

    def search(
        self,
        user: str,
        vector: Sequence[float],
        k: int = 10,
        strategy: Strategy = Strategy.FOLDER_PRE_FILTER,
    ) -> SearchResult:
        if strategy is Strategy.POST_FILTER:
            return self._post_filter(user, vector, k)
        if strategy is Strategy.PRE_FILTER:
            return self._pre_filter(user, vector, k)
        if strategy is Strategy.FOLDER_PRE_FILTER:
            return self._folder_pre_filter(user, vector, k)
        raise ValueError(f"unknown strategy: {strategy}")

    def _post_filter(self, user: str, vector: Sequence[float], k: int) -> SearchResult:
        stats = SearchStats(Strategy.POST_FILTER)
        allowed_hits: list[Hit] = []
        checked: set[int] = set()
        limit = min(k * self._overfetch, self._max_candidates)

        while True:
            with _Timer() as t:
                hits = self._store.search(vector, limit)
            stats.vector_ms += t.ms

            # Each round re-runs the search with a bigger limit; only check the new tail.
            new_hits = [h for h in hits if h.id not in checked]
            with _Timer() as t:
                allowed = self._authz.batch_check(user, "viewer", [f"document:{h.id}" for h in new_hits])
            stats.authz_ms += t.ms
            stats.authz_calls += 1
            stats.candidates_checked += len(new_hits)
            checked.update(h.id for h in new_hits)
            allowed_hits.extend(h for h in new_hits if f"document:{h.id}" in allowed)

            exhausted = len(hits) < limit
            if len(allowed_hits) >= k or exhausted or limit >= self._max_candidates:
                break
            limit = min(limit * 4, self._max_candidates)

        return SearchResult(allowed_hits[:k], stats)

    def _pre_filter(self, user: str, vector: Sequence[float], k: int) -> SearchResult:
        stats = SearchStats(Strategy.PRE_FILTER)
        with _Timer() as t:
            documents = self._authz.list_objects(user, "viewer", "document")
        stats.authz_ms += t.ms
        stats.authz_calls += 1
        stats.listed_objects = len(documents)
        if not documents:
            return SearchResult([], stats)

        doc_ids = [_object_id(d) for d in documents]
        with _Timer() as t:
            hits = self._store.search(vector, k, filter=in_filter("id", doc_ids))
        stats.vector_ms += t.ms
        return SearchResult(hits, stats)

    def _folder_pre_filter(self, user: str, vector: Sequence[float], k: int) -> SearchResult:
        stats = SearchStats(Strategy.FOLDER_PRE_FILTER)
        with _Timer() as t:
            folders_future = self._pool.submit(self._authz.list_objects, user, "viewer", "folder")
            direct_future = self._pool.submit(self._authz.list_objects, user, "direct_viewer", "document")
            folders, direct = folders_future.result(), direct_future.result()
        stats.authz_ms += t.ms
        stats.authz_calls += 2
        stats.listed_objects = len(folders) + len(direct)

        clauses = []
        if folders:
            clauses.append(in_filter("folder_id", [_object_id(f) for f in folders]))
        if direct:
            clauses.append(in_filter("id", [_object_id(d) for d in direct]))
        if not clauses:
            return SearchResult([], stats)

        with _Timer() as t:
            hits = self._store.search(vector, k, filter=" or ".join(clauses))
        stats.vector_ms += t.ms
        return SearchResult(hits, stats)

    def close(self) -> None:
        self._pool.shutdown()


def _object_id(obj: str) -> int:
    """'document:42' -> 42"""
    return int(obj.split(":", 1)[1])
