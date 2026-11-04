"""Synthetic organization for benchmarks and tests.

Documents are spread evenly across folders. Each benchmark user can see a fixed
fraction of the folders through a team, plus a few documents shared with them
directly. Because the generator knows exactly who can see what, it can compute
the ground-truth top-k for every query.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from parag.authz import Authorizer, Tuple
from parag.store import ChunkStore

ORG = "organization:acme"


@dataclass
class UserAccess:
    name: str
    team: str | None
    folders: np.ndarray  # folder IDs visible through the team or organization
    direct_docs: np.ndarray  # documents shared with the user directly
    org_member: bool = False

    @property
    def subject(self) -> str:
        return f"user:{self.name}"


@dataclass
class SyntheticOrg:
    n_folders: int
    docs_per_folder: int
    vectors: np.ndarray  # (n_docs, dim), L2-normalized
    folder_of: np.ndarray  # (n_docs,) folder ID of each document
    users: dict[str, UserAccess] = field(default_factory=dict)
    centroids: np.ndarray | None = None  # only set for clustered data

    @property
    def n_docs(self) -> int:
        return len(self.folder_of)

    def authorized_docs(self, user: str) -> np.ndarray:
        access = self.users[user]
        in_folders = np.flatnonzero(np.isin(self.folder_of, access.folders))
        return np.union1d(in_folders, access.direct_docs)

    def ground_truth(self, user: str, query: np.ndarray, k: int) -> np.ndarray:
        """Exact top-k document IDs among the documents `user` can see."""
        allowed = self.authorized_docs(user)
        scores = self.vectors[allowed] @ query
        top = np.argsort(-scores)[:k]
        return allowed[top]

    def queries(self, n: int, rng: np.random.Generator) -> np.ndarray:
        dim = self.vectors.shape[1]
        if self.centroids is None:
            q = rng.standard_normal((n, dim))
        else:
            # Each query targets the topic of a random folder, which the user may not be able to see.
            topics = rng.integers(0, self.n_folders, size=n)
            q = self.centroids[topics] + 0.5 * rng.standard_normal((n, dim)) / np.sqrt(dim)
        return _normalize(q)

    def tuples(self) -> list[Tuple]:
        tuples: list[Tuple] = [
            (f"folder:{f}", "parent", f"document:{d}") for d, f in enumerate(self.folder_of.tolist())
        ]
        org_folders_added = False
        for access in self.users.values():
            if access.org_member:
                tuples.append((access.subject, "member", ORG))
                if not org_folders_added:
                    tuples += [(f"{ORG}#member", "viewer", f"folder:{f}") for f in range(self.n_folders)]
                    org_folders_added = True
            if access.team:
                tuples.append((access.subject, "member", f"team:{access.team}"))
                tuples += [(f"team:{access.team}#member", "viewer", f"folder:{f}") for f in access.folders.tolist()]
            tuples += [(access.subject, "direct_viewer", f"document:{d}") for d in access.direct_docs.tolist()]
        return tuples

    def load(self, authz: Authorizer, store: ChunkStore) -> None:
        authz.write_tuples(self.tuples())
        store.insert(
            [
                {"id": d, "folder_id": int(self.folder_of[d]), "text": "", "vector": self.vectors[d].tolist()}
                for d in range(self.n_docs)
            ]
        )


def build_org(
    *,
    n_folders: int,
    docs_per_folder: int,
    access_fractions: list[float],
    direct_shares: int,
    dim: int,
    distribution: str,
    seed: int = 7,
) -> SyntheticOrg:
    rng = np.random.default_rng(seed)
    n_docs = n_folders * docs_per_folder
    folder_of = np.repeat(np.arange(n_folders), docs_per_folder)

    centroids = None
    if distribution == "uniform":
        vectors = rng.standard_normal((n_docs, dim))
    elif distribution == "clustered":
        # Documents in the same folder share a topic, as they would in a real team space.
        centroids = _normalize(rng.standard_normal((n_folders, dim)))
        vectors = centroids[folder_of] + 0.5 * rng.standard_normal((n_docs, dim)) / np.sqrt(dim)
    else:
        raise ValueError(f"unknown distribution: {distribution}")

    org = SyntheticOrg(n_folders, docs_per_folder, _normalize(vectors).astype(np.float32), folder_of, centroids=centroids)

    for fraction in access_fractions:
        name = f"u{fraction * 100:g}pct"
        if fraction >= 1.0:
            org.users[name] = UserAccess(name, None, np.arange(n_folders), np.array([], dtype=int), org_member=True)
            continue
        folders = np.sort(rng.choice(n_folders, size=max(1, round(fraction * n_folders)), replace=False))
        hidden = np.flatnonzero(~np.isin(folder_of, folders))
        direct = np.sort(rng.choice(hidden, size=min(direct_shares, len(hidden)), replace=False))
        org.users[name] = UserAccess(name, f"team_{name}", folders, direct)

    return org


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)
