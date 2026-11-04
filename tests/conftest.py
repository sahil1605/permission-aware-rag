import os
import uuid

import httpx
import pytest

from parag import Authorizer, ChunkStore, PermissionAwareSearcher
from parag.synthetic import build_org

FGA_URL = os.getenv("FGA_API_URL", "http://127.0.0.1:8080")


@pytest.fixture(scope="session")
def fga_url() -> str:
    try:
        httpx.get(f"{FGA_URL}/healthz", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip(f"OpenFGA is not running at {FGA_URL} (start it with `openfga run`)")
    return FGA_URL


@pytest.fixture(scope="module")
def env(fga_url, tmp_path_factory):
    """A small synthetic org loaded into a fresh OpenFGA store and Milvus Lite collection."""
    org = build_org(
        n_folders=20,
        docs_per_folder=30,
        access_fractions=[0.05, 0.25, 1.0],
        direct_shares=3,
        dim=32,
        distribution="clustered",
    )
    authz = Authorizer.bootstrap(fga_url, f"test-{uuid.uuid4().hex[:8]}")
    store = ChunkStore(str(tmp_path_factory.mktemp("milvus") / "test.db"), dim=32)
    store.create()
    org.load(authz, store)
    searcher = PermissionAwareSearcher(store, authz)
    yield org, authz, searcher
    searcher.close()
    store.close()
    authz.close()
