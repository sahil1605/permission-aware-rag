import numpy as np
import pytest

from parag import Strategy

K = 5


@pytest.mark.parametrize("strategy", list(Strategy))
def test_never_returns_documents_the_user_cannot_view(env, strategy):
    org, _, searcher = env
    queries = org.queries(25, np.random.default_rng(0))
    for user, access in org.users.items():
        allowed = set(org.authorized_docs(user).tolist())
        for q in queries:
            hits = searcher.search(access.subject, q, K, strategy).hits
            assert {h.id for h in hits} <= allowed, f"{strategy} leaked a document to {user}"


@pytest.mark.parametrize("strategy", [Strategy.PRE_FILTER, Strategy.FOLDER_PRE_FILTER])
def test_pre_filters_return_the_exact_top_k(env, strategy):
    org, _, searcher = env
    queries = org.queries(25, np.random.default_rng(1))
    for user, access in org.users.items():
        for q in queries:
            hits = searcher.search(access.subject, q, K, strategy).hits
            assert [h.id for h in hits] == org.ground_truth(user, q, K).tolist()


@pytest.mark.parametrize("strategy", list(Strategy))
def test_directly_shared_document_is_retrievable(env, strategy):
    org, _, searcher = env
    access = org.users["u5pct"]
    shared = int(access.direct_docs[0])
    hits = searcher.search(access.subject, org.vectors[shared], K, strategy).hits
    assert hits[0].id == shared


@pytest.mark.parametrize("strategy", list(Strategy))
def test_revoking_team_membership_removes_access(env, strategy):
    org, authz, searcher = env
    team = org.users["u25pct"].team
    membership = ("user:temp", "member", f"team:{team}")
    target = int(np.flatnonzero(np.isin(org.folder_of, org.users["u25pct"].folders))[0])

    authz.write_tuples([membership])
    try:
        hits = searcher.search("user:temp", org.vectors[target], K, strategy).hits
        assert hits and hits[0].id == target
    finally:
        authz.delete_tuples([membership])

    assert searcher.search("user:temp", org.vectors[target], K, strategy).hits == []


def test_user_with_no_access_gets_nothing(env):
    org, _, searcher = env
    for strategy in Strategy:
        assert searcher.search("user:nobody", org.vectors[0], K, strategy).hits == []
