"""Benchmark the three permission-filtering strategies on a synthetic organization.

For every user and strategy it records latency, recall against the exact top-k among
the documents the user may see, and leaks (results the user may not see).

    python bench/run_benchmark.py --distribution clustered --label default-limits
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

from parag import Authorizer, ChunkStore, PermissionAwareSearcher, Strategy
from parag.synthetic import build_org

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fga-url", default="http://127.0.0.1:8080")
    parser.add_argument("--milvus-uri", default="data/bench.db")
    parser.add_argument("--label", required=True, help="name for the output files, e.g. the OpenFGA config used")
    parser.add_argument("--distribution", choices=["uniform", "clustered"], default="clustered")
    parser.add_argument("--folders", type=int, default=200)
    parser.add_argument("--docs-per-folder", type=int, default=250)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--fractions", default="0.01,0.05,0.2,0.5,1.0")
    parser.add_argument("--direct-shares", type=int, default=20)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    org = build_org(
        n_folders=args.folders,
        docs_per_folder=args.docs_per_folder,
        access_fractions=[float(f) for f in args.fractions.split(",")],
        direct_shares=args.direct_shares,
        dim=args.dim,
        distribution=args.distribution,
    )
    print(f"{org.n_docs} documents in {org.n_folders} folders, {len(org.users)} users, {args.distribution} vectors")

    authz = Authorizer.bootstrap(args.fga_url, f"bench-{args.label}-{args.distribution}")
    store = ChunkStore(args.milvus_uri, collection=f"bench_{args.distribution}", dim=args.dim)
    store.create(drop_existing=True)
    start = time.perf_counter()
    org.load(authz, store)
    print(f"loaded {len(org.tuples())} tuples and {org.n_docs} vectors in {time.perf_counter() - start:.1f}s")

    searcher = PermissionAwareSearcher(store, authz)
    rng = np.random.default_rng(42)
    queries = org.queries(args.queries, rng)
    rows = []

    for user, access in org.users.items():
        allowed = set(org.authorized_docs(user).tolist())
        fraction = len(allowed) / org.n_docs
        truths = [set(org.ground_truth(user, q, args.k).tolist()) for q in queries]
        for strategy in Strategy:
            for q in queries[:3]:  # warm-up, not recorded
                searcher.search(access.subject, q, args.k, strategy)
            for i, q in enumerate(queries):
                result = searcher.search(access.subject, q, args.k, strategy)
                ids = [h.id for h in result.hits]
                leaks = sum(d not in allowed for d in ids)
                # Cross-check the generator's ground truth against OpenFGA itself on a sample.
                if i < 2 and ids:
                    allowed_by_fga = authz.batch_check(access.subject, "viewer", [f"document:{d}" for d in ids])
                    if len(allowed_by_fga) != len(ids) - leaks:
                        raise RuntimeError(f"OpenFGA and ground truth disagree for {user}")
                truth = truths[i]
                rows.append(
                    {
                        "label": args.label,
                        "distribution": args.distribution,
                        "user": user,
                        "access_fraction": round(fraction, 4),
                        "strategy": strategy.value,
                        "query": i,
                        "total_ms": round(result.stats.total_ms, 3),
                        "authz_ms": round(result.stats.authz_ms, 3),
                        "vector_ms": round(result.stats.vector_ms, 3),
                        "authz_calls": result.stats.authz_calls,
                        "candidates_checked": result.stats.candidates_checked,
                        "listed_objects": result.stats.listed_objects,
                        "hits": len(ids),
                        "recall": len(truth & set(ids)) / len(truth),
                        "leaks": leaks,
                    }
                )
            print(_summary_line(rows[-args.queries :]))

    searcher.close()
    store.close()
    authz.close()

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"{args.label}-{args.distribution}"
    with open(RESULTS_DIR / f"{stem}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS_DIR / f"{stem}.md").write_text(_summary_table(rows))
    print(f"wrote {RESULTS_DIR / stem}.csv and .md")


def _percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def _summary_line(rows: list[dict]) -> str:
    r = rows[0]
    lat = [x["total_ms"] for x in rows]
    return (
        f"{r['user']:>9} {r['strategy']:>18}  p50 {_percentile(lat, 50):7.1f}ms  p95 {_percentile(lat, 95):7.1f}ms  "
        f"recall {statistics.mean(x['recall'] for x in rows):.3f}  leaks {sum(x['leaks'] for x in rows)}"
    )


def _summary_table(rows: list[dict]) -> str:
    lines = [
        "| user can see | strategy | p50 ms | p95 ms | mean recall@k | min recall@k | candidates checked | IDs listed | leaks |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["access_fraction"], r["strategy"]), []).append(r)
    for (fraction, strategy), g in sorted(groups.items(), key=lambda kv: (kv[0][0], list(Strategy).index(kv[0][1]))):
        lat = [x["total_ms"] for x in g]
        lines.append(
            f"| {fraction:.1%} | {strategy} | {_percentile(lat, 50):.1f} | {_percentile(lat, 95):.1f} "
            f"| {statistics.mean(x['recall'] for x in g):.3f} | {min(x['recall'] for x in g):.2f} "
            f"| {statistics.mean(x['candidates_checked'] for x in g):.0f} "
            f"| {statistics.mean(x['listed_objects'] for x in g):.0f} | {sum(x['leaks'] for x in g)} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
