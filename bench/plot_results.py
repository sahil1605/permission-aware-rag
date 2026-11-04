"""Draw the README chart from the benchmark CSVs.

    python bench/plot_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"

SURFACE = "#fcfcfb"
GRID = "#e4e3df"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
COLORS = {"post_filter": "#2a78d6", "pre_filter": "#eb6834", "folder_pre_filter": "#1baf7a"}
MARKERS = {"post_filter": "o", "pre_filter": "s", "folder_pre_filter": "D"}


def summarize(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["pct"] = df["access_fraction"] * 100
    return df.groupby(["strategy", "pct"]).agg(
        p95_ms=("total_ms", lambda s: np.percentile(s, 95)), recall=("recall", "mean")
    ).reset_index()


def style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 20, 50, 100], ["1%", "5%", "20%", "50%", "100%"])
    ax.minorticks_off()
    ax.set_xlabel("Share of documents the user can see", color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def line(ax, d: pd.DataFrame, strategy: str, y: str, label: str | None = None) -> None:
    ax.plot(
        d["pct"], d[y], color=COLORS[strategy], linewidth=2, marker=MARKERS[strategy], markersize=7,
        markeredgecolor=SURFACE, markeredgewidth=2, solid_capstyle="round", label=label,
    )


def main() -> None:
    uncapped = summarize(RESULTS / "no-list-limit-clustered.csv")
    capped = summarize(RESULTS / "default-limits-clustered.csv")

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    style(left, "p95 search latency (ListObjects uncapped)", "milliseconds")
    style(right, "Recall@10 (OpenFGA default limits)", "recall vs. exact top-10")

    # post_filter and folder_pre_filter end close together at 100%, so nudge their labels apart.
    label_dy = {"post_filter": 7, "pre_filter": 0, "folder_pre_filter": -7}
    for strategy in COLORS:
        d = uncapped[uncapped.strategy == strategy]
        line(left, d, strategy, "p95_ms", label=strategy)
        last = d.iloc[-1]
        left.annotate(f"{last.p95_ms:.0f} ms", (last.pct, last.p95_ms), xytext=(8, label_dy[strategy]),
                      textcoords="offset points", va="center", color=TEXT, fontsize=9)
        line(right, capped[capped.strategy == strategy], strategy, "recall")
    left.set_ylim(0, None)
    left.set_xlim(0.8, 160)
    right.set_ylim(-0.03, 1.08)
    right.set_xlim(0.8, 130)

    fig.legend(loc="upper center", ncol=3, frameon=False, labelcolor=TEXT, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, -0.04,
             "50,000 chunks in 200 folders, clustered embeddings, 50 queries per point, OpenFGA 1.21 on Postgres, Milvus Lite FLAT index, Apple M4.",
             color=TEXT_SECONDARY, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = RESULTS / "summary.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
