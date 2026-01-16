"""
Visualisation module for the Customer Segmentation project.

Generates publication-quality charts saved to outputs/plots/:
  1. Elbow curve  (inertia vs k)
  2. Silhouette score bar chart
  3. PCA scatter plot of clusters
  4. Radar / spider chart of segment profiles
  5. Feature distribution box plots per segment
  6. Spending vs frequency scatter (annotated)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm

PLOT_DIR = "outputs/plots"
SEGMENT_COLORS = {
    "High-Value Customers": "#E63946",
    "Frequent Buyers":      "#457B9D",
    "Occasional Buyers":    "#2A9D8F",
    "New Customers":        "#E9C46A",
}
FIGSIZE_WIDE   = (14, 6)
FIGSIZE_SQUARE = (10, 8)
DPI            = 150


def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {path}")


# ── 1. Elbow Curve ─────────────────────────────────────────────────────────────
def plot_elbow(elbow_results: list[dict], final_k: int = 4) -> None:
    ks       = [r["k"] for r in elbow_results]
    inertias = [r["inertia"] for r in elbow_results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, inertias, "o-", color="#457B9D", linewidth=2.5, markersize=8)
    ax.axvline(final_k, color="#E63946", linestyle="--", linewidth=1.5,
               label=f"Chosen k = {final_k}")
    ax.set_title("Elbow Method — Optimal Number of Clusters", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "01_elbow_curve.png")


# ── 2. Silhouette Bar Chart ────────────────────────────────────────────────────
def plot_silhouette_bars(elbow_results: list[dict], final_k: int = 4) -> None:
    ks   = [r["k"] for r in elbow_results]
    sils = [r["silhouette"] for r in elbow_results]
    colors = ["#E63946" if k == final_k else "#A8DADC" for k in ks]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(ks, sils, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_title("Silhouette Score by Number of Clusters", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Silhouette Score (higher = better)", fontsize=12)
    ax.set_ylim(0, max(sils) * 1.15)
    ax.set_xticks(ks)
    ax.grid(axis="y", alpha=0.3)
    patch = mpatches.Patch(color="#E63946", label=f"Chosen k = {final_k}")
    ax.legend(handles=[patch], fontsize=11)
    fig.tight_layout()
    _save(fig, "02_silhouette_scores.png")


# ── 3. PCA Cluster Scatter ─────────────────────────────────────────────────────
def plot_pca_clusters(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    segments = df["segment"].unique()
    for seg in segments:
        mask = df["segment"] == seg
        ax.scatter(
            df.loc[mask, "pca_x"],
            df.loc[mask, "pca_y"],
            label=seg,
            color=SEGMENT_COLORS.get(seg, "grey"),
            alpha=0.7,
            edgecolors="white",
            linewidths=0.4,
            s=55,
        )

    ax.set_title("Customer Segments — PCA 2-D Projection", fontsize=14, fontweight="bold")
    ax.set_xlabel("Principal Component 1", fontsize=11)
    ax.set_ylabel("Principal Component 2", fontsize=11)
    ax.legend(title="Segment", fontsize=10, title_fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "03_pca_clusters.png")


# ── 4. Radar / Spider Chart ────────────────────────────────────────────────────
def plot_radar(df: pd.DataFrame) -> None:
    radar_metrics = [
        "purchase_frequency",
        "total_spending",
        "avg_order_value",
        "loyalty_score",
        "days_since_last_purchase",
    ]
    labels_display = [
        "Purchase\nFrequency",
        "Total\nSpending",
        "Avg Order\nValue",
        "Loyalty\nScore",
        "Recency\n(days)",
    ]

    # Normalise each metric 0–1 across the dataset for fair radar comparison
    norm_df = df.copy()
    for col in radar_metrics:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max != col_min:
            norm_df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            norm_df[col] = 0.5

    # Invert recency so that higher = more recent (better)
    norm_df["days_since_last_purchase"] = 1 - norm_df["days_since_last_purchase"]

    segment_means = norm_df.groupby("segment")[radar_metrics].mean()

    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for seg, row in segment_means.iterrows():
        values = row.tolist() + row.tolist()[:1]
        color  = SEGMENT_COLORS.get(seg, "grey")
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=seg)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_display, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title("Segment Profile Radar Chart\n(normalised metrics)",
                 fontsize=13, fontweight="bold", y=1.10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    fig.tight_layout()
    _save(fig, "04_radar_segments.png")


# ── 5. Box Plots ───────────────────────────────────────────────────────────────
def plot_feature_distributions(df: pd.DataFrame) -> None:
    features = {
        "Total Spending ($)":          "total_spending",
        "Purchase Frequency":          "purchase_frequency",
        "Avg Order Value ($)":         "avg_order_value",
        "Days Since Last Purchase":    "days_since_last_purchase",
        "Loyalty Score":               "loyalty_score",
    }

    segment_order = [
        "High-Value Customers",
        "Frequent Buyers",
        "Occasional Buyers",
        "New Customers",
    ]
    segment_order = [s for s in segment_order if s in df["segment"].unique()]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for idx, (title, col) in enumerate(features.items()):
        ax = axes_flat[idx]
        data_by_seg = [df.loc[df["segment"] == seg, col].dropna() for seg in segment_order]
        bp = ax.boxplot(
            data_by_seg,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        for patch, seg in zip(bp["boxes"], segment_order):
            patch.set_facecolor(SEGMENT_COLORS.get(seg, "grey"))
            patch.set_alpha(0.8)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(1, len(segment_order) + 1))
        ax.set_xticklabels(
            [s.replace(" ", "\n") for s in segment_order],
            fontsize=8,
        )
        ax.grid(axis="y", alpha=0.3)

    # Use last subplot for a legend
    ax_legend = axes_flat[-1]
    ax_legend.axis("off")
    handles = [
        mpatches.Patch(color=SEGMENT_COLORS[s], label=s)
        for s in segment_order
    ]
    ax_legend.legend(handles=handles, loc="center", fontsize=11,
                     title="Customer Segments", title_fontsize=12, frameon=False)

    fig.suptitle("Feature Distributions by Customer Segment",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "05_feature_distributions.png")


# ── 6. Spending vs Frequency Annotated Scatter ─────────────────────────────────
def plot_spend_vs_frequency(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for seg in df["segment"].unique():
        mask = df["segment"] == seg
        ax.scatter(
            df.loc[mask, "purchase_frequency"],
            df.loc[mask, "total_spending"],
            label=seg,
            color=SEGMENT_COLORS.get(seg, "grey"),
            alpha=0.65,
            edgecolors="white",
            linewidths=0.3,
            s=50,
        )

    # Annotate quadrant labels
    ax.text(0.97, 0.95, "High Value\n(Big Spenders)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#E63946", style="italic")
    ax.text(0.97, 0.10, "Frequent Low-Spend",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#457B9D", style="italic")

    ax.set_title("Customer Spending vs Purchase Frequency by Segment",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Purchase Frequency (orders / year)", fontsize=11)
    ax.set_ylabel("Total Spending ($)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(title="Segment", fontsize=10, title_fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "06_spend_vs_frequency.png")


# ── 7. Segment Size Pie Chart ──────────────────────────────────────────────────
def plot_segment_pie(df: pd.DataFrame) -> None:
    counts = df["segment"].value_counts()
    colors = [SEGMENT_COLORS.get(s, "grey") for s in counts.index]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=None,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
        t.set_color("white")

    ax.legend(
        wedges,
        [f"{seg}  ({cnt})" for seg, cnt in zip(counts.index, counts.values)],
        title="Segment (count)",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=10,
    )
    ax.set_title("Customer Segment Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "07_segment_distribution_pie.png")


# ── Master function ────────────────────────────────────────────────────────────
def generate_all_plots(df: pd.DataFrame, elbow_results: list[dict],
                       final_k: int = 4) -> None:
    """Generate and save every chart in sequence."""
    print("\n[Visualisation] Generating all plots …")
    plot_elbow(elbow_results, final_k)
    plot_silhouette_bars(elbow_results, final_k)
    plot_pca_clusters(df)
    plot_radar(df)
    plot_feature_distributions(df)
    plot_spend_vs_frequency(df)
    plot_segment_pie(df)
    print(f"[Visualisation] All plots saved to {PLOT_DIR}/\n")


if __name__ == "__main__":
    # Quick standalone test: load output from pipeline and re-plot
    df = pd.read_csv("outputs/segmented_customers.csv")
    elbow = pd.read_csv("outputs/reports/elbow_data.csv").to_dict("records")
    generate_all_plots(df, elbow)
