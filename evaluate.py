"""
Evaluation & Business Reporting Module

Generates a structured HTML report plus a plain-text evaluation log:
  - Clustering quality metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
  - Segment-level statistics
  - Comparison of predicted segments vs ground-truth labels (optional)
  - Business action recommendations per segment
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

OUTPUT_DIR   = "outputs"
REPORTS_DIR  = os.path.join(OUTPUT_DIR, "reports")

# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Return the three standard unsupervised clustering quality metrics."""
    return {
        "silhouette_score":        round(silhouette_score(X, labels), 4),
        "davies_bouldin_score":    round(davies_bouldin_score(X, labels), 4),
        "calinski_harabasz_score": round(calinski_harabasz_score(X, labels), 4),
        "n_clusters":              int(len(np.unique(labels))),
        "n_samples":               int(len(labels)),
    }


def compute_ground_truth_metrics(df: pd.DataFrame) -> dict | None:
    """
    Compare ML segments vs the true_segment column (simulation only).
    In a real production scenario this column would not exist.
    """
    if "true_segment" not in df.columns or "segment" not in df.columns:
        return None

    # Encode strings to ints for sklearn metrics
    true_labels = pd.factorize(df["true_segment"])[0]
    pred_labels = pd.factorize(df["segment"])[0]

    return {
        "adjusted_rand_index":        round(adjusted_rand_score(true_labels, pred_labels), 4),
        "normalized_mutual_info":     round(normalized_mutual_info_score(true_labels, pred_labels), 4),
    }


def compute_segment_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate key KPIs per segment for the report table."""
    stats = df.groupby("segment").agg(
        customers=("customer_id", "count"),
        pct=("customer_id", lambda x: round(len(x) / len(df) * 100, 1)),
        avg_frequency=("purchase_frequency", "mean"),
        avg_spending=("total_spending", "mean"),
        avg_order_value=("avg_order_value", "mean"),
        avg_recency_days=("days_since_last_purchase", "mean"),
        avg_loyalty=("loyalty_score", "mean"),
        top_category=("top_product_category", lambda x: x.mode()[0]),
    ).round(2)
    return stats


# ── HTML Report ───────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Customer Segmentation Report</title>
<style>
  body  {{ font-family: 'Segoe UI', Arial, sans-serif; background:#f4f6f9; color:#333; margin:0; padding:0; }}
  .wrap {{ max-width:1100px; margin:40px auto; padding:0 20px; }}
  h1   {{ color:#1a3c5e; border-bottom:3px solid #457B9D; padding-bottom:10px; }}
  h2   {{ color:#457B9D; margin-top:40px; }}
  .meta {{ background:#fff; border-radius:8px; padding:16px 24px; box-shadow:0 2px 6px rgba(0,0,0,.08); margin-bottom:28px; }}
  .meta p {{ margin:4px 0; font-size:14px; color:#555; }}
  .metric-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin-bottom:32px; }}
  .metric-card {{ background:#fff; border-radius:8px; padding:20px; text-align:center; box-shadow:0 2px 6px rgba(0,0,0,.08); }}
  .metric-card .label {{ font-size:13px; color:#888; margin-bottom:6px; }}
  .metric-card .value {{ font-size:26px; font-weight:700; color:#1a3c5e; }}
  .metric-card .hint  {{ font-size:11px; color:#aaa; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:8px; overflow:hidden;
            box-shadow:0 2px 6px rgba(0,0,0,.08); margin-bottom:32px; }}
  th   {{ background:#457B9D; color:#fff; padding:12px 14px; text-align:left; font-size:13px; }}
  td   {{ padding:10px 14px; font-size:13px; border-bottom:1px solid #eef0f3; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:nth-child(even) td {{ background:#f9fafb; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; color:#fff; }}
  .High-Value   {{ background:#E63946; }}
  .Frequent     {{ background:#457B9D; }}
  .Occasional   {{ background:#2A9D8F; }}
  .New          {{ background:#E9A020; color:#333; }}
  .strategy {{ font-size:12px; color:#555; font-style:italic; }}
  footer {{ text-align:center; font-size:12px; color:#aaa; margin:40px 0 20px; }}
  .plots {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; margin-bottom:32px; }}
  .plot-card {{ background:#fff; border-radius:8px; overflow:hidden; box-shadow:0 2px 6px rgba(0,0,0,.08); }}
  .plot-card img {{ width:100%; display:block; }}
  .plot-card p {{ padding:10px 14px; margin:0; font-size:12px; color:#666; text-align:center; }}
  .gt-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Customer Segmentation Report</h1>
  <div class="meta">
    <p><strong>Generated:</strong> {timestamp}</p>
    <p><strong>Dataset:</strong> {n_samples} customers</p>
    <p><strong>Algorithm:</strong> K-Means (k-means++ init, 20 restarts)</p>
    <p><strong>Number of Segments:</strong> {n_clusters}</p>
    <p><strong>Features Used:</strong> {features}</p>
  </div>

  <h2>Clustering Quality Metrics</h2>
  <div class="metric-grid">
    <div class="metric-card">
      <div class="label">Silhouette Score</div>
      <div class="value">{silhouette}</div>
      <div class="hint">Range −1 to 1 &nbsp;|&nbsp; Higher = better</div>
    </div>
    <div class="metric-card">
      <div class="label">Davies-Bouldin Index</div>
      <div class="value">{davies_bouldin}</div>
      <div class="hint">Lower = more compact clusters</div>
    </div>
    <div class="metric-card">
      <div class="label">Calinski-Harabasz Score</div>
      <div class="value">{calinski_harabasz}</div>
      <div class="hint">Higher = better defined clusters</div>
    </div>
    {gt_cards}
  </div>

  <h2>Segment Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Segment</th>
        <th>Customers</th>
        <th>Share</th>
        <th>Avg Frequency</th>
        <th>Avg Spending</th>
        <th>Avg Order Value</th>
        <th>Avg Recency (days)</th>
        <th>Loyalty Score</th>
        <th>Top Category</th>
      </tr>
    </thead>
    <tbody>
      {segment_rows}
    </tbody>
  </table>

  <h2>Marketing Strategies</h2>
  <table>
    <thead><tr><th>Segment</th><th>Recommended Action</th></tr></thead>
    <tbody>{strategy_rows}</tbody>
  </table>

  <h2>Visualisations</h2>
  <div class="plots">
    {plot_cards}
  </div>

  <footer>Customer Segmentation ML Project &nbsp;|&nbsp; K-Means Clustering &nbsp;|&nbsp; Generated {timestamp}</footer>
</div>
</body>
</html>
"""

STRATEGIES = {
    "High-Value Customers": "Premium loyalty programmes, dedicated account managers, early access to new products, upsell high-margin items.",
    "Frequent Buyers":      "Reward cards, bundle offers, subscription discounts, personalised reorder reminders.",
    "Occasional Buyers":    "Re-engagement email campaigns, seasonal promotions, win-back offers, targeted cross-sell.",
    "New Customers":        "Welcome discounts, onboarding guides, follow-up calls, nurture sequences to drive second purchase.",
}

BADGE_CLASS = {
    "High-Value Customers": "High-Value",
    "Frequent Buyers":      "Frequent",
    "Occasional Buyers":    "Occasional",
    "New Customers":        "New",
}

PLOT_META = [
    ("01_elbow_curve.png",          "Elbow Curve — choosing optimal k"),
    ("02_silhouette_scores.png",    "Silhouette Scores by k"),
    ("03_pca_clusters.png",         "PCA 2-D Cluster Projection"),
    ("04_radar_segments.png",       "Segment Profile Radar"),
    ("05_feature_distributions.png","Feature Distribution Box Plots"),
    ("06_spend_vs_frequency.png",   "Spending vs Purchase Frequency"),
    ("07_segment_distribution_pie.png", "Segment Size Distribution"),
]


def _build_segment_rows(stats: pd.DataFrame) -> str:
    rows = []
    for seg, row in stats.iterrows():
        badge_cls = BADGE_CLASS.get(seg, "")
        rows.append(
            f"<tr>"
            f"<td><span class='badge {badge_cls}'>{seg}</span></td>"
            f"<td>{int(row['customers'])}</td>"
            f"<td>{row['pct']}%</td>"
            f"<td>{row['avg_frequency']:.1f}</td>"
            f"<td>${row['avg_spending']:,.0f}</td>"
            f"<td>${row['avg_order_value']:,.0f}</td>"
            f"<td>{row['avg_recency_days']:.0f}</td>"
            f"<td>{row['avg_loyalty']:.1f}</td>"
            f"<td>{row['top_category']}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _build_strategy_rows(stats: pd.DataFrame) -> str:
    rows = []
    for seg in stats.index:
        badge_cls = BADGE_CLASS.get(seg, "")
        strategy  = STRATEGIES.get(seg, "—")
        rows.append(
            f"<tr>"
            f"<td><span class='badge {badge_cls}'>{seg}</span></td>"
            f"<td class='strategy'>{strategy}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _build_plot_cards() -> str:
    cards = []
    for filename, caption in PLOT_META:
        rel_path = f"../plots/{filename}"
        cards.append(
            f"<div class='plot-card'>"
            f"<img src='{rel_path}' alt='{caption}'/>"
            f"<p>{caption}</p>"
            f"</div>"
        )
    return "\n".join(cards)


def generate_html_report(
    df: pd.DataFrame,
    X: np.ndarray,
    metrics: dict,
    stats: pd.DataFrame,
    gt_metrics: dict | None,
    feature_cols: list[str],
) -> str:
    """Render and save the HTML report. Returns the file path."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    gt_cards = ""
    if gt_metrics:
        gt_cards = (
            f"<div class='metric-card'>"
            f"<div class='label'>Adjusted Rand Index</div>"
            f"<div class='value'>{gt_metrics['adjusted_rand_index']}</div>"
            f"<div class='hint'>vs ground-truth labels (simulation only)</div>"
            f"</div>"
            f"<div class='metric-card'>"
            f"<div class='label'>Norm. Mutual Info</div>"
            f"<div class='value'>{gt_metrics['normalized_mutual_info']}</div>"
            f"<div class='hint'>vs ground-truth labels</div>"
            f"</div>"
        )

    html = _HTML_TEMPLATE.format(
        timestamp        = datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_samples        = metrics["n_samples"],
        n_clusters       = metrics["n_clusters"],
        features         = ", ".join(feature_cols),
        silhouette       = metrics["silhouette_score"],
        davies_bouldin   = metrics["davies_bouldin_score"],
        calinski_harabasz= metrics["calinski_harabasz_score"],
        gt_cards         = gt_cards,
        segment_rows     = _build_segment_rows(stats),
        strategy_rows    = _build_strategy_rows(stats),
        plot_cards       = _build_plot_cards(),
    )

    path = os.path.join(REPORTS_DIR, "segmentation_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Report] HTML report saved → {path}")
    return path


def generate_json_summary(metrics: dict, gt_metrics: dict | None, stats: pd.DataFrame) -> str:
    """Save a machine-readable JSON metrics file."""
    payload = {
        "generated_at":      datetime.now().isoformat(),
        "clustering_metrics": metrics,
        "ground_truth_metrics": gt_metrics,
        "segment_stats":     stats.reset_index().to_dict(orient="records"),
    }
    path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[Report] JSON metrics saved → {path}")
    return path


def run_evaluation(df: pd.DataFrame, X: np.ndarray, feature_cols: list[str]) -> None:
    """Full evaluation flow called from main.py after the pipeline."""
    labels = df["cluster_id"].values

    metrics    = compute_clustering_metrics(X, labels)
    gt_metrics = compute_ground_truth_metrics(df)
    stats      = compute_segment_stats(df)

    print("\n[Evaluation] Clustering Quality")
    print(f"  Silhouette Score        : {metrics['silhouette_score']}  (good ≥ 0.40)")
    print(f"  Davies-Bouldin Index    : {metrics['davies_bouldin_score']}  (good ≤ 1.0)")
    print(f"  Calinski-Harabasz Score : {metrics['calinski_harabasz_score']}")

    if gt_metrics:
        print("\n[Evaluation] vs Ground-Truth Labels (simulation)")
        print(f"  Adjusted Rand Index     : {gt_metrics['adjusted_rand_index']}")
        print(f"  Norm. Mutual Info       : {gt_metrics['normalized_mutual_info']}")

    generate_html_report(df, X, metrics, stats, gt_metrics, feature_cols)
    generate_json_summary(metrics, gt_metrics, stats)


if __name__ == "__main__":
    df       = pd.read_csv("outputs/segmented_customers.csv")
    # Rebuild X from saved CSV for standalone use
    from clustering_pipeline import engineer_features, build_feature_matrix
    df_eng   = engineer_features(df)
    X, fcols, _ = build_feature_matrix(df_eng)
    run_evaluation(df, X, fcols)
