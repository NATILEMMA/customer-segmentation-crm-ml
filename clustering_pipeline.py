"""
Customer Segmentation Pipeline — K-Means Clustering

Full ML pipeline:
  1. Load / generate CRM dataset
  2. Feature engineering & preprocessing
  3. Optimal-k selection via Elbow method + Silhouette analysis
  4. Final K-Means model training
  5. Segment labelling & business interpretation
  6. Export enriched dataset with assigned segments
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
K_RANGE       = range(2, 9)          # evaluate 2 to 8 clusters
FINAL_K       = 4                    # business decision: 4 customer segments
DATA_PATH     = "data/crm_customers.csv"
OUTPUT_DIR    = "outputs"

NUMERIC_FEATURES = [
    "purchase_frequency",
    "total_spending",
    "avg_order_value",
    "days_since_last_purchase",
    "loyalty_score",
]

CATEGORICAL_FEATURES = [
    "top_product_category",
    "location",
]

# Segment name mapping after inspecting cluster centroids
SEGMENT_NAMES = {
    0: "High-Value Customers",
    1: "Frequent Buyers",
    2: "Occasional Buyers",
    3: "New Customers",
}

SEGMENT_STRATEGIES = {
    "High-Value Customers": (
        "Premium loyalty programmes, dedicated account managers, "
        "early access to new products, upsell high-margin items."
    ),
    "Frequent Buyers": (
        "Reward cards, bundle offers, subscription discounts, "
        "personalised reorder reminders."
    ),
    "Occasional Buyers": (
        "Re-engagement email campaigns, seasonal promotions, "
        "win-back offers, targeted cross-sell."
    ),
    "New Customers": (
        "Welcome discounts, onboarding guides, follow-up calls, "
        "nurture sequences to drive second purchase."
    ),
}
# ──────────────────────────────────────────────────────────────────────────────


def load_or_generate_data(path: str) -> pd.DataFrame:
    """Load existing dataset or generate a fresh synthetic one."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"[Data] Loaded {len(df)} records from {path}")
    else:
        from generate_dataset import generate_crm_dataset
        df = generate_crm_dataset(output_path=path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived RFM-style features and encode categoricals.

    RFM = Recency, Frequency, Monetary — the industry standard
    framework for customer value analysis.
    """
    df = df.copy()

    # Recency: invert days so higher = more recent
    df["recency_score"] = 1 / (df["days_since_last_purchase"] + 1)

    # Monetary density: spend per visit
    df["spend_per_visit"] = df["total_spending"] / (df["purchase_frequency"] + 1)

    # Encode categorical columns as ordinal integers
    le = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Select and scale features for clustering."""
    derived = ["recency_score", "spend_per_visit"]
    encoded = [f"{c}_encoded" for c in CATEGORICAL_FEATURES if f"{c}_encoded" in df.columns]

    feature_cols = NUMERIC_FEATURES + derived + encoded
    feature_cols = [c for c in feature_cols if c in df.columns]

    X_raw = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, feature_cols, scaler


def find_optimal_k(X: np.ndarray) -> dict:
    """
    Evaluate cluster counts 2–8 using:
      - Inertia (Elbow method)
      - Silhouette Score  (higher = better separated clusters)
      - Davies-Bouldin Index (lower = more compact clusters)
    """
    results = []
    print("\n[Elbow & Silhouette Analysis]")
    print(f"{'K':>4}  {'Inertia':>12}  {'Silhouette':>12}  {'Davies-Bouldin':>16}")
    print("-" * 50)

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X)

        inertia   = km.inertia_
        sil       = silhouette_score(X, labels)
        db        = davies_bouldin_score(X, labels)

        results.append({"k": k, "inertia": inertia, "silhouette": sil, "davies_bouldin": db})
        print(f"{k:>4}  {inertia:>12.1f}  {sil:>12.4f}  {db:>16.4f}")

    best_sil_k = max(results, key=lambda r: r["silhouette"])["k"]
    print(f"\n[Optimal K by Silhouette] → k = {best_sil_k}")
    return results


def train_final_model(X: np.ndarray, k: int = FINAL_K) -> KMeans:
    """Train the final K-Means model with the chosen k."""
    model = KMeans(
        n_clusters=k,
        init="k-means++",   # smart centroid initialisation
        n_init=20,           # run 20 times, pick best result
        max_iter=500,
        random_state=RANDOM_SEED,
    )
    model.fit(X)
    print(f"\n[Model] K-Means trained  k={k}  inertia={model.inertia_:.1f}")
    return model


def assign_segment_labels(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Map raw cluster IDs to meaningful business segment names.

    Strategy: rank clusters by average total_spending descending,
    then assign High-Value → Frequent → Occasional → New.
    This keeps labels stable regardless of KMeans random centroid order.
    """
    df = df.copy()
    df["cluster_id"] = labels

    cluster_spending = (
        df.groupby("cluster_id")["total_spending"].mean().sort_values(ascending=False)
    )
    rank_to_name = {
        rank: name
        for rank, name in zip(range(FINAL_K), SEGMENT_NAMES.values())
    }
    cluster_to_name = {
        cluster_id: rank_to_name[rank]
        for rank, cluster_id in enumerate(cluster_spending.index)
    }

    df["segment"] = df["cluster_id"].map(cluster_to_name)
    df["marketing_strategy"] = df["segment"].map(SEGMENT_STRATEGIES)
    return df


def pca_coordinates(X: np.ndarray) -> np.ndarray:
    """Reduce to 2-D for scatter-plot visualisation."""
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    return pca.fit_transform(X)


def print_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Print a business-readable summary table of each segment."""
    summary = df.groupby("segment").agg(
        customer_count=("customer_id", "count"),
        avg_purchase_frequency=("purchase_frequency", "mean"),
        avg_total_spending=("total_spending", "mean"),
        avg_order_value=("avg_order_value", "mean"),
        avg_days_since_last_purchase=("days_since_last_purchase", "mean"),
        avg_loyalty_score=("loyalty_score", "mean"),
        top_category=("top_product_category", lambda x: x.mode()[0]),
        top_location=("location", lambda x: x.mode()[0]),
    ).round(2)

    print("\n" + "=" * 70)
    print("  CUSTOMER SEGMENT SUMMARY")
    print("=" * 70)
    for seg, row in summary.iterrows():
        print(f"\n  [{seg}]")
        print(f"    Customers              : {row['customer_count']}")
        print(f"    Avg Purchase Frequency : {row['avg_purchase_frequency']:.1f} orders/year")
        print(f"    Avg Total Spending     : ${row['avg_total_spending']:,.0f}")
        print(f"    Avg Order Value        : ${row['avg_order_value']:,.0f}")
        print(f"    Avg Days Since Last    : {row['avg_days_since_last_purchase']:.0f} days")
        print(f"    Avg Loyalty Score      : {row['avg_loyalty_score']:.1f}/100")
        print(f"    Top Category           : {row['top_category']}")
        print(f"    Top Location           : {row['top_location']}")
        print(f"    Strategy               : {SEGMENT_STRATEGIES[seg]}")

    print("\n" + "=" * 70)
    return summary


def run_pipeline() -> pd.DataFrame:
    """End-to-end pipeline execution."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1 — Data
    df = load_or_generate_data(DATA_PATH)

    # 2 — Feature engineering
    df_eng = engineer_features(df)
    X, feature_cols, scaler = build_feature_matrix(df_eng)
    print(f"[Features] Using {len(feature_cols)} features: {feature_cols}")

    # 3 — Optimal-k analysis
    elbow_results = find_optimal_k(X)

    # 4 — Train final model
    model = train_final_model(X, k=FINAL_K)
    labels = model.labels_

    # 5 — Evaluate final model
    sil   = silhouette_score(X, labels)
    db    = davies_bouldin_score(X, labels)
    print(f"[Evaluation] Final model — Silhouette: {sil:.4f}  Davies-Bouldin: {db:.4f}")

    # 6 — Assign human-readable segment names
    df_result = assign_segment_labels(df_eng, labels)

    # 7 — 2-D PCA coordinates (for plots)
    coords = pca_coordinates(X)
    df_result["pca_x"] = coords[:, 0]
    df_result["pca_y"] = coords[:, 1]

    # 8 — Summary
    summary = print_segment_summary(df_result)

    # 9 — Save outputs
    out_csv = os.path.join(OUTPUT_DIR, "segmented_customers.csv")
    df_result.to_csv(out_csv, index=False)
    print(f"\n[Output] Segmented dataset saved → {out_csv}")

    elbow_df = pd.DataFrame(elbow_results)
    elbow_df.to_csv(os.path.join(OUTPUT_DIR, "reports", "elbow_data.csv"), index=False)

    summary.to_csv(os.path.join(OUTPUT_DIR, "reports", "segment_summary.csv"))
    print(f"[Output] Reports saved → {OUTPUT_DIR}/reports/")

    return df_result, model, X, elbow_results, sil


if __name__ == "__main__":
    run_pipeline()
