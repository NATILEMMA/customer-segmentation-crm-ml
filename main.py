"""
main.py — Customer Segmentation Entry Point

Run this file to execute the complete pipeline:
  python main.py

Steps:
  1. Generate synthetic CRM dataset (if not already present)
  2. Feature engineering & preprocessing
  3. Elbow + Silhouette analysis
  4. Train K-Means (k=4)
  5. Assign business segment labels
  6. Generate all charts
  7. Generate HTML + JSON evaluation report
"""

import time
from generate_dataset    import generate_crm_dataset
from clustering_pipeline import (
    load_or_generate_data,
    engineer_features,
    build_feature_matrix,
    find_optimal_k,
    train_final_model,
    assign_segment_labels,
    pca_coordinates,
    print_segment_summary,
    DATA_PATH,
    OUTPUT_DIR,
    FINAL_K,
)
from visualize  import generate_all_plots
from evaluate   import run_evaluation

import os
import pandas as pd


def main() -> None:
    start = time.time()
    print("=" * 65)
    print("  Customer Segmentation Using K-Means Clustering")
    print("  CRM-Style Dataset — ML Portfolio Project")
    print("=" * 65)

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        generate_crm_dataset(output_path=DATA_PATH)

    df = load_or_generate_data(DATA_PATH)

    # ── Step 2: Feature Engineering ───────────────────────────────────────────
    df_eng = engineer_features(df)
    X, feature_cols, scaler = build_feature_matrix(df_eng)
    print(f"\n[Pipeline] {len(feature_cols)} features after engineering:")
    print(f"           {feature_cols}")

    # ── Step 3: Optimal-k Analysis ────────────────────────────────────────────
    elbow_results = find_optimal_k(X)

    # ── Step 4: Train Final Model ─────────────────────────────────────────────
    model  = train_final_model(X, k=FINAL_K)
    labels = model.labels_

    # ── Step 5: Segment Assignment ────────────────────────────────────────────
    df_result = assign_segment_labels(df_eng, labels)

    # Add PCA coordinates for plotting
    coords = pca_coordinates(X)
    df_result["pca_x"] = coords[:, 0]
    df_result["pca_y"] = coords[:, 1]

    # ── Step 6: Summary ───────────────────────────────────────────────────────
    print_segment_summary(df_result)

    # Save enriched dataset
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, "segmented_customers.csv")
    df_result.to_csv(out_csv, index=False)
    print(f"[Pipeline] Segmented dataset → {out_csv}")

    # ── Step 7: Visualisations ────────────────────────────────────────────────
    generate_all_plots(df_result, elbow_results, final_k=FINAL_K)

    # ── Step 8: Evaluation Report ─────────────────────────────────────────────
    run_evaluation(df_result, X, feature_cols)

    elapsed = time.time() - start
    print("\n" + "=" * 65)
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Results → {OUTPUT_DIR}/")
    print(f"  Open    → {OUTPUT_DIR}/reports/segmentation_report.html")
    print("=" * 65)


if __name__ == "__main__":
    main()
