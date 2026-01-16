# Customer Segmentation for CRM Using Machine Learning

Groups customers into four segments based on purchasing behaviour.

## Segments

| Segment | Characteristics | Goal |
|---|---|---|
| **High-Value Customers** | High spend, moderate frequency | Retention & upsell |
| **Frequent Buyers** | High order volume, lower ticket | Loyalty rewards |
| **Occasional Buyers** | Low frequency, medium spend | Re-engagement |
| **New Customers** | Very recent, few purchases | Onboarding |

## Project Structure

```
customer_segmentation_ml/
├── main.py                  <- Run this
├── generate_dataset.py      <- Synthetic CRM dataset
├── clustering_pipeline.py   <- Feature engineering + K-Means
├── visualize.py             <- 7 charts
├── evaluate.py              <- Metrics + HTML report
├── data/
└── outputs/
    ├── plots/
    └── reports/
```

## Data Fields

| Field | Description |
|---|---|
| `purchase_frequency` | Orders per year |
| `total_spending` | Total revenue ($) |
| `avg_order_value` | Average order ($) |
| `days_since_last_purchase` | Recency |
| `loyalty_score` | RFM score (0-100) |

## ML Technique

K-Means Clustering with k-means++ initialisation.

- **Elbow method** for optimal k selection
- **Silhouette score** for cluster quality
- **Davies-Bouldin index** for compactness

## Quick Start

```bash
pip install -r requirements.txt
python main.py
start outputs/reports/segmentation_report.html
```

## Evaluation Metrics

| Metric | Target |
|---|---|
| Silhouette Score | >= 0.40 |
| Davies-Bouldin Index | <= 1.0 |
| Adjusted Rand Index | >= 0.80 |

## Tools

- Python 3.11+, Pandas, NumPy
- scikit-learn (KMeans, StandardScaler, PCA, metrics)
- Matplotlib
