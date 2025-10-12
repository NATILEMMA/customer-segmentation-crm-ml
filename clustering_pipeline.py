import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

RANDOM_SEED = 42
FINAL_K = 4
DATA_PATH = "data/crm_customers.csv"

def run_pipeline():
    df = pd.read_csv(DATA_PATH)
    features = ["purchase_frequency", "total_spending", "days_since_last_purchase"]
    X = df[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=FINAL_K, random_state=RANDOM_SEED, n_init=10)
    model.fit(X_scaled)
    df["cluster_id"] = model.labels_
    print(df["cluster_id"].value_counts())
    df.to_csv("outputs/segmented.csv", index=False)

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    run_pipeline()
