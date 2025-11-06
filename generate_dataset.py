import numpy as np
import pandas as pd
import random
import os
from datetime import datetime, timedelta

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

PRODUCT_CATEGORIES = ["Electronics", "Office Supplies", "Furniture", "Software", "Services"]
LOCATIONS = ["Riyadh", "Jeddah", "Dammam", "Dubai", "Cairo", "Amman", "Kuwait City"]

def _random_date(min_days, max_days):
    delta = random.randint(min_days, max_days)
    return (datetime.today() - timedelta(days=delta)).strftime("%Y-%m-%d")

SEGMENT_PROFILES = [
    {"label": "High-Value",  "n": 150, "freq": (15,40), "spend": (50000,200000),  "days": (1,60)},
    {"label": "Frequent",    "n": 200, "freq": (30,80), "spend": (5000,30000),    "days": (1,30)},
    {"label": "Occasional",  "n": 300, "freq": (3,12),  "spend": (1000,15000),    "days": (60,300)},
    {"label": "New",         "n": 150, "freq": (1,4),   "spend": (200,3000),      "days": (1,20)},
]

def generate_crm_dataset(output_path="data/crm_customers.csv"):
    records = []
    cid = 1001
    for profile in SEGMENT_PROFILES:
        for _ in range(profile["n"]):
            freq    = random.randint(*profile["freq"])
            spend   = round(np.random.uniform(*profile["spend"]), 2)
            days    = random.randint(*profile["days"])
            aov     = round(spend / max(freq, 1), 2)
            loyalty = round(min(100, (100 - days * 0.4 + freq * 1.5 + spend / 2000) / 3), 1)
            records.append({
                "customer_id": f"CUST-{cid:05d}",
                "purchase_frequency": freq,
                "total_spending": spend,
                "avg_order_value": aov,
                "days_since_last_purchase": days,
                "last_purchase_date": _random_date(days, days),
                "top_product_category": random.choice(PRODUCT_CATEGORIES),
                "location": random.choice(LOCATIONS),
                "loyalty_score": loyalty,
                "true_segment": profile["label"],
            })
            cid += 1
    df = pd.DataFrame(records).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} customer records -> {output_path}")
    return df

if __name__ == "__main__":
    generate_crm_dataset()
