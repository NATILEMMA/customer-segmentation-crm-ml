import numpy as np
import pandas as pd
import random
import os
from datetime import datetime, timedelta

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

LOCATIONS = ["Riyadh", "Jeddah", "Dammam", "Dubai", "Cairo"]

def generate_crm_dataset(output_path="data/crm_customers.csv"):
    records = []
    for i in range(500):
        records.append({
            "customer_id": f"CUST-{1000+i:05d}",
            "purchase_frequency": random.randint(1, 60),
            "total_spending": round(random.uniform(500, 180000), 2),
            "days_since_last_purchase": random.randint(1, 300),
            "location": random.choice(LOCATIONS),
        })
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records -> {output_path}")
    return df

if __name__ == "__main__":
    generate_crm_dataset()
