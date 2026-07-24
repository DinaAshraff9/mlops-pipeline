"""
Generate a sample fraud detection dataset for testing the pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 10_000

# Simulate credit card transaction features
df = pd.DataFrame({
    "amount":           np.random.exponential(scale=100, size=N).round(2),
    "hour_of_day":      np.random.randint(0, 24, size=N),
    "merchant_category": np.random.randint(0, 20, size=N),
    "distance_from_home": np.random.exponential(scale=50, size=N).round(2),
    "n_transactions_1h": np.random.poisson(3, size=N),
    "n_transactions_24h": np.random.poisson(15, size=N),
    "avg_amount_7d":    np.random.exponential(scale=80, size=N).round(2),
    "is_international": np.random.randint(0, 2, size=N),
    "card_age_months":  np.random.randint(1, 120, size=N),
    "credit_score":     np.random.randint(300, 850, size=N),
})

# Simulate fraud label (imbalanced: ~3% fraud)
fraud_score = (
    (df["amount"] > 500).astype(int) * 2 +
    (df["is_international"] == 1).astype(int) * 1.5 +
    (df["distance_from_home"] > 200).astype(int) * 2 +
    (df["n_transactions_1h"] > 5).astype(int) * 1.5 +
    np.random.normal(0, 1, N)
)
df["is_fraud"] = (fraud_score > 4.5).astype(int)

print(f"Dataset shape:  {df.shape}")
print(f"Fraud rate:     {df['is_fraud'].mean():.2%}")
print(f"Sample:\n{df.head()}")

output_path = Path(__file__).parent.parent / "data" / "dataset.csv"
output_path.parent.mkdir(exist_ok=True)
df.to_csv(output_path, index=False)
print(f"\n✅ Dataset saved to {output_path}")
