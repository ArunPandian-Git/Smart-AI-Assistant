import os
import numpy as np
import pandas as pd

# Make the example reproducible
np.random.seed(42)

# Generate two years of daily dates
dates = pd.date_range(
    start="2024-01-01",
    end="2025-12-31",
    freq="D"
)

# Generate a gradual sales trend
trend = np.arange(len(dates)) * 50

# Generate weekly variations
weekly_pattern = 2000 * np.sin(
    2 * np.pi * np.arange(len(dates)) / 7
)

# Generate random fluctuations
noise = np.random.normal(
    loc=0,
    scale=1500,
    size=len(dates)
)

# Calculate daily revenue
revenue = (
    20000
    + trend
    + weekly_pattern
    + noise
)

# Keep revenue positive
revenue = np.maximum(revenue, 1000)

# Create a DataFrame
df = pd.DataFrame({
    "Date": dates,
    "Revenue": revenue.round(2)
})

# Create the directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save the dataset
df.to_csv(
    "data/daily_sales.csv",
    index=False
)

print(df.head())
print("\nDataset shape:", df.shape)
print("\nDataset saved successfully!")
