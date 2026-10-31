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

#Train the Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load the generated dataset
df = pd.read_csv("data/daily_sales.csv")

df["Date"] = pd.to_datetime(df["Date"])

# Sort dates
df = df.sort_values("Date").reset_index(drop=True)

# Feature engineering
df["DayNumber"] = (
    df["Date"] - df["Date"].min()
).dt.days

df["DayOfWeek"] = df["Date"].dt.dayofweek

# Define input features and target
X = df[["DayNumber", "DayOfWeek"]]
y = df["Revenue"]

# Chronological 80/20 split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train the model
model = LinearRegression()

model.fit(X_train, y_train)

# Predict sales for the test period
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Training Completed!")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.4f}")

# Display actual vs predicted sales
results = pd.DataFrame({
    "Date": df["Date"].iloc[split_index:],
    "Actual Revenue": y_test.values,
    "Predicted Revenue": y_pred.round(2)
})

print("\nActual vs Predicted Sales:")
print(results.head(10))

#model prediction,trained & evaluation
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#Predict Future Sales
# Generate the next seven dates
last_date = df["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=7,
    freq="D"
)

# Create features for future dates
future_df = pd.DataFrame({
    "Date": future_dates
})

future_df["DayNumber"] = (
    future_df["Date"] - df["Date"].min()
).dt.days

future_df["DayOfWeek"] = (
    future_df["Date"].dt.dayofweek
)

# Predict future revenue
future_features = future_df[
    ["DayNumber", "DayOfWeek"]
]

future_df["PredictedRevenue"] = model.predict(
    future_features
).round(2)

print("\nNext Seven Days Sales Prediction:")
print(future_df)
