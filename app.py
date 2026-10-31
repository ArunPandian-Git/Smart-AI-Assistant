import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.subheader("Sales Prediction")

# Load historical data
df = pd.read_csv("data/daily_sales.csv")

df["Date"] = pd.to_datetime(df["Date"])

# Create features
df["DayNumber"] = (
    df["Date"] - df["Date"].min()
).dt.days

df["DayOfWeek"] = df["Date"].dt.dayofweek

X = df[["DayNumber", "DayOfWeek"]]
y = df["Revenue"]

# Chronological split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train the model
model = LinearRegression()

model.fit(X_train, y_train)

# Evaluate on test data
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2 = st.columns(2)

col1.metric("Test MAE", f"₹{mae:,.2f}")
col2.metric("Test R²", f"{r2:.3f}")

# Select forecast horizon
days = st.slider(
    "Number of days to forecast",
    min_value=1,
    max_value=30,
    value=7
)

# Create future dates
last_date = df["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=days,
    freq="D"
)

future_df = pd.DataFrame({
    "Date": future_dates
})

future_df["DayNumber"] = (
    future_df["Date"] - df["Date"].min()
).dt.days

future_df["DayOfWeek"] = (
    future_df["Date"].dt.dayofweek
)

future_df["PredictedRevenue"] = model.predict(
    future_df[["DayNumber", "DayOfWeek"]]
)

# Display forecast table
st.subheader("Future Sales Forecast")

st.dataframe(
    future_df,
    use_container_width=True
)

# Plot the forecast
fig = px.line(
    future_df,
    x="Date",
    y="PredictedRevenue",
    markers=True,
    title="Predicted Daily Revenue"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
