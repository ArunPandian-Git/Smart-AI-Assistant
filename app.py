import pandas as pd
import streamlit as st
import plotly.express as px

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.subheader("SmartSales AI - Sales Prediction")

try:

    # 1. Load dataset
    file_path = Path("data/daily_sales.csv")

    if not file_path.exists():
        st.error("Sales dataset not found: data/daily_sales.csv")
        st.stop()

    df = pd.read_csv(file_path)

    # 2. Validate columns
    required_columns = ["Date", "Revenue"]

    if not all(col in df.columns for col in required_columns):
        st.error("Dataset must contain Date and Revenue columns.")
        st.stop()

    # 3. Preprocessing
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")

    df = df.dropna(subset=["Date", "Revenue"])

    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 10:
        st.error("At least 10 valid records are required.")
        st.stop()

    # 4. Feature Engineering
    df["DayNumber"] = (
        df["Date"] - df["Date"].min()
    ).dt.days

    df["DayOfWeek"] = df["Date"].dt.dayofweek

    features = ["DayNumber", "DayOfWeek"]

    X = df[features]
    y = df["Revenue"]

    # 5. Chronological Train-Test Split
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # 6. Model Training
    model = LinearRegression()

    model.fit(X_train, y_train)

    # 7. Prediction
    y_pred = model.predict(X_test)

    # 8. Evaluation
    mae = mean_absolute_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)

    col1.metric("Test MAE", f"₹{mae:,.2f}")

    col2.metric("Test R²", f"{r2:.3f}")

    # 9. Actual vs Predicted
    comparison = pd.DataFrame({
        "Date": df["Date"].iloc[split_index:],
        "Actual Revenue": y_test.values,
        "Predicted Revenue": y_pred
    })

    fig1 = px.line(
        comparison,
        x="Date",
        y=["Actual Revenue", "Predicted Revenue"],
        title="Actual vs Predicted Sales"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # 10. Future Forecast
    st.subheader("Future Sales Forecast")

    days = st.slider(
        "Select Forecast Days",
        min_value=1,
        max_value=30,
        value=7
    )

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
        future_df[features]
    )

    # 11. Display Results
    st.dataframe(
        future_df,
        use_container_width=True
    )

    # 12. Forecast Graph
    fig2 = px.line(
        future_df,
        x="Date",
        y="PredictedRevenue",
        markers=True,
        title="Future Daily Revenue Prediction"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.success("Sales prediction completed successfully!")

except Exception as e:

    st.error(f"Application Error: {e}")

