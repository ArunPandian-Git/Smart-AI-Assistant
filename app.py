import streamlit as st
import pandas as pd
import plotly.express as px

from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Smart AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #161B26;
}

.metric-card {
    background-color: #1C2535;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #303B50;
}

</style>
""", unsafe_allow_html=True)


# Application title
st.title("🤖 Smart AI Assistant")

st.markdown(
    "### Intelligent Sales Analytics & Prediction Platform"
)

st.write("---")


# Sidebar
st.sidebar.title("SMART AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Sales Analytics",
        "Product Analysis",
        "Regional Analysis",
        "Sales Prediction",
        "AI Assistant",
        "Business Insights",
        "Data Explorer"
    ]
)


# Upload dataset
uploaded_file = st.sidebar.file_uploader(
    "Upload Sales Dataset",
    type=["csv", "xlsx"]
)

# =====================================
# DATASET LOADING AND CLEANING
# =====================================

import re

try:

    # Load dataset
    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):

            uploaded_file.seek(0)

            # Automatically detect delimiter
            df = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python"
            )

        else:
            df = pd.read_excel(uploaded_file)

    else:

        file_path = Path("data/sales.csv")

        if not file_path.exists():
            st.warning("Please upload a sales dataset.")
            st.stop()

        df = pd.read_csv(
            file_path,
            sep=None,
            engine="python"
        )

    # Clean column names
    df.columns = [
        re.sub(
            r"_+",
            "_",
            re.sub(
                r"[^a-zA-Z0-9]+",
                "_",
                str(col).strip().strip("|_")
            )
        ).strip("_").lower()
        for col in df.columns
    ]

    # Remove empty columns
    df = df.loc[:, df.columns != ""]

    # Standardize column names
    column_mapping = {
        "date": "Date",
        "order_date": "Date",
        "sales_date": "Date",
        "transaction_date": "Date",
        "order_id": "Order_ID",
        "product": "Product",
        "category": "Category",
        "region": "Region",
        "quantity": "Quantity",
        "unit_price": "Unit_Price",
        "revenue": "Revenue",
        "profit": "Profit"
    }

    df = df.rename(columns=column_mapping)

    # Validate required columns
    required_columns = [
        "Date",
        "Order_ID",
        "Product",
        "Category",
        "Region",
        "Quantity",
        "Unit_Price",
        "Revenue",
        "Profit"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        st.error(
            f"Missing columns: {missing_columns}"
        )

        st.write(
            "Available columns:",
            df.columns.tolist()
        )

        st.stop()

    # Convert date
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Convert numerical columns
    numeric_columns = [
        "Quantity",
        "Unit_Price",
        "Revenue",
        "Profit"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove invalid records
    df = df.dropna(
        subset=["Date", "Revenue", "Profit"]
    )

    st.sidebar.success("Dataset loaded successfully!")
        # Dashboard Filters
    if df.empty:
        st.error("No valid records found in dataset.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Dashboard Filters")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    products = sorted(df["Product"].dropna().unique())

    selected_products = st.sidebar.multiselect(
        "Select Products",
        products,
        default=products
    )

    regions = sorted(df["Region"].dropna().unique())

    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        regions,
        default=regions
    )

    filtered_df = df.copy()

    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["Date"].dt.date >= date_range[0]) &
            (filtered_df["Date"].dt.date <= date_range[1])
        ]

    filtered_df = filtered_df[
        filtered_df["Product"].isin(selected_products)
    ]

    filtered_df = filtered_df[
        filtered_df["Region"].isin(selected_regions)
    ]
    except Exception as e:

    st.error(f"Dataset loading error: {e}")
    st.stop()

  # =====================================
# ADVANCED DASHBOARD
# =====================================

if page == "Dashboard":

    st.subheader("📊 Business Performance Overview")

    if filtered_df.empty:
        st.warning("No records found for the selected filters.")

    else:
        # KPI calculations
        total_revenue = filtered_df["Revenue"].sum()
        total_profit = filtered_df["Profit"].sum()
        total_orders = filtered_df["Order_ID"].nunique()
        total_products = filtered_df["Product"].nunique()

        average_order_value = (
            total_revenue / total_orders
            if total_orders > 0 else 0
        )

        profit_margin = (
            (total_profit / total_revenue) * 100
            if total_revenue != 0 else 0
        )

        # KPI cards
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "💰 Total Revenue",
            f"₹{total_revenue:,.0f}"
        )

        col2.metric(
            "📦 Total Orders",
            f"{total_orders:,}"
        )

        col3.metric(
            "💵 Total Profit",
            f"₹{total_profit:,.0f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "🛍️ Total Products",
            f"{total_products:,}"
        )

        col5.metric(
            "📈 Average Order Value",
            f"₹{average_order_value:,.0f}"
        )

        col6.metric(
            "Profit Margin",
            f"{profit_margin:.2f}%"
        )

        st.write("---")

        # Monthly Revenue Trend
        st.subheader("📈 Monthly Revenue Trend")

        monthly_sales = (
            filtered_df.groupby(
                filtered_df["Date"].dt.to_period("M")
            )["Revenue"]
            .sum()
            .reset_index()
        )

        monthly_sales["Date"] = monthly_sales["Date"].astype(str)

        fig = px.line(
            monthly_sales,
            x="Date",
            y="Revenue",
            markers=True,
            title="Monthly Revenue"
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(
            fig,
            use_container_width=True
        )



# Sales Analytics
elif page == "Sales Analytics":

    st.header("📊 Sales Analytics")

    st.dataframe(
        df,
        use_container_width=True
    )

# Product Analysis
elif page == "Product Analysis":

    st.header("📦 Product Analysis")

    if filtered_df.empty:
        st.warning("No product data available for selected filters.")

    else:
        result = filtered_df.groupby(
            "Product"
        )["Revenue"].sum().reset_index()

        st.bar_chart(
            result,
            x="Product",
            y="Revenue"
        )

        # Regional Analysis
        st.subheader("🌍 Regional Performance")

        region_sales = (
            filtered_df.groupby("Region")["Revenue"]
            .sum()
            .reset_index()
        )

        fig3 = px.pie(
            region_sales,
            names="Region",
            values="Revenue",
            hole=0.4,
            title="Revenue by Region"
        )

        fig3.update_layout(template="plotly_dark")

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        # Category Analysis
        st.subheader("📊 Category Performance")

        category_sales = (
            filtered_df.groupby("Category")["Revenue"]
            .sum()
            .reset_index()
        )

        fig4 = px.bar(
            category_sales,
            x="Category",
            y="Revenue",
            title="Revenue by Category"
        )

        fig4.update_layout(template="plotly_dark")

        st.plotly_chart(
            fig4,
            use_container_width=True
        )


# Sales Prediction
elif page == "Sales Prediction":

    st.header("🔮 Sales Prediction")

    st.info(
        "Machine Learning forecasting will be integrated in Stage 5."
    )


# AI Assistant
elif page == "AI Assistant":

    st.header("🤖 AI Sales Assistant")

    question = st.chat_input(
        "Ask anything about your sales data..."
    )

    if question:

        st.chat_message("user").write(question)

        st.chat_message("assistant").write(
            "AI integration will be implemented in Stage 6."
        )


# Business Insights
elif page == "Business Insights":

    st.header("💡 Business Insights")

    st.info(
        "Automated business recommendations will be implemented in Stage 7."
    )


# Data Explorer
elif page == "Data Explorer":

    st.header("📁 Data Explorer")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.download_button(
        "Download Sales Data",
        df.to_csv(index=False),
        "sales_data.csv",
        "text/csv"
    )
