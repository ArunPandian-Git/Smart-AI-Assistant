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

# =====================================
# CUSTOM DASHBOARD THEME
# =====================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #171321,
        #30204b,
        #21152f
    );
    color: #ffffff;
}

/* Main content spacing */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #15121b;
    border-right: 1px solid #30283c;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #ffffff;
}

/* Sidebar navigation */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background-color: #211c2b;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 7px;
}

/* Selected navigation */
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background-color: #8985ff;
    color: white;
}

/* Headings */
h1, h2, h3 {
    color: #ffffff !important;
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(
        145deg,
        #302744,
        #241e32
    );
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #46375c;
    min-height: 135px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.25);
}

.kpi-title {
    color: #c7bfd8;
    font-size: 14px;
}

.kpi-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: bold;
    margin-top: 12px;
}

.kpi-icon {
    font-size: 23px;
}

/* Chart containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(35, 29, 48, 0.9);
    border: 1px solid #453653;
    border-radius: 16px;
    padding: 12px;
}

/* Buttons */
.stButton button {
    background-color: #8985ff;
    color: white;
    border-radius: 9px;
    border: none;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #46375c;
    border-radius: 12px;
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

st.sidebar.markdown("""
<div style="
    padding:15px 5px 25px 5px;
    font-size:25px;
    font-weight:bold;
    color:#9995ff;
">
    🤖 SMART AI
</div>

<div style="
    color:#aaa;
    font-size:12px;
    margin-bottom:20px;
">
    SALES INTELLIGENCE PLATFORM
</div>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR NAVIGATION
# =====================================

st.sidebar.markdown("""
<div style="font-size:25px;font-weight:bold;color:#9995ff;">
    🤖 SMART AI
</div>
<div style="font-size:12px;color:#aaaaaa;">
    SALES INTELLIGENCE PLATFORM
</div>
""", unsafe_allow_html=True)

# ADD NAVIGATION HERE
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

# ==========================================
# SMART AI - PREMIUM DASHBOARD
# ==========================================

if page == "Dashboard":

    # Dashboard heading
    st.markdown("""
    <div style="padding:10px 0 20px 0;">
        <h1 style="font-size:32px;">
            📊 Business Dashboard
        </h1>
        <p style="color:#bdb3d1;">
            Monitor your sales performance and business growth
        </p>
    </div>
    """, unsafe_allow_html=True)

    if filtered_df.empty:

        st.warning("No data available for selected filters.")

    else:

        # ==================================
        # KPI CALCULATIONS
        # ==================================

        revenue = filtered_df["Revenue"].sum()
        profit = filtered_df["Profit"].sum()
        orders = filtered_df["Order_ID"].nunique()
        products = filtered_df["Product"].nunique()

        avg_order = revenue / orders if orders else 0

        margin = (
            profit / revenue * 100
            if revenue else 0
        )

        # ==================================
        # KPI CARDS
        # ==================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">💰</div>
                <div class="kpi-title">Total Revenue</div>
                <div class="kpi-value">₹{revenue:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">📦</div>
                <div class="kpi-title">Total Orders</div>
                <div class="kpi-value">{orders:,}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">💵</div>
                <div class="kpi-title">Total Profit</div>
                <div class="kpi-value">₹{profit:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">🛍️</div>
                <div class="kpi-title">Total Products</div>
                <div class="kpi-value">{products:,}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        col5, col6 = st.columns(2)

        with col5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">📈 Average Order Value</div>
                <div class="kpi-value">₹{avg_order:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col6:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">🎯 Profit Margin</div>
                <div class="kpi-value">{margin:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.divider()

        # ==================================
        # MONTHLY REVENUE TREND
        # ==================================

        st.subheader("📈 Revenue Performance")

        monthly_sales = (
            filtered_df.groupby(
                filtered_df["Date"].dt.to_period("M")
            )["Revenue"]
            .sum()
            .reset_index()
        )

        monthly_sales["Date"] = (
            monthly_sales["Date"].astype(str)
        )

        with st.container(border=True):

            fig1 = px.line(
                monthly_sales,
                x="Date",
                y="Revenue",
                markers=True,
                title="Monthly Revenue Trend"
            )

            fig1.update_traces(
                line=dict(color="#a855f7", width=4),
                marker=dict(size=9)
            )

            fig1.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )

        # ==================================
        # PRODUCT AND REGIONAL ANALYSIS
        # ==================================

        col1, col2 = st.columns(2)

        with col1:

            with st.container(border=True):

                st.subheader("📦 Product Performance")

                product_sales = (
                    filtered_df.groupby("Product")["Revenue"]
                    .sum()
                    .reset_index()
                    .sort_values("Revenue", ascending=False)
                )

                fig2 = px.bar(
                    product_sales,
                    x="Product",
                    y="Revenue",
                    color="Product",
                    title="Revenue by Product"
                )

                fig2.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    height=350
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

        with col2:

            with st.container(border=True):

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
                    hole=0.55,
                    title="Revenue by Region"
                )

                fig3.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=350
                )

                st.plotly_chart(
                    fig3,
                    use_container_width=True
                )

        # ==================================
        # CATEGORY ANALYSIS
        # ==================================

        col1, col2 = st.columns(2)

        with col1:

            with st.container(border=True):

                st.subheader("📊 Category Analysis")

                category_sales = (
                    filtered_df.groupby("Category")["Revenue"]
                    .sum()
                    .reset_index()
                )

                fig4 = px.bar(
                    category_sales,
                    x="Category",
                    y="Revenue",
                    color="Category",
                    title="Revenue by Category"
                )

                fig4.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    height=330
                )

                st.plotly_chart(
                    fig4,
                    use_container_width=True
                )

        with col2:

            with st.container(border=True):

                st.subheader("🏆 Top Selling Products")

                top_products = (
                    filtered_df.groupby("Product")
                    .agg(
                        Revenue=("Revenue", "sum"),
                        Quantity=("Quantity", "sum"),
                        Profit=("Profit", "sum")
                    )
                    .reset_index()
                    .sort_values(
                        "Revenue",
                        ascending=False
                    )
                    .head(5)
                )

                st.dataframe(
                    top_products,
                    use_container_width=True,
                    hide_index=True
                )

        # ==================================
        # SALES DATA SUMMARY
        # ==================================

        st.divider()

        st.subheader("📋 Recent Sales Transactions")

        recent_sales = filtered_df.sort_values(
            "Date",
            ascending=False
        ).head(10)

        st.dataframe(
            recent_sales,
            use_container_width=True,
            hide_index=True
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
