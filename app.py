import streamlit as st
import pandas as pd
import plotly.express as px

from groq import Groq
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from io import BytesIO

# ==========================================
# PREMIUM PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart AI Assistant | Sales Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://docs.streamlit.io/",
        "Report a bug": "https://github.com/",
        "About": """
        # Smart AI Assistant

        Intelligent Sales Analytics & Prediction Platform.

        Powered by Python, Machine Learning,
        Data Analytics and Generative AI.
        """
    }
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


/* =====================================
   PREMIUM SIDEBAR NAVIGATION
===================================== */

/* Navigation buttons */

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: linear-gradient(
        120deg,
        #211c2b,
        #292139
    );

    padding: 12px 15px;
    border-radius: 11px;
    margin-bottom: 8px;
    border: 1px solid #39304d;
    transition: all 0.3s ease;
}

/* Hover effect */

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #403258;
    border-color: #8985ff;
}

/* Selected navigation */

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(
        100deg,
        #6855c9,
        #9275e9
    );

    border: 1px solid #a99aff;
    box-shadow: 0 4px 15px rgba(137,133,255,0.25);
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

# =====================================
# PREMIUM SIDEBAR BRANDING
# =====================================

st.sidebar.markdown("""
<div style="
    padding:20px 10px;
    margin-bottom:20px;
    background:linear-gradient(135deg,#302744,#211c2b);
    border:1px solid #51416b;
    border-radius:15px;
">

    <div style="font-size:25px;font-weight:800;color:#9995ff;">
        🤖 SMART AI
    </div>

    <div style="
        font-size:11px;
        color:#c7bfd8;
        margin-top:8px;
        letter-spacing:1px;
    ">
        SALES INTELLIGENCE PLATFORM
    </div>

    <div style="
        font-size:10px;
        color:#8e83aa;
        margin-top:12px;
    ">
        PREMIUM ANALYTICS WORKSPACE
    </div>

</div>
""", unsafe_allow_html=True)
# ==========================================
# PREMIUM APPLICATION HERO SECTION
# ==========================================

st.markdown("""
<style>

.hero-container {
    background: linear-gradient(
        120deg,
        #302044,
        #38265c,
        #211a35
    );

    padding: 35px 35px;
    border-radius: 20px;
    border: 1px solid #594477;
    margin-bottom: 25px;

    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

.hero-title {
    font-size: 42px;
    font-weight: 850;
    letter-spacing: -1px;

    background: linear-gradient(
        90deg,
        #ffffff,
        #b9a5ff,
        #8f83ff
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 8px;
}

.hero-subtitle {
    color: #d1c5e7;
    font-size: 16px;
    margin-top: 5px;
}

.hero-badge {
    display: inline-block;
    background: rgba(137,133,255,0.15);
    color: #b9a5ff;
    border: 1px solid #7665c9;
    border-radius: 30px;
    padding: 7px 15px;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
    margin-top: 15px;
}

.hero-description {
    color: #aaa0bd;
    font-size: 13px;
    margin-top: 15px;
}

</style>

<div class="hero-container">

    <div class="hero-title">
        🤖 Smart AI Assistant
    </div>

    <div class="hero-subtitle">
        Intelligent Sales Analytics & Prediction Platform
    </div>

    <div class="hero-badge">
        ✦ AI POWERED BUSINESS INTELLIGENCE
    </div>

    <div class="hero-description">
        Transform your sales data into actionable business insights
        with Machine Learning, Advanced Analytics and Generative AI.
    </div>

</div>
""", unsafe_allow_html=True)


# =====================================
# PREMIUM SIDEBAR BRANDING
# =====================================

st.sidebar.markdown("""
<div style="
    padding:20px 10px;
    margin-bottom:20px;
    background:linear-gradient(135deg,#302744,#211c2b);
    border:1px solid #51416b;
    border-radius:15px;
">
    <div style="
        font-size:25px;
        font-weight:800;
        color:#9995ff;
    ">
        🤖 SMART AI
    </div>

    <div style="
        font-size:11px;
        color:#c7bfd8;
        margin-top:8px;
        letter-spacing:1px;
    ">
        SALES INTELLIGENCE PLATFORM
    </div>

    <div style="
        font-size:10px;
        color:#8e83aa;
        margin-top:12px;
    ">
        PREMIUM ANALYTICS WORKSPACE
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR NAVIGATION
# =====================================

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

# ==========================================
# PREMIUM SIDEBAR NAVIGATION
# ==========================================

st.sidebar.markdown("""
<div style="
    font-size:12px;
    color:#a99bc5;
    font-weight:700;
    letter-spacing:2px;
    margin-bottom:12px;
">
    WORKSPACE MENU
</div>
""", unsafe_allow_html=True)

page_options = {
    "🏠  Dashboard": "Dashboard",
    "📊  Sales Analytics": "Sales Analytics",
    "📦  Product Analysis": "Product Analysis",
    "🌍  Regional Analysis": "Regional Analysis",
    "🔮  Sales Prediction": "Sales Prediction",
    "🤖  AI Assistant": "AI Assistant",
    "💡  Business Insights": "Business Insights",
    "📁  Data Explorer": "Data Explorer"
}

selected_page = st.sidebar.radio(
    "Navigate",
    list(page_options.keys()),
    label_visibility="collapsed"
)

page = page_options[selected_page]


# ==========================================
# PREMIUM DATASET UPLOAD
# ==========================================

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="
    font-size:12px;
    color:#a99bc5;
    font-weight:700;
    letter-spacing:2px;
    margin-bottom:12px;
">
    DATA MANAGEMENT
</div>

<div style="
    background:linear-gradient(135deg,#302744,#211c2b);
    padding:15px;
    border-radius:13px;
    border:1px solid #51416b;
    margin-bottom:12px;
">
    <div style="font-size:22px;">📂</div>

    <div style="
        color:white;
        font-size:15px;
        font-weight:700;
        margin-top:8px;
    ">
        Upload Sales Dataset
    </div>

    <div style="
        color:#b9accd;
        font-size:11px;
        margin-top:5px;
    ">
        Import your business data for intelligent analysis.
    </div>

    <div style="
        color:#9988c0;
        font-size:10px;
        margin-top:10px;
    ">
        SUPPORTED FORMATS: CSV • XLSX
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Choose your dataset",
    type=["csv", "xlsx"],
    label_visibility="collapsed"
)

if uploaded_file is not None:

    st.sidebar.success("✅ Dataset uploaded successfully!")

    st.sidebar.markdown(
        f"""
        <div style="
            background:#211c2b;
            padding:12px;
            border-radius:10px;
            border:1px solid #51416b;
            margin-top:8px;
        ">
            <div style="color:#b9a5ff;font-size:12px;">
                FILE NAME
            </div>
            <div style="color:white;font-size:13px;font-weight:600;">
                {uploaded_file.name}
            </div>
            <div style="color:#aaa;font-size:11px;margin-top:5px;">
                Size: {uploaded_file.size / 1024:.2f} KB
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.sidebar.info("Upload a dataset to begin analysis.")


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

# =====================================================
# PREMIUM SALES ANALYTICS
# =====================================================

elif page == "Sales Analytics":

    st.title("📊 Sales Intelligence Center")

    st.markdown(
        "Analyze revenue, profitability, sales trends and business growth."
    )

    st.divider()

    if filtered_df.empty:

        st.warning("No sales data available for the selected filters.")

    else:

        # =========================================
        # 1. KPI CALCULATIONS
        # =========================================

        total_revenue = filtered_df["Revenue"].sum()
        total_profit = filtered_df["Profit"].sum()
        total_orders = filtered_df["Order_ID"].nunique()
        total_quantity = filtered_df["Quantity"].sum()

        avg_order_value = (
            total_revenue / total_orders if total_orders else 0
        )

        profit_margin = (
            total_profit / total_revenue * 100
            if total_revenue else 0
        )

        # =========================================
        # 2. PREMIUM KPI CARDS
        # =========================================

        st.subheader("💎 Business Performance Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "💰 Total Revenue",
                f"₹{total_revenue:,.0f}"
            )

        with col2:
            st.metric(
                "📈 Total Profit",
                f"₹{total_profit:,.0f}"
            )

        with col3:
            st.metric(
                "📦 Total Orders",
                f"{total_orders:,}"
            )

        col4, col5, col6 = st.columns(3)

        with col4:
            st.metric(
                "🛒 Units Sold",
                f"{total_quantity:,}"
            )

        with col5:
            st.metric(
                "💳 Average Order Value",
                f"₹{avg_order_value:,.0f}"
            )

        with col6:
            st.metric(
                "🎯 Profit Margin",
                f"{profit_margin:.2f}%"
            )

        st.divider()

        # =========================================
        # 3. REVENUE TREND
        # =========================================

        st.subheader("📈 Revenue Growth Analysis")

        monthly_revenue = (
            filtered_df.groupby(
                filtered_df["Date"].dt.to_period("M")
            )
            .agg(
                Revenue=("Revenue", "sum"),
                Profit=("Profit", "sum")
            )
            .reset_index()
        )

        monthly_revenue["Month"] = (
            monthly_revenue["Date"].astype(str)
        )

        fig = px.line(
            monthly_revenue,
            x="Month",
            y=["Revenue", "Profit"],
            markers=True,
            title="Monthly Revenue and Profit Trend"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =========================================
        # 4. DAILY SALES ANALYSIS
        # =========================================

        st.subheader("📅 Daily Sales Performance")

        daily_sales = (
            filtered_df.groupby(
                filtered_df["Date"].dt.date
            )["Revenue"]
            .sum()
            .reset_index()
        )

        daily_sales.columns = ["Date", "Revenue"]

        fig_daily = px.bar(
            daily_sales,
            x="Date",
            y="Revenue",
            title="Daily Revenue Performance",
            text_auto=".2s"
        )

        fig_daily.update_layout(
            template="plotly_dark",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig_daily,
            use_container_width=True
        )

        st.divider()

        # =========================================
        # 5. REVENUE AND PROFIT DISTRIBUTION
        # =========================================

        st.subheader("💹 Revenue vs Profit Analysis")

        col1, col2 = st.columns(2)

        with col1:

            fig_profit = px.bar(
                monthly_revenue,
                x="Month",
                y="Revenue",
                title="Monthly Revenue",
                text_auto=".2s"
            )

            fig_profit.update_layout(
                template="plotly_dark",
                height=350
            )

            st.plotly_chart(
                fig_profit,
                use_container_width=True
            )

        with col2:

            fig_margin = px.bar(
                monthly_revenue,
                x="Month",
                y="Profit",
                title="Monthly Profit",
                text_auto=".2s"
            )

            fig_margin.update_layout(
                template="plotly_dark",
                height=350
            )

            st.plotly_chart(
                fig_margin,
                use_container_width=True
            )

        st.divider()

        # =========================================
        # 6. BUSINESS INSIGHTS
        # =========================================

        st.subheader("🧠 Business Performance Insights")

        best_day = daily_sales.loc[
            daily_sales["Revenue"].idxmax()
        ]

        st.info(
            f"""
            **Business Summary**

            - Total Revenue: ₹{total_revenue:,.2f}
            - Total Profit: ₹{total_profit:,.2f}
            - Total Orders: {total_orders}
            - Average Order Value: ₹{avg_order_value:,.2f}
            - Profit Margin: {profit_margin:.2f}%
            - Best Sales Date: {best_day['Date']}
            - Revenue on Best Sales Date: ₹{best_day['Revenue']:,.2f}
            """
        )

        # =========================================
        # 7. TRANSACTION DATA
        # =========================================

        st.subheader("📋 Detailed Sales Transactions")

        st.dataframe(
            filtered_df.sort_values(
                "Date",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# PREMIUM PRODUCT ANALYSIS
# =====================================================

elif page == "Product Analysis":

    st.title("📦 Product Intelligence Center")

    st.markdown(
        "Discover your best-selling products, revenue contribution "
        "and product profitability."
    )

    st.divider()

    if filtered_df.empty:

        st.warning("No product data available for selected filters.")

    else:

        # =========================================
        # 1. PRODUCT SUMMARY
        # =========================================

        product_summary = (
            filtered_df.groupby("Product")
            .agg(
                Revenue=("Revenue", "sum"),
                Profit=("Profit", "sum"),
                Quantity=("Quantity", "sum"),
                Orders=("Order_ID", "nunique")
            )
            .reset_index()
            .sort_values("Revenue", ascending=False)
        )

        best_product = product_summary.iloc[0]

        total_product_revenue = product_summary["Revenue"].sum()

        total_product_profit = product_summary["Profit"].sum()

        total_products = len(product_summary)

        # =========================================
        # 2. PRODUCT KPI CARDS
        # =========================================

        st.subheader("💎 Product Performance Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🛍️ Total Products",
                total_products
            )

        with col2:
            st.metric(
                "💰 Product Revenue",
                f"₹{total_product_revenue:,.0f}"
            )

        with col3:
            st.metric(
                "📈 Product Profit",
                f"₹{total_product_profit:,.0f}"
            )

        with col4:
            st.metric(
                "🏆 Best-Selling Product",
                best_product["Product"]
            )

        st.divider()

        # =========================================
        # 3. PRODUCT REVENUE RANKING
        # =========================================

        st.subheader("🏆 Product Revenue Ranking")

        fig_product = px.bar(
            product_summary,
            x="Product",
            y="Revenue",
            color="Product",
            text_auto=".2s",
            title="Revenue Generated by Each Product"
        )

        fig_product.update_layout(
            template="plotly_dark",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )

        st.plotly_chart(
            fig_product,
            use_container_width=True
        )

        # =========================================
        # 4. PRODUCT PROFIT ANALYSIS
        # =========================================

        st.subheader("💹 Product Profitability")

        fig_profit = px.bar(
            product_summary,
            x="Product",
            y="Profit",
            color="Product",
            text_auto=".2s",
            title="Profit Contribution by Product"
        )

        fig_profit.update_layout(
            template="plotly_dark",
            height=400,
            showlegend=False
        )

        st.plotly_chart(
            fig_profit,
            use_container_width=True
        )

        st.divider()

        # =========================================
        # 5. PRODUCT REVENUE CONTRIBUTION
        # =========================================

        st.subheader("🍩 Product Revenue Contribution")

        fig_pie = px.pie(
            product_summary,
            names="Product",
            values="Revenue",
            hole=0.55,
            title="Percentage Revenue Share by Product"
        )

        fig_pie.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

        st.divider()

        # =========================================
        # 6. CATEGORY ANALYSIS
        # =========================================

        st.subheader("📊 Category Performance")

        category_summary = (
            filtered_df.groupby("Category")
            .agg(
                Revenue=("Revenue", "sum"),
                Profit=("Profit", "sum"),
                Quantity=("Quantity", "sum")
            )
            .reset_index()
            .sort_values("Revenue", ascending=False)
        )

        fig_category = px.bar(
            category_summary,
            x="Category",
            y="Revenue",
            color="Category",
            text_auto=".2s",
            title="Revenue by Product Category"
        )

        fig_category.update_layout(
            template="plotly_dark",
            height=380,
            showlegend=False
        )

        st.plotly_chart(
            fig_category,
            use_container_width=True
        )

        # =========================================
        # 7. TOP PRODUCT INSIGHTS
        # =========================================

        st.subheader("🧠 Product Business Insights")

        st.success(
            f"""
            **Top Performing Product: {best_product['Product']}**

            - Revenue: ₹{best_product['Revenue']:,.2f}
            - Profit: ₹{best_product['Profit']:,.2f}
            - Quantity Sold: {best_product['Quantity']}
            - Orders: {best_product['Orders']}
            """
        )

        # =========================================
        # 8. PRODUCT PERFORMANCE TABLE
        # =========================================

        st.subheader("📋 Complete Product Performance")

        st.dataframe(
            product_summary,
            use_container_width=True,
            hide_index=True
        )

        # =========================================
        # 9. DOWNLOAD PRODUCT REPORT
        # =========================================

        product_csv = product_summary.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Product Analysis Report",
            data=product_csv,
            file_name="product_analysis_report.csv",
            mime="text/csv"
        )


# =====================================================
# PREMIUM REGIONAL ANALYSIS
# =====================================================

elif page == "Regional Analysis":

    st.title("🌍 Regional Business Intelligence")

    st.markdown(
        "Explore sales distribution, revenue contribution "
        "and profitability across business regions."
    )

    st.divider()

    if filtered_df.empty:

        st.warning("No regional data available.")

    else:

        region_summary = (
            filtered_df.groupby("Region")
            .agg(
                Revenue=("Revenue", "sum"),
                Profit=("Profit", "sum"),
                Orders=("Order_ID", "nunique"),
                Quantity=("Quantity", "sum")
            )
            .reset_index()
            .sort_values("Revenue", ascending=False)
        )

        best_region = region_summary.iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🌍 Total Regions",
                len(region_summary)
            )

        with col2:
            st.metric(
                "💰 Total Revenue",
                f"₹{region_summary['Revenue'].sum():,.0f}"
            )

        with col3:
            st.metric(
                "🏆 Top Region",
                best_region["Region"]
            )

        st.divider()
        # ==========================================
        # PREMIUM REGIONAL INTELLIGENCE
        # ==========================================

        st.markdown("## 🌍 Regional Intelligence")

        st.caption(
            "Analyze revenue, profitability and business performance across regions."
        )

        # Regional KPI calculations
        total_revenue = region_summary["Revenue"].sum()
        total_profit = region_summary["Profit"].sum()
        total_regions = region_summary["Region"].nunique()

        top_region = region_summary.iloc[0]["Region"]
        top_region_revenue = region_summary.iloc[0]["Revenue"]

        profit_margin = (
            total_profit / total_revenue * 100
            if total_revenue > 0 else 0
        )

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🌍 Total Regions", total_regions)

        with col2:
            st.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")

        with col3:
            st.metric("📈 Total Profit", f"₹{total_profit:,.0f}")

        with col4:
            st.metric("🏆 Profit Margin", f"{profit_margin:.2f}%")

        st.divider()

        # Revenue comparison
        st.subheader("📊 Regional Revenue Performance")

        fig_region = px.bar(
            region_summary,
            x="Region",
            y="Revenue",
            color="Revenue",
            text_auto=".2s",
            title="Revenue Comparison Across Regions",
            color_continuous_scale="Purples"
        )

        fig_region.update_layout(
            template="plotly_dark",
            height=420,
            showlegend=False
        )

        st.plotly_chart(
            fig_region,
            use_container_width=True
        )

        # Profit distribution
        st.subheader("💎 Regional Profit Distribution")

        col1, col2 = st.columns([1.5, 1])

        with col1:

            fig_region_profit = px.pie(
                region_summary,
                names="Region",
                values="Profit",
                hole=0.6,
                title="Profit Contribution by Region"
            )

            fig_region_profit.update_layout(
                template="plotly_dark",
                height=420
            )

            st.plotly_chart(
                fig_region_profit,
                use_container_width=True
            )

        with col2:

            st.markdown("### 🏆 Top Performing Region")

            st.markdown(
                f"""
                <div style="
                    background:linear-gradient(135deg,#30204b,#514078);
                    padding:25px;
                    border-radius:18px;
                    border:1px solid #8170b5;
                ">
                    <p style="color:#c9b9ff;">BEST REGION</p>
                    <h2 style="color:white;">🌍 {top_region}</h2>
                    <h3 style="color:#b8a4ff;">
                        ₹{top_region_revenue:,.2f}
                    </h3>
                    <p style="color:#ddd;">
                        Highest revenue-generating region.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Regional performance table
        st.divider()

        st.subheader("📋 Regional Performance Report")

        st.dataframe(
            region_summary,
            use_container_width=True,
            hide_index=True
        )

        # Business insights
        st.divider()

        st.subheader("💡 Regional Business Insights")

        st.success(
            f"""
            🏆 **Best Performing Region**

            {top_region} generates the highest revenue:
            ₹{top_region_revenue:,.2f}
            """
        )

        st.info(
            f"""
            📊 **Business Overview**

            Your business operates across {total_regions} regions.

            Overall profit margin: {profit_margin:.2f}%
            """
        )

# ==========================================
# STAGE 5 - MACHINE LEARNING SALES PREDICTION
# ==========================================

elif page == "Sales Prediction":

    st.title("🔮 AI Sales Prediction")

    st.markdown(
        "### Predict future business revenue using Machine Learning"
    )

    st.write("---")

    # Prepare historical sales data
    daily_sales = (
        filtered_df.assign(
            Date=filtered_df["Date"].dt.normalize()
        )
        .groupby("Date")["Revenue"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )

    if len(daily_sales) < 5:

        st.warning(
            "Please upload at least 5 different dates of sales data "
            "to train the prediction model."
        )

        st.info(
            "For better predictions, use at least 30-90 days "
            "of historical sales data."
        )

    else:

        # ==================================
        # STEP 1: PREPARE DATA
        # ==================================

        daily_sales["Day_Number"] = (
            daily_sales["Date"] -
            daily_sales["Date"].min()
        ).dt.days

        X = daily_sales[["Day_Number"]]

        y = daily_sales["Revenue"]

        # ==================================
        # STEP 2: TRAIN AND TEST DATA
        # ==================================

        test_size = max(
            1,
            round(len(daily_sales) * 0.2)
        )

        X_train = X.iloc[:-test_size]
        X_test = X.iloc[-test_size:]

        y_train = y.iloc[:-test_size]
        y_test = y.iloc[-test_size:]

        # ==================================
        # STEP 3: TRAIN MODEL
        # ==================================

        model = LinearRegression()

        model.fit(X_train, y_train)

        # ==================================
        # STEP 4: MODEL EVALUATION
        # ==================================

        test_predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            test_predictions
        )

        st.subheader("📊 Model Performance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Training Records",
                len(X_train)
            )

        with col2:
            st.metric(
                "Mean Absolute Error",
                f"₹{mae:,.2f}"
            )

        with col3:

            if len(y_test) >= 2:

                r2 = r2_score(
                    y_test,
                    test_predictions
                )

                st.metric(
                    "R² Score",
                    f"{r2:.2f}"
                )

            else:

                st.metric(
                    "R² Score",
                    "Not available"
                )

        st.caption(
            "MAE measures the average absolute prediction error. "
            "R² indicates how well the model explains variation "
            "in the test data."
        )

        st.write("---")

        # ==================================
        # STEP 5: FUTURE SALES PREDICTION
        # ==================================

        st.subheader("🚀 Predict Future Sales")

        prediction_days = st.selectbox(
            "Select prediction period",
            [7, 15, 30, 60, 90],
            index=2
        )

        # Train final model using all historical data
        final_model = LinearRegression()

        final_model.fit(X, y)

        # Generate future dates
        last_date = daily_sales["Date"].max()

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=prediction_days,
            freq="D"
        )

        future_day_numbers = (
            future_dates - daily_sales["Date"].min()
        ).days

        future_X = pd.DataFrame({
            "Day_Number": future_day_numbers
        })

        # Predict revenue
        future_predictions = final_model.predict(
            future_X
        )

        # Revenue cannot be negative
        future_predictions = future_predictions.clip(
            min=0
        )

        # Create prediction dataframe
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Revenue": future_predictions
        })

        # ==================================
        # STEP 6: DISPLAY PREDICTIONS
        # ==================================

        total_predicted_revenue = (
            forecast_df["Predicted_Revenue"].sum()
        )

        average_predicted_revenue = (
            forecast_df["Predicted_Revenue"].mean()
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                f"Predicted Revenue ({prediction_days} Days)",
                f"₹{total_predicted_revenue:,.2f}"
            )

        with col2:

            st.metric(
                "Average Daily Predicted Revenue",
                f"₹{average_predicted_revenue:,.2f}"
            )

        st.write("---")

        # ==================================
        # STEP 7: VISUALIZATION
        # ==================================

        st.subheader("📈 Actual vs Predicted Revenue")

        actual_chart = daily_sales[
            ["Date", "Revenue"]
        ].copy()

        actual_chart["Type"] = "Actual Revenue"

        actual_chart = actual_chart.rename(
            columns={
                "Revenue": "Amount"
            }
        )

        prediction_chart = forecast_df[
            ["Date", "Predicted_Revenue"]
        ].copy()

        prediction_chart["Type"] = "Predicted Revenue"

        prediction_chart = prediction_chart.rename(
            columns={
                "Predicted_Revenue": "Amount"
            }
        )

        chart_data = pd.concat(
            [
                actual_chart,
                prediction_chart
            ],
            ignore_index=True
        )

        fig = px.line(
            chart_data,
            x="Date",
            y="Amount",
            color="Type",
            markers=True,
            title="Historical and Future Sales Forecast"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
            yaxis_title="Revenue (₹)",
            xaxis_title="Date"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==================================
        # STEP 8: PREDICTION TABLE
        # ==================================

        st.subheader("📋 Future Sales Prediction Table")

        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True
        )

        # ==================================
        # STEP 9: DOWNLOAD PREDICTIONS
        # ==================================

        csv = forecast_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Sales Predictions",
            data=csv,
            file_name="sales_predictions.csv",
            mime="text/csv"
        )


# ==========================================
# AI SALES ASSISTANT - GROQ INTEGRATION
# ==========================================

elif page == "AI Assistant":

    st.title("🤖 AI Sales Assistant")

    st.markdown(
        "### Ask questions about your business data"
    )

    st.caption(
        "Powered by Groq AI and Llama 3.3"
    )

    # Check dataset
    if filtered_df.empty:

        st.warning(
            "No sales data available. Please change your filters."
        )

    else:

        # ==================================
        # CREATE DATASET CONTEXT
        # ==================================

        total_revenue = filtered_df["Revenue"].sum()
        total_profit = filtered_df["Profit"].sum()
        total_orders = filtered_df["Order_ID"].nunique()

        product_summary = (
            filtered_df.groupby("Product")
            .agg(
                Revenue=("Revenue", "sum"),
                Quantity=("Quantity", "sum"),
                Profit=("Profit", "sum")
            )
            .sort_values("Revenue", ascending=False)
            .reset_index()
        )

        region_summary = (
            filtered_df.groupby("Region")["Revenue"]
            .sum()
            .sort_values(ascending=False)
        )

        category_summary = (
            filtered_df.groupby("Category")["Revenue"]
            .sum()
            .sort_values(ascending=False)
        )

        data_context = f"""
        SALES DATA SUMMARY

        Total Revenue: ₹{total_revenue:,.2f}

        Total Profit: ₹{total_profit:,.2f}

        Total Orders: {total_orders}

        Total Products: {filtered_df['Product'].nunique()}

        Total Quantity Sold: {filtered_df['Quantity'].sum()}

        Product Performance:
        {product_summary.to_string(index=False)}

        Regional Performance:
        {region_summary.to_string()}

        Category Performance:
        {category_summary.to_string()}
        """

        # ==================================
        # CHAT HISTORY
        # ==================================

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Clear chat button
        if st.button("🗑️ Clear Conversation"):

            st.session_state.chat_history = []
            st.rerun()

        # ==================================
        # DISPLAY OLD MESSAGES
        # ==================================

        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # ==================================
        # CHAT INPUT
        # ==================================

        question = st.chat_input(
            "Ask anything about your sales data..."
        )

        if question:

            # Display user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.markdown(question)

            # ==================================
            # GROQ API CONNECTION
            # ==================================

            try:

                api_key = st.secrets["GROQ_API_KEY"]

                client = Groq(api_key=api_key)

                system_prompt = f"""
                You are Smart AI, an intelligent sales analytics assistant.

                Answer questions using the supplied sales dataset.

                IMPORTANT RULES:

                1. Use the dataset information provided below.
                2. Do not invent revenue, profit, orders or quantities.
                3. If information is unavailable, clearly say so.
                4. Explain answers in simple English.
                5. Use bullet points when useful.
                6. Provide business recommendations when requested.
                7. Format currency using Indian Rupees (₹).

                DATASET INFORMATION:

                {data_context}
                """

                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]

                # Include recent conversation
                messages.extend(
                    st.session_state.chat_history[-10:]
                )

                with st.chat_message("assistant"):

                    with st.spinner("AI is analyzing your sales data..."):

                        response = client.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=messages,
                            temperature=0.3,
                            max_completion_tokens=1000
                        )

                        answer = response.choices[0].message.content

                        st.markdown(answer)

                # Save assistant response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error(
                    f"AI Assistant Error: {str(e)}"
                )

# ==========================================
# STAGE 7 - PREMIUM BUSINESS INSIGHTS
# ==========================================

elif page == "Business Insights":

    st.header("💡 Business Intelligence Center")

    st.markdown("""
    Discover business performance, identify opportunities,
    and receive actionable recommendations from your sales data.
    """)

    st.divider()

    if filtered_df.empty:

        st.warning("No sales data available. Please change your filters.")

    else:

        # ----------------------------------
        # 1. PREPARE DATA
        # ----------------------------------

        insight_df = filtered_df.copy()

        insight_df["Date"] = pd.to_datetime(
            insight_df["Date"],
            errors="coerce"
        )

        for col in ["Revenue", "Profit", "Quantity"]:
            insight_df[col] = pd.to_numeric(
                insight_df[col],
                errors="coerce"
            ).fillna(0)

        insight_df = insight_df.dropna(subset=["Date"])

        if insight_df.empty:

            st.warning("No valid sales dates found.")

        else:

            total_revenue = insight_df["Revenue"].sum()
            total_profit = insight_df["Profit"].sum()
            total_orders = insight_df["Order_ID"].nunique()
            total_quantity = insight_df["Quantity"].sum()

            profit_margin = (
                (total_profit / total_revenue) * 100
                if total_revenue != 0 else 0
            )

            # ----------------------------------
            # 2. BUSINESS PERFORMANCE KPIs
            # ----------------------------------

            st.subheader("📊 Business Performance Overview")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Total Revenue",
                f"₹{total_revenue:,.0f}"
            )

            c2.metric(
                "Total Profit",
                f"₹{total_profit:,.0f}"
            )

            c3.metric(
                "Total Orders",
                f"{total_orders:,}"
            )

            c4.metric(
                "Profit Margin",
                f"{profit_margin:.2f}%"
            )

            st.divider()

            # ----------------------------------
            # 3. SALES TREND ANALYSIS
            # ----------------------------------

            st.subheader("📈 Sales Trend Analysis")

            daily_sales = (
                insight_df.groupby("Date")["Revenue"]
                .sum()
                .reset_index()
                .sort_values("Date")
            )

            if len(daily_sales) >= 2:

                first_sales = daily_sales["Revenue"].iloc[0]
                last_sales = daily_sales["Revenue"].iloc[-1]

                if first_sales != 0:

                    growth = (
                        (last_sales - first_sales) / first_sales
                    ) * 100

                else:
                    growth = 0

                if growth > 0:

                    st.success(
                        f"📈 Sales increased by {growth:.2f}% "
                        "between the first and last recorded sales dates."
                    )

                elif growth < 0:

                    st.warning(
                        f"📉 Sales decreased by {abs(growth):.2f}% "
                        "between the first and last recorded sales dates."
                    )

                else:

                    st.info("Sales remained stable between the selected dates.")

            else:

                st.info("Add more sales dates to analyze sales growth.")

            fig_trend = px.line(
                daily_sales,
                x="Date",
                y="Revenue",
                markers=True,
                title="Daily Revenue Trend",
                template="plotly_dark"
            )

            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig_trend,
                use_container_width=True
            )

            st.divider()

            # ----------------------------------
            # 4. PRODUCT PERFORMANCE
            # ----------------------------------

            st.subheader("🏆 Product Performance Insights")

            product_summary = (
                insight_df.groupby("Product")
                .agg(
                    Revenue=("Revenue", "sum"),
                    Profit=("Profit", "sum"),
                    Quantity=("Quantity", "sum")
                )
                .reset_index()
                .sort_values("Revenue", ascending=False)
            )

            if not product_summary.empty:

                best_product = product_summary.iloc[0]

                st.success(
                    f"🏆 Top Product: {best_product['Product']} "
                    f"with ₹{best_product['Revenue']:,.0f} revenue."
                )

                fig_product = px.bar(
                    product_summary,
                    x="Product",
                    y="Revenue",
                    color="Profit",
                    title="Product Revenue Performance",
                    template="plotly_dark",
                    text_auto=True
                )

                fig_product.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )

                st.plotly_chart(
                    fig_product,
                    use_container_width=True
                )

            st.divider()

            # ----------------------------------
            # 5. REGIONAL PERFORMANCE
            # ----------------------------------

            st.subheader("🌍 Regional Business Insights")

            region_summary = (
                insight_df.groupby("Region")
                .agg(
                    Revenue=("Revenue", "sum"),
                    Profit=("Profit", "sum")
                )
                .reset_index()
                .sort_values("Revenue", ascending=False)
            )

            if not region_summary.empty:

                best_region = region_summary.iloc[0]

                st.success(
                    f"🌍 Highest Revenue Region: {best_region['Region']} "
                    f"with ₹{best_region['Revenue']:,.0f} revenue."
                )

                fig_region = px.pie(
                    region_summary,
                    names="Region",
                    values="Revenue",
                    hole=0.55,
                    title="Revenue Contribution by Region",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig_region,
                    use_container_width=True
                )

            st.divider()

            # ----------------------------------
            # 6. AUTOMATED BUSINESS RECOMMENDATIONS
            # ----------------------------------

            st.subheader("🚀 Smart Business Recommendations")

            recommendations = []

            # Revenue recommendation

            if total_revenue > 0:

                recommendations.append(
                    "💰 Revenue: Continue monitoring revenue trends "
                    "and focus on products generating the highest sales."
                )

            else:

                recommendations.append(
                    "⚠️ Revenue: No revenue recorded. Review your "
                    "sales transactions and dataset."
                )

            # Profit recommendation

            if profit_margin < 10:

                recommendations.append(
                    "⚠️ Profitability: Profit margin is below 10%. "
                    "Review operating costs, discounts and product pricing."
                )

            elif profit_margin >= 20:

                recommendations.append(
                    "✅ Profitability: Your profit margin is healthy. "
                    "Continue monitoring costs and pricing."
                )

            else:

                recommendations.append(
                    "📊 Profitability: Monitor your profit margin "
                    "and look for opportunities to improve it."
                )

            # Product recommendation

            if not product_summary.empty:

                low_profit_products = product_summary[
                    product_summary["Profit"] <= 0
                ]

                if not low_profit_products.empty:

                    products = ", ".join(
                        low_profit_products["Product"].astype(str).tolist()
                    )

                    recommendations.append(
                        f"📦 Product Alert: Review these products with "
                        f"zero or negative profit: {products}."
                    )

                else:

                    recommendations.append(
                        "📦 Product Strategy: Review your top-selling "
                        "products and maintain sufficient inventory."
                    )

            # Regional recommendation

            if len(region_summary) >= 2:

                weakest_region = region_summary.iloc[-1]

                recommendations.append(
                    f"🌍 Regional Opportunity: {weakest_region['Region']} "
                    "has the lowest revenue among the selected regions. "
                    "Investigate customer demand and marketing opportunities."
                )

            # Quantity recommendation

            if total_quantity > 0 and total_orders > 0:

                avg_quantity = total_quantity / total_orders

                recommendations.append(
                    f"🛒 Order Analysis: Average quantity per order is "
                    f"{avg_quantity:.2f}. Explore product bundles "
                    "and cross-selling opportunities."
                )

            for i, recommendation in enumerate(recommendations, 1):

                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #302744, #241e32);
                        padding: 18px;
                        border-radius: 12px;
                        border: 1px solid #51416b;
                        margin-bottom: 12px;
                    ">
                        <h4 style="color:#b9a5ff;">
                            Recommendation {i}
                        </h4>
                        <p style="color:white;">
                            {recommendation}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.divider()

            # ----------------------------------
            # 7. DOWNLOAD BUSINESS REPORT
            # ----------------------------------

            st.subheader("📥 Export Business Insights")

            report = f"""
SMART AI ASSISTANT
BUSINESS INSIGHTS REPORT

BUSINESS PERFORMANCE
--------------------
Total Revenue: Rs. {total_revenue:,.2f}
Total Profit: Rs. {total_profit:,.2f}
Total Orders: {total_orders}
Total Quantity Sold: {total_quantity}
Profit Margin: {profit_margin:.2f}%

TOP PRODUCT
-----------
{best_product['Product'] if not product_summary.empty else 'Not available'}

TOP REGION
----------
{best_region['Region'] if not region_summary.empty else 'Not available'}

BUSINESS RECOMMENDATIONS
------------------------
""" + "\n".join(
                f"{i}. {item}"
                for i, item in enumerate(recommendations, 1)
            )

            st.download_button(
                label="📥 Download Business Insights Report",
                data=report,
                file_name="business_insights_report.txt",
                mime="text/plain",
                use_container_width=True
            )

# ==========================================
# PREMIUM DATA EXPLORER
# ==========================================

elif page == "Data Explorer":

    st.header("📁 Data Intelligence Explorer")

    st.markdown("""
    Explore, search, filter, analyze and export
    your sales dataset using an interactive data workspace.
    """)

    st.divider()

    if filtered_df.empty:

        st.warning("No data available. Please change your dashboard filters.")

    else:

        # ----------------------------------
        # 1. DATA PREPARATION
        # ----------------------------------

        explorer_df = filtered_df.copy()

        # ----------------------------------
        # 2. DATASET OVERVIEW
        # ----------------------------------

        st.subheader("📊 Dataset Overview")

        total_rows = len(explorer_df)
        total_columns = len(explorer_df.columns)
        duplicate_rows = explorer_df.duplicated().sum()
        missing_values = explorer_df.isnull().sum().sum()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Records", f"{total_rows:,}")
        c2.metric("Total Columns", f"{total_columns:,}")
        c3.metric("Duplicate Rows", f"{duplicate_rows:,}")
        c4.metric("Missing Values", f"{missing_values:,}")

        st.divider()

        # ----------------------------------
        # 3. SEARCH AND FILTER
        # ----------------------------------

        st.subheader("🔍 Search & Filter Dataset")

        col1, col2 = st.columns(2)

        with col1:

            search_text = st.text_input(
                "Search records",
                placeholder="Search product, region or order ID..."
            )

        with col2:

            selected_columns = st.multiselect(
                "Select columns to display",
                explorer_df.columns.tolist(),
                default=explorer_df.columns.tolist()
            )

        col3, col4 = st.columns(2)

        with col3:

            if "Product" in explorer_df.columns:

                products = sorted(
                    explorer_df["Product"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_products = st.multiselect(
                    "Filter by Product",
                    products,
                    default=products
                )

            else:
                selected_products = []

        with col4:

            if "Region" in explorer_df.columns:

                regions = sorted(
                    explorer_df["Region"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_regions = st.multiselect(
                    "Filter by Region",
                    regions,
                    default=regions
                )

            else:
                selected_regions = []

        # ----------------------------------
        # 4. APPLY FILTERS
        # ----------------------------------

        search_df = explorer_df.copy()

        if "Product" in search_df.columns:

            search_df = search_df[
                search_df["Product"].astype(str).isin(
                    selected_products
                )
            ]

        if "Region" in search_df.columns:

            search_df = search_df[
                search_df["Region"].astype(str).isin(
                    selected_regions
                )
            ]

        if search_text.strip():

            search_mask = search_df.astype(str).apply(
                lambda column: column.str.contains(
                    search_text.strip(),
                    case=False,
                    na=False,
                    regex=False
                )
            ).any(axis=1)

            search_df = search_df[search_mask]

        st.divider()

        # ----------------------------------
        # 5. FILTERED DATA TABLE
        # ----------------------------------

        st.subheader("📋 Dataset Records")

        st.caption(
            f"Displaying {len(search_df):,} records "
            f"out of {len(explorer_df):,} available records."
        )

        if search_df.empty:

            st.warning("No matching records found.")

        else:

            display_columns = [
                col for col in selected_columns
                if col in search_df.columns
            ]

            if display_columns:

                display_df = search_df[display_columns]

                column_config = {}

                if "Revenue" in display_columns:

                    column_config["Revenue"] = st.column_config.NumberColumn(
                        "Revenue",
                        format="₹ %.2f"
                    )

                if "Profit" in display_columns:

                    column_config["Profit"] = st.column_config.NumberColumn(
                        "Profit",
                        format="₹ %.2f"
                    )

                if "Unit_Price" in display_columns:

                    column_config["Unit_Price"] = st.column_config.NumberColumn(
                        "Unit Price",
                        format="₹ %.2f"
                    )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    height=450,
                    column_config=column_config
                )

            else:

                st.warning("Please select at least one column.")

        st.divider()

        # ----------------------------------
        # 6. DATA QUALITY ANALYSIS
        # ----------------------------------

        st.subheader("🧹 Data Quality Monitor")

        quality_tab1, quality_tab2, quality_tab3 = st.tabs(
            [
                "Missing Values",
                "Duplicate Records",
                "Column Information"
            ]
        )

        with quality_tab1:

            missing_summary = (
                explorer_df.isnull()
                .sum()
                .reset_index()
            )

            missing_summary.columns = [
                "Column",
                "Missing Values"
            ]

            missing_summary["Missing Percentage"] = (
                missing_summary["Missing Values"]
                / len(explorer_df) * 100
            ).round(2)

            st.dataframe(
                missing_summary,
                use_container_width=True,
                hide_index=True
            )

            if missing_values == 0:

                st.success("Excellent! No missing values found.")

            else:

                st.warning(
                    f"Found {missing_values} missing values. "
                    "Review these columns before analysis."
                )

        with quality_tab2:

            st.metric(
                "Duplicate Records",
                f"{duplicate_rows:,}"
            )

            if duplicate_rows == 0:

                st.success("No duplicate records detected.")

            else:

                st.warning(
                    f"{duplicate_rows} duplicate records detected."
                )

                st.dataframe(
                    explorer_df[
                        explorer_df.duplicated(keep=False)
                    ],
                    use_container_width=True
                )

        with quality_tab3:

            column_info = pd.DataFrame({
                "Column Name": explorer_df.columns,
                "Data Type": explorer_df.dtypes.astype(str).values,
                "Non-Null Count": explorer_df.notnull().sum().values,
                "Unique Values": explorer_df.nunique().values
            })

            st.dataframe(
                column_info,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # ----------------------------------
        # 7. DATA EXPORT CENTER
        # ----------------------------------

        st.subheader("📥 Data Export Center")

        st.markdown(
            "Download the complete filtered dataset or "
            "only the records matching your search."
        )

        export_df = search_df.copy()

        export_col1, export_col2 = st.columns(2)

        with export_col1:

            csv_data = export_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="smart_ai_sales_data.csv",
                mime="text/csv",
                use_container_width=True
            )

        with export_col2:

            excel_buffer = BytesIO()

            with pd.ExcelWriter(
                excel_buffer,
                engine="openpyxl"
            ) as writer:

                export_df.to_excel(
                    writer,
                    index=False,
                    sheet_name="Sales Data"
                )

                quality_summary = pd.DataFrame({
                    "Column": explorer_df.columns,
                    "Missing Values": explorer_df.isnull().sum().values,
                    "Unique Values": explorer_df.nunique().values
                })

                quality_summary.to_excel(
                    writer,
                    index=False,
                    sheet_name="Data Quality"
                )

            st.download_button(
                label="📊 Download Excel Report",
                data=excel_buffer.getvalue(),
                file_name="smart_ai_sales_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        st.success(
            "Your Data Explorer workspace is ready!"
        )
