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


# Load dataset
if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

else:

    file_path = Path("data/sales.csv")

    if file_path.exists():
        df = pd.read_csv(file_path)
    else:
        st.warning("Please upload a sales dataset.")
        st.stop()


# Convert date
df["Date"] = pd.to_datetime(df["Date"])

# Dashboard
if page == "Dashboard":

    st.subheader("Business Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Revenue",
            f"₹{df['Revenue'].sum():,.0f}"
        )

    with col2:
        st.metric(
            "Total Orders",
            f"{df['Order_ID'].nunique():,}"
        )

    with col3:
        st.metric(
            "Total Profit",
            f"₹{df['Profit'].sum():,.0f}"
        )

    with col4:
        st.metric(
            "Total Products",
            f"{df['Product'].nunique():,}"
        )

    st.write("---")

    # Revenue chart
    monthly_sales = (
        df.groupby(df["Date"].dt.to_period("M"))["Revenue"]
        .sum()
        .reset_index()
    )

    monthly_sales["Date"] = (
        monthly_sales["Date"].astype(str)
    )

    fig = px.line(
        monthly_sales,
        x="Date",
        y="Revenue",
        title="Monthly Revenue Trend",
        markers=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Product chart
    product_sales = (
        df.groupby("Product")["Revenue"]
        .sum()
        .reset_index()
    )

    fig2 = px.bar(
        product_sales,
        x="Product",
        y="Revenue",
        title="Product Performance"
    )

    st.plotly_chart(
        fig2,
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

    result = df.groupby(
        "Product"
    )["Revenue"].sum().reset_index()

    st.bar_chart(
        result,
        x="Product",
        y="Revenue"
    )


# Regional Analysis
elif page == "Regional Analysis":

    st.header("🌍 Regional Analysis")

    result = df.groupby(
        "Region"
    )["Revenue"].sum().reset_index()

    st.bar_chart(
        result,
        x="Region",
        y="Revenue"
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
