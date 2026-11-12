# ============================================================
# SMART AI ASSISTANT - SALES INTELLIGENCE PLATFORM
# Stable replacement for the previous app.py
# ============================================================

from pathlib import Path
from io import BytesIO
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from groq import Groq
except Exception:
    Groq = None

# --------------------------- PAGE CONFIG ---------------------------
st.set_page_config(
    page_title="Smart AI Assistant | Business Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- PREMIUM THEME ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: radial-gradient(circle at 10% 0%, #342451 0, #171321 42%, #100e18 100%); color:#f5f1ff; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#171321,#21182e); border-right:1px solid #46345e; }
.block-container { padding-top:1.6rem; padding-bottom:2.5rem; max-width:1500px; }
.hero { padding:30px 34px; border:1px solid #624b82; border-radius:22px; background:linear-gradient(120deg,rgba(110,75,160,.28),rgba(40,28,58,.55)); margin-bottom:22px; }
.hero h1 { color:#d4c2ff; font-size:2.25rem; margin:0; font-weight:800; }
.hero p { color:#c7badb; margin:.5rem 0 0 0; }
.kpi { background:linear-gradient(145deg,#2b213b,#1b1725); border:1px solid #514064; padding:18px 20px; border-radius:17px; min-height:112px; }
.kpi-label { color:#b9a9cf; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:1px; }
.kpi-value { color:#fff; font-size:25px; font-weight:800; margin-top:8px; }
.small-note { color:#b6a9c9; font-size:12px; }
div[data-testid="stMetric"] { background:#241c31; border:1px solid #49375e; padding:14px; border-radius:14px; }
.stButton>button { border-radius:10px; font-weight:600; }
</style>
<div class="hero">
  <h1>🤖 Smart AI Assistant</h1>
  <p>Intelligent Sales Analytics · Machine Learning Predictions · Generative AI Business Insights</p>
  <p class="small-note">Turn raw business data into clear, actionable decisions.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------- HELPERS ---------------------------
def money(value):
    return f"₹{float(value):,.2f}"

def clean_columns(frame):
    frame = frame.copy()
    frame.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", str(c).strip())).strip("_").lower()
        for c in frame.columns
    ]
    aliases = {
        "orderdate": "date", "order_date": "date", "sales_date": "date",
        "transaction_date": "date", "invoicedate": "date",
        "orderid": "order_id", "invoice": "order_id", "invoice_id": "order_id",
        "item": "product", "product_name": "product",
        "product_category": "category",
        "sales_region": "region", "location": "region",
        "units": "quantity", "qty": "quantity",
        "price": "unit_price", "selling_price": "unit_price",
        "sales": "revenue", "amount": "revenue", "total_sales": "revenue",
        "gross_profit": "profit",
    }
    frame = frame.rename(columns={c: aliases.get(c, c) for c in frame.columns})
    return frame

def load_data(upload):
    if upload is not None:
        upload.seek(0)
        if upload.name.lower().endswith(".csv"):
            try:
                raw = pd.read_csv(upload, sep=None, engine="python")
            except Exception:
                upload.seek(0)
                raw = pd.read_csv(upload)
        else:
            raw = pd.read_excel(upload)
    else:
        candidates = [
            Path("data/sales.csv"),
            Path("sales.csv"),
            Path("data/Sales.csv"),
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            return None, "No default dataset found. Upload a CSV or XLSX file."
        raw = pd.read_csv(path, sep=None, engine="python")

    df = clean_columns(raw)
    if "date" not in df.columns:
        df["date"] = pd.date_range("2024-01-01", periods=len(df), freq="D")
    if "order_id" not in df.columns:
        df["order_id"] = [f"ORD-{i+1:05d}" for i in range(len(df))]
    for col, default in [("product", "Unknown Product"), ("category", "Uncategorized"), ("region", "Unknown Region")]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str)
    if "quantity" not in df.columns:
        df["quantity"] = 1
    if "unit_price" not in df.columns:
        df["unit_price"] = np.nan
    if "revenue" not in df.columns:
        df["revenue"] = pd.to_numeric(df["quantity"], errors="coerce") * pd.to_numeric(df["unit_price"], errors="coerce")
    if "profit" not in df.columns:
        df["profit"] = 0

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["quantity", "unit_price", "revenue", "profit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "revenue"])
    df["quantity"] = df["quantity"].fillna(0)
    df["profit"] = df["profit"].fillna(0)
    df["order_id"] = df["order_id"].fillna("Unknown").astype(str)
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        return None, "The dataset has no usable rows. Check the Date and Revenue columns."
    return df, None

def chart_layout(fig, height=370):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=15, r=15, t=55, b=15),
        font=dict(color="#eee7fa"),
    )
    return fig

def kpi_card(label, value, icon):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{icon} {label}</div>'
        f'<div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

# --------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.markdown("## 🤖 SMART AI")
    st.caption("BUSINESS INTELLIGENCE SUITE")
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Dashboard", "Sales Analytics", "Product Analysis", "Regional Analysis",
         "Sales Prediction", "AI Assistant", "Business Insights", "Data Explorer"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("### 📂 Data Management")
    uploaded_file = st.file_uploader("Upload sales dataset", type=["csv", "xlsx"])
    st.caption("Supported formats: CSV and Excel (.xlsx)")
    st.divider()
    st.markdown("### 🔐 AI Configuration")
    st.caption("Add GROQ_API_KEY in Streamlit Secrets to enable AI chat.")
    st.markdown("[Project repository](https://github.com/ArunPandian-Git/Smart-AI-Assistant)")

# --------------------------- LOAD + FILTER DATA ---------------------------
try:
    df, load_error = load_data(uploaded_file)
except Exception as exc:
    df, load_error = None, f"Unable to read dataset: {exc}"

if df is None:
    st.warning(load_error)
    st.info("Required recommended fields: Date, Order_ID, Product, Category, Region, Quantity, Unit_Price, Revenue, Profit. Date and Revenue are essential; other fields are safely defaulted.")
    st.stop()

with st.sidebar:
    st.success(f"Loaded {len(df):,} records")
    min_date, max_date = df["date"].min().date(), df["date"].max().date()
    date_filter = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    product_options = sorted(df["product"].unique().tolist())
    region_options = sorted(df["region"].unique().tolist())
    selected_products = st.multiselect("Products", product_options, default=product_options)
    selected_regions = st.multiselect("Regions", region_options, default=region_options)

filtered = df.copy()
if isinstance(date_filter, (tuple, list)) and len(date_filter) == 2:
    filtered = filtered[filtered["date"].dt.date.between(date_filter[0], date_filter[1])]
filtered = filtered[filtered["product"].isin(selected_products)]
filtered = filtered[filtered["region"].isin(selected_regions)]

# --------------------------- SHARED SUMMARY ---------------------------
revenue = float(filtered["revenue"].sum()) if not filtered.empty else 0.0
profit = float(filtered["profit"].sum()) if not filtered.empty else 0.0
orders = int(filtered["order_id"].nunique()) if not filtered.empty else 0
units = float(filtered["quantity"].sum()) if not filtered.empty else 0.0
margin = (profit / revenue * 100) if revenue else 0.0
aov = revenue / orders if orders else 0.0

# --------------------------- DASHBOARD ---------------------------
if page == "Dashboard":
    st.title("📊 Executive Dashboard")
    st.caption("A real-time overview of your selected business data.")
    if filtered.empty:
        st.warning("No records match the selected filters.")
    else:
        cols = st.columns(4)
        with cols[0]: kpi_card("Total Revenue", money(revenue), "💰")
        with cols[1]: kpi_card("Total Profit", money(profit), "📈")
        with cols[2]: kpi_card("Total Orders", f"{orders:,}", "📦")
        with cols[3]: kpi_card("Profit Margin", f"{margin:.1f}%", "🎯")
        st.write("")
        c1, c2 = st.columns([1.5, 1])
        monthly = filtered.assign(month=filtered["date"].dt.to_period("M").astype(str)).groupby("month", as_index=False).agg(Revenue=("revenue", "sum"), Profit=("profit", "sum"))
        with c1:
            st.plotly_chart(chart_layout(px.line(monthly, x="month", y=["Revenue", "Profit"], markers=True, title="Monthly Revenue & Profit")), use_container_width=True)
        with c2:
            by_region = filtered.groupby("region", as_index=False)["revenue"].sum()
            st.plotly_chart(chart_layout(px.pie(by_region, names="region", values="revenue", hole=.58, title="Revenue by Region")), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            products = filtered.groupby("product", as_index=False)["revenue"].sum().nlargest(10, "revenue")
            st.plotly_chart(chart_layout(px.bar(products, x="product", y="revenue", title="Top 10 Products", color="product")), use_container_width=True)
        with c4:
            categories = filtered.groupby("category", as_index=False)["profit"].sum().sort_values("profit", ascending=False)
            st.plotly_chart(chart_layout(px.bar(categories, x="category", y="profit", title="Profit by Category", color="category")), use_container_width=True)
        st.subheader("Latest Transactions")
        st.dataframe(filtered.sort_values("date", ascending=False).head(12), use_container_width=True, hide_index=True)

# --------------------------- SALES ANALYTICS ---------------------------
elif page == "Sales Analytics":
    st.title("📈 Sales Intelligence Center")
    if filtered.empty:
        st.warning("No data available for the selected filters.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Revenue", money(revenue))
        c2.metric("Profit", money(profit))
        c3.metric("Average Order Value", money(aov))
        monthly = filtered.assign(month=filtered["date"].dt.to_period("M").astype(str)).groupby("month", as_index=False).agg(Revenue=("revenue", "sum"), Profit=("profit", "sum"), Units=("quantity", "sum"))
        st.plotly_chart(chart_layout(px.line(monthly, x="month", y=["Revenue", "Profit"], markers=True, title="Monthly Performance")), use_container_width=True)
        daily = filtered.groupby(filtered["date"].dt.date, as_index=False)["revenue"].sum()
        daily.columns = ["Date", "Revenue"]
        c4, c5 = st.columns(2)
        with c4: st.plotly_chart(chart_layout(px.bar(daily, x="Date", y="Revenue", title="Daily Revenue")), use_container_width=True)
        with c5: st.plotly_chart(chart_layout(px.histogram(filtered, x="revenue", nbins=30, title="Order Revenue Distribution")), use_container_width=True)
        st.subheader("Performance summary")
        st.write({"Transactions": len(filtered), "Unique orders": orders, "Units sold": units, "Profit margin": f"{margin:.2f}%", "Best sales day": str(daily.loc[daily["Revenue"].idxmax(), "Date"]) if not daily.empty else "N/A"})

# --------------------------- PRODUCT ANALYSIS ---------------------------
elif page == "Product Analysis":
    st.title("📦 Product Intelligence")
    if filtered.empty:
        st.warning("No product data for these filters.")
    else:
        product_summary = filtered.groupby(["product", "category"], as_index=False).agg(Revenue=("revenue", "sum"), Profit=("profit", "sum"), Units=("quantity", "sum"), Orders=("order_id", "nunique"))
        product_summary = product_summary.sort_values("Revenue", ascending=False)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(chart_layout(px.bar(product_summary.head(15), x="product", y="Revenue", color="category", title="Top Products by Revenue")), use_container_width=True)
        with c2: st.plotly_chart(chart_layout(px.scatter(product_summary, x="Revenue", y="Profit", size="Units", color="category", hover_name="product", title="Product Revenue vs Profit")), use_container_width=True)
        st.subheader("Product performance table")
        st.dataframe(product_summary, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download product report", product_summary.to_csv(index=False).encode("utf-8"), "product_analysis.csv", "text/csv")

# --------------------------- REGIONAL ANALYSIS ---------------------------
elif page == "Regional Analysis":
    st.title("🌍 Regional Intelligence")
    if filtered.empty:
        st.warning("No regional data for these filters.")
    else:
        region_summary = filtered.groupby("region", as_index=False).agg(Revenue=("revenue", "sum"), Profit=("profit", "sum"), Orders=("order_id", "nunique"), Units=("quantity", "sum"))
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(chart_layout(px.bar(region_summary.sort_values("Revenue", ascending=False), x="region", y="Revenue", color="region", title="Revenue by Region")), use_container_width=True)
        with c2: st.plotly_chart(chart_layout(px.pie(region_summary, names="region", values="Profit", hole=.5, title="Profit Contribution")), use_container_width=True)
        st.dataframe(region_summary.sort_values("Revenue", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download regional report", region_summary.to_csv(index=False).encode("utf-8"), "regional_analysis.csv", "text/csv")

# --------------------------- SALES PREDICTION ---------------------------
elif page == "Sales Prediction":
    st.title("🔮 Sales Forecast Studio")
    st.write("Aggregate historical revenue by day and train a regression model to estimate future daily revenue.")
    if len(filtered) < 5:
        st.warning("At least five valid records are recommended for prediction.")
    else:
        daily = filtered.groupby(filtered["date"].dt.normalize(), as_index=False)["revenue"].sum()
        daily.columns = ["date", "revenue"]
        daily = daily.sort_values("date")
        daily["day_index"] = (daily["date"] - daily["date"].min()).dt.days
        horizon = st.slider("Forecast horizon (days)", 7, 90, 30)
        model_choice = st.selectbox("Model", ["Linear Regression", "Random Forest"])
        if st.button("🚀 Train model and forecast", type="primary"):
            if len(daily) < 5:
                st.error("Not enough daily observations.")
            else:
                split = max(1, int(len(daily) * .8))
                X = daily[["day_index"]]
                y = daily["revenue"]
                model = LinearRegression() if model_choice == "Linear Regression" else RandomForestRegressor(n_estimators=150, random_state=42)
                model.fit(X.iloc[:split], y.iloc[:split])
                test_pred = model.predict(X.iloc[split:]) if split < len(daily) else np.array([])
                if len(test_pred):
                    st.metric("Validation MAE", money(mean_absolute_error(y.iloc[split:], test_pred)))
                    if len(y.iloc[split:]) > 1:
                        score = r2_score(y.iloc[split:], test_pred)
                        st.metric("Validation R²", f"{score:.3f}")
                # Refit on all historical observations for final forecast
                model.fit(X, y)
                future_dates = pd.date_range(daily["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
                future_x = (future_dates - daily["date"].min()).days.to_numpy().reshape(-1, 1)
                forecast = np.maximum(model.predict(future_x), 0)
                forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Revenue": forecast})
                hist = daily.rename(columns={"date": "Date", "revenue": "Revenue"})
                fig = px.line(hist, x="Date", y="Revenue", title="Historical Revenue and Forecast")
                fig.add_scatter(x=forecast_df["Date"], y=forecast_df["Predicted Revenue"], mode="lines+markers", name="Forecast")
                st.plotly_chart(chart_layout(fig, 430), use_container_width=True)
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download forecast", forecast_df.to_csv(index=False).encode("utf-8"), "sales_forecast.csv", "text/csv")
                st.caption("Forecasts are estimates based on historical daily revenue; they are not guaranteed outcomes.")

# --------------------------- AI ASSISTANT ---------------------------
elif page == "AI Assistant":
    st.title("🧠 Smart AI Business Assistant")
    st.caption("Ask questions about sales performance, trends, products, and business decisions.")
    api_key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role": "assistant", "content": "Hello! I can help interpret your sales data. Ask me about revenue, profit, products, regions, or business strategy."}]
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    prompt = st.chat_input("Ask your business question...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            if not api_key or Groq is None:
                answer = "AI service is not configured. Add GROQ_API_KEY under Streamlit Cloud → App → Settings → Secrets. Your analytics and prediction pages work without this key."
                st.info(answer)
            else:
                context = (
                    f"Dataset rows: {len(filtered)}; revenue: {revenue:.2f}; profit: {profit:.2f}; "
                    f"orders: {orders}; units: {units:.0f}; profit margin: {margin:.2f}%. "
                    f"Top products: {filtered.groupby('product')['revenue'].sum().nlargest(5).to_dict() if not filtered.empty else {}}. "
                    f"Regional revenue: {filtered.groupby('region')['revenue'].sum().to_dict() if not filtered.empty else {}}."
                )
                try:
                    client = Groq(api_key=api_key)
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "You are a concise senior business data analyst. Use the supplied dataset summary, distinguish facts from recommendations, and never invent data. " + context},
                            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages[-8:]],
                        ],
                        temperature=0.3,
                        max_tokens=900,
                    )
                    answer = response.choices[0].message.content or "No response returned."
                    st.markdown(answer)
                except Exception as exc:
                    answer = f"AI request failed: {exc}. Check the Groq key and network connection."
                    st.error(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    if st.button("🧹 Clear conversation"):
        st.session_state.chat_messages = []
        st.rerun()

# --------------------------- BUSINESS INSIGHTS ---------------------------
elif page == "Business Insights":
    st.title("💡 Business Insights & Recommendations")
    if filtered.empty:
        st.warning("No data available for insights.")
    else:
        top_product = filtered.groupby("product")["revenue"].sum().idxmax()
        top_region = filtered.groupby("region")["revenue"].sum().idxmax()
        best_category = filtered.groupby("category")["profit"].sum().idxmax()
        low_products = filtered.groupby("product")["revenue"].sum().sort_values().head(3)
        st.success(f"🏆 Revenue leader: **{top_product}**")
        st.success(f"🌍 Strongest region: **{top_region}**")
        st.info(f"💎 Most profitable category: **{best_category}**")
        st.subheader("Actionable observations")
        st.markdown(f"- Focus on retaining customers and stock availability for **{top_product}**, the highest-revenue product in the selected data.")
        st.markdown(f"- Review regional campaigns and sales coverage in **{top_region}**, which leads revenue.")
        st.markdown(f"- Review pricing, discounts, and operating costs for the three lowest-revenue products: **{', '.join(low_products.index.astype(str))}**.")
        st.markdown(f"- Current profit margin is **{margin:.2f}%**. Track margin alongside revenue so growth does not come at the expense of profitability.")
        st.caption("These observations are calculated from the filtered dataset, not external market research.")

# --------------------------- DATA EXPLORER ---------------------------
elif page == "Data Explorer":
    st.title("🗂️ Data Explorer")
    st.caption("Inspect, filter, and export the data currently loaded in the workspace.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(filtered):,}")
    c2.metric("Columns", f"{filtered.shape[1]:,}")
    c3.metric("Missing cells", f"{int(filtered.isna().sum().sum()):,}")
    search_text = st.text_input("Search across text fields")
    view = filtered.copy()
    if search_text:
        mask = view.astype(str).apply(lambda col: col.str.contains(search_text, case=False, na=False)).any(axis=1)
        view = view[mask]
    st.dataframe(view, use_container_width=True, hide_index=True)
    csv_bytes = view.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data (CSV)", csv_bytes, "smart_ai_filtered_data.csv", "text/csv")
    with st.expander("Data quality report"):
        quality = pd.DataFrame({"Column": df.columns, "Data Type": [str(t) for t in df.dtypes], "Missing Values": df.isna().sum().values, "Missing %": (df.isna().mean().values * 100).round(2)})
        st.dataframe(quality, use_container_width=True, hide_index=True)

# --------------------------- FOOTER ---------------------------
st.divider()
st.markdown(
    '<div style="text-align:center;color:#9f91b4;font-size:12px;">'
    'SMART AI ASSISTANT · SALES INTELLIGENCE PLATFORM · Built with Streamlit, Python, Plotly, Scikit-learn & Groq'
    '</div>',
    unsafe_allow_html=True,
)
