# ============================================================
# SMART AI ASSISTANT — PREMIUM PRODUCT INTELLIGENCE
# Robust Streamlit edition | Backend-only dataset discovery
# ============================================================
from pathlib import Path
from io import BytesIO
import json, re, sqlite3, html
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from groq import Groq
except Exception:
    Groq = None
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
except Exception:
    RandomForestRegressor = None
    LinearRegression = None

# Optional parsers are imported safely so a missing optional package
# does not prevent the application from starting.
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None
try:
    from docx import Document
except Exception:
    Document = None
try:
    import yaml
except Exception:
    yaml = None

st.set_page_config(
    page_title="Smart AI Assistant | Product Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- PREMIUM UI ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(circle at 12% 0%,#282047 0,#12111d 42%,#0c0d14 100%);color:#f5f2ff;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#171426,#201a32);border-right:1px solid #403454;}
.block-container {max-width:1500px;padding-top:1.7rem;padding-bottom:3rem;}
.hero {padding:30px 34px;border:1px solid #574775;border-radius:22px;background:linear-gradient(115deg,rgba(111,76,164,.30),rgba(30,25,45,.65));margin-bottom:22px;}
.hero h1 {font-size:2.2rem;font-weight:800;color:#e1d4ff;margin:0;}
.hero p {color:#c9bfdc;margin:.45rem 0 0;}
.eyebrow {letter-spacing:2px;color:#bda2ff;font-size:11px;font-weight:800;}
.panel {padding:18px 20px;border:1px solid #403450;border-radius:17px;background:rgba(35,29,49,.78);}
div[data-testid="stMetric"] {background:#211b2d;border:1px solid #443654;padding:14px;border-radius:14px;}
.stButton>button[kind="primary"] {background:linear-gradient(90deg,#7045c5,#a16bea);border:0;}
[data-testid="stDataFrame"] {border:1px solid #443654;border-radius:12px;}
.small {font-size:12px;color:#aaa0bd;}
</style>
<div class="hero">
  <div class="eyebrow">✦ INTELLIGENCE • ANALYTICS • DECISION SUPPORT</div>
  <h1>Smart AI Assistant</h1>
  <p>Product Intelligence Workspace</p>
  <p>Turn product catalogues and operational data into clear, actionable business insight.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------- DATA ENGINE ----------------------------
DATA_ROOT = Path(__file__).resolve().parent / "datasets"
FORMAT_FOLDERS = {
    "CSV": {".csv"}, "Excel": {".xlsx", ".xls"},
    "JSON": {".json"}, "JSONL": {".jsonl", ".ndjson"},
    "Parquet": {".parquet"}, "Database": {".db", ".sqlite", ".sqlite3"},
    "XML": {".xml"}, "TXT": {".txt"}, "PDF": {".pdf"},
    "DOCX": {".docx"}, "HTML": {".html", ".htm"},
    "YAML": {".yaml", ".yml"},
}
SUPPORTED = ["csv","xlsx","xls","json","jsonl","ndjson","parquet","db","sqlite","sqlite3",
             "xml","txt","pdf","docx","html","htm","yaml","yml"]

ALIASES = {
    "product_id": ["product_id","id","sku","item_id","product_code"],
    "product": ["product","product_name","name","item","title"],
    "category": ["category","product_category","department","type"],
    "brand": ["brand","manufacturer","vendor"],
    "description": ["description","product_description","details","summary","text"],
    "price": ["price","unit_price","selling_price","sale_price","cost"],
    "discount": ["discount","discount_percent","discount_percentage"],
    "rating": ["rating","average_rating","stars"],
    "reviews_count": ["reviews_count","review_count","reviews"],
    "stock_quantity": ["stock_quantity","stock","inventory","quantity_in_stock"],
    "supplier": ["supplier","supplier_name"],
    "availability": ["availability","status","stock_status"],
    "country": ["country"], "state": ["state"], "city": ["city"],
    "region": ["region","location","market","territory"],
    "created_date": ["created_date","date","order_date","sales_date","transaction_date"],
    "quantity": ["quantity","units","qty","units_sold"],
    "revenue": ["revenue","sales","total_sales","sales_amount","total"],
    "profit": ["profit","net_profit","gross_profit"],
    "image_url": ["image_url","image","photo_url"], "tags": ["tags","keywords","tag"],
}

def clean_col(value):
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip())).strip("_").lower()

def records_to_frame(obj):
    if isinstance(obj, list):
        return pd.json_normalize(obj) if obj and isinstance(obj[0], dict) else pd.DataFrame({"description":[str(x) for x in obj]})
    if isinstance(obj, dict):
        for key in ("products","data","items","records","catalogue","catalog"):
            if key in obj and isinstance(obj[key], (list,dict)):
                return records_to_frame(obj[key])
        return pd.json_normalize(obj)
    return pd.DataFrame({"description":[str(obj)]})

def parse_file(name, raw):
    ext = Path(name).suffix.lower()
    bio = BytesIO(raw)
    if ext == ".csv": return pd.read_csv(bio, sep=None, engine="python")
    if ext in (".xlsx",".xls"): return pd.read_excel(bio)
    if ext == ".json": return records_to_frame(json.loads(raw.decode("utf-8-sig")))
    if ext in (".jsonl",".ndjson"): return pd.read_json(bio, lines=True)
    if ext == ".parquet": return pd.read_parquet(bio)
    if ext in (".db",".sqlite",".sqlite3"):
        con = sqlite3.connect(":memory:")
        con.deserialize(raw)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)["name"].tolist()
        if not tables: raise ValueError("No tables found in SQLite database.")
        preferred = next((t for t in tables if t.lower() in ("products","product","catalog","catalogue")), tables[0])
        out = pd.read_sql_query('SELECT * FROM "' + preferred.replace('"','""') + '"', con)
        con.close()
        return out
    if ext == ".xml": return pd.read_xml(bio)
    if ext in (".html",".htm"):
        try:
            tables = pd.read_html(bio)
            if tables: return max(tables,key=len)
        except Exception: pass
        return pd.DataFrame({"description":[re.sub(r"\s+"," ",html.unescape(re.sub(r"<[^>]+>"," ",raw.decode("utf-8","ignore")))).strip()]})
    if ext == ".txt":
        text = raw.decode("utf-8-sig","ignore")
        try: return pd.read_csv(BytesIO(raw), sep=None, engine="python")
        except Exception: return pd.DataFrame({"description":[text]})
    if ext == ".pdf":
        if PdfReader is None: raise ImportError("PDF support needs pypdf in requirements.txt.")
        return pd.DataFrame({"description":["\n".join(p.extract_text() or "" for p in PdfReader(bio).pages)]})
    if ext == ".docx":
        if Document is None: raise ImportError("DOCX support needs python-docx in requirements.txt.")
        doc = Document(bio)
        return pd.DataFrame({"description":["\n".join(p.text for p in doc.paragraphs)]})
    if ext in (".yaml",".yml"):
        if yaml is None: raise ImportError("YAML support needs PyYAML in requirements.txt.")
        return records_to_frame(yaml.safe_load(raw.decode("utf-8-sig")))
    raise ValueError(f"Unsupported format: {ext}")

def normalize(frame):
    df = frame.copy()
    df.columns = [clean_col(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    rename = {}
    for target, candidates in ALIASES.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found and found != target: rename[found] = target
    df = df.rename(columns=rename)
    if "product" not in df.columns:
        text_col = next((c for c in df.columns if df[c].astype(str).str.len().gt(25).any()), None)
        if text_col:
            df["product"] = "Document knowledge"
            df["description"] = df[text_col].astype(str)
        else:
            df["product"] = [f"Product {i+1}" for i in range(len(df))]
    defaults = {
        "product_id":[f"PRD-{i+1:05d}" for i in range(len(df))],
        "category":"Uncategorized","brand":"Unknown","description":"",
        "price":0,"discount":0,"rating":np.nan,"reviews_count":0,
        "stock_quantity":0,"supplier":"Not specified","availability":"Unknown",
        "country":"Unspecified","state":"Unspecified","city":"Unspecified",
        "region":"Unspecified","created_date":pd.NaT,"quantity":0,
        "revenue":np.nan,"profit":0,"image_url":"","tags":"",
    }
    for col, default in defaults.items():
        if col not in df.columns: df[col] = default
    for col in ["price","discount","rating","reviews_count","stock_quantity","quantity","revenue","profit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["price","discount","reviews_count","stock_quantity","quantity","profit"]:
        df[col] = df[col].fillna(0)
    df["revenue"] = df["revenue"].fillna(df["price"] * df["quantity"])
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    for col in ["product","category","brand","availability","country","state","city","region","description","supplier","tags","product_id"]:
        df[col] = df[col].fillna("Unknown").astype(str)
    df["discounted_price"] = (df["price"] * (1-df["discount"].clip(0,100)/100)).round(2)
    return df

def load_backend_datasets():
    frames, reports = [], []
    if not DATA_ROOT.exists(): return frames, reports
    for folder, extensions in FORMAT_FOLDERS.items():
        directory = DATA_ROOT / folder
        if not directory.is_dir(): continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in extensions:
                try:
                    frames.append(parse_file(path.name, path.read_bytes()))
                    reports.append(f"Loaded {path.name}")
                except Exception as exc:
                    reports.append(f"Skipped {path.name}: {exc}")
    return frames, reports

# ---------------------------- NAVIGATION / INPUT ----------------------------
with st.sidebar:
    st.markdown("## ✦ SMART AI")
    st.caption("PRODUCT INTELLIGENCE SUITE")
    page = st.radio("WORKSPACE", ["Overview", "Product Intelligence", "AI Intelligence Studio",
                                  "Catalogue", "Sales Analytics", "Regional Intelligence",
                                  "Data Explorer", "Data Quality"])
    st.divider()
    uploads = st.file_uploader("Import datasets", type=SUPPORTED, accept_multiple_files=True,
                               help="CSV, Excel, JSON, JSONL, Parquet, SQLite, XML, TXT, PDF, DOCX, HTML and YAML.")
    st.caption("Multi-format ingestion • Private backend discovery")
    st.divider()
    st.markdown("**DATA CONTROLS**")
    st.caption("Use filters to refine the active workspace.")

frames, ingestion_report = [], []
if uploads:
    for file in uploads:
        try:
            frames.append(parse_file(file.name, file.getvalue()))
            ingestion_report.append(f"Loaded {file.name}")
        except Exception as exc:
            ingestion_report.append(f"Could not load {file.name}: {exc}")
else:
    frames, ingestion_report = load_backend_datasets()

if frames:
    try:
        data = normalize(pd.concat(frames, ignore_index=True, sort=False))
    except Exception as exc:
        st.error(f"Dataset normalization failed: {exc}")
        st.stop()
    data_source = "Your dataset"
else:
    data = normalize(pd.DataFrame([
        {"product_id":"PRD-1001","product":"Aero Wireless Headphones","category":"Electronics","brand":"NovaTech","description":"Wireless headphones with noise cancellation and long battery life.","price":4999,"discount":10,"rating":4.6,"reviews_count":284,"stock_quantity":42,"supplier":"Nova Distribution","availability":"In Stock","country":"India","state":"Tamil Nadu","city":"Chennai","region":"South","quantity":3,"revenue":13497,"profit":3600},
        {"product_id":"PRD-1002","product":"Ergo Office Chair","category":"Furniture","brand":"WorkNest","description":"Ergonomic mesh chair with adjustable lumbar support.","price":8999,"discount":5,"rating":4.4,"reviews_count":156,"stock_quantity":18,"supplier":"WorkNest Supply","availability":"In Stock","country":"India","state":"Karnataka","city":"Bengaluru","region":"South","quantity":2,"revenue":17098,"profit":4200},
        {"product_id":"PRD-1003","product":"Smart Fitness Watch","category":"Wearables","brand":"PulseIQ","description":"Smart watch with GPS, activity and heart rate tracking.","price":6499,"discount":15,"rating":4.2,"reviews_count":391,"stock_quantity":0,"supplier":"PulseIQ India","availability":"Out of Stock","country":"India","state":"Maharashtra","city":"Mumbai","region":"West","quantity":5,"revenue":27500,"profit":7000},
        {"product_id":"PRD-1004","product":"Ceramic Coffee Maker","category":"Home Appliances","brand":"BrewCraft","description":"Programmable coffee maker with reusable filter.","price":3299,"discount":0,"rating":4.1,"reviews_count":98,"stock_quantity":27,"supplier":"BrewCraft Traders","availability":"In Stock","country":"India","state":"Delhi","city":"New Delhi","region":"North","quantity":4,"revenue":13196,"profit":3100},
        {"product_id":"PRD-1005","product":"Ultralight Laptop Stand","category":"Accessories","brand":"DeskPro","description":"Foldable aluminium laptop stand with adjustable height.","price":1799,"discount":8,"rating":4.7,"reviews_count":512,"stock_quantity":64,"supplier":"DeskPro Wholesale","availability":"In Stock","country":"India","state":"Kerala","city":"Kochi","region":"South","quantity":8,"revenue":13240,"profit":5100},
    ]))
    data_source = "Demo data"

with st.sidebar:
    st.success(f"{len(data):,} records ready")
    categories = sorted(data["category"].unique().tolist())
    brands = sorted(data["brand"].unique().tolist())
    regions = sorted(data["region"].unique().tolist())
    chosen_categories = st.multiselect("Categories", categories, default=categories)
    chosen_brands = st.multiselect("Brands", brands, default=brands)
    chosen_regions = st.multiselect("Regions", regions, default=regions)

filtered = data[
    data["category"].isin(chosen_categories) &
    data["brand"].isin(chosen_brands) &
    data["region"].isin(chosen_regions)
].copy()

revenue = float(filtered["revenue"].sum())
profit = float(filtered["profit"].sum())
stock = float(filtered["stock_quantity"].sum())
avg_price = float(filtered["price"].mean()) if not filtered.empty else 0
margin = profit/revenue*100 if revenue else 0

# ---------------------------- PAGE CONTENT ----------------------------
if page == "Overview":
    st.title("Executive Overview")
    st.caption(f"{data_source} • A real-time view of your product and sales dataset.")
    a,b,c,d = st.columns(4)
    a.metric("Product records", f"{len(filtered):,}")
    b.metric("Catalogue value", f"₹{(filtered['price']*filtered['stock_quantity']).sum():,.0f}")
    c.metric("Revenue", f"₹{revenue:,.0f}")
    d.metric("Inventory units", f"{stock:,.0f}")
    left,right = st.columns(2)
    with left:
        if not filtered.empty:
            chart = filtered.groupby("category",as_index=False).agg(Products=("product_id","count"))
            st.plotly_chart(px.bar(chart,x="category",y="Products",color="category",title="Products by category",template="plotly_dark"),use_container_width=True)
    with right:
        if not filtered.empty:
            chart = filtered.groupby("brand",as_index=False).agg(AvgPrice=("price","mean")).sort_values("AvgPrice",ascending=False)
            st.plotly_chart(px.bar(chart,x="brand",y="AvgPrice",title="Average price by brand",template="plotly_dark"),use_container_width=True)
    st.subheader("Top-rated products")
    st.dataframe(filtered.sort_values("rating",ascending=False)[["product_id","product","brand","category","price","rating","stock_quantity"]].head(12),use_container_width=True,hide_index=True)

elif page == "Product Intelligence":
    st.title("Product Intelligence Center")
    st.caption("Understand product mix, customer sentiment, pricing and inventory health.")
    a,b,c,d = st.columns(4)
    a.metric("Unique products", f"{filtered['product'].nunique():,}")
    b.metric("Average rating", f"{filtered['rating'].mean():.2f}/5" if filtered["rating"].notna().any() else "N/A")
    c.metric("Average discount", f"{filtered['discount'].mean():.1f}%")
    d.metric("Out of stock", f"{(filtered['stock_quantity']<=0).sum():,}")
    left,right = st.columns(2)
    with left:
        if not filtered.empty:
            chart=filtered.groupby("category",as_index=False).agg(AvgPrice=("price","mean"))
            st.plotly_chart(px.bar(chart,x="category",y="AvgPrice",color="category",title="Average price by category",template="plotly_dark"),use_container_width=True)
    with right:
        rated=filtered[filtered["rating"].notna()]
        if not rated.empty:
            st.plotly_chart(px.scatter(rated,x="price",y="rating",size="reviews_count",color="category",hover_name="product",title="Price vs customer rating",template="plotly_dark"),use_container_width=True)
    st.subheader("Product performance")
    st.dataframe(filtered,use_container_width=True,hide_index=True)

elif page == "AI Intelligence Studio":
    st.title("✦ AI Intelligence Studio")
    st.caption("AI-powered business copilot, predictive demand modelling and strategic recommendations.")

    tab_copilot, tab_forecast, tab_recommend, tab_insights = st.tabs([
        "AI Business Copilot", "Demand Forecast", "Product Recommendations", "Strategic Insights"
    ])

    # Compact, privacy-conscious dataset context for the language model.
    def build_business_context(frame):
        if frame.empty:
            return "The active dataset is empty after applying filters."
        top_products = frame.nlargest(8, "revenue")[["product", "category", "brand", "price", "quantity", "revenue", "profit", "rating", "stock_quantity"]]
        category_summary = frame.groupby("category", as_index=False).agg(
            products=("product_id", "count"), revenue=("revenue", "sum"), profit=("profit", "sum"))
        region_summary = frame.groupby("region", as_index=False).agg(
            revenue=("revenue", "sum"), products=("product_id", "count"))
        return (
            f"Dataset rows: {len(frame)}. Revenue: {frame['revenue'].sum():.2f}. "
            f"Profit: {frame['profit'].sum():.2f}. Units: {frame['quantity'].sum():.0f}. "
            f"Inventory units: {frame['stock_quantity'].sum():.0f}. "
            f"Average rating: {frame['rating'].mean():.2f}.\n"
            f"Category summary: {category_summary.to_dict(orient='records')}\n"
            f"Regional summary: {region_summary.to_dict(orient='records')}\n"
            f"Top products: {top_products.to_dict(orient='records')}"
        )

    with tab_copilot:
        st.subheader("Ask your business data")
        st.write("Ask questions in natural language about product performance, revenue, profit, stock, categories or regions.")
        api_key = ""
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            api_key = ""
        if Groq is None:
            st.warning("Groq package is not installed. Add `groq` to requirements.txt and redeploy to enable the AI copilot.")
        elif not api_key:
            st.info("Connect the AI engine by adding GROQ_API_KEY under Streamlit Cloud → App settings → Secrets.")
        else:
            st.success("AI engine connected · Groq")
        if "smart_ai_chat" not in st.session_state:
            st.session_state.smart_ai_chat = []
        for message in st.session_state.smart_ai_chat:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        prompt = st.chat_input("Example: Which products should I restock first and why?")
        if prompt:
            st.session_state.smart_ai_chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                if Groq is not None and api_key:
                    try:
                        client = Groq(api_key=api_key)
                        history = [{"role": "system", "content": (
                            "You are Smart AI Assistant, an expert product and business intelligence analyst. "
                            "Answer only from the supplied dataset context; clearly say when data is insufficient. "
                            "Use concise headings, calculations and actionable recommendations. Do not invent facts.\n\n"
                            + build_business_context(filtered)
                        )}]
                        history += [{"role": m["role"], "content": m["content"]} for m in st.session_state.smart_ai_chat[-8:]]
                        answer = client.chat.completions.create(
                            model="openai/gpt-oss-120b", messages=history,
                            temperature=0.25, max_tokens=1200
                        ).choices[0].message.content
                    except Exception as exc:
                        answer = f"The AI request could not be completed: {exc}. Check your Groq API key, model access and app logs."
                else:
                    answer = "The AI copilot is not connected yet. Add your GROQ_API_KEY in Streamlit Secrets and ensure `groq` is in requirements.txt. Meanwhile, use the Strategic Insights and Product Recommendations tabs for dataset-driven analysis."
                st.markdown(answer)
            st.session_state.smart_ai_chat.append({"role": "assistant", "content": answer})
        if st.button("Clear conversation", key="clear_ai_chat"):
            st.session_state.smart_ai_chat = []
            st.rerun()

    with tab_forecast:
        st.subheader("Machine-learning demand forecast")
        st.caption("Forecasts are generated only when a usable date and quantity/sales history exists.")
        date_rows = filtered.dropna(subset=["created_date"]).copy()
        target = st.selectbox("Forecast target", ["quantity", "revenue"], key="forecast_target")
        if len(date_rows) < 4 or date_rows["created_date"].nunique() < 3:
            st.info("At least 4 dated records across 3 or more distinct dates are required. Add a date column and historical sales/quantity data.")
        else:
            date_rows["period"] = date_rows["created_date"].dt.to_period("M").dt.to_timestamp()
            series = date_rows.groupby("period")[target].sum().sort_index().asfreq("MS", fill_value=0)
            if len(series) < 3 or series.sum() == 0:
                st.info("There is not enough non-zero monthly history to build a meaningful forecast.")
            elif LinearRegression is None:
                st.warning("Install scikit-learn to enable forecasting.")
            else:
                X = np.arange(len(series)).reshape(-1, 1)
                model_choice = st.selectbox("Model", ["Linear Regression", "Random Forest"])
                if model_choice == "Random Forest" and RandomForestRegressor is not None and len(series) >= 5:
                    model = RandomForestRegressor(n_estimators=150, random_state=42)
                else:
                    model = LinearRegression()
                model.fit(X, series.values)
                horizon = st.slider("Forecast horizon (months)", 1, 12, 3)
                future_x = np.arange(len(series), len(series)+horizon).reshape(-1, 1)
                predictions = np.maximum(model.predict(future_x), 0)
                future_dates = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
                forecast_df = pd.DataFrame({"Month": future_dates, "Forecast": predictions})
                history_df = pd.DataFrame({"Month": series.index, "Actual": series.values})
                st.plotly_chart(px.line(forecast_df, x="Month", y="Forecast", markers=True,
                                        title=f"{target.title()} forecast · {model_choice}", template="plotly_dark"), use_container_width=True)
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                st.caption("Forecasts are estimates from the available history, not guaranteed outcomes.")

    with tab_recommend:
        st.subheader("Intelligent product recommendations")
        if filtered.empty:
            st.info("No products match the current filters.")
        else:
            rec = filtered.copy()
            rec["opportunity_score"] = (
                rec["rating"].fillna(0).clip(0,5) * 15
                + np.log1p(rec["reviews_count"].clip(lower=0)) * 5
                + rec["profit"].clip(lower=0).rank(pct=True) * 25
                + (rec["stock_quantity"] > 0).astype(int) * 10
                - (rec["stock_quantity"] <= 0).astype(int) * 15
            )
            rec["recommendation_reason"] = np.select(
                [rec["stock_quantity"] <= 0, rec["rating"].fillna(0) >= 4.5, rec["profit"] > rec["profit"].median()],
                ["Restock opportunity", "Strong customer rating", "Above-median profit contribution"],
                default="Review pricing and promotion")
            st.dataframe(rec.sort_values("opportunity_score", ascending=False)[
                ["product", "category", "brand", "price", "rating", "reviews_count", "stock_quantity", "profit", "opportunity_score", "recommendation_reason"]
            ].head(20), use_container_width=True, hide_index=True)
            st.caption("Opportunity score is a transparent heuristic based on rating, reviews, profit and stock—not a trained recommendation model.")

    with tab_insights:
        st.subheader("Automated strategic insights")
        if filtered.empty:
            st.info("No data available for insights.")
        else:
            top_category = filtered.groupby("category")["revenue"].sum().sort_values(ascending=False)
            top_brand = filtered.groupby("brand")["revenue"].sum().sort_values(ascending=False)
            low_stock = filtered[filtered["stock_quantity"] <= 5]
            no_stock = filtered[filtered["stock_quantity"] <= 0]
            low_rated = filtered[filtered["rating"].notna() & (filtered["rating"] < 3.5)]
            insights = []
            if not top_category.empty:
                insights.append(("Category leader", f"{top_category.index[0]} is the top revenue category at ₹{top_category.iloc[0]:,.2f}."))
            if not top_brand.empty:
                insights.append(("Brand leader", f"{top_brand.index[0]} leads brand revenue at ₹{top_brand.iloc[0]:,.2f}."))
            if len(no_stock):
                insights.append(("Stockout alert", f"{len(no_stock)} product records have zero stock. Review replenishment priorities."))
            if len(low_stock):
                insights.append(("Inventory watch", f"{len(low_stock)} product records have 5 or fewer units available."))
            if len(low_rated):
                insights.append(("Customer experience", f"{len(low_rated)} products are rated below 3.5. Investigate reviews and product quality."))
            if margin < 10 and revenue > 0:
                insights.append(("Margin watch", f"Current profit margin is {margin:.1f}%. Review discounting, costs and pricing."))
            for title, detail in insights:
                st.markdown(f"**{title}**  \n{detail}")
                st.divider()
            if not insights:
                st.success("No immediate risk signals were detected in the currently filtered data.")
            st.download_button("Export strategic insights", "\n".join(f"{a}: {b}" for a,b in insights),
                               "strategic_insights.txt", "text/plain")

elif page == "Catalogue":
    st.title("Product Catalogue")
    query=st.text_input("Search products, descriptions, brands or tags")
    view=filtered.copy()
    if query:
        mask=view.astype(str).apply(lambda col:col.str.contains(query,case=False,na=False)).any(axis=1)
        view=view[mask]
    st.caption(f"{len(view):,} matching records")
    st.dataframe(view,use_container_width=True,hide_index=True)
    st.download_button("Export catalogue CSV",view.to_csv(index=False).encode("utf-8"),"product_catalogue.csv","text/csv")

elif page == "Sales Analytics":
    st.title("Sales Intelligence")
    a,b,c,d=st.columns(4)
    a.metric("Revenue",f"₹{revenue:,.2f}")
    b.metric("Profit",f"₹{profit:,.2f}")
    c.metric("Units sold",f"{filtered['quantity'].sum():,.0f}")
    d.metric("Profit margin",f"{margin:.2f}%")
    if filtered["created_date"].notna().any():
        dated=filtered.dropna(subset=["created_date"]).copy()
        dated["Month"]=dated["created_date"].dt.to_period("M").astype(str)
        monthly=dated.groupby("Month",as_index=False).agg(Revenue=("revenue","sum"),Profit=("profit","sum"))
        st.plotly_chart(px.line(monthly,x="Month",y=["Revenue","Profit"],markers=True,title="Monthly sales performance",template="plotly_dark"),use_container_width=True)
    else:
        st.info("Add a date/created_date column to your dataset to unlock time-series sales trends.")
    st.dataframe(filtered[["product","quantity","revenue","profit","region","created_date"]],use_container_width=True,hide_index=True)

elif page == "Regional Intelligence":
    st.title("Regional Intelligence")
    dimension=st.selectbox("Analyse geography by",["country","state","city","region"])
    geo=filtered.groupby(dimension,as_index=False).agg(Products=("product_id","count"),Revenue=("revenue","sum"),Profit=("profit","sum"),Stock=("stock_quantity","sum"))
    st.plotly_chart(px.bar(geo,x=dimension,y="Revenue",color=dimension,title=f"Revenue by {dimension}",template="plotly_dark"),use_container_width=True)
    st.dataframe(geo.sort_values("Revenue",ascending=False),use_container_width=True,hide_index=True)

elif page == "Data Explorer":
    st.title("Data Explorer")
    st.caption("Inspect, search, and export the active filtered dataset.")
    q=st.text_input("Search across all columns")
    view=filtered.copy()
    if q:
        view=view[view.astype(str).apply(lambda col:col.str.contains(q,case=False,na=False)).any(axis=1)]
    st.write(f"**{len(view):,} rows** · **{len(view.columns)} columns**")
    st.dataframe(view,use_container_width=True,hide_index=True)
    st.download_button("Download CSV",view.to_csv(index=False).encode("utf-8"),"smart_ai_export.csv","text/csv")
    st.download_button("Download JSON",view.to_json(orient="records",indent=2,date_format="iso").encode("utf-8"),"smart_ai_export.json","application/json")

elif page == "Data Quality":
    st.title("Data Quality & Governance")
    missing=filtered.isna().sum().rename("Missing values").to_frame()
    missing["Missing %"]=(missing["Missing values"]/max(len(filtered),1)*100).round(1)
    a,b,c=st.columns(3)
    a.metric("Rows",f"{len(filtered):,}")
    b.metric("Columns",f"{len(filtered.columns):,}")
    c.metric("Duplicate rows",f"{filtered.duplicated().sum():,}")
    st.subheader("Column completeness")
    st.dataframe(missing,use_container_width=True)
    st.subheader("Ingestion status")
    if ingestion_report:
        for item in ingestion_report: st.write("• "+item)
    else:
        st.info("No local or uploaded files were detected. The workspace is currently using its demo dataset.")

st.divider()
st.markdown('<div class="small">SMART AI ASSISTANT · PREMIUM PRODUCT INTELLIGENCE · Data processing runs within this Streamlit app.</div>',unsafe_allow_html=True)
