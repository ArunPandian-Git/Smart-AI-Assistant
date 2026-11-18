# ============================================================
# SMART AI ASSISTANT — PRODUCT & SALES INTELLIGENCE PLATFORM
# Premium multi-format edition | Streamlit deployment-ready
# ============================================================
from pathlib import Path
from io import BytesIO
import os, re, json, sqlite3, html
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

# Optional document parsers. The app remains usable if one is not installed.
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
.block-container { padding-top:1.6rem; padding-bottom:2.5rem; max-width:1550px; }
.hero { padding:30px 34px; border:1px solid #624b82; border-radius:22px; background:linear-gradient(120deg,rgba(110,75,160,.28),rgba(40,28,58,.55)); margin-bottom:22px; }
.hero h1 { color:#d4c2ff; font-size:2.25rem; margin:0; font-weight:800; }
.hero p { color:#c7badb; margin:.5rem 0 0 0; }
.kpi { background:linear-gradient(145deg,#2b213b,#1b1725); border:1px solid #514064; padding:18px 20px; border-radius:17px; min-height:112px; }
.kpi-label { color:#b9a9cf; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:1px; }
.kpi-value { color:#fff; font-size:25px; font-weight:800; margin-top:8px; }
.small-note { color:#b6a9c9; font-size:12px; }
div[data-testid="stMetric"] { background:#241c31; border:1px solid #49375e; padding:14px; border-radius:14px; }
.stButton>button[kind="primary"] { background:linear-gradient(90deg,#7c3aed,#a855f7); border:0; }
</style>
<div class="hero">
  <h1>🤖 Smart AI Assistant</h1>
  <p>Product & Sales Intelligence Platform</p>
  <div style="color:#b9a5ff;font-size:12px;font-weight:800;letter-spacing:2px;margin-top:14px;">✦ MULTI-FORMAT DATA INTELLIGENCE</div>
  <p>Transform product catalogues and sales records into actionable intelligence with advanced analytics, machine learning and generative AI.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------- CONSTANTS ---------------------------
FORMAT_TYPES = ["csv", "xlsx", "json", "jsonl", "parquet", "db", "sqlite",
                "xml", "txt", "pdf", "docx", "html", "yaml", "yml"]
FORMAT_FOLDERS = {
    "CSV": ["csv"], "Excel": ["xlsx"], "JSON": ["json"],
    "JSONL": ["jsonl"], "Parquet": ["parquet"],
    "Database": ["db", "sqlite"], "XML": ["xml"], "TXT": ["txt"],
    "PDF": ["pdf"], "DOCX": ["docx"], "HTML": ["html"],
    "YAML": ["yaml", "yml"],
}
ALIASES = {
    "product_id": ["product_id", "id", "sku", "item_id", "product_code"],
    "product": ["product", "product_name", "name", "item", "title"],
    "category": ["category", "product_category", "department", "type"],
    "brand": ["brand", "manufacturer", "vendor"],
    "description": ["description", "product_description", "details", "summary", "text"],
    "price": ["price", "unit_price", "selling_price", "sale_price", "cost"],
    "discount": ["discount", "discount_percent", "discount_percentage"],
    "rating": ["rating", "average_rating", "stars"],
    "reviews_count": ["reviews_count", "review_count", "reviews"],
    "stock_quantity": ["stock_quantity", "stock", "inventory", "quantity_in_stock"],
    "supplier": ["supplier", "supplier_name"],
    "availability": ["availability", "status", "stock_status"],
    "region": ["region", "location", "market", "territory"],
    "date": ["date", "order_date", "sales_date", "transaction_date", "created_date"],
    "order_id": ["order_id", "transaction_id", "invoice_id"],
    "quantity": ["quantity", "units", "qty", "units_sold"],
    "revenue": ["revenue", "sales", "total_sales", "sales_amount", "total"],
    "profit": ["profit", "net_profit", "gross_profit"],
    "image_url": ["image_url", "image", "photo_url"],
    "tags": ["tags", "keywords", "tag"],
}

def clean_col(value):
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip())).strip("_").lower()

def records_to_frame(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            return pd.json_normalize(obj)
        return pd.DataFrame({"description": [str(x) for x in obj]})
    if isinstance(obj, dict):
        for key in ("products", "data", "items", "records", "catalogue", "catalog"):
            if key in obj and isinstance(obj[key], (list, dict)):
                return records_to_frame(obj[key])
        # A single product object or a dictionary of records
        if obj and all(isinstance(v, dict) for v in obj.values()):
            return pd.json_normalize(obj, sep="_").T.reset_index()
        return pd.json_normalize(obj)
    return pd.DataFrame({"description": [str(obj)]})

def parse_uploaded(file):
    name = file.name
    ext = Path(name).suffix.lower().lstrip(".")
    raw = file.getvalue()
    bio = BytesIO(raw)
    if ext == "csv":
        return pd.read_csv(bio, sep=None, engine="python")
    if ext == "xlsx":
        return pd.read_excel(bio)
    if ext == "json":
        return records_to_frame(json.loads(raw.decode("utf-8-sig")))
    if ext == "jsonl":
        return pd.read_json(bio, lines=True)
    if ext == "parquet":
        return pd.read_parquet(bio)
    if ext in ("db", "sqlite"):
        # SQLite uploads are read-only in-memory copies for safe inspection.
        tmp = sqlite3.connect(":memory:")
        tmp.deserialize(raw)
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", tmp
        )["name"].tolist()
        if not tables:
            raise ValueError("The SQLite database contains no tables.")
        preferred = next((t for t in tables if t.lower() in ("products", "product", "catalog", "catalogue")), tables[0])
        frame = pd.read_sql_query(f'SELECT * FROM "{preferred}"', tmp)
        frame.attrs["source_table"] = preferred
        frame.attrs["available_tables"] = tables
        tmp.close()
        return frame
    if ext == "xml":
        try:
            return pd.read_xml(bio)
        except Exception:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(raw)
            rows = []
            for node in list(root):
                rows.append({child.tag: child.text for child in list(node)})
            return pd.DataFrame(rows)
    if ext == "html":
        tables = pd.read_html(bio)
        if tables:
            return max(tables, key=len)
        text_content = re.sub(r"<[^>]+>", " ", raw.decode("utf-8", "ignore"))
        return pd.DataFrame({"description": [re.sub(r"\s+", " ", html.unescape(text_content)).strip()]})
    if ext == "txt":
        content = raw.decode("utf-8-sig", "ignore")
        try:
            return pd.read_csv(BytesIO(raw), sep=None, engine="python")
        except Exception:
            return pd.DataFrame({"description": [content]})
    if ext == "pdf":
        if PdfReader is None:
            raise ImportError("Install pypdf to read PDF files.")
        reader = PdfReader(bio)
        pages = [page.extract_text() or "" for page in reader.pages]
        return pd.DataFrame({"description": ["\n".join(pages)]})
    if ext == "docx":
        if Document is None:
            raise ImportError("Install python-docx to read DOCX files.")
        doc = Document(bio)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        tables = []
        for table in doc.tables:
            rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            if len(rows) > 1:
                tables.append(pd.DataFrame(rows[1:], columns=rows[0]))
        if tables:
            return pd.concat(tables, ignore_index=True)
        return pd.DataFrame({"description": ["\n".join(paragraphs)]})
    if ext in ("yaml", "yml"):
        if yaml is None:
            raise ImportError("Install PyYAML to read YAML files.")
        return records_to_frame(yaml.safe_load(raw.decode("utf-8-sig")))
    raise ValueError(f"Unsupported file type: .{ext}")

def normalize_products(frame):
    df = frame.copy()
    df.columns = [clean_col(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    mapping = {}
    for target, candidates in ALIASES.items():
        for candidate in candidates:
            if candidate in df.columns:
                mapping[candidate] = target
                break
    df = df.rename(columns=mapping)
    if "product" not in df.columns:
        # Text-only documents still become searchable product knowledge records.
        text_col = next((c for c in df.columns if df[c].astype(str).str.len().gt(20).any()), None)
        if text_col:
            df["product"] = Path(str(df.iloc[0].get("source_file", "Document knowledge"))).stem if len(df) else "Document knowledge"
            df["description"] = df[text_col].astype(str)
        else:
            df["product"] = [f"Product {i+1}" for i in range(len(df))]
    defaults = {
        "product_id": [f"PRD-{i+1:05d}" for i in range(len(df))],
        "category": "Uncategorized", "brand": "Unknown", "description": "",
        "price": 0, "discount": 0, "rating": np.nan, "reviews_count": 0,
        "stock_quantity": 0, "supplier": "Not specified",
        "availability": "Unknown", "region": "Unspecified",
        "date": pd.NaT, "order_id": "", "quantity": 0,
        "revenue": np.nan, "profit": 0, "image_url": "", "tags": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    for col in ["price", "discount", "rating", "reviews_count", "stock_quantity", "quantity", "revenue", "profit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["price"] = df["price"].fillna(0)
    df["quantity"] = df["quantity"].fillna(0)
    df["revenue"] = df["revenue"].fillna(df["price"] * df["quantity"])
    df["profit"] = df["profit"].fillna(0)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["product", "category", "brand", "availability", "region"]:
        df[col] = df[col].fillna("Unknown").astype(str)
    df["product_id"] = df["product_id"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["discounted_price"] = (df["price"] * (1 - df["discount"].fillna(0).clip(0, 100) / 100)).round(2)
    return df

def load_local_folder():
    root = Path("datasets")
    frames, notes = [], []
    if not root.exists():
        return None, []
    for folder, extensions in FORMAT_FOLDERS.items():
        directory = root / folder
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower().lstrip(".") in extensions:
                try:
                    with path.open("rb") as f:
                        class LocalFile:
                            name = path.name
                            def getvalue(self): return f.read()
                        raw_frame = parse_uploaded(LocalFile())
                    raw_frame["source_file"] = path.name
                    frames.append(raw_frame)
                    notes.append(path.name)
                except Exception as exc:
                    notes.append(f"{path.name} (skipped: {exc})")
    if not frames:
        return None, notes
    return pd.concat(frames, ignore_index=True, sort=False), notes

def money(value):
    return f"₹{value:,.2f}"

def chart_layout(fig, height=370):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=15, r=15, t=55, b=15),
                      font=dict(color="#eee7fa"))
    return fig

def kpi_card(label, value, icon):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{icon} {label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

def export_frame(frame, fmt):
    buf = BytesIO()
    if fmt == "CSV":
        return frame.to_csv(index=False).encode("utf-8")
    if fmt == "Excel":
        frame.to_excel(buf, index=False, engine="openpyxl")
        return buf.getvalue()
    if fmt == "JSON":
        return frame.to_json(orient="records", indent=2, date_format="iso").encode("utf-8")
    if fmt == "JSONL":
        return frame.to_json(orient="records", lines=True, date_format="iso").encode("utf-8")
    if fmt == "Parquet":
        frame.to_parquet(buf, index=False)
        return buf.getvalue()
    if fmt == "XML":
        return frame.to_xml(index=False, root_name="products", row_name="product").encode("utf-8")
    if fmt == "HTML":
        return frame.to_html(index=False).encode("utf-8")
    if fmt == "TXT":
        return frame.to_csv(index=False, sep="\t").encode("utf-8")
    if fmt == "YAML":
        if yaml is None: raise ImportError("Install PyYAML to export YAML.")
        return yaml.safe_dump(json.loads(frame.to_json(orient="records", date_format="iso")), sort_keys=False, allow_unicode=True).encode("utf-8")
    if fmt == "DOCX":
        if Document is None: raise ImportError("Install python-docx to export DOCX.")
        doc = Document()
        doc.add_heading("Smart AI Assistant — Product Export", 0)
        doc.add_paragraph(f"Records: {len(frame)}")
        table = doc.add_table(rows=1, cols=len(frame.columns))
        table.style = "Light Shading Accent 1"
        for i, col in enumerate(frame.columns): table.rows[0].cells[i].text = str(col)
        for _, row in frame.head(500).iterrows():
            cells = table.add_row().cells
            for i, value in enumerate(row): cells[i].text = str(value)[:500]
        doc.save(buf)
        return buf.getvalue()
    raise ValueError("This export format is not supported here.")

# --------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.markdown("## 🤖 SMART AI")
    st.caption("PRODUCT & SALES INTELLIGENCE SUITE")
    st.divider()
    page = st.radio("WORKSPACE", [
        "Dashboard", "Product Analysis", "Product Catalogue",
        "Sales Analytics", "Regional Analysis", "Sales Prediction",
        "AI Assistant", "Business Insights", "Data Explorer", "Format Converter"
    ])
    st.divider()
    st.markdown("### 📂 Multi-Format Data Management")
    uploaded_files = st.file_uploader(
        "Upload product / sales datasets",
        type=FORMAT_TYPES, accept_multiple_files=True,
        help="CSV, XLSX, JSON, JSONL, Parquet, SQLite DB, XML, TXT, PDF, DOCX, HTML and YAML."
    )
    st.caption("12 dataset formats • Multiple files supported")
    st.divider()
    st.markdown("### 🔐 AI Configuration")
    st.caption("Add GROQ_API_KEY in Streamlit Secrets to enable AI chat.")
    st.markdown("[Project repository](https://github.com/ArunPandian-Git/Smart-AI-Assistant)")
    st.divider()
    st.markdown("### 📁 Local dataset folders")
    st.code("\n".join([f"datasets/{folder}/" for folder in FORMAT_FOLDERS]), language="text")

# --------------------------- DATA INGESTION ---------------------------
raw_frames, load_messages = [], []
if uploaded_files:
    for uploaded in uploaded_files:
        try:
            part = parse_uploaded(uploaded)
            part["source_file"] = uploaded.name
            raw_frames.append(part)
            load_messages.append(f"✓ {uploaded.name}: {len(part):,} rows")
        except Exception as exc:
            load_messages.append(f"✕ {uploaded.name}: {exc}")
else:
    local_df, local_notes = load_local_folder()
    if local_df is not None:
        raw_frames.append(local_df)
        load_messages.extend(local_notes)

if not raw_frames:
    # A helpful product schema sample keeps the app usable on first launch.
    sample = pd.DataFrame([
        {"product_id":"PRD-1001","product":"Aero Wireless Headphones","category":"Electronics","brand":"NovaTech","description":"Bluetooth over-ear headphones with active noise cancellation and 30-hour battery life.","price":4999,"discount":10,"rating":4.6,"reviews_count":284,"stock_quantity":42,"supplier":"Nova Distribution","availability":"In Stock","region":"South","quantity":3,"revenue":13497,"profit":3600},
        {"product_id":"PRD-1002","product":"Ergo Office Chair","category":"Furniture","brand":"WorkNest","description":"Ergonomic mesh chair with adjustable lumbar support and breathable backrest.","price":8999,"discount":5,"rating":4.4,"reviews_count":156,"stock_quantity":18,"supplier":"WorkNest Supply","availability":"In Stock","region":"West","quantity":2,"revenue":17098,"profit":4200},
        {"product_id":"PRD-1003","product":"Smart Fitness Watch","category":"Wearables","brand":"PulseIQ","description":"Water-resistant smartwatch with heart-rate tracking, GPS and sleep insights.","price":6499,"discount":15,"rating":4.2,"reviews_count":391,"stock_quantity":0,"supplier":"PulseIQ India","availability":"Out of Stock","region":"North","quantity":5,"revenue":27500,"profit":7000},
        {"product_id":"PRD-1004","product":"Ceramic Coffee Maker","category":"Home Appliances","brand":"BrewCraft","description":"Programmable 1.5 litre coffee maker with reusable filter and keep-warm function.","price":3299,"discount":0,"rating":4.1,"reviews_count":98,"stock_quantity":27,"supplier":"BrewCraft Traders","availability":"In Stock","region":"East","quantity":4,"revenue":13196,"profit":3100},
        {"product_id":"PRD-1005","product":"Ultralight Laptop Stand","category":"Accessories","brand":"DeskPro","description":"Foldable aluminium laptop stand with six height settings and anti-slip pads.","price":1799,"discount":8,"rating":4.7,"reviews_count":512,"stock_quantity":64,"supplier":"DeskPro Wholesale","availability":"In Stock","region":"South","quantity":8,"revenue":13240,"profit":5100},
    ])
    raw_frames = [sample]
    st.info("Demo catalogue loaded. Upload your own files or place datasets in the folders shown in the sidebar.")

try:
    raw_df = pd.concat(raw_frames, ignore_index=True, sort=False)
    df = normalize_products(raw_df)
except Exception as exc:
    st.error(f"Dataset processing error: {exc}")
    st.stop()

with st.sidebar:
    st.success(f"Loaded {len(df):,} product / transaction records")
    if load_messages:
        with st.expander("Ingestion report"):
            for message in load_messages: st.caption(message)
    categories = sorted(df["category"].dropna().unique().tolist())
    brands = sorted(df["brand"].dropna().unique().tolist())
    selected_categories = st.multiselect("Categories", categories, default=categories)
    selected_brands = st.multiselect("Brands", brands, default=brands)
    price_max = float(df["price"].max()) if len(df) else 0
    price_range = st.slider("Price range (₹)", 0.0, max(price_max, 1.0), (0.0, max(price_max, 1.0)))
    regions = sorted(df["region"].dropna().unique().tolist())
    selected_regions = st.multiselect("Regions", regions, default=regions)

filtered = df[
    df["category"].isin(selected_categories) &
    df["brand"].isin(selected_brands) &
    df["region"].isin(selected_regions) &
    df["price"].between(price_range[0], price_range[1])
].copy()

revenue = float(filtered["revenue"].sum()) if not filtered.empty else 0.0
profit = float(filtered["profit"].sum()) if not filtered.empty else 0.0
units = float(filtered["quantity"].sum()) if not filtered.empty else 0.0
orders = int(filtered["order_id"].replace("", np.nan).nunique()) if not filtered.empty else 0
if orders == 0: orders = len(filtered)
margin = profit / revenue * 100 if revenue else 0.0
avg_price = float(filtered["price"].mean()) if not filtered.empty else 0.0
stock_total = float(filtered["stock_quantity"].fillna(0).sum()) if not filtered.empty else 0.0

# --------------------------- DASHBOARD ---------------------------
if page == "Dashboard":
    st.title("📊 Executive Intelligence Dashboard")
    st.caption("A consolidated view of catalogue value, pricing, stock and sales performance.")
    if filtered.empty:
        st.warning("No records match the selected filters.")
    else:
        cols = st.columns(4)
        with cols[0]: kpi_card("Catalogue Records", f"{len(filtered):,}", "📦")
        with cols[1]: kpi_card("Catalogue Value", money((filtered["price"] * filtered["stock_quantity"]).sum()), "💎")
        with cols[2]: kpi_card("Average Price", money(avg_price), "🏷️")
        with cols[3]: kpi_card("Units in Stock", f"{stock_total:,.0f}", "🏬")
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            cat = filtered.groupby("category", as_index=False).agg(Products=("product_id","count"), Stock=("stock_quantity","sum"))
            st.plotly_chart(chart_layout(px.bar(cat, x="category", y="Products", color="category", title="Catalogue Distribution by Category")), use_container_width=True)
        with c2:
            brand = filtered.groupby("brand", as_index=False)["price"].mean().sort_values("price", ascending=False).head(12)
            st.plotly_chart(chart_layout(px.bar(brand, x="brand", y="price", title="Average Price by Brand")), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(chart_layout(px.histogram(filtered, x="price", color="category", title="Product Price Distribution")), use_container_width=True)
        with c4:
            stock = filtered.assign(stock_status=np.where(filtered["stock_quantity"] > 0, "In Stock", "Out of Stock")).groupby("stock_status", as_index=False).size()
            st.plotly_chart(chart_layout(px.pie(stock, names="stock_status", values="size", hole=.58, title="Inventory Availability")), use_container_width=True)
        st.subheader("⭐ Highest-rated products")
        st.dataframe(filtered.sort_values("rating", ascending=False)[["product_id","product","brand","category","price","rating","reviews_count","stock_quantity"]].head(10), use_container_width=True, hide_index=True)

# --------------------------- PRODUCT ANALYSIS ---------------------------
elif page == "Product Analysis":
    st.title("📦 Product Intelligence Center")
    st.caption("Discover category performance, pricing patterns, customer ratings and inventory signals.")
    if filtered.empty:
        st.warning("No product data for these filters.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Products", f"{filtered['product'].nunique():,}")
        c2.metric("Average Rating", f"{filtered['rating'].mean():.2f} / 5" if filtered["rating"].notna().any() else "N/A")
        c3.metric("Average Discount", f"{filtered['discount'].mean():.1f}%")
        c4.metric("Out of Stock", f"{(filtered['stock_quantity'].fillna(0)<=0).sum():,}")
        c1,c2 = st.columns(2)
        with c1:
            cat = filtered.groupby("category", as_index=False).agg(Avg_Price=("price","mean"), Products=("product_id","count"))
            st.plotly_chart(chart_layout(px.bar(cat, x="category", y="Avg_Price", color="category", title="Average Price by Category")), use_container_width=True)
        with c2:
            st.plotly_chart(chart_layout(px.scatter(filtered, x="price", y="rating", size="reviews_count", color="category", hover_name="product", title="Price vs Customer Rating")), use_container_width=True)
        st.subheader("Product performance table")
        st.dataframe(filtered.sort_values("price", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download product report", filtered.to_csv(index=False).encode("utf-8"), "product_analysis.csv", "text/csv")

# --------------------------- PRODUCT CATALOGUE ---------------------------
elif page == "Product Catalogue":
    st.title("🛍️ Product Catalogue")
    search = st.text_input("Search products, descriptions, brands or tags")
    view = filtered.copy()
    if search:
        mask = view.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        view = view[mask]
    st.caption(f"{len(view):,} matching catalogue entries")
    for start in range(0, min(len(view), 60), 3):
        cols = st.columns(3)
        for col, (_, row) in zip(cols, view.iloc[start:start+3].iterrows()):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {row['product']}")
                    st.caption(f"{row['brand']} · {row['category']}")
                    st.markdown(f"**{money(row['discounted_price'])}**  ")
                    if row["discount"] > 0: st.caption(f"~~{money(row['price'])}~~ · {row['discount']:.0f}% off")
                    st.write(str(row["description"])[:260] or "No description supplied.")
                    st.caption(f"⭐ {row['rating'] if pd.notna(row['rating']) else 'N/A'} · Stock: {row['stock_quantity']:,.0f} · {row['availability']}")
                    if str(row["image_url"]).startswith(("http://","https://")):
                        st.image(row["image_url"], use_container_width=True)
    if len(view) > 60: st.info("Showing first 60 matching product cards. Use Data Explorer to inspect and export the complete result.")

# --------------------------- SALES ANALYTICS ---------------------------
elif page == "Sales Analytics":
    st.title("📈 Sales Intelligence Center")
    st.caption("Sales metrics are calculated when quantity, revenue, profit or order fields are present.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Revenue", money(revenue))
    c2.metric("Profit", money(profit))
    c3.metric("Units Sold", f"{units:,.0f}")
    c4.metric("Profit Margin", f"{margin:.2f}%")
    if filtered["date"].notna().any():
        dated = filtered.dropna(subset=["date"]).copy()
        monthly = dated.assign(month=dated["date"].dt.to_period("M").astype(str)).groupby("month", as_index=False).agg(Revenue=("revenue","sum"), Profit=("profit","sum"), Units=("quantity","sum"))
        st.plotly_chart(chart_layout(px.line(monthly, x="month", y=["Revenue","Profit"], markers=True, title="Monthly Revenue & Profit")), use_container_width=True)
        daily = dated.groupby(dated["date"].dt.date, as_index=False)["revenue"].sum()
        daily.columns = ["Date","Revenue"]
        st.plotly_chart(chart_layout(px.bar(daily, x="Date", y="Revenue", title="Daily Revenue Performance")), use_container_width=True)
    else:
        st.info("No transaction dates were detected. Upload sales records with a Date / Order_Date column to activate time-series analysis.")
    st.subheader("Sales record detail")
    st.dataframe(filtered[["product","category","quantity","price","revenue","profit","date","region"]], use_container_width=True, hide_index=True)

# --------------------------- REGIONAL ANALYSIS ---------------------------
elif page == "Regional Analysis":
    st.title("🌍 Regional Intelligence")
    summary = filtered.groupby("region", as_index=False).agg(Revenue=("revenue","sum"), Profit=("profit","sum"), Products=("product_id","count"), Stock=("stock_quantity","sum"))
    if summary.empty: st.warning("No regional records.")
    else:
        c1,c2 = st.columns(2)
        with c1: st.plotly_chart(chart_layout(px.bar(summary, x="region", y="Revenue", color="region", title="Revenue by Region")), use_container_width=True)
        with c2: st.plotly_chart(chart_layout(px.pie(summary, names="region", values="Stock", hole=.5, title="Inventory by Region")), use_container_width=True)
        st.dataframe(summary, use_container_width=True, hide_index=True)

# --------------------------- SALES PREDICTION ---------------------------
elif page == "Sales Prediction":
    st.title("🔮 Sales & Demand Forecast Studio")
    dated = filtered.dropna(subset=["date"]).copy()
    if len(dated) < 5:
        st.info("At least five dated sales observations are recommended. Add Date, Quantity and Revenue fields to activate forecasting.")
    else:
        daily = dated.groupby(dated["date"].dt.normalize(), as_index=False).agg(revenue=("revenue","sum"), demand=("quantity","sum")).sort_values("date")
        daily["day_index"] = (daily["date"] - daily["date"].min()).dt.days
        target = st.selectbox("Forecast target", ["Revenue","Product demand"], index=0)
        target_col = "revenue" if target == "Revenue" else "demand"
        horizon = st.slider("Forecast horizon (days)", 7, 90, 30)
        model_choice = st.selectbox("Model", ["Linear Regression","Random Forest"])
        if st.button("🚀 Train model and forecast", type="primary"):
            if daily[target_col].nunique() < 2: st.warning("The selected target has insufficient variation for a useful forecast.")
            else:
                split = max(1, int(len(daily)*.8))
                X, y = daily[["day_index"]], daily[target_col]
                model = LinearRegression() if model_choice == "Linear Regression" else RandomForestRegressor(n_estimators=150, random_state=42)
                model.fit(X.iloc[:split], y.iloc[:split])
                test = model.predict(X.iloc[split:]) if split < len(daily) else np.array([])
                if len(test):
                    st.metric("Validation MAE", f"{mean_absolute_error(y.iloc[split:], test):,.2f}")
                    if len(test)>1: st.metric("Validation R²", f"{r2_score(y.iloc[split:], test):.3f}")
                model.fit(X,y)
                dates = pd.date_range(daily["date"].max()+pd.Timedelta(days=1), periods=horizon, freq="D")
                future_x = (dates-daily["date"].min()).days.to_numpy().reshape(-1,1)
                forecast = np.maximum(model.predict(future_x),0)
                forecast_df = pd.DataFrame({"Date":dates, f"Predicted {target}":forecast})
                fig = px.line(daily, x="date", y=target_col, title=f"Historical and Forecast {target}")
                fig.add_scatter(x=forecast_df["Date"], y=forecast_df[f"Predicted {target}"], mode="lines+markers", name="Forecast")
                st.plotly_chart(chart_layout(fig,430), use_container_width=True)
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download forecast", forecast_df.to_csv(index=False).encode(), "smart_ai_forecast.csv", "text/csv")
                st.caption("Forecasts are estimates based on the uploaded history, not guaranteed outcomes.")

# --------------------------- AI ASSISTANT ---------------------------
elif page == "AI Assistant":
    st.title("🧠 Smart AI Product & Business Assistant")
    st.caption("Ask questions about products, pricing, stock, category performance and sales.")
    api_key = (st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None) or os.getenv("GROQ_API_KEY")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role":"assistant","content":"Welcome! I can analyze your product catalogue and sales data. Ask about top-rated products, price comparisons, stock risks, category performance or revenue."}]
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    prompt = st.chat_input("Ask your product intelligence question...")
    if prompt:
        st.session_state.chat_messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            if not api_key or Groq is None:
                answer = "AI service is not configured. Add GROQ_API_KEY under Streamlit Cloud → App → Settings → Secrets. All dashboard, analytics and conversion features work without this key."
                st.info(answer)
            else:
                top_products = filtered.groupby("product")["revenue"].sum().nlargest(8).to_dict()
                top_rated = filtered.nlargest(8, "rating")[["product","rating","price"]].to_dict("records")
                context = (
                    f"Rows: {len(filtered)}; total price value: {float((filtered.price*filtered.stock_quantity).sum()):.2f}; "
                    f"revenue: {revenue:.2f}; profit: {profit:.2f}; stock units: {stock_total:.0f}; "
                    f"average rating: {filtered.rating.mean()}; top products by revenue: {top_products}; "
                    f"top rated products: {top_rated}; categories: {filtered.category.value_counts().to_dict()}."
                )
                try:
                    client = Groq(api_key=api_key)
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[{"role":"system","content":"You are a senior product intelligence and business analyst. Answer only from the supplied dataset context. Clearly separate observed facts from recommendations and never invent metrics. "+context},
                                  *[{"role":m["role"],"content":m["content"]} for m in st.session_state.chat_messages[-8:]]],
                        temperature=.3, max_tokens=1000,
                    )
                    answer = response.choices[0].message.content or "No response returned."
                    st.markdown(answer)
                except Exception as exc:
                    answer = f"AI request failed: {exc}. Check the Groq key and network connection."
                    st.error(answer)
        st.session_state.chat_messages.append({"role":"assistant","content":answer})
    if st.button("🧹 Clear conversation"):
        st.session_state.chat_messages = []
        st.rerun()

# --------------------------- BUSINESS INSIGHTS ---------------------------
elif page == "Business Insights":
    st.title("💡 Business Insights & Recommendations")
    if filtered.empty: st.warning("No data available for insights.")
    else:
        top_product = filtered.groupby("product")["revenue"].sum().idxmax()
        top_category = filtered.groupby("category")["revenue"].sum().idxmax()
        top_brand = filtered.groupby("brand")["rating"].mean().idxmax() if filtered["rating"].notna().any() else "N/A"
        out_count = int((filtered["stock_quantity"].fillna(0)<=0).sum())
        st.success(f"🏆 Revenue leader: **{top_product}**")
        st.success(f"📦 Leading category by revenue: **{top_category}**")
        st.info(f"⭐ Highest average-rated brand: **{top_brand}**")
        st.subheader("Actionable observations")
        st.markdown(f"- Review replenishment for **{out_count}** products with zero or unavailable stock.")
        st.markdown(f"- Prioritize visibility for **{top_product}**, the leading product by recorded revenue.")
        st.markdown(f"- Evaluate pricing and assortment in **{top_category}**, the leading revenue category.")
        st.markdown(f"- Average catalogue discount is **{filtered['discount'].mean():.1f}%**; compare discount depth with profit margin ({margin:.2f}%) before increasing promotions.")
        st.caption("Insights are calculated from the selected dataset; they are not external market research.")

# --------------------------- DATA EXPLORER ---------------------------
elif page == "Data Explorer":
    st.title("🗂️ Universal Data Explorer")
    st.caption("Inspect, search, validate and export the normalized dataset.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", f"{len(filtered):,}")
    c2.metric("Columns", f"{filtered.shape[1]:,}")
    c3.metric("Missing cells", f"{int(filtered.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(filtered.duplicated().sum()):,}")
    search_text = st.text_input("Search across all fields")
    view = filtered.copy()
    if search_text:
        mask = view.astype(str).apply(lambda col: col.str.contains(search_text, case=False, na=False)).any(axis=1)
        view = view[mask]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered data (CSV)", view.to_csv(index=False).encode("utf-8"), "smart_ai_filtered_data.csv", "text/csv")
    with st.expander("Data quality report"):
        quality = pd.DataFrame({"Column":df.columns,"Data Type":[str(t) for t in df.dtypes],"Missing Values":df.isna().sum().values,"Missing %":(df.isna().mean().values*100).round(2)})
        st.dataframe(quality, use_container_width=True, hide_index=True)
        st.write("Duplicate rows:", int(df.duplicated().sum()))
        st.write("Blank product names:", int(df["product"].astype(str).str.strip().eq("").sum()))

# --------------------------- FORMAT CONVERTER ---------------------------
elif page == "Format Converter":
    st.title("🔄 Multi-Format Dataset Converter")
    st.caption("Convert the currently filtered product dataset into a portable format.")
    target = st.selectbox("Export format", ["CSV","Excel","JSON","JSONL","Parquet","XML","HTML","TXT","YAML","DOCX"])
    if st.button("⚙️ Prepare export", type="primary"):
        try:
            payload = export_frame(filtered, target)
            extensions = {"CSV":"csv","Excel":"xlsx","JSON":"json","JSONL":"jsonl","Parquet":"parquet","XML":"xml","HTML":"html","TXT":"txt","YAML":"yaml","DOCX":"docx"}
            mime = {"CSV":"text/csv","Excel":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet","JSON":"application/json","JSONL":"application/json","Parquet":"application/octet-stream","XML":"application/xml","HTML":"text/html","TXT":"text/plain","YAML":"text/yaml","DOCX":"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
            st.download_button(f"⬇️ Download {target}", payload, f"smart_ai_products.{extensions[target]}", mime[target])
            st.success(f"Export prepared: {len(filtered):,} records.")
        except Exception as exc:
            st.error(f"Export failed: {exc}")
    st.info("PDF and SQLite are supported as ingestion sources. The converter currently exports CSV, Excel, JSON, JSONL, Parquet, XML, HTML, TXT, YAML and DOCX.")

# --------------------------- FOOTER ---------------------------
st.divider()
st.markdown(
    '<div style="text-align:center;color:#9f91b4;font-size:12px;">'
    'SMART AI ASSISTANT · PRODUCT & SALES INTELLIGENCE PLATFORM · '
    'Multi-format ingestion · Plotly · Scikit-learn · Groq'
    '</div>', unsafe_allow_html=True
)
