# ============================================================
# SMART AI ASSISTANT — UPLOAD-DRIVEN PRODUCT & BUSINESS INTELLIGENCE
# No built-in/demo dataset. Nothing is analysed until a file is uploaded.
# ============================================================
from pathlib import Path
from io import BytesIO
import json, re, sqlite3, html
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from sklearn.linear_model import LinearRegression
except Exception:
    LinearRegression = None
try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:
    RandomForestRegressor = None
try:
    from groq import Groq
except Exception:
    Groq = None
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

st.set_page_config(page_title="Smart AI Assistant | Intelligence Studio",
                   page_icon="✦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(circle at 10% 0%,#292044 0,#12111d 43%,#0b0d14 100%);color:#f4f0ff}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#171426,#211a32);border-right:1px solid #403451}
.block-container{max-width:1500px;padding-top:1.6rem}
.hero{padding:28px 32px;border:1px solid #594779;border-radius:22px;background:linear-gradient(115deg,rgba(111,76,164,.32),rgba(30,25,45,.72));margin-bottom:22px}
.hero h1{font-size:2.15rem;font-weight:800;color:#e4d8ff;margin:0}
.hero p{color:#c9bfdc;margin:.45rem 0 0}
.eyebrow{letter-spacing:2px;color:#bea5ff;font-size:11px;font-weight:800}
div[data-testid="stMetric"]{background:#211b2d;border:1px solid #443654;padding:14px;border-radius:14px}
.panel{padding:18px;border:1px solid #403450;border-radius:16px;background:rgba(35,29,49,.78)}
.small{font-size:12px;color:#aaa0bd}
</style>
<div class="hero">
<div class="eyebrow">✦ UPLOAD • UNDERSTAND • PREDICT • DECIDE</div>
<h1>Smart AI Assistant</h1>
<p>Upload your own business file to unlock product, sales, value and predictive intelligence.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------- INGESTION -----------------------
SUPPORTED = ["csv","xlsx","xls","json","jsonl","ndjson","parquet","xml","txt","pdf","docx","html","htm","yaml","yml","db","sqlite","sqlite3"]

ALIASES = {
    "product_id":["product_id","id","sku","item_id","product_code"],
    "product":["product","product_name","name","item","title"],
    "category":["category","product_category","department","type"],
    "brand":["brand","manufacturer","vendor"],
    "description":["description","product_description","details","summary"],
    "price":["price","unit_price","selling_price","sale_price","unit_cost","cost"],
    "discount":["discount","discount_percent","discount_percentage"],
    "rating":["rating","average_rating","stars"],
    "reviews_count":["reviews_count","review_count","reviews"],
    "stock_quantity":["stock_quantity","stock","inventory","quantity_in_stock","available_stock"],
    "supplier":["supplier","supplier_name"],
    "availability":["availability","status","stock_status"],
    "country":["country"],"state":["state"],"city":["city"],
    "region":["region","market","territory"],
    "created_date":["created_date","date","order_date","sales_date","transaction_date","timestamp"],
    "quantity":["quantity","units","qty","units_sold","sales_quantity"],
    "revenue":["revenue","sales","total_sales","sales_amount","total_revenue"],
    "profit":["profit","net_profit","gross_profit"],
    "image_url":["image_url","image","photo_url"],
    "tags":["tags","keywords","tag"],
}

def norm_col(x):
    return re.sub(r"_+","_",re.sub(r"[^a-zA-Z0-9]+","_",str(x).strip())).strip("_").lower()

def json_frame(obj):
    if isinstance(obj,list):
        return pd.json_normalize(obj) if obj and isinstance(obj[0],dict) else pd.DataFrame({"content":[str(x) for x in obj]})
    if isinstance(obj,dict):
        for k in ("products","data","items","records","rows","results"):
            if k in obj and isinstance(obj[k],(list,dict)):
                return json_frame(obj[k])
        return pd.json_normalize(obj)
    return pd.DataFrame({"content":[str(obj)]})

def parse_upload(file):
    name=file.name
    raw=file.getvalue()
    ext=Path(name).suffix.lower()
    bio=BytesIO(raw)
    if ext==".csv": return pd.read_csv(bio,sep=None,engine="python")
    if ext in (".xlsx",".xls"): return pd.read_excel(bio)
    if ext==".json": return json_frame(json.loads(raw.decode("utf-8-sig")))
    if ext in (".jsonl",".ndjson"): return pd.read_json(bio,lines=True)
    if ext==".parquet": return pd.read_parquet(bio)
    if ext==".xml": return pd.read_xml(bio)
    if ext in (".html",".htm"):
        try:
            tables=pd.read_html(bio)
            if tables: return max(tables,key=len)
        except Exception: pass
        txt=re.sub(r"\s+"," ",html.unescape(re.sub(r"<[^>]+>"," ",raw.decode("utf-8","ignore"))))
        return pd.DataFrame({"content":[txt]})
    if ext==".txt":
        try: return pd.read_csv(bio,sep=None,engine="python")
        except Exception: return pd.DataFrame({"content":[raw.decode("utf-8-sig","ignore")]})
    if ext==".pdf":
        if PdfReader is None: raise ImportError("Install pypdf to read PDF files.")
        return pd.DataFrame({"content":["\n".join(p.extract_text() or "" for p in PdfReader(bio).pages)]})
    if ext==".docx":
        if Document is None: raise ImportError("Install python-docx to read DOCX files.")
        doc=Document(bio)
        return pd.DataFrame({"content":["\n".join(p.text for p in doc.paragraphs)]})
    if ext in (".yaml",".yml"):
        if yaml is None: raise ImportError("Install PyYAML to read YAML files.")
        return json_frame(yaml.safe_load(raw.decode("utf-8-sig")))
    if ext in (".db",".sqlite",".sqlite3"):
        con=sqlite3.connect(":memory:")
        con.deserialize(raw)
        tabs=pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'",con)["name"].tolist()
        if not tabs: raise ValueError("No tables found in SQLite database.")
        table=next((t for t in tabs if t.lower() in ("products","product","sales","transactions")),tabs[0])
        df=pd.read_sql_query('SELECT * FROM "'+table.replace('"','""')+'"',con)
        con.close()
        return df
    raise ValueError(f"Unsupported file type: {ext}")

def normalize(df):
    x=df.copy()
    x.columns=[norm_col(c) for c in x.columns]
    x=x.loc[:,~pd.Index(x.columns).duplicated()]
    ren={}
    for target,choices in ALIASES.items():
        found=next((c for c in choices if c in x.columns),None)
        if found and found!=target: ren[found]=target
    x=x.rename(columns=ren)
    numeric_candidates=["price","discount","rating","reviews_count","stock_quantity","quantity","revenue","profit"]
    for c in numeric_candidates:
        if c in x.columns:
            # Handles currency symbols and comma-separated numeric values.
            x[c]=pd.to_numeric(x[c].astype(str).str.replace(r"[^0-9.\-]","",regex=True),errors="coerce")
    if "created_date" in x.columns:
        x["created_date"]=pd.to_datetime(x["created_date"],errors="coerce")
    if "product" not in x.columns:
        x["product"]=x["product_id"].astype(str) if "product_id" in x.columns else [f"Record {i+1}" for i in range(len(x))]
    return x

def money(v):
    return f"₹{v:,.2f}" if pd.notna(v) else "Not available"

# ----------------------- SIDEBAR -----------------------
with st.sidebar:
    st.markdown("## ✦ SMART AI")
    st.caption("UPLOAD-DRIVEN INTELLIGENCE")
    page=st.radio("WORKSPACE",["Upload & Overview","Product Intelligence","Worth & Valuation",
                               "Sales Analytics","AI Intelligence Studio","Data Explorer","Data Quality"])
    st.divider()
    st.markdown("**YOUR DATA ONLY**")
    st.caption("No sample records or preloaded demo dataset are used.")
    st.caption("Upload one file to start.")

# ----------------------- UPLOAD GATE -----------------------
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df=None
    st.session_state.uploaded_name=None

if page=="Upload & Overview" or st.session_state.uploaded_df is None:
    st.subheader("Upload your dataset")
    st.write("Choose one file. The application will inspect its columns, detect useful business fields, and build analysis from that file only.")
    uploaded=st.file_uploader("Upload one dataset",type=SUPPORTED,key="primary_upload")
    if uploaded is not None:
        try:
            raw_df=parse_upload(uploaded)
            if raw_df is None or raw_df.empty:
                st.error("The uploaded file contains no readable records.")
                st.stop()
            st.session_state.uploaded_df=normalize(raw_df)
            st.session_state.uploaded_name=uploaded.name
            st.success(f"Successfully loaded {len(raw_df):,} rows and {len(raw_df.columns):,} columns.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not read this file: {e}")
            st.info("For Excel install openpyxl; for Parquet pyarrow; for PDF pypdf; for DOCX python-docx; for YAML PyYAML.")
    if st.session_state.uploaded_df is None:
        st.info("No dataset is loaded. Upload a file above to activate the intelligence workspace.")
        st.markdown("**Supported formats:** CSV, Excel, JSON, JSONL, Parquet, XML, TXT, PDF, DOCX, HTML, YAML and SQLite.")
        st.markdown("**Important:** PDF/DOCX/TXT are treated as text documents. Structured product valuation and forecasting require suitable numeric/date columns.")
        st.stop()

df=st.session_state.uploaded_df.copy()
name=st.session_state.uploaded_name or "Uploaded file"

with st.sidebar:
    st.success(f"Loaded: {name}")
    if st.button("Remove dataset / upload another",use_container_width=True):
        st.session_state.uploaded_df=None
        st.session_state.uploaded_name=None
        st.rerun()

# Filters apply only when matching columns exist.
filtered=df.copy()
with st.sidebar:
    filter_cols=[c for c in ["category","brand","country","state","region"] if c in df.columns]
    for col in filter_cols:
        vals=sorted(df[col].dropna().astype(str).unique().tolist())
        selected=st.multiselect(col.replace("_"," ").title(),vals,default=vals)
        filtered=filtered[filtered[col].astype(str).isin(selected)]

cols=set(filtered.columns)
has_price="price" in cols
has_stock="stock_quantity" in cols
has_quantity="quantity" in cols
has_revenue="revenue" in cols
has_profit="profit" in cols
has_date="created_date" in cols and filtered["created_date"].notna().any()

# ----------------------- COMMON HEADER -----------------------
st.caption(f"ACTIVE DATASET: {name}  ·  {len(filtered):,} filtered rows  ·  {len(filtered.columns)} columns")

# ----------------------- OVERVIEW -----------------------
if page=="Upload & Overview":
    st.title("Dataset Overview")
    a,b,c,d=st.columns(4)
    a.metric("Records",f"{len(filtered):,}")
    b.metric("Columns",f"{len(filtered.columns):,}")
    b2=int(filtered.duplicated().sum())
    c.metric("Duplicate rows",f"{b2:,}")
    d.metric("Missing cells",f"{int(filtered.isna().sum().sum()):,}")
    st.subheader("Dataset preview")
    st.dataframe(filtered.head(100),use_container_width=True,hide_index=True)
    st.subheader("Detected fields")
    detected=[c for c in ["product","category","brand","price","stock_quantity","quantity","revenue","profit","rating","created_date","country","state","region"] if c in cols]
    st.write(", ".join(detected) if detected else "No standard product/sales fields detected. All original columns are retained below.")
    st.download_button("Download normalized dataset",filtered.to_csv(index=False).encode("utf-8"),"analysed_dataset.csv","text/csv")

# ----------------------- PRODUCT INTELLIGENCE -----------------------
elif page=="Product Intelligence":
    st.title("Product Intelligence")
    if "product" in cols:
        a,b,c=st.columns(3)
        a.metric("Unique products",f"{filtered['product'].nunique():,}")
        c.metric("Product categories",f"{filtered['category'].nunique():,}" if "category" in cols else "N/A")
        b.metric("Brands",f"{filtered['brand'].nunique():,}" if "brand" in cols else "N/A")
        if "category" in cols:
            cat=filtered.groupby("category",as_index=False).size()
            st.plotly_chart(px.bar(cat,x="category",y="size",color="category",title="Records by category",template="plotly_dark"),use_container_width=True)
        if has_price:
            st.metric("Average listed price",money(filtered["price"].mean()))
            if "category" in cols:
                p=filtered.groupby("category",as_index=False)["price"].mean()
                st.plotly_chart(px.bar(p,x="category",y="price",title="Average price by category",template="plotly_dark"),use_container_width=True)
        if "rating" in cols:
            st.metric("Average rating",f"{filtered['rating'].mean():.2f}/5" if filtered["rating"].notna().any() else "N/A")
        st.dataframe(filtered,use_container_width=True,hide_index=True)
    else:
        st.warning("No product/name column was detected. Use Data Explorer to inspect the uploaded columns.")

# ----------------------- WORTH & VALUATION -----------------------
elif page=="Worth & Valuation":
    st.title("Worth & Inventory Valuation")
    if has_price:
        st.metric("Average unit price",money(filtered["price"].mean()))
        st.metric("Total listed price value (sum of row prices)",money(filtered["price"].sum()))
        if has_stock:
            inv=(filtered["price"].fillna(0)*filtered["stock_quantity"].fillna(0)).sum()
            st.metric("Estimated inventory value (price × stock)",money(inv))
            st.caption("This is an estimate based on the uploaded unit price and stock quantity. It is not a company valuation.")
            val=filtered.assign(estimated_inventory_value=filtered["price"].fillna(0)*filtered["stock_quantity"].fillna(0))
            st.dataframe(val.sort_values("estimated_inventory_value",ascending=False),use_container_width=True,hide_index=True)
            if "product" in cols:
                top=val.groupby("product",as_index=False)["estimated_inventory_value"].sum().nlargest(15,"estimated_inventory_value")
                st.plotly_chart(px.bar(top,x="product",y="estimated_inventory_value",title="Top products by estimated inventory value",template="plotly_dark"),use_container_width=True)
        else:
            st.warning("No stock/inventory quantity column found. Upload stock_quantity or inventory to calculate total inventory value.")
    else:
        st.warning("No price/value column was detected. The app cannot calculate product worth without a numeric price, cost, or value field.")
        st.write("Detected columns:",", ".join(filtered.columns.astype(str)))
    if has_revenue:
        st.metric("Total recorded revenue",money(filtered["revenue"].sum()))
    if has_profit:
        st.metric("Total recorded profit",money(filtered["profit"].sum()))

# ----------------------- SALES -----------------------
elif page=="Sales Analytics":
    st.title("Sales Analytics")
    if has_revenue:
        st.metric("Total revenue",money(filtered["revenue"].sum()))
    else:
        st.info("Revenue column not detected.")
    if has_profit:
        st.metric("Total profit",money(filtered["profit"].sum()))
    if has_quantity:
        st.metric("Total quantity",f"{filtered['quantity'].sum():,.2f}")
    if has_date and has_revenue:
        t=filtered.dropna(subset=["created_date"]).copy()
        t["Period"]=t["created_date"].dt.to_period("M").astype(str)
        monthly=t.groupby("Period",as_index=False)["revenue"].sum()
        st.plotly_chart(px.line(monthly,x="Period",y="revenue",markers=True,title="Revenue over time",template="plotly_dark"),use_container_width=True)
    elif has_date and has_quantity:
        t=filtered.dropna(subset=["created_date"]).copy()
        t["Period"]=t["created_date"].dt.to_period("M").astype(str)
        monthly=t.groupby("Period",as_index=False)["quantity"].sum()
        st.plotly_chart(px.line(monthly,x="Period",y="quantity",markers=True,title="Quantity over time",template="plotly_dark"),use_container_width=True)
    else:
        st.info("For time-based sales analysis, include a date column plus revenue or quantity.")
    st.dataframe(filtered,use_container_width=True,hide_index=True)

# ----------------------- AI INTELLIGENCE STUDIO -----------------------
elif page=="AI Intelligence Studio":
    st.title("AI Intelligence Studio")
    st.caption("Analysis and predictions are based exclusively on the uploaded dataset.")
    tab1,tab2,tab3=st.tabs(["Business Copilot","Demand Forecast","Smart Insights"])
    with tab1:
        st.subheader("Ask your data")
        question=st.text_area("Ask a business question",placeholder="Example: Which products have the highest inventory value?")
        api_key=st.text_input("Groq API key (optional)",type="password",help="Alternatively set GROQ_API_KEY in Streamlit Secrets.")
        if st.button("Ask AI",type="primary",disabled=not question.strip()):
            key=api_key or st.secrets.get("GROQ_API_KEY","")
            if not key or Groq is None:
                st.warning("Configure GROQ_API_KEY in Streamlit Secrets and add groq to requirements.txt to enable conversational AI.")
            else:
                try:
                    # Send compact schema + computed summary, not the entire raw file.
                    summary={"file":name,"rows":len(filtered),"columns":list(filtered.columns),
                             "numeric_summary":filtered.select_dtypes(include=np.number).describe().round(2).to_dict(),
                             "sample_records":filtered.head(12).replace({np.nan:None}).to_dict(orient="records")}
                    client=Groq(api_key=key)
                    response=client.chat.completions.create(
                        model = "openai/gpt-oss-120b",
                        messages=[
                            {"role":"system","content":"You are a careful business data analyst. Answer only from supplied dataset context. State assumptions and limitations. Do not invent missing values. Clearly distinguish observed facts from estimates."},
                            {"role":"user","content":"Dataset context:\n"+json.dumps(summary,default=str)+"\n\nQuestion: "+question}
                        ],temperature=0.2,max_tokens=1200)
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"AI request failed: {e}")
    with tab2:
        st.subheader("Demand forecasting")
        if not (has_date and has_quantity):
            st.warning("Forecasting needs both a usable date column and a numeric quantity/units-sold column. Your uploaded file does not currently contain both.")
        elif LinearRegression is None:
            st.error("Install scikit-learn to enable forecasting.")
        else:
            t=filtered.dropna(subset=["created_date","quantity"]).copy()
            t["period"]=t["created_date"].dt.to_period("M").dt.to_timestamp()
            series=t.groupby("period")["quantity"].sum().sort_index()
            if len(series)<3:
                st.warning("At least 3 distinct monthly periods are needed for a basic forecast.")
            else:
                horizon=st.slider("Months to forecast",1,12,3)
                X=np.arange(len(series)).reshape(-1,1)
                model=RandomForestRegressor(n_estimators=150,random_state=42) if RandomForestRegressor and len(series)>=6 else LinearRegression()
                model.fit(X,series.values)
                future_x=np.arange(len(series),len(series)+horizon).reshape(-1,1)
                pred=np.maximum(0,model.predict(future_x))
                future_dates=pd.date_range(series.index[-1]+pd.offsets.MonthBegin(1),periods=horizon,freq="MS")
                forecast=pd.DataFrame({"period":future_dates,"quantity":pred.round(2),"type":"Forecast"})
                history=pd.DataFrame({"period":series.index,"quantity":series.values,"type":"Actual"})
                combined=pd.concat([history,forecast],ignore_index=True)
                st.plotly_chart(px.line(combined,x="period",y="quantity",color="type",markers=True,title="Actual demand and model forecast",template="plotly_dark"),use_container_width=True)
                st.dataframe(forecast,use_container_width=True,hide_index=True)
                st.caption("Forecast is an estimate from historical monthly quantities, not a guarantee. Sparse or irregular data reduces reliability.")
    with tab3:
        st.subheader("Dataset-derived business signals")
        if has_price and has_stock:
            value=filtered.assign(_value=filtered["price"].fillna(0)*filtered["stock_quantity"].fillna(0))
            st.metric("Estimated inventory value",money(value["_value"].sum()))
            st.write("Highest inventory value products")
            if "product" in cols:
                st.dataframe(value.groupby("product",as_index=False)["_value"].sum().nlargest(10,"_value"),use_container_width=True,hide_index=True)
        if has_stock:
            low=filtered[filtered["stock_quantity"]<=0]
            st.metric("Rows with zero/negative stock",f"{len(low):,}")
            if not low.empty: st.dataframe(low.head(20),use_container_width=True,hide_index=True)
        if "rating" in cols:
            rated=filtered.dropna(subset=["rating"])
            if not rated.empty:
                st.metric("Average customer rating",f"{rated['rating'].mean():.2f}/5")
                if "product" in cols:
                    st.write("Lowest-rated products")
                    st.dataframe(rated.sort_values("rating").head(10),use_container_width=True,hide_index=True)
        if not (has_price or has_stock or "rating" in cols):
            st.info("No standard price, stock or rating fields detected. Explore your raw columns in Data Explorer.")

# ----------------------- EXPLORER -----------------------
elif page=="Data Explorer":
    st.title("Data Explorer")
    search=st.text_input("Search all columns")
    view=filtered.copy()
    if search:
        view=view[view.astype(str).apply(lambda c:c.str.contains(search,case=False,na=False)).any(axis=1)]
    st.caption(f"{len(view):,} rows · {len(view.columns)} columns")
    st.dataframe(view,use_container_width=True,hide_index=True)
    st.download_button("Download analysed CSV",view.to_csv(index=False).encode("utf-8"),"smart_ai_analysed_data.csv","text/csv")
    st.download_button("Download JSON",view.to_json(orient="records",indent=2,date_format="iso").encode("utf-8"),"smart_ai_analysed_data.json","application/json")

# ----------------------- DATA QUALITY -----------------------
elif page=="Data Quality":
    st.title("Data Quality")
    a,b,c=st.columns(3)
    a.metric("Rows",f"{len(filtered):,}")
    b.metric("Columns",f"{len(filtered.columns):,}")
    c.metric("Duplicate rows",f"{filtered.duplicated().sum():,}")
    quality=pd.DataFrame({"Missing values":filtered.isna().sum()})
    quality["Missing %"]=(quality["Missing values"]/max(len(filtered),1)*100).round(1)
    quality["Data type"]=filtered.dtypes.astype(str)
    st.dataframe(quality,use_container_width=True)
    st.subheader("Column profile")
    st.dataframe(pd.DataFrame({"Column":filtered.columns,"Data type":filtered.dtypes.astype(str).values,
                               "Unique values":[filtered[c].nunique(dropna=True) for c in filtered.columns],
                               "Missing":[filtered[c].isna().sum() for c in filtered.columns]}),use_container_width=True,hide_index=True)

st.divider()
st.markdown('<div class="small">SMART AI ASSISTANT · UPLOAD-DRIVEN INTELLIGENCE · No bundled demo or sample data.</div>',unsafe_allow_html=True)
