from pathlib import Path
from io import BytesIO
import os, re, json, sqlite3, html
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

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

st.set_page_config(page_title="Smart AI Assistant | Intelligence Cloud", page_icon="✦", layout="wide")

# ── Premium visual system ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 10% 0%,#202753 0%,#0b1020 42%,#080b14 100%);color:#edf3ff}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1424,#111827);border-right:1px solid #293956}
.block-container{max-width:1580px;padding-top:1.2rem;padding-bottom:3rem}
h1,h2,h3{font-family:Manrope,sans-serif;letter-spacing:-.5px}
.hero{padding:30px 34px;border:1px solid #35496d;border-radius:24px;background:linear-gradient(115deg,rgba(99,102,241,.24),rgba(34,211,238,.08),rgba(139,92,246,.12));margin-bottom:22px;box-shadow:0 18px 55px #0003}
.eyebrow{color:#8be9ff;font-size:11px;font-weight:800;letter-spacing:2.4px}
.hero-title{font:800 2.35rem Manrope,sans-serif;color:#f5f7ff;margin:8px 0}
.hero-sub{color:#b7c5df}
.kpi{background:linear-gradient(145deg,#19263e,#0f1728);border:1px solid #2b3b59;border-radius:18px;padding:18px 20px;min-height:112px}
.kpi-label{color:#9fb0ca;font-size:11px;font-weight:800;letter-spacing:1.2px}
.kpi-value{color:#f4f7ff;font-size:26px;font-weight:800;margin-top:9px}
.kpi-foot{color:#7dd3fc;font-size:11px;margin-top:4px}
div[data-testid="stMetric"]{background:#111b2c;border:1px solid #2a3b58;padding:14px;border-radius:15px}
.stButton>button[kind="primary"]{background:linear-gradient(95deg,#6d5dfc,#8b5cf6,#0891b2);border:0;color:white;font-weight:700}
.stButton>button{border-radius:10px;border-color:#344663}
[data-testid="stDataFrame"]{border:1px solid #2a3b58;border-radius:12px;overflow:hidden}
.nav-label{color:#91a4c2;font-size:10px;font-weight:800;letter-spacing:1.5px;margin:10px 0 2px}
</style>
<div class="hero">
<div class="eyebrow">✦ INTELLIGENCE CLOUD · PRODUCT · SALES · DATA</div>
<div class="hero-title">Smart AI Assistant</div>
<div class="hero-sub">A unified command centre for product intelligence, commercial performance and data-driven decisions.</div>
</div>
""", unsafe_allow_html=True)

# ── Backend ingestion: local folders stay private and are never rendered ──
FOLDERS = {
    "CSV": {"csv"}, "Excel": {"xlsx","xls"}, "JSON": {"json"},
    "JSONL": {"jsonl","ndjson"}, "Parquet": {"parquet"},
    "Database": {"db","sqlite","sqlite3"}, "XML": {"xml"}, "TXT": {"txt"},
    "PDF": {"pdf"}, "DOCX": {"docx"}, "HTML": {"html","htm"}, "YAML": {"yaml","yml"}
}
UPLOAD_TYPES = sorted(set().union(*FOLDERS.values()))
DATA_ROOT = Path(__file__).resolve().parent / "datasets"

ALIASES = {
    "product":["product","product_name","name","item","title"],
    "product_id":["product_id","id","sku","item_id","product_code"],
    "category":["category","product_category","department","type"],
    "brand":["brand","manufacturer","vendor"],
    "description":["description","product_description","details","summary","text"],
    "price":["price","unit_price","selling_price","sale_price","cost"],
    "discount":["discount","discount_percent","discount_percentage"],
    "rating":["rating","average_rating","stars"],
    "reviews_count":["reviews_count","review_count","reviews"],
    "stock_quantity":["stock_quantity","stock","inventory","quantity_in_stock"],
    "supplier":["supplier","supplier_name"], "availability":["availability","status","stock_status"],
    "region":["region","market","territory"], "country":["country","nation"], "state":["state","province"],
    "date":["date","order_date","sales_date","transaction_date","created_date"],
    "order_id":["order_id","transaction_id","invoice_id"], "quantity":["quantity","units","qty","units_sold"],
    "revenue":["revenue","sales","total_sales","sales_amount","total"], "profit":["profit","net_profit","gross_profit"],
    "materials":["materials","material"], "colours":["colours","colors","colour","color"],
    "tags":["tags","keywords","tag"], "image_url":["image_url","image","photo_url"]
}

def clean_col(v):
    return re.sub(r"_+","_",re.sub(r"[^a-zA-Z0-9]+","_",str(v).strip())).strip("_").lower()

def to_frame(obj):
    if isinstance(obj,list):
        return pd.json_normalize(obj) if obj and all(isinstance(x,dict) for x in obj) else pd.DataFrame({"description":[str(x) for x in obj]})
    if isinstance(obj,dict):
        for k in ("products","data","items","records","catalogue","catalog"):
            if k in obj and isinstance(obj[k],(list,dict)): return to_frame(obj[k])
        return pd.json_normalize(obj)
    return pd.DataFrame({"description":[str(obj)]})

def parse_file(name, raw):
    ext=Path(name).suffix.lower().lstrip("."); bio=BytesIO(raw)
    if ext=="csv": return pd.read_csv(bio,sep=None,engine="python")
    if ext in ("xlsx","xls"): return pd.read_excel(bio)
    if ext=="json": return to_frame(json.loads(raw.decode("utf-8-sig")))
    if ext in ("jsonl","ndjson"): return pd.read_json(bio,lines=True)
    if ext=="parquet": return pd.read_parquet(bio)
    if ext in ("db","sqlite","sqlite3"):
        con=sqlite3.connect(":memory:"); con.deserialize(raw)
        tables=pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",con)["name"].tolist()
        if not tables: con.close(); raise ValueError("Database has no tables")
        table=next((t for t in tables if t.lower() in ("products","product","sales","transactions")),tables[0])
        result=pd.read_sql_query('SELECT * FROM "'+table.replace('"','""')+'"',con); con.close(); return result
    if ext=="xml":
        try: return pd.read_xml(bio)
        except Exception:
            import xml.etree.ElementTree as ET
            root=ET.fromstring(raw); return pd.DataFrame([{c.tag:c.text for c in list(n)} for n in list(root)])
    if ext in ("html","htm"):
        try:
            tables=pd.read_html(bio)
            if tables: return max(tables,key=len)
        except ValueError: pass
        text=re.sub(r"<[^>]+>"," ",raw.decode("utf-8","ignore"))
        return pd.DataFrame({"description":[re.sub(r"\s+"," ",html.unescape(text)).strip()]})
    if ext=="txt":
        text=raw.decode("utf-8-sig","ignore")
        try: return pd.read_csv(BytesIO(raw),sep=None,engine="python")
        except Exception: return pd.DataFrame({"description":[text]})
    if ext=="pdf":
        if PdfReader is None: raise ImportError("Add pypdf to requirements.txt")
        return pd.DataFrame({"description":["\n".join(p.extract_text() or "" for p in PdfReader(bio).pages)]})
    if ext=="docx":
        if Document is None: raise ImportError("Add python-docx to requirements.txt")
        doc=Document(bio); tables=[]
        for table in doc.tables:
            rows=[[c.text.strip() for c in r.cells] for r in table.rows]
            if len(rows)>1 and rows[0]: tables.append(pd.DataFrame(rows[1:],columns=rows[0]))
        return pd.concat(tables,ignore_index=True) if tables else pd.DataFrame({"description":["\n".join(p.text for p in doc.paragraphs)]})
    if ext in ("yaml","yml"):
        if yaml is None: raise ImportError("Add PyYAML to requirements.txt")
        return to_frame(yaml.safe_load(raw.decode("utf-8-sig")))
    raise ValueError("Unsupported file type: "+ext)

def discover_local():
    paths=[]
    for folder,exts in FOLDERS.items():
        directory=DATA_ROOT/folder
        if directory.is_dir():
            paths += [p for p in directory.iterdir() if p.is_file() and p.suffix.lower().lstrip(".") in exts]
    return paths

def load_local():
    frames=[]; report=[]
    for path in discover_local():
        try:
            part=parse_file(path.name,path.read_bytes())
            if not part.empty:
                part["source_file"]=path.name; frames.append(part)
                report.append({"Source":path.name,"Rows":len(part),"Status":"Loaded"})
        except Exception as e:
            report.append({"Source":path.name,"Rows":0,"Status":"Skipped: "+str(e)[:100]})
    return (pd.concat(frames,ignore_index=True,sort=False) if frames else None),report

def normalize(frame):
    d=frame.copy(); d.columns=[clean_col(c) for c in d.columns]
    d=d.loc[:,~pd.Index(d.columns).duplicated()]
    mapping={}
    for target,candidates in ALIASES.items():
        for c in candidates:
            if c in d.columns: mapping[c]=target; break
    d=d.rename(columns=mapping)
    if "product" not in d:
        text_col=next((c for c in d if d[c].astype(str).str.len().gt(20).any()),None)
        d["product"]="Document knowledge" if text_col else [f"Product {i+1}" for i in range(len(d))]
        if text_col and "description" not in d: d["description"]=d[text_col].astype(str)
    defaults={"product_id":[f"PRD-{i+1:05d}" for i in range(len(d))],"category":"Uncategorized","brand":"Unknown",
      "description":"","price":0,"discount":0,"rating":np.nan,"reviews_count":0,"stock_quantity":0,
      "supplier":"Not specified","availability":"Unknown","region":"Unspecified","country":"Unspecified",
      "state":"Unspecified","date":pd.NaT,"order_id":"","quantity":0,"revenue":np.nan,"profit":0,
      "materials":"Unspecified","colours":"Unspecified","tags":"","image_url":""}
    for c,v in defaults.items():
        if c not in d: d[c]=v
    for c in ["price","discount","rating","reviews_count","stock_quantity","quantity","revenue","profit"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d["price"]=d.price.fillna(0); d["quantity"]=d.quantity.fillna(0)
    d["revenue"]=d.revenue.fillna(d.price*d.quantity); d["profit"]=d.profit.fillna(0)
    d["date"]=pd.to_datetime(d.date,errors="coerce")
    for c in ["product","category","brand","availability","region","country","state","materials","colours"]:
        d[c]=d[c].fillna("Unknown").astype(str)
    d["discounted_price"]=(d.price*(1-d.discount.fillna(0).clip(0,100)/100)).round(2)
    return d

def secret(key):
    try: return st.secrets.get(key)
    except Exception: return os.getenv(key)

def cash(v): return f"₹{v:,.2f}"

def chart(fig):
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      height=360,margin=dict(l=12,r=12,t=55,b=12),font=dict(color="#e7efff"))
    return fig

def card(label,value,sub):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-foot">{sub}</div></div>',unsafe_allow_html=True)

# ── Navigation: no backend paths or folder tree shown in the UI ─────────
PAGES = [
    "Executive Dashboard", "Product Intelligence", "Product Catalogue",
    "Sales Intelligence", "Regional Intelligence", "Demand Forecast Studio",
    "AI Business Copilot", "Strategic Insights", "Data Explorer",
    "Data Quality", "Format Studio"
]
with st.sidebar:
    st.markdown("## ✦ SMART AI")
    st.caption("INTELLIGENCE WORKSPACE · PREMIUM")
    st.divider()
    st.markdown('<div class="nav-label">WORKSPACE NAVIGATION</div>',unsafe_allow_html=True)
    page = st.radio("Workspace", PAGES, label_visibility="collapsed")
    st.divider()
    st.markdown("**DATA INGESTION**")
    uploads=st.file_uploader("Upload business data",type=UPLOAD_TYPES,accept_multiple_files=True)
    st.caption("Multi-file ingestion · Automatic schema harmonization")
    st.divider()
    st.markdown("**AI ENGINE**")
    st.caption("Configure GROQ_API_KEY in Streamlit Secrets to enable the copilot.")
    st.markdown("[Project repository](https://github.com/ArunPandian-Git/Smart-AI-Assistant)")

# ── Load uploaded files first; otherwise scan private backend datasets ──
frames=[]; report=[]
if uploads:
    for f in uploads:
        try:
            part=parse_file(f.name,f.getvalue())
            if not part.empty:
                part["source_file"]=f.name; frames.append(part)
                report.append({"Source":f.name,"Rows":len(part),"Status":"Uploaded"})
        except Exception as e: report.append({"Source":f.name,"Rows":0,"Status":"Failed: "+str(e)[:100]})
else:
    local,report=load_local()
    if local is not None: frames=[local]

if frames:
    try: df=normalize(pd.concat(frames,ignore_index=True,sort=False)); mode="Connected dataset"
    except Exception as e: st.error(f"Data preparation error: {e}"); st.stop()
else:
    sample=pd.DataFrame([
      ["PRD-1001","Aero Wireless Headphones","Electronics","NovaTech",4999,10,4.6,284,42,"South","India","Tamil Nadu","2026-01-05",3,13497,3600],
      ["PRD-1002","Ergo Office Chair","Furniture","WorkNest",8999,5,4.4,156,18,"West","India","Maharashtra","2026-01-12",2,17098,4200],
      ["PRD-1003","Smart Fitness Watch","Wearables","PulseIQ",6499,15,4.2,391,0,"North","India","Delhi","2026-02-03",5,27500,7000],
      ["PRD-1004","Ceramic Coffee Maker","Home Appliances","BrewCraft",3299,0,4.1,98,27,"East","India","West Bengal","2026-02-14",4,13196,3100],
      ["PRD-1005","Ultralight Laptop Stand","Accessories","DeskPro",1799,8,4.7,512,64,"South","India","Karnataka","2026-03-08",8,13240,5100]],
      columns=["product_id","product","category","brand","price","discount","rating","reviews_count","stock_quantity","region","country","state","date","quantity","revenue","profit"])
    df=normalize(sample); mode="Demo workspace"

with st.sidebar:
    st.divider()
    st.success(f"{len(df):,} records ready")
    st.caption(mode)
    if report:
        with st.expander("Processing summary"):
            st.dataframe(pd.DataFrame(report),use_container_width=True,hide_index=True)
    st.markdown("**REFINE ANALYSIS**")
    cats=sorted(df.category.unique()); brands=sorted(df.brand.unique()); regions=sorted(df.region.unique())
    fc=st.multiselect("Categories",cats,default=cats)
    fb=st.multiselect("Brands",brands,default=brands)
    fr=st.multiselect("Regions",regions,default=regions)
    pmax=max(float(df.price.max()),1.0)
    pr=st.slider("Price range (₹)",0.0,pmax,(0.0,pmax))

data=df[df.category.isin(fc)&df.brand.isin(fb)&df.region.isin(fr)&df.price.between(pr[0],pr[1])].copy()
rev=float(data.revenue.sum()); prof=float(data.profit.sum()); units=float(data.quantity.sum())
margin=prof/rev*100 if rev else 0; stock=float(data.stock_quantity.sum())

# ── Executive dashboard ────────────────────────────────────────────────
if page=="Executive Dashboard":
    st.title("Executive Intelligence Dashboard")
    st.caption("A command view of product performance, commercial outcomes and inventory health.")
    a,b,c,d=st.columns(4)
    with a: card("CATALOGUE RECORDS",f"{len(data):,}","Records in current view")
    with b: card("REVENUE",cash(rev),"Recorded revenue")
    with c: card("GROSS PROFIT",cash(prof),f"{margin:.1f}% calculated margin")
    with d: card("INVENTORY UNITS",f"{stock:,.0f}","Current stock quantity")
    x,y=st.columns(2)
    with x:
        s=data.groupby("category",as_index=False).agg(Records=("product_id","count"))
        st.plotly_chart(chart(px.bar(s,x="category",y="Records",color="category",title="Portfolio mix by category")),use_container_width=True)
    with y:
        s=data.assign(Status=np.where(data.stock_quantity>0,"In stock","Out of stock")).groupby("Status",as_index=False).size()
        st.plotly_chart(chart(px.pie(s,names="Status",values="size",hole=.6,title="Inventory availability")),use_container_width=True)
    x,y=st.columns(2)
    with x:
        s=data.groupby("brand",as_index=False).revenue.sum().nlargest(10,"revenue")
        st.plotly_chart(chart(px.bar(s,x="brand",y="revenue",title="Top brands by revenue")),use_container_width=True)
    with y:
        dated=data.dropna(subset=["date"])
        if len(dated):
            s=dated.assign(Month=dated.date.dt.to_period("M").astype(str)).groupby("Month",as_index=False).revenue.sum()
            st.plotly_chart(chart(px.line(s,x="Month",y="revenue",markers=True,title="Revenue trajectory")),use_container_width=True)
        else: st.info("Add transaction dates to activate time-series analytics.")
    st.subheader("Performance leaders")
    st.dataframe(data.nlargest(10,"revenue")[["product_id","product","brand","category","price","revenue","profit","stock_quantity"]],use_container_width=True,hide_index=True)

elif page=="Product Intelligence":
    st.title("Product Intelligence")
    a,b,c,d=st.columns(4); a.metric("Products",f"{data.product.nunique():,}")
    b.metric("Average rating",f"{data.rating.mean():.2f}/5" if data.rating.notna().any() else "N/A")
    c.metric("Average discount",f"{data.discount.mean():.1f}%"); d.metric("Out of stock",int((data.stock_quantity<=0).sum()))
    x,y=st.columns(2)
    with x:
        s=data.groupby("category",as_index=False).agg(Avg_Price=("price","mean"))
        st.plotly_chart(chart(px.bar(s,x="category",y="Avg_Price",color="category",title="Average price by category")),use_container_width=True)
    with y: st.plotly_chart(chart(px.scatter(data,x="price",y="rating",size="reviews_count",color="category",hover_name="product",title="Price vs rating")),use_container_width=True)
    st.dataframe(data,use_container_width=True,hide_index=True)

elif page=="Product Catalogue":
    st.title("Product Catalogue")
    q=st.text_input("Search products, descriptions, brands or tags")
    view=data.copy()
    if q: view=view[view.astype(str).apply(lambda col:col.str.contains(q,case=False,na=False)).any(axis=1)]
    st.caption(f"{len(view):,} matching entries")
    for start in range(0,min(len(view),90),3):
        cols=st.columns(3)
        for col,(_,r) in zip(cols,view.iloc[start:start+3].iterrows()):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {r['product']}"); st.caption(f"{r['brand']} · {r['category']}")
                    st.markdown(f"**{cash(r['discounted_price'])}**")
                    st.write(str(r.description)[:220] or "No description supplied.")
                    st.caption(f"★ {r.rating if pd.notna(r.rating) else 'N/A'} · Stock {r.stock_quantity:,.0f}")
                    if str(r.get("image_url","")).startswith(("http://","https://")): st.image(r.image_url,use_container_width=True)
    if len(view)>90: st.info("Showing first 90 product cards. Use Data Explorer for the complete dataset.")

elif page=="Sales Intelligence":
    st.title("Sales Intelligence")
    a,b,c,d=st.columns(4); a.metric("Revenue",cash(rev)); b.metric("Profit",cash(prof)); c.metric("Units sold",f"{units:,.0f}"); d.metric("Profit margin",f"{margin:.2f}%")
    dated=data.dropna(subset=["date"])
    if len(dated):
        s=dated.assign(Month=dated.date.dt.to_period("M").astype(str)).groupby("Month",as_index=False).agg(Revenue=("revenue","sum"),Profit=("profit","sum"),Units=("quantity","sum"))
        st.plotly_chart(chart(px.line(s,x="Month",y=["Revenue","Profit"],markers=True,title="Monthly revenue and profit")),use_container_width=True)
        st.plotly_chart(chart(px.bar(s,x="Month",y="Units",title="Monthly unit movement")),use_container_width=True)
    else: st.info("No usable dates were detected in this dataset.")
    st.dataframe(data[["product","category","quantity","price","revenue","profit","date","country","state","region"]],use_container_width=True,hide_index=True)

elif page=="Regional Intelligence":
    st.title("Regional Intelligence")
    dimension=st.selectbox("Geographic view",["region","country","state"])
    s=data.groupby(dimension,as_index=False).agg(Revenue=("revenue","sum"),Profit=("profit","sum"),Products=("product_id","count"),Stock=("stock_quantity","sum"))
    x,y=st.columns(2)
    with x: st.plotly_chart(chart(px.bar(s,x=dimension,y="Revenue",color=dimension,title=f"Revenue by {dimension}")),use_container_width=True)
    with y: st.plotly_chart(chart(px.bar(s,x=dimension,y="Profit",color=dimension,title=f"Profit by {dimension}")),use_container_width=True)
    st.dataframe(s,use_container_width=True,hide_index=True)

elif page=="Demand Forecast Studio":
    st.title("Demand Forecast Studio")
    dated=data.dropna(subset=["date"])
    if len(dated)<5: st.info("At least five dated observations are recommended. Include date, quantity and revenue fields.")
    else:
        daily=dated.groupby(dated.date.dt.normalize(),as_index=False).agg(revenue=("revenue","sum"),demand=("quantity","sum")).sort_values("date")
        target=st.selectbox("Forecast target",["Revenue","Product demand"]); col="revenue" if target=="Revenue" else "demand"
        horizon=st.slider("Forecast horizon (days)",7,90,30); choice=st.selectbox("Model",["Linear Regression","Random Forest"])
        if st.button("Train model & forecast",type="primary"):
            daily["day_index"]=(daily.date-daily.date.min()).dt.days
            if daily[col].nunique()<2: st.warning("Not enough target variation to train a forecast.")
            else:
                split=max(1,int(len(daily)*.8)); X=daily[["day_index"]]; y=daily[col]
                model=LinearRegression() if choice=="Linear Regression" else RandomForestRegressor(n_estimators=150,random_state=42)
                model.fit(X.iloc[:split],y.iloc[:split])
                if split<len(daily): st.metric("Validation MAE",f"{mean_absolute_error(y.iloc[split:],model.predict(X.iloc[split:])):,.2f}")
                model.fit(X,y); dates=pd.date_range(daily.date.max()+pd.Timedelta(days=1),periods=horizon)
                future=((dates-daily.date.min()).days.to_numpy()).reshape(-1,1); values=np.maximum(model.predict(future),0)
                out=pd.DataFrame({"Date":dates,f"Predicted {target}":values})
                fig=px.line(daily,x="date",y=col,title=f"Historical and forecast {target}")
                fig.add_scatter(x=out.Date,y=out.iloc[:,1],mode="lines+markers",name="Forecast")
                st.plotly_chart(chart(fig),use_container_width=True); st.dataframe(out,use_container_width=True,hide_index=True)
                st.download_button("Download forecast",out.to_csv(index=False).encode(),"smart_ai_forecast.csv","text/csv")
                st.caption("Forecasts are estimates based on supplied history, not guaranteed outcomes.")

elif page=="AI Business Copilot":
    st.title("AI Business Copilot")
    st.caption("Ask questions about the current dataset. Answers are grounded in a compact data summary.")
    key=secret("GROQ_API_KEY")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[{"role":"assistant","content":"Welcome. Ask me about revenue, products, profitability, inventory, regions or trends."}]
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    prompt=st.chat_input("Ask a business question...")
    if prompt:
        st.session_state.chat_history.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            if not key or Groq is None:
                answer="AI is not configured. Add GROQ_API_KEY in Streamlit Cloud → App settings → Secrets. Analytics and data tools remain available."
                st.info(answer)
            else:
                context={"records":len(data),"revenue":rev,"profit":prof,"margin_pct":margin,"stock_units":stock,
                         "top_products":data.groupby("product").revenue.sum().nlargest(8).to_dict(),
                         "category_counts":data.category.value_counts().head(10).to_dict(),
                         "regional_revenue":data.groupby("region").revenue.sum().to_dict()}
                try:
                    client=Groq(api_key=key)
                    response=client.chat.completions.create(model="openai/gpt-oss-120b",
                        messages=[{"role":"system","content":"You are a senior business analyst. Use only this dataset summary; never invent metrics. Separate facts from recommendations. "+json.dumps(context,default=str)}]+st.session_state.chat_history[-8:],
                        temperature=.25,max_tokens=1000)
                    answer=response.choices[0].message.content or "No response returned."; st.markdown(answer)
                except Exception as e: answer=f"AI request failed: {str(e)[:250]}"; st.error(answer)
        st.session_state.chat_history.append({"role":"assistant","content":answer})
    if st.button("Clear conversation"): st.session_state.chat_history=[]; st.rerun()

elif page=="Strategic Insights":
    st.title("Strategic Insights")
    if data.empty: st.info("No data available.")
    else:
        top_product=data.groupby("product").revenue.sum().idxmax()
        top_category=data.groupby("category").revenue.sum().idxmax()
        top_region=data.groupby("region").revenue.sum().idxmax()
        a,b,c=st.columns(3); a.metric("Revenue leader",top_product); b.metric("Leading category",top_category); c.metric("Leading region",top_region)
        st.info(f"Inventory review: {(data.stock_quantity<=0).sum()} records show zero stock.")
        st.info(f"Product focus: {top_product} leads recorded revenue.")
        st.info(f"Category strategy: review assortment and margin performance in {top_category}.")
        st.info(f"Average recorded discount is {data.discount.mean():.1f}%; compare discount depth with profitability.")

elif page=="Data Explorer":
    st.title("Universal Data Explorer")
    a,b,c,d=st.columns(4); a.metric("Records",f"{len(data):,}"); b.metric("Fields",data.shape[1])
    c.metric("Missing cells",int(data.isna().sum().sum())); d.metric("Duplicate rows",int(data.duplicated().sum()))
    q=st.text_input("Search all fields"); view=data.copy()
    if q: view=view[view.astype(str).apply(lambda col:col.str.contains(q,case=False,na=False)).any(axis=1)]
    st.dataframe(view,use_container_width=True,hide_index=True)
    st.download_button("Download filtered data",view.to_csv(index=False).encode("utf-8-sig"),"smart_ai_data.csv","text/csv")

elif page=="Data Quality":
    st.title("Data Quality & Validation")
    quality=pd.DataFrame({"Field":df.columns,"Type":[str(t) for t in df.dtypes],"Missing":df.isna().sum().values,"Missing %":(df.isna().mean().values*100).round(2)})
    st.dataframe(quality,use_container_width=True,hide_index=True)
    st.metric("Duplicate rows",int(df.duplicated().sum()))
    with st.expander("Ingestion processing report"):
        if report: st.dataframe(pd.DataFrame(report),use_container_width=True,hide_index=True)
        else: st.write("No local/uploaded sources detected; demo data is active.")

elif page=="Format Studio":
    st.title("Multi-Format Studio")
    st.caption("Export the current filtered data into a portable format.")
    target=st.selectbox("Output format",["CSV","Excel","JSON","JSONL","Parquet","XML","HTML","TXT","YAML","DOCX"])
    if st.button("Prepare export",type="primary"):
        try:
            buf=BytesIO()
            if target=="CSV": payload=data.to_csv(index=False).encode("utf-8-sig")
            elif target=="Excel": data.to_excel(buf,index=False); payload=buf.getvalue()
            elif target=="JSON": payload=data.to_json(orient="records",indent=2,date_format="iso").encode()
            elif target=="JSONL": payload=data.to_json(orient="records",lines=True,date_format="iso").encode()
            elif target=="Parquet": data.to_parquet(buf,index=False); payload=buf.getvalue()
            elif target=="XML": payload=data.to_xml(index=False,root_name="records",row_name="record").encode()
            elif target=="HTML": payload=data.to_html(index=False).encode()
            elif target=="TXT": payload=data.to_csv(index=False,sep="\\t").encode()
            elif target=="YAML":
                if yaml is None: raise ImportError("Install PyYAML")
                payload=yaml.safe_dump(json.loads(data.to_json(orient="records",date_format="iso")),sort_keys=False,allow_unicode=True).encode()
            else:
                if Document is None: raise ImportError("Install python-docx")
                doc=Document(); doc.add_heading("Smart AI Assistant Export",0)
                cols=list(data.columns[:12]); table=doc.add_table(rows=1,cols=len(cols)); table.style="Light Shading Accent 1"
                for i,c in enumerate(cols): table.rows[0].cells[i].text=str(c)
                for _,row in data[cols].head(400).iterrows():
                    cells=table.add_row().cells
                    for i,v in enumerate(row): cells[i].text=str(v)[:250]
                doc.save(buf); payload=buf.getvalue()
            extensions={"CSV":"csv","Excel":"xlsx","JSON":"json","JSONL":"jsonl","Parquet":"parquet","XML":"xml","HTML":"html","TXT":"txt","YAML":"yaml","DOCX":"docx"}
            st.download_button("Download "+target,payload,"smart_ai_export."+extensions[target])
            st.success(f"Export prepared for {len(data):,} records.")
        except Exception as e: st.error(f"Export failed: {e}")
    st.caption("PDF and SQLite are ingestion formats. Parquet operations require PyArrow.")

st.divider()
st.markdown('<div style="text-align:center;color:#8293ae;font-size:11px;letter-spacing:1px">SMART AI ASSISTANT · INTELLIGENCE WORKSPACE · ANALYTICS · MACHINE LEARNING · GENERATIVE AI</div>',unsafe_allow_html=True)
