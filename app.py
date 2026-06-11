import os, io, json, datetime, warnings
import numpy as np, pandas as pd, joblib
import streamlit as st
warnings.filterwarnings("ignore")

st.set_page_config(page_title="DiaGuide - AI Diabetes Prediction",
                   page_icon="🩺", layout="centered")

st.markdown("""
<style>
/* ── Sidebar ─────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d2d7a 0%,#1a4fa0 60%,#1565c0 100%) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.18) !important; }

/* Nav radio buttons - inline circle + text */
[data-testid="stSidebar"] .stRadio > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}
[data-testid="stSidebar"] .stRadio > label {
    display: none !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 10px 14px !important;
    border-radius: 10px !important;
    margin: 3px 0 !important;
    transition: background .2s !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:hover {
    background: rgba(255,255,255,0.14) !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label {
    font-size: .97rem !important;
    font-weight: 500 !important;
    padding: 0 !important;
    margin: 0 !important;
    cursor: pointer !important;
    line-height: 1.2 !important;
    display: inline !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] input:checked ~ div {
    background: rgba(255,255,255,0.18) !important;
    border-radius: 10px !important;
}

/* ── Buttons ─────────────────────────────── */
.stFormSubmitButton > button {
    background: linear-gradient(135deg,#1a4fa0,#1565c0) !important;
    color: white !important; font-weight: 700 !important;
    border-radius: 10px !important; border: none !important;
    width: 100% !important; padding: 14px !important;
    font-size: 1.05rem !important; letter-spacing: .3px !important;
    box-shadow: 0 4px 14px rgba(26,79,160,.35) !important;
    transition: all .2s !important;
}
.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg,#153d80,#1a4fa0) !important;
    box-shadow: 0 6px 20px rgba(26,79,160,.5) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg,#1a4fa0,#1565c0) !important;
    color: white !important; font-weight: 700 !important;
    border-radius: 10px !important; border: none !important;
    box-shadow: 0 4px 14px rgba(26,79,160,.3) !important;
}
.stButton > button {
    background: #f0f4f8 !important; color: #1a4fa0 !important;
    font-weight: 600 !important; border-radius: 8px !important;
    border: 1.5px solid #d0daea !important;
}

/* ── Risk banners ────────────────────────── */
.risk-low  { background:linear-gradient(135deg,#16a34a,#22c55e); color:white; padding:30px 24px; border-radius:16px; text-align:center; margin:16px 0; box-shadow:0 6px 20px rgba(22,163,74,.3); }
.risk-mod  { background:linear-gradient(135deg,#d97706,#f59e0b); color:white; padding:30px 24px; border-radius:16px; text-align:center; margin:16px 0; box-shadow:0 6px 20px rgba(217,119,6,.3); }
.risk-high { background:linear-gradient(135deg,#dc2626,#ef4444); color:white; padding:30px 24px; border-radius:16px; text-align:center; margin:16px 0; box-shadow:0 6px 20px rgba(220,38,38,.3); }

/* ── Cards ───────────────────────────────── */
.sbox { background:#eef4ff; border-radius:12px; padding:18px; text-align:center; margin:4px; box-shadow:0 2px 8px rgba(26,79,160,.08); }
.bbox { background:#f4f7ff; border-radius:10px; padding:13px 18px; margin:6px 0; border-left:4px solid #1a4fa0; }
.hrow { background:#f4f7ff; border-radius:10px; padding:13px 18px; margin:8px 0; border-left:4px solid #1a4fa0; }

/* ── Input fields ────────────────────────── */
.stNumberInput input {
    border-radius: 8px !important;
    border: 1.5px solid #d0daea !important;
    font-size: .95rem !important;
}
.stNumberInput input:focus {
    border-color: #1a4fa0 !important;
    box-shadow: 0 0 0 3px rgba(26,79,160,.1) !important;
}
</style>""", unsafe_allow_html=True)

# ── Models ────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MD   = os.path.join(BASE, "models_saved")

@st.cache_resource
def load_models():
    try:
        e  = joblib.load(os.path.join(MD,"ensemble.pkl"))
        r  = joblib.load(os.path.join(MD,"rf.pkl"))
        l  = joblib.load(os.path.join(MD,"lr.pkl"))
        s  = joblib.load(os.path.join(MD,"scaler.pkl"))
        fi = json.load(open(os.path.join(MD,"feature_importance.json")))
        return e,r,l,s,fi,True
    except: return None,None,None,None,{},False

ens,rf_m,lr_m,sc,feat_imp,ready = load_models()

FEAT    = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
           "Insulin","BMI","DiabetesPedigreeFunction","Age"]
MEDIANS = {"Glucose":117.0,"BloodPressure":72.0,
           "SkinThickness":23.0,"Insulin":30.5,"BMI":32.0}

def preprocess(v):
    v2=[]
    for col,x in zip(FEAT,v):
        x=float(x)
        if col in MEDIANS and x==0: x=MEDIANS[col]
        v2.append(x)
    return sc.transform(np.array([v2]))

def risk_level(p):
    return "Low" if p*100<40 else ("Moderate" if p*100<65 else "High")

def ada_class(g):
    return ("Normoglycaemic" if g<100 else
            "Prediabetes" if g<126 else "Provisional Diabetes Diagnosis")

def get_recs(level,bmi,age):
    r={}
    if level=="High":
        r["🥗 Dietary Guidelines"]=[
            "Strictly limit refined carbohydrates (white rice, white bread, sugary drinks)",
            "Follow low GI diet: oats, lentils, brown rice",
            "Eat 5-6 small meals daily to stabilise blood glucose",
            "Increase dietary fibre to 25-35g per day"]
        r["🏥 Clinical Monitoring"]=[
            "Consult an endocrinologist immediately",
            "Monitor fasting blood glucose daily",
            "HbA1c test every 3 months",
            "Annual kidney function tests and eye examination"]
    elif level=="Moderate":
        r["🥗 Dietary Guidelines"]=[
            "Reduce refined sugar and processed food consumption",
            "Add vegetables, legumes, whole grains to every meal",
            "Replace sweet drinks with water or green tea"]
        r["🏥 Clinical Monitoring"]=[
            "Fasting glucose and HbA1c test within 2 weeks",
            "Physician follow-up every 3-6 months"]
    else:
        r["🥗 Dietary Guidelines"]=[
            "Maintain a balanced diet with fruits, vegetables, whole grains",
            "Keep sugar intake moderate",
            "Drink 8-10 glasses of water daily"]
        r["🏥 Clinical Monitoring"]=[
            "Annual fasting glucose screening",
            "Health check-ups every 6-12 months"]
    r["🏃 Physical Activity"]=(
        ["150 mins moderate aerobic activity per week",
         "Daily 30-minute brisk walk",
         "Resistance training 2-3 times per week"]
        if level in("High","Moderate") else
        ["At least 150 mins activity per week","Daily 30-minute brisk walk"])
    ls=[]
    if bmi>=30: ls.append("Weight loss critical — target 5-10% body weight reduction")
    if age>45:  ls.append("Increased screening frequency recommended due to age")
    ls+=["7-9 hours of quality sleep per night",
         "Manage stress through mindfulness or meditation",
         "Avoid smoking — worsens insulin resistance significantly"]
    r["💡 Lifestyle"]=ls
    return r

def make_pdf(vals,res,recs):
    try:
        from fpdf import FPDF
        pdf=FPDF(); pdf.set_margins(15,15,15); pdf.set_auto_page_break(True,18); pdf.add_page()
        pdf.set_fill_color(26,79,160); pdf.set_xy(0,0)
        pdf.set_font("Helvetica","B",16); pdf.set_text_color(255,255,255)
        pdf.cell(210,12,"  DiaGuide - AI Diabetes Prediction Report",fill=True,ln=True)
        pdf.set_xy(0,12); pdf.set_font("Helvetica","",7)
        pdf.cell(210,6," ADA 2024 Standards of Care  |  WHO Diabetes Prevention Programme",fill=True,ln=True)
        pdf.ln(4); pdf.set_text_color(0,0,0); pdf.set_font("Helvetica","",9)
        pdf.cell(0,5,"Report Date: "+datetime.datetime.now().strftime("%d %b %Y  %H:%M"),ln=True); pdf.ln(2)
        lv=res["level"]
        col={"Low":(34,139,34),"Moderate":(200,120,0),"High":(180,0,0)}.get(lv,(80,80,80))
        pdf.set_fill_color(*col); pdf.set_text_color(255,255,255); pdf.set_font("Helvetica","B",10)
        pdf.cell(0,9,"  RISK: "+lv.upper()+"   Probability: "+str(round(res['prob'],1))+"% ADA: "+res['ada'],fill=True,ln=True)
        pdf.ln(3); pdf.set_text_color(0,0,0); pdf.set_font("Helvetica","B",10); pdf.cell(0,6,"Biomarker Profile",ln=True)
        W=[60,30,50,30]; pdf.set_font("Helvetica","B",8); pdf.set_fill_color(26,79,160); pdf.set_text_color(255,255,255)
        for h,w in zip(["Parameter","Value","Normal Range","Status"],W): pdf.cell(w,7,h,border=1,fill=True)
        pdf.ln()
        brows=[("Glucose (mg/dL)",vals[1],"70-99",70,99),("Blood Pressure",vals[2],"60-80",60,80),
               ("BMI (kg/m2)",vals[5],"18.5-24.9",18.5,24.9),("Skin Thickness",vals[3],"10-40",10,40),
               ("Insulin (uU/mL)",vals[4],"2-25",2,25),("Age (years)",vals[7],"N/A",None,None),
               ("Pregnancies",vals[0],"N/A",None,None),("Pedigree Function",vals[6],"0.08-2.42",None,None)]
        pdf.set_font("Helvetica","",8); tog=True
        for p,v,n,lo,hi in brows:
            pdf.set_fill_color(245,248,255) if tog else pdf.set_fill_color(255,255,255)
            pdf.set_text_color(0,0,0)
            if lo is not None: st2,sc2=("Normal",(0,130,0)) if lo<=float(v)<=hi else ("Abnormal",(180,0,0))
            else: st2,sc2="-",(80,80,80)
            pdf.cell(W[0],6,p,border=1,fill=True); pdf.cell(W[1],6,str(round(float(v),1)),border=1,fill=True)
            pdf.cell(W[2],6,n,border=1,fill=True); pdf.set_text_color(*sc2)
            pdf.cell(W[3],6,st2,border=1,fill=True); pdf.set_text_color(0,0,0); pdf.ln(); tog=not tog
        pdf.ln(3); pdf.set_font("Helvetica","B",10); pdf.cell(0,6,"Model Confidence Scores",ln=True)
        pdf.set_font("Helvetica","",9)
        for nm,val in [("Random Forest",res['rf']),("Logistic Regression",res['lr']),("Ensemble (Final)",res['prob']/100)]:
            pdf.cell(80,6,nm); pdf.cell(0,6,str(round(val*100,1))+"%",ln=True)
        pdf.add_page()
        pdf.set_font("Helvetica","B",13); pdf.cell(0,8,"Personalised Lifestyle Recommendations",ln=True)
        pdf.set_font("Helvetica","I",8); pdf.cell(0,5,"ADA 2024 Standards of Care | WHO Diabetes Prevention Programme",ln=True); pdf.ln(3)
        sc_col={"Dietary":(30,130,50),"Physical":(30,80,160),"Clinical":(150,30,30),"Lifestyle":(110,50,150)}
        for sec,items in recs.items():
            if not items: continue
            clean=sec.split(" ",1)[-1] if sec[0] in "🥗🏥🏃💡" else sec
            color=next((v for k,v in sc_col.items() if k in clean),(60,60,60))
            pdf.set_fill_color(*color); pdf.set_text_color(255,255,255); pdf.set_font("Helvetica","B",9)
            pdf.cell(0,7,"  "+clean,fill=True,ln=True); pdf.set_text_color(0,0,0); pdf.set_font("Helvetica","",8)
            for item in items:
                pdf.set_x(18); pdf.multi_cell(0,5,"- "+item.encode("latin-1","replace").decode("latin-1"))
            pdf.ln(1)
        pdf.set_y(-10); pdf.set_fill_color(26,79,160); pdf.set_xy(0,pdf.get_y())
        pdf.set_font("Helvetica","",6); pdf.set_text_color(255,255,255)
        
        buf=io.BytesIO(); buf.write(pdf.output()); buf.seek(0); return buf.getvalue()
    except: return None

# ── Session state ─────────────────────────────────────
for k,v in [("history",[]),("result",None),("show_result",False)]:
    if k not in st.session_state: st.session_state[k]=v

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    # Logo area
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px 0">
        <div style="font-size:3rem;margin-bottom:8px">🩺</div>
        <div style="font-size:1.6rem;font-weight:800;letter-spacing:1px">DiaGuide</div>
        <div style="font-size:.75rem;opacity:.7;margin-top:4px;letter-spacing:.5px">
            AI DIABETES PREDICTION
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Navigation
    page = st.radio("nav", [ "🔬  New Prediction", "📋  History", "ℹ️  About"], label_visibility="collapsed")

    st.markdown("---")

    # Status
    if ready:
        st.markdown("""
        <div style="background:rgba(34,197,94,.2);border:1px solid rgba(34,197,94,.4);
             border-radius:10px;padding:10px 14px;text-align:center;margin-bottom:8px">
            <div style="font-size:1.2rem">✅</div>
            <div style="font-size:.8rem;font-weight:700;margin-top:2px">Models Loaded</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(239,68,68,.2);border:1px solid rgba(239,68,68,.4);
             border-radius:10px;padding:10px 14px;text-align:center">
            <div style="font-size:1.2rem">⚠️</div>
            <div style="font-size:.8rem;font-weight:700">Run train_model.py</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    
# ══════════════════════════════════════════════════════
# PAGE: NEW PREDICTION
# ══════════════════════════════════════════════════════
if "New Prediction" in page:

    # ── RESULT VIEW ──────────────────────────────────
    if st.session_state.show_result and st.session_state.result:
        R = st.session_state.result
        st.title("✅ Prediction Result")
        if st.button("⬅️  Run New Prediction", key="back_btn"):
            st.session_state.show_result = False
            st.session_state.result = None
            st.rerun()
        st.markdown("---")

        css={"Low":"risk-low","Moderate":"risk-mod","High":"risk-high"}[R["lv"]]
        em ={"Low":"🟢","Moderate":"🟡","High":"🔴"}[R["lv"]]
        st.markdown(f"""
<div class="{css}">
  <div style="font-size:3.2rem;font-weight:800;margin-bottom:8px">{R['ep']*100:.1f}%</div>
  <div style="font-size:1.4rem;font-weight:700;margin-bottom:6px">{em} Risk Level: {R['lv']}</div>
  <div style="font-size:.95rem;opacity:.9">ADA Classification: <b>{R['ada']}</b></div>
</div>""", unsafe_allow_html=True)

        st.markdown("#### 📊 Model Confidence Scores")
        st.markdown(f"""
<div style="display:flex;gap:12px;margin:12px 0">
  <div class="sbox" style="flex:1">
    <div style="font-size:.78rem;color:#555;margin-bottom:6px">🌲 Random Forest</div>
    <div style="font-size:1.7rem;font-weight:800;color:#1a4fa0">{R['rfp']*100:.1f}%</div>
  </div>
  <div class="sbox" style="flex:1">
    <div style="font-size:.78rem;color:#555;margin-bottom:6px">📈 Logistic Regression</div>
    <div style="font-size:1.7rem;font-weight:800;color:#1a4fa0">{R['lrp']*100:.1f}%</div>
  </div>
  <div class="sbox" style="flex:1">
    <div style="font-size:.78rem;color:#555;margin-bottom:6px">🎯 Ensemble (Final)</div>
    <div style="font-size:1.7rem;font-weight:800;color:#1a4fa0">{R['ep']*100:.1f}%</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.progress(min(R["ep"],1.0))

        def bstat(val,lo,hi,unit=""):
            if val==0: return f"<b>{val}{unit}</b> &nbsp;—&nbsp; <span style='color:#888'>ℹ️ Unknown</span>"
            ok=lo<=val<=hi; c="#16a34a" if ok else "#dc2626"
            t="✅ Normal" if ok else "⚠️ Abnormal"
            return f"<b>{val}{unit}</b> &nbsp;—&nbsp; <span style='color:{c}'>{t}</span>"

        st.markdown("#### 🔬 Biomarker Status")
        st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:12px 0">
  <div class="bbox">🩸 <b>Glucose</b><br>{bstat(R['g'],70,99,' mg/dL')}</div>
  <div class="bbox">⚖️ <b>BMI</b><br>{bstat(R['b'],18.5,24.9,' kg/m²')}</div>
  <div class="bbox">💉 <b>Blood Pressure</b><br>{bstat(R['bp'],60,80,' mmHg')}</div>
  <div class="bbox">🔬 <b>Insulin</b><br>{bstat(R['insulin'],2,25,' uU/mL')}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📋 Personalised Recommendations")
        st.caption("Based on ADA 2024 Standards of Care and WHO Diabetes Prevention Programme")
        for sec,items in R["recs"].items():
            if items:
                with st.expander(sec, expanded=True):
                    for item in items: st.write(f"▸ {item}")

        st.markdown("---")
        if R["pdf_b"]:
            st.download_button("📄 Download Clinical PDF Report", R["pdf_b"],
                f"DiaGuide_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                "application/pdf", use_container_width=True, key="dl_pdf")

    # ── FORM VIEW ────────────────────────────────────
    else:
        if not ready:
            st.error("⚠️ Models not loaded. Run: python train_model.py")
            st.stop()

        st.title("🔬 Diabetes Risk Assessment")
        st.markdown("Enter all 8 biomarker values. Enter **0** for any unknown value.")
        st.markdown("---")

        with st.form(key="pred_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                glucose = st.number_input("🩸 Glucose Level (mg/dL) — Normal: 70–99",    0.0,400.0, 0.0,  1.0)
                bmi     = st.number_input("⚖️ BMI (kg/m²) — Normal: 18.5–24.9",           0.0, 70.0, 0.0,  0.1)
                age     = st.number_input("🎂 Age (years)",                                1,   120,  25)
                bp      = st.number_input("💉 Blood Pressure (mmHg) — Normal: 60–80",     0.0,150.0, 0.0,  1.0)
            with col2:
                skin    = st.number_input("📏 Skin Thickness (mm) — enter 0 if unknown",  0.0,100.0, 0.0,  1.0)
                insulin = st.number_input("🔬 Insulin (uU/mL) — enter 0 if unknown",      0.0,900.0, 0.0,  1.0)
                dpf     = st.number_input("🧬 Diabetes Pedigree Function — 0.5 if unknown",0.0,3.0,  0.5,0.001,format="%.3f")
                preg    = st.number_input("🤰 Number of Pregnancies — 0 if male",          0,   20,   0)
                st.markdown("")
            go = st.form_submit_button("🔬 Analyse Diabetes Risk", use_container_width=True)

        if go:
            g=float(glucose); b=float(bmi)
            if g<=0:
                st.error("⚠️ Glucose Level is required (must be > 0)")
            elif b<=0:
                st.error("⚠️ BMI is required (must be > 0)")
            else:
                vals=[float(preg),g,float(bp),float(skin),
                      float(insulin),b,float(dpf),float(age)]
                with st.spinner("🔬 Running ML analysis..."):
                    fs =preprocess(vals)
                    ep =float(ens.predict_proba(fs)[0][1])
                    rfp=float(rf_m.predict_proba(fs)[0][1])
                    lrp=float(lr_m.predict_proba(fs)[0][1])
                    lv =risk_level(ep)
                    ada=ada_class(g)
                    recs=get_recs(lv,b,float(age))
                res_obj={"prob":round(ep*100,2),"level":lv,"ada":ada,"rf":rfp,"lr":lrp}
                pdf_b  =make_pdf(vals,res_obj,recs)
                st.session_state.result={
                    "vals":vals,"res":res_obj,"recs":recs,
                    "ep":ep,"rfp":rfp,"lrp":lrp,"lv":lv,"ada":ada,"pdf_b":pdf_b,
                    "g":g,"b":b,"bp":float(bp),"insulin":float(insulin)}
                st.session_state.history.append({
                    "id":len(st.session_state.history)+1,
                    "date":datetime.datetime.now().strftime("%d %b %Y %H:%M"),
                    "glucose":g,"bmi":b,"age":float(age),
                    "risk_prob":round(ep*100,2),"risk_level":lv,"ada_class":ada,
                    "vals":vals,"res":res_obj,"recs":recs,"pdf_bytes":pdf_b})
                st.session_state.show_result=True
                st.rerun()

# ══════════════════════════════════════════════════════
elif "History" in page:
    st.title("📋 Prediction History")
    st.markdown("---")
    if not st.session_state.history:
        st.info("📭 No predictions yet. Go to **New Prediction** to run your first analysis.")
    else:
        hist=list(reversed(st.session_state.history))
        total=len(hist)
        high=sum(1 for h in hist if h['risk_level']=="High")
        mod =sum(1 for h in hist if h['risk_level']=="Moderate")
        low =sum(1 for h in hist if h['risk_level']=="Low")
        st.markdown(f"""
<div style="display:flex;gap:12px;margin-bottom:18px">
  <div class="sbox" style="flex:1"><div style="font-size:.78rem;color:#555">📊 Total</div>
    <div style="font-size:2rem;font-weight:800;color:#1a4fa0">{total}</div></div>
  <div class="sbox" style="flex:1"><div style="font-size:.78rem;color:#555">🔴 High</div>
    <div style="font-size:2rem;font-weight:800;color:#dc2626">{high}</div></div>
  <div class="sbox" style="flex:1"><div style="font-size:.78rem;color:#555">🟡 Moderate</div>
    <div style="font-size:2rem;font-weight:800;color:#d97706">{mod}</div></div>
  <div class="sbox" style="flex:1"><div style="font-size:.78rem;color:#555">🟢 Low</div>
    <div style="font-size:2rem;font-weight:800;color:#16a34a">{low}</div></div>
</div>""", unsafe_allow_html=True)
        rows=[]
        for h in hist:
            badge={"Low":"🟢 Low","Moderate":"🟡 Moderate","High":"🔴 High"}[h['risk_level']]
            rows.append({"#":h['id'],"Date":h['date'],"Glucose":h['glucose'],
                         "BMI":h['bmi'],"Age":h['age'],
                         "Risk %":f"{h['risk_prob']}%","Level":badge,"ADA Class":h['ada_class']})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.markdown("---")
        st.markdown("#### 📄 Download PDF Reports")
        for h in hist:
            em={"Low":"🟢","Moderate":"🟡","High":"🔴"}[h['risk_level']]
            st.markdown(f"""
<div class="hrow">
  <b>#{h['id']}</b> &nbsp;|&nbsp; {h['date']} &nbsp;|&nbsp;
  {em} <b>{h['risk_level']}</b> &nbsp;|&nbsp; Risk: <b>{h['risk_prob']}%</b>
  &nbsp;|&nbsp; ADA: {h['ada_class']}
</div>""", unsafe_allow_html=True)
            if h.get("pdf_bytes"):
                st.download_button(f"📄 Download PDF #{h['id']}",
                    h["pdf_bytes"],f"DiaGuide_Report_{h['id']}.pdf",
                    "application/pdf",key=f"pdf_{h['id']}")

# ══════════════════════════════════════════════════════
elif "About" in page:
    st.title("ℹ️ About DiaGuide")
    st.markdown("---")
    st.markdown("""
<div style="background:linear-gradient(135deg,#eef4ff,#f4f7ff);border-radius:14px;
     padding:24px 28px;border-left:5px solid #1a4fa0;margin-bottom:20px">
<h4 style="color:#1a4fa0;margin-bottom:12px"> Overview</h4>
<p style="color:#333;line-height:1.7">
DiaGuide is an AI-powered diabetes prediction and lifestyle recommendation system
built using machine learning on the PIMA Indians Diabetes Dataset. It uses a
soft-voting ensemble of Random Forest and Logistic Regression classifiers to
predict diabetes risk and provides personalised recommendations based on
ADA 2024 Standards of Care.
</p>
</div>""", unsafe_allow_html=True)

    st.markdown("""
| Field | Details |
|---|---|
| **Technology** | Python, Streamlit, Scikit-learn |
| **ML Models** | Random Forest + Logistic Regression |
| **Ensemble Method** | Soft-Voting Classifier |
| **Dataset** | PIMA Indians Diabetes Dataset |
| **Records** | 768 patients, 8 features |
| **CV Accuracy** | 76.68% (±2.95%) |
| **ROC-AUC Score** | 0.8156 |
| **Test Accuracy** | 70.13% |
    """)
    st.markdown("---")
    st.markdown("### 🏥 ADA 2024 Glucose Classification")
    st.markdown("""
| Fasting Glucose | Classification | Action |
|---|---|---|
| < 100 mg/dL | 🟢 Normoglycaemic | Annual screening |
| 100–125 mg/dL | 🟡 Prediabetes | Lifestyle changes, monitor |
| ≥ 126 mg/dL | 🔴 Provisional Diabetes | Immediate medical consultation |
    """)
    st.markdown("---")
    st.markdown("### 📊 Feature Descriptions")
    st.dataframe(pd.DataFrame({
        "Feature":["Glucose","BMI","Age","Blood Pressure","Skin Thickness",
                   "Insulin","Pregnancies","Pedigree Function"],
        "Description":["Fasting plasma glucose (mg/dL)","Body Mass Index (kg/m²)",
                       "Patient age (years)","Diastolic blood pressure (mmHg)",
                       "Triceps skinfold thickness (mm)","2-hour serum insulin (uU/mL)",
                       "Number of pregnancies","Diabetes family history score"],
        "Normal Range":["70–99","18.5–24.9","N/A","60–80","10–40","2–25","N/A","0.08–2.42"]
    }), use_container_width=True, hide_index=True)
    st.markdown("---")
    
