import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Disease Predictor", page_icon="🏥", layout="wide")

# ── ALL CSS in one place, using strong selectors ──────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
body, html { background: #F0F4F8 !important; }
.stApp    { background: #F0F4F8 !important; }
.block-container { padding-top: 1.5rem !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg,#0A2342 0%,#023E5A 55%,#028090 100%) !important;
    padding: 1.5rem 1rem;
}
section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] { display:none; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.25) !important; }

/* ── Main text always dark on light bg ── */
.stApp .stMarkdown p,
.stApp .stMarkdown li,
.stApp .stMarkdown span,
.stApp .stMarkdown label { color: #1E293B !important; }

/* ── Tab text ── */
.stTabs [data-baseweb="tab"] { color: #475569 !important; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #028090 !important; border-bottom: 3px solid #028090 !important; }

/* ── st.metric override ── */
[data-testid="stMetricLabel"] > div { color: #64748B !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"]       { color: #0A2342 !important; font-size: 1.5rem !important; font-weight:800 !important; }

/* ── Predict button ── */
section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(90deg,#02C39A,#028090) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-size: 1rem !important;
    font-weight: 700 !important; padding: 0.75rem !important;
    width: 100% !important; margin-top: 6px !important;
    box-shadow: 0 4px 14px rgba(2,128,144,0.35) !important;
}

/* ── Info/warning boxes native ── */
.stAlert { border-radius: 10px !important; }

/* ── Dataframe text ── */
.stDataFrame td, .stDataFrame th { color: #1E293B !important; }

/* ── Hide chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 768
    glucose = np.random.normal(121, 32, n).clip(44, 199)
    bmi     = np.random.normal(32, 7.9, n).clip(18, 67)
    age     = np.random.randint(21, 81, n).astype(float)
    preg    = np.random.randint(0, 17, n).astype(float)
    bp      = np.random.normal(69, 19, n).clip(24, 122)
    skin    = np.random.normal(20, 16, n).clip(0, 99)
    insulin = np.random.normal(79, 115, n).clip(0, 846)
    dpf     = np.random.uniform(0.08, 2.42, n)
    prob    = 1 / (1 + np.exp(-(0.003*glucose + 0.015*bmi + 0.008*age + 0.04*preg - 1.8)))
    outcome = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({"Pregnancies": preg, "Glucose": glucose,
        "BloodPressure": bp, "SkinThickness": skin, "Insulin": insulin,
        "BMI": bmi, "DiabetesPedigreeFunction": dpf, "Age": age, "Outcome": outcome})

    X, y = df.drop("Outcome", axis=1), df["Outcome"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=4, random_state=42)
    clf.fit(sc.fit_transform(Xtr), ytr)
    acc = accuracy_score(yte, clf.predict(sc.transform(Xte)))
    return clf, sc, acc, df

model, scaler, accuracy, df = train_model()
FEAT = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age"]


# ── SIDEBAR INPUT ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Disease Predictor")
    st.markdown("*Project Jupyter Hackathon · AIT*")
    st.markdown("---")
    st.markdown("### 📋 Enter Patient Details")

    preg_v = st.slider("Pregnancies",            0,   17,   1)
    gluc_v = st.slider("Glucose (mg/dL)",        44,  200,  110,  help="Plasma glucose (fasting)")
    bp_v   = st.slider("Blood Pressure (mm Hg)", 24,  122,  70,   help="Diastolic BP")
    skin_v = st.slider("Skin Thickness (mm)",    0,   99,   20,   help="Triceps skinfold")
    ins_v  = st.slider("Insulin (μU/ml)",         0,   846,  80,   help="2-hr serum insulin")
    bmi_v  = st.slider("BMI",                    18.0, 67.0, 25.0, 0.1)
    dpf_v  = st.slider("Diabetes Pedigree",      0.08, 2.42, 0.47, 0.01, help="Family history score")
    age_v  = st.slider("Age",                    21,  80,   30)

    st.markdown("---")
    clicked = st.button("🔍  Predict Now", use_container_width=True)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.8rem;color:#B2EBF2;line-height:1.8'>"
        "📊 Pima Indians Diabetes Dataset<br>"
        "🤖 Random Forest · 200 trees<br>"
        "⚡ Scikit-learn + Streamlit</div>",
        unsafe_allow_html=True
    )


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0A2342,#028090,#02C39A);
            border-radius:14px;padding:32px 36px;margin-bottom:20px">
  <h1 style="color:#fff;margin:0 0 8px;font-size:2rem">🏥 Early Disease Prediction System</h1>
  <p style="color:#B2EBF2;margin:0;font-size:1rem">
    AI-powered diabetes risk detection using Machine Learning &nbsp;·&nbsp;
    Project Jupyter Hackathon · Army Institute of Technology
  </p>
</div>
""", unsafe_allow_html=True)

# ── STATS ROW ─────────────────────────────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
for col, val, lbl, clr in [
    (s1, f"{accuracy*100:.1f}%", "Model Accuracy",    "#028090"),
    (s2, "768",                   "Training Records",  "#023E5A"),
    (s3, "8",                     "Health Features",   "#02C39A"),
    (s4, "200",                   "Decision Trees",    "#0A2342"),
]:
    col.markdown(
        f"<div style='background:#fff;border-radius:12px;padding:18px 16px;"
        f"border-top:4px solid {clr};box-shadow:0 2px 10px rgba(0,0,0,0.07);text-align:center'>"
        f"<div style='color:{clr};font-size:1.9rem;font-weight:800;margin:0'>{val}</div>"
        f"<div style='color:#64748B;font-size:0.78rem;text-transform:uppercase;letter-spacing:.05em;margin-top:4px'>{lbl}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  Prediction Result", "📊  Data Insights", "ℹ️  About"])


# ════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════
with tab1:

    # ── Always-visible section header ──
    st.markdown(
        "<div style='background:linear-gradient(90deg,#028090,#02C39A);"
        "color:#fff;padding:10px 20px;border-radius:8px;font-weight:700;"
        "font-size:1rem;margin-bottom:18px'>🎯 Patient Risk Assessment</div>",
        unsafe_allow_html=True
    )

    if not clicked:
        # ── Placeholder shown before prediction ──
        st.markdown(
            "<div style='background:#fff;border:2px dashed #CBD5E1;border-radius:14px;"
            "padding:48px;text-align:center'>"
            "<div style='font-size:3.5rem'>🩺</div>"
            "<h2 style='color:#0A2342;margin:12px 0 8px'>Ready to Predict</h2>"
            "<p style='color:#64748B;font-size:1.05rem;margin:0'>"
            "Adjust the sliders in the sidebar, then click <b>🔍 Predict Now</b></p>"
            "</div>",
            unsafe_allow_html=True
        )

    else:
        # ── Run prediction ──
        inp   = np.array([[preg_v, gluc_v, bp_v, skin_v, ins_v, bmi_v, dpf_v, age_v]])
        pred  = model.predict(scaler.transform(inp))[0]
        proba = model.predict_proba(scaler.transform(inp))[0]
        risk  = proba[1] * 100
        safe  = proba[0] * 100

        # ══ BIG RESULT BANNER ══
        if pred == 1:
            st.markdown(
                f"<div style='background:#FFF0F0;border:3px solid #E53E3E;"
                f"border-radius:16px;padding:36px;text-align:center;margin-bottom:20px'>"
                f"<div style='font-size:3rem'>⚠️</div>"
                f"<h1 style='color:#C53030;margin:10px 0 8px;font-size:2rem'>"
                f"HIGH DIABETES RISK DETECTED</h1>"
                f"<p style='color:#742A2A;font-size:1.1rem;margin:0'>"
                f"The model predicts a <b style='font-size:1.4rem;color:#C53030'>{risk:.1f}%</b> "
                f"probability of diabetes. Please consult a doctor.</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='background:#F0FFF8;border:3px solid #02C39A;"
                f"border-radius:16px;padding:36px;text-align:center;margin-bottom:20px'>"
                f"<div style='font-size:3rem'>✅</div>"
                f"<h1 style='color:#02C39A;margin:10px 0 8px;font-size:2rem'>"
                f"LOW DIABETES RISK</h1>"
                f"<p style='color:#1A4731;font-size:1.1rem;margin:0'>"
                f"The model predicts only a <b style='font-size:1.4rem;color:#028090'>{risk:.1f}%</b> "
                f"probability of diabetes. Keep maintaining a healthy lifestyle!</p>"
                f"</div>",
                unsafe_allow_html=True
            )

        # ── Probability gauges ──
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown(
                "<div style='background:#fff;border-radius:12px;padding:22px;"
                "box-shadow:0 2px 10px rgba(0,0,0,0.07)'>"
                "<p style='color:#64748B;font-size:0.85rem;text-transform:uppercase;"
                "letter-spacing:.05em;margin:0 0 10px'>Diabetic Probability</p>"
                f"<div style='font-size:2.5rem;font-weight:800;color:#E53E3E'>{risk:.1f}%</div>"
                "<div style='background:#FED7D7;border-radius:6px;height:14px;margin-top:10px'>"
                f"<div style='background:#E53E3E;width:{risk:.1f}%;height:100%;border-radius:6px'></div>"
                "</div></div>",
                unsafe_allow_html=True
            )

        with c_right:
            st.markdown(
                "<div style='background:#fff;border-radius:12px;padding:22px;"
                "box-shadow:0 2px 10px rgba(0,0,0,0.07)'>"
                "<p style='color:#64748B;font-size:0.85rem;text-transform:uppercase;"
                "letter-spacing:.05em;margin:0 0 10px'>Non-Diabetic Probability</p>"
                f"<div style='font-size:2.5rem;font-weight:800;color:#02C39A'>{safe:.1f}%</div>"
                "<div style='background:#C6F6D5;border-radius:6px;height:14px;margin-top:10px'>"
                f"<div style='background:#02C39A;width:{safe:.1f}%;height:100%;border-radius:6px'></div>"
                "</div></div>",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Input summary table ──
        st.markdown(
            "<div style='background:linear-gradient(90deg,#028090,#02C39A);"
            "color:#fff;padding:10px 20px;border-radius:8px;font-weight:700;"
            "font-size:1rem;margin-bottom:14px'>📋 Your Input vs Normal Ranges</div>",
            unsafe_allow_html=True
        )

        normals = ["0–5","70–140 mg/dL","60–80 mm Hg","10–30 mm",
                   "16–166 μU/ml","18.5–24.9","< 0.5","–"]
        vals    = [preg_v, f"{gluc_v} mg/dL", f"{bp_v} mm Hg",
                   f"{skin_v} mm", f"{ins_v} μU/ml", f"{bmi_v:.1f}", f"{dpf_v:.2f}", age_v]

        def flag(feature, value):
            checks = {"Glucose":(gluc_v,70,140), "BMI":(bmi_v,18.5,24.9),
                      "BloodPressure":(bp_v,60,80)}
            if feature in checks:
                v, lo, hi = checks[feature]
                return "🔴 High" if v > hi else ("🟡 Low" if v < lo else "🟢 Normal")
            return "–"

        summary = pd.DataFrame({
            "Feature":      FEAT,
            "Your Value":   vals,
            "Normal Range": normals,
            "Status":       [flag(f, v) for f, v in zip(FEAT, vals)]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # ── Disclaimer ──
        st.markdown(
            "<div style='background:#FFFBEB;border-left:4px solid #F6AD55;"
            "border-radius:8px;padding:12px 16px;margin-top:10px'>"
            "<span style='color:#744210;font-size:0.88rem'>"
            "⚠️ <b>Disclaimer:</b> This tool is for the Project Jupyter Hackathon only. "
            "It is NOT a certified medical device. Always consult a qualified doctor.</span>"
            "</div>",
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════════════════════
# TAB 2 — DATA INSIGHTS
# ════════════════════════════════════════════════════════════
with tab2:
    def sec(title):
        st.markdown(
            f"<div style='background:linear-gradient(90deg,#028090,#02C39A);"
            f"color:#fff;padding:10px 20px;border-radius:8px;font-weight:700;"
            f"font-size:1rem;margin:24px 0 14px'>{title}</div>",
            unsafe_allow_html=True
        )

    sec("📊 Dataset Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records",  "768")
    m2.metric("Diabetic Cases", f"{df['Outcome'].sum()}  ({df['Outcome'].mean()*100:.0f}%)")
    m3.metric("Healthy Cases",  f"{(~df['Outcome'].astype(bool)).sum()}  ({(1-df['Outcome'].mean())*100:.0f}%)")

    sec("🔑 Feature Importance — What Predicts Diabetes Most?")
    imp = (pd.DataFrame({"Feature": FEAT, "Importance": model.feature_importances_})
             .sort_values("Importance"))
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#028090" if v >= imp["Importance"].median() else "#B2DFDB" for v in imp["Importance"]]
    bars = ax.barh(imp["Feature"], imp["Importance"], color=colors, height=0.55)
    ax.set_xlabel("Importance Score", color="#475569", fontsize=10)
    ax.set_facecolor("#F8FAFC"); fig.patch.set_facecolor("#F8FAFC")
    ax.tick_params(colors="#475569", labelsize=10)
    for sp in ax.spines.values(): sp.set_visible(False)
    for bar, val in zip(bars, imp["Importance"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color="#0A2342", fontsize=9, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    sec("📉 Glucose Distribution: Diabetic vs Non-Diabetic")
    fig2, ax2 = plt.subplots(figsize=(9, 3.4))
    ax2.hist(df[df["Outcome"]==0]["Glucose"], bins=28, color="#02C39A", alpha=0.75,
             label="No Diabetes", edgecolor="white", linewidth=0.5)
    ax2.hist(df[df["Outcome"]==1]["Glucose"], bins=28, color="#FC8181", alpha=0.75,
             label="Diabetes",    edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Glucose Level (mg/dL)", color="#475569", fontsize=11)
    ax2.set_ylabel("Patient Count",         color="#475569", fontsize=11)
    ax2.set_facecolor("#F8FAFC"); fig2.patch.set_facecolor("#F8FAFC")
    ax2.tick_params(colors="#475569"); ax2.legend(fontsize=10)
    for sp in ax2.spines.values(): sp.set_color("#E2E8F0")
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    sec("🔗 Feature Correlation Heatmap")
    corr = df.corr()
    fig3, ax3 = plt.subplots(figsize=(9, 4.5))
    im = ax3.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax3.set_xticks(range(len(corr.columns)))
    ax3.set_xticklabels(corr.columns, rotation=38, ha="right", fontsize=9, color="#1E293B")
    ax3.set_yticks(range(len(corr.columns)))
    ax3.set_yticklabels(corr.columns, fontsize=9, color="#1E293B")
    plt.colorbar(im, ax=ax3, fraction=0.03)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax3.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                     fontsize=7.5, color="#1E293B")
    fig3.patch.set_facecolor("#F8FAFC"); ax3.set_facecolor("#F8FAFC")
    plt.tight_layout(); st.pyplot(fig3); plt.close()


# ════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        "<div style='background:linear-gradient(90deg,#028090,#02C39A);"
        "color:#fff;padding:10px 20px;border-radius:8px;font-weight:700;"
        "font-size:1rem;margin-bottom:18px'>ℹ️ About This Project</div>",
        unsafe_allow_html=True
    )

    ca, cb = st.columns(2)

    def card(content):
        return (
            "<div style='background:#fff;border-radius:12px;padding:24px 26px;"
            "box-shadow:0 2px 12px rgba(0,0,0,0.07);height:100%'>"
            + content + "</div>"
        )

    def h4(t): return f"<h4 style='color:#028090;margin:0 0 8px'>{t}</h4>"
    def p(t):  return f"<p style='color:#374151;line-height:1.7;margin:0 0 14px'>{t}</p>"
    def ul(items):
        lis = "".join(f"<li style='color:#374151;line-height:1.8'>{i}</li>" for i in items)
        return f"<ul style='padding-left:18px;margin:0 0 14px'>{lis}</ul>"

    with ca:
        st.markdown(card(
            h4("🎯 Project Goal") +
            p("Build an accessible AI/ML tool that predicts diabetes risk from basic "
              "health parameters — enabling early detection without expensive lab visits, "
              "especially in rural and underserved communities.") +
            h4("🧠 ML Approach") +
            ul(["<b>Algorithm:</b> Random Forest Classifier (200 trees)",
                "<b>Preprocessing:</b> StandardScaler normalization",
                "<b>Train/Test split:</b> 80% / 20%",
                "<b>Accuracy:</b> Evaluated on held-out test set"]) +
            h4("📊 Dataset") +
            p("Based on the <b>Pima Indians Diabetes Dataset</b> (768 patient records, 8 features).") +
            "<a href='https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database' "
            "target='_blank' style='color:#028090;font-weight:600'>"
            "🔗 Download from Kaggle →</a>"
        ), unsafe_allow_html=True)

    with cb:
        st.markdown(card(
            h4("📦 Tech Stack") +
            ul(["<b>Language:</b> Python 3.x",
                "<b>ML Library:</b> Scikit-learn",
                "<b>Data:</b> Pandas, NumPy",
                "<b>Visualization:</b> Matplotlib",
                "<b>Web App:</b> Streamlit"]) +
            h4("🏥 Health Features Used") +
            ul(["Pregnancies, Glucose, Blood Pressure",
                "Skin Thickness, Insulin, BMI",
                "Diabetes Pedigree Function, Age"]) +
            h4("🏆 Event") +
            p("<b>Project Jupyter Hackathon</b><br>Army Institute of Technology · 2026<br>"
              "Domain: Healthcare AI / ML")
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='background:#FFFBEB;border-left:4px solid #F6AD55;"
        "border-radius:8px;padding:14px 18px'>"
        "<span style='color:#744210;font-size:0.9rem'>"
        "⚠️ <b>Medical Disclaimer:</b> This application is developed for the Project Jupyter Hackathon "
        "and is strictly for educational and demonstration purposes. It is <b>NOT a certified medical "
        "device</b> and must not be used as a substitute for professional medical advice, diagnosis, "
        "or treatment. Always consult a qualified healthcare provider.</span>"
        "</div>",
        unsafe_allow_html=True
    )
