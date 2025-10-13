import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Skill Gap Analyzer — PRO", layout="wide", page_icon="💼")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg,#0f172a,#1e293b); color: #e6eef6;}
.main-title { text-align:center; font-size:2.4rem; color:#7cfff0; font-weight:700; }
.card { background: rgba(255,255,255,0.04); padding:16px; border-radius:12px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); }
.small { font-size:0.9rem; color:#cfe9f5 }
</style>
""", unsafe_allow_html=True)

def load_lottie_url(url):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except:
        return None

def st_lottie_safe(lottie_json, height=150):
    try:
        from streamlit_lottie import st_lottie
        if lottie_json:
            st_lottie(lottie_json, height=height)
    except:
        pass

LOTTIE_SUCCESS = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json")

@st.cache_data
def generate_synthetic_data(n=500, random_state=42):
    np.random.seed(random_state)
    years = ["1st","2nd","3rd","4th"]
    branches = ["CSE","ECE","AIML","IT","EEE","MECH","CIVIL"]
    goals_pool = ["Software Engineer","Data Scientist","AI Engineer","Frontend Developer","DevOps Engineer","Product Manager","Data Analyst","Embedded Engineer"]
    missing_skills_pool = ["Cloud Computing","Deep Learning","NLP","Frontend Frameworks","DevOps","Project Management","Databases","Computer Vision","Model Deployment"]
    rows = []
    for _ in range(n):
        year = np.random.choice(years, p=[0.15,0.35,0.3,0.2])
        branch = np.random.choice(branches, p=[0.25,0.15,0.12,0.15,0.12,0.12,0.09])
        goal = np.random.choice(goals_pool)
        base = {
            "Python": int(np.clip(np.round(np.random.normal(3 + (years.index(year)*0.2), 1)),1,5)),
            "Java": int(np.clip(np.round(np.random.normal(3 + (0 if branch in ['CSE','IT','AIML'] else -0.3), 1)),1,5)),
            "SQL": int(np.clip(np.round(np.random.normal(3 + (0.1*years.index(year)), 1)),1,5)),
            "WebDev": int(np.clip(np.round(np.random.normal(3 - (0.2 if branch in ['CIVIL','MECH'] else 0), 1)),1,5)),
            "Comm": int(np.clip(np.round(np.random.normal(3 - (0.1*years.index(year)), 1)),1,5)),
            "ProblemSolving": int(np.clip(np.round(np.random.normal(3 + (0.1*years.index(year)), 1)),1,5))
        }
        completed_courses = int(np.clip(np.random.poisson(3 + years.index(year)), 0, 20))
        learn_hours = int(np.clip(np.round(np.random.normal(6 + years.index(year)*1.5, 2)), 1, 40))
        low_skills = []
        if base["WebDev"] <= 2:
            low_skills.append("Frontend Frameworks")
        if base["Python"] <= 2 and base["ProblemSolving"] <= 2:
            low_skills.append("Deep Learning")
        if base["SQL"] <= 2:
            low_skills.append("Databases")
        if not low_skills:
            low_skills = [np.random.choice(missing_skills_pool)]
        missing_skill = np.random.choice(low_skills)
        rows.append({
            "Year_of_Study": year,
            "Degree_Branch": branch,
            "Career_Goal": goal,
            "Python_Skill": base["Python"],
            "Java_Skill": base["Java"],
            "SQL_Skill": base["SQL"],
            "WebDev_Skill": base["WebDev"],
            "Communication_Skill": base["Comm"],
            "ProblemSolving_Skill": base["ProblemSolving"],
            "Completed_Courses": completed_courses,
            "Learning_Hours_per_Week": learn_hours,
            "Missing_Skill": missing_skill
        })
    return pd.DataFrame(rows)

st.markdown("<div class='card'><h1 class='main-title'>Skill Gap Analyzer — PRO</h1></div>", unsafe_allow_html=True)
st.markdown("---")
st.header("1) Dataset — Generate or Upload")

col_a, col_b = st.columns([2,1])
with col_a:
    synth_size = st.slider("Synthetic dataset size", 200, 2000, 500, step=100)
    if st.button("Generate Synthetic Dataset"):
        df_generated = generate_synthetic_data(n=synth_size)
        st.success(f"Synthetic dataset created: {len(df_generated)} records")
        st.dataframe(df_generated.head(10))
        st.session_state["df"] = df_generated
with col_b:
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"])
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)
            if "Missing_Skill" not in df_up.columns:
                st.warning("Missing 'Missing_Skill' column.")
            else:
                st.session_state["df"] = df_up.copy()
                st.success("Dataset loaded.")
                st.dataframe(df_up.head())
        except Exception as e:
            st.error(f"Load error: {e}")

if "df" in st.session_state:
    df = st.session_state["df"]
else:
    df = generate_synthetic_data(n=500)
    st.session_state["df"] = df

st.markdown("---")
st.header("2) Model Training & Evaluation")

target = "Missing_Skill"
X = df.drop(columns=[target])
y = df[target].astype(str)
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

transformers = []
if numeric_cols:
    transformers.append(("num", StandardScaler(), numeric_cols))
if cat_cols:
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))

preprocessor = ColumnTransformer(transformers)
rf = RandomForestClassifier(random_state=42)
pipeline = Pipeline([("pre", preprocessor), ("clf", rf)])
param_grid = {"clf__n_estimators": [150, 250], "clf__max_depth": [None, 12], "clf__min_samples_split": [2, 5]}
st.info("Training model with GridSearchCV (3-fold)")
with st.spinner("Training..."):
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring="accuracy", verbose=0)
    grid.fit(X, y)
    best_model = grid.best_estimator_
    cv_score = grid.best_score_ * 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred) * 100
    class_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
st.success(f"Training completed — CV accuracy: {cv_score:.2f}%  •  Test accuracy: {test_acc:.2f}%")
st.write("Best parameters:", grid.best_params_)
st.dataframe(pd.DataFrame(class_report).transpose().round(3))
st.session_state["model"] = best_model
st.session_state["features"] = {"numeric": numeric_cols, "categorical": cat_cols}
st.markdown("---")
st.header("3) Predict Missing Skill")

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        branch = st.selectbox("Degree Branch", sorted(df["Degree_Branch"].unique()))
        year = st.selectbox("Year of Study", sorted(df["Year_of_Study"].unique()))
        goal = st.selectbox("Career Goal", sorted(df["Career_Goal"].unique()))
    with col2:
        py = st.slider("Python Skill (1-5)", 1, 5, 3)
        java = st.slider("Java Skill (1-5)", 1, 5, 3)
        sql = st.slider("SQL Skill (1-5)", 1, 5, 3)
    with col3:
        web = st.slider("WebDev Skill (1-5)", 1, 5, 3)
        comm = st.slider("Communication (1-5)", 1, 5, 3)
        prob = st.slider("Problem Solving (1-5)", 1, 5, 3)
    submitted = st.form_submit_button("Analyze My Skill Gap")
    if submitted:
        if "model" not in st.session_state:
            st.error("Train model first.")
        else:
            input_df = pd.DataFrame([{
                "Year_of_Study": year,
                "Degree_Branch": branch,
                "Career_Goal": goal,
                "Python_Skill": py,
                "Java_Skill": java,
                "SQL_Skill": sql,
                "WebDev_Skill": web,
                "Communication_Skill": comm,
                "ProblemSolving_Skill": prob,
