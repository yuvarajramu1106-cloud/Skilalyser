# ============================================
# 💼 UNIVERSAL SKILL GAP ANALYZER — PRO (All Upgrades)
# ============================================
"""
Features:
- Synthetic dataset generator (configurable size) for robust training
- Model training with GridSearchCV + cross-validation
- Improved prediction UI: sliders + categorical choices
- Curated free course links (no API keys)
- Professional Lottie animation (success) instead of balloons
- Radar chart (Plotly) to visualize user's skill profile
- Session-based progress tracker with CSV download
- In-app list of recommended upgrades (next steps)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import json
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
from datetime import datetime

# -------------------------
# Page config & CSS
# -------------------------
st.set_page_config(page_title="Skill Gap Analyzer — PRO", layout="wide", page_icon="💼")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg,#0f172a,#1e293b); color: #e6eef6;}
.main-title { text-align:center; font-size:2.4rem; color:#7cfff0; font-weight:700; }
.card { background: rgba(255,255,255,0.04); padding:16px; border-radius:12px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); }
.small { font-size:0.9rem; color:#cfe9f5 }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Lottie loader
# -------------------------
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except:
        return None

LOTTIE_SUCCESS = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json")

def st_lottie_safe(lottie_json, height=150):
    try:
        # only import when needed
        from streamlit_lottie import st_lottie
        if lottie_json:
            st_lottie(lottie_json, height=height)
    except Exception:
        pass

# -------------------------
# Synthetic dataset generator
# -------------------------
@st.cache_data
def generate_synthetic_data(n=500, random_state=42):
    np.random.seed(random_state)
    years = ["1st","2nd","3rd","4th"]
    branches = ["CSE","ECE","AIML","IT","EEE","MECH","CIVIL"]
    goals_pool = [
        "Software Engineer","Data Scientist","AI Engineer","Frontend Developer",
        "DevOps Engineer","Product Manager","Data Analyst","Embedded Engineer"
    ]
    missing_skills_pool = [
        "Cloud Computing","Deep Learning","NLP","Frontend Frameworks",
        "DevOps","Project Management","Databases","Computer Vision","Model Deployment"
    ]

    rows = []
    for _ in range(n):
        year = np.random.choice(years, p=[0.15,0.35,0.3,0.2])
        branch = np.random.choice(branches, p=[0.25,0.15,0.12,0.15,0.12,0.12,0.09])
        goal = np.random.choice(goals_pool)
        # skill scores correlated with branch and year (simple heuristic)
        base = {
            "Python": np.clip(int(np.random.normal(3 + (years.index(year)*0.2), 1)),1,5),
            "Java": np.clip(int(np.random.normal(3 + (0 if branch in ['CSE','IT','AIML'] else -0.3), 1)),1,5),
            "SQL": np.clip(int(np.random.normal(3 + (0.1*years.index(year)), 1)),1,5),
            "WebDev": np.clip(int(np.random.normal(3 - (0.2 if branch in ['CIVIL','MECH'] else 0), 1)),1,5),
            "Comm": np.clip(int(np.random.normal(3 - (0.1*years.index(year)), 1)),1,5),
            "ProblemSolving": np.clip(int(np.random.normal(3 + (0.1*years.index(year)), 1)),1,5)
        }
        # Completed courses and learning hours
        completed_courses = int(np.random.poisson(3 + years.index(year)))
        learn_hours = int(np.random.normal(6 + years.index(year)*1.5, 2))
        # Missing skill label chosen based on low domain competence
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
            "Learning_Hours_per_Week": max(1, learn_hours),
            "Missing_Skill": missing_skill
        })
    return pd.DataFrame(rows)

# -------------------------
# Load / prepare dataset
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'>"
            "<h1 class='main-title'>Skill Gap Analyzer — PRO</h1>"
            "<div style='text-align:right'><span class='small'>Professional ML + Recommendations</span></div>"
            "</div>", unsafe_allow_html=True)
st.markdown("</div>")

st.markdown("---")
st.header("1) Dataset — Generate or Upload")

col_a, col_b = st.columns([2,1])
with col_a:
    st.markdown("**Use synthetic dataset (recommended for demo/training)**")
    synth_size = st.slider("Synthetic dataset size (records)", 200, 2000, 500, step=100)
    if st.button("Generate Synthetic Dataset"):
        df = generate_synthetic_data(n=synth_size)
        st.success(f"Synthetic dataset created: {len(df)} records")
        st.dataframe(df.head(10))
        st.session_state["df"] = df
else:
    # check session
    if "df" in st.session_state:
        df = st.session_state["df"]
    else:
        df = generate_synthetic_data(n=500)
        st.session_state["df"] = df

with col_b:
    uploaded = st.file_uploader("Or upload CSV/XLSX (optional)", type=["csv","xlsx"])
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)
            # minimal validation
            if "Missing_Skill" not in df_up.columns:
                st.warning("Uploaded file does not contain 'Missing_Skill' column. Use synthetic or upload labeled data.")
            else:
                df = df_up.copy()
                st.session_state["df"] = df
                st.success("Dataset uploaded and loaded.")
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to load file: {e}")

st.markdown("---")

# -------------------------
# Preprocess & Train Model
# -------------------------
st.header("2) Model Training & Evaluation (Automatic)")

# guard
if df is None or len(df) < 50:
    st.warning("Dataset too small. Generate synthetic data (>=200) for robust model training.")
else:
    # Prepare features/labels
    target = "Missing_Skill"
    X = df.drop(columns=[target])
    y = df[target].astype(str)

    # Identify numeric & categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Preprocessor
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
    ])

    # Pipeline with RandomForest
    rf = RandomForestClassifier(random_state=42)
    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", rf)
    ])

    # Hyperparam grid (kept small to be quick on Streamlit Cloud)
    param_grid = {
        "clf__n_estimators": [150, 250],
        "clf__max_depth": [None, 12],
        "clf__min_samples_split": [2, 5]
    }

    st.info("Training RandomForest with GridSearchCV (3-fold). This may take ~30-90s depending on dataset size.")

    with st.spinner("Training model..."):
        grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring="accuracy", verbose=0)
        grid.fit(X, y)
        best_model = grid.best_estimator_
        # metrics
        cv_score = grid.best_score_ * 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred) * 100
        class_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    st.success(f"Training completed — CV accuracy: {cv_score:.2f}%  •  Test accuracy: {test_acc:.2f}%")
    st.write("**Best parameters:**", grid.best_params_)
    # show classification report summary
    report_df = pd.DataFrame(class_report).transpose()
    st.dataframe(report_df.round(3))

    # store model & metadata in session
    st.session_state["model"] = best_model
    st.session_state["features"] = {"numeric": numeric_cols, "categorical": cat_cols}
    st.session_state["label_encoder_classes"] = sorted(df[target].unique())

st.markdown("---")

# -------------------------
# Prediction UI (improved)
# -------------------------
st.header("3) Predict Missing Skill — Interactive")

# Create input form
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

    submitted = st.form_submit_button("Analyze My Skill Gap (PRO)")
    if submitted:
        if "model" not in st.session_state:
            st.error("Model not trained yet. Please generate dataset and train.")
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
                "Completed_Courses": 0,
                "Learning_Hours_per_Week": 0
            }])
            model = st.session_state["model"]
            pred = model.predict(input_df)[0]
            st_lottie_safe(LOTTIE_SUCCESS, height=160)
            st.success(f"Predicted Missing Skill: **{pred}**")
            st.info("Interpretation: This is a machine-suggested area you should prioritize. Use the recommended free courses below.")

            # Save to session progress tracker
            if "progress" not in st.session_state:
                st.session_state["progress"] = []
            st.session_state["progress"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "branch": branch,
                "year": year,
                "goal": goal,
                "python": py,
                "java": java,
                "sql": sql,
                "webdev": web,
                "comm": comm,
                "problem": prob,
                "predicted_missing": pred
            })

            # Visualize user skill radar chart
            skill_df = pd.DataFrame({
                "skill": ["Python","Java","SQL","WebDev","Communication","ProblemSolving"],
                "score": [py, java, sql, web, comm, prob]
            })
            fig = px.line_polar(skill_df, r="score", theta="skill", line_close=True,
                                title="Your Skill Profile (1-5)", range_r=[0,5])
            st.plotly_chart(fig, use_container_width=True)

            # Display recommended free courses (curated)
            st.markdown("### 🎓 Curated Free Courses & Resources (No API required)")
            # curated mapping
            curated = {
                "Cloud Computing": [
                    ("Google Cloud - Fundamentals (Coursera) — Audit free", "https://www.coursera.org/learn/gcp-fundamentals"),
                    ("AWS Cloud Practitioner Essentials (free training)", "https://www.aws.training/Details/Curriculum?id=20685"),
                    ("Microsoft Learn — Azure Fundamentals", "https://learn.microsoft.com/en-us/training/paths/azure-fundamentals/")
                ],
                "Deep Learning": [
                    ("Deep Learning Specialization (Coursera) — Audit", "https://www.coursera.org/specializations/deep-learning"),
                    ("FreeCodeCamp Deep Learning tutorials", "https://www.freecodecamp.org/news/tag/deep-learning/"),
                    ("YouTube: Deep Learning Crash Course (search)", "https://www.youtube.com/results?search_query=deep+learning+crash+course")
                ],
                "NLP": [
                    ("Coursera NLP courses (audit) — search", "https://www.coursera.org/search?query=natural%20language%20processing&price=Free"),
                    ("FreeCodeCamp NLP articles & tutorials", "https://www.freecodecamp.org/news/tag/nlp/"),
                    ("YouTube: NLP tutorial (search)", "https://www.youtube.com/results?search_query=nlp+tutorial")
                ],
                "Frontend Frameworks": [
                    ("freeCodeCamp Front End Development", "https://www.freecodecamp.org/learn/front-end-development-libraries/"),
                    ("Kaggle / YouTube Frontend playlists (search)", "https://www.youtube.com/results?search_query=react+full+course+free"),
                    ("Coursera: Front-End Web Development (audit)", "https://www.coursera.org/search?query=frontend&price=Free")
                ],
                "DevOps": [
                    ("Google Cloud / AWS DevOps training (free modules)", "https://cloud.google.com/training"),
                    ("Microsoft Learn — DevOps modules", "https://learn.microsoft.com/en-us/training/browse/?terms=devops"),
                    ("YouTube: DevOps Full Course (search)", "https://www.youtube.com/results?search_query=devops+full+course+free")
                ],
                "Project Management": [
                    ("Google Project Management: Professional Certificate (audit where possible)", "https://www.coursera.org/professional-certificates/google-project-management"),
                    ("edX project management courses (search)", "https://www.edx.org/learn/project-management"),
                    ("YouTube: Project Management basics (search)", "https://www.youtube.com/results?search_query=project+management+course+free")
                ],
                "Databases": [
                    ("Kaggle SQL courses", "https://www.kaggle.com/learn/SQL"),
                    ("Mode SQL Tutorial", "https://mode.com/sql-tutorial/"),
                    ("freeCodeCamp SQL articles", "https://www.freecodecamp.org/news/tag/sql/")
                ],
                "Computer Vision": [
                    ("Coursera / free resources (search)", "https://www.coursera.org/search?query=computer%20vision&price=Free"),
                    ("freeCodeCamp Computer Vision tag", "https://www.freecodecamp.org/news/tag/computer-vision/"),
                    ("YouTube: Computer Vision course (search)", "https://www.youtube.com/results?search_query=computer+vision+course+free")
                ],
                "Model Deployment": [
                    ("FastAPI + Docker tutorials (YouTube search)", "https://www.youtube.com/results?search_query=fastapi+docker+deployment+tutorial"),
                    ("AWS/GCP deployment guides (official docs)", "https://cloud.google.com/community/tutorials"),
                    ("FreeCodeCamp deployment articles", "https://www.freecodecamp.org/news/tag/deployment/")
                ]
            }
            # show curated links for predicted skill, fallback to general search links
            recommended = curated.get(pred, [
                ("FreeCodeCamp search", f"https://www.freecodecamp.org/news/search/?query={pred}"),
                ("Coursera Free search", f"https://www.coursera.org/search?query={pred}&price=Free"),
                ("YouTube search", f"https://www.youtube.com/results?search_query={pred}+free+course")
            ])
            for name, url in recommended:
                st.markdown(f"- [{name}]({url})")

st.markdown("---")

# -------------------------
# Progress Tracker
# -------------------------
st.header("4) Progress Tracker (Session)")
if "progress" in st.session_state and st.session_state["progress"]:
    progress_df = pd.DataFrame(st.session_state["progress"])
    st.write("Recent analyses (session):")
    st.dataframe(progress_df.tail(10))
    # download CSV
    csv = progress_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download session progress CSV", data=csv, file_name="skillgap_progress.csv", mime="text/csv")
else:
    st.info("No session progress yet — run an analysis to record entries.")

st.markdown("---")

# -------------------------
# Professional Upgrade Recommendations (in-app)
# -------------------------
st.header("5) Recommended Next Upgrades (Professional)")
st.markdown("""
Below are **concrete improvements** to make this project production-grade:
- **Increase labeled dataset size** — collect 500–2000 real student records or synthesize more varied examples.
- **Feature engineering** — extract more features (GPA, project count, internships, resume keywords).
- **Model ensemble & calibration** — try XGBoost/LightGBM, model stacking, and probability calibration.
- **Explainability** — add SHAP feature importance to explain predictions to users.
- **Quality ranking for courses** — scrape view counts / ratings and rank recommended resources.
- **Persistence** — store user profiles and progress in a small DB (SQLite or Airtable) rather than session state.
- **Resume analyzer** — implement an uploader that extracts text, maps keywords to skills, and auto-fills sliders.
- **CI/CD & tests** — add unit tests for preprocessing, model training and deploy via GitHub Actions.
""")

st.markdown("---")
st.markdown("<div style='text-align:center; color:#9fb3c8'>Built: Skill Gap Analyzer — PRO • No external payment required • Lottie animations for UI polish</div>", unsafe_allow_html=True)
