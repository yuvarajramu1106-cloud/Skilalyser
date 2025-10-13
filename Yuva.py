import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

st.set_page_config(page_title="Skill Gap Analyzer — PRO", layout="wide", page_icon="💼")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg,#0f172a,#1e293b); color: #e6eef6;}
.main-title { text-align:center; font-size:2.4rem; color:#7cfff0; font-weight:700; }
.card { background: rgba(255,255,255,0.04); padding:16px; border-radius:12px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); }
.small { font-size:0.9rem; color:#cfe9f5 }
h3 { color: #00F5A0; font-weight:600; }
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
        branch = np.random.choice(branches)
        goal = np.random.choice(goals_pool)
        base = {
            "Python": int(np.clip(np.round(np.random.normal(3, 1)),1,5)),
            "Java": int(np.clip(np.round(np.random.normal(3, 1)),1,5)),
            "SQL": int(np.clip(np.round(np.random.normal(3, 1)),1,5)),
            "WebDev": int(np.clip(np.round(np.random.normal(3, 1)),1,5)),
            "Communication": int(np.clip(np.round(np.random.normal(3, 1)),1,5)),
            "ProblemSolving": int(np.clip(np.round(np.random.normal(3, 1)),1,5))
        }
        completed_courses = int(np.clip(np.random.poisson(3), 0, 20))
        learn_hours = int(np.clip(np.round(np.random.normal(6,2)), 1, 40))
        low_skills = []
        if base["WebDev"] <= 2: low_skills.append("Frontend Frameworks")
        if base["Python"] <= 2 and base["ProblemSolving"] <= 2: low_skills.append("Deep Learning")
        if base["SQL"] <= 2: low_skills.append("Databases")
        if not low_skills: low_skills = [np.random.choice(missing_skills_pool)]
        missing_skill = np.random.choice(low_skills)
        rows.append({
            "Year_of_Study": year,
            "Degree_Branch": branch,
            "Career_Goal": goal,
            "Python_Skill": base["Python"],
            "Java_Skill": base["Java"],
            "SQL_Skill": base["SQL"],
            "WebDev_Skill": base["WebDev"],
            "Communication_Skill": base["Communication"],
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
st.header("2) Model Accuracy")

if "df" in st.session_state:
    target = "Missing_Skill"
    X = df.drop(columns=[target])
    y = df[target].astype(str)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    transformers = []
    if numeric_cols: transformers.append(("num", StandardScaler(), numeric_cols))
    if cat_cols: transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    preprocessor = ColumnTransformer(transformers)

    rf = RandomForestClassifier(random_state=42)
    pipeline = Pipeline([("pre", preprocessor), ("clf", rf)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred) * 100

    st.markdown(f"<h3>Model Test Accuracy: {test_acc:.2f}%</h3>", unsafe_allow_html=True)
else:
    st.info("Model not trained yet or dataset missing.")

st.markdown("---")
st.header("3) Predict Missing Skill Dynamically")

career_skills = {
    "Software Engineer":["Python","Java","SQL","ProblemSolving"],
    "Data Scientist":["Python","SQL","ProblemSolving","Communication"],
    "AI Engineer":["Python","Deep Learning","ProblemSolving","SQL"],
    "Frontend Developer":["JavaScript","WebDev","Communication","ProblemSolving"],
    "DevOps Engineer":["Python","DevOps","Communication"],
    "Product Manager":["Communication","ProblemSolving","Project Management"],
    "Data Analyst":["Python","SQL","Communication"],
    "Embedded Engineer":["C/C++","ProblemSolving","Communication"]
}

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        branch = st.selectbox("Degree Branch", sorted(df["Degree_Branch"].unique()))
        year = st.selectbox("Year of Study", sorted(df["Year_of_Study"].unique()))
        goal = st.selectbox("Career Goal", sorted(df["Career_Goal"].unique()))
    with col2:
        st.markdown("### Relevant Skills")
        skills_to_show = career_skills.get(goal, ["Python","Java","SQL","WebDev","Communication","ProblemSolving"])
        skill_inputs = {}
        for skill in skills_to_show:
            skill_inputs[skill] = st.slider(f"{skill} Skill (1-5)", 1, 5, 3)

    submitted = st.form_submit_button("Analyze My Skill Gap")
    if submitted:
        input_df = pd.DataFrame([{
            "Year_of_Study": year,
            "Degree_Branch": branch,
            "Career_Goal": goal,
            "Python_Skill": skill_inputs.get("Python",3),
            "Java_Skill": skill_inputs.get("Java",3),
            "SQL_Skill": skill_inputs.get("SQL",3),
            "WebDev_Skill": skill_inputs.get("WebDev",3),
            "Communication_Skill": skill_inputs.get("Communication",3),
            "ProblemSolving_Skill": skill_inputs.get("ProblemSolving",3),
            "Completed_Courses": 0,
            "Learning_Hours_per_Week": 0
        }])
        pipeline.fit(X, y)
        pred = pipeline.predict(input_df)[0]
        st_lottie_safe(LOTTIE_SUCCESS, height=160)
        st.success(f"Predicted Missing Skill: {pred}")

        curated = {
            "Cloud Computing": [("Google Cloud Fundamentals","https://www.coursera.org/learn/gcp-fundamentals"),
                                ("AWS Cloud Practitioner Essentials","https://www.aws.training/Details/Curriculum?id=20685"),
                                ("Azure Fundamentals","https://learn.microsoft.com/en-us/training/paths/azure-fundamentals/")],
            "Deep Learning": [("Deep Learning Specialization","https://www.coursera.org/specializations/deep-learning"),
                              ("FreeCodeCamp Tutorials","https://www.freecodecamp.org/news/tag/deep-learning/"),
                              ("YouTube Crash Course","https://www.youtube.com/results?search_query=deep+learning+course")],
            "NLP": [("Coursera NLP","https://www.coursera.org/search?query=natural%20language%20processing&price=Free"),
                    ("FreeCodeCamp NLP","https://www.freecodecamp.org/news/tag/nlp/"),
                    ("YouTube NLP","https://www.youtube.com/results?search_query=nlp+tutorial")],
            "Frontend Frameworks": [("freeCodeCamp Front End","https://www.freecodecamp.org/learn/front-end-development-libraries/"),
                                    ("React Full Course","https://www.youtube.com/results?search_query=react+full+course+free"),
                                    ("Coursera Frontend","https://www.coursera.org/search?query=frontend&price=Free")],
            "DevOps": [("Google Cloud Training","https://cloud.google.com/training"),
                       ("Microsoft Learn DevOps","https://learn.microsoft.com/en-us/training/browse/?terms=devops"),
                       ("YouTube DevOps","https://www.youtube.com/results?search_query=devops+full+course+free")],
            "Project Management": [("Google Project Management","https://www.coursera.org/professional-certificates/google-project-management"),
                                   ("edX PM","https://www.edx.org/learn/project-management"),
                                   ("YouTube PM
