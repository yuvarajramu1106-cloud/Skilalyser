import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px
import requests

st.set_page_config(page_title="Skill Gap Analyzer", layout="wide", page_icon="💼")

# CSS for dark theme with variety of light colors
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    color: #FFFFFF !important;
}
.main-title { text-align:center; font-size:2.4rem; font-weight:700; background: -webkit-linear-gradient(#00F5A0,#ADFF2F,#00FFFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
h3 { color:#FFD700 !important; }
h4 { color:#ADFF2F !important; }
h5 { color:#00FFFF !important; }
p, label, span, div { color: #FFFFFF !important; }
a { color:#FF69B4 !important; text-decoration:none; }
.stSlider > div > div[data-baseweb="slider"] > div { background: #1e293b !important; }
.stButton > button { background: linear-gradient(90deg, #00DBDE, #FC00FF) !important; color: #FFFFFF !important; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Lottie Animation Loader
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

LOTTIE_SUCCESS = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_sSF6EG.json")

# Synthetic Data Generator
@st.cache_data
def generate_synthetic_data(n=500, random_state=42):
    np.random.seed(random_state)
    years = ["1st","2nd","3rd","4th"]
    branches = ["CSE","ECE","AIML","IT","EEE","MECH","CIVIL"]
    goals_pool = ["Software Engineer","Data Scientist","AI Engineer","Frontend Developer",
                  "DevOps Engineer","Product Manager","Data Analyst","Embedded Engineer"]
    missing_skills_pool = ["Cloud Computing","Deep Learning","NLP","Frontend Frameworks",
                           "DevOps","Project Management","Databases","Computer Vision","Model Deployment"]
    rows = []
    for _ in range(n):
        year = np.random.choice(years)
        branch = np.random.choice(branches)
        goal = np.random.choice(goals_pool)
        skills = {
            "Python": np.random.randint(1,6),
            "Java": np.random.randint(1,6),
            "SQL": np.random.randint(1,6),
            "WebDev": np.random.randint(1,6),
            "Communication": np.random.randint(1,6),
            "ProblemSolving": np.random.randint(1,6)
        }
        low_skills = []
        if skills["WebDev"] <= 2: low_skills.append("Frontend Frameworks")
        if skills["Python"] <= 2 and skills["ProblemSolving"] <= 2: low_skills.append("Deep Learning")
        if skills["SQL"] <= 2: low_skills.append("Databases")
        if not low_skills: low_skills = [np.random.choice(missing_skills_pool)]
        missing_skill = np.random.choice(low_skills)
        rows.append({
            "Year_of_Study": year,
            "Degree_Branch": branch,
            "Career_Goal": goal,
            "Python_Skill": skills["Python"],
            "Java_Skill": skills["Java"],
            "SQL_Skill": skills["SQL"],
            "WebDev_Skill": skills["WebDev"],
            "Communication_Skill": skills["Communication"],
            "ProblemSolving_Skill": skills["ProblemSolving"],
            "Completed_Courses": np.random.randint(0,10),
            "Learning_Hours_per_Week": np.random.randint(1,20),
            "Missing_Skill": missing_skill
        })
    return pd.DataFrame(rows)

# Title
st.markdown("<h1 class='main-title'>💼 Skill Gap Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #00F5A0'>", unsafe_allow_html=True)

# Dataset Section
st.header("1) Dataset — Generate or Upload")
col1, col2 = st.columns([2,1])
with col1:
    synth_size = st.slider("Synthetic dataset size", 200, 2000, 500, step=100)
    if st.button("Generate Synthetic Dataset"):
        df_generated = generate_synthetic_data(n=synth_size)
        st.session_state["df"] = df_generated
        st.dataframe(df_generated.head(10).style.set_properties(**{'color': '#FFFFFF', 'background-color': '#1e293b'}))
with col2:
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"])
    if uploaded:
        if uploaded.name.endswith(".csv"):
            df_up = pd.read_csv(uploaded)
        else:
            df_up = pd.read_excel(uploaded)
        st.session_state["df"] = df_up
        st.dataframe(df_up.head(10).style.set_properties(**{'color': '#FFFFFF', 'background-color': '#1e293b'}))

# Load dataset
if "df" in st.session_state:
    df = st.session_state["df"]
else:
    df = generate_synthetic_data(n=500)
    st.session_state["df"] = df

# Model Accuracy Section
st.markdown("---")
st.header("2) Model Accuracy")
X = df.drop(columns=["Missing_Skill"])
y = df["Missing_Skill"]
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
transformers = []
if num_cols: transformers.append(('num', StandardScaler(), num_cols))
if cat_cols: transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
preprocessor = ColumnTransformer(transformers)
rf_model = RandomForestClassifier(random_state=42)
pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", rf_model)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"Model Accuracy: {acc*100:.2f}%")

# Skill Prediction Section
st.markdown("---")
st.header("3) Predict Your Skill Gap")

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

with st.form("skill_form"):
    col1, col2 = st.columns(2)
    with col1:
        branch = st.selectbox("Degree Branch", sorted(df["Degree_Branch"].unique()))
        year = st.selectbox("Year of Study", sorted(df["Year_of_Study"].unique()))
        goal = st.selectbox("Career Goal", sorted(df["Career_Goal"].unique()))
    with col2:
        st.markdown("### Rate Your Skills (1-5)")
        skills_to_show = career_skills.get(goal, ["Python","Java","SQL","WebDev","Communication","ProblemSolving"])
        skill_inputs = {}
        for skill in skills_to_show:
            skill_inputs[skill] = st.select_slider(skill, options=[1,2,3,4,5], value=3)
    submitted = st.form_submit_button("Analyze My Skill Gap")

if submitted:
    input_data = pd.DataFrame([{
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
    pred_skill = pipeline.predict(input_data)[0]
    st_lottie_safe(LOTTIE_SUCCESS, height=200)
    st.success(f"Predicted Missing Skill: {pred_skill}", icon="🎯")

    # Radar chart
    skill_df = pd.DataFrame({"Skill": list(skill_inputs.keys()), "Score": list(skill_inputs.values())})
    fig = px.line_polar(skill_df, r="Score", theta="Skill", line_close=True, range_r=[0,5], 
                        title="Your Skill Profile", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Free course recommendations
    course_links = {
        "Cloud Computing": ["https://www.coursera.org/learn/gcp-fundamentals", "https://www.aws.training/Details/Curriculum?id=20685"],
        "Deep Learning": ["https://www.coursera.org/specializations/deep-learning", "https://www.freecodecamp.org/news/tag/deep-learning/"],
        "Frontend Frameworks": ["https://www.freecodecamp.org/learn/front-end-development-libraries/"],
        "DevOps": ["https://cloud.google.com/training", "https://learn.microsoft.com/en-us/training/browse/?terms=devops"],
        "Project Management": ["https://www.coursera.org/professional-certificates/google-project-management"],
        "Databases": ["https://www.kaggle.com/learn/SQL"]
    }
    recs = course_links.get(pred_skill, ["https://www.freecodecamp.org/news"])
    st.markdown("### Recommended Free Courses")
    for link in recs:
        st.markdown(f"[{link}]({link})")
