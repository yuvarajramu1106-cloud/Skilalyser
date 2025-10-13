# skillgap_analyzer.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------
# 🌟 PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="Skill Gap Analyzer", layout="wide", page_icon="💼")

# ---------------------------------
# 🎨 CUSTOM STYLING
# ---------------------------------
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1F1C2C, #928DAB);
        color: white;
    }
    .main-title {
        text-align: center;
        font-size: 2.5em;
        color: #00F5A0;
        text-shadow: 1px 1px 10px rgba(0, 245, 160, 0.6);
        animation: fadeInDown 1s ease-in-out;
    }
    @keyframes fadeInDown {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .stContainer {
        background: rgba(255,255,255,0.08);
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 4px 30px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #00DBDE, #FC00FF);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7em 1.5em;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #FC00FF, #00DBDE);
    }
    h2, h3 {
        color: #FAD961;
    }
    [data-testid="stDataFrame"] {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------
# 🧠 HEADER
# ---------------------------------
st.markdown("<h1 class='main-title'>💼 Universal Skill Gap Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#EDEDED;'>Analyze your skills, identify the gap, and advance your career professionally.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------
# 📂 DATASET SECTION
# ---------------------------------
st.markdown("<div class='stContainer'>", unsafe_allow_html=True)
st.header("📥 Upload or Use Sample Data")

uploaded_file = st.file_uploader("Upload your dataset (CSV or XLSX)", type=["csv", "xlsx"])
use_sample = st.checkbox("Use sample dataset", value=False)

def create_sample_data():
    data = {
        "Year_of_Study": ["2nd", "3rd", "4th", "2nd", "3rd"],
        "Degree_Branch": ["B.E CSE", "B.E ECE", "B.E AIML", "B.E EEE", "B.E IT"],
        "Python_Skill(1-5)": [4, 3, 5, 2, 1],
        "Java_Skill(1-5)": [3, 4, 2, 3, 2],
        "C_C++_Skill(1-5)": [4, 2, 5, 1, 2],
        "SQL_Skill(1-5)": [3, 3, 4, 2, 1],
        "WebDev_Skill(1-5)": [4, 2, 5, 3, 2],
        "Communication_Skill(1-5)": [4, 5, 3, 4, 2],
        "ProblemSolving_Skill(1-5)": [5, 4, 4, 3, 2],
        "Leadership_Skill(1-5)": [3, 4, 3, 2, 1],
        "Teamwork_Skill(1-5)": [5, 4, 3, 4, 3],
        "Completed_Courses": [3, 5, 6, 2, 1],
        "Career_Goal": ["Software Engineer", "Data Scientist", "AI Engineer", "Developer", "Civil Engineer"],
        "Current_Role": ["Intern", "Student", "Research Assistant", "Student", "Trainee"],
        "Desired_Role": ["Backend Developer", "ML Engineer", "AI Developer", "Frontend Dev", "Design Engineer"],
        "Industry_Interest": ["IT", "AI", "AI", "Software", "Construction"],
        "Recent_Certifications": ["Python Basics", "Machine Learning", "Deep Learning", "Java Dev", "AutoCAD"],
        "Learning_Hours_per_Week": [10, 12, 8, 5, 6],
        "Learning_Method": ["Online", "Offline", "Online", "Hybrid", "Online"],
        "Last_Training": ["Python", "ML", "DL", "Java", "AutoCAD"],
        "Missing_Skills": ["Cloud", "Deep Learning", "NLP", "Frontend", "Project Management"],
        "Confidence_Level(1-10)": [8, 7, 9, 6, 5],
        "Challenges": ["Time management", "Lack of resources", "Practical exposure", "Motivation", "Guidance"],
        "Need_Recommendations": ["Yes", "Yes", "No", "Yes", "Yes"]
    }
    return pd.DataFrame(data)

# Load dataset
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
elif use_sample:
    df = create_sample_data()
else:
    st.warning("⚠ Please upload a dataset or use the sample dataset.")
    st.stop()

if "Name" in df.columns:
    df = df.drop(columns=["Name"])

st.write("### 📊 Dataset Preview")
st.dataframe(df.head())
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------
# 🔧 MODEL TRAINING
# ---------------------------------
st.markdown("<div class='stContainer'>", unsafe_allow_html=True)
st.header("🧠 Model Training & Evaluation")

target_col = "Missing_Skills"
feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols]
y = df[target_col]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = RandomForestClassifier(n_estimators=200, random_state=42)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

st.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------
# 🎯 PREDICTION SECTION
# ---------------------------------
st.markdown("<div class='stContainer'>", unsafe_allow_html=True)
st.header("🔮 Predict Missing Skill (5 Quick Questions)")

col1, col2, col3 = st.columns(3)
with col1:
    branch = st.selectbox("🎓 Degree Branch", [
        "B.E CSE", "B.E ECE", "B.E EEE", "B.E IT", "B.E AIML", "B.E AIDS", "B.E CS"
    ])
with col2:
    year = st.selectbox("📘 Year of Study", sorted(df["Year_of_Study"].unique()))
with col3:
    goal = st.selectbox("🚀 Career Goal", sorted(df["Career_Goal"].unique()))

col4, col5 = st.columns(2)
with col4:
    current_role = st.selectbox("💼 Current Role", sorted(df["Current_Role"].unique()))
with col5:
    desired_role = st.selectbox("🌠 Desired Role", sorted(df["Desired_Role"].unique()))

certification = st.selectbox("🏅 Recent Certification", sorted(df["Recent_Certifications"].unique()))

# Professional Lottie Animation Loader
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# A reliable professional Lottie animation (career success theme)
success_anim = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_touohxv0.json")

if st.button("✨ Analyze My Skill Gap"):
    try:
        avg_numeric = df[numeric_features].mean().to_dict()
        input_data = pd.DataFrame([avg_numeric])

        for col in categorical_features:
            input_data[col] = df[col].mode()[0] if col in df.columns else "Unknown"

        input_data["Degree_Branch"] = branch
        input_data["Year_of_Study"] = year
        input_data["Career_Goal"] = goal
        input_data["Current_Role"] = current_role
        input_data["Desired_Role"] = desired_role
        input_data["Recent_Certifications"] = certification

        # ✅ Reindex safely in case of missing columns
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        pred = pipeline.predict(input_data)
        predicted_skill = le.inverse_transform(pred)[0]

        with st.spinner("Analyzing your profile and predicting missing skill..."):
            st.success(f"🎯 Predicted Missing Skill: **{predicted_skill}**")
            st.info(f"💡 Focus on improving **{predicted_skill}** to align with your career goal: **{goal}**")

        # ✅ Professional animation
        if success_anim:
            st_lottie(success_anim, height=200, key="success_anim")
        else:
            st.info("✅ Analysis completed successfully!")

    except Exception as e:
        st.error(f"⚠ Prediction failed: {e}")

st.markdown("</div>", unsafe_allow_html=True)
