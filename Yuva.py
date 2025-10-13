# ============================================
# 💼 UNIVERSAL SKILL GAP ANALYZER (Final Version)
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
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

# Custom CSS Styling
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
    div[data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.1);
        color: white !important;
        border-radius: 8px;
    }
    h2, h3 { color: #FAD961; }
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
st.markdown("<p style='text-align:center; color:#EDEDED;'>Discover what’s holding you back — and bridge your skill gap today.</p>", unsafe_allow_html=True)
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
        "Degree_Branch": ["CSE", "ECE", "AIML", "EEE", "CIVIL"],
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
        "Industry_Interest": ["IT", "AI", "AI", "Software", "Construction"],
        "Learning_Hours_per_Week": [10, 12, 8, 5, 6],
        "Learning_Method": ["Online", "Offline", "Online", "Hybrid", "Online"],
        "Last_Training": ["Python", "ML", "DL", "Java", "AutoCAD"],
        "Desired Role": ["Backend Developer", "ML Engineer", "AI Developer", "Frontend Dev", "Design Engineer"],
        "Missing_Skills": ["Cloud", "Deep Learning", "NLP", "Frontend", "Project Management"],
        "Confidence_Level(1-10)": [8, 7, 9, 6, 5],
        "Challenges": ["Time management", "Lack of resources", "Practical exposure", "Motivation", "Guidance"],
        "Need_Recommendations": ["Yes", "Yes", "No", "Yes", "Yes"]
    }
    return pd.DataFrame(data)

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
st.header("🔮 Predict Missing Skill (Only 3 Questions)")

col1, col2, col3 = st.columns(3)
with col1:
    branch = st.selectbox("1️⃣ Select your Degree Branch", sorted(df["Degree_Branch"].unique()))
with col2:
    year = st.selectbox("2️⃣ Select your Year of Study", sorted(df["Year_of_Study"].unique()))
with col3:
    goal = st.selectbox("3️⃣ Select your Career Goal", sorted(df["Career_Goal"].unique()))

predicted_skill = None

if st.button("✨ Analyze My Skill Gap"):
    try:
        avg_numeric = df[numeric_features].mean().to_dict()
        input_data = pd.DataFrame([avg_numeric])

        for col in categorical_features:
            input_data[col] = df[col].mode()[0] if col in df.columns else "Unknown"

        input_data["Degree_Branch"] = branch
        input_data["Year_of_Study"] = year
        input_data["Career_Goal"] = goal
        input_data = input_data[X.columns]

        pred = pipeline.predict(input_data)
        predicted_skill = le.inverse_transform(pred)[0]

        st.success(f"🎯 Predicted Missing Skill: **{predicted_skill}**")
        st.info(f"💡 Tip: Focus on learning **{predicted_skill}** to move closer to your dream role.")
        st.balloons()
    except Exception as e:
        st.error(f"⚠ Prediction failed: {e}")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------
# 🎓 COURSE RECOMMENDATION (NO API KEY)
# ---------------------------------
st.markdown("<div class='stContainer'>", unsafe_allow_html=True)
st.header("🎓 Recommended Free Courses")

def get_free_courses(skill):
    links = []
    if not skill:
        return []
    try:
        links.append(("FreeCodeCamp Courses", f"https://www.freecodecamp.org/news/search/?query={skill}"))
        links.append(("Kaggle Learn Courses", f"https://www.kaggle.com/learn/search?query={skill}"))
        links.append(("Coursera Free Courses", f"https://www.coursera.org/search?query={skill}&price=Free"))
        links.append(("YouTube Tutorials", f"https://www.youtube.com/results?search_query={skill}+free+course"))
    except Exception as e:
        st.error(f"⚠️ Error fetching links: {e}")
    return links

if predicted_skill:
    if st.button("🎯 Get Free Course Recommendations"):
        st.subheader(f"Top Free Learning Resources for '{predicted_skill}' 👇")
        with st.spinner("Fetching course links..."):
            courses = get_free_courses(predicted_skill)
            for name, url in courses:
                st.markdown(f"- [{name}]({url})")
st.markdown("</div>", unsafe_allow_html=True)
