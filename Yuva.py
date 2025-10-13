# skillgap_analyzer.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- New imports for APIs ---
import requests
from urllib.parse import urlencode, quote_plus
import math
import time

# ---------------------------------
# 🌟 PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="Skill Gap Analyzer", layout="wide", page_icon="💼")

# Custom CSS (kept as before) - shortened for brevity; keep your full CSS if desired
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #1F1C2C, #928DAB); color: white; }
    .main-title { text-align:center; font-size:2.3em; color:#00F5A0; }
    .stContainer { background: rgba(255,255,255,0.08); padding:1rem; border-radius:12px; margin-bottom:18px; }
    div.stButton > button { background: linear-gradient(90deg, #00DBDE, #FC00FF); color:white; border-radius:8px; }
    h2,h3 { color:#FAD961; }
    </style>
""", unsafe_allow_html=True)

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

# Load dataset logic
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
    df = df.drop(columns=["Name"], errors="ignore")

st.write("### 📊 Dataset Preview")
st.dataframe(df.head())
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------
# 🔧 MODEL TRAINING (unchanged)
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
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features)
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
# 🔗 Real-time Recommendation helpers
# ---------------------------------

# --- Helper: YouTube Data API (search)
def youtube_search(query, max_results=5, api_key=None):
    """
    Returns a list of dicts: {title, url, channelTitle, publishedAt, description}
    Requires an API key (YouTube Data API v3).
    """
    results = []
    if not api_key:
        return results

    base = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video,playlist",
        "maxResults": min(10, max_results),
        "order": "relevance",
        "safeSearch": "none",
        "key": api_key
    }
    try:
        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        for item in data.get("items", [])[:max_results]:
            kind = item["id"].get("kind", "")
            if "video" in kind:
                url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            elif "playlist" in kind:
                url = f"https://www.youtube.com/playlist?list={item['id']['playlistId']}"
            else:
                url = ""
            results.append({
                "title": item["snippet"]["title"],
                "url": url,
                "channel": item["snippet"].get("channelTitle", ""),
                "publishedAt": item["snippet"].get("publishedAt", ""),
                "description": item["snippet"].get("description", "")
            })
    except Exception as e:
        st.error(f"YouTube API error: {e}")
    return results

# --- Helper: Udemy search (optional) - placeholder using unofficial public endpoints / affiliate
def udemy_search(query, max_results=5, client_id=None, client_secret=None):
    """
    Optional: call Udemy API if credentials are provided.
    Udemy offers an Affiliate / GraphQL API. This code uses a simple search endpoint pattern as an example.
    (When using Udemy API in production, follow Udemy docs and auth flow.)
    """
    results = []
    if not client_id or not client_secret:
        return results
    # Example: Udemy has different APIs; many require token exchange. Here is a simple placeholder.
    try:
        # Placeholder - this won't work until you implement the proper Udemy auth & endpoints per their docs.
        # Keep this as a template for when you register for Udemy API access.
        base = "https://www.udemy.com/api-2.0/courses/"
        params = {"search": query, "page_size": max_results}
        r = requests.get(base, params=params, auth=(client_id, client_secret), timeout=10)
        r.raise_for_status()
        data = r.json()
        for c in data.get("results", [])[:max_results]:
            results.append({
                "title": c.get("title"),
                "url": c.get("url") or f"https://www.udemy.com{c.get('url')}",
                "platform": "Udemy",
                "is_paid": c.get("is_paid", None)
            })
    except Exception as e:
        # Keep silent if Udemy not configured; show nothing
        st.write("")  # noop
    return results

# --- Helper: edX Course Catalog (optional)
def edx_search(query, max_results=5, client_id=None, client_secret=None):
    """
    Optional: edX Course Catalog search (requires access token / client credentials).
    This is a placeholder to show where you would call edX Course Catalog API.
    """
    results = []
    if not client_id or not client_secret:
        return results
    try:
        # Implement OAuth2 token retrieval and call endpoints per edX docs when you have credentials.
        pass
    except Exception as e:
        pass
    return results

# --- Aggregator
def get_course_recommendations(skill, youtube_api_key=None, udemy_creds=None, edx_creds=None, top_n=5):
    """
    Query available APIs and return a list of recommendations.
    Returns: dict with keys: 'youtube', 'udemy', 'edx', each a list of items.
    """
    # Build search query phrases to fetch high-quality tutorial/playlist content
    queries = [
        f"{skill} tutorial playlist",
        f"{skill} full course",
        f"{skill} for beginners",
        f"{skill} crash course"
    ]

    yt_items = []
    # Try multiple queries until we have enough results
    if youtube_api_key:
        for q in queries:
            if len(yt_items) >= top_n:
                break
            items = youtube_search(q, max_results=top_n, api_key=youtube_api_key)
            # append unique urls
            for it in items:
                if it["url"] and all(existing["url"] != it["url"] for existing in yt_items):
                    yt_items.append(it)
                    if len(yt_items) >= top_n:
                        break
            time.sleep(0.05)  # small pause to avoid burst
    # Udemy / edX (optional)
    udemy_items = []
    if udemy_creds:
        udemy_items = udemy_search(skill, max_results=top_n, client_id=udemy_creds.get("id"), client_secret=udemy_creds.get("secret"))

    edx_items = []
    if edx_creds:
        edx_items = edx_search(skill, max_results=top_n, client_id=edx_creds.get("id"), client_secret=edx_creds.get("secret"))

    return {"youtube": yt_items[:top_n], "udemy": udemy_items[:top_n], "edx": edx_items[:top_n]}

# ---------------------------------
# 🎯 PREDICTION + LIVE RECOMMENDATIONS
# ---------------------------------
st.markdown("<div class='stContainer'>", unsafe_allow_html=True)
st.header("🔮 Predict Missing Skill & Get Live Course Recommendations")

col1, col2, col3 = st.columns(3)
with col1:
    branch = st.selectbox("1️⃣ Select your Degree Branch", sorted(df["Degree_Branch"].unique()))
with col2:
    year = st.selectbox("2️⃣ Select your Year of Study", sorted(df["Year_of_Study"].unique()))
with col3:
    goal = st.selectbox("3️⃣ Select your Career Goal", sorted(df["Career_Goal"].unique()))

# --- Load API keys from streamlit secrets (recommended)
# To set secrets, create ~/.streamlit/secrets.toml locally with:
# [youtube]
# key = "YOUR_KEY"
#
# [udemy]
# id = "YOUR_ID"
# secret = "YOUR_SECRET"
#
# [edx]
# id = "YOUR_ID"
# secret = "YOUR_SECRET"

youtube_api_key = st.secrets.get("youtube", {}).get("key") if st.secrets else None
udemy_creds = st.secrets.get("udemy") if st.secrets else None
edx_creds = st.secrets.get("edx") if st.secrets else None

# Also show small UI guidance to add keys
with st.expander("🔐 API Keys / Setup (click to expand)"):
    st.markdown("""
    **YouTube Data API (required for free video-based recommendations)**  
    1. Create a Google Cloud project, enable *YouTube Data API v3*, create an API key.  
    2. Put your key in Streamlit secrets as:
    ```
    [youtube]
    key = "YOUR_YOUTUBE_API_KEY"
    ```
    **Udemy / edX (optional)** — register for API credentials and add to `secrets.toml`.  
    See the docs for each provider (links shown in the app code comments).
    """)

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

        # --- Fetch live recommendations
        recs = get_course_recommendations(predicted_skill, youtube_api_key=youtube_api_key, udemy_creds=udemy_creds, edx_creds=edx_creds, top_n=6)

        # Display YouTube results prominently (free)
        st.subheader("📺 Free Video Courses / Playlists (YouTube)")
        if recs["youtube"]:
            for idx, item in enumerate(recs["youtube"], 1):
                st.markdown(f"**{idx}. [{item['title']}]({item['url']})**  \n_Channel: {item['channel']} • Published: {item.get('publishedAt','N/A')}_")
                if item.get("description"):
                    st.markdown(f"<small>{item['description'][:300]}...</small>", unsafe_allow_html=True)
        else:
            st.warning("No YouTube results found. Make sure you added a valid YouTube API key in Streamlit secrets.")

        # Udemy / edX optional
        if recs["udemy"]:
            st.subheader("💼 Udemy (optional) — live results")
            for u in recs["udemy"]:
                st.markdown(f"- [{u.get('title')}]({u.get('url')}) • Platform: Udemy")
        if recs["edx"]:
            st.subheader("🎓 edX (optional) — live results")
            for e in recs["edx"]:
                st.markdown(f"- [{e.get('title')}]({e.get('url')}) • Platform: edX")

        # Learning Path suggestions
        st.subheader("🗺️ Suggested Learning Path")
        st.markdown(f"""
        - ⏰ **4–10 hours/week** focusing on the resources above.  
        - 🧩 Do **2 small projects** (1 guided tutorial + 1 self project).  
        - 🧾 Keep a learning log and re-run the analyzer monthly to re-check gaps.  
        - 🤝 Join relevant communities (Discord/Reddit/StackOverflow) for practical help.
        """)

    except Exception as e:
        st.error(f"⚠ Prediction failed: {e}")

st.markdown("</div>", unsafe_allow_html=True)
