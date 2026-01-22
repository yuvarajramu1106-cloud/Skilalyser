import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.title("Automatic Construction Material Counting")

# Load model
model = YOLO("yolov8n.pt")

# Upload image
uploaded_file = st.file_uploader("Upload construction material image", type=["jpg","png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detection
    results = model(image)

    brick = cement = steel = 0

    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls[0])]
            if cls == "brick":
                brick += 1
            elif cls == "cement":
                cement += 1
            elif cls == "steel":
                steel += 1

    # Calculations
    brick_volume = brick * (0.19 * 0.09 * 0.09)
    cement_weight = cement * 50

    st.subheader("Results")
    st.write("Brick Count:", brick)
    st.write("Brick Volume (m³):", round(brick_volume, 3))
    st.write("Cement Bags:", cement)
    st.write("Cement Weight (kg):", cement_weight)numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
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
st.header("🔮 Predict Missing Skill ")

col1, col2, col3 = st.columns(3)
with col1:
    branch = st.selectbox("1️⃣ Select your Degree Branch", sorted(df["Degree_Branch"].unique()))
with col2:
    year = st.selectbox("2️⃣ Select your Year of Study", sorted(df["Year_of_Study"].unique()))
with col3:
    goal = st.selectbox("3️⃣ Select your Career Goal", sorted(df["Career_Goal"].unique()))

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

        st_lottie_safe(LOTTIE_SUCCESS, height=160)
        st.success(f"🎯 Predicted Missing Skill: **{predicted_skill}**")
        st.info(f"💡 Tip: Focus on learning **{predicted_skill}** to move closer to your dream role.")

        # Free course recommendations
        curated = {
            "Cloud": [("Google Cloud Fundamentals", "https://www.coursera.org/learn/gcp-fundamentals"),
                      ("AWS Cloud Practitioner Essentials", "https://www.aws.training/Details/Curriculum?id=20685"),
                      ("Azure Fundamentals", "https://learn.microsoft.com/en-us/training/paths/azure-fundamentals/")],
            "Deep Learning": [("Deep Learning Specialization", "https://www.coursera.org/specializations/deep-learning"),
                              ("FreeCodeCamp DL Tutorials", "https://www.freecodecamp.org/news/tag/deep-learning/"),
                              ("YouTube Crash Course", "https://www.youtube.com/results?search_query=deep+learning+course")],
            "NLP": [("Coursera NLP", "https://www.coursera.org/search?query=natural%20language%20processing&price=Free"),
                    ("FreeCodeCamp NLP", "https://www.freecodecamp.org/news/tag/nlp/"),
                    ("YouTube NLP", "https://www.youtube.com/results?search_query=nlp+tutorial")],
            "Frontend": [("freeCodeCamp Front End", "https://www.freecodecamp.org/learn/front-end-development-libraries/"),
                         ("React Full Course", "https://www.youtube.com/results?search_query=react+full+course+free"),
                         ("Coursera Frontend", "https://www.coursera.org/search?query=frontend&price=Free")],
            "Project Management": [("Google Project Management", "https://www.coursera.org/professional-certificates/google-project-management"),
                                   ("edX PM", "https://www.edx.org/learn/project-management"),
                                   ("YouTube PM", "https://www.youtube.com/results?search_query=project+management+course+free")]
        }

        recs = curated.get(predicted_skill, [("FreeCodeCamp Search", f"https://www.freecodecamp.org/news/search/?query={predicted_skill}")])
        st.markdown("### 🎓 Free Course Recommendations:")
        for name, url in recs:
            st.markdown(f"- [{name}]({url})")

    except Exception as e:
        st.error(f"⚠ Prediction failed: {e}")

st.markdown("</div>", unsafe_allow_html=True)
