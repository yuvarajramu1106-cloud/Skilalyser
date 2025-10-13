import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load dataset for UI
df = pd.read_csv("skill_gap_dataset.csv")

# Load model and encoders
model = pickle.load(open("models/skill_gap_model.pkl", "rb"))
le_stream = pickle.load(open("models/le_stream.pkl", "rb"))
mlb = pickle.load(open("models/mlb.pkl", "rb"))

# Free course suggestions
free_courses = {
    "Python": "https://www.coursera.org/learn/python",
    "Java": "https://www.udemy.com/course/java-tutorial/",
    "C++": "https://www.learncpp.com/",
    "SQL": "https://www.khanacademy.org/computing/computer-programming/sql",
    "Machine Learning": "https://www.coursera.org/learn/machine-learning",
    "Data Analysis": "https://www.coursera.org/learn/data-analysis",
    "Web Development": "https://www.freecodecamp.org/",
    "React": "https://reactjs.org/tutorial/tutorial.html",
    "Flutter": "https://flutter.dev/docs/get-started/codelab",
    "AI": "https://www.coursera.org/learn/ai-for-everyone",
    "Deep Learning": "https://www.deeplearning.ai/",
    "Communication": "https://www.edx.org/course/communication-skills"
}

st.title("Skill Gap Analyzer")

# User Inputs
name = st.text_input("Name")
stream = st.selectbox("Stream", df['Stream'].unique())
gpa = st.slider("GPA", 5.0, 10.0, 7.0)
projects = st.number_input("Number of Projects", 0, 10, 1)
internships = st.number_input("Number of Internships", 0, 5, 0)
skills_input = st.multiselect("Select your skills", list(mlb.classes_))

if st.button("Analyze Skill Gap"):
    # Encode inputs
    stream_enc = le_stream.transform([stream])[0]
    skills_enc = mlb.transform([skills_input])
    features = np.concatenate(([gpa, projects, internships, stream_enc], skills_enc[0])).reshape(1,-1)
    
    # Predict skill gap
    gap_pred = model.predict(features)[0]
    st.success(f"Predicted Skill Gap: {gap_pred}")
    
    # Suggest free courses for missing skills
    st.write("Recommended Free Courses:")
    for skill in mlb.classes_:
        if skill not in skills_input and skill in free_courses:
            st.markdown(f"- [{skill}]({free_courses[skill]})")
