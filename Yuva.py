# skillgap_analyser_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ----------------------------
# Load ML model
# ----------------------------
# Make sure your model is saved as 'skillgap_model.pkl'
with open("skillgap_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Skillgap Analyser", layout="wide")
st.title("Skillgap Analyser 🧠")
st.markdown("""
Upload a CSV file with skills data and enter your details to analyze your skill gap.
""")

# ----------------------------
# Step 1: CSV Upload
# ----------------------------
st.header("Step 1: Upload CSV")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully!")
    st.dataframe(df.head())

    # Automatically detect skill columns (excluding name, experience, role)
    skill_columns = [col for col in df.columns if col.lower() not in ["name", "experience", "role"]]

    # ----------------------------
    # Step 2: User Input
    # ----------------------------
    st.header("Step 2: Enter Your Details")
    name = st.text_input("Name")
    experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=1)
    
    roles_from_csv = df['role'].unique() if 'role' in df.columns else ["Developer", "Data Scientist", "Analyst", "Manager", "Other"]
    role = st.selectbox("Current Role", roles_from_csv)
    
    st.markdown(f"List your skills separated by commas (from CSV columns: {', '.join(skill_columns)})")
    skills_input = st.text_area("Your Skills")
    skills_list = [skill.strip().lower() for skill in skills_input.split(",")]

    # ----------------------------
    # Step 3: Predict & Visualize
    # ----------------------------
    if st.button("Analyse Skill Gap"):
        # Create user skill vector
        user_skill_vector = [1 if skill.lower() in skills_list else 0 for skill in skill_columns]

        # Encode role
        role_dict = {r: i for i, r in enumerate(roles_from_csv)}
        role_encoded = role_dict.get(role, 0)

        # Model input
        X = np.array([[experience, role_encoded] + user_skill_vector])

        # Predict skill gap
        gap_score = model.predict(X)[0]  # numeric prediction
        st.success(f"Hello {name}! Your estimated skill gap score is: **{gap_score:.2f}**")

        # Identify missing skills
        missing_skills = [skill for skill, has_skill in zip(skill_columns, user_skill_vector) if not has_skill]
        if missing_skills:
            st.info(f"Recommended skills to improve: {', '.join(missing_skills)}")
        else:
            st.info("You have all key skills!")

        # ----------------------------
        # Radar Chart Visualization
        # ----------------------------
        fig = go.Figure()

        # User skills
        fig.add_trace(go.Scatterpolar(
            r=user_skill_vector,
            theta=skill_columns,
            fill='toself',
            name='Your Skills'
        ))

        # Recommended skills (inverse of user skills)
        recommended_vector = [0 if val == 1 else 1 for val in user_skill_vector]
        fig.add_trace(go.Scatterpolar(
            r=recommended_vector,
            theta=skill_columns,
            fill='toself',
            name='Recommended Skills'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,1])
            ),
            showlegend=True,
            title="Skill Gap Radar Chart"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.header("Insights")
        st.markdown("""
        - Radar chart shows your current skills vs recommended skills.  
        - Focus on missing skills to close the gap.  
        - Update CSV and skill input over time to track progress.
        """)
