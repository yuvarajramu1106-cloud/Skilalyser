import streamlit as st
import pandas as pd

st.set_page_config(page_title="Skill Gap Analyser", layout="wide")
st.title("🔍 Universal Skill Gap Analyser")

# ---------------------------
# Step 1: Upload CSV
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Preview of Dataset")
    st.dataframe(df.head())

    # ---------------------------
    # Step 2: Map Columns
    # ---------------------------
    st.subheader("🗂️ Map Your Columns")
    role_col = st.selectbox("Select column for Current Role", df.columns)
    desired_col = st.selectbox("Select column for Desired Role", df.columns)
    skills_col = st.selectbox("Select column for Skills (comma separated)", df.columns)

    # ---------------------------
    # Step 3: Charts
    # ---------------------------
    st.subheader("📈 Insights from Dataset")

    if desired_col:
        st.write("**Distribution of Desired Roles**")
        st.bar_chart(df[desired_col].value_counts())

    if skills_col:
        st.write("**Top Skills in Dataset**")
        all_skills = []
        for s in df[skills_col].dropna():
            all_skills.extend([x.strip() for x in str(s).split(",")])
        skill_series = pd.Series(all_skills)
        st.bar_chart(skill_series.value_counts().head(15))

    # ---------------------------
    # Step 4: User Input Form
    # ---------------------------
    st.subheader("📝 Enter Your Details")

    with st.form("user_input_form"):
        name = st.text_input("Your Name")
        skills_input = st.text_area("Your Current Skills (comma separated)")
        experience = st.text_input("Your Last Work Experience (e.g., Intern, Student, Project, Junior Developer)")

        submitted = st.form_submit_button("🔮 Analyse My Skill Gap")

    # ---------------------------
    # Step 5: Skill Gap Analysis
    # ---------------------------
    if submitted and name and skills_input:
        user_skills = [s.strip() for s in skills_input.split(",")]

        role_skills_map = {}

        for _, row in df.iterrows():
            desired_role = row[desired_col]
            role_skills = [s.strip() for s in str(row[skills_col]).split(",")]
            if desired_role not in role_skills_map:
                role_skills_map[desired_role] = []
            role_skills_map[desired_role].extend(role_skills)

        # Average required skills for each role
        for role in role_skills_map:
            role_skills_map[role] = list(set(role_skills_map[role]))

        # Compare user skills with each role
        best_match = None
        best_score = -1
        missing_skills = []

        for role, req_skills in role_skills_map.items():
            overlap = len(set(user_skills) & set(req_skills))
            score = overlap / len(req_skills) if req_skills else 0
            if score > best_score:
                best_score = score
                best_match = role
                missing_skills = list(set(req_skills) - set(user_skills))

        # ---------------------------
        # Step 6: Show Results
        # ---------------------------
        st.success(f"### ✅ Skill Gap Analysis for {name}")
        st.write(f"**Last Work Experience:** {experience}")
        st.write(f"**Suggested Desired Role:** {best_match}")
        st.write(f"**Your Skills:** {', '.join(user_skills)}")
        st.write(f"**Missing Skills:** {', '.join(missing_skills) if missing_skills else 'None 🎉'}")
        st.write(f"**Success Percentage:** {round(best_score*100, 2)} %")
