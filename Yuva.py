import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Skill Gap Analyser", layout="wide")
st.title("🔍 Skill Gap Analyser (Flexible CSV with Charts)")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📂 Preview of your data")
    st.dataframe(df.head())

    st.write("### 🔑 Map Your Columns")
    # Let user map columns
    current_role_col = st.selectbox("Select column for Current Role", df.columns)
    desired_role_col = st.selectbox("Select column for Desired Role", df.columns)
    skills_col = st.selectbox("Select column for Skills (comma separated)", df.columns)

    if st.button("Process Data"):
        try:
            results = []
            all_skills = []

            for _, row in df.iterrows():
                current_role = row[current_role_col]
                desired_role = row[desired_role_col]
                skills = [s.strip() for s in str(row[skills_col]).split(",")]
                all_skills.extend(skills)

                results.append({
                    "Current Role": current_role,
                    "Desired Role": desired_role,
                    "Skills": skills
                })

            processed_df = pd.DataFrame(results)
            st.success("✅ Data processed successfully!")
            st.write(processed_df.head())

            # --- Charts Section ---
            st.write("## 📊 Insights from Data")

            # 1. Desired Role distribution
            st.subheader("Distribution of Desired Roles")
            fig, ax = plt.subplots()
            processed_df["Desired Role"].value_counts().plot(kind="bar", ax=ax)
            ax.set_xlabel("Desired Roles")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # 2. Skills frequency
            st.subheader("Most Common Skills")
            skill_series = pd.Series(all_skills)
            fig2, ax2 = plt.subplots()
            skill_series.value_counts().head(10).plot(kind="barh", ax=ax2)
            ax2.set_xlabel("Count")
            ax2.set_ylabel("Skills")
            st.pyplot(fig2)

            # 3. Current vs Desired role pie chart
            st.subheader("Current vs Desired Roles (Pie)")
            role_counts = processed_df["Current Role"].value_counts().add(
                processed_df["Desired Role"].value_counts(), fill_value=0
            )
            fig3, ax3 = plt.subplots()
            role_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax3)
            ax3.set_ylabel("")
            st.pyplot(fig3)

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")