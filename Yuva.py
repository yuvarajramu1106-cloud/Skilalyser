import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Skill Gap Analyser", layout="wide")
st.title("🔍 Skill Gap Analyser with Prediction & Success Rate")

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

    if st.button("Process & Train Model"):
        try:
            results = []
            all_skills = []

            for _, row in df.iterrows():
                current_role = str(row[current_role_col])
                desired_role = str(row[desired_role_col])
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
            st.bar_chart(processed_df["Desired Role"].value_counts())

            # 2. Skills frequency
            st.subheader("Most Common Skills")
            skill_series = pd.Series(all_skills)
            st.bar_chart(skill_series.value_counts().head(10))

            # --- Prediction Section ---
            st.write("## 🔮 Predict Desired Role")

            # Encode Current Role + Desired Role
            le_role = LabelEncoder()
            le_desired = LabelEncoder()

            processed_df["CurrentRole_enc"] = le_role.fit_transform(processed_df["Current Role"])
            processed_df["DesiredRole_enc"] = le_desired.fit_transform(processed_df["Desired Role"])

            # Encode Skills (multi-hot encoding)
            mlb = MultiLabelBinarizer()
            skill_matrix = mlb.fit_transform(processed_df["Skills"])
            skill_df = pd.DataFrame(skill_matrix, columns=mlb.classes_)

            X = pd.concat([processed_df[["CurrentRole_enc"]], skill_df], axis=1)
            y = processed_df["DesiredRole_enc"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Model performance
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"✅ Model trained with accuracy: **{acc*100:.2f}%**")

            # Prediction UI
            st.subheader("Make a Prediction")
            input_role = st.selectbox("Select Current Role", processed_df["Current Role"].unique())
            input_skills = st.multiselect("Select Skills", mlb.classes_)

            if st.button("Predict Desired Role"):
                role_enc = le_role.transform([input_role])[0]
                skill_vector = np.zeros(len(mlb.classes_))
                for s in input_skills:
                    if s in mlb.classes_:
                        skill_vector[list(mlb.classes_).index(s)] = 1

                input_data = np.concatenate(([role_enc], skill_vector)).reshape(1, -1)

                # Prediction with probability
                pred_enc = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]

                predicted_role = le_desired.inverse_transform([pred_enc])[0]
                success_rate = np.max(proba) * 100

                st.success(f"🔮 Predicted Desired Role: **{predicted_role}**")
                st.info(f"✅ Success Percentage: **{success_rate:.2f}%**")

                # Show top 3 possible roles with probabilities
                st.subheader("📌 Top 3 Role Predictions")
                top_indices = np.argsort(proba)[::-1][:3]
                for i in top_indices:
                    role = le_desired.inverse_transform([i])[0]
                    st.write(f"- {role}: {proba[i]*100:.2f}%")

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
