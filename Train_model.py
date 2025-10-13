import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("skill_gap_dataset.csv")

# Encode Stream
le_stream = LabelEncoder()
df['Stream_enc'] = le_stream.fit_transform(df['Stream'])

# Encode Skills
df['Skills_list'] = df['Skills'].apply(lambda x: x.split(", "))
mlb = MultiLabelBinarizer()
skills_encoded = mlb.fit_transform(df['Skills_list'])
skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_)

# Combine features
X = pd.concat([df[['GPA','Projects','Internships','Stream_enc']], skills_df], axis=1)
y = df['Skill_Gap']

# Train-test split: 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
import os
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/skill_gap_model.pkl", "wb"))
pickle.dump(le_stream, open("models/le_stream.pkl", "wb"))
pickle.dump(mlb, open("models/mlb.pkl", "wb"))

print("Model and encoders saved in 'models/' folder")
