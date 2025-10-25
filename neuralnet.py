"""
Train an exercise classifier from pose landmark data.
-----------------------------------------------------
Input : exercise_data.csv
Output: exercise_model.pkl (saved model file)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---------- Load the dataset ----------
CSV_PATH = "exercise_data.csv"
data = pd.read_csv(CSV_PATH)

# Separate features (X) and labels (y)
X = data.drop(columns=["label"])
y = data["label"]

# ---------- Split data ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- Train model ----------
print("🧠 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,      # number of trees
    max_depth=15,          # prevent overfitting
    random_state=42
)
model.fit(X_train, y_train)

# ---------- Evaluate ----------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc*100:.2f}%\n")
print(classification_report(y_test, y_pred))

# ---------- Save model ----------
joblib.dump(model, "exercise_model.pkl")
print("💾 Saved trained model as exercise_model.pkl")