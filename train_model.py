import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("data/aircraft_engine_data.csv")

# Assume last column is target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Scale features
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Train model
# ----------------------------
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train_scaled, y_train)

# ----------------------------
# Evaluation
# ----------------------------
pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, pred)

print("\nModel Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))

# ----------------------------
# Save artifacts
# ----------------------------
Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel & scaler saved successfully!")
