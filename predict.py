import joblib
import numpy as np

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Example values
temperature = 620
pressure = 40.1
vibration = 0.011
fuel_flow = 1700
rpm = 12000

features = np.array([
    temperature, pressure, vibration, fuel_flow, rpm
]).reshape(1, -1)

features_scaled = scaler.transform(features)
pred = model.predict(features_scaled)[0]
prob = model.predict_proba(features_scaled).max()

print(f"\nPrediction: {pred}")
print(f"Probability: {prob:.4f}")
