import joblib
import numpy as np

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Example sensor inputs
temperature = 630
vibration = 0.015
pressure = 44
rpm = 12800
fuel_flow = 2100

values = np.array([
    temperature, vibration, pressure, rpm, fuel_flow
]).reshape(1, -1)

scaled_values = scaler.transform(values)
pred = model.predict(scaled_values)[0]
prob = model.predict_proba(scaled_values).max()

print(f"\nPrediction: {pred}")
print(f"Probability: {prob:.4f}")
