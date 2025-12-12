from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/")
def home():
    return {"message": "Aircraft Predictive Maintenance API Running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([
        data["temperature"],
        data["pressure"],
        data["vibration"],
        data["fuel_flow"],
        data["rpm"]
    ]).reshape(1, -1)

    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled).max()

    return jsonify({
        "prediction": str(pred),
        "probability": float(prob)
    })

if __name__ == "__main__":
    app.run(debug=True)
