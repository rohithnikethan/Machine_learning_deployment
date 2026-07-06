from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

try:
    model = joblib.load("logistic_regression_model.joblib")
    scaler = joblib.load("scaler.joblib")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


@app.route("/")
def home():
    return jsonify({
        "message": "Diabetes Prediction API is running."
    })


@app.route("/predict", methods=["POST"])
def predict():

    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded."}), 500

    data = request.get_json()

    if data is None:
        return jsonify({"error": "JSON data expected."}), 400

    if "features" not in data:
        return jsonify({
            "error": "Missing 'features' key in JSON."
        }), 400

    features = data["features"]

    if not isinstance(features, list):
        return jsonify({
            "error": "'features' must be a list."
        }), 400

    if len(features) != len(FEATURE_NAMES):
        return jsonify({
            "error": f"Expected {len(FEATURE_NAMES)} features, received {len(features)}."
        }), 400

    try:
        # Convert list to NumPy array with shape (1, 8)
        input_data = np.array([features], dtype=float)

        # Scale the input
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]

        return jsonify({
            "prediction": int(prediction),
            "probability_no_diabetes": round(float(probabilities[0]), 4),
            "probability_diabetes": round(float(probabilities[1]), 4)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
