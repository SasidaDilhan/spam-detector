from flask import Flask, request, jsonify
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("models/spam_detector_pipeline.joblib")

# -----------------------------
# Create Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Default spam threshold
# -----------------------------
THRESHOLD = 0.80

# -----------------------------
# Health check endpoint
# -----------------------------
@app.route("/")
def index():
    return "Spam Detector API is running!"

# -----------------------------
# Predict endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    messages = data.get("messages", [])

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    # Predict probabilities
    probs = model.predict_proba(messages)
    predictions = ['spam' if p[1] > THRESHOLD else 'ham' for p in probs]

    # Return results
    results = []
    for msg, pred, prob in zip(messages, predictions, probs):
        results.append({
            "message": msg,
            "prediction": pred,
            "spam_prob": round(prob[1], 2)
        })

    return jsonify(results)

# -----------------------------
# Predict single message endpoint
# -----------------------------
@app.route("/predict_single", methods=["POST"])
def predict_single():
    data = request.get_json()
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Predict probability
    prob = model.predict_proba([message])[0]
    prediction = 'spam' if prob[1] > THRESHOLD else 'ham'

    return jsonify({
        "message": message,
        "prediction": prediction,
        "spam_prob": round(prob[1], 2)
    })


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
