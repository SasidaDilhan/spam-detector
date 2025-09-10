import joblib

# -----------------------------
# Load trained pipeline
# -----------------------------
model = joblib.load("models/spam_detector_pipeline.joblib")

# -----------------------------
# Test messages
# -----------------------------
test_messages = [
    "Congratulations! You won a free iPhone",
    "Hi John, I will send the report tomorrow.",
    "You have been selected for a $1000 gift card",
    "Meeting at 10 AM, please confirm",
    "Claim your free vacation now!"
]

# -----------------------------
# Custom threshold (optional)
# -----------------------------
threshold = 0.35  # Lower threshold catches more spam

# -----------------------------
# Predict using probabilities
# -----------------------------
probs = model.predict_proba(test_messages)
predictions = ['spam' if p[1] > threshold else 'ham' for p in probs]

# -----------------------------
# Display results
# -----------------------------
for msg, pred, prob in zip(test_messages, predictions, probs):
    print(f"Message: {msg}")
    print(f"Predicted: {pred} (spam prob: {prob[1]:.2f})")
    print("-" * 50)
