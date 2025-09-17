import requests

# API endpoint
url_single = "http://127.0.0.1:5000/predict_single"
url_multiple = "http://127.0.0.1:5000/predict"

# High-confidence test messages
messages = [
    "Congratulations! You won a free iPhone",
    "Hi John, I will send the report tomorrow.",
    "Claim your free vacation now!",
    "Meeting at 10 AM, please confirm",
    "You have been selected for a $1000 gift card"
]

# -----------------------------
# Test multiple messages
# -----------------------------
response = requests.post(url_multiple, json={"messages": messages})
results = response.json()

# Print results
print("----- Multi-Message Test -----")
for r in results:
    # Only consider spam if probability >= 0.8
    if r['spam_prob'] >= 0.8:
        label = 'spam'
    else:
        label = 'ham'

    print(f"Message: {r['message']}")
    print(f"Predicted: {label} (spam prob: {r['spam_prob']})")
    print("-" * 50)

# -----------------------------
# Test a single message
# -----------------------------
single_msg = {"message": "You have won a free iPhone!"}
response_single = requests.post(url_single, json=single_msg)
res = response_single.json()

label = 'spam' if res['spam_prob'] >= 0.8 else 'ham'
print("----- Single Message Test -----")
print(f"Message: {res['message']}")
print(f"Predicted: {label} (spam prob: {res['spam_prob']})")
print("-" * 50)


# single_msg = {"message": "You have won a free iPhone!"}
# response_single = requests.post(url_single, json=single_msg)
# res = response_single.json()

# label = 'spam' if res['spam_prob'] >= 0.8 else 'ham'
# print("----- Single Message Test -----")
# print(f"Message: {res['message']}")
# print(f"Predicted: {label} (spam prob: {res['spam_prob']})")
# print("-" * 50)
