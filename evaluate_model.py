import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re, string

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Rename columns if needed
if "v1" in df.columns and "v2" in df.columns:
    df = df.rename(columns={"v1": "label", "v2": "text"})

# Keep only necessary columns
df = df[["label", "text"]]

# -----------------------------
# Text cleaning function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

df['text_clean'] = df['text'].apply(clean_text)

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# -----------------------------
# Load trained pipeline
# -----------------------------
model = joblib.load("models/spam_detector_pipeline.joblib")

# -----------------------------
# Make predictions on test set
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluate metrics
# -----------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label='spam')
rec = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')
cm = confusion_matrix(y_test, y_pred, labels=['ham','spam'])

print("----- Spam Detector Evaluation -----")
print("Accuracy       :", acc)
print("Precision (spam):", prec)
print("Recall (spam)   :", rec)
print("F1-score (spam) :", f1)
print("Confusion Matrix:\n", cm)
