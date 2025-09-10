import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
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
# Create pipeline (TF-IDF + Naive Bayes)
# -----------------------------
pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=1.0)),
    ('clf', MultinomialNB())
])

# -----------------------------
# Train the model
# -----------------------------
pipeline_nb.fit(X_train, y_train)

# -----------------------------
# Save trained pipeline
# -----------------------------
joblib.dump(pipeline_nb, "models/spam_detector_pipeline.joblib")
print("Model trained and saved to models/spam_detector_pipeline.joblib")
