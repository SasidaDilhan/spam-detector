# spam_detector_starter.py
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import re
import string

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = text.strip()
    return text

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    # remove URLs, emails, numbers, and non-alphanum (keep spaces)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # remove stopwords (optional)
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)

def load_data(path):
    df = pd.read_csv(path, encoding="latin-1")
    
    # Rename columns if they exist
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})
    elif "label" not in df.columns or "text" not in df.columns:
        raise ValueError("No text column found in CSV. Expected columns: 'label' and 'text'.")
    
    # Keep only necessary columns
    df = df[["label", "text"]]
    
    return df


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]   # probability for positive class label
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, pos_label='spam'))
    print("Recall:", recall_score(y_test, y_pred, pos_label='spam'))
    print("F1:", f1_score(y_test, y_pred, pos_label='spam'))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=['spam','ham'])
    print("Confusion Matrix:\n", cm)
    if y_proba is not None:
        try:
            y_test_bin = (y_test == 'spam').astype(int)
            auc = roc_auc_score(y_test_bin, y_proba)
            print("ROC AUC:", auc)
            fpr, tpr, _ = roc_curve(y_test_bin, y_proba)
            plt.figure()
            plt.plot(fpr, tpr)
            plt.title('ROC curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.grid(True)
            plt.show()
        except Exception as e:
            print("ROC AUC error:", e)

def main():
    # 1) Load data
    df = load_data('data/spam.csv')
    print("Loaded:", df.shape)
    print(df['label'].value_counts())

# ✅ Create cleaned column
    df['text_clean'] = df['text'].apply(clean_text)

    

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    # 3) Pipeline with TF-IDF + Multinomial NB
    pipeline_nb = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=2)),
        ('clf', MultinomialNB())
    ])

    pipeline_nb.fit(X_train, y_train)
    print("\n--- MultinomialNB Evaluation ---")
    evaluate_model(pipeline_nb, X_test, y_test)

    # 4) Try Logistic Regression (often better)
    pipeline_lr = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=2)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipeline_lr.fit(X_train, y_train)
    print("\n--- Logistic Regression Evaluation ---")
    evaluate_model(pipeline_lr, X_test, y_test)

    # 5) Quick gridsearch to tune LR C
    param_grid = {
        'tfidf__ngram_range': [(1,1), (1,2)],
        'clf__C': [0.1, 1.0, 10.0]
    }
    gs = GridSearchCV(pipeline_lr, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    gs.fit(X_train, y_train)
    print("\nBest params:", gs.best_params_)
    print("Best CV score:", gs.best_score_)
    best_model = gs.best_estimator_
    print("\n--- Best Model Evaluation ---")
    evaluate_model(best_model, X_test, y_test)

    # 6) Save the best pipeline
    joblib.dump(best_model, 'spam_detector_pipeline.joblib')
    print("Saved model to spam_detector_pipeline.joblib")

if __name__ == "__main__":
    main()
