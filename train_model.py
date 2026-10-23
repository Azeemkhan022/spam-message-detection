
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    f1_score,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "SMSSpamCollection"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "spam_classifier.pkl"

# Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
)

# Clean data
df = df.dropna(subset=["label", "message"])
df["label"] = df["label"].str.strip().str.lower()
df["message"] = df["message"].astype(str).str.strip()
df = df[df["label"].isin(["ham", "spam"]) & df["message"].ne("")]
df = df.drop_duplicates(subset=["label", "message"])

print("Dataset size:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())

X = df["message"]
y = df["label"]

# Keep a final test set separate from model selection
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Measure how well each model detects spam during cross-validation
spam_f1 = make_scorer(f1_score, pos_label="spam", zero_division=0)

models = {
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
        )),
        ("classifier", MultinomialNB()),
    ]),
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ]),
}

print("\nComparing models using 5-fold cross-validation...")

best_name = None
best_model = None
best_score = -1

for name, candidate in models.items():
    scores = cross_val_score(
        candidate,
        X_train,
        y_train,
        cv=5,
        scoring=spam_f1,
        n_jobs=-1,
    )

    mean_score = scores.mean()
    print(f"{name} spam F1-score: {mean_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_name = name
        best_model = candidate

print(f"\nSelected model: {best_name}")
print("Cross-validation spam F1-score:", round(best_score, 4))

# Train the selected model on all training data
best_model.fit(X_train, y_train)

# Evaluate once on the held-out test set
predictions = best_model.predict(X_test)

print("\nTest accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=["ham", "spam"],
    zero_division=0,
))

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

# Save the complete trained pipeline
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, MODEL_PATH)

print("\nModel saved successfully:", MODEL_PATH)
print("Remember: model probabilities are estimates, not guarantees.")
