from pathlib import Path
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "news.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def build_pipeline(classifier):
    """Create word-level and character-level TF-IDF features plus a classifier."""
    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=100000,
            sublinear_tf=True,
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=100000,
            sublinear_tf=True,
        )),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", classifier),
    ])


def main():
    if not DATA_PATH.exists():
        raise SystemExit(
            "Dataset not found at data/news.csv. Create a labelled CSV with "
            "columns named text and label. See data/README.md."
        )

    df = pd.read_csv(DATA_PATH)
    if not {"text", "label"}.issubset(df.columns):
        raise SystemExit("data/news.csv must have columns named text and label.")

    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]

    if df["label"].nunique() < 2:
        raise SystemExit("The dataset must contain at least two distinct labels.")
    if df["label"].value_counts().min() < 2:
        raise SystemExit("Each label must have at least two examples for a stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "MultinomialNB": MultinomialNB(),
        "LinearSVC": LinearSVC(class_weight="balanced"),
    }

    rows = []
    fitted_models = {}

    for name, classifier in candidates.items():
        pipeline = build_pipeline(classifier)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_macro": float(precision_score(
                y_test, predictions, average="macro", zero_division=0
            )),
            "recall_macro": float(recall_score(
                y_test, predictions, average="macro", zero_division=0
            )),
            "f1_macro": float(f1_score(
                y_test, predictions, average="macro", zero_division=0
            )),
        })
        fitted_models[name] = pipeline
        print(f"\n--- {name} ---")
        print(classification_report(y_test, predictions, zero_division=0))

    comparison = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    comparison.to_csv(ROOT / "model_comparison.csv", index=False)

    best_name = str(comparison.iloc[0]["model"])
    best_model = fitted_models[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    # Compatibility copy for the original app, if it expects this filename.
    joblib.dump(best_model, ROOT / "fake_news_model.pkl")

    best_predictions = best_model.predict(X_test)
    class_names = sorted(df["label"].unique().tolist())
    metrics = {
        "dataset_name": DATA_PATH.name,
        "text_column": "text",
        "label_column": "label",
        "rows_after_cleaning": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "best_model": best_name,
        "accuracy": float(accuracy_score(y_test, best_predictions)),
        "precision_macro": float(precision_score(
            y_test, best_predictions, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            y_test, best_predictions, average="macro", zero_division=0
        )),
        "f1_macro": float(f1_score(
            y_test, best_predictions, average="macro", zero_division=0
        )),
        "class_names": class_names,
        "confusion_matrix": confusion_matrix(
            y_test, best_predictions, labels=class_names
        ).tolist(),
        "random_state": 42,
        "test_size": 0.2,
        "note": (
            "These results describe this dataset and split only; they do not "
            "prove real-world truth detection."
        ),
    }
    (ROOT / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print(f"\nBest model by macro F1: {best_name}")
    print("Saved models/best_model.pkl, fake_news_model.pkl,")
    print("model_metrics.json, and model_comparison.csv")


if __name__ == "__main__":
    main()
