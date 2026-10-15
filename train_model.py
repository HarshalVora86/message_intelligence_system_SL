"""
Train and export the complete preprocessing + classification pipelines.

Run:
    python train_model.py

Output:
    model/spam_svc_pipeline.pkl   (Linear SVC pipeline)
    model/metrics.json
"""
import json
import pickle
from pathlib import Path

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_FILE = "Message_Intelligence_Dataset_5200.xlsx"
DATA_CANDIDATES = [
    Path("dataset") / DATA_FILE,
    Path("Dataset") / DATA_FILE,
    Path(DATA_FILE),
]
TARGET = "spam_label"
DROP_COLUMNS = ["message_id", "message_text", "timestamp"]

FEATURES = [
    "message_length",
    "word_count",
    "num_urls",
    "num_digits",
    "num_special_chars",
    "spam_keyword_score",
    "legit_keyword_score",
    "sender_activity_score",
    "sender_account_age_days",
    "messages_sent_last_24h",
    "hour_of_day",
    "day_of_week",
]


def load_data():
    for path in DATA_CANDIDATES:
        if path.exists():
            return pd.read_excel(path)
    raise FileNotFoundError(
        f"{DATA_FILE} not found. Put it in the 'dataset' folder."
    )


def build_pipeline():
    # imputer and scaler are fitted on the training data only
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="linear")),
        ]
    )


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred)),
        "F1": float(f1_score(y_test, y_pred)),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def main():
    df = load_data()
    df = df.drop(columns=DROP_COLUMNS)

    X = df[FEATURES]
    y = df[TARGET]

    # same split as the notebook (test_size=0.2, random_state=50)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=50
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)
    metrics.update(
        {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "features": FEATURES,
            "target": TARGET,
        }
    )

    Path("model").mkdir(exist_ok=True)

    with open("model/spam_svc_pipeline.pkl", "wb") as file:
        pickle.dump(pipeline, file)

    with open("model/metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nModel trained successfully.")
    print("Pipeline: model/spam_svc_pipeline.pkl")
    print("\nEvaluation (test set):")
    for key, value in metrics.items():
        if key != "features":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
