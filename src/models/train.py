import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn


def load_data():
    X = pd.read_parquet("data/processed/features.parquet")
    y = pd.read_parquet("data/processed/target.parquet")["target"]
    return X, y


def train():

    # ===== DATA =====
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ===== PIPELINE =====
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # ===== MLFLOW =====
    mlflow.set_experiment("trading_pipeline")

    with mlflow.start_run():

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        # log params
        mlflow.log_param("model", "logistic_regression")

        # log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc)

        # log model
        mlflow.sklearn.log_model(pipe, "model")

        print("\n=== RESULTS ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"ROC-AUC: {roc:.4f}")


if __name__ == "__main__":
    train()