import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn

import joblib

def load_data():
    X = pd.read_parquet("data/processed/features.parquet")
    y = pd.read_parquet("data/processed/target.parquet")["target"]
    return X, y


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


def train():

    X, y = load_data()

    # time series → no shuffle
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    mlflow.set_experiment("trading_pipeline")

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "random_forest": Pipeline([
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }

    # results = []
    best_score = -1
    best_model = None
    best_model_name = None

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)

            metrics = evaluate(model, X_test, y_test)

            # log params
            mlflow.log_param("model", name)

            # log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # log model
            mlflow.sklearn.log_model(model, name="model")

            print(f"\n=== {name.upper()} ===")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

            # results.append((name, metrics["roc_auc"]))
            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model = model
                best_model_name = name

    # seleccionar mejor modelo
    # best_model = max(results, key=lambda x: x[1])
    # print("\nBEST MODEL:", best_model)
    print(f"\nBEST MODEL: {best_model_name}")
    print(f"BEST ROC AUC: {best_score:.4f}")

    joblib.dump(best_model, "models/best_model.pkl")

    print("Model saved to models/best_model.pkl")


if __name__ == "__main__":
    train()