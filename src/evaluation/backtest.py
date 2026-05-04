import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn


def load_data():
    X = pd.read_parquet("data/processed/features.parquet")
    y = pd.read_parquet("data/processed/target.parquet")["target"]
    prices = pd.read_parquet("data/raw/market_data.parquet")["SPY"]

    return X, y, prices


def backtest():

    # ===== LOAD =====
    X, y, prices = load_data()

    split = int(len(X) * 0.8)

    X_test = X.iloc[split:]
    prices_test = prices.iloc[split:]

    # ===== LOAD MODEL (último run) =====
    model_uri = "runs:/latest/model"

    # fallback si falla latest
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("trading_pipeline")
    runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"

    model = mlflow.sklearn.load_model(model_uri)

    # ===== PREDICTIONS =====
    preds = model.predict(X_test)

    # returns
    returns = prices_test.pct_change().shift(-1)

    # convertir preds a serie con índice correcto
    preds = pd.Series(preds, index=X_test.index)

    # alinear
    returns = returns.loc[preds.index]

    # estrategia
    strategy_returns = returns * preds

    # ===== CUMULATIVE =====
    cum_strategy = (1 + strategy_returns.fillna(0)).cumprod()
    cum_buy_hold = (1 + returns.fillna(0)).cumprod()

    # ===== PLOT =====
    plt.figure()
    plt.plot(cum_strategy, label="Strategy")
    plt.plot(cum_buy_hold, label="Buy & Hold")
    plt.legend()
    plt.title("Strategy vs Buy & Hold")
    plt.show()

    print("\nFinal Strategy Return:", cum_strategy.iloc[-1])
    print("Final Buy & Hold Return:", cum_buy_hold.iloc[-1])


if __name__ == "__main__":
    backtest()