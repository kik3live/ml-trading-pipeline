import pandas as pd
from pathlib import Path

def build_features():

    df = pd.read_parquet("data/raw/market_data.parquet")

    returns = df.pct_change()

    features = pd.DataFrame(index=df.index)

    # features para todos los activos
    for col in df.columns:
        features[f"{col}_ret"] = returns[col]
        features[f"{col}_lag1"] = returns[col].shift(1)
        features[f"{col}_lag3"] = returns[col].shift(3)
        features[f"{col}_ma5"] = df[col].rolling(5).mean()

    # 🎯 TARGET: solo SPY (más estable)
    target = (returns["SPY"].shift(-1) > 0).astype(int)

    # limpieza
    features = features.dropna()
    target = target.loc[features.index]

    # asegurar carpetas
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # guardar
    features_path = "data/processed/features.parquet"
    target_path = "data/processed/target.parquet"

    features.to_parquet(features_path)
    target.to_frame(name="target").to_parquet(target_path)

    print("Features built:", features.shape)
    print("Saved features to:", features_path)
    print("Saved target to:", target_path)


if __name__ == "__main__":
    build_features()