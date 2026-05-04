import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = ["AAPL", "SPY", "NVDA"]

def download_data():
    df = yf.download(TICKERS, start="2015-01-01", end="2024-12-31")

    # 🔥 manejar multiindex correctamente
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    else:
        df = df

    df = df.dropna()

    Path("data/raw").mkdir(parents=True, exist_ok=True)

    path = "data/raw/market_data.parquet"
    df.to_parquet(path)

    print("Data downloaded:", df.shape)
    print("Saved to:", path)


if __name__ == "__main__":
    download_data()