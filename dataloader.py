import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import io
import base64
import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import StandardScaler


def download_price_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Downloads adjusted close prices for the given tickers and handles both single/multi ticker formats.
    """
    print(f"Downloading data for tickers: {tickers} from {start} to {end}")
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)
    print(len(data))

    # Handle case where data is empty (no data fetched)
    if data.empty:
        raise ValueError(f"No data fetched for tickers: {tickers}. Check your ticker symbols or date range.")

    # Handle single vs multi-ticker
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" not in data.columns.levels[0]:
            raise KeyError("'Adj Close' not found in data. Available columns: {}".format(data.columns.levels[0]))
        data = data["Adj Close"]  
    else:
        data = data["Adj Close"]

    data = data.dropna()
    return data


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily log returns from price data.
    """
    log_returns = np.log(price_df / price_df.shift(1))
    log_returns = log_returns.dropna()
    return log_returns

def add_features(log_returns: pd.DataFrame, window_size: int = 30, include_seasonality: bool = True) -> pd.DataFrame:
    """
    Adds features: momentum, moving average, volatility for each asset.
    """
    momentum = log_returns.cumsum() # cumulative log returns
    moving_avg = log_returns.rolling(window=window_size).mean() # rolling avg
    volatility = log_returns.rolling(window=window_size).std() # rolling std dev

    # Normalize each feature group
    def zscore(df):
        return (df - df.mean()) / (df.std() + 1e-8)

    features = {
        "log_return": log_returns,
        "momentum": zscore(momentum),
        "moving_avg": zscore(moving_avg),
        "volatility": zscore(volatility)
    }

    if include_seasonality:
        quarterly_returns = log_returns.rolling(window=63).sum()
        yearly_returns = log_returns.rolling(window=252).sum()
        features["quarterly_return"] = zscore(quarterly_returns)
        features["yearly_return"] = zscore(yearly_returns)

    combined = pd.concat({feat: features[feat] for feat in features}, axis=1)
    return combined.dropna()


def load_data(tickers: List[str], start: str = "2025-01-01", end: str = "2025-03-31", window_size: int = 30) -> pd.DataFrame:
    """
    Full pipeline: download → compute log returns → return clean dataset.
    """
    print("Loading data...")
    prices = download_price_data(tickers, start, end)
    log_returns = compute_log_returns(prices)
    return add_features(log_returns, window_size)

def load_eval_data(tickers: List[str], start: str, end: str, window_size: int = 30) -> pd.DataFrame:
    """
    Use for evaluation (excludes quarterly/yearly returns to avoid long lookback).
    """
    print("Loading evaluation data...")
    prices = download_price_data(tickers, start, end)
    log_returns = compute_log_returns(prices)
    return add_features(log_returns, window_size, include_seasonality=False)



# Existing functions from earlier...
def plot_stock_performance(tickers: List[str], start: str, end: str) -> str:
    """
    Plot the stock performance (adjusted close prices) for given tickers over the specified date range.
    The plot is returned as a base64-encoded PNG image.
    """
    # Download stock data for the tickers
    print(f"Plotting stock performance for tickers: {tickers} from {start} to {end}")
    data = download_price_data(tickers, start, end)

    # Create a plot
    plt.figure(figsize=(10, 6))
    for ticker in tickers:
        plt.plot(data[ticker], label=ticker)

    # Add labels and title
    plt.title(f"Stock Performance of {' '.join(tickers)} from {start} to {end}")
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend(loc='upper left')

    # Save the plot to a BytesIO object and encode it as base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close()

    return img_base64


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "JPM", "UNH", "HD", "V"]
    start_date = "2020-01-01"
    end_date = "2025-01-01"

    log_returns = load_data(tickers, start=start_date, end=end_date)

    print("\nLog returns loaded successfully!")
    print(log_returns.head())
    print(f"\nDate Range: {log_returns.index.min().date()} → {log_returns.index.max().date()}")
    print(f"Shape: {log_returns.shape}")
    print(f"\nColumns (Tickers): {log_returns.columns.tolist()}")

    #print("\nFirst few rows of log returns:")
    print(log_returns.tail())
