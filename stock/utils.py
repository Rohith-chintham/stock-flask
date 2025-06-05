import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def load_data(ticker, start, end):
    """
    Downloads stock data using yfinance and returns a single-column DataFrame with 'Close' prices.
    Handles both normal and multi-index cases.
    """
    try:
        # Download data
        data = yf.download(ticker, start=start, end=end)

        # Check for empty data
        if data.empty:
            return pd.DataFrame()

        # Handle multi-index (occurs when more than one ticker is passed)
        if isinstance(data.columns, pd.MultiIndex):
            if ('Close', ticker) in data.columns:
                close_data = data['Close'][ticker].to_frame(name='Close')
            else:
                raise KeyError(f"('Close', '{ticker}') not found in multi-index data.")
        else:
            if 'Close' in data.columns:
                close_data = data[['Close']]
            else:
                raise KeyError("'Close' not found in columns.")

        close_data.dropna(inplace=True)
        return close_data

    except Exception as e:
        print("Data load error:", e)
        return pd.DataFrame()


def plot_data(df):
    """
    Returns a matplotlib plot of the closing price.
    """
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Close'], label='Closing Price', color='blue')
    ax.set_title('Stock Closing Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)
    return fig
