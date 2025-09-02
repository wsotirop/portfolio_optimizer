# data/fetch_data.py


import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import warnings

def get_price_data(tickers, start_date, end_date):
    """
    Fetches unadjusted OHLC + 'Adj Close' by disabling auto_adjust.
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )
    adj = data["Adj Close"]
    # drop columns where all prices are NaN and rows with any NaN
    adj = adj.dropna(how="all", axis=1).dropna()
    return adj


def get_fama_french_factors(start, end, freq="D"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        df = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start, end)[0]
    df.index = pd.to_datetime(df.index, format="%Y%m%d", errors="coerce")
    df = df / 100.0
    return df

