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
    """Fetch the Fama–French research factors (daily or monthly).

    Parameters
    ----------
    start : str | datetime
    end   : str | datetime
    freq  : {"D","daily","M","monthly"}
    """
    freq_key = str(freq).lower()
    if freq_key in {"d", "daily"}:
        dataset = "F-F_Research_Data_Factors_daily"
        date_fmt = "%Y%m%d"
    else:
        dataset = "F-F_Research_Data_Factors"
        date_fmt = "%Y%m"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        df = web.DataReader(dataset, "famafrench", start, end)[0]

    # Convert integer-like index to Timestamp
    df.index = pd.to_datetime(df.index.astype(str), format=date_fmt, errors="coerce")

    # Provided in percent; convert to decimals
    return df / 100.0

