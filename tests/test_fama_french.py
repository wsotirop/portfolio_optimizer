# tests/test_fama_french.py
import pandas as pd
from data import fetch_data

def test_get_fama_french_factors_monthly(monkeypatch):
    calls = {}
    def fake_datareader(name, source, start, end):
        calls["dataset"] = name
        df = pd.DataFrame(
            {"Mkt-RF":[1.0], "SMB":[2.0], "HML":[3.0], "RF":[0.5]},
            index=[202001]  # YYYYMM
        )
        return {0: df}
    monkeypatch.setattr(fetch_data.web, "DataReader", fake_datareader)

    df = fetch_data.get_fama_french_factors("2020-01-01","2020-12-31",freq="M")
    assert calls["dataset"] == "F-F_Research_Data_Factors"
    assert df.index[0] == pd.Timestamp("2020-01-01")
    assert df.loc[pd.Timestamp("2020-01-01"), "Mkt-RF"] == 0.01  # %→decimal

def test_get_fama_french_factors_daily(monkeypatch):
    calls = {}
    def fake_datareader(name, source, start, end):
        calls["dataset"] = name
        df = pd.DataFrame({"Mkt-RF":[1.0]}, index=[20200101])  # YYYYMMDD
        return {0: df}
    monkeypatch.setattr(fetch_data.web, "DataReader", fake_datareader)

    df = fetch_data.get_fama_french_factors("2020-01-01","2020-12-31",freq="daily")
    assert calls["dataset"] == "F-F_Research_Data_Factors_daily"
    assert df.index[0] == pd.Timestamp("2020-01-01")
