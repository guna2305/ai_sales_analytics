import pandas as pd
from prophet import Prophet


def build_prophet_df(series: pd.Series) -> pd.DataFrame:
    """series: index=datetime, values=sales"""
    dfp = pd.DataFrame({"ds": series.index, "y": series.values})
    dfp = dfp.dropna().sort_values("ds")
    return dfp


def prophet_fit_predict(series: pd.Series, periods: int, freq: str = "M") -> tuple[pd.DataFrame, str]:
    """
    Returns forecast_df (Prophet output) and model_name.
    freq: "W" or "M"
    """
    dfp = build_prophet_df(series)

    # Prophet is robust for seasonality. Enable yearly by default for monthly/weekly sales.
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=(freq == "W"),
        daily_seasonality=False,
        seasonality_mode="additive",
    )

    m.fit(dfp)

    future = m.make_future_dataframe(periods=periods, freq=freq, include_history=True)
    forecast = m.predict(future)  # includes history + future

    return forecast, "Prophet"