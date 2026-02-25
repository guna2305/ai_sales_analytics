import pandas as pd

def build_series(df_clean: pd.DataFrame, freq: str) -> pd.Series:
    """
    Convert df_clean -> time series indexed by date.
    freq: "W" weekly, "M" monthly
    """
    s = (
        df_clean.set_index("date")["sales"]
        .resample(freq)
        .sum()
        .sort_index()
    )
    # Ensure a continuous index (fills missing periods with 0)
    s = s.asfreq(freq, fill_value=0)
    return s


def seasonal_naive_forecast(series: pd.Series, steps: int, season_len: int) -> pd.Series:
    """
    Fallback forecast if statsmodels not available.
    Repeats the last season_len values into the future.
    """
    if len(series) < season_len:
        # if too short, repeat last value
        last_val = float(series.iloc[-1])
        future_vals = [last_val] * steps
    else:
        pattern = series.iloc[-season_len:].tolist()
        future_vals = [pattern[i % season_len] for i in range(steps)]
    # Build future index
    future_idx = pd.date_range(
        start=series.index[-1] + series.index.freq,
        periods=steps,
        freq=series.index.freq
    )
    return pd.Series(future_vals, index=future_idx, name="forecast")


def fit_and_forecast(series: pd.Series, steps: int, seasonal_periods: int | None = None):
    """
    Returns:
      fitted: pd.Series on historical dates
      forecast: pd.Series on FUTURE dates only
      model_name: str
    """
    # Try a proper forecasting model first
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        if seasonal_periods and len(series) >= seasonal_periods * 2:
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            ).fit(optimized=True)
            model_name = f"ExponentialSmoothing(seasonal={seasonal_periods})"
        else:
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            model_name = "ExponentialSmoothing(trend)"

        fitted = model.fittedvalues
        forecast = model.forecast(steps)
        # Ensure future index is correct
        forecast.index = pd.date_range(
            start=series.index[-1] + series.index.freq,
            periods=steps,
            freq=series.index.freq
        )
        forecast.name = "forecast"
        return fitted, forecast, model_name

    except Exception:
        # Fallback: seasonal naive (still produces FUTURE dates correctly)
        season_len = seasonal_periods or (12 if series.index.freqstr.startswith("M") else 52)
        fitted = series.shift(season_len)  # naive fitted for history
        forecast = seasonal_naive_forecast(series, steps=steps, season_len=season_len)
        return fitted, forecast, f"SeasonalNaive(season={season_len})"
