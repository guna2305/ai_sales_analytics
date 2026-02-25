import pandas as pd


def compute_forecast_insights(history: pd.Series, future_forecast: pd.Series) -> dict:
    """
    history: actual series (indexed by date)
    future_forecast: forecasted values for FUTURE dates only
    """
    insights = {}

    # Growth %: compare last actual period vs average of future horizon (or first future period)
    last_actual = float(history.iloc[-1]) if len(history) else 0.0
    future_avg = float(future_forecast.mean()) if len(future_forecast) else 0.0

    if last_actual != 0:
        growth_pct = (future_avg - last_actual) / abs(last_actual) * 100.0
    else:
        growth_pct = None

    insights["last_actual"] = last_actual
    insights["future_avg"] = future_avg
    insights["growth_pct"] = growth_pct

    # Peak month (peak period)
    if len(future_forecast):
        peak_date = future_forecast.idxmax()
        peak_value = float(future_forecast.max())
        insights["peak_period"] = str(peak_date.date())
        insights["peak_value"] = peak_value
    else:
        insights["peak_period"] = None
        insights["peak_value"] = None

    # Trend direction: compare first vs last forecast value
    if len(future_forecast) >= 2:
        first = float(future_forecast.iloc[0])
        last = float(future_forecast.iloc[-1])
        if last > first:
            trend = "Upward"
        elif last < first:
            trend = "Downward"
        else:
            trend = "Flat"
        insights["trend_direction"] = trend
        insights["trend_change"] = last - first
    else:
        insights["trend_direction"] = None
        insights["trend_change"] = None

    # Volatility measure: coefficient of variation on forecast horizon
    # (std / mean) – simple, interpretable
    if len(future_forecast) and float(future_forecast.mean()) != 0:
        volatility = float(future_forecast.std() / abs(future_forecast.mean()))
    else:
        volatility = None
    insights["volatility_cv"] = volatility

    return insights


def insights_to_text(insights: dict, freq_label: str, horizon: int) -> str:
    parts = []
    parts.append(f"Forecast horizon: next {horizon} {freq_label.lower()} periods.")

    gp = insights.get("growth_pct")
    if gp is not None:
        parts.append(f"Expected growth vs last actual: {gp:.2f}% (based on future average vs last period).")
    else:
        parts.append("Growth % could not be computed (last actual is 0 or missing).")

    if insights.get("peak_period") is not None:
        parts.append(f"Peak expected period: {insights['peak_period']} with forecast {insights['peak_value']:.2f}.")
    else:
        parts.append("Peak period not available.")

    if insights.get("trend_direction") is not None:
        parts.append(f"Trend direction across horizon: {insights['trend_direction']} (change {insights['trend_change']:.2f}).")
    else:
        parts.append("Trend direction not available.")

    v = insights.get("volatility_cv")
    if v is not None:
        parts.append(f"Forecast volatility (CV): {v:.3f} (higher means more variation).")
    else:
        parts.append("Volatility measure not available.")

    return " ".join(parts)