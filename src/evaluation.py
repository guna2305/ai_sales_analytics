import numpy as np
import pandas as pd
from prophet import Prophet


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = _safe_mape(y_true, y_pred)
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }


def prophet_backtest(series: pd.Series, horizon: int, freq: str = "M") -> tuple[pd.DataFrame, dict]:
    """
    Backtest using the last `horizon` periods as holdout.
    Returns:
      comparison_df: columns = date, actual, predicted
      metrics: mae, rmse, mape
    """
    if len(series) <= horizon + 3:
        raise ValueError("Not enough periods for backtesting. Need more historical data.")

    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]

    train_df = pd.DataFrame({"ds": train.index, "y": train.values})

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=(freq == "W"),
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    model.fit(train_df)

    future = model.make_future_dataframe(periods=horizon, freq=freq, include_history=False)
    forecast = model.predict(future)

    comparison_df = pd.DataFrame({
        "date": test.index,
        "actual": test.values,
        "predicted": forecast["yhat"].values
    })

    metrics = compute_regression_metrics(
        comparison_df["actual"].values,
        comparison_df["predicted"].values
    )

    return comparison_df, metrics