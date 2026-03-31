import pandas as pd
from prophet import Prophet


def make_default_holidays(country: str = "US") -> pd.DataFrame:
    """
    Small built-in holiday calendar for retail-style forecasting.
    """
    current_year = pd.Timestamp.today().year
    years = [current_year - 1, current_year, current_year + 1, current_year + 2]

    rows = []
    for y in years:
        rows.extend([
            {"holiday": "new_year", "ds": pd.Timestamp(f"{y}-01-01"), "lower_window": 0, "upper_window": 1},
            {"holiday": "independence_day", "ds": pd.Timestamp(f"{y}-07-04"), "lower_window": 0, "upper_window": 1},
            {"holiday": "thanksgiving", "ds": pd.Timestamp(f"{y}-11-27"), "lower_window": 0, "upper_window": 1},
            {"holiday": "black_friday", "ds": pd.Timestamp(f"{y}-11-28"), "lower_window": 0, "upper_window": 2},
            {"holiday": "christmas", "ds": pd.Timestamp(f"{y}-12-25"), "lower_window": 0, "upper_window": 2},
        ])

    holidays_df = pd.DataFrame(rows)
    holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])
    return holidays_df


def prepare_prophet_training_df(
    df: pd.DataFrame,
    freq: str = "M",
    date_col: str = "date",
    sales_col: str = "sales",
    regressors: list | None = None,
) -> pd.DataFrame:
    """
    Convert raw sales data into Prophet-ready aggregated dataframe.
    Required output columns: ds, y
    Optional regressor columns are aggregated too.
    """
    regressors = regressors or []

    agg_map = {sales_col: "sum"}
    for reg in regressors:
        if reg == "promo_flag":
            agg_map[reg] = "max"
        else:
            agg_map[reg] = "mean"

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, sales_col]).copy()

    grouped = (
        temp.groupby(pd.Grouper(key=date_col, freq=freq))
        .agg(agg_map)
        .reset_index()
        .rename(columns={date_col: "ds", sales_col: "y"})
    )

    grouped = grouped.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    return grouped


def prophet_fit_predict(
    train_df: pd.DataFrame,
    periods: int,
    freq: str = "M",
    regressor_cols: list | None = None,
    holidays_df: pd.DataFrame | None = None,
):
    """
    Train Prophet using aggregated dataframe.

    train_df must contain:
    - ds
    - y
    - optional regressor columns
    """
    regressor_cols = regressor_cols or []

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=(freq in ["D", "W"]),
        daily_seasonality=False,
        seasonality_mode="additive",
        holidays=holidays_df,
    )

    for col in regressor_cols:
        model.add_regressor(col)

    fit_df = train_df.copy()
    fit_df["ds"] = pd.to_datetime(fit_df["ds"])
    model.fit(fit_df)

    future = model.make_future_dataframe(periods=periods, freq=freq, include_history=True)

    for col in regressor_cols:
        non_null = fit_df[col].dropna()
        last_val = non_null.iloc[-1] if len(non_null) > 0 else 0
        future[col] = last_val

        future = future.merge(
            fit_df[["ds", col]],
            on="ds",
            how="left",
            suffixes=("", "_train")
        )

        future[col] = future[f"{col}_train"].combine_first(future[col])
        future = future.drop(columns=[f"{col}_train"])

    forecast = model.predict(future)

    return forecast, model, "Prophet + Regressors + Holidays"