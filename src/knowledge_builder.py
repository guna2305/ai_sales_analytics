import pandas as pd


def make_schema_doc(df_clean: pd.DataFrame) -> str:
    cols = ", ".join(df_clean.columns.tolist())
    return f"Dataset schema contains these columns: {cols}. Required business fields are date and sales. Optional analytical fields include category and store."


def make_kpi_doc(kpis: dict) -> str:
    return (
        f"Business KPIs summary: Total sales are {kpis['total_sales']:.2f}. "
        f"Total records are {kpis['records']}. "
        f"Date range is from {kpis['date_min']} to {kpis['date_max']}. "
        f"Average daily sales are {kpis['avg_sales_per_day']:.2f}."
    )


def make_recent_trend_doc(series: pd.Series) -> str:
    tail = series.tail(6)
    values = "; ".join([f"{idx.date()}: {float(val):.2f}" for idx, val in tail.items()])
    return f"Recent historical sales trend across latest periods: {values}."


def make_top_entities_docs(df_clean: pd.DataFrame) -> list[str]:
    docs = []

    if "category" in df_clean.columns:
        top_cat = (
            df_clean.groupby("category")["sales"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        bottom_cat = (
            df_clean.groupby("category")["sales"]
            .sum()
            .sort_values(ascending=True)
            .head(5)
        )
        docs.append(
            "Top categories by sales: " +
            "; ".join([f"{k}: {v:.2f}" for k, v in top_cat.items()])
        )
        docs.append(
            "Bottom categories by sales: " +
            "; ".join([f"{k}: {v:.2f}" for k, v in bottom_cat.items()])
        )

    if "store" in df_clean.columns:
        top_store = (
            df_clean.groupby("store")["sales"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        bottom_store = (
            df_clean.groupby("store")["sales"]
            .sum()
            .sort_values(ascending=True)
            .head(5)
        )
        docs.append(
            "Top stores by sales: " +
            "; ".join([f"{k}: {v:.2f}" for k, v in top_store.items()])
        )
        docs.append(
            "Bottom stores by sales: " +
            "; ".join([f"{k}: {v:.2f}" for k, v in bottom_store.items()])
        )

    return docs


def make_forecast_doc(future_df: pd.DataFrame) -> str:
    pairs = "; ".join([f"{d.date()}: {float(v):.2f}" for d, v in zip(future_df["date"], future_df["yhat"])])
    return f"Forecasted sales for future periods are: {pairs}."


def make_backtest_doc(metrics: dict) -> str:
    mape_txt = f"{metrics['mape']:.2f}%" if pd.notna(metrics["mape"]) else "not available"
    return (
        f"Forecast model backtest metrics: "
        f"MAE is {metrics['mae']:.2f}, "
        f"RMSE is {metrics['rmse']:.2f}, "
        f"MAPE is {mape_txt}."
    )


def make_anomaly_doc(series: pd.Series) -> str:
    pct = series.pct_change().dropna()
    if pct.empty:
        return "No anomaly summary available."

    max_rise_idx = pct.idxmax()
    max_drop_idx = pct.idxmin()

    return (
        f"Strongest positive change happened around {max_rise_idx.date()} with change {pct.max() * 100:.2f}%. "
        f"Strongest negative change happened around {max_drop_idx.date()} with change {pct.min() * 100:.2f}%."
    )