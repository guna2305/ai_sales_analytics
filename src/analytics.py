import pandas as pd


def compute_kpis(df: pd.DataFrame) -> dict:
    total_sales = float(df["sales"].sum())
    records = len(df)
    days = max(1, (df["date"].max() - df["date"].min()).days + 1)
    avg_daily_sales = total_sales / days

    return {
        "total_sales": total_sales,
        "records": records,
        "avg_daily_sales": avg_daily_sales,
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
    }


def aggregate_sales(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    return (
        df.set_index("date")["sales"]
        .resample(freq)
        .sum()
        .reset_index()
        .rename(columns={"sales": "total_sales"})
    )
