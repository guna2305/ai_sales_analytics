import pandas as pd

def make_schema_doc(df_clean: pd.DataFrame) -> str:
    cols = ", ".join(df_clean.columns.tolist())
    return f"Dataset schema: columns={cols}. Standard fields include date and sales, optional category and store."

def make_top_entities_docs(df_clean: pd.DataFrame) -> list[str]:
    docs = []
    if "category" in df_clean.columns:
        top_cat = (
            df_clean.groupby("category")["sales"].sum()
            .sort_values(ascending=False).head(5)
        )
        docs.append("Top categories by sales: " + "; ".join([f"{k}: {v:.2f}" for k, v in top_cat.items()]))

    if "store" in df_clean.columns:
        top_store = (
            df_clean.groupby("store")["sales"].sum()
            .sort_values(ascending=False).head(5)
        )
        docs.append("Top stores by sales: " + "; ".join([f"{k}: {v:.2f}" for k, v in top_store.items()]))

    return docs

def make_recent_trend_doc(series: pd.Series) -> str:
    tail = series.tail(6)
    pairs = "; ".join([f"{idx.date()}: {float(val):.2f}" for idx, val in tail.items()])
    return f"Recent sales trend (last periods): {pairs}"

def make_forecast_table_doc(future_df: pd.DataFrame) -> str:
    pairs = "; ".join([f"{d.date()}: {float(v):.2f}" for d, v in zip(future_df["date"], future_df["yhat"])])
    return "Future forecast values: " + pairs