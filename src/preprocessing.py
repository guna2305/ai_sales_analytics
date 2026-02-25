import pandas as pd


def preprocess_sales_data(df_raw: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    df = df_raw.copy()

    rename_map = {
        col_map["date"]: "date",
        col_map["sales"]: "sales",
    }

    if col_map.get("category"):
        rename_map[col_map["category"]] = "category"
    if col_map.get("store"):
        rename_map[col_map["store"]] = "store"

    df = df.rename(columns=rename_map)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["sales"].dtype == "object":
        df["sales"] = (
            df["sales"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("₹", "", regex=False)
        )

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    df = df.dropna(subset=["date", "sales"])
    df = df.sort_values("date")

    return df
