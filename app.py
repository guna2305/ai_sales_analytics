# app.py

import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# project modules
# -----------------------------
from src.forecasting import build_series
from src.forecast_prophet import prophet_fit_predict
from src.insights import compute_forecast_insights, insights_to_text
from src.rag_pipeline import RAGIndex, build_docs_from_outputs


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="AI Sales Analytics", layout="wide")
st.title("AI-Powered Sales Analytics System")

st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Go to", ["Data Upload", "Data Preview", "Dashboards", "Forecast Results"])


# -----------------------------
# Helpers
# -----------------------------
def load_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def guess_default(columns, keywords):
    """Return index of the first column that contains any keyword (case-insensitive)."""
    cols_lower = [c.lower() for c in columns]
    for kw in keywords:
        for i, col in enumerate(cols_lower):
            if kw in col:
                return i
    return 0


def preprocess_sales_data(df_raw: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    df = df_raw.copy()

    # Rename to standard schema
    rename_map = {
        col_map["date"]: "date",
        col_map["sales"]: "sales",
    }
    if col_map.get("category"):
        rename_map[col_map["category"]] = "category"
    if col_map.get("store"):
        rename_map[col_map["store"]] = "store"

    df = df.rename(columns=rename_map)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Clean sales (remove commas, currency if any)
    if df["sales"].dtype == "object":
        df["sales"] = (
            df["sales"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("₹", "", regex=False)
        )
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["date", "sales"]).copy()
    df = df.sort_values("date")

    # Optional columns cleanup
    if "category" in df.columns:
        df["category"] = df["category"].astype(str)
    if "store" in df.columns:
        df["store"] = df["store"].astype(str)

    return df


def compute_kpis(df_clean: pd.DataFrame) -> dict:
    return {
        "records": int(len(df_clean)),
        "date_min": df_clean["date"].min(),
        "date_max": df_clean["date"].max(),
        "total_sales": float(df_clean["sales"].sum()),
        "avg_sales_per_day": float(
            df_clean.set_index("date")["sales"].resample("D").sum().mean()
        ) if len(df_clean) else 0.0,
    }


# =========================================================
# Page: Data Upload
# =========================================================
if page == "Data Upload":
    st.subheader("Upload Sales Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_raw = load_csv(uploaded_file)
            st.session_state["df_raw"] = df_raw

            st.success("File uploaded successfully.")
            st.write("Dataset shape:", df_raw.shape)

            st.write("Columns detected:")
            cols = df_raw.columns.astype(str).tolist()
            st.code(", ".join(cols))

            st.write("Preview (first 10 rows):")
            st.dataframe(df_raw.head(10), use_container_width=True)

            # Column Mapping UI
            st.markdown("---")
            st.subheader("Column Mapping")

            date_default = guess_default(cols, ["order date", "date", "invoice date", "transaction date"])
            sales_default = guess_default(cols, ["total revenue", "sales", "revenue", "amount", "total", "price"])

            category_default = 0
            for kw in ["item type", "category", "product", "department"]:
                idx = guess_default(cols, [kw])
                if idx != 0 or (cols and kw in cols[0].lower()):
                    category_default = idx
                    break

            store_default = 0
            for kw in ["store", "branch", "outlet", "location"]:
                idx = guess_default(cols, [kw])
                if idx != 0 or (cols and kw in cols[0].lower()):
                    store_default = idx
                    break

            date_col = st.selectbox("Select Date Column (required)", cols, index=date_default, key="date_col")
            sales_col = st.selectbox("Select Sales Column (required)", cols, index=sales_default, key="sales_col")

            category_col = st.selectbox(
                "Select Category Column (optional)",
                ["None"] + cols,
                index=(category_default + 1) if category_default is not None else 0,
                key="category_col"
            )
            store_col = st.selectbox(
                "Select Store Column (optional)",
                ["None"] + cols,
                index=(store_default + 1) if store_default is not None else 0,
                key="store_col"
            )

            if st.button("Confirm Column Mapping"):
                st.session_state["col_map"] = {
                    "date": date_col,
                    "sales": sales_col,
                    "category": None if category_col == "None" else category_col,
                    "store": None if store_col == "None" else store_col,
                }
                st.success("Column mapping saved successfully.")

                df_clean = preprocess_sales_data(df_raw, st.session_state["col_map"])
                st.session_state["df_clean"] = df_clean

                # If new data, invalidate old RAG index
                if "rag_index" in st.session_state:
                    del st.session_state["rag_index"]

                if df_clean.empty:
                    st.error("After cleaning, no valid rows remain. Re-check your date/sales mapping.")
                else:
                    st.success("Data preprocessing successful.")
                    st.write("Cleaned dataset shape:", df_clean.shape)
                    st.write("Cleaned preview (first 10 rows):")
                    st.dataframe(df_clean.head(10), use_container_width=True)

        except Exception as e:
            st.error("Could not read the CSV file. Please upload a valid CSV.")
            st.write("Error details:", str(e))


# =========================================================
# Page: Data Preview
# =========================================================
elif page == "Data Preview":
    st.subheader("Data Preview")

    if "df_raw" not in st.session_state:
        st.info("Upload a dataset first from the Data Upload page.")
        st.stop()

    df_raw = st.session_state["df_raw"]
    st.write("Raw dataset shape:", df_raw.shape)
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("Current Column Mapping")
    if "col_map" not in st.session_state:
        st.warning("Column mapping not configured yet.")
        st.stop()
    st.json(st.session_state["col_map"])

    st.markdown("---")
    st.subheader("Cleaned Data (Standard Schema)")
    if "df_clean" not in st.session_state:
        st.info("Click 'Confirm Column Mapping' in Data Upload to generate cleaned data.")
        st.stop()

    df_clean = st.session_state["df_clean"]
    if df_clean.empty:
        st.error("Cleaned dataset is empty. Fix mapping in Data Upload.")
        st.stop()

    st.write("Cleaned dataset shape:", df_clean.shape)
    st.write("Columns:", df_clean.columns.tolist())
    st.dataframe(df_clean.head(20), use_container_width=True)

    st.markdown("---")
    st.subheader("Quick Summary")
    kpis = compute_kpis(df_clean)
    st.write("Date range:", kpis["date_min"], "to", kpis["date_max"])
    st.write("Total sales:", float(kpis["total_sales"]))


# =========================================================
# Page: Dashboards
# =========================================================
elif page == "Dashboards":
    st.subheader("Dashboards")

    if "df_clean" not in st.session_state or st.session_state["df_clean"].empty:
        st.info("Upload data and confirm column mapping first.")
        st.stop()

    df = st.session_state["df_clean"].copy()

    # Date filter
    min_date = df["date"].min()
    max_date = df["date"].max()
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start date", value=min_date.date())
    with c2:
        end_date = st.date_input("End date", value=max_date.date())

    df_f = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))].copy()

    kpis = compute_kpis(df_f)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sales", f"{kpis['total_sales']:.2f}")
    k2.metric("Records", f"{kpis['records']}")
    k3.metric("From", str(kpis["date_min"]))
    k4.metric("To", str(kpis["date_max"]))

    st.markdown("---")
    st.subheader("Sales Trend")

    gran = st.selectbox("Trend granularity", ["Daily", "Weekly", "Monthly"], index=2)
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[gran]

    series = build_series(df_f, freq=freq)
    trend_df = pd.DataFrame({"date": series.index, "sales": series.values})
    fig = px.line(trend_df, x="date", y="sales", title=f"Sales Trend ({gran})")
    st.plotly_chart(fig, use_container_width=True)

    if "category" in df_f.columns:
        st.markdown("---")
        st.subheader("Category-wise Sales")
        cat_df = df_f.groupby("category")["sales"].sum().sort_values(ascending=False).head(20).reset_index()
        fig_cat = px.bar(cat_df, x="category", y="sales", title="Top Categories by Sales")
        st.plotly_chart(fig_cat, use_container_width=True)

    if "store" in df_f.columns:
        st.markdown("---")
        st.subheader("Store-wise Sales")
        store_df = df_f.groupby("store")["sales"].sum().sort_values(ascending=False).head(20).reset_index()
        fig_store = px.bar(store_df, x="store", y="sales", title="Top Stores by Sales")
        st.plotly_chart(fig_store, use_container_width=True)


# =========================================================
# Page: Forecast Results (Prophet + Insights + Build RAG KB)
# =========================================================
elif page == "Forecast Results":
    st.subheader("Forecast Results (Meta Prophet)")

    if "df_clean" not in st.session_state or st.session_state["df_clean"].empty:
        st.info("Upload data and confirm column mapping first.")
        st.stop()

    df = st.session_state["df_clean"].copy().sort_values("date")

    # Controls
    freq_label = st.selectbox("Forecast granularity", ["Weekly", "Monthly"], index=1)
    freq = "W" if freq_label == "Weekly" else "M"
    horizon = st.slider("Forecast horizon (future periods)", 4, 52, 12)

    # Build time series used for Prophet
    series = build_series(df, freq=freq)

    if len(series) < 8:
        st.warning("Not enough time periods to forecast. Add more data or choose a coarser frequency.")
        st.stop()

    forecast_df, model_name = prophet_fit_predict(series, periods=horizon, freq=freq)
    st.caption(f"Model used: {model_name}")

    # Prophet returns history + future; split so future is FUTURE only
    last_hist_date = series.index.max()

    hist_part = forecast_df[forecast_df["ds"] <= last_hist_date].copy()
    fut_part = forecast_df[forecast_df["ds"] > last_hist_date].copy()

    # ---------------- Graph 1: Historical Actual vs Fitted
    hist_plot_df = pd.DataFrame({"date": series.index, "actual": series.values}).merge(
        hist_part[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "fitted"}),
        on="date",
        how="left"
    )

    fig_hist = px.line(
        hist_plot_df,
        x="date",
        y=["actual", "fitted"],
        title="Historical: Actual vs Fitted (sanity/accuracy check)"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ---------------- Graph 2: Future Forecast ONLY
    future_plot_df = fut_part[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"}).copy()
    fig_future = px.line(
        future_plot_df,
        x="date",
        y="yhat",
        title=f"Future Forecast (next {horizon} {freq_label.lower()} periods)"
    )
    st.plotly_chart(fig_future, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Future Forecast Table")
        st.dataframe(future_plot_df, use_container_width=True)

    with c2:
        st.subheader("Historical (tail)")
        st.dataframe(hist_plot_df.tail(24), use_container_width=True)

    # ---------------- Insight Summary
    future_series = pd.Series(future_plot_df["yhat"].values, index=future_plot_df["date"])
    ins = compute_forecast_insights(history=series, future_forecast=future_series)
    ins_text = insights_to_text(ins, freq_label=freq_label, horizon=horizon)

    st.markdown("---")
    st.subheader("Forecast Insight Summary")
    st.write(ins_text)

    # ---------------- Build / Rebuild Knowledge Base (Embeddings + FAISS)
    st.markdown("---")
    st.subheader("Build AI Knowledge Base (Embeddings + FAISS)")

    kpis = compute_kpis(df)
    kpis_text = (
        f"KPIs: Total sales={kpis['total_sales']:.2f}, Records={kpis['records']}, "
        f"Date range={kpis['date_min']} to {kpis['date_max']}, "
        f"Avg daily sales={kpis['avg_sales_per_day']:.2f}."
    )

    forecast_table_text = "Future forecast rows: " + "; ".join(
        [f"{d.date()}: {v:.2f}" for d, v in zip(future_plot_df["date"], future_plot_df["yhat"])]
    )

    if st.button("Build / Rebuild Knowledge Base"):
        docs = build_docs_from_outputs(kpis_text, ins_text, forecast_table_text)
        rag = RAGIndex()
        rag.build(docs)
        st.session_state["rag_index"] = rag
        st.success("Knowledge base built successfully (sentence-transformers + FAISS).")

    if "rag_index" in st.session_state:
        st.info("Knowledge base is ready. Next step: Chatbot page using local LLM + retrieval.")