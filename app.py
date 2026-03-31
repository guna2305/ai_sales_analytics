import streamlit as st
import pandas as pd
import plotly.express as px

from src.forecasting import build_series
from src.forecast_prophet import (
    prophet_fit_predict,
    prepare_prophet_training_df,
    make_default_holidays,
)
from src.insights import compute_forecast_insights, insights_to_text
from src.evaluation import prophet_backtest
from src.rag_answering import template_answer, build_grounded_prompt
from src.knowledge_builder import (
    make_schema_doc,
    make_kpi_doc,
    make_recent_trend_doc,
    make_top_entities_docs,
    make_forecast_doc,
    make_backtest_doc,
    make_anomaly_doc,
)

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="AI Sales Analytics System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# Session state
# =========================================================
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# =========================================================
# Global styling
# =========================================================
st.markdown("""
<style>
[data-testid="stSidebar"] {
    display: none;
}
[data-testid="collapsedControl"] {
    display: none;
}
header[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
.stApp {
    background:
        radial-gradient(circle at top left, rgba(37,99,235,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(6,182,212,0.14), transparent 26%),
        linear-gradient(135deg, #0b1220 0%, #111827 45%, #0f172a 100%);
}
.main .block-container {
    padding-top: 1.3rem;
    padding-bottom: 3rem;
    max-width: 1420px;
}
.main-title {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
    letter-spacing: -0.7px;
    color: #f8fafc;
}
.subtle-text {
    color: #cbd5e1;
    font-size: 1rem;
    margin-bottom: 1.35rem;
}
.section-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.028));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 1.15rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 18px 50px rgba(0,0,0,0.16);
    backdrop-filter: blur(6px);
}
.section-title {
    font-size: 1.28rem;
    font-weight: 750;
    margin-bottom: 0.9rem;
    color: #f8fafc;
}
.small-muted {
    color: #94a3b8;
    font-size: 0.92rem;
}
.info-chip {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(96,165,250,0.20);
    color: #dbeafe;
    font-size: 0.86rem;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}
.chat-shell {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1rem;
    background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(17,24,39,0.98));
    box-shadow: 0 18px 50px rgba(0,0,0,0.22);
}
.chat-header {
    font-weight: 750;
    font-size: 1.2rem;
    margin-bottom: 0.2rem;
    color: #f8fafc;
}
.chat-sub {
    color: #94a3b8;
    font-size: 0.93rem;
    margin-bottom: 0.9rem;
}
.user-msg {
    background: rgba(37,99,235,0.16);
    border: 1px solid rgba(96,165,250,0.18);
    border-radius: 14px;
    padding: 0.8rem;
    margin-bottom: 0.65rem;
}
.bot-msg {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 0.8rem;
    margin-bottom: 0.85rem;
}
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.07);
    padding: 0.85rem 0.7rem;
    border-radius: 16px;
}
hr {
    border-color: rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# Helpers
# =========================================================
def load_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def guess_default(columns, keywords):
    cols_lower = [c.lower() for c in columns]
    for kw in keywords:
        for i, col in enumerate(cols_lower):
            if kw in col:
                return i
    return 0


def preprocess_sales_data(df_raw: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    df = df_raw.copy()

    rename_map = {
        col_map["date"]: "date",
        col_map["sales"]: "sales",
    }

    optional_fields = ["category", "store", "price", "base_price", "promo_flag"]
    for field in optional_fields:
        if col_map.get(field):
            rename_map[col_map[field]] = field

    df = df.rename(columns=rename_map)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["sales"].dtype == "object":
        df["sales"] = (
            df["sales"].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("₹", "", regex=False)
        )

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "base_price" in df.columns:
        df["base_price"] = pd.to_numeric(df["base_price"], errors="coerce")

    if "promo_flag" in df.columns:
        df["promo_flag"] = (
            df["promo_flag"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0})
        )
        df["promo_flag"] = pd.to_numeric(df["promo_flag"], errors="coerce").fillna(0).astype(int)
    else:
        df["promo_flag"] = 0

    if "base_price" in df.columns and "price" in df.columns:
        df["discount_pct"] = ((df["base_price"] - df["price"]) / df["base_price"]) * 100
        df["discount_pct"] = df["discount_pct"].replace([float("inf"), -float("inf")], pd.NA).fillna(0.0)
    else:
        df["discount_pct"] = 0.0

    df = df.dropna(subset=["date", "sales"]).copy()
    df = df.sort_values("date")

    if "category" in df.columns:
        df["category"] = df["category"].astype(str)

    if "store" in df.columns:
        df["store"] = df["store"].astype(str)

    return df


def compute_kpis(df_clean: pd.DataFrame) -> dict:
    if df_clean.empty:
        return {
            "records": 0,
            "date_min": None,
            "date_max": None,
            "total_sales": 0.0,
            "avg_sales_per_day": 0.0,
        }

    daily = df_clean.set_index("date")["sales"].resample("D").sum()

    return {
        "records": int(len(df_clean)),
        "date_min": df_clean["date"].min(),
        "date_max": df_clean["date"].max(),
        "total_sales": float(df_clean["sales"].sum()),
        "avg_sales_per_day": float(daily.mean()) if len(daily) else 0.0,
    }


def clear_ai_cache():
    for key in [
        "rag_index",
        "chat_history",
        "__series_used",
        "__forecast_future_df",
        "__forecast_ins_text",
        "__backtest_metrics",
        "__forecast_driver_summary",
        "__forecast_level_selected",
        "__forecast_regressors_used",
        "__holiday_usage",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["chat_history"] = []


def summarize_drivers(df: pd.DataFrame) -> str:
    parts = []

    if "price" in df.columns and df["price"].notna().sum() > 0:
        parts.append(f"Average observed price is {df['price'].mean():.2f}.")

    if "discount_pct" in df.columns and df["discount_pct"].notna().sum() > 0:
        parts.append(f"Average discount percentage is {df['discount_pct'].mean():.2f}%.")

    if "promo_flag" in df.columns:
        promo_rate = 100 * df["promo_flag"].mean()
        parts.append(f"Promotions were active in {promo_rate:.1f}% of records.")

    if "category" in df.columns:
        top_cat = df.groupby("category")["sales"].sum().sort_values(ascending=False).head(1)
        if not top_cat.empty:
            parts.append(f"Top revenue category is {top_cat.index[0]}.")

    if "store" in df.columns:
        top_store = df.groupby("store")["sales"].sum().sort_values(ascending=False).head(1)
        if not top_store.empty:
            parts.append(f"Top revenue store is {top_store.index[0]}.")

    return " ".join(parts) if parts else "No additional business drivers were available in the uploaded dataset."


# =========================================================
# Header
# =========================================================
st.markdown('<div class="main-title">AI-Powered Sales Analytics System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle-text">Upload a sales dataset, explore business dashboards, generate business-aware forecasts using Meta Prophet, and ask grounded questions using the AI assistant.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div>
        <span class="info-chip">Meta Prophet</span>
        <span class="info-chip">Price Impact</span>
        <span class="info-chip">Promotions</span>
        <span class="info-chip">Holiday Effects</span>
        <span class="info-chip">Category / Store Forecasting</span>
        <span class="info-chip">AI Assistant</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# 1) Upload + Mapping
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">1) Data Upload and Column Mapping</div>', unsafe_allow_html=True)

left_col, right_col = st.columns([1.05, 1.15], gap="large")

with left_col:
    st.markdown("**Upload Sales Dataset**")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            df_raw = load_csv(uploaded_file)
            st.session_state["df_raw"] = df_raw

            st.success("File uploaded successfully.")
            st.write("Dataset shape:", df_raw.shape)

            with st.expander("Preview raw dataset", expanded=False):
                st.dataframe(df_raw.head(20), use_container_width=True)

        except Exception as e:
            st.error("Could not read the CSV file.")
            st.write("Error details:", str(e))

with right_col:
    st.markdown("**Column Mapping**")

    if "df_raw" not in st.session_state:
        st.info("Upload a dataset first to enable mapping.")
    else:
        df_raw = st.session_state["df_raw"]
        cols = df_raw.columns.astype(str).tolist()

        date_default = guess_default(cols, ["order date", "date", "invoice date", "transaction date"])
        sales_default = guess_default(cols, ["total revenue", "sales", "revenue", "amount", "total", "price"])
        price_default = guess_default(cols, ["price", "selling price", "unit price"])
        base_price_default = guess_default(cols, ["base price", "mrp", "list price", "regular price"])
        promo_default = guess_default(cols, ["promo", "promotion", "discount flag", "campaign"])

        date_col = st.selectbox("Date column", cols, index=date_default)
        sales_col = st.selectbox("Sales column", cols, index=sales_default)
        category_col = st.selectbox("Category column (optional)", ["None"] + cols, index=0)
        store_col = st.selectbox("Store column (optional)", ["None"] + cols, index=0)
        price_col = st.selectbox("Price column (optional)", ["None"] + cols, index=min(price_default + 1, len(cols)))
        base_price_col = st.selectbox("Base price column (optional)", ["None"] + cols,
                                      index=min(base_price_default + 1, len(cols)))
        promo_col = st.selectbox("Promotion flag column (optional)", ["None"] + cols,
                                 index=min(promo_default + 1, len(cols)))

        if st.button("Confirm Mapping and Clean Data", type="primary"):
            col_map = {
                "date": date_col,
                "sales": sales_col,
                "category": None if category_col == "None" else category_col,
                "store": None if store_col == "None" else store_col,
                "price": None if price_col == "None" else price_col,
                "base_price": None if base_price_col == "None" else base_price_col,
                "promo_flag": None if promo_col == "None" else promo_col,
            }

            st.session_state["col_map"] = col_map
            df_clean = preprocess_sales_data(df_raw, col_map)
            st.session_state["df_clean"] = df_clean
            clear_ai_cache()

            if df_clean.empty:
                st.error("After preprocessing, no valid rows remain. Please re-check your selected columns.")
            else:
                st.success("Data cleaning completed successfully.")
                with st.expander("Preview cleaned dataset", expanded=False):
                    st.dataframe(df_clean.head(20), use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 2) Data Preview
# =========================================================
with st.expander("2) Data Preview", expanded=False):
    if "df_raw" not in st.session_state:
        st.info("Upload a dataset first.")
    else:
        st.subheader("Raw Dataset")
        st.dataframe(st.session_state["df_raw"].head(20), use_container_width=True)

        st.subheader("Current Mapping")
        if "col_map" in st.session_state:
            st.json(st.session_state["col_map"])
        else:
            st.warning("Mapping not configured yet.")

        if "df_clean" in st.session_state and not st.session_state["df_clean"].empty:
            st.subheader("Cleaned Dataset")
            st.dataframe(st.session_state["df_clean"].head(20), use_container_width=True)

st.divider()

# =========================================================
# 3) Dashboards
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">3) Dashboards</div>', unsafe_allow_html=True)

if "df_clean" not in st.session_state or st.session_state["df_clean"].empty:
    st.info("Please complete data cleaning first.")
else:
    df = st.session_state["df_clean"].copy()

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1.15], gap="large")
    with filter_col1:
        start_date = st.date_input("Start date", value=df["date"].min().date())
    with filter_col2:
        end_date = st.date_input("End date", value=df["date"].max().date())
    with filter_col3:
        trend_granularity = st.selectbox("Trend granularity", ["Daily", "Weekly", "Monthly"], index=2)

    df_f = df[
        (df["date"] >= pd.to_datetime(start_date)) &
        (df["date"] <= pd.to_datetime(end_date))
        ].copy()

    kpis = compute_kpis(df_f)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Sales", f"{kpis['total_sales']:.2f}")
    with m2:
        st.metric("Records", f"{kpis['records']}")
    with m3:
        st.metric("From", str(kpis["date_min"]).split(" ")[0] if kpis["date_min"] is not None else "-")
    with m4:
        st.metric("To", str(kpis["date_max"]).split(" ")[0] if kpis["date_max"] is not None else "-")

    if "price" in df_f.columns and df_f["price"].notna().sum() > 0:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Average Price", f"{df_f['price'].mean():.2f}")
        with c2:
            st.metric("Avg Discount %", f"{df_f['discount_pct'].mean():.2f}")
        with c3:
            st.metric("Promo Rate %", f"{100 * df_f['promo_flag'].mean():.1f}")

    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
    trend_series = build_series(df_f, freq=freq_map[trend_granularity])
    trend_df = pd.DataFrame({"date": trend_series.index, "sales": trend_series.values})

    fig_trend = px.line(trend_df, x="date", y="sales", title=f"Sales Trend ({trend_granularity})")
    st.plotly_chart(fig_trend, use_container_width=True)

    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        if "category" in df_f.columns:
            cat_df = (
                df_f.groupby("category")["sales"]
                .sum()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )
            fig_cat = px.bar(cat_df, x="category", y="sales", title="Top Categories by Sales")
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Category column not available in this dataset.")

    with chart_col2:
        if "store" in df_f.columns:
            store_df = (
                df_f.groupby("store")["sales"]
                .sum()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )
            fig_store = px.bar(store_df, x="store", y="sales", title="Top Stores by Sales")
            st.plotly_chart(fig_store, use_container_width=True)
        else:
            st.info("Store column not available in this dataset.")

st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# =========================================================
# 4) Forecasting
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">4) Forecasting (Business-Aware Prophet)</div>', unsafe_allow_html=True)

if "df_clean" not in st.session_state or st.session_state["df_clean"].empty:
    st.info("Please complete data cleaning first.")
else:
    df = st.session_state["df_clean"].copy().sort_values("date")

    fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 1.3], gap="large")

    with fc1:
        forecast_level = st.selectbox("Forecast level", ["Total Sales", "Category", "Store"], index=0)

    with fc2:
        forecast_granularity = st.selectbox("Forecast granularity", ["Weekly", "Monthly"], index=1)

    with fc3:
        forecast_horizon = st.slider("Forecast horizon", 4, 52, 12)

    with fc4:
        use_holidays = st.checkbox("Use holiday calendar", value=True)

    df_model = df.copy()

    if forecast_level == "Category":
        if "category" not in df.columns:
            st.warning("Category column not available for category-level forecasting.")
            df_model = None
        else:
            selected_category = st.selectbox(
                "Select category",
                sorted(df["category"].dropna().unique().tolist())
            )
            df_model = df[df["category"] == selected_category].copy()

    elif forecast_level == "Store":
        if "store" not in df.columns:
            st.warning("Store column not available for store-level forecasting.")
            df_model = None
        else:
            selected_store = st.selectbox(
                "Select store",
                sorted(df["store"].dropna().unique().tolist())
            )
            df_model = df[df["store"] == selected_store].copy()

    if df_model is not None:
        forecast_freq = "W" if forecast_granularity == "Weekly" else "M"

        regressor_cols = []
        for col in ["price", "promo_flag", "discount_pct"]:
            if col in df_model.columns and df_model[col].notna().sum() > 0:
                regressor_cols.append(col)

        train_df = prepare_prophet_training_df(
            df_model,
            freq=forecast_freq,
            date_col="date",
            sales_col="sales",
            regressors=regressor_cols
        )

        if len(train_df) < 8:
            st.warning("Not enough aggregated time periods to forecast. Add more data or use a coarser frequency.")
        else:
            holidays_df = make_default_holidays() if use_holidays else None

            forecast_df, model, model_name = prophet_fit_predict(
                train_df=train_df,
                periods=forecast_horizon,
                freq=forecast_freq,
                regressor_cols=regressor_cols,
                holidays_df=holidays_df
            )

            st.caption(f"Model used: {model_name}")

            last_hist_date = train_df["ds"].max()

            hist_part = forecast_df[forecast_df["ds"] <= last_hist_date].copy()
            fut_part = forecast_df[forecast_df["ds"] > last_hist_date].copy()

            hist_plot_df = train_df[["ds", "y"]].rename(columns={"ds": "date", "y": "actual"}).merge(
                hist_part[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "fitted"}),
                on="date",
                how="left"
            )

            fig_hist = px.line(
                hist_plot_df,
                x="date",
                y=["actual", "fitted"],
                title="Historical: Actual vs Fitted"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            future_plot_df = fut_part[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"}).copy()

            fig_future = px.line(
                future_plot_df,
                x="date",
                y="yhat",
                title=f"Future Forecast ({forecast_level})"
            )
            st.plotly_chart(fig_future, use_container_width=True)

            future_series = pd.Series(future_plot_df["yhat"].values, index=future_plot_df["date"])
            history_series = pd.Series(train_df["y"].values, index=train_df["ds"])

            forecast_insights = compute_forecast_insights(history=history_series, future_forecast=future_series)
            forecast_insights_text = insights_to_text(
                forecast_insights,
                freq_label=forecast_granularity,
                horizon=forecast_horizon
            )

            driver_summary = summarize_drivers(df_model)

            s1, s2 = st.columns([1.1, 0.9], gap="large")

            with s1:
                st.subheader("Forecast Insight Summary")
                st.write(forecast_insights_text)

            with s2:
                st.subheader("Business Driver Summary")
                st.write(driver_summary)

            st.subheader("Forecast Inputs Used")
            if regressor_cols:
                st.write("This forecast used:", ", ".join(regressor_cols))
            else:
                st.write("This forecast used historical sales only. No optional business regressors were available.")

            st.write(f"Holiday calendar applied: {'Yes' if use_holidays else 'No'}")

            try:
                holdout = min(forecast_horizon, max(2, len(history_series) // 4))
                backtest_df, backtest_metrics = prophet_backtest(history_series, horizon=holdout, freq=forecast_freq)

                bt1, bt2, bt3 = st.columns(3)
                with bt1:
                    st.metric("MAE", f"{backtest_metrics['mae']:.2f}")
                with bt2:
                    st.metric("RMSE", f"{backtest_metrics['rmse']:.2f}")
                with bt3:
                    mape_txt = f"{backtest_metrics['mape']:.2f}%" if pd.notna(backtest_metrics["mape"]) else "N/A"
                    st.metric("MAPE", mape_txt)

                fig_backtest = px.line(
                    backtest_df,
                    x="date",
                    y=["actual", "predicted"],
                    title="Backtest: Actual vs Predicted"
                )
                st.plotly_chart(fig_backtest, use_container_width=True)

                st.session_state["__backtest_metrics"] = backtest_metrics
            except Exception as e:
                st.warning(f"Backtest not available yet: {str(e)}")

            with st.expander("Forecast output tables", expanded=False):
                left_t, right_t = st.columns(2)
                with left_t:
                    st.write("Future Forecast")
                    st.dataframe(future_plot_df, use_container_width=True)
                with right_t:
                    st.write("Historical Actual vs Fitted")
                    st.dataframe(hist_plot_df.tail(24), use_container_width=True)

            st.session_state["__series_used"] = history_series
            st.session_state["__forecast_future_df"] = future_plot_df
            st.session_state["__forecast_ins_text"] = forecast_insights_text
            st.session_state["__forecast_driver_summary"] = driver_summary
            st.session_state["__forecast_level_selected"] = forecast_level
            st.session_state["__forecast_regressors_used"] = regressor_cols
            st.session_state["__holiday_usage"] = use_holidays

st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# =========================================================
# 5) AI Knowledge Base
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">5) AI Knowledge Base (Embeddings + FAISS)</div>', unsafe_allow_html=True)

if "df_clean" not in st.session_state or st.session_state["df_clean"].empty:
    st.info("Please complete data cleaning first.")
else:
    df = st.session_state["df_clean"]
    kpis = compute_kpis(df)

    kb1, kb2 = st.columns([1, 1.2], gap="large")

    with kb1:
        st.markdown("**Included Knowledge Sources**")
        st.write("- Dataset schema")
        st.write("- KPI summary")
        st.write("- Recent trend summary")
        st.write("- Forecast summary")
        st.write("- Forecast values")
        st.write("- Top categories / stores")
        st.write("- Backtest metrics")
        st.write("- Anomaly summary")
        st.write("- Business driver summary")

    with kb2:
        if st.button("Build / Rebuild Knowledge Base", type="primary"):
            try:
                from src.rag_pipeline import RAGIndex, Doc

                series = st.session_state.get("__series_used", None)
                future_df = st.session_state.get("__forecast_future_df", None)
                ins_text = st.session_state.get("__forecast_ins_text", "Forecast insights not generated yet.")
                driver_summary = st.session_state.get("__forecast_driver_summary", "Driver summary not available.")
                metrics = st.session_state.get("__backtest_metrics", None)
                forecast_level_used = st.session_state.get("__forecast_level_selected", "Total Sales")
                regressors_used = st.session_state.get("__forecast_regressors_used", [])
                holiday_used = st.session_state.get("__holiday_usage", False)

                docs = []
                docs.append(Doc(text=make_schema_doc(df), meta={"type": "schema"}))
                docs.append(Doc(text=make_kpi_doc(kpis), meta={"type": "kpi"}))

                if series is not None:
                    docs.append(Doc(text=make_recent_trend_doc(series), meta={"type": "trend"}))
                    docs.append(Doc(text=make_anomaly_doc(series), meta={"type": "anomaly"}))

                if future_df is not None and not future_df.empty:
                    docs.append(Doc(text=make_forecast_doc(future_df), meta={"type": "forecast"}))

                docs.append(Doc(text=ins_text, meta={"type": "forecast_insights"}))
                docs.append(Doc(text=driver_summary, meta={"type": "business_drivers"}))
                docs.append(Doc(
                    text=f"Forecast level selected: {forecast_level_used}. Regressors used: {', '.join(regressors_used) if regressors_used else 'none'}. Holiday calendar applied: {'Yes' if holiday_used else 'No'}.",
                    meta={"type": "forecast_config"}
                ))

                if metrics is not None:
                    docs.append(Doc(text=make_backtest_doc(metrics), meta={"type": "evaluation"}))

                for txt in make_top_entities_docs(df):
                    docs.append(Doc(text=txt, meta={"type": "entities"}))

                rag_index = RAGIndex()
                rag_index.build(docs)
                st.session_state["rag_index"] = rag_index

                st.success("Knowledge base built successfully.")

            except Exception as e:
                st.error(f"Failed to build knowledge base: {str(e)}")

        if "rag_index" in st.session_state:
            st.info("Knowledge base status: READY")
        else:
            st.warning("Knowledge base status: NOT BUILT YET")

st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# =========================================================
# 6) AI Assistant (On-page)
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">6) AI Assistant</div>', unsafe_allow_html=True)

left_space, chat_col = st.columns([0.82, 1.18], gap="large")

with left_space:
    st.markdown(
        """
        **What you can ask**

        - How do promotions affect sales?
        - Which category is strongest?
        - Was holiday effect included in the forecast?
        - What business drivers were used?
        - Explain the forecast trend in simple words.
        """
    )

with chat_col:
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header">AI Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-sub">Ask about sales trends, forecasts, price impact, promotions, holidays, categories, stores, and KPI insights.</div>',
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Clear chat", key="clear_inline_chat"):
            st.session_state["chat_history"] = []
            st.rerun()
    with col_b:
        st.write("")

    if "rag_index" not in st.session_state:
        st.info("Build the Knowledge Base first to enable the assistant.")
    else:
        rag_index = st.session_state["rag_index"]

        mode = st.selectbox(
            "Answer mode",
            ["Template (Cloud-safe)", "Local Ollama (requires Ollama running)"],
            key="inline_mode"
        )

        top_k = st.slider(
            "Top-k retrieved chunks",
            2, 10, 5,
            key="inline_topk"
        )

        for role, message in st.session_state["chat_history"]:
            if role == "user":
                st.markdown(f'<div class="user-msg"><b>You:</b><br>{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg"><b>Assistant:</b><br>{message}</div>', unsafe_allow_html=True)

        with st.form("inline_assistant_form", clear_on_submit=True):
            user_query = st.text_input(
                "Ask a question",
                placeholder="How do promotions and price changes affect sales?",
                key="inline_user_query"
            )
            submitted = st.form_submit_button("Send")

        if submitted and user_query:
            st.session_state["chat_history"].append(("user", user_query))

            try:
                results = rag_index.search_with_scores(user_query, k=top_k)
            except Exception:
                docs = rag_index.search(user_query, k=top_k)
                results = [(0.0, doc) for doc in docs]

            with st.expander("Retrieved context (debug)", expanded=False):
                for score, doc in results:
                    st.write(f"score={score:.3f} | type={doc.meta.get('type')}")
                    st.write(doc.text)

            if mode.startswith("Template"):
                answer = template_answer(user_query, results)
            else:
                try:
                    from src.llm_local import ollama_generate

                    prompt = build_grounded_prompt(user_query, results)
                    answer = ollama_generate(prompt)
                except Exception as e:
                    answer = f"Ollama call failed: {str(e)}"

            st.session_state["chat_history"].append(("assistant", answer))
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)