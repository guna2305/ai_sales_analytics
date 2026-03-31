import streamlit as st
import pandas as pd
import plotly.express as px

from src.forecasting import build_series
from src.forecast_prophet import prophet_fit_predict
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
if "show_chatbot" not in st.session_state:
    st.session_state.show_chatbot = False

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
.main .block-container {
    padding-top: 1.4rem;
    padding-bottom: 5.5rem;
    max-width: 1400px;
}
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}
.subtle-text {
    color: #a8acb3;
    font-size: 1rem;
    margin-bottom: 1.4rem;
}
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}
.small-muted {
    color: #a8acb3;
    font-size: 0.92rem;
}
.answer-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 0.85rem;
    margin-bottom: 0.75rem;
}

/* Floating chatbot button */
.st-key-chat_fab {
    position: fixed;
    right: 24px;
    bottom: 24px;
    z-index: 10000;
    width: 72px;
}

.st-key-chat_fab button {
    width: 72px !important;
    height: 72px !important;
    border-radius: 999px !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    background: linear-gradient(135deg, #2563eb, #06b6d4) !important;
    color: white !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    box-shadow: 0 16px 35px rgba(0,0,0,0.35) !important;
}

/* Floating chatbot window */
.floating-chat-window {
    position: fixed;
    right: 24px;
    bottom: 110px;
    width: 420px;
    max-width: calc(100vw - 24px);
    height: 620px;
    max-height: 78vh;
    overflow-y: auto;
    padding: 1rem;
    border-radius: 22px;
    background: #0f172a;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 22px 60px rgba(0,0,0,0.45);
    z-index: 9999;
}

.floating-chat-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.15rem;
}

.floating-chat-subtitle {
    color: #a8acb3;
    font-size: 0.9rem;
    margin-bottom: 0.85rem;
}

.user-msg {
    background: rgba(59,130,246,0.16);
    border: 1px solid rgba(96,165,250,0.20);
    border-radius: 14px;
    padding: 0.75rem;
    margin-bottom: 0.65rem;
}

.bot-msg {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 0.75rem;
    margin-bottom: 0.85rem;
}

@media (max-width: 768px) {
    .floating-chat-window {
        right: 12px;
        left: 12px;
        width: auto;
        bottom: 95px;
        height: 70vh;
    }

    .st-key-chat_fab {
        right: 16px;
        bottom: 16px;
    }
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
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["chat_history"] = []


# =========================================================
# Header
# =========================================================
st.markdown('<div class="main-title">AI-Powered Sales Analytics System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle-text">Upload a sales dataset, explore business dashboards, generate forecasts using Meta Prophet, and ask grounded questions using the AI assistant.</div>',
    unsafe_allow_html=True,
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

        date_col = st.selectbox("Date column", cols, index=date_default)
        sales_col = st.selectbox("Sales column", cols, index=sales_default)
        category_col = st.selectbox("Category column (optional)", ["None"] + cols, index=0)
        store_col = st.selectbox("Store column (optional)", ["None"] + cols, index=0)

        if st.button("Confirm Mapping and Clean Data", type="primary"):
            col_map = {
                "date": date_col,
                "sales": sales_col,
                "category": None if category_col == "None" else category_col,
                "store": None if store_col == "None" else store_col,
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
st.markdown('<div class="section-title">4) Forecasting (Meta Prophet)</div>', unsafe_allow_html=True)

if "df_clean" not in st.session_state or st.session_state["df_clean"].empty:
    st.info("Please complete data cleaning first.")
else:
    df = st.session_state["df_clean"].copy().sort_values("date")

    fc1, fc2, fc3 = st.columns([1, 1, 1.2], gap="large")
    with fc1:
        forecast_granularity = st.selectbox("Forecast granularity", ["Weekly", "Monthly"], index=1)
    with fc2:
        forecast_horizon = st.slider("Forecast horizon (future periods)", 4, 52, 12)
    with fc3:
        st.markdown(
            '<div class="small-muted">Historical graph is for model sanity-check. Future graph shows prediction periods only.</div>',
            unsafe_allow_html=True
        )

    forecast_freq = "W" if forecast_granularity == "Weekly" else "M"
    series = build_series(df, freq=forecast_freq)

    if len(series) < 8:
        st.warning("Not enough time periods to forecast. Add more data or use a coarser frequency.")
    else:
        forecast_df, model_name = prophet_fit_predict(series, periods=forecast_horizon, freq=forecast_freq)
        st.caption(f"Model used: {model_name}")

        last_hist_date = series.index.max()

        hist_part = forecast_df[forecast_df["ds"] <= last_hist_date].copy()
        fut_part = forecast_df[forecast_df["ds"] > last_hist_date].copy()

        hist_plot_df = pd.DataFrame({
            "date": series.index,
            "actual": series.values
        }).merge(
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
            title=f"Future Forecast (next {forecast_horizon} {forecast_granularity.lower()} periods)"
        )
        st.plotly_chart(fig_future, use_container_width=True)

        future_series = pd.Series(future_plot_df["yhat"].values, index=future_plot_df["date"])
        forecast_insights = compute_forecast_insights(history=series, future_forecast=future_series)
        forecast_insights_text = insights_to_text(
            forecast_insights,
            freq_label=forecast_granularity,
            horizon=forecast_horizon
        )

        st.subheader("Forecast Insight Summary")
        st.write(forecast_insights_text)

        st.subheader("Forecast Evaluation")
        try:
            holdout = min(forecast_horizon, max(2, len(series) // 4))
            backtest_df, backtest_metrics = prophet_backtest(series, horizon=holdout, freq=forecast_freq)

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

        st.session_state["__series_used"] = series
        st.session_state["__forecast_future_df"] = future_plot_df
        st.session_state["__forecast_ins_text"] = forecast_insights_text

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

    with kb2:
        if st.button("Build / Rebuild Knowledge Base", type="primary"):
            try:
                from src.rag_pipeline import RAGIndex, Doc

                series = st.session_state.get("__series_used", None)
                future_df = st.session_state.get("__forecast_future_df", None)
                ins_text = st.session_state.get("__forecast_ins_text", "Forecast insights not generated yet.")
                metrics = st.session_state.get("__backtest_metrics", None)

                docs = []
                docs.append(Doc(text=make_schema_doc(df), meta={"type": "schema"}))
                docs.append(Doc(text=make_kpi_doc(kpis), meta={"type": "kpi"}))

                if series is not None:
                    docs.append(Doc(text=make_recent_trend_doc(series), meta={"type": "trend"}))
                    docs.append(Doc(text=make_anomaly_doc(series), meta={"type": "anomaly"}))

                if future_df is not None and not future_df.empty:
                    docs.append(Doc(text=make_forecast_doc(future_df), meta={"type": "forecast"}))

                docs.append(Doc(text=ins_text, meta={"type": "forecast_insights"}))

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


# =========================================================
# 6) Floating AI Assistant
# =========================================================
if st.session_state.show_chatbot:
    st.markdown('<div class="floating-chat-window">', unsafe_allow_html=True)
    st.markdown('<div class="floating-chat-title">AI Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="floating-chat-subtitle">Ask about sales trends, forecasts, categories, stores, and KPI insights.</div>',
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Clear chat", key="clear_floating_chat"):
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
            key="floating_mode"
        )

        top_k = st.slider(
            "Top-k retrieved chunks",
            2, 10, 5,
            key="floating_topk"
        )

        for role, message in st.session_state["chat_history"]:
            if role == "user":
                st.markdown(f'<div class="user-msg"><b>You:</b><br>{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg"><b>Assistant:</b><br>{message}</div>', unsafe_allow_html=True)

        with st.form("floating_assistant_form", clear_on_submit=True):
            user_query = st.text_input(
                "Ask a question",
                placeholder="How can I improve my sales?",
                key="floating_user_query"
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


fab_icon = "✕" if st.session_state.show_chatbot else "◉"
if st.button(fab_icon, key="chat_fab", help="Open AI Assistant"):
    st.session_state.show_chatbot = not st.session_state.show_chatbot
    st.rerun()