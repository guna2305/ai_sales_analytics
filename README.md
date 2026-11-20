
LIVE LINK: https://aisalesanalytics.streamlit.app/

# AI-Powered Sales Analytics

Turn a sales CSV into interactive dashboards, time-series forecasts, and a searchable business knowledge base.

This project brings data preparation, Prophet forecasting, and retrieval-based question answering into one Streamlit application. It is designed for exploring retail, store, and category sales without writing analysis code.

**Start here:** [Run locally](#run-locally) · [User walkthrough](#user-walkthrough) · [How it works](#how-it-works)

> **Viewing the project:** This README explains the application and its workflow. To open the interactive dashboard, follow the local setup below. No public hosted demo URL is provided here.

## What you can do

| Capability | What you see |
| --- | --- |
| CSV upload and column mapping | A preview of your raw data and a standardized, cleaned dataset |
| Sales dashboard | Total sales, record count, date range, trends, and optional category/store comparisons |
| Business-aware forecasts | Weekly or monthly Prophet forecasts for total sales, a category, or a store |
| Forecast review | Historical fitted values, future estimates, forecast tables, and backtest metrics when available |
| Searchable knowledge base | Summaries of KPIs, forecasts, business drivers, top entities, and evaluation results |
| AI assistant | Retrieved findings in template mode, or locally generated answers through Ollama |

## How it works

1. **Upload and map:** Select the date and sales columns in your CSV, plus any optional business fields.
2. **Clean:** The app parses dates, converts sales values to numbers, removes rows with invalid dates or sales, and sorts the records.
3. **Analyze:** Pandas aggregates the data and Plotly displays trends and comparisons.
4. **Forecast:** Prophet models weekly or monthly sales with available price, promotion, and discount regressors and an optional holiday calendar.
5. **Build knowledge:** Analytical summaries are embedded with Sentence Transformers and indexed with FAISS.
6. **Ask questions:** Relevant summaries are retrieved for each question. Template mode displays those findings; Ollama mode uses them as context for a local language model.

## Run locally

You need Git and a Python environment compatible with the packages in [requirements.txt](requirements.txt). Python 3.11 is a practical starting point; dependencies are currently unpinned.

```bash
git clone https://github.com/guna2305/ai_sales_analytics.git
cd ai_sales_analytics
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install and start:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally **http://localhost:8501**. The first knowledge-base build may download the embedding model.

### Optional: local AI answers

Template mode does not require Ollama or an API key. For generated answers, install and start Ollama on the same machine as Streamlit, then download the model used by [src/llm_local.py](src/llm_local.py):

```bash
ollama pull llama3
```

Select **Local Ollama (requires Ollama running)** in the assistant. The app calls `http://localhost:11434/api/generate`. On a hosted server, localhost refers to that server, so a visitor's own Ollama installation is not automatically available.

## Prepare your CSV

| Field | Required? | Example |
| --- | --- | --- |
| Date | Yes | `2026-01-01` |
| Sales | Yes | `1250.50` |
| Category | No | `Electronics` |
| Store | No | `Store A` |
| Price | No | `90` |
| Base price | No | `100` |
| Promotion flag | No | `1` / `0`, `yes` / `no` |

Column names can differ: map them in the interface. Use distinct source columns for different fields, and explicitly choose **None** for optional fields you do not have; the interface may preselect a column.

The forecast requires **at least eight aggregated weekly or monthly periods**. A few sample rows can demonstrate upload and charts, but are insufficient for forecasting. Use a longer sales history for a meaningful demonstration.

## User walkthrough

1. Upload your CSV and inspect the raw preview.
2. Map the fields and click **Confirm Mapping and Clean Data**.
3. Review **Dashboards**, adjust the date range, and explore daily, weekly, or monthly trends.
4. In **Forecasting**, choose total sales, category, or store; select weekly/monthly granularity and a forecast horizon.
5. Review forecast charts, business-driver summaries, output tables, and available MAE, RMSE, and MAPE metrics.
6. Click **Build / Rebuild Knowledge Base** after generating the forecast.
7. Start with **Template (Cloud-safe)** and ask:
   - “Which category has the highest sales?”
   - “Explain the forecast trend.”
   - “What business drivers were used?”
   - “Was a holiday calendar included?”
8. Open **Retrieved context (debug)** to inspect the information behind an answer.

## Working notes and limitations

- Dashboard date filters affect the dashboard; forecasting and knowledge-base construction use the cleaned dataset, with the selected forecast category/store applied separately.
- Rebuild the knowledge base after changing data or forecast settings so the assistant uses updated summaries.
- Forecasts are estimates, and retrieved context does not guarantee that generated answers are correct. Review the source values and backtest results.
- Price and promotion summaries describe the supplied data; they do not establish a causal effect.
- Data and chat state are held in the Streamlit session. This repository does not provide a persistent user account or saved-analysis workflow.
- Setup instructions are based on the source code; they are not a claim of a fresh end-to-end runtime test.

## Code guide

| File | Purpose |
| --- | --- |
| [app.py](app.py) | Streamlit interface, upload, mapping, dashboard, and workflow |
| [src/forecast_prophet.py](src/forecast_prophet.py) | Prophet training and prediction |
| [src/evaluation.py](src/evaluation.py) | Forecast backtesting |
| [src/knowledge_builder.py](src/knowledge_builder.py) | Analytics-to-text knowledge documents |
| [src/rag_pipeline.py](src/rag_pipeline.py) | Embeddings and FAISS retrieval |
| [src/rag_answering.py](src/rag_answering.py) | Template answers and grounded prompts |
| [src/llm_local.py](src/llm_local.py) | Optional Ollama integration |

Built with Python, Streamlit, Pandas, Plotly, Prophet, Sentence Transformers, and FAISS.


An end-to-end intelligent sales analytics platform that enables businesses to upload sales datasets, explore insights through interactive dashboards, forecast future sales using machine learning, and query business insights using an AI-powered assistant.

This project demonstrates the integration of data engineering, time-series forecasting, and retrieval-augmented AI systems within a modern web-based analytics application.

The application is deployed using Streamlit and supports dynamic datasets, making it adaptable for multiple retail and sales environments.

Project Overview

Modern businesses generate large amounts of sales data but often lack simple tools to analyze trends and generate forecasts without complex data science workflows.
This project addresses that problem by providing a unified system where users can:
Upload any sales dataset
Automatically map and clean data
Visualize trends and key performance indicators
Forecast future sales using time-series modeling
Interact with an AI assistant to understand business insights
The system is designed to be dataset-agnostic, meaning it can work with different retail datasets such as supermarket, e-commerce, or store-level transaction data.

Key Features

Dynamic Data Ingestion
Users can upload CSV datasets containing sales information. The system detects available columns and allows manual column mapping for:
Date
Sales value
Product category (optional)
Store/location (optional)
This design ensures the application works across different dataset structures.
Automated Data Preprocessing
After column mapping, the system performs preprocessing steps including:
Date parsing and formatting
Cleaning currency symbols and commas
Converting sales values to numeric format
Removing invalid rows
Sorting data chronologically
Creating a standardized schema for analytics
Interactive Sales Dashboards
The platform provides interactive visualizations to analyze sales performance.

Key analytics include:
Total sales metrics
Sales trend visualization
Date range filtering
Category-wise sales distribution
Store-wise performance breakdown
Aggregated trends (daily, weekly, monthly)
These dashboards help identify patterns, seasonality, and performance variations across products or stores.

Time-Series Forecasting with Meta Prophet

The forecasting engine uses Meta Prophet, a robust time-series forecasting library designed for business data.
The forecasting pipeline performs the following:
Converts aggregated sales data into Prophet format (ds, y)
Trains the forecasting model
Generates predictions for future periods
Separates historical fitted values and future forecasts
Visualizes results using interactive charts
This allows businesses to anticipate future demand trends.
Automated Forecast Insight Generation

Beyond forecasting, the system automatically generates business insights from model predictions, including:

Expected growth percentage
Peak forecast period
Trend direction
Forecast volatility measure
These insights help decision-makers quickly understand model results without manual interpretation.

AI Knowledge Base using Embeddings and Vector Search
To enable intelligent question answering, the system converts analytical outputs into a knowledge base using:
Sentence embeddings
Vector similarity search
FAISS vector database
Information stored in the knowledge base includes:
KPI summaries
Forecast insights
Forecasted sales values
This allows semantic search over business analytics.
AI Assistant (Retrieval-Augmented System)
The AI assistant retrieves relevant information from the vector database and generates contextual responses to user queries.

Example queries include:

"What is the expected sales trend for the next months?"

"Which period has the highest predicted sales?"

"What is the overall sales growth?"

The chatbot ensures responses are grounded in the dataset and forecasting results.
