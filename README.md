AI- Powered sales analytics system

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
