import streamlit as st
import pandas as pd
from data_ingestion import get_yahoo_data, get_alpha_data, get_news_data
from credit_scoring import credit_score_model
from explainability import explain_score

# -------------------------------
# Streamlit Dashboard App
# -------------------------------

st.set_page_config(page_title="CredTech - Explainable Credit Intelligence", layout="wide")

st.title("📊 CredTech - Explainable Credit Intelligence Platform")
st.write("Real-Time Creditworthiness Scoring with Explainability")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (default: AAPL)", value="AAPL")

if st.sidebar.button("Run Analysis"):
    # -------------------------------
    # Step 1: Data Ingestion
    # -------------------------------
    st.subheader("📥 Data Ingestion")
    try:
        yahoo_data = get_yahoo_data(ticker)
        alpha_data = get_alpha_data(ticker)
        news_data = get_news_data(ticker)

        st.write("### Yahoo Finance Data (Recent Prices)")
        st.dataframe(yahoo_data.tail())

        st.write("### Alpha Vantage Data (Daily Prices)")
        st.dataframe(alpha_data.head())

        st.write("### Recent News")
        st.dataframe(news_data[["title", "publishedAt", "source"]].head())
    except Exception as e:
        st.error(f"Error in data ingestion: {e}")

    # -------------------------------
    # Step 2: Credit Scoring
    # -------------------------------
    st.subheader("💳 Credit Scoring Engine")
    try:
        score, model_used = credit_score_model(yahoo_data)
        st.metric(label="Creditworthiness Score", value=score)
        st.write(f"Model Used: {model_used}")
    except Exception as e:
        st.error(f"Error in credit scoring: {e}")

    # -------------------------------
    # Step 3: Explainability
    # -------------------------------
    st.subheader("🔍 Explainability Insights")
    try:
        explanation = explain_score(yahoo_data)
        st.write(explanation)
    except Exception as e:
        st.error(f"Error in explainability: {e}")

else:
    st.info("Enter a stock ticker in the sidebar and click *Run Analysis*")
