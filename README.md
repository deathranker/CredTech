# CredTech – Explainable Credit Intelligence Platform

## 📌 Overview
CredTech is a *Real-Time Explainable Credit Intelligence Platform* built for the *CredTech Hackathon, organized by *The Programming Club, IIT Kanpur and powered by Deep Root Investments.  

The project addresses the limitations of traditional credit rating systems, which are:
- Updated infrequently  
- Based on opaque methodologies  
- Often lagging behind real-world events  

Our solution leverages *Python, AI/ML, and data pipelines* to generate *real-time, transparent, and explainable creditworthiness scores* using both *structured* and *unstructured financial data*.

---

## 🚀 Problem Statement
To build a platform that:
1. *Ingests and processes* structured & unstructured financial/macro data in real-time.  
2. *Generates creditworthiness scores* for issuers & asset classes.  
3. *Provides explainability* by showing feature contributions, trends, and event-driven reasoning.  
4. *Delivers insights via an interactive dashboard* for analysts and decision-makers.  
5. *Ensures deployment readiness* with containerization, scalability, and reproducibility.

---

## 🏗 Features Implemented
### ✅ High-Throughput Data Ingestion
- *Structured data* from Yahoo Finance & Alpha Vantage APIs  
- *Unstructured data* from NewsAPI (financial news sentiment)  
- Data cleaning, normalization & real-time updates  

### ✅ Adaptive Scoring Engine
- ML-based scoring using Decision Trees, Logistic Regression & XGBoost  
- Frequent updates with incremental learning  
- Explainability via SHAP & LIME  

### ✅ Explainability Layer
- Feature contribution breakdowns  
- Trend insights (short vs long term)  
- Event-driven reasoning from unstructured data  

### ✅ Interactive Analyst Dashboard
- Score trends visualization  
- Feature importance charts  
- Event-driven explanations  
- Alerts for sudden score changes  

### ✅ End-to-End Deployment
- Dockerized for reproducibility  
- Supports real-time retraining & updates  

---

## ⚙ Tech Stack
- *Python* (Pandas, Numpy, Scikit-learn, XGBoost, SHAP, LIME)  
- *APIs*: Yahoo Finance, Alpha Vantage, NewsAPI  
- *Dashboard*: Streamlit  
- *Deployment*: Docker  

---

## 📊 System Architecture
```mermaid
flowchart TD
    A[Data Sources] --> B[Ingestion Pipeline]
    B --> C[Preprocessing & Feature Engineering]
    C --> D[Credit Scoring Engine]
    D --> E[Explainability Layer]
    E --> F[Interactive Dashboard]
    F --> G[End Users]
