import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from newsapi import NewsApiClient
import pandas as pd
import os

DEMO_MODE = True  # ✅ Toggle this to False for real APIs during final demo

def get_yahoo_data(ticker="AAPL"):
    if DEMO_MODE:
        # Fallback demo data
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        prices = pd.Series([150 + i*0.5 for i in range(30)], index=dates)
        df = pd.DataFrame({"Close": prices})
        return df
    else:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1mo")
        return df

def get_alpha_data(symbol="AAPL"):
    if DEMO_MODE:
        # Fallback demo data
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        prices = pd.Series([100 + i*0.3 for i in range(30)], index=dates)
        df = pd.DataFrame({"close": prices})
        return df
    else:
        ts = TimeSeries(key=os.getenv("ALPHA_VANTAGE_KEY"), output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
        return data

def get_news_data(query="finance"):
    if DEMO_MODE:
        # Fallback demo news
        return pd.DataFrame([
            {"title": "Company X reports steady growth", "publishedAt": "2024-02-01", "source": "Demo News"},
            {"title": "Markets remain stable amid global trends", "publishedAt": "2024-02-02", "source": "Demo News"},
            {"title": "Investors show confidence in tech sector", "publishedAt": "2024-02-03", "source": "Demo News"},
        ])
    else:
        newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
        articles = newsapi.get_everything(q=query, language="en", sort_by="relevancy")
        return pd.DataFrame(articles['articles'])

if _name_ == "_main_":
    print("Yahoo Data Sample:\n", get_yahoo_data().head())
    print("Alpha Vantage Data Sample:\n", get_alpha_data().head())
    print("News Data Sample:\n", get_news_data().head())
