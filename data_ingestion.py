import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from newsapi import NewsApiClient
import pandas as pd
import os

# Yahoo Finance Example
def get_yahoo_data(ticker="AAPL"):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1mo")
    return df

# Alpha Vantage Example
def get_alpha_data(symbol="AAPL"):
    ts = TimeSeries(key=os.getenv("ALPHA_VANTAGE_KEY"), output_format='pandas')
    data, meta = ts.get_daily(symbol=symbol, outputsize='compact')
    return data

# News API Example
def get_news_data(query="finance"):
    newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
    articles = newsapi.get_everything(q=query, language="en", sort_by="relevancy")
    return pd.DataFrame(articles['articles'])

if _name_ == "_main_":
    print("Yahoo Data Sample:\n", get_yahoo_data().head())
    print("Alpha Vantage Data Sample:\n", get_alpha_data().head())
    print("News Data Sample:\n", get_news_data().head())
