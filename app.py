import streamlit as st
import requests
from textblob import TextBlob
import pandas as pd
import time

# Set your API Key here
API_KEY = "65af9d35def2491cbf609f0916e0dabf"
URL = f"https://newsapi.org/v2/top-headlines?language=en&apiKey={API_KEY}"

st.title("📰 Real-Time News Sentiment Dashboard")

def fetch_news():
    response = requests.get(URL)
    data = response.json()
    headlines = [article['title'] for article in data['articles'] if article['title']]
    return headlines

# Auto-refresh every 60 seconds
while True:
    news = fetch_news()
    results = [{"headline": h,
                "sentiment": "Positive" if TextBlob(h).sentiment.polarity > 0 else "Negative"}
               for h in news]

    df = pd.DataFrame(results)
    st.write(df)

    # Sentiment summary chart
    sentiment_counts = df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    time.sleep(60)
