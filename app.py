import streamlit as st
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

# ------------------------
# Streamlit UI
# ------------------------
st.title("Real-Time News Sentiment Dashboard")

# Refresh interval (seconds)
REFRESH_INTERVAL = 30

# GNews API Key
API_KEY = "API_KEY"  # <-- Replace with your key
URL = f"https://gnews.io/api/v4/top-headlines?country=in&max=10&apikey={API_KEY}"

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Sentiment function
def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    return "Positive" if score >= 0.0 else "Negative"

# Function to color-code sentiment
def color_sentiment(val):
    color = 'green' if val == 'Positive' else 'red'
    return f'color: {color}'

# Containers for smooth updates
news_container = st.empty()
chart_container = st.empty()

# Main loop for auto-refresh
while True:
    response = requests.get(URL)
    data = response.json()
    articles = data.get('articles', [])

    if articles:
        df = pd.DataFrame(articles)

        if 'description' not in df.columns:
            df['description'] = ""

        # Combine title + description
        df['full_text'] = df['title'] + " " + df['description']
        df['sentiment'] = df['full_text'].apply(get_sentiment)

        # Apply color styling
        styled_df = df[['title', 'sentiment']].style.applymap(color_sentiment, subset=['sentiment'])

        # Update news table
        news_container.subheader("Latest News with Sentiment")
        news_container.dataframe(styled_df)

        # Update sentiment bar chart
        chart_container.subheader("Sentiment Distribution")
        chart_container.bar_chart(df['sentiment'].value_counts())

    else:
        news_container.warning("No news fetched. Check your API key or internet connection.")

    time.sleep(REFRESH_INTERVAL)
