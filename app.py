import streamlit as st
import pandas as pd
import requests
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import altair as alt
import time

# Download VADER lexicon (only once)
nltk.download('vader_lexicon')

# Spark session caching
@st.cache_resource
def init_spark():
    return SparkSession.builder \
        .appName("NewsSentimentStreaming") \
        .master("local[*]") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

spark_session = init_spark()

# Get API key from secrets
API_KEY = "65af9d35def2491cbf609f0916e0dabf"

# Fetch news from NewsAPI
def get_news(query, limit=5):
    api_url = f"https://newsapi.org/v2/everything?q={query}&pageSize={limit}&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(api_url)
    articles = response.json().get("articles", [])
    return pd.DataFrame([{"headline": art["title"]} for art in articles if art.get("title")])

# Sentiment labeling with VADER
def assign_sentiment(df_input):
    analyzer = SentimentIntensityAnalyzer()
    def classify(text):
        compound_score = analyzer.polarity_scores(text)["compound"]
        if compound_score > 0.05:
            return "Positive"
        elif compound_score < -0.05:
            return "Negative"
        else:
            return "Neutral"
    df_input["sentiment"] = df_input["headline"].apply(classify)
    label_encoding = {"Positive": 2.0, "Neutral": 1.0, "Negative": 0.0}
    df_input["label"] = df_input["sentiment"].map(label_encoding)
    return df_input

# Streamlit UI
st.title("📰 PySpark News Sentiment Dashboard")

search_topic = st.text_input("Topic", "technology")
mode_option = st.radio("Mode", ["Bootstrap", "Predict"])
article_count = st.slider("Number of articles", 3, 20, 5)

if st.button("Run"):
    df_news = get_news(search_topic, article_count)
    if df_news.empty:
        st.error("No news found.")
    else:
        if mode_option == "Bootstrap":
            # Label data
            df_labeled = assign_sentiment(df_news)
            spark_df = spark_session.createDataFrame(df_labeled)

            # Spark ML Pipeline
            tokenizer_stage = Tokenizer(inputCol="headline", outputCol="words")
            words_df = tokenizer_stage.transform(spark_df)

            hashing_stage = HashingTF(inputCol="words", outputCol="rawFeatures")
            features_df = hashing_stage.transform(words_df)

            idf_stage = IDF(inputCol="rawFeatures", outputCol="features")
            idf_model = idf_stage.fit(features_df)
            scaled_df = idf_model.transform(features_df)

            lr_model = LogisticRegression(maxIter=10, regParam=0.001)
            trained_model = lr_model.fit(scaled_df)

            # Training accuracy
            evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="accuracy")
            training_accuracy = evaluator.evaluate(trained_model.transform(scaled_df))

            # Store in session
            st.session_state.update({
                "model": trained_model,
                "tokenizer": tokenizer_stage,
                "hashingTF": hashing_stage,
                "idf_model": idf_model
            })

            st.success("Model trained successfully!")
            st.info(f"Training Accuracy: {training_accuracy:.2%}")
            st.subheader("Bootstrap Data")
            st.write(df_labeled[["headline", "sentiment"]])

        elif mode_option == "Predict":
            if "model" not in st.session_state:
                st.error("Run Bootstrap first!")
            else:
                # Convert new data to Spark DataFrame
                spark_df = spark_session.createDataFrame(df_news)

                # Transform data using cached pipeline
                tokenizer_stage = st.session_state["tokenizer"]
                hashing_stage = st.session_state["hashingTF"]
                idf_model = st.session_state["idf_model"]
                trained_model = st.session_state["model"]

                words_df = tokenizer_stage.transform(spark_df)
                features_df = hashing_stage.transform(words_df)
                scaled_df = idf_model.transform(features_df)
                predictions = trained_model.transform(scaled_df)

                predictions_pd = predictions.select("headline", "prediction").toPandas()
                label_decoder = {2.0: "Positive", 1.0: "Neutral", 0.0: "Negative"}
                predictions_pd["sentiment"] = predictions_pd["prediction"].map(label_decoder)

                st.subheader("Predictions")
                st.write(predictions_pd[["headline", "sentiment"]])

                # Real-time streaming simulation (display each row with delay)
                st.subheader("Streaming Simulation")
                for _, row in predictions_pd.iterrows():
                    st.write(f"**{row['headline']}** → {row['sentiment']}")
                    time.sleep(0.5)

                # Altair bar chart for sentiment distribution
                chart_df = predictions_pd.groupby("sentiment").size().reset_index(name='count')
                sentiment_chart = alt.Chart(chart_df).mark_bar().encode(
                    x='sentiment',
                    y='count',
                    color='sentiment'
                )

                st.altair_chart(sentiment_chart, use_container_width=True)
