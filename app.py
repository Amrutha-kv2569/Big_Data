# Install dependencies (run once)
!pip install pyspark requests nltk altair vega_datasets

import requests
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import altair as alt
import time

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize Spark session
spark = SparkSession.builder \
    .appName("NewsSentimentColab") \
    .master("local[*]") \
    .config("spark.ui.showConsoleProgress", "false") \
    .getOrCreate()

# ====== Inputs ======
NEWS_API_KEY = "65af9d35def2491cbf609f0916e0dabf"  # Replace with your API key
topic = "technology"
num_articles = 5
mode = "Bootstrap"  # Options: "Bootstrap" or "Predict"

# ====== Fetch news from NewsAPI ======
def fetch_news(topic, page_size=5):
    url = f"https://newsapi.org/v2/everything?q={topic}&pageSize={page_size}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    r = requests.get(url)
    articles = r.json().get("articles", [])
    return pd.DataFrame([{"title": a["title"]} for a in articles if a.get("title")])

# ====== Sentiment labeling with VADER ======
def label_sentiment(df_pd):
    sia = SentimentIntensityAnalyzer()
    def classify_title(title):
        score = sia.polarity_scores(title)["compound"]
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"
    df_pd["sentiment"] = df_pd["title"].apply(classify_title)
    # Map to numeric labels for Spark ML
    label_map = {"Positive": 2.0, "Neutral": 1.0, "Negative": 0.0}
    df_pd["label"] = df_pd["sentiment"].map(label_map)
    return df_pd

# ====== Main workflow ======
df_pd = fetch_news(topic, num_articles)
if df_pd.empty:
    print("No news found.")
else:
    if mode == "Bootstrap":
        print("Bootstrapping model with fetched news...")
        df_pd = label_sentiment(df_pd)
        df_spark = spark.createDataFrame(df_pd)

        # Spark ML Pipeline (manually, no Streamlit)
        tokenizer = Tokenizer(inputCol="title", outputCol="words")
        words_data = tokenizer.transform(df_spark)

        hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
        featurized_data = hashingTF.transform(words_data)

        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idf_model = idf.fit(featurized_data)
        rescaled_data = idf_model.transform(featurized_data)

        lr = LogisticRegression(maxIter=10, regParam=0.001)
        model = lr.fit(rescaled_data)

        # Training accuracy
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy")
        train_accuracy = evaluator.evaluate(model.transform(rescaled_data))

        print(f"Training Accuracy: {train_accuracy:.2%}")
        display(df_pd[["title", "sentiment"]])

    elif mode == "Predict":
        try:
            # Ensure model exists (run Bootstrap first)
            model, tokenizer, hashingTF, idf_model
        except NameError:
            print("Run Bootstrap first!")
        else:
            df_spark = spark.createDataFrame(df_pd)

            words_data = tokenizer.transform(df_spark)
            featurized_data = hashingTF.transform(words_data)
            rescaled_data = idf_model.transform(featurized_data)
            predictions = model.transform(rescaled_data)

            predictions_df = predictions.select("title", "prediction").toPandas()
            label_map_reverse = {2.0: "Positive", 1.0: "Neutral", 0.0: "Negative"}
            predictions_df["sentiment"] = predictions_df["prediction"].map(label_map_reverse)

            print("Predictions:")
            display(predictions_df[["title", "sentiment"]])

            # Simulate streaming
            print("Streaming Simulation:")
            for idx, row in predictions_df.iterrows():
                print(f"{row['title']} → {row['sentiment']}")
                time.sleep(0.5)

            # Altair bar chart
            chart_data = predictions_df.groupby("sentiment").size().reset_index(name='count')
            chart = alt.Chart(chart_data).mark_bar().encode(
                x='sentiment',
                y='count',
                color='sentiment'
            )
            chart.display()
