pip install pyspark
import requests

API_KEY = "65af9d35def2491cbf609f0916e0dabf"
URL = f"https://newsapi.org/v2/top-headlines?language=en&apiKey={API_KEY}"

def fetch_news():
    response = requests.get(URL)
    data = response.json()
    headlines = [article['title'] for article in data['articles'] if article['title']]
    return headlines

headlines = fetch_news()
print(headlines[:5])
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("NewsSentimentClassification") \
    .getOrCreate()
df = spark.createDataFrame([(h,) for h in headlines], ["headline"])
df.show(truncate=False)
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline

tokenizer = Tokenizer(inputCol="headline", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from textblob import TextBlob

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return "Positive" if polarity > 0 else "Negative"

sentiment_udf = udf(get_sentiment, StringType())
df = df.withColumn("label", sentiment_udf(df["headline"]))
df.show(truncate=False)
