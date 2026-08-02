"""
NLP Sentiment Analysis Training Script.

Trains TF-IDF Vectorizer + Logistic Regression model on product review samples.
Output Artifacts:
- models/sentiment_model.pkl
- models/vectorizer.pkl
"""

import os
from pathlib import Path
import pickle
import re
import string
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Paths
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

# English Stopwords
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few",
    "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "you",
    "your", "yours", "yourself", "yourselves"
}

# Simple Lemmatization Map for common product review terms
LEMMA_DICT = {
    "loved": "love", "loves": "love", "loving": "love",
    "liked": "like", "likes": "like", "liking": "like",
    "great": "great", "best": "good", "better": "good",
    "bought": "buy", "buying": "buy", "buys": "buy",
    "worked": "work", "working": "work", "works": "work",
    "horrible": "bad", "terrible": "bad", "worst": "bad",
    "products": "product", "items": "item", "purchases": "purchase",
    "shoes": "shoe", "bags": "bag", "clothes": "clothing",
    "delivers": "deliver", "delivered": "deliver", "delivering": "deliver",
    "returns": "return", "returned": "return", "returning": "return",
}


def preprocess_text(text: str) -> str:
    """
    Text preprocessing pipeline:
    1. Lowercasing
    2. Punctuation removal
    3. Stopword removal
    4. Lemmatization
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower().strip()

    # 2. Punctuation removal via regex
    text = re.sub(r"[^\w\s]", " ", text)

    # 3. Tokenize & Filter stopwords
    tokens = text.split()
    filtered_tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]

    # 4. Lemmatization
    lemmatized = [LEMMA_DICT.get(w, w) for w in filtered_tokens]

    return " ".join(lemmatized)


# Synthetic training dataset of product reviews
SAMPLE_REVIEWS = [
    # Positive Reviews
    ("Exceptional product quality! Super fast delivery and wonderful packaging.", "positive"),
    ("I love these shoes! Extremely comfortable, durable, and stylish.", "positive"),
    ("Great customer service, friendly staff, and easy return policy.", "positive"),
    ("Works perfectly as expected. Highly recommend this brand to everyone!", "positive"),
    ("Very satisfied with my purchase. The item exceeded my expectations.", "positive"),
    ("Fantastic store experience. Will definitely buy from here again!", "positive"),
    ("Smooth checkout process and quick shipping. 5 stars!", "positive"),
    ("High quality fabric and true to size fit. Loving this bag!", "positive"),
    ("Super helpful customer support agent resolved my issue immediately.", "positive"),
    ("Great value for money. Discount codes worked flawlessly.", "positive"),
    
    # Negative Reviews
    ("Terrible quality. The zipper broke on the first day of use.", "negative"),
    ("Horrible customer service. Nobody answered my phone call or email.", "negative"),
    ("Extremely disappointed. Package arrived damaged and missing items.", "negative"),
    ("Size chart is completely wrong. Shoes were way too small and uncomfortable.", "negative"),
    ("Waste of money. Product stopped working after two hours.", "negative"),
    ("Very slow delivery! Took three weeks to arrive with no tracking info.", "negative"),
    ("Return process was a nightmare and they charged hidden restocking fees.", "negative"),
    ("Cheap materials and poor craftsmanship. Do not buy this item.", "negative"),
    ("Item description was misleading and fake. Very angry customer.", "negative"),
    ("Received wrong color and bad quality fabric. Requesting a full refund.", "negative"),
    
    # Neutral Reviews
    ("The product is average. Works okay, nothing extraordinary.", "neutral"),
    ("Received item on time. Standard quality for the price paid.", "neutral"),
    ("Packaging was fine. Sizing is okay but color is slightly different.", "neutral"),
    ("Standard delivery timeframe. The product meets basic specifications.", "neutral"),
    ("Decent customer service. The issue was handled eventually.", "neutral"),
    ("Product functions as described in the manual.", "neutral"),
]


def train_sentiment_model():
    """Builds, fits, and exports TF-IDF + Logistic Regression sentiment model."""
    print("Preprocessing review texts...")
    raw_texts, labels = zip(*SAMPLE_REVIEWS)
    processed_texts = [preprocess_text(txt) for txt in raw_texts]

    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X_train = vectorizer.fit_transform(processed_texts)

    print("Fitting Logistic Regression Classifier...")
    model = LogisticRegression(C=1.0, max_iter=200)
    model.fit(X_train, labels)

    # Ensure output directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving vectorizer to {VECTORIZER_PATH}...")
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saving sentiment model to {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Sentiment analysis model and vectorizer saved successfully!")


if __name__ == "__main__":
    train_sentiment_model()
