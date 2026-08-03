"""
NLP Service Module for Smart Retail & Customer Intelligence Platform.

Contains:
1. TextPreprocessor: lowercasing, punctuation cleaning, stopword filtering, and lemmatization.
2. SentimentAnalyzerService: TF-IDF + Logistic Regression model loading, text vectorization, and sentiment prediction.
"""

import os
from pathlib import Path
import pickle
import re
import string
from typing import Any, Dict, List, Optional, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
SENTIMENT_MODEL_PATH = DEFAULT_MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = DEFAULT_MODEL_DIR / "vectorizer.pkl"

# Custom English Stopwords
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

# Rule-based lemmatization mapping for retail reviews
LEMMA_MAPPING = {
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
    "running": "run", "runs": "run", "ran": "run",
}


class TextPreprocessor:
    """Text preprocessing engine."""

    def __init__(self, stop_words: Optional[set] = None):
        self.stop_words = stop_words if stop_words is not None else STOP_WORDS

    def preprocess(self, text: str) -> str:
        """
        Execute text preprocessing pipeline:
        1. Lowercasing & trimming
        2. Punctuation removal
        3. Stopword removal
        4. Lemmatization
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Lowercase
        text = text.lower().strip()

        # 2. Punctuation & Non-alphanumeric cleaning
        text = re.sub(r"[^\w\s]", " ", text)

        # 3. Tokenize & Stopword filtering
        tokens = text.split()
        filtered = [w for w in tokens if w not in self.stop_words and len(w) > 1]

        # 4. Lemmatization
        lemmatized = [LEMMA_MAPPING.get(w, w) for w in filtered]

        return " ".join(lemmatized)


def preprocess_text(text: str) -> str:
    """Convenience function for text preprocessing."""
    processor = TextPreprocessor()
    return processor.preprocess(text)


class SentimentAnalyzerService:
    """
    Service for product review sentiment classification using TF-IDF + Logistic Regression.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        vectorizer_path: Optional[Union[str, Path]] = None,
    ):
        self.model_path = Path(model_path) if model_path else SENTIMENT_MODEL_PATH
        self.vectorizer_path = Path(vectorizer_path) if vectorizer_path else VECTORIZER_PATH
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.vectorizer = None
        self.load_or_train_model()

    def load_or_train_model(self):
        """Load stored sentiment model & vectorizer PKL files or train fresh instance."""
        if self.model_path.exists() and self.vectorizer_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                with open(self.vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
                print(f"[SentimentAnalyzer] Loaded model from {self.model_path}")
                return
            except Exception as e:
                print(f"[SentimentAnalyzer] Error loading PKL artifacts: {e}. Re-building model...")

        self._train_and_save_default_model()

    def _train_and_save_default_model(self):
        """Train default TF-IDF + Logistic Regression model on sample reviews."""
        import importlib
        train_mod = importlib.import_module("notebooks.02_train_sentiment_model")
        train_mod.train_sentiment_model()
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of input text.

        :param text: Raw review or customer feedback text string.
        :return: Sentiment result dictionary (sentiment, confidence_score, probabilities).
        """
        clean_text = self.preprocessor.preprocess(text)
        if not clean_text:
            return {
                "raw_text": text,
                "clean_text": "",
                "sentiment": "neutral",
                "confidence_score": 0.5,
                "class_probabilities": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
            }

        X_tfidf = self.vectorizer.transform([clean_text])
        probs = self.model.predict_proba(X_tfidf)[0]
        classes = self.model.classes_

        top_idx = int(np_argmax(probs))
        predicted_sentiment = str(classes[top_idx])
        confidence_score = float(probs[top_idx])

        probabilities = {str(c): float(probs[i]) for i, c in enumerate(classes)}

        return {
            "raw_text": text,
            "clean_text": clean_text,
            "sentiment": predicted_sentiment,
            "confidence_score": round(confidence_score, 4),
            "class_probabilities": probabilities,
        }


def np_argmax(arr):
    """Helper argmax function."""
    return max(range(len(arr)), key=lambda i: arr[i])


# Singleton instance
_sentiment_service: Optional[SentimentAnalyzerService] = None


def get_sentiment_analyzer_service() -> SentimentAnalyzerService:
    global _sentiment_service
    if _sentiment_service is None:
        _sentiment_service = SentimentAnalyzerService()
    return _sentiment_service
