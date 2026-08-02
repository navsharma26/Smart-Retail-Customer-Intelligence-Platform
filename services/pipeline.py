"""
Unified Machine Learning & Customer Intelligence Pipeline.

Loads all saved models into memory once on application startup:
- Product Image Classifier (MobileNetV2: product_classifier.h5)
- Face Recognition & Visit Database (face_db.pkl)
- NLP Sentiment Analyzer (sentiment_model.pkl & vectorizer.pkl)
- Hybrid Chatbot (data/intents.json)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np

from services.chatbot_service import HybridChatbotService, get_chatbot_service
from services.cv_service import (
    FaceRecognitionDBService,
    ProductClassifierService,
    get_face_recognition_db_service,
    get_product_classifier_service,
)
from services.cv_utils import CVProcessor
from services.nlp_service import SentimentAnalyzerService, get_sentiment_analyzer_service


class UnifiedPlatformPipeline:
    """
    Unified ML & Customer Intelligence Pipeline Manager.
    Loads and manages all model instances in memory.
    """

    def __init__(self):
        print("================================================================")
        print("Initializing Unified Smart Retail ML Pipeline...")
        print("================================================================")
        self.cv_processor = CVProcessor()
        self.product_classifier = get_product_classifier_service()
        self.face_db_service = get_face_recognition_db_service()
        self.sentiment_service = get_sentiment_analyzer_service()
        self.chatbot_service = get_chatbot_service()

        # In-memory session telemetry for dashboard analytics
        self.sentiment_history: List[Dict[str, Any]] = []

    def recognize_face(
        self, input_data: Union[str, Path, bytes, np.ndarray], tolerance: float = 0.6
    ) -> Dict[str, Any]:
        """
        Ingest facial image, compare against customer database, and log visit.

        :param input_data: Image file path, raw bytes, or NumPy matrix.
        :param tolerance: Distance threshold for positive match.
        :return: Customer identification and visit log outcome.
        """
        image_bgr = self.cv_processor.load_image(input_data)
        res = self.face_db_service.identify_customer(image_bgr, tolerance=tolerance)
        return res

    def classify_product(
        self, input_data: Union[str, Path, bytes, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Classify product image using MobileNetV2.

        :param input_data: Image file path, raw bytes, or NumPy matrix.
        :return: Product category and confidence predictions.
        """
        image_bgr = self.cv_processor.load_image(input_data)
        res = self.product_classifier.classify(image_bgr)
        return res

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of customer feedback or review.

        :param text: Customer text string.
        :return: Sentiment prediction and confidence score.
        """
        res = self.sentiment_service.analyze_sentiment(text)
        # Log to in-memory sentiment history for real-time telemetry
        self.sentiment_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sentiment": res["sentiment"],
                "confidence_score": res["confidence_score"],
            }
        )
        return res

    def chatbot_query(self, query: str) -> Dict[str, Any]:
        """
        Process user question with Hybrid Chatbot (Rule matching + TF-IDF fallback).

        :param query: Customer question string.
        :return: Bot answer and matching telemetry.
        """
        return self.chatbot_service.get_response(query)

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """
        Aggregate system-wide real-time metrics:
        - Total registered customers
        - Total logged visits across all customers
        - Recent visit activities stream
        - Sentiment distribution breakdown (positive, negative, neutral counts & percentages)
        - System model health indicators
        """
        all_customers = self.face_db_service.get_all_customers()
        total_customers = len(all_customers)
        total_visits = sum(c["total_visits"] for c in all_customers)

        recent_visits = sorted(
            [c for c in all_customers if c.get("last_visit")],
            key=lambda x: x["last_visit"],
            reverse=True,
        )[:10]

        # Calculate sentiment trends
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for item in self.sentiment_history:
            s = item.get("sentiment", "neutral")
            sentiment_counts[s] = sentiment_counts.get(s, 0) + 1

        total_sentiments = len(self.sentiment_history)
        if total_sentiments > 0:
            sentiment_percentages = {
                k: round((v / total_sentiments) * 100, 2)
                for k, v in sentiment_counts.items()
            }
        else:
            sentiment_percentages = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "customer_analytics": {
                "total_registered_customers": total_customers,
                "total_logged_visits": total_visits,
                "recent_customer_activity": recent_visits,
            },
            "sentiment_telemetry": {
                "total_analyzed_reviews": total_sentiments,
                "counts": sentiment_counts,
                "percentages": sentiment_percentages,
            },
            "models_loaded": {
                "product_classifier": "MobileNetV2 (models/product_classifier.h5)",
                "face_recognition_db": "FaceDB PKL (models/face_db.pkl)",
                "sentiment_analyzer": "TF-IDF + LogisticRegression (models/sentiment_model.pkl)",
                "hybrid_chatbot": "Rule-Based + TF-IDF (data/intents.json)",
            },
        }


# Global singleton instance
_unified_pipeline: Optional[UnifiedPlatformPipeline] = None


def get_unified_pipeline() -> UnifiedPlatformPipeline:
    """Retrieve singleton instance of UnifiedPlatformPipeline."""
    global _unified_pipeline
    if _unified_pipeline is None:
        _unified_pipeline = UnifiedPlatformPipeline()
    return _unified_pipeline
