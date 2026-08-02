"""
Services package for Smart Retail & Customer Intelligence Platform.
"""

from .cv_utils import CVProcessor, process_image_pipeline
from .cv_service import (
    ProductClassifierService,
    FaceRecognitionDBService,
    get_product_classifier_service,
    get_face_recognition_db_service,
)
from .nlp_service import (
    TextPreprocessor,
    SentimentAnalyzerService,
    preprocess_text,
    get_sentiment_analyzer_service,
)
from .chatbot_service import (
    HybridChatbotService,
    get_chatbot_service,
)
from .pipeline import (
    UnifiedPlatformPipeline,
    get_unified_pipeline,
)

__all__ = [
    "CVProcessor",
    "process_image_pipeline",
    "ProductClassifierService",
    "FaceRecognitionDBService",
    "get_product_classifier_service",
    "get_face_recognition_db_service",
    "TextPreprocessor",
    "SentimentAnalyzerService",
    "preprocess_text",
    "get_sentiment_analyzer_service",
    "HybridChatbotService",
    "get_chatbot_service",
    "UnifiedPlatformPipeline",
    "get_unified_pipeline",
]
