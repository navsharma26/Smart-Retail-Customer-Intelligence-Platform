"""
Pydantic Data Schemas for API Requests & Responses.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --- Computer Vision Schemas ---
class BoundingBox(BaseModel):
    """Face bounding box coordinates schema."""

    x: int = Field(..., description="Top-left X coordinate")
    y: int = Field(..., description="Top-left Y coordinate")
    w: int = Field(..., description="Width of bounding box")
    h: int = Field(..., description="Height of bounding box")


class DimensionSchema(BaseModel):
    """Image dimensions schema."""

    width: int
    height: int


class FaceDetectionResponse(BaseModel):
    """Response schema for face detection endpoint."""

    status: str = "success"
    faces_count: int
    bounding_boxes: List[BoundingBox]
    image_dimensions: DimensionSchema


class CVPipelineResponse(BaseModel):
    """Response schema for complete computer vision pipeline."""

    status: str = "success"
    original_dimensions: DimensionSchema
    processed_dimensions: DimensionSchema
    faces_detected_count: int
    bounding_boxes: List[BoundingBox]
    gray_image_base64: Optional[str] = None
    edges_image_base64: Optional[str] = None


class ProductClassificationResponse(BaseModel):
    """Response schema for MobileNetV2 Product Image Classification."""

    status: str = "success"
    predicted_category: str = Field(..., description="Predicted category ('shoes', 'bags', 'electronics', 'clothing', 'groceries')")
    confidence_score: float = Field(..., description="Prediction confidence float (0.0 to 1.0)")
    confidence_percentage: float = Field(..., description="Prediction confidence percentage (0% to 100%)")
    class_probabilities: Dict[str, float] = Field(..., description="Probability distribution across all 5 classes")


class CustomerFaceRecognitionResponse(BaseModel):
    """Response schema for facial customer identification and visit logging."""

    status: str = "success"
    matched: bool
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    confidence_score: float = 0.0
    distance: Optional[float] = None
    total_visits: Optional[int] = None
    last_visit: Optional[str] = None
    visit_history: Optional[List[str]] = None
    message: Optional[str] = None


class CustomerRegistrationResponse(BaseModel):
    """Response schema for customer facial registration."""

    status: str = "success"
    message: str
    customer_id: str
    name: str
    registered_at: str


class CustomerProfile(BaseModel):
    """Customer profile item schema."""

    customer_id: str
    name: str
    registered_at: Optional[str] = None
    total_visits: int
    last_visit: Optional[str] = None


# --- NLP & Sentiment Analysis Schemas ---
class PreprocessTextRequest(BaseModel):
    """Request payload for text preprocessing."""

    text: str = Field(..., example="I LOVED these shoes! Very fast delivery.")


class PreprocessTextResponse(BaseModel):
    """Response payload for preprocessed text."""

    status: str = "success"
    raw_text: str
    processed_text: str


class SentimentAnalysisRequest(BaseModel):
    """Request payload for Sentiment Analysis."""

    text: str = Field(..., example="Exceptional product quality and super fast shipping!")


class SentimentAnalysisResponse(BaseModel):
    """Response payload for Sentiment Analysis."""

    status: str = "success"
    raw_text: str
    clean_text: str
    sentiment: str = Field(..., description="'positive', 'negative', or 'neutral'")
    confidence_score: float = Field(..., description="Confidence float 0.0 - 1.0")
    class_probabilities: Dict[str, float]


# --- Hybrid Chatbot Schemas ---
class ChatbotQueryRequest(BaseModel):
    """Request payload for Chatbot queries."""

    query: str = Field(..., example="What is your return policy for shoes?")
    customer_id: Optional[str] = Field("guest", description="Optional customer ID identifier")


class ChatbotQueryResponse(BaseModel):
    """Response payload for Hybrid Chatbot interaction."""

    status: str = "success"
    query: str
    matched_intent: str
    matching_method: str = Field(..., description="'rule_based', 'tfidf_fallback', or 'fallback_default'")
    confidence: float
    response: str


class IntentSummary(BaseModel):
    """FAQ Intent category schema."""

    tag: str
    keywords: List[str]
    sample_patterns_count: int


# --- Dashboard Analytics Schemas ---
class CustomerAnalyticsSchema(BaseModel):
    total_registered_customers: int
    total_logged_visits: int
    recent_customer_activity: List[CustomerProfile]


class SentimentTelemetrySchema(BaseModel):
    total_analyzed_reviews: int
    counts: Dict[str, int]
    percentages: Dict[str, float]


class DashboardStatsResponse(BaseModel):
    """Response schema for GET /dashboard/stats endpoint."""

    status: str = "success"
    timestamp: str
    customer_analytics: CustomerAnalyticsSchema
    sentiment_telemetry: SentimentTelemetrySchema
    models_loaded: Dict[str, str]


# --- System Schemas ---
class HealthCheckResponse(BaseModel):
    """API Health Check response schema."""

    status: str = "healthy"
    version: str = "1.0.0"
    app_name: str = "Smart Retail & Customer Intelligence Platform API"
