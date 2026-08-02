"""
FastAPI Application Entry Point for Smart Retail & Customer Intelligence Platform.

Provides unified endpoints and loads ML models into memory once on application startup.
"""

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chatbot_router, cv_router, nlp_router
from app.schemas import (
    ChatbotQueryRequest,
    ChatbotQueryResponse,
    CustomerFaceRecognitionResponse,
    DashboardStatsResponse,
    HealthCheckResponse,
    ProductClassificationResponse,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
)
from services.pipeline import get_unified_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager.
    Pre-loads all Machine Learning models into memory on application startup.
    """
    print("==================================================================")
    print("Pre-loading Machine Learning models into memory (MobileNetV2, FaceDB, TF-IDF Sentiment, Hybrid Chatbot)...")
    pipeline = get_unified_pipeline()
    print("All ML models pre-loaded and ready for inference!")
    print("==================================================================")
    yield
    print("Shutting down Smart Retail & Customer Intelligence Platform API...")


app = FastAPI(
    title="Smart Retail & Customer Intelligence Platform API",
    description=(
        "Enterprise Unified AI Backend integrating Computer Vision (Face Recognition & MobileNetV2 Product Classification), "
        "Natural Language Processing (Sentiment Analysis), and Conversational AI (Hybrid FAQ Chatbot)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register v1 Modular Routers
app.include_router(cv_router, prefix="/api/v1")
app.include_router(nlp_router, prefix="/api/v1")
app.include_router(chatbot_router, prefix="/api/v1")


# --- Top-Level Unified API Endpoints ---

@app.post(
    "/recognize-face",
    response_model=CustomerFaceRecognitionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Core Unified Services"],
    summary="Recognize face image, identify customer ID, and log visit timestamp",
)
async def recognize_face_endpoint(
    file: UploadFile = File(..., description="Facial image file"),
    tolerance: float = Query(
        0.6, ge=0.1, le=1.5, description="Face match distance threshold"
    ),
):
    """Accepts an uploaded facial image, identifies customer against face_db.pkl, and logs visit timestamp."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image.",
        )

    try:
        contents = await file.read()
        pipeline = get_unified_pipeline()
        res = pipeline.recognize_face(contents, tolerance=tolerance)

        return CustomerFaceRecognitionResponse(
            status="success",
            matched=res["matched"],
            customer_id=res.get("customer_id"),
            customer_name=res.get("customer_name"),
            confidence_score=res.get("confidence_score", 0.0),
            distance=res.get("distance"),
            total_visits=res.get("total_visits"),
            last_visit=res.get("last_visit"),
            visit_history=res.get("visit_history"),
            message=res.get("message"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face recognition failure: {str(e)}",
        )


@app.post(
    "/classify-product",
    response_model=ProductClassificationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Core Unified Services"],
    summary="Classify retail product image into 5 categories using MobileNetV2",
)
async def classify_product_endpoint(
    file: UploadFile = File(..., description="Product image file")
):
    """Accepts a product image file and predicts category ('shoes', 'bags', 'electronics', 'clothing', 'groceries')."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image.",
        )

    try:
        contents = await file.read()
        pipeline = get_unified_pipeline()
        res = pipeline.classify_product(contents)

        return ProductClassificationResponse(
            status="success",
            predicted_category=res["predicted_category"],
            confidence_score=res["confidence_score"],
            confidence_percentage=res["confidence_percentage"],
            class_probabilities=res["class_probabilities"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Product classification failure: {str(e)}",
        )


@app.post(
    "/analyze-sentiment",
    response_model=SentimentAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Core Unified Services"],
    summary="Analyze sentiment of customer product review text",
)
async def analyze_sentiment_endpoint(payload: SentimentAnalysisRequest):
    """Accepts text payload and predicts sentiment label ('positive', 'negative', 'neutral') and score."""
    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text field cannot be empty.",
        )

    try:
        pipeline = get_unified_pipeline()
        res = pipeline.analyze_sentiment(payload.text)

        return SentimentAnalysisResponse(
            status="success",
            raw_text=res["raw_text"],
            clean_text=res["clean_text"],
            sentiment=res["sentiment"],
            confidence_score=res["confidence_score"],
            class_probabilities=res["class_probabilities"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failure: {str(e)}",
        )


@app.post(
    "/chatbot",
    response_model=ChatbotQueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Core Unified Services"],
    summary="Query Hybrid Retail Chatbot",
)
async def chatbot_endpoint(payload: ChatbotQueryRequest):
    """Accepts a customer message and returns bot answer via rule-based matching with TF-IDF fallback."""
    if not payload.query or not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query message cannot be empty.",
        )

    try:
        pipeline = get_unified_pipeline()
        res = pipeline.chatbot_query(payload.query)

        return ChatbotQueryResponse(
            status="success",
            query=res["query"],
            matched_intent=res["matched_intent"],
            matching_method=res["matching_method"],
            confidence=res["confidence"],
            response=res["response"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chatbot execution failure: {str(e)}",
        )


@app.get(
    "/dashboard/stats",
    response_model=DashboardStatsResponse,
    status_code=status.HTTP_200_OK,
    tags=["Core Unified Services"],
    summary="Retrieve aggregated JSON stats for customer visits, sentiment trends, and model status",
)
async def dashboard_stats_endpoint():
    """Returns aggregated real-time metrics for retail customer visits, sentiment trends, and model health."""
    try:
        pipeline = get_unified_pipeline()
        stats = pipeline.get_dashboard_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dashboard stats retrieval failure: {str(e)}",
        )


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="Application Health Check",
)
async def health_check():
    """Returns operational status of the API."""
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        app_name="Smart Retail & Customer Intelligence Platform API",
    )


@app.get("/", tags=["System"])
async def root():
    """Root endpoint hint."""
    return {
        "message": "Welcome to Smart Retail & Customer Intelligence Platform API",
        "documentation": "/docs",
        "health_check": "/health",
        "unified_endpoints": [
            "/recognize-face",
            "/classify-product",
            "/analyze-sentiment",
            "/chatbot",
            "/dashboard/stats",
        ],
    }
