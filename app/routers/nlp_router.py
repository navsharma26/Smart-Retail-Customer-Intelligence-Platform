"""
Natural Language Processing Router Endpoints.
"""

from fastapi import APIRouter, HTTPException, status

from app.schemas import (
    PreprocessTextRequest,
    PreprocessTextResponse,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
)
from services.nlp_service import get_sentiment_analyzer_service, preprocess_text

router = APIRouter(prefix="/nlp", tags=["Natural Language Processing Services"])


@router.post(
    "/preprocess",
    response_model=PreprocessTextResponse,
    status_code=status.HTTP_200_OK,
    summary="Clean and preprocess text (Lowercasing, punctuation/stopword removal, lemmatization)",
)
async def preprocess_text_endpoint(payload: PreprocessTextRequest):
    """Cleans input text string."""
    try:
        clean = preprocess_text(payload.text)
        return PreprocessTextResponse(
            status="success", raw_text=payload.text, processed_text=clean
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing error: {str(e)}",
        )


@router.post(
    "/analyze-sentiment",
    response_model=SentimentAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze sentiment of product reviews using TF-IDF + Logistic Regression",
)
async def analyze_sentiment_endpoint(payload: SentimentAnalysisRequest):
    """Predicts sentiment ('positive', 'negative', 'neutral') and returns confidence probabilities."""
    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text field cannot be empty.",
        )

    try:
        service = get_sentiment_analyzer_service()
        res = service.analyze_sentiment(payload.text)

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
            detail=f"Sentiment analysis error: {str(e)}",
        )
