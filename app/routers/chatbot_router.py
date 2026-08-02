"""
Conversational Intelligence Chatbot Router Endpoints.
"""

from typing import List
from fastapi import APIRouter, HTTPException, status

from app.schemas import (
    ChatbotQueryRequest,
    ChatbotQueryResponse,
    IntentSummary,
)
from services.chatbot_service import get_chatbot_service

router = APIRouter(prefix="/chatbot", tags=["Conversational Intelligence Services"])


@router.post(
    "/query",
    response_model=ChatbotQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Hybrid Chatbot (Rule-Based Keyword Match with TF-IDF Fallback)",
)
async def chatbot_query(payload: ChatbotQueryRequest):
    """Processes customer query using rule-based exact keyword matching and TF-IDF similarity fallback."""
    if not payload.query or not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query field cannot be empty.",
        )

    try:
        service = get_chatbot_service()
        res = service.get_response(payload.query)

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
            detail=f"Chatbot execution error: {str(e)}",
        )


@router.get(
    "/intents",
    response_model=List[IntentSummary],
    status_code=status.HTTP_200_OK,
    summary="List all 20 retail FAQ intent categories in data/intents.json",
)
async def list_intents():
    """Returns summary list of all supported retail FAQ intents."""
    service = get_chatbot_service()
    return service.get_all_intents()
