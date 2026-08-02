"""
FastAPI Routers package.
"""

from .cv_router import router as cv_router
from .nlp_router import router as nlp_router
from .chatbot_router import router as chatbot_router

__all__ = ["cv_router", "nlp_router", "chatbot_router"]
