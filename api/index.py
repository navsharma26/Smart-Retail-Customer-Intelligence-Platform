"""
Vercel Serverless Function entrypoint for FastAPI application.
"""

from app.main import app

# Export FastAPI app instance for Vercel WSGI/ASGI handler
app = app
