# Use Python 3.10 slim base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV (libgl1-mesa-glx, libglib2.0-0) and C++ compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download default spaCy model and NLTK resources
RUN python -m spacy download en_core_web_sm || true && \
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" || true

# Copy models, data, app, services, notebooks, and tests
COPY models ./models
COPY data ./data
COPY app ./app
COPY services ./services
COPY notebooks ./notebooks
COPY tests ./tests
COPY README.md .

# Expose FastAPI Uvicorn port
EXPOSE 8000

# Entrypoint to launch FastAPI application via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
