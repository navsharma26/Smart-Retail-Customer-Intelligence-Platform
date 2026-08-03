FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime system libraries for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download required NLTK resources
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" || true

# Copy application source code and pre-trained models
COPY models ./models
COPY data ./data
COPY app ./app
COPY services ./services
COPY notebooks ./notebooks
COPY tests ./tests
COPY vercel.json .
COPY README.md .

EXPOSE 8000

# Start Uvicorn app server
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
