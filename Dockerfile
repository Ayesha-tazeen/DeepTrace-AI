# DeepTrace AI - Production Docker Container
FROM python:3.11-slim

WORKDIR /app

# Install system runtime dependencies for OpenCV, audio processing and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . /app

# Ensure model and data storage directories exist
RUN mkdir -p /app/data/raw /app/data/processed /app/data/temp /app/logs /app/models

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Launch the unified application
CMD ["python", "app.py"]
