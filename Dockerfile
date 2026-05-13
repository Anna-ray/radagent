# RadAgent v2 — Vultr Deployment Dockerfile
# Author: Rayane Aggoune
# 
# CPU-only container for public demo deployment
# Heavy inference (specialist, federation) runs on desktop
# This serves cached results + light routing/autonomy/dictation

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY radagent/ ./radagent/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Copy pre-cached demo results (created by deployment script)
COPY runs/ ./runs/

# Create directories for uploads and temp files
RUN mkdir -p /app/uploads /app/temp

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run FastAPI server
CMD ["uvicorn", "radagent.app.server:app", "--host", "0.0.0.0", "--port", "8000"]

# Made with Bob
