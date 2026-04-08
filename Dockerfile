FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY env/ ./env/
COPY sample_data.py .
COPY openenv.yaml .
COPY app.py .
COPY inference.py .
COPY pyproject.toml .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 7860

# Set environment
ENV PYTHONPATH=/app

# ✅ FIXED health check (IMPORTANT)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run FastAPI server
CMD ["python", "app.py"]