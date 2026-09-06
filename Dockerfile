# =========================================================
# BhoomiMitra AI — Production / Staging Docker Container
# =========================================================
FROM python:3.11-slim

# Set environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    APP_ENV=staging

WORKDIR /app

# Install system dependencies (ffmpeg for audio conversion, libpq for postgres, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and assets
COPY . .

# Ensure entrypoint script is executable
RUN chmod +x /app/scripts/start_staging.sh || true

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command: run migrations and start gunicorn with uvicorn workers
CMD ["sh", "-c", "alembic upgrade head && gunicorn -w 2 -k uvicorn.workers.UvicornWorker --timeout 90 -b 0.0.0.0:${PORT} src.main:app"]
