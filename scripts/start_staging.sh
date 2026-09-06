#!/bin/sh
set -e

echo "========================================================"
echo " Starting BhoomiMitra AI Staging Server"
echo "========================================================"

echo "[1/2] Running database migrations..."
alembic upgrade head

echo "[2/2] Launching Gunicorn ASGI server..."
exec gunicorn -w 2 -k uvicorn.workers.UvicornWorker --timeout 90 -b "0.0.0.0:${PORT:-8000}" src.main:app
