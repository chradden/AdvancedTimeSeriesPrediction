FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (including curl for healthcheck)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy API requirements (minimal for production)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API (simple version without torch)
CMD ["python", "api_simple.py"]
