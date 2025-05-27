# Use official Python slim image
FROM python:3.10-slim

# Set environment variables early
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static/plots fastf1_cache

# Environment variables for Flask
ENV FLASK_ENV=production \
    FLASK_APP=app.py \
    PORT=5000

# Expose Flask port
EXPOSE 5000

# Run app using gunicorn (production-grade WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--access-logfile", "-", "--error-logfile", "-", "app:app"]