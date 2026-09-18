# ==========================================================
# Oral Cancer (OSCC-CNN) Web Application Dockerfile
# ==========================================================
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5001 \
    FLASK_DEBUG=0

WORKDIR /app

# Install python dependencies first to take advantage of Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --progress-bar on -r requirements.txt

# Copy application source code
COPY . .

# Expose the application port
EXPOSE 5001

# Run with Gunicorn in production mode
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "--timeout", "120", "app:app"]
