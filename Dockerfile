# Base image with Python 3.10
FROM python:3.10-slim

# Install system dependencies required for Java, OCR (Tesseract), and PDF processing (Poppler)
RUN apt-get update && apt-get install -y \
    default-jre \
    maven \
    tesseract-ocr \
    poppler-utils \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt .
COPY pom.xml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Download spacy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application
COPY . .

# Ensure upload, data, and report directories exist
RUN mkdir -p /app/uploads /app/data/processed /app/reports

# Initialize the SQLite Database
RUN python src/python/db.py

# Expose the port the app runs on
EXPOSE 5000

# Set environment variables for production
ENV FLASK_APP=src/python/app.py
ENV FLASK_ENV=production

# Run the application using gunicorn for production stability
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "src.python.app:app"]
