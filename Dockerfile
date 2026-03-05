# Use slim Python image
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for LightGBM and NLTK
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Expose Flask port
EXPOSE 8080

# Run the Flask app
CMD ["python3", "flask_app/app.py"]