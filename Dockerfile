# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and YOLOv8
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Set the environment variable so Streamlit knows where the backend is
# When running in the same container, localhost works
ENV API_URL="http://localhost:8000/api/v1"

# Make entrypoint executable just in case
RUN chmod +x /app/entrypoint.sh

# Run both services
CMD ["/app/entrypoint.sh"]
