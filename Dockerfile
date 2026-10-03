# Use a lightweight python base
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest
COPY requirements.txt .

# Install python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and streamlit codebase
COPY best_pneumonia_vgg16_model.keras .
COPY app.py .

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to verify the web service is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Boot command
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
