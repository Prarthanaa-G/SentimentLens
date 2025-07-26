
# Use lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies required for curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Download large model from external source into Artifacts directory
# Replace the URL below with your actual model link
# Download large model from Hugging Face during build
RUN mkdir -p Artifacts && \
    curl -L "https://huggingface.co/Prarthana1/sentiment-analysis-model/resolve/main/model.safetensors" -o Artifacts/model.safetensors

RUN mkdir Logs
# Copy the rest of your codebase into the container
COPY . .

# Set environment variables inside Docker
ENV ARTIFACTS_DIR=Artifacts
ENV DATA_DIR=Data
ENV LOGS_DIR=Logs

# Expose Streamlit default port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "Scripts/app.py"]

