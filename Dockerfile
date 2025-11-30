# Use Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install FFmpeg (Crucial for audio processing)
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create the static directory if it doesn't exist
RUN mkdir -p static

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run the application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]