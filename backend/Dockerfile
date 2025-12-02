FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for librosa (audio processing)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Copy models directory (explicitly, though COPY . . covers it if it's in the dir)
# But good to be explicit or ensure it's there.
# Since we moved models into backend/, COPY . . will include it.
# No extra command needed if models is inside backend/ and we COPY . .
# However, let's just ensure we don't exclude it in .dockerignore if it exists.

# Expose the port
EXPOSE 8000

# Command to run the application with hot reloading
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
