FROM --platform=linux/arm64 python:3.11-slim

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gnupg \
      python3-opencv \
      build-essential \
      python3-dev \
      libatlas-base-dev \
      libprotobuf-dev \
      protobuf-compiler \
      ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY recognizer.py .
CMD ["python3", "recognizer.py"]
