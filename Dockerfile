# --- Stage 1: Builder ---
FROM python:3.9-slim-bullseye AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libatlas-base-dev \
      libprotobuf-dev \
      protobuf-compiler \
      gdal-bin \
      libgdal-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

WORKDIR /app
COPY requirements.txt .

RUN --mount=type=cache,id=pipcache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.9-slim-bullseye

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      gnupg \
      ca-certificates \
      ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Coral TPU runtime
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/coral.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/coral.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" > /etc/apt/sources.list.d/coral-edgetpu.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends libedgetpu1-std && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

COPY --from=builder /install /usr/local

# Copy app and model
WORKDIR /app
COPY app.py camera.py draw.py ./
COPY models/model_edgetpu.tflite ./models/model_edgetpu.tflite

EXPOSE 80

CMD ["python", "app.py"]
