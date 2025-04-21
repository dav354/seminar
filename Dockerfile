FROM --platform=linux/arm64 python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gnupg \
      python3-opencv \
 && echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
      > /etc/apt/sources.list.d/coral-edgetpu.list \
 && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | apt-key add - \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      libedgetpu1-std \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY recognizer.py .

CMD ["python", "recognizer.py"]
