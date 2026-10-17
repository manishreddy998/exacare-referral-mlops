FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app/src:/app \
    PYTHONUNBUFFERED=1 \
    OMP_THREAD_LIMIT=1 \
    DAGSTER_HOME=/app/.dagster_home
RUN mkdir -p /app/.dagster_home && touch /app/.dagster_home/dagster.yaml
