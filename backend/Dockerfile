FROM python:3.13-slim AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        git \
        curl \
        ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS final

RUN mkdir -p /app/files/docs \
    /app/files/hf_cache \
    /app/files/qdrant_storage

ENV HF_HOME=/app/files/hf_cache
ENV TRANSFORMERS_CACHE=/app/files/hf_cache
ENV HF_DATASETS_CACHE=/app/files/hf_cache
ENV COQUI_TOS_AGREED=1

EXPOSE 8000

COPY . .

CMD ["python", "manage.py", "runserver", "--noreload", "0.0.0.0:8000"]
