FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS cuda_libs

FROM python:3.13-slim AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ffmpeg \
        wget \
        apt-transport-https \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=cuda_libs /usr/local/cuda/lib64/libc* /usr/local/lib/
COPY --from=cuda_libs /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/lib/
RUN ldconfig

RUN pip install --no-cache-dir --upgrade pip

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

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

EXPOSE 8000

COPY . .

CMD ["python", "manage.py", "runserver", "--noreload", "0.0.0.0:8000"]