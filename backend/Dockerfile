FROM nvidia/cuda:12.9.1-base-ubuntu20.04 AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-distutils \
    python3-pip \
    build-essential \
    git \
    curl \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3 1

RUN pip install --no-cache-dir --upgrade pip
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

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