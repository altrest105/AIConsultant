FROM python:3.13-slim AS base

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git ffmpeg build-essential ca-certificates \
 && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends cuda-toolkit-13-0 libcudnn9-cuda-13 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir nvidia-cudnn-cu13

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu130

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

ENV PATH="/opt/venv/bin:$PATH"

CMD ["python", "manage.py", "runserver", "--noreload", "0.0.0.0:8000"]
