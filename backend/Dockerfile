FROM python:3.13

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
RUN pip install --no-cache-dir django djangorestframework django-cors-headers coqui-tts \
    faster-whisper langchain langchain-community langchain-core pydub \
    qdrant-client rank-bm25 sentence-transformers python-docx tqdm

EXPOSE 8000
COPY . .

CMD ["python", "manage.py", "runserver", "--noreload", "0.0.0.0:8000"]
