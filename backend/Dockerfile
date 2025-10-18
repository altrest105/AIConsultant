FROM python:3.13

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# создаём изолированное окружение
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
COPY . .

CMD ["python", "manage.py", "runserver", "--noreload", "0.0.0.0:8000"]
