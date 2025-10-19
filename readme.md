# AI Consultant

Интеллектуальная консультационная система на базе RAG (Retrieval-Augmented Generation) для работы по предоставленной базе знаний.

## 📋 Описание проекта

AI Consultant - это веб-приложение, которое использует технологии обработки естественного языка и векторный поиск для предоставления точных ответов на вопросы. Система построена на основе Django backend и Vue.js frontend, с интеграцией speech-to-text и text-to-speech функционала.

## 🏗️ Архитектура

Проект состоит из следующих компонентов:

- **Backend**: Django REST API
- **Frontend**: Vue.js
- **Docker**: Контейнеризация приложения

## 📁 Структура проекта

```
.
├── backend/             # Django backend приложение
│   ├── backend/         # Основные настройки Django
│   ├── qa/              # Модуль вопросов-ответов (RAG)
│   ├── stt/             # Модуль распознавания речи
│   ├── tts/             # Модуль синтеза речи
│   ├── files/           # Файловое хранилище
│   └── requirements.txt 
├── frontend/            # Vue.js frontend приложение
│   ├── src/             # Исходный код
│   ├── public/          # Статические файлы
│   └── package.json
├── benchmark/           # Бенчмарки и тестовые данные
│   ├── benchmark.json   # Итоговый бенчмарк
│   ├── chunks.json      # Чанки документов
│   └── main.py          # Скрипт бенчмаркинга
├── readme.md            # Описание проекта
└── docker-compose.yml   # Docker Compose конфигурация
```

## 🚀 Быстрый старт

### Оборудование

Программа тестировалась на следующем аппаратном обеспечении:
- NVIDIA RTX 4070 Ti (12 ГБ VRAM)
- 32 ГБ RAM
- AMD Ryzen 9 9900X
- 50 GB свободного места

### Запуск с Docker

Необходимо [установить Docker](https://www.docker.com/) и запустить его, после этого вписывать команды
```bash
# Клонировать репозиторий
git clone https://github.com/altrest105/AIConsultant
cd AIConsultant

# Запустить все сервисы
docker-compose up -d
```

Приложение будет доступно по адресу:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000

### Запуск для разработки

#### Backend

```bash
cd backend

# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установить зависимости
pip install -r requirements.txt

# Применить миграции
python manage.py migrate

# Запустить сервер разработки
python manage.py runserver --noreload
```

#### Frontend

```bash
cd frontend

# Установить зависимости
npm install

# Запустить dev-сервер
npm run dev
```

## 🔧 Основные модули

### QA (Question-Answering)

Модуль реализует RAG-систему для ответов на вопросы:
- Векторный поиск по документам
- Генерация ответов на основе контекста
- Ранжирование релевантных фрагментов

### STT (Speech-to-Text)

Распознавание речи для голосового ввода вопросов.

### TTS (Text-to-Speech)

Синтез речи для озвучивания ответов системы.

## 📊 Бенчмаркинг

Для оценки качества системы используется модуль бенчмаркинга:

```bash
cd benchmark
python main.py
```

Бенчмарк для кейса №3 хранится в [`benchmark/benchmark.json`](benchmark/benchmark.json).

## 📝 API Endpoints

- `POST /api/qa/answer/` - Отправить вопрос системе
- `POST /api/stt/recognize/` - Распознать речь
- `POST /api/tts/synthesize/` - Синтезировать речь