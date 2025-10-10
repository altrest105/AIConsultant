import os
import logging
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Глобальная переменная для модели
WHISPER_MODEL = None


def initialize_model():
    """Инициализирует глобальную модель Whisper для распознавания речи.
    
    Пытается загрузить модель на GPU с использованием float16.
    При неудаче автоматически переключается на CPU с int8.
    Использует глобальную переменную для кэширования модели.
    
    Returns:
        WhisperModel: Инициализированная модель Whisper (medium).
        
    Raises:
        RuntimeError: Если не удалось инициализировать модель ни на GPU, ни на CPU.
        
    Note:
        - При повторном вызове возвращает уже загруженную модель
        - GPU версия использует compute_type="float16"
        - CPU версия использует compute_type="int8" для оптимизации
    """
    global WHISPER_MODEL
    
    if WHISPER_MODEL is not None:
        return WHISPER_MODEL
    
    try:
        # Попытка использовать GPU
        logger.info("Инициализация модели на GPU...")
        WHISPER_MODEL = WhisperModel("medium", device="cuda", compute_type="float16")
        logger.info("✅ Модель загружена на GPU")
    except Exception as e:
        logger.warning(f"Не удалось загрузить на GPU: {e}")
        try:
            # Fallback на CPU
            logger.info("Инициализация модели на CPU...")
            WHISPER_MODEL = WhisperModel("medium", device="cpu", compute_type="int8")
            logger.info("✅ Модель загружена на CPU")
        except Exception as cpu_error:
            logger.error(f"Ошибка инициализации модели: {cpu_error}")
            raise RuntimeError("Не удалось инициализировать модель")
    
    return WHISPER_MODEL


def transcribe_audio(file_path):
    """Транскрибирует аудиофайл в текст на русском языке.
    
    Использует модель Whisper для распознавания речи с настройками VAD
    (Voice Activity Detection) для фильтрации тишины и улучшения качества.
    
    Args:
        file_path (str): Путь к аудиофайлу для транскрибации.
            Поддерживаемые форматы: WAV, MP3, M4A, FLAC и другие.
    
    Returns:
        str: Распознанный текст, объединенный из всех сегментов.
            Пробелы между сегментами автоматически добавляются.
    
    Raises:
        FileNotFoundError: Если указанный файл не существует.
        Exception: При ошибках во время транскрибации.
        
    Note:
        Используемые параметры:
        - language="ru" - русский язык
        - beam_size=5 - размер луча для декодирования
        - vad_filter=True - фильтрация по голосовой активности
        - vad_threshold=0.5 - порог уверенности VAD
        - min_speech_duration_ms=250 - минимальная длительность речи
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    # Инициализация модели если не инициализирована
    model = initialize_model()
    
    try:
        # Транскрибация на русском языке
        segments, info = model.transcribe(
            file_path,
            language="ru",
            beam_size=5,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
            }
        )
        
        # Сбор текста из сегментов
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        
        text = ' '.join(text_parts)
        
        logger.info(f"✅ Транскрибация завершена. Длина текста: {len(text)} символов")
        return text
        
    except Exception as e:
        logger.error(f"Ошибка транскрибации: {e}")
        raise