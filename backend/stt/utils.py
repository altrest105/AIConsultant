import os
import logging
from django.conf import settings
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Глобальная переменная для модели
WHISPER_MODEL = None


def get_model():
    global WHISPER_MODEL
    
    if WHISPER_MODEL is not None:
        return WHISPER_MODEL
    
    try:
        model_size = settings.STT_CONFIG.get("MODEL_SIZE", "large")
        gpu_compute_type = settings.STT_CONFIG.get("GPU_COMPUTE_TYPE", "float16")
        cpu_compute_type = settings.STT_CONFIG.get("CPU_COMPUTE_TYPE", "int8")

        # Попытка использовать GPU
        logger.info("🔄 Загрузка Whisper модели на GPU...")
        WHISPER_MODEL = WhisperModel(model_size, device="cuda", compute_type=gpu_compute_type)
        logger.info("✅ Whisper модель загружена на GPU")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось загрузить на GPU: {e}")
        try:
            # Fallback на CPU
            logger.info("🔄 Загрузка Whisper модели на CPU...")
            WHISPER_MODEL = WhisperModel(model_size, device="cpu", compute_type=cpu_compute_type)
            logger.info("✅ Whisper модель загружена на CPU")
        except Exception as cpu_error:
            logger.error(f"❌ Ошибка инициализации модели: {cpu_error}")
            raise RuntimeError("Не удалось инициализировать модель")
    
    return WHISPER_MODEL


def transcribe_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    # Получение модели
    model = get_model()
    
    try:
        language = settings.STT_CONFIG.get("LANGUAGE", "ru")
        beam_size = settings.STT_CONFIG.get("BEAM_SIZE", 5)
        vad_threshold = settings.STT_CONFIG.get("VAD_THRESHOLD", 0.5)
        vad_min_speech_duration_ms = settings.STT_CONFIG.get("VAD_MIN_SPEECH_DURATION_MS", 250)

        # Транскрибация на русском языке
        segments, info = model.transcribe(
            file_path,
            language=language,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters={
                "threshold": vad_threshold,
                "min_speech_duration_ms": vad_min_speech_duration_ms,
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