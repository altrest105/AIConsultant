import os
import tempfile
import logging
import torch
from TTS.api import TTS
from django.conf import settings

logger = logging.getLogger(__name__)

# Глобальная переменная для модели
TTS_MODEL = None
REFERENCE_WAV_PATH = os.path.abspath(os.path.join(settings.BASE_DIR, '..', 'docs', 'audio.wav'))


def initialize_model():
    """Инициализация модели TTS"""
    global TTS_MODEL
    
    if TTS_MODEL is not None:
        return TTS_MODEL
    
    try:
        logger.info("Инициализация TTS модели...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        TTS_MODEL = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        TTS_MODEL.to(device)
        logger.info(f"✅ TTS модель загружена на устройство: {device}")
    except Exception as e:
        logger.error(f"Ошибка инициализации TTS модели: {e}")
        raise RuntimeError("Не удалось инициализировать TTS модель")
    
    return TTS_MODEL


def text_to_speech(text):
    """
    Конвертирует текст в речь на русском языке
    
    Args:
        text: Текст для синтеза
    
    Returns:
        str: Путь к сгенерированному аудиофайлу
    """
    if not text or not text.strip():
        raise ValueError("Текст не может быть пустым")
    
    # Инициализация модели если не инициализирована
    model = initialize_model()
    
    # Создаем временный файл для результата
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        output_path = tmp.name
    
    try:
        logger.info(f"Генерация речи для текста: {text[:50]}...")
        
        # Используем дефолтный или предоставленный голос
        if not os.path.exists(REFERENCE_WAV_PATH):
            print(text)
            logger.warning(f"⚠️ Файл спикера не найден: {REFERENCE_WAV_PATH}, будет использован голос по умолчанию модели")
            model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker="Aaron Dreschner",
                language="ru"
        )
        else:
            print(text)
            model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=REFERENCE_WAV_PATH,
                language="ru"
        )
        
        logger.info(f"✅ Речь успешно сгенерирована: {output_path}")
        return output_path
        
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        logger.error(f"Ошибка генерации речи: {e}")
        raise