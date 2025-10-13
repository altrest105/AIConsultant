from django.apps import AppConfig
from .utils import get_model
import logging

logger = logging.getLogger(__name__)


class TtsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'tts'

    def ready(self):
        try:
            logger.info("🔄 Инициализация TTS модели при запуске Django...")
            get_model()
            logger.info("✅ TTS модель инициализирована при старте Django.")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации TTS модели при старте: {e}")
