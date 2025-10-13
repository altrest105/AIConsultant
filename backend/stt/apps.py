from django.apps import AppConfig
from .utils import get_model
import logging

logger = logging.getLogger(__name__)


class SttConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'stt'

    def ready(self):
        try:
            logger.info("🔄 Инициализация Whisper модели при запуске Django...")
            get_model()
            logger.info("✅ Whisper модель инициализирована при старте Django.")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации Whisper модели при старте: {e}")