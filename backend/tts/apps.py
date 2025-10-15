from django.apps import AppConfig
from django.conf import settings
from .utils import get_model
import logging

logger = logging.getLogger(__name__)


class TtsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'tts'

    def ready(self):
        try:
            if settings.TTS_CONFIG.get("TTS_IS_ACTIVE", True):
                logger.info("⌛ Инициализация TTS модели при запуске Django...")
                get_model()
                logger.info("✅ TTS модель инициализирована при старте Django.")
            else:
                logger.info("ℹ️ TTS отключен в настройках, пропуск инициализации модели.")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации TTS модели при старте: {e}")
