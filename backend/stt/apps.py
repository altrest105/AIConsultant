from django.apps import AppConfig
from django.conf import settings
from .utils import get_model
import logging

logger = logging.getLogger(__name__)


class SttConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'stt'

    def ready(self):
        try:
            if settings.STT_CONFIG.get("STT_IS_ACTIVE", True):
                logger.info("⌛ Инициализация Whisper модели при запуске Django...")
                get_model()
                logger.info("✅ Whisper модель инициализирована при старте Django.")
            else:
                logger.info("ℹ️ STT отключен в настройках, пропуск инициализации модели.")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации Whisper модели при старте: {e}")