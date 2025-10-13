from django.apps import AppConfig
import logging
import os
from . import utils
from django.conf import settings

logger = logging.getLogger(__name__)


class QaConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'qa'

    def ready(self):
        logger.info("⌛ Инициализация всех компонентов QA при запуске Django...")
        logger.info("\n--- Шаг 1/2: Инициализация моделей...")
        utils.initialize_qa_system()
        logger.info("\n--- Шаг 2/2: Индексация базы знаний...")
        docs_folder_path = settings.QA_CONFIG.get("QA_DOCX", os.path.abspath(os.path.join(settings.BASE_DIR, 'files', 'docs')))
        docs_folder_path.mkdir(parents=True, exist_ok=True)
        if not (docs_folder_path / "data.docx").exists():
             logger.warning(f"⚠️ Документ 'data.docx' не найден в папке '{docs_folder_path}'.")
        utils.index_knowledge_base(docs_folder_path)
        if not utils.FULL_INDEXED_DOCUMENTS:
            logger.error("❌ Корпус пуст после индексации. Проверьте, есть ли DOCX-файлы.")
        logger.info("✅ Все компоненты QA инициализированы при запуске Django.")
