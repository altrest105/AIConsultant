import os
import tempfile
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .utils import transcribe_audio

logger = logging.getLogger(__name__)


class STTRecognizeView(APIView):
    """API endpoint для распознавания русской речи из аудиофайлов.
    
    Принимает аудиофайл и возвращает распознанный текст на русском языке.
    Использует модель Whisper (medium) для транскрибации с поддержкой
    Voice Activity Detection (VAD) для улучшения качества.
    
    Attributes:
        parser_classes (list): Список парсеров для обработки multipart/form-data.
    
    Note:
        Поддерживаемые форматы аудио: WAV, MP3, M4A, FLAC и другие.
        Временные файлы автоматически удаляются после обработки.
    """
    
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        """Обрабатывает POST-запрос для распознавания речи из аудиофайла.
        
        Args:
            request (Request): HTTP-запрос с аудиофайлом.
                Ожидает файл в поле 'file' с Content-Type: multipart/form-data.
        
        Returns:
            Response: JSON-ответ с распознанным текстом или ошибкой.
                Успех (200): {'text': 'Распознанный текст'}
                Ошибка (400): {'error': 'Файл не предоставлен'}
                Ошибка (404): {'error': 'Файл не найден'}
                Ошибка (500): {'error': 'Описание ошибки'}
        
        Raises:
            FileNotFoundError: Если временный файл недоступен.
            Exception: При ошибках транскрибации или обработки файла.
            
        Example:
            POST /api/stt/recognize/
            Content-Type: multipart/form-data
            file: audio.wav
            
            Response:
            {
                "text": "Привет, это распознанная речь"
            }
            
        Note:
            - Файл сохраняется во временную директорию
            - После обработки временный файл удаляется автоматически
            - Поддерживается chunked upload для больших файлов
        """
        # Получение файла
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return Response(
                {'error': 'Файл не предоставлен'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Сохранение во временный файл
        temp_path = None
        try:
            # Создание временного файла с правильным расширением
            suffix = os.path.splitext(uploaded_file.name)[1] or '.wav'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
                temp_path = tmp.name
            
            logger.info(f"Обработка файла: {uploaded_file.name}")
            
            # Транскрибация аудио в текст
            text = transcribe_audio(temp_path)
            
            return Response({
                'text': text
            }, status=status.HTTP_200_OK)
            
        except FileNotFoundError as e:
            logger.error(f"Файл не найден: {e}")
            return Response(
                {'error': 'Файл не найден'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        finally:
            # Удаление временного файла
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.info(f"✅ Временный файл удален: {temp_path}")
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл: {e}")