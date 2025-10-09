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
    """Распознавание русской речи из аудиофайла"""
    
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
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
            # Создание временного файла
            suffix = os.path.splitext(uploaded_file.name)[1] or '.wav'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
                temp_path = tmp.name
            
            logger.info(f"Обработка файла: {uploaded_file.name}")
            
            # Транскрибация
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
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл: {e}")