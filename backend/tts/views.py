import os
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from django.http import HttpResponse
from .utils import text_to_speech

logger = logging.getLogger(__name__)


class TTSSynthesizeView(APIView):
    """Синтез русской речи из текста"""
    
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        # Получение текста
        text = request.data.get('text')
        if not text:
            return Response(
                {'error': 'Текст не предоставлен'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        audio_path = None
        
        try:
            logger.info(f"Синтез речи для текста: {text[:50]}...")
            
            # Проверка на кодировку
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="ignore")
            # Генерация аудио
            audio_path = text_to_speech(text)
            
            # Читаем файл в память
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # Удаляем файл сразу после чтения
            try:
                os.remove(audio_path)
                logger.info(f"✅ Временный файл удален: {audio_path}")
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл: {e}")
            
            # Отправляем из памяти
            response = HttpResponse(audio_data, content_type='audio/wav')
            response['Content-Disposition'] = 'attachment; filename="speech.wav"'
            
            return response
            
        except ValueError as e:
            logger.error(f"Ошибка валидации: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            
            # Очистка в случае ошибки
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )