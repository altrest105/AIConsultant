import os
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework import status
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from .utils import text_to_speech_combined, text_to_speech_streaming

logger = logging.getLogger(__name__)


class TTSSynthesizeView(APIView):
    """API endpoint для синтеза русской речи из текста.
    
    Принимает текст и возвращает единый WAV-файл с озвученным текстом.
    Поддерживает различные форматы данных: multipart/form-data, 
    application/x-www-form-urlencoded и application/json.
    
    Attributes:
        parser_classes (list): Список парсеров для обработки входящих данных.
    """
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        """Обрабатывает POST-запрос для синтеза речи.
        
        Args:
            request (Request): HTTP-запрос с текстом для озвучивания.
                Ожидает параметр 'text' в теле запроса.
        
        Returns:
            Response/HttpResponse: 
                - HttpResponse с WAV-файлом при успехе (200 OK)
                - Response с ошибкой при неудаче (400/500)
        
        Raises:
            ValueError: Если текст пустой или невалидный.
            Exception: При любых других ошибках генерации речи.
            
        Example:
            POST /api/tts/synthesize/
            Content-Type: application/json
            {"text": "Привет, мир!"}
            
        Note:
            Временные файлы автоматически удаляются после отправки.
        """
        # Получение текста из разных источников
        text = request.data.get('text') or request.POST.get('text')
        
        if not text:
            return Response(
                {'error': 'Текст не предоставлен'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        audio_path = None
        
        try:
            logger.info(f"Синтез речи для текста: {text[:50]}...")
            
            # Генерация объединенного аудио
            audio_path = text_to_speech_combined(text)
            
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


@csrf_exempt
@require_http_methods(["POST"])
def text_to_speech_stream_view(request):
    """API endpoint для потоковой генерации речи.
    
    Генерирует аудио потоково, отправляя данные по мере готовности каждого
    предложения. Подходит для длинных текстов и снижения задержки первого байта.
    
    Args:
        request (HttpRequest): HTTP POST-запрос с JSON-телом.
            Ожидает параметр 'text' в JSON-теле запроса.
    
    Returns:
        StreamingHttpResponse: Потоковый ответ с WAV-данными при успехе (200 OK).
        JsonResponse: JSON с описанием ошибки при неудаче (400/500).
    
    Raises:
        json.JSONDecodeError: Если тело запроса содержит невалидный JSON.
        ValueError: Если текст пустой.
        Exception: При ошибках генерации речи.
    
    Example:
        POST /api/tts/stream/
        Content-Type: application/json
        {"text": "Это длинный текст для потоковой генерации."}
        
    Note:
        - Буферизация отключена для минимальной задержки
        - Поддерживает nginx через заголовок X-Accel-Buffering
        - Временные файлы автоматически удаляются после обработки
    """
    try:
        # Парсим JSON из тела запроса
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        
        if not text:
            return JsonResponse({
                'error': 'Текст не может быть пустым'
            }, status=400)
        
        # Создаем генератор для потоковой передачи
        def audio_stream():
            """Генератор для потоковой передачи аудио-чанков.
            
            Yields:
                bytes: WAV-данные для каждого обработанного предложения.
                
            Raises:
                Exception: При ошибках генерации аудио.
            """
            try:
                for audio_chunk in text_to_speech_streaming(text):
                    yield audio_chunk
            except Exception as e:
                logger.error(f"❌ Ошибка при генерации аудио: {e}")
                raise
        
        # Возвращаем StreamingHttpResponse
        response = StreamingHttpResponse(
            audio_stream(),
            content_type='audio/wav'
        )
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'  # Отключаем буферизацию для nginx
        
        return response
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Неверный формат JSON'
        }, status=400)
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        return JsonResponse({
            'error': str(e)
        }, status=500)