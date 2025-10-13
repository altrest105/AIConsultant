import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import JSONParser, FormParser
from rest_framework import status
from django.http import StreamingHttpResponse
from .utils import text_to_speech_streaming

logger = logging.getLogger(__name__)


class TTSStreamSynthesizeView(APIView):
    # Указываем, что ожидаем только JSON данные
    parser_classes = [JSONParser, FormParser] 

    def post(self, request, *args, **kwargs):
        try:
            # Данные уже распарсены в request.data
            text = request.data.get('text', '').strip()
            
            if not text:
                return Response(
                    {'error': 'Текст не может быть пустым'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            logger.info(f"Начало потокового синтеза речи для текста: {text[:50]}...")

            # Создаем генератор для потоковой передачи
            def audio_stream():
                try:
                    # Генератор text_to_speech_streaming возвращает чанки аудио
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
            response['X-Accel-Buffering'] = 'no'
            
            logger.info("✅ StreamingHttpResponse возвращен клиенту.")
            return response
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка в TTSStreamSynthesizeView: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )