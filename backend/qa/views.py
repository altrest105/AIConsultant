from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from . import utils

class QAView(APIView):
    def post(self, request, *args, **kwargs):
        question = request.data.get('question', '').strip()

        # Валидация входных данных
        if not question:
            return Response(
                {"error": "Поле 'question' является обязательным. Передайте текст вопроса в теле запроса."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Вызов QA модуля
        try:
            answer_data = utils.answer_question(question)

            # Возвращаем ответ
            return Response(answer_data, status=status.HTTP_200_OK)

        except Exception as e:
            # Обработка ошибок
            return Response(
                {"error": "Произошла внутренняя ошибка при обработке вопроса.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )