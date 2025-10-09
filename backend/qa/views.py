import os
import time
import logging
import uuid
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from .models import Document, Context, QuestionAnswer, ChatSession, ChatMessage, BenchmarkResult
from .utils import (
    parse_docx_document, split_text_into_chunks, generate_embeddings,
    create_faiss_index, search_similar_contexts, generate_answer,
    generate_qa_triplets, calculate_metrics, calculate_retrieval_metrics,
    initialize_models, CONTEXTS_STORE
)

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name='dispatch')
class DocumentUploadView(APIView):
    """Загрузка и обработка документов"""
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return Response(
                {'error': 'Файл не предоставлен'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not uploaded_file.name.endswith('.docx'):
            return Response(
                {'error': 'Поддерживаются только DOCX файлы'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Сохраняем файл
            file_path = f"docs/{uploaded_file.name}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
            
            # Парсим документ
            content = parse_docx_document(file_path)
            
            # Создаем запись в БД
            document = Document.objects.create(
                title=uploaded_file.name,
                file_path=file_path,
                content=content
            )
            
            # Обрабатываем документ асинхронно
            self.process_document(document)
            
            return Response({
                'document_id': document.id,
                'message': 'Документ успешно загружен и обрабатывается'
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки документа: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def process_document(self, document):
        """Обработка документа: создание чанков, эмбеддингов и QA пар"""
        try:
            # Инициализируем модели
            initialize_models()
            
            # Разбиваем на чанки
            chunks = split_text_into_chunks(document.content)
            
            # Создаем контексты
            contexts = []
            for i, chunk in enumerate(chunks):
                context = Context.objects.create(
                    document=document,
                    text=chunk,
                    chunk_index=i
                )
                contexts.append(context)
            
            # Генерируем эмбеддинги
            chunk_texts = [ctx.text for ctx in contexts]
            embeddings = generate_embeddings(chunk_texts)
            
            # Сохраняем эмбеддинги
            for context, embedding in zip(contexts, embeddings):
                context.embedding = embedding.tolist()
                context.save()
            
            # Создаем FAISS индекс
            create_faiss_index(embeddings)
            
            # Обновляем глобальное хранилище контекстов
            global CONTEXTS_STORE
            CONTEXTS_STORE.extend(chunk_texts)
            
            # Генерируем QA триплеты для бенчмарка
            triplets = generate_qa_triplets(document.content)
            
            for triplet in triplets:
                # Находим соответствующий контекст
                context = contexts[triplet['chunk_index']]
                
                QuestionAnswer.objects.create(
                    document=document,
                    context=context,
                    question=triplet['question'],
                    answer=triplet['answer']
                )
            
            document.processed = True
            document.save()
            
            logger.info(f"✅ Документ {document.title} успешно обработан")
            
        except Exception as e:
            logger.error(f"Ошибка обработки документа: {e}")


@method_decorator(csrf_exempt, name='dispatch')
class ChatBotView(APIView):
    """Основной чат-бот для вопросов и ответов"""
    
    def post(self, request):
        question = request.data.get('question')
        session_id = request.data.get('session_id')
        
        if not question:
            return Response(
                {'error': 'Вопрос не предоставлен'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            start_time = time.time()
            
            # Создаем или получаем сессию
            if not session_id:
                session_id = str(uuid.uuid4())
            
            session, created = ChatSession.objects.get_or_create(
                session_id=session_id,
                defaults={'user': request.user if request.user.is_authenticated else None}
            )
            
            # Ищем релевантные контексты
            retrieved_contexts = search_similar_contexts(question, top_k=10)
            
            # Извлекаем тексты контекстов для генерации ответа
            context_texts = [ctx['context'] for ctx in retrieved_contexts[:3]]
            
            # Генерируем ответ
            answer = generate_answer(question, context_texts)
            
            # Вычисляем уверенность (средний score топ-3 контекстов)
            confidence = sum(ctx['score'] for ctx in retrieved_contexts[:3]) / 3 if retrieved_contexts else 0.0
            
            response_time = time.time() - start_time
            
            # Сохраняем сообщение
            message = ChatMessage.objects.create(
                session=session,
                question=question,
                answer=answer,
                retrieved_contexts=[{
                    'context': ctx['context'][:200] + '...',
                    'score': ctx['score'],
                    'rank': ctx['rank']
                } for ctx in retrieved_contexts],
                confidence_score=confidence,
                response_time=response_time
            )
            
            return Response({
                'session_id': session_id,
                'answer': answer,
                'confidence': confidence,
                'response_time': response_time,
                'retrieved_contexts': retrieved_contexts[:5]  # Возвращаем топ-5 для анализа
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Ошибка чат-бота: {e}")
            return Response(
                {'error': 'Произошла ошибка при обработке вопроса'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class BenchmarkView(APIView):
    """Запуск бенчмарка QA системы"""
    
    def post(self, request):
        document_id = request.data.get('document_id')
        
        if not document_id:
            return Response(
                {'error': 'ID документа не предоставлен'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            document = get_object_or_404(Document, id=document_id)
            qa_pairs = QuestionAnswer.objects.filter(document=document)
            
            results = []
            
            for qa_pair in qa_pairs:
                # Получаем ответ от системы
                retrieved_contexts = search_similar_contexts(qa_pair.question, top_k=10)
                context_texts = [ctx['context'] for ctx in retrieved_contexts[:3]]
                generated_answer = generate_answer(qa_pair.question, context_texts)
                
                # Сохраняем сгенерированный ответ
                qa_pair.generated_answer = generated_answer
                qa_pair.save()
                
                # Вычисляем метрики качества ответов
                answer_metrics = calculate_metrics(generated_answer, qa_pair.answer)
                
                # Вычисляем метрики ретривера
                retrieval_metrics = calculate_retrieval_metrics(retrieved_contexts, qa_pair.context.text)
                
                # Создаем запись результата
                benchmark_result = BenchmarkResult.objects.create(
                    qa_pair=qa_pair,
                    bleurt_score=answer_metrics.get('bleurt_score'),
                    semantic_similarity_score=answer_metrics.get('semantic_similarity'),
                    rouge_l_score=answer_metrics.get('rouge_l'),
                    ndcg_10=retrieval_metrics.get('ndcg_10'),
                    mrr_10=retrieval_metrics.get('mrr_10'),
                    map_100=retrieval_metrics.get('map_100')
                )
                
                results.append({
                    'question': qa_pair.question,
                    'ground_truth': qa_pair.answer,
                    'generated_answer': generated_answer,
                    'metrics': {**answer_metrics, **retrieval_metrics},
                    'retrieved_contexts': retrieved_contexts[:10]
                })
            
            # Вычисляем средние метрики
            avg_metrics = self.calculate_average_metrics(results)
            
            return Response({
                'document_title': document.title,
                'total_qa_pairs': len(results),
                'average_metrics': avg_metrics,
                'detailed_results': results
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Ошибка бенчмарка: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def calculate_average_metrics(self, results):
        """Вычисление средних метрик"""
        if not results:
            return {}
        
        metrics_sums = {}
        for result in results:
            for metric, value in result['metrics'].items():
                if value is not None:
                    metrics_sums[metric] = metrics_sums.get(metric, 0) + value
        
        avg_metrics = {}
        for metric, total in metrics_sums.items():
            avg_metrics[f'avg_{metric}'] = total / len(results)
        
        return avg_metrics


@method_decorator(csrf_exempt, name='dispatch')
class DocumentListView(APIView):
    """Список загруженных документов"""
    
    def get(self, request):
        documents = Document.objects.all().order_by('-uploaded_at')
        
        data = []
        for doc in documents:
            qa_count = QuestionAnswer.objects.filter(document=doc).count()
            context_count = Context.objects.filter(document=doc).count()
            
            data.append({
                'id': doc.id,
                'title': doc.title,
                'uploaded_at': doc.uploaded_at,
                'processed': doc.processed,
                'qa_pairs_count': qa_count,
                'contexts_count': context_count
            })
        
        return Response(data, status=status.HTTP_200_OK)