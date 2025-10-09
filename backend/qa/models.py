from django.db import models
from django.contrib.auth.models import User


class Document(models.Model):
    """Модель для хранения документов"""
    title = models.CharField(max_length=255, verbose_name="Название документа")
    file_path = models.CharField(max_length=500, verbose_name="Путь к файлу")
    content = models.TextField(verbose_name="Содержимое документа")
    uploaded_at = models.DateTimeField(auto_now_add=True, verbose_name="Дата загрузки")
    processed = models.BooleanField(default=False, verbose_name="Обработан")
    
    def __str__(self):
        return self.title


class Context(models.Model):
    """Модель для хранения контекстов из документов"""
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='contexts')
    text = models.TextField(verbose_name="Текст контекста")
    chunk_index = models.IntegerField(verbose_name="Индекс чанка")
    embedding = models.JSONField(null=True, blank=True, verbose_name="Эмбеддинг")
    
    def __str__(self):
        return f"Context {self.chunk_index} from {self.document.title}"


class QuestionAnswer(models.Model):
    """Модель для хранения пар вопрос-ответ (бенчмарк)"""
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='qa_pairs')
    context = models.ForeignKey(Context, on_delete=models.CASCADE, related_name='qa_pairs')
    question = models.TextField(verbose_name="Вопрос")
    answer = models.TextField(verbose_name="Ответ")
    generated_answer = models.TextField(null=True, blank=True, verbose_name="Сгенерированный ответ")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"QA: {self.question[:50]}..."


class ChatSession(models.Model):
    """Модель для хранения сессий чата"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Chat Session {self.session_id}"


class ChatMessage(models.Model):
    """Модель для хранения сообщений чата"""
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    question = models.TextField(verbose_name="Вопрос пользователя")
    answer = models.TextField(verbose_name="Ответ системы")
    retrieved_contexts = models.JSONField(null=True, blank=True, verbose_name="Найденные контексты")
    confidence_score = models.FloatField(null=True, blank=True, verbose_name="Уверенность")
    response_time = models.FloatField(null=True, blank=True, verbose_name="Время ответа (сек)")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Message: {self.question[:50]}..."


class BenchmarkResult(models.Model):
    """Модель для хранения результатов бенчмарка"""
    qa_pair = models.ForeignKey(QuestionAnswer, on_delete=models.CASCADE)
    bleurt_score = models.FloatField(null=True, blank=True)
    semantic_similarity_score = models.FloatField(null=True, blank=True)
    rouge_l_score = models.FloatField(null=True, blank=True)
    ndcg_10 = models.FloatField(null=True, blank=True)
    mrr_10 = models.FloatField(null=True, blank=True)
    map_100 = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Benchmark for {self.qa_pair.question[:30]}..."