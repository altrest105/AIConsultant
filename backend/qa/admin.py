from django.contrib import admin
from .models import Document, Context, QuestionAnswer, ChatSession, ChatMessage, BenchmarkResult


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['title', 'uploaded_at', 'processed']
    list_filter = ['processed', 'uploaded_at']
    search_fields = ['title', 'content']


@admin.register(Context)
class ContextAdmin(admin.ModelAdmin):
    list_display = ['document', 'chunk_index']
    list_filter = ['document']
    search_fields = ['text']


@admin.register(QuestionAnswer)
class QuestionAnswerAdmin(admin.ModelAdmin):
    list_display = ['question', 'document', 'created_at']
    list_filter = ['document', 'created_at']
    search_fields = ['question', 'answer']


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'user', 'created_at']
    list_filter = ['created_at']


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['session', 'question', 'confidence_score', 'created_at']
    list_filter = ['session', 'created_at']
    search_fields = ['question', 'answer']


@admin.register(BenchmarkResult)
class BenchmarkResultAdmin(admin.ModelAdmin):
    list_display = ['qa_pair', 'bleurt_score', 'semantic_similarity_score', 'created_at']
    list_filter = ['created_at']