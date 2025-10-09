from django.urls import path
from .views import DocumentUploadView, ChatBotView, BenchmarkView, DocumentListView

app_name = 'qa'

urlpatterns = [
    path('upload/', DocumentUploadView.as_view(), name='upload_document'),
    path('chat/', ChatBotView.as_view(), name='chatbot'),
    path('benchmark/', BenchmarkView.as_view(), name='benchmark'),
    path('documents/', DocumentListView.as_view(), name='document_list'),
]