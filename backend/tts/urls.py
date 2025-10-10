from django.urls import path
from . import views

app_name = 'tts'

urlpatterns = [
    path('synthesize/', views.text_to_speech_stream_view, name='synthesize'),
]