from django.urls import path
from . import views
from .views import TTSStreamSynthesizeView

app_name = 'tts'

urlpatterns = [
    path('synthesize/', TTSStreamSynthesizeView.as_view(), name='synthesize'),
]