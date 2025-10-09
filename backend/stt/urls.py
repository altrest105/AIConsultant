from django.urls import path
from .views import STTRecognizeView

app_name = 'stt'

urlpatterns = [
    path('recognize/', STTRecognizeView.as_view(), name='recognize'),
]