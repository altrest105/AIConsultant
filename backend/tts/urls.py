from django.urls import path
from .views import TTSSynthesizeView

app_name = 'tts'

urlpatterns = [
    path('synthesize/', TTSSynthesizeView.as_view(), name='synthesize'),
]