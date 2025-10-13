from django.urls import path
from .views import QAView

app_name = 'qa'

urlpatterns = [
    path('answer/', QAView.as_view(), name='answer_question'),
]