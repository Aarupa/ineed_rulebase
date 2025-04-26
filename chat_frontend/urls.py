from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat, name='chat'),
    path('api/chat/', views.get_response, name='get_response'),
    path('clear-history/', views.clear_history, name='clear_history'),
]
