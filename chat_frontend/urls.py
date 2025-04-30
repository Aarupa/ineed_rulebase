from django.urls import path
from . import views,main
from . import lm  # Importing the lm module for the neural chat functionality

urlpatterns = [
    path('', views.chat, name='chat'),  # URL for the chat view
    path('api/chat/', main.get_response, name='get_response'),  # URL for the chatbot response
     path('clear-history/', views.clear_history, name='clear_history'),
     path('api/generate/',lm.query_neural_chat,name='query_neural_chat')
]
