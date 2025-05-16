from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat, name='chat'),
    path('gmtt/', views.gmtt, name='gmtt'),  # This line maps the root URL to the chat view
    path('api/chat/', views.get_response, name='get_response'),
    # path('clear-history/', views.clear_history, name='clear_history'),
    path('api/gmtt/', views.gmtt_response, name='gmtt_response'),  # New endpoint for Give Me Trees
]