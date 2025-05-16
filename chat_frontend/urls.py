from django.urls import path
from . import views

urlpatterns = [
    path('indeed/', views.indeed_view),
    path('api/indeed/', views.indeed_chat),
    path('gmtt/', views.gmtt_view),
    path('api/gmtt/', views.gmtt_chat),
]