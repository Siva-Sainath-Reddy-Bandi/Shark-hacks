from django.contrib import admin
from django.urls import path
from . import views

# All of the endpoints will start from here
urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login, name='login'),
    path('report/', views.report, name='report'),
    path('ml/', views.ml, name='model'),
    path('transactions/', views.transactions, name='transactions'),
    path('signup/', views.signup, name='signup')
]
