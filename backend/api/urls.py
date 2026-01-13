from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.get_dashboard_data),
    path('train/', views.train_model),
    path('backtest/', views.run_backtest),
    path('sync/', views.sync_trades),
    path('predict/', views.get_prediction),
]
