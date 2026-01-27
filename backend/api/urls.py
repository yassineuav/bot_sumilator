from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.get_dashboard_data),
    path('train/', views.train_model),
    path('backtest/', views.run_backtest),
    path('sync/', views.sync_trades),
    path('predict/', views.get_prediction),
    path('alpaca/account/', views.get_alpaca_account),
    path('manual-test/run/', views.run_manual_test),
    path('manual-test/save/', views.save_manual_trade),
    path('manual-test/history/', views.get_manual_history),
    path('manual-test/clear/', views.clear_manual_history),
    path('manual-test/auto-backtest/', views.run_auto_backtest),
    path('train/lstm/', views.train_lstm),
    path('train/lstm/status/', views.get_lstm_status),
    path('train/lstm/versions/', views.list_lstm_versions),
    path('train/lstm/rollback/', views.rollback_lstm_version),
    path('predict/lstm/', views.predict_lstm),
    path('auto-trade/execute/', views.execute_auto_trade),
    path('history-data/', views.get_history_data),
    path('train/multitimeframe/', views.train_multitimeframe_model),
    path('train/cascade/', views.train_cascade_system),
    path('status/cascade/', views.get_cascade_status),
    path('health/', views.health_check),
]
