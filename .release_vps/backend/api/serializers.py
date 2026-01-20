from rest_framework import serializers
from .models import Trade, Performance, ManualTrade

class TradeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Trade
        fields = '__all__'

class PerformanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Performance
        fields = '__all__'

class ManualTradeSerializer(serializers.ModelSerializer):
    class Meta:
        model = ManualTrade
        fields = '__all__'
