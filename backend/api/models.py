from django.db import models

class Trade(models.Model):
    entry_time = models.DateTimeField()
    exit_time = models.DateTimeField(null=True, blank=True)
    symbol = models.CharField(max_length=10, default='SPY')
    type = models.CharField(max_length=10) # CALL / PUT
    entry_price = models.FloatField()
    exit_price = models.FloatField(null=True, blank=True)
    pnl_pct = models.FloatField(null=True, blank=True)
    pnl_val = models.FloatField(null=True, blank=True)
    exit_reason = models.CharField(max_length=100, null=True, blank=True)
    confidence = models.FloatField()

    def __str__(self):
        return f"{self.symbol} {self.type} @ {self.entry_price}"

class Performance(models.Model):
    symbol = models.CharField(max_length=10)
    win_rate = models.FloatField()
    total_pnl_pct = models.FloatField()
    total_trades = models.IntegerField()
    updated_at = models.DateTimeField(auto_now=True)
