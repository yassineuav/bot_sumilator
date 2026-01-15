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

class ManualTrade(models.Model):
    symbol = models.CharField(max_length=10)
    timestamp = models.DateTimeField()
    prediction = models.CharField(max_length=10) # CALL / PUT / NEUTRAL
    confidence = models.FloatField()
    expected_move_pct = models.FloatField(null=True, blank=True)
    
    # Option details
    option_strike = models.FloatField(null=True, blank=True)
    option_premium = models.FloatField(null=True, blank=True)
    option_expiry = models.CharField(max_length=20, null=True, blank=True)
    option_type = models.CharField(max_length=10, null=True, blank=True)
    
    # Entry details
    entry_price = models.FloatField()
    take_profit = models.FloatField()
    stop_loss = models.FloatField()
    position_size = models.FloatField() # % of account
    
    # Result
    result = models.CharField(max_length=20) # TP / SL / Expired / Open
    pnl_pct = models.FloatField()
    pnl_val = models.FloatField()
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Manual {self.symbol} @ {self.timestamp} - {self.result}"
