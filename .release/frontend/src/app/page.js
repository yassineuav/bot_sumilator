"use client";
import React, { useState, useEffect } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import PredictionChart from '../components/PredictionChart';
import { ArrowUpRight, ArrowDownRight, Activity, TrendingUp, DollarSign, Shield, RefreshCw } from 'lucide-react';
import StatCard from '@/components/StatCard';
import { ThemeToggle } from '@/components/ThemeToggle';

export default function Dashboard() {
  const [data, setData] = useState({
    latest_trades: [],
    performance: [],
    equity_curve: [],
    stats: {
      total_trades: 0,
      win_rate: 0,
      total_pnl: 0,
      current_equity: 1000
    }
  });
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [alpacaAccount, setAlpacaAccount] = useState(null);
  const [config, setConfig] = useState({
    symbol: 'SPY',
    interval: '15m',
    stop_loss: 10,
    take_profit: 100,
    risk_pct: 10,
    use0DTE: true,
    isAutoTrading: false
  });

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${apiUrl}/api/dashboard/`);
      const json = await res.json();
      setData(json);
    } catch (error) {
      console.error("Failed to fetch dashboard data:", error);
    }
    setLoading(false);
  };

  /* Existing Logic */
  const fetchAlpacaAccount = async () => {
    try {
      const res = await fetch(`${apiUrl}/api/alpaca/account/`);
      const json = await res.json();
      if (res.ok) {
        setAlpacaAccount(json);
      }
    } catch (error) {
      console.error("Failed to fetch Alpaca account:", error);
    }
  };

  const [systemStatus, setSystemStatus] = useState({ backend: 'offline', market: 'unknown' });

  const fetchSystemStatus = async () => {
    try {
      const res = await fetch(`${apiUrl}/api/health/`);
      if (res.ok) {
        const json = await res.json();
        setSystemStatus({
          backend: 'online',
          market: json.market_status || 'unknown'
        });
      } else {
        setSystemStatus(prev => ({ ...prev, backend: 'offline' }));
      }
    } catch (e) {
      setSystemStatus(prev => ({ ...prev, backend: 'offline' }));
    }
  };

  const fetchPrediction = async () => {
    setPredicting(true);
    try {
      const res = await fetch(`${apiUrl}/api/predict/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      const json = await res.json();
      if (res.ok) {
        setPrediction(json);
      } else {
        console.error("Prediction failed:", json.error);
      }
    } catch (error) {
      console.error("Prediction fetch failed:", error);
    }
    setPredicting(false);
  };

  const syncData = async () => {
    setLoading(true);
    try {
      await fetch(`${apiUrl}/api/sync/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: config.symbol,
          interval: config.interval
        })
      });
      await fetchData();
      await fetchAlpacaAccount();
    } catch (error) {
      console.error("Sync failed:", error);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
    fetchPrediction();
    fetchAlpacaAccount();
    fetchSystemStatus();
    // Poll status every 30s
    const interval = setInterval(fetchSystemStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // Auto Trader Polling Effect
  useEffect(() => {
    let intervalId;
    if (config.isAutoTrading) {
      const runAutoTrade = async () => {
        if (predicting) return;

        // 1. Get latest prediction
        await fetchPrediction();

        // 2. If signal is valid, execute trade (handled by backend or implicit in predict hook if extended, 
        // but here we will add a separate execute call for safety and specific logic)
        // Actually, best pattern: Check if we have a NEW prediction that matches criteria
        // For simplicity in this user request, we trigger the endpoint which checks everything.
        try {
          await fetch(`${apiUrl}/api/auto-trade/execute/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
          });
          // Refresh account data after potential trade
          await fetchAlpacaAccount();
          await fetchData();
        } catch (e) {
          console.error("Auto trade loop error:", e);
        }
      };

      // Poll based on interval (roughly)
      // 1m = 60000ms
      const pollMs = config.interval === '1m' ? 60000 : config.interval === '5m' ? 300000 : 900000;

      // Run immediately on start
      runAutoTrade();
      intervalId = setInterval(runAutoTrade, pollMs);
    }
    return () => clearInterval(intervalId);
  }, [config.isAutoTrading, config.interval]);

  // Calculate stats from trades if performance is empty (fallback)
  const totalEquity = alpacaAccount ? alpacaAccount.equity : (data.stats?.current_equity || 1000);
  const buyingPower = alpacaAccount ? alpacaAccount.buying_power : 0;
  const winRate = data.stats?.win_rate || 0;

  return (
    <div className="flex-1 space-y-4 p-4 md:p-8 pt-6 min-h-screen bg-background">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 md:gap-0">
        <div className="flex items-center gap-4">
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground">Dashboard</h2>

          {/* Status Indicators */}
          <div className="hidden md:flex items-center gap-2">
            <div className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase border flex items-center gap-1.5 ${systemStatus.backend === 'online' ? 'bg-green-500/10 text-green-500 border-green-500/20' : 'bg-red-500/10 text-red-500 border-red-500/20'}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${systemStatus.backend === 'online' ? 'bg-green-500' : 'bg-red-500'} ${systemStatus.backend === 'online' && 'animate-pulse'}`}></span>
              Backend: {systemStatus.backend}
            </div>
            <div className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase border flex items-center gap-1.5 ${systemStatus.market === 'open' ? 'bg-green-500/10 text-green-500 border-green-500/20' : 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20'}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${systemStatus.market === 'open' ? 'bg-green-500' : 'bg-yellow-500'}`}></span>
              Market: {systemStatus.market}
            </div>
          </div>

          {alpacaAccount && (
            <div className="hidden sm:flex items-center gap-2 px-3 py-1 bg-primary/10 rounded-full text-[10px] font-black uppercase tracking-widest text-primary border border-primary/20">
              <span className={`w-2 h-2 rounded-full ${alpacaAccount.status === 'ACTIVE' ? 'bg-green-500' : 'bg-yellow-500'} animate-pulse`} />
              Alpaca {alpacaAccount.status}
            </div>
          )}
        </div>
        <div className="flex items-center space-x-2 w-full md:w-auto justify-end">
          <div className="hidden md:block">
            <ThemeToggle />
          </div>
          <button
            onClick={syncData}
            disabled={loading}
            className="bg-primary text-primary-foreground px-4 py-2 rounded-lg hover:bg-primary/90 transition-colors flex items-center justify-center gap-2 disabled:opacity-50 w-full md:w-auto text-sm"
          >
            <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
            {loading ? "Syncing..." : "Sync Data"}
          </button>
        </div>
      </div>

      <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard title="Account Equity" value={`$${parseFloat(totalEquity).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`} icon={DollarSign} trend={0} />
        <StatCard title="Buying Power" value={`$${parseFloat(buyingPower).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`} icon={Shield} trend={0} />
        <StatCard title="Win Rate" value={`${winRate.toFixed(1)}%`} icon={TrendingUp} trend={0} />
        <StatCard title="Day Trades Today" value={alpacaAccount ? alpacaAccount.daytrade_count.toString() : (data.stats?.total_trades || 0).toString()} icon={Activity} />
      </div>

      <div className="grid gap-4 grid-cols-1 lg:grid-cols-7">
        <div className="lg:col-span-4 space-y-4">
          {/* Main Chart Card */}
          <div className="rounded-xl border bg-card text-card-foreground p-6 shadow-sm">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Recent Performance</h3>
              <div className="flex items-center gap-2 px-3 py-1 bg-muted rounded-full text-xs font-medium">
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                Live Data
              </div>
            </div>
            <div className="h-[300px]" style={{ height: "300px" }}>
              <ResponsiveContainer width="100%" height="100%" minHeight={100}>
                <AreaChart data={data.equity_curve}>
                  <defs>
                    <linearGradient id="colorPnL" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8884d8" stopOpacity={0.1} />
                      <stop offset="95%" stopColor="#8884d8" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                  <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => `$${value}`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))', color: 'hsl(var(--card-foreground))' }}
                  />
                  <Area type="monotone" dataKey="equity" stroke="hsl(var(--primary))" strokeWidth={2} fillOpacity={1} fill="url(#colorPnL)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Neural Predictor Card - Redesigned to match UI */}
          <div className="md:col-span-3 rounded-xl border bg-card text-card-foreground shadow p-4 md:p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-lg flex items-center">
                <Activity className="h-5 w-5 mr-2 text-primary" />
                Neural Trading Signal
              </h3>
              <span className="text-xs text-muted-foreground bg-muted px-2 py-1 rounded-full">{prediction?.timestamp ? new Date(prediction.timestamp).toLocaleTimeString() : 'Waiting...'}</span>
            </div>

            {prediction ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex flex-col space-y-1 p-3 bg-muted/50 rounded-lg">
                    <span className="text-xs text-muted-foreground font-medium uppercase">Signal</span>
                    <div className="flex items-center space-x-2">
                      {prediction.signal === 'CALL' ? <ArrowUpRight className="h-5 w-5 text-green-500" /> : <ArrowDownRight className="h-5 w-5 text-red-500" />}
                      <span className={`text-xl font-bold ${prediction.signal === 'CALL' ? 'text-green-500' : 'text-red-500'}`}>{prediction.signal}</span>
                    </div>
                  </div>
                  <div className="flex flex-col space-y-1 p-3 bg-muted/50 rounded-lg">
                    <span className="text-xs text-muted-foreground font-medium uppercase">Confidence</span>
                    <span className="text-xl font-bold">{prediction.confidence?.toFixed(1)}%</span>
                  </div>
                </div>

                <div className="relative pt-4">
                  <span className="absolute -top-1 left-0 text-[10px] font-medium text-muted-foreground uppercase italic underline decoration-primary/30">chart</span>
                  <div className="h-[350px]">
                    <PredictionChart data={prediction.chart_data} prediction={prediction} />
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-6 pt-4 bg-muted/30 rounded-xl p-4 border border-border/50">
                  <div className="text-center">
                    <p className="text-[10px] text-muted-foreground uppercase font-black tracking-widest mb-1">Contract</p>
                    <p className="font-bold text-sm truncate">{prediction.option_data.contract.split(' (Est.)')[0]}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-[10px] text-muted-foreground uppercase font-black tracking-widest mb-1">Est. Premium</p>
                    <p className="font-black text-lg text-primary">${prediction.option_data.entry.toFixed(2)}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-[10px] text-muted-foreground uppercase font-black tracking-widest mb-1">Target TP</p>
                    <p className="font-black text-lg text-green-500">${prediction.option_data.target.toFixed(2)}</p>
                  </div>
                </div>
              </div>
            ) : predicting ? (
              <div className="h-[400px] flex flex-col items-center justify-center text-muted-foreground gap-4">
                <RefreshCw className="animate-spin w-8 h-8 opacity-20" />
                <p className="text-sm font-medium animate-pulse">Running Neural Inference Engine...</p>
              </div>
            ) : (
              <div className="h-[400px] flex flex-col items-center justify-center text-muted-foreground gap-2">
                <p className="text-sm">Click "Update Prediction" to fetch latest signal.</p>
              </div>
            )}

          </div>

          {/* Auto Trade Configuration */}
          <div className="md:col-span-3 rounded-xl border bg-card text-card-foreground shadow p-4 md:p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-lg flex items-center gap-2">
                <RefreshCw className={`h-5 w-5 text-primary ${config.isAutoTrading ? 'animate-spin' : ''}`} />
                Auto Trade Configuration
              </h3>
              <div className="flex items-center gap-2">
                <span className={`text-xs font-bold uppercase ${config.isAutoTrading ? 'text-green-500' : 'text-muted-foreground'}`}>
                  {config.isAutoTrading ? 'RUNNING' : 'STOPPED'}
                </span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    className="sr-only peer"
                    checked={config.isAutoTrading || false}
                    onChange={(e) => setConfig(prev => ({ ...prev, isAutoTrading: e.target.checked }))}
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                </label>
              </div>
            </div>

            <div className={`grid grid-cols-1 md:grid-cols-2 gap-6 transition-opacity duration-300 ${!config.isAutoTrading && 'opacity-75'}`}>
              <div className="space-y-4">
                <h4 className="text-sm font-black uppercase text-muted-foreground tracking-widest">Risk Management</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label className="text-xs font-bold">Risk Per Trade (%)</label>
                    <input
                      type="number"
                      value={config.risk_pct || 10}
                      onChange={(e) => setConfig(prev => ({ ...prev, risk_pct: parseFloat(e.target.value) }))}
                      className="w-full bg-muted border rounded-lg px-3 py-2 text-sm font-bold"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-bold">Take Profit (%)</label>
                    <input
                      type="number"
                      value={config.take_profit || 100}
                      onChange={(e) => setConfig(prev => ({ ...prev, take_profit: parseFloat(e.target.value) }))}
                      className="w-full bg-muted border rounded-lg px-3 py-2 text-sm font-bold text-green-500"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-bold">Stop Loss (%)</label>
                    <input
                      type="number"
                      value={config.stop_loss || 10}
                      onChange={(e) => setConfig(prev => ({ ...prev, stop_loss: parseFloat(e.target.value) }))}
                      className="w-full bg-muted border rounded-lg px-3 py-2 text-sm font-bold text-red-500"
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h4 className="text-sm font-black uppercase text-muted-foreground tracking-widest">Strategy Settings</h4>
                <div className="space-y-3">
                  <label className="flex items-center gap-3 p-3 border rounded-xl bg-muted/20 cursor-pointer hover:bg-muted/40 transition">
                    <input
                      type="checkbox"
                      checked={config.use0DTE || false}
                      onChange={(e) => setConfig(prev => ({ ...prev, use0DTE: e.target.checked }))}
                      className="w-4 h-4 text-primary rounded border-gray-300 focus:ring-primary"
                    />
                    <div className="flex-1">
                      <p className="text-sm font-bold">Enable 0DTE Strategy</p>
                      <p className="text-[10px] text-muted-foreground">Expires same day. High risk, high reward.</p>
                    </div>
                  </label>

                  <div className="flex gap-2">
                    {['1m', '5m', '15m'].map(tf => (
                      <button
                        key={tf}
                        onClick={() => setConfig(prev => ({ ...prev, interval: tf }))}
                        className={`flex-1 py-2 rounded-lg text-xs font-bold border transition-all ${config.interval === tf ? 'bg-primary text-primary-foreground border-primary' : 'bg-background hover:bg-muted'}`}
                      >
                        {tf}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-3 rounded-xl border bg-card text-card-foreground p-6 shadow-sm">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-semibold">Recent Trades</h3>
            <span className="text-xs bg-muted px-2 py-0.5 rounded-md text-muted-foreground">{data.latest_trades.length} Records</span>
          </div>
          <div className="space-y-4 max-h-[780px] overflow-y-auto pr-2 custom-scrollbar">
            {data.latest_trades.length === 0 ? (
              <div className="h-40 flex flex-col items-center justify-center border-2 border-dashed rounded-xl border-muted">
                <p className="text-muted-foreground text-xs">No trades recorded yet.</p>
              </div>
            ) : (
              data.latest_trades.map((trade, i) => (
                <div key={i} className="flex items-center justify-between p-3 rounded-xl border bg-muted/20 hover:bg-muted/40 transition-all group overflow-hidden relative">
                  <div className="flex items-center gap-3 relative z-10">
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center font-black text-sm transition-transform group-hover:scale-110 ${trade.type === 'CALL' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-800' : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-800'}`}>
                      {trade.type === 'CALL' ? 'C' : 'P'}
                    </div>
                    <div>
                      <p className="font-bold text-sm flex items-center gap-1.5">
                        {trade.symbol || "SPY"}
                        <span className="text-[10px] font-medium text-muted-foreground uppercase bg-muted/50 px-1.5 rounded-sm">0DTE</span>
                      </p>
                      <p className="text-[10px] text-muted-foreground">{new Date(trade.exit_time).toLocaleString()}</p>
                    </div>
                  </div>
                  <div className="text-right relative z-10">
                    <p className={`font-black text-sm ${parseFloat(trade.pnl_val) >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                      {parseFloat(trade.pnl_val) >= 0 ? '+' : ''}{parseFloat(trade.pnl_val).toFixed(2)}
                    </p>
                    <p className="text-[10px] font-medium text-muted-foreground">Entry: ${parseFloat(trade.entry_price).toFixed(2)}</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div >
    </div >
  );
}
