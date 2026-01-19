"use client";
import React, { useState, useEffect } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceArea
} from 'recharts';
import { Activity, TrendingUp, DollarSign, Shield, RefreshCw } from 'lucide-react';
import StatCard from '@/components/StatCard';
import { ThemeToggle } from '@/components/ThemeToggle';

export default function Dashboard() {
  const [data, setData] = useState({
    latest_trades: [],
    performance: [],
    equity_curve: []
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
    take_profit: 50
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
  }, []);

  // Calculate stats from trades if performance is empty (fallback)
  const totalEquity = alpacaAccount ? alpacaAccount.equity : (1000 + (data.latest_trades.reduce((acc, t) => acc + (parseFloat(t.pnl_val) || 0), 0)));
  const buyingPower = alpacaAccount ? alpacaAccount.buying_power : 0;
  const winRate = data.latest_trades.filter(t => parseFloat(t.pnl_val) > 0).length / (data.latest_trades.length || 1) * 100;

  return (
    <div className="flex-1 space-y-4 p-4 md:p-8 pt-6 min-h-screen bg-background">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 md:gap-0">
        <div className="flex items-center gap-4">
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground">Dashboard</h2>
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
        <StatCard title="Day Trades Today" value={alpacaAccount ? alpacaAccount.daytrade_count.toString() : data.latest_trades.length.toString()} icon={Activity} />
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
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.latest_trades.slice().reverse()}>
                  <defs>
                    <linearGradient id="colorPnL" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8884d8" stopOpacity={0.1} />
                      <stop offset="95%" stopColor="#8884d8" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                  <XAxis dataKey="exit_time" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => `$${value}`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))', color: 'hsl(var(--card-foreground))' }}
                  />
                  <Area type="monotone" dataKey="pnl_val" stroke="hsl(var(--primary))" strokeWidth={2} fillOpacity={1} fill="url(#colorPnL)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Neural Predictor Card - Redesigned to match UI */}
          <div className="rounded-xl border bg-card text-card-foreground p-6 shadow-sm">
            <div className="flex justify-between items-center mb-6">
              <div className="flex items-center gap-3">
                <Shield className="w-5 h-5 text-primary" />
                <h3 className="text-lg font-bold">Neural Trading Signal</h3>
              </div>
              <button
                onClick={fetchPrediction}
                className="text-xs text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
                disabled={predicting}
              >
                <RefreshCw size={12} className={predicting ? "animate-spin" : ""} />
                Update Prediction
              </button>
            </div>

            {prediction ? (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-4 pb-4 border-b border-border">
                  <div>
                    <p className="text-[10px] uppercase font-bold text-muted-foreground tracking-widest mb-1">Signal Status</p>
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${prediction.signal === 'CALL' ? 'bg-green-500' : prediction.signal === 'PUT' ? 'bg-red-500' : 'bg-yellow-500'}`} />
                      <p className="font-black text-xl italic">{prediction.signal} SIGNAL</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-[10px] uppercase font-bold text-muted-foreground tracking-widest mb-1">Confidence</p>
                    <p className="font-black text-xl">{(prediction.confidence * 100).toFixed(1)}%</p>
                  </div>
                </div>

                <div className="relative pt-4">
                  <span className="absolute -top-1 left-0 text-[10px] font-medium text-muted-foreground uppercase italic underline decoration-primary/30">chart</span>
                  <div className="h-[350px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={prediction.chart_data}>
                        <defs>
                          <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.1} />
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="2 2" vertical={false} stroke="hsl(var(--border))" opacity={0.5} />
                        <XAxis
                          dataKey="Datetime"
                          hide={false}
                          stroke="hsl(var(--muted-foreground))"
                          fontSize={9}
                          label={{ value: "time interval", position: 'insideBottomRight', offset: 0, fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                        />
                        <YAxis domain={['auto', 'auto']} hide />
                        <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))', fontSize: 10 }} />

                        {/* Prediction Zones - TradingView Style */}
                        <ReferenceLine
                          y={prediction.entry_price}
                          stroke="#3b82f6"
                          strokeWidth={2}
                          strokeDasharray="3 3"
                          label={{ value: 'enter point', position: 'right', fill: '#3b82f6', fontSize: 12, fontWeight: 'bold', offset: 10 }}
                        />

                        {prediction.signal === 'CALL' ? (
                          <>
                            {/* Take Profit Zone (Top) */}
                            <ReferenceArea
                              y1={prediction.entry_price}
                              y2={prediction.tp_price}
                              fill="#22c55e"
                              fillOpacity={0.25}
                              stroke="#22c55e"
                              strokeWidth={1}
                            />
                            <ReferenceLine
                              y={prediction.tp_price}
                              stroke="#22c55e"
                              strokeWidth={1}
                              label={{ value: 'Take profit', position: 'right', fill: '#22c55e', fontSize: 12, fontWeight: 'bold', offset: 10 }}
                            />

                            {/* Stop Loss Zone (Bottom) */}
                            <ReferenceArea
                              y1={prediction.entry_price}
                              y2={prediction.sl_price}
                              fill="#ef4444"
                              fillOpacity={0.25}
                              stroke="#ef4444"
                              strokeWidth={1}
                            />
                            <ReferenceLine
                              y={prediction.sl_price}
                              stroke="#ef4444"
                              strokeWidth={1}
                              label={{ value: 'stop loss 5% 10%', position: 'right', fill: '#ef4444', fontSize: 12, fontWeight: 'bold', offset: 10 }}
                            />
                          </>
                        ) : (
                          <>
                            {/* Stop Loss Zone (Top for PUT) */}
                            <ReferenceArea
                              y1={prediction.entry_price}
                              y2={prediction.sl_price}
                              fill="#ef4444"
                              fillOpacity={0.25}
                              stroke="#ef4444"
                              strokeWidth={1}
                            />
                            <ReferenceLine
                              y={prediction.sl_price}
                              stroke="#ef4444"
                              strokeWidth={1}
                              label={{ value: 'stop loss 5% 10%', position: 'right', fill: '#ef4444', fontSize: 12, fontWeight: 'bold', offset: 10 }}
                            />

                            {/* Take Profit Zone (Bottom for PUT) */}
                            <ReferenceArea
                              y1={prediction.entry_price}
                              y2={prediction.tp_price}
                              fill="#22c55e"
                              fillOpacity={0.25}
                              stroke="#22c55e"
                              strokeWidth={1}
                            />
                            <ReferenceLine
                              y={prediction.tp_price}
                              stroke="#22c55e"
                              strokeWidth={1}
                              label={{ value: 'Take profit', position: 'right', fill: '#22c55e', fontSize: 12, fontWeight: 'bold', offset: 10 }}
                            />
                          </>
                        )}

                        <Area
                          type="bundle"
                          dataKey="Close"
                          stroke="hsl(var(--primary))"
                          strokeWidth={2}
                          fillOpacity={1}
                          fill="url(#colorPrice)"
                          animationDuration={1500}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
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
      </div>
    </div>
  );
}
