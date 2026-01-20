"use client";
import React, { useState } from 'react';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    BarChart, Bar, Cell, ReferenceLine, ReferenceArea
} from 'recharts';
import { Settings, Play, RefreshCw, AlertTriangle, CheckCircle } from 'lucide-react';

const PredictionCard = ({ prediction }) => (
    <div className="p-6 rounded-xl border bg-card text-card-foreground shadow-sm mb-6">
        <div className="flex justify-between items-start mb-4">
            <div>
                <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full animate-pulse ${prediction.signal === 'CALL' ? 'bg-green-500' : prediction.signal === 'PUT' ? 'bg-red-500' : 'bg-yellow-500'}`} />
                    <h3 className="text-xl font-black uppercase tracking-tight">
                        {prediction.signal === 'CALL' ? 'Bullish' : prediction.signal === 'PUT' ? 'Bearish' : 'Neutral'} Signal
                    </h3>
                </div>
                <p className="text-sm text-muted-foreground font-medium">Model Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
            </div>
            <div className="text-right">
                <p className="text-2xl font-black tabular-nums">${prediction.entry_price.toFixed(2)}</p>
                <p className="text-[10px] uppercase font-bold text-muted-foreground tracking-widest">Entry Price</p>
            </div>
        </div>

        {/* Sentiment Bar */}
        {prediction.probs && (
            <div className="mb-6 space-y-1.5">
                <div className="flex justify-between text-[10px] uppercase font-bold tracking-widest text-muted-foreground px-1">
                    <span>Bearish ({(prediction.probs.bearish * 100).toFixed(0)}%)</span>
                    <span>Neutral ({(prediction.probs.neutral * 100).toFixed(0)}%)</span>
                    <span>Bullish ({(prediction.probs.bullish * 100).toFixed(0)}%)</span>
                </div>
                <div className="h-2 w-full rounded-full flex overflow-hidden bg-muted/20 border border-muted">
                    <div
                        style={{ width: `${prediction.probs.bearish * 100}%` }}
                        className="h-full bg-gradient-to-r from-red-600 to-red-400 transition-all duration-500"
                    />
                    <div
                        style={{ width: `${prediction.probs.neutral * 100}%` }}
                        className="h-full bg-gray-400 transition-all duration-500"
                    />
                    <div
                        style={{ width: `${prediction.probs.bullish * 100}%` }}
                        className="h-full bg-gradient-to-r from-green-400 to-green-600 transition-all duration-500"
                    />
                </div>
            </div>
        )}

        {prediction.signal !== 'NEUTRAL' && prediction.option_data && (
            <div className="grid grid-cols-3 gap-4 mb-6 bg-muted/30 p-4 rounded-xl border border-primary/10 relative overflow-hidden group">
                <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <div className="text-center relative">
                    <p className="text-[10px] text-muted-foreground uppercase font-black tracking-widest mb-1">Target Option</p>
                    <p className="font-bold text-xs truncate">{prediction.option_data.contract}</p>
                    <p className="text-[10px] font-medium text-muted-foreground pt-1">IV: {(prediction.option_data.volatility * 100).toFixed(1)}%</p>
                </div>
                <div className="text-center relative">
                    <p className="text-[10px] text-muted-foreground uppercase font-black tracking-widest mb-1">Est. Premium</p>
                    <p className="font-black text-xl text-indigo-500 tabular-nums">${prediction.option_data.entry.toFixed(2)}</p>
                </div>
                <div className="text-center relative">
                    <p className="text-[10px] text-muted-foreground uppercase font-black tracking-widest mb-1">TP / SL Targets</p>
                    <p className="font-bold text-sm text-green-500 tabular-nums">${prediction.option_data.target.toFixed(2)}</p>
                    <p className="font-bold text-sm text-red-500 tabular-nums">${prediction.option_data.stop.toFixed(2)}</p>
                </div>
            </div>
        )}

        <div className="h-[300px]" style={{ height: "300px" }}>
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={prediction.chart_data}>
                    <defs>
                        <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#2563eb" stopOpacity={0.1} />
                            <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                    <XAxis dataKey="Datetime" hide />
                    <YAxis domain={['auto', 'auto']} stroke="hsl(var(--muted-foreground))" tickFormatter={(val) => `$${val}`} />
                    <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }} />

                    <ReferenceLine y={prediction.entry_price} stroke="blue" strokeDasharray="3 3" label="Entry" />

                    {prediction.signal === 'CALL' && (
                        <>
                            <ReferenceArea
                                y1={prediction.entry_price}
                                y2={prediction.tp_price}
                                fill="#22c55e"
                                fillOpacity={0.2}
                                stroke="#16a34a"
                                label={{ value: `TP: $${prediction.tp_price.toFixed(2)}`, position: 'insideTopRight', fill: '#16a34a', fontSize: 10 }}
                            />
                            <ReferenceArea
                                y1={prediction.entry_price}
                                y2={prediction.sl_price}
                                fill="#ef4444"
                                fillOpacity={0.2}
                                stroke="#dc2626"
                                label={{ value: `SL: $${prediction.sl_price.toFixed(2)}`, position: 'insideBottomRight', fill: '#dc2626', fontSize: 10 }}
                            />
                        </>
                    )}
                    {prediction.signal === 'PUT' && (
                        <>
                            <ReferenceArea
                                y1={prediction.entry_price}
                                y2={prediction.tp_price}
                                fill="#22c55e"
                                fillOpacity={0.2}
                                stroke="#16a34a"
                                label={{ value: `TP: $${prediction.tp_price.toFixed(2)}`, position: 'insideBottomRight', fill: '#16a34a', fontSize: 10 }}
                            />
                            <ReferenceArea
                                y1={prediction.entry_price}
                                y2={prediction.sl_price}
                                fill="#ef4444"
                                fillOpacity={0.2}
                                stroke="#dc2626"
                                label={{ value: `SL: $${prediction.sl_price.toFixed(2)}`, position: 'insideTopRight', fill: '#dc2626', fontSize: 10 }}
                            />
                        </>
                    )}

                    <Area type="monotone" dataKey="Close" stroke="hsl(var(--primary))" fillOpacity={1} fill="url(#colorPrice)" />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    </div>
);

export default function RiskManagement() {
    const [formData, setFormData] = useState({
        symbol: 'SPY',
        initial_balance: 1000,
        risk_pct: 20,
        stop_loss: 10,
        take_profit: 50,
        bullish_threshold: 70,
        bearish_threshold: 70,
        zero_dte: true
    });

    const [prediction, setPrediction] = useState(null);
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    const timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'];
    const [timeframe, setTimeframe] = useState('15m');
    const [trainingStatus, setTrainingStatus] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const handleInputChange = (e) => {
        const { name, value, type, checked } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }));
    };

    const trainCurrentModel = async () => {
        setTrainingStatus('Training...');
        try {
            const res = await fetch(`${apiUrl}/api/train/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: formData.symbol,
                    interval: timeframe
                })
            });
            const data = await res.json();
            if (res.ok) {
                setTrainingStatus('Done');
                setTimeout(() => setTrainingStatus(null), 2000);
            } else {
                setTrainingStatus('Error');
                alert("Training Error: " + data.error);
                setTimeout(() => setTrainingStatus(null), 3000);
            }
        } catch (e) {
            setTrainingStatus('Error');
            alert(e.message);
        }
    };

    const trainAndSimulate = async () => {
        setLoading(true);
        setResult(null);
        try {
            const res = await fetch(`${apiUrl}/api/backtest/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: formData.symbol,
                    interval: timeframe,
                    initial_balance: formData.initial_balance,
                    risk_pct: formData.risk_pct,
                    stop_loss: formData.stop_loss,
                    take_profit: formData.take_profit,
                    zero_dte: formData.zero_dte
                })
            });
            const data = await res.json();
            if (res.ok) {
                setResult(data);
            } else {
                alert("Backtest Error: " + data.error);
            }
        } catch (e) {
            alert(e.message);
        }
        setLoading(false);
    };

    const fetchPrediction = async () => {
        setLoading(true);
        setPrediction(null);
        try {
            const res = await fetch(`${apiUrl}/api/predict/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: formData.symbol,
                    interval: timeframe,
                    stop_loss: formData.stop_loss,
                    take_profit: formData.take_profit,
                    bullish_threshold: formData.bullish_threshold,
                    bearish_threshold: formData.bearish_threshold
                })
            });
            const data = await res.json();
            if (res.ok) {
                setPrediction(data);
            } else {
                alert("Prediction Error: " + data.error);
            }
        } catch (e) {
            alert(e.message);
        }
        setLoading(false);
    };

    return (
        <div className="space-y-6 p-4 md:p-8">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <h1 className="text-2xl md:text-3xl font-bold tracking-tight">Risk & Predictor</h1>
                <button
                    onClick={trainCurrentModel}
                    className="flex items-center justify-center gap-2 px-4 py-2 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 transition w-full md:w-auto text-sm"
                    disabled={trainingStatus !== null}
                >
                    <RefreshCw className={`w-4 h-4 ${trainingStatus === 'Training...' ? 'animate-spin' : ''}`} />
                    {trainingStatus || `Retrain ${timeframe} Model`}
                </button>
            </div>

            {/* Mobile Prediction Priority View */}
            {prediction && (
                <div className="lg:hidden animate-in fade-in slide-in-from-top-4 duration-500">
                    <PredictionCard prediction={prediction} />
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Configuration Card */}
                <div className="lg:col-span-1 space-y-6">
                    <div className="p-6 rounded-xl border bg-card text-card-foreground shadow-sm space-y-6">
                        <div className="flex items-center gap-2 pb-4 border-b">
                            <Settings className="w-5 h-5 text-primary" />
                            <h2 className="font-semibold text-lg">Configuration</h2>
                        </div>

                        <div className="space-y-4">
                            <div>
                                <label className="text-sm font-medium mb-1 block">Symbol</label>
                                <input
                                    type="text"
                                    name="symbol"
                                    value={formData.symbol}
                                    onChange={handleInputChange}
                                    className="w-full p-2 rounded-md border bg-background uppercase font-bold"
                                    placeholder="e.g. SPY"
                                />
                            </div>

                            <div>
                                <label className="text-sm font-medium mb-2 block">Timeframe Horizon</label>
                                <div className="flex flex-wrap gap-1 rounded-md shadow-sm" role="group">
                                    {timeframes.map((tf) => (
                                        <button
                                            key={tf}
                                            onClick={() => setTimeframe(tf)}
                                            className={`px-3 py-1 text-xs font-medium border first:rounded-l-md last:rounded-r-md transition-colors
                        ${timeframe === tf
                                                    ? 'bg-primary text-primary-foreground border-primary'
                                                    : 'bg-background text-foreground border-input hover:bg-accent hover:text-accent-foreground'}`}
                                        >
                                            {tf}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div>
                                <label className="text-sm font-medium mb-1 block">Initial Balance ($)</label>
                                <input
                                    type="number"
                                    name="initial_balance"
                                    value={formData.initial_balance}
                                    onChange={handleInputChange}
                                    className="w-full p-2 rounded-md border bg-background"
                                />
                            </div>

                            <div>
                                <label className="text-sm font-medium mb-1 block">Risk Per Trade (%)</label>
                                <input
                                    type="number"
                                    name="risk_pct"
                                    value={formData.risk_pct}
                                    onChange={handleInputChange}
                                    className="w-full p-2 rounded-md border bg-background"
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="text-sm font-medium mb-1 block uppercase text-[10px] tracking-widest text-muted-foreground">Stop Loss (%)</label>
                                    <input
                                        type="number"
                                        name="stop_loss"
                                        value={formData.stop_loss}
                                        onChange={handleInputChange}
                                        className="w-full p-2 rounded-md border bg-background text-red-500 font-bold"
                                    />
                                </div>
                                <div>
                                    <label className="text-sm font-medium mb-1 block uppercase text-[10px] tracking-widest text-muted-foreground">Take Profit (%)</label>
                                    <input
                                        type="number"
                                        name="take_profit"
                                        value={formData.take_profit}
                                        onChange={handleInputChange}
                                        className="w-full p-2 rounded-md border bg-background text-green-500 font-bold"
                                    />
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4 pt-2 border-t border-dashed">
                                <div>
                                    <label className="text-sm font-medium mb-1 block uppercase text-[10px] tracking-widest text-muted-foreground">Bullish Threshold (%)</label>
                                    <input
                                        type="number"
                                        name="bullish_threshold"
                                        value={formData.bullish_threshold}
                                        onChange={handleInputChange}
                                        className="w-full p-2 rounded-md border bg-background text-green-600 font-black"
                                        min="0"
                                        max="100"
                                    />
                                </div>
                                <div>
                                    <label className="text-sm font-medium mb-1 block uppercase text-[10px] tracking-widest text-muted-foreground">Bearish Threshold (%)</label>
                                    <input
                                        type="number"
                                        name="bearish_threshold"
                                        value={formData.bearish_threshold}
                                        onChange={handleInputChange}
                                        className="w-full p-2 rounded-md border bg-background text-red-600 font-black"
                                        min="0"
                                        max="100"
                                    />
                                </div>
                            </div>

                            <div className="flex items-center justify-between p-3 rounded-lg border bg-muted/20">
                                <span className="text-sm font-medium">Enable 0DTE Strategy</span>
                                <input
                                    type="checkbox"
                                    name="zero_dte"
                                    checked={formData.zero_dte}
                                    onChange={handleInputChange}
                                    className="w-5 h-5 accent-primary"
                                />
                            </div>

                            <div className="flex flex-col gap-2">
                                <button
                                    onClick={fetchPrediction}
                                    disabled={loading}
                                    className="w-full py-3 bg-indigo-600 text-white font-bold rounded-lg hover:bg-indigo-700 transition flex justify-center items-center gap-2"
                                >
                                    {loading ? <RefreshCw className="animate-spin w-5 h-5" /> : <CheckCircle className="w-5 h-5" />}
                                    Get Prediction
                                </button>
                                <button
                                    onClick={trainAndSimulate}
                                    disabled={loading}
                                    className="w-full py-3 bg-secondary text-secondary-foreground font-bold rounded-lg hover:bg-secondary/80 transition flex justify-center items-center gap-2"
                                >
                                    {loading ? <RefreshCw className="animate-spin w-5 h-5" /> : <Play className="w-5 h-5" />}
                                    Run Backtest
                                </button>
                            </div>
                        </div>
                    </div>
                    {/* Desktop Prediction Card */}
                    {prediction && (
                        <div className="hidden lg:block">
                            <PredictionCard prediction={prediction} />
                        </div>
                    )}
                </div>

                {/* Results Area */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Backtest Results */}
                    {result ? (
                        <>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div className="p-4 rounded-xl border bg-card shadow-sm">
                                    <p className="text-xs md:text-sm text-muted-foreground">Final Balance</p>
                                    <p className="text-lg md:text-2xl font-bold">${result.final_balance.toFixed(2)}</p>
                                </div>
                                <div className="p-4 rounded-xl border bg-card shadow-sm">
                                    <p className="text-xs md:text-sm text-muted-foreground">Total Trades</p>
                                    <p className="text-lg md:text-2xl font-bold">{result.total_trades}</p>
                                </div>
                                <div className="p-4 rounded-xl border bg-card shadow-sm">
                                    <p className="text-xs md:text-sm text-muted-foreground">Win Rate</p>
                                    <p className="text-lg md:text-2xl font-bold">{(result.summary.win_rate * 100).toFixed(1)}%</p>
                                </div>
                                <div className="p-4 rounded-xl border bg-card shadow-sm">
                                    <p className="text-xs md:text-sm text-muted-foreground">Profit Factor</p>
                                    <p className="text-lg md:text-2xl font-bold">
                                        {(result.trades.filter(t => t.pnl_val > 0).reduce((a, b) => a + b.pnl_val, 0) /
                                            Math.abs(result.trades.filter(t => t.pnl_val < 0).reduce((a, b) => a + b.pnl_val, 0) || 1)).toFixed(2)}
                                    </p>
                                </div>
                            </div>

                            <div className="p-6 rounded-xl border bg-card shadow-sm h-[400px]" style={{ height: "400px" }}>
                                <h3 className="font-semibold mb-4">Equity Curve ({timeframe})</h3>
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={result.equity_curve}>
                                        <defs>
                                            <linearGradient id="colorEq" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#8884d8" stopOpacity={0.1} />
                                                <stop offset="95%" stopColor="#8884d8" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                                        <XAxis dataKey="Datetime" hide />
                                        <YAxis domain={['auto', 'auto']} stroke="hsl(var(--muted-foreground))" tickFormatter={(val) => `$${val}`} />
                                        <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }} />
                                        <Area type="monotone" dataKey="equity" stroke="hsl(var(--primary))" fillOpacity={1} fill="url(#colorEq)" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>

                            <div className="p-6 rounded-xl border bg-card shadow-sm h-[400px]" style={{ height: "400px" }}>
                                <h3 className="font-semibold mb-4">Trade P&L Distribution (%)</h3>
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={result.trades}>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                                        <XAxis dataKey="exit_time" hide />
                                        <YAxis stroke="hsl(var(--muted-foreground))" tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} />
                                        <Tooltip
                                            formatter={(value) => `${(value * 100).toFixed(2)}%`}
                                            contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }}
                                        />
                                        <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" />
                                        <Bar dataKey="pnl_pct">
                                            {result.trades.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.pnl_pct >= 0 ? '#22c55e' : '#ef4444'} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </>
                    ) : (
                        <div className="h-full flex flex-col items-center justify-center p-12 border-2 border-dashed rounded-xl text-muted-foreground">
                            <AlertTriangle className="w-12 h-12 mb-4 opacity-50" />
                            <p className="text-lg">No backtest results yet</p>
                            <p className="text-sm">Click "Run Train & Backtest"</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
