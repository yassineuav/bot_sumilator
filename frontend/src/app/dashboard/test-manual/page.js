"use client";
import React, { useState, useEffect, useMemo } from 'react';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    ReferenceLine, ReferenceArea, BarChart, Bar, Cell
} from 'recharts';
import PredictionChart from '@/components/PredictionChart';
import {
    Settings, Play, RefreshCw, AlertTriangle,
    CheckCircle, Clock, ArrowRight, History,
    TrendingUp, TrendingDown, Target, Shield,
    ChevronRight, FastForward, Rewind, Trash2,
    Activity, Zap, Info
} from 'lucide-react';
import TimelineSlider from '@/components/TimelineSlider';

// --- Black-Scholes Math Utils ---
const normPDF = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

const normCDF = (x) => {
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989423 * Math.exp(-x * x / 2);
    const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return x > 0 ? 1 - p : p;
};

const calculateGreeks = (S, K, T, r, sigma, type) => {
    if (T <= 0) {
        return {
            price: type === 'CALL' ? Math.max(0, S - K) : Math.max(0, K - S),
            delta: type === 'CALL' ? (S > K ? 1 : 0) : (S < K ? -1 : 0),
            theta: 0
        };
    }

    const d1 = (Math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);

    let price, delta, theta;
    if (type === 'CALL') {
        price = S * normCDF(d1) - K * Math.exp(-r * T) * normCDF(d2);
        delta = normCDF(d1);
        theta = -(S * normPDF(d1) * sigma) / (2 * Math.sqrt(T)) - r * K * Math.exp(-r * T) * normCDF(d2);
    } else {
        price = K * Math.exp(-r * T) * normCDF(-d2) - S * normCDF(-d1);
        delta = normCDF(d1) - 1;
        theta = -(S * normPDF(d1) * sigma) / (2 * Math.sqrt(T)) + r * K * Math.exp(-r * T) * normCDF(-d2);
    }

    return { price, delta, theta: theta / 365.0 }; // Daily Theta
};
// -------------------------------

export default function TestManual() {
    const [config, setConfig] = useState({
        symbol: 'SPY',
        interval: '15m',
        symbol: 'SPY',
        interval: '15m',
        period: '60d',
        model_type: 'xgb'
    });
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    const [historyData, setHistoryData] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [predicting, setPredicting] = useState(false);
    const [manualHistory, setManualHistory] = useState([]);
    const [autoPredict, setAutoPredict] = useState(false);
    const [autoBacktesting, setAutoBacktesting] = useState(false);

    // Balance Management
    const [balance, setBalance] = useState(1000);

    // Strategy / Trade Inputs
    const [tradeInputs, setTradeInputs] = useState({
        risk_pct: 10,
        stop_loss_pct: 50,
        take_profit_pct: 200,
        position_size: 20 // Defaulting to 20% as requested
    });

    // Active Trade Simulation
    const [activeTrade, setActiveTrade] = useState(null);
    const [currentOptionPrice, setCurrentOptionPrice] = useState(null);
    const [greeks, setGreeks] = useState({ delta: 0, theta: 0 });
    const [simulationResult, setSimulationResult] = useState(null);

    const timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'];

    // Auto-predict effect
    useEffect(() => {
        if (autoPredict && !predicting && !activeTrade) {
            runPredictor();
        }
    }, [currentIndex, autoPredict]);

    useEffect(() => {
        fetchHistoryData();
        fetchManualHistory();
    }, [config.symbol, config.interval]);

    const fetchHistoryData = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${apiUrl}/api/history-data/?symbol=${config.symbol}&interval=${config.interval}&period=${config.period}`);
            const data = await res.json();
            if (res.ok) {
                setHistoryData(data);
                setCurrentIndex(data.length - 1);
            } else {
                console.error("Failed to fetch history:", data.error);
            }
        } catch (e) {
            console.error("Fetch error:", e);
        }
        setLoading(false);
    };

    const fetchManualHistory = async () => {
        try {
            const res = await fetch(`${apiUrl}/api/manual-test/history/`);
            const data = await res.json();
            if (res.ok) setManualHistory(data);
        } catch (e) {
            console.error("History fetch error:", e);
        }
    };

    const clearManualHistory = async () => {
        if (!confirm("Are you sure you want to clear all manual test history?")) return;
        try {
            const res = await fetch(`${apiUrl}/api/manual-test/clear/`, {
                method: 'DELETE'
            });
            if (res.ok) {
                setManualHistory([]);
            } else {
                alert("Failed to clear history");
            }
        } catch (e) {
            alert(e.message);
        }
    };

    const runPredictor = async () => {
        if (!historyData[currentIndex]) return;

        setPredicting(true);
        setPrediction(null);
        try {
            const currentPoint = historyData[currentIndex];
            const res = await fetch(`${apiUrl}/api/manual-test/run/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: config.symbol,
                    interval: config.interval,
                    timestamp: currentPoint.Datetime,
                    model_type: config.model_type
                })
            });
            const data = await res.json();
            if (res.ok) {
                setPrediction(data);
            } else {
                alert("Error: " + data.error);
            }
        } catch (e) {
            alert(e.message);
        }
        setPredicting(false);
    };

    const runAutoBacktest = async () => {
        if (!historyData[currentIndex]) return;
        if (!confirm(`Start auto backtest from ${historyData[currentIndex].Datetime} until today? This will overwrite/append to your journal.`)) return;

        setAutoBacktesting(true);
        try {
            const currentPoint = historyData[currentIndex];
            const res = await fetch(`${apiUrl}/api/manual-test/auto-backtest/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: config.symbol,
                    interval: config.interval,
                    timestamp: currentPoint.Datetime,
                    risk_pct: tradeInputs.risk_pct,
                    stop_loss_pct: tradeInputs.stop_loss_pct,
                    take_profit_pct: tradeInputs.take_profit_pct,
                    position_size: tradeInputs.position_size
                })
            });
            const data = await res.json();
            if (res.ok) {
                alert(data.message);
                fetchManualHistory();
            } else {
                alert("Error: " + data.error);
            }
        } catch (e) {
            alert(e.message);
        }
        setAutoBacktesting(false);
    };

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        if (value === '') {
            setTradeInputs(prev => ({ ...prev, [name]: '' }));
            return;
        }
        const parsed = parseFloat(value);
        if (!isNaN(parsed)) {
            setTradeInputs(prev => ({ ...prev, [name]: parsed }));
        }
    };

    // Filtered data for the chart (up to current index)
    const visibleData = useMemo(() => {
        return historyData.slice(Math.max(0, currentIndex - 200), currentIndex + 1);
    }, [historyData, currentIndex]);

    // Journal Stats calculation
    const stats = useMemo(() => {
        const totalTrades = manualHistory.length;
        const totalWins = manualHistory.filter(t => t.result === 'TP').length;
        const totalLosses = manualHistory.filter(t => t.result === 'SL').length;
        const winRate = totalTrades > 0 ? (totalWins / totalTrades) * 100 : 0;
        return { totalTrades, totalWins, totalLosses, winRate };
    }, [manualHistory]);

    const startTrade = () => {
        if (!prediction || !prediction.option) return;

        const entryPoint = historyData[currentIndex];
        const option = prediction.option;

        // Realistic 0DTE Initialization - Sync with Backend ManualTester
        // Backend uses T = 0.5 / 365.0 (which is 12 hours)
        const initialHoursLeft = 12.0;
        const T = initialHoursLeft / (24 * 365);
        const r = 0.05;
        const sigma = 0.20; // Sync with backend vol

        // Use the premium from the recommendation as the starting point
        const entryPremium = option.premium;

        // Calculate initial greeks at entry
        const res = calculateGreeks(entryPoint.Close, option.strike, T, r, sigma, option.type);

        const newTrade = {
            entry_timestamp: entryPoint.Datetime,
            entry_index: currentIndex,
            entry_price: entryPoint.Close,
            symbol: config.symbol,
            type: option.type,
            option_strike: option.strike,
            option_premium: entryPremium, // Use the recommended premium
            initial_hours_left: initialHoursLeft,
            sigma: sigma,
            r: r,
            tp_price: entryPremium * (1 + tradeInputs.take_profit_pct / 100),
            sl_price: entryPremium * (1 - tradeInputs.stop_loss_pct / 100),
            size_pct: tradeInputs.position_size,
            status: 'OPEN',
            steps: 0
        };

        setActiveTrade(newTrade);
        setCurrentOptionPrice(entryPremium);
        setGreeks({ delta: res.delta, theta: res.theta });
        setSimulationResult(null);
    };

    const advanceTimeAndCheck = () => {
        if (!activeTrade || currentIndex >= historyData.length - 1) return;

        const nextIndex = currentIndex + 1;
        setCurrentIndex(nextIndex);

        const currentPoint = historyData[nextIndex];
        const stepsElapsed = nextIndex - activeTrade.entry_index;

        // Calculate time decay based on interval
        const intervalMinutes = parseInt(config.interval) || 15; // fallback to 15 if parse fails (e.g. '1d')
        // Special case for daily: maybe 6.5 hours per day
        const minutesToSub = config.interval.includes('d') ? 390 : intervalMinutes;

        const currentHoursLeft = Math.max(0, activeTrade.initial_hours_left - (stepsElapsed * minutesToSub / 60));
        const T = currentHoursLeft / (24 * 365);

        const res = calculateGreeks(
            currentPoint.Close,
            activeTrade.option_strike,
            T,
            activeTrade.r,
            activeTrade.sigma,
            activeTrade.type
        );

        setCurrentOptionPrice(res.price);
        setGreeks({ delta: res.delta, theta: res.theta });

        if (res.price >= activeTrade.tp_price) {
            finalizeTrade('TP', res.price);
        } else if (res.price <= activeTrade.sl_price) {
            finalizeTrade('SL', res.price);
        }
    };

    const finalizeTrade = async (result, finalPremium) => {
        const pnl_pct = (finalPremium - activeTrade.option_premium) / activeTrade.option_premium;
        const entryPoint = historyData[activeTrade.entry_index];
        const investment = (activeTrade.size_pct / 100) * balance;
        const pnl_val = investment * pnl_pct;

        const payload = {
            symbol: activeTrade.symbol,
            timestamp: entryPoint.Datetime,
            prediction: prediction.prediction.signal,
            confidence: prediction.prediction.confidence,
            option_strike: activeTrade.option_strike,
            option_premium: activeTrade.option_premium,
            option_type: activeTrade.type,
            entry_price: activeTrade.entry_price,
            take_profit: activeTrade.tp_price,
            stop_loss: activeTrade.sl_price,
            position_size: activeTrade.size_pct,
            result: result,
            pnl_pct: pnl_pct * 100,
            pnl_val: pnl_val
        };

        try {
            const res = await fetch(`${apiUrl}/api/manual-test/save/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (res.ok) {
                setSimulationResult({ result, pnl: pnl_pct * 100, pnl_val });
                setBalance(prev => prev + pnl_val);
                setActiveTrade(null);
                setCurrentOptionPrice(null);
                fetchManualHistory();
            }
        } catch (e) {
            console.error("Save error:", e);
        }
    };

    return (
        <div className="flex flex-col gap-4 md:gap-6 animate-in fade-in duration-500 p-4 md:p-0">
            {/* Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center bg-card p-6 rounded-2xl border shadow-sm gap-4">
                <div>
                    <h1 className="text-2xl md:text-3xl font-black tracking-tight flex items-center gap-3">
                        <History className="text-primary w-8 h-8" />
                        TEST MANUAL
                    </h1>
                    <p className="text-xs md:text-sm text-muted-foreground font-medium">Verify human psychology vs AI logic in historical back-in-time test mode.</p>
                </div>

                <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 w-full md:w-auto">
                    <div className="bg-background border px-4 py-2 rounded-xl flex flex-col items-center sm:items-end justify-center">
                        <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Account Balance</p>
                        <p className="text-lg md:text-xl font-black tabular-nums text-primary">${balance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
                    </div>
                    <div className="flex bg-muted p-1 rounded-lg border overflow-x-auto no-scrollbar">
                        {timeframes.map(tf => (
                            <button
                                key={tf}
                                onClick={() => setConfig(prev => ({ ...prev, interval: tf }))}
                                className={`px-3 md:px-4 py-1.5 rounded-md text-[10px] md:text-xs font-bold transition-all whitespace-nowrap ${config.interval === tf ? 'bg-background shadow-sm text-primary' : 'text-muted-foreground hover:text-foreground'}`}
                            >
                                {tf}
                            </button>
                        ))}
                    </div>

                    <div className="flex bg-muted p-1 rounded-lg border">
                        {['xgb', 'lstm', 'hybrid'].map(m => (
                            <button
                                key={m}
                                onClick={() => setConfig(prev => ({ ...prev, model_type: m }))}
                                className={`px-3 md:px-4 py-1.5 rounded-md text-[10px] md:text-xs font-bold transition-all uppercase ${config.model_type === m ? 'bg-background shadow-sm text-indigo-500' : 'text-muted-foreground hover:text-foreground'}`}
                            >
                                {m}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Main Content (Chart & Timeline) */}
                <div className="lg:col-span-8 space-y-6">
                    {/* Timeline Slider */}
                    <TimelineSlider
                        data={historyData}
                        currentIndex={currentIndex}
                        onChange={setCurrentIndex}
                        interval={config.interval}
                    />

                    {/* Main Pricing Chart */}
                    <div className="bg-card border rounded-2xl p-4 md:p-6 shadow-sm min-h-[400px] md:min-h-[500px] relative overflow-hidden">
                        {loading && (
                            <div className="absolute inset-0 bg-background/50 backdrop-blur-sm z-50 flex items-center justify-center">
                                <RefreshCw className="w-10 h-10 animate-spin text-primary" />
                            </div>
                        )}

                        <div className="h-[350px] md:h-[450px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={visibleData}>
                                    <defs>
                                        <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.15} />
                                            <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                                    <XAxis
                                        dataKey="Datetime"
                                        hide={true}
                                    />
                                    <YAxis
                                        domain={['auto', 'auto']}
                                        stroke="hsl(var(--muted-foreground))"
                                        tickFormatter={val => `$${val.toFixed(1)}`}
                                        className="text-[10px] font-bold"
                                        orientation="right"
                                    />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: 'hsl(var(--card))', borderRadius: '12px', border: '1px solid hsl(var(--border))', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                                        labelStyle={{ fontWeight: 'bold', fontSize: '12px' }}
                                    />

                                    <Area
                                        type="monotone"
                                        dataKey="Close"
                                        stroke="hsl(var(--primary))"
                                        strokeWidth={2}
                                        fillOpacity={1}
                                        fill="url(#colorPrice)"
                                        animationDuration={300}
                                    />

                                    {/* Cursor logic marker */}
                                    <ReferenceLine x={historyData[currentIndex]?.Datetime} stroke="hsl(var(--primary))" strokeWidth={2} />

                                    {activeTrade && (
                                        <ReferenceLine y={activeTrade.entry_price} stroke="yellow" strokeDasharray="5 5" label={{ value: 'MANUAL ENTRY', position: 'left', fill: 'yellow', fontSize: 10, fontWeight: 'bold' }} />
                                    )}
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Auto Backtest Trigger */}
                        <div className="absolute bottom-4 left-6 right-6 flex justify-between items-center bg-background/80 backdrop-blur-md p-3 rounded-xl border shadow-lg">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-primary/10 rounded-lg">
                                    <Activity className="w-5 h-5 text-primary" />
                                </div>
                                <div>
                                    <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Automation</p>
                                    <p className="text-xs font-bold">Fast-forward from this point</p>
                                </div>
                            </div>
                            <button
                                onClick={runAutoBacktest}
                                disabled={autoBacktesting || loading}
                                className="flex items-center gap-2 bg-primary hover:bg-primary/90 text-primary-foreground px-6 py-2.5 rounded-lg text-sm font-black transition shadow-lg disabled:opacity-50"
                            >
                                {autoBacktesting ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                                {autoBacktesting ? 'RUNNING BACKTEST...' : 'START AUTO BACKTEST'}
                            </button>
                        </div>
                    </div>

                    {/* Simulation Controls (Visible when trade is active) */}
                    {activeTrade && (
                        <div className="bg-primary/5 border-2 border-primary border-dashed rounded-2xl p-6 space-y-4 animate-in zoom-in-95 duration-300">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-4">
                                    <div className={`p-3 rounded-full ${activeTrade.type === 'CALL' ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500'}`}>
                                        {activeTrade.type === 'CALL' ? <TrendingUp className="w-8 h-8" /> : <TrendingDown className="w-8 h-8" />}
                                    </div>
                                    <div>
                                        <p className="text-sm font-black uppercase tracking-widest text-primary">Active {activeTrade.type} Trade (Strike: ${activeTrade.option_strike})</p>
                                        <div className="flex items-center gap-6">
                                            <p className="text-2xl font-black">Option Price: <span className="text-indigo-500">${currentOptionPrice?.toFixed(2)}</span></p>
                                            <div className="flex items-center gap-4 bg-background/50 px-3 py-1 rounded-lg border border-primary/20">
                                                <div className="flex items-center gap-1">
                                                    <Zap className="w-3 h-3 text-yellow-500" />
                                                    <span className="text-[10px] font-black">Δ {greeks.delta.toFixed(2)}</span>
                                                </div>
                                                <div className="flex items-center gap-1">
                                                    <Clock className="w-3 h-3 text-blue-500" />
                                                    <span className="text-[10px] font-black">Θ {greeks.theta.toFixed(3)}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className="text-xs font-bold text-muted-foreground uppercase">Exit Targets</p>
                                    <p className="font-black">TP: <span className="text-green-500">${activeTrade.tp_price.toFixed(2)}</span> | SL: <span className="text-red-500">${activeTrade.sl_price.toFixed(2)}</span></p>
                                </div>
                            </div>

                            <div className="flex justify-center pt-2">
                                <button
                                    onClick={advanceTimeAndCheck}
                                    className="flex items-center gap-3 px-12 py-5 bg-primary text-primary-foreground rounded-2xl font-black hover:scale-105 transition shadow-indigo-500/25 shadow-2xl"
                                >
                                    <FastForward className="w-6 h-6" />
                                    ADVANCE TO NEXT CANDLE
                                </button>
                            </div>
                        </div>
                    )}

                    {simulationResult && (
                        <div className={`p-6 rounded-2xl border-2 flex items-center justify-between animate-in slide-in-from-top-4 duration-500 ${simulationResult.result === 'TP' ? 'bg-green-500/10 border-green-500' : 'bg-red-500/10 border-red-500'}`}>
                            <div className="flex items-center gap-4">
                                {simulationResult.result === 'TP' ? <CheckCircle className="w-10 h-10 text-green-500" /> : <AlertTriangle className="w-10 h-10 text-red-500" />}
                                <div>
                                    <p className="text-2xl font-black uppercase tracking-tighter">Trade Finalized: {simulationResult.result === 'TP' ? 'Profit Hit!' : 'Stop Loss Hit'}</p>
                                    <div className="flex gap-4">
                                        <p className={`text-xl font-bold ${simulationResult.pnl > 0 ? 'text-green-500' : 'text-red-500'}`}>
                                            PnL: {simulationResult.pnl > 0 ? '+' : ''}{simulationResult.pnl.toFixed(2)}%
                                        </p>
                                        <p className={`text-xl font-bold ${simulationResult.pnl_val >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                                            ({simulationResult.pnl_val >= 0 ? '+' : ''}${Math.abs(simulationResult.pnl_val).toFixed(2)})
                                        </p>
                                    </div>
                                </div>
                            </div>
                            <button onClick={() => setSimulationResult(null)} className="p-3 bg-muted hover:bg-muted/80 rounded-full transition">
                                <ArrowRight className="w-6 h-6" />
                            </button>
                        </div>
                    )}
                </div>

                {/* Sidebar (Controls & Prediction) */}
                <div className="lg:col-span-4 space-y-6">
                    {/* Manual Run Card */}
                    <div className="bg-card border rounded-2xl p-6 shadow-sm space-y-6">
                        <div className="flex items-center justify-between border-b pb-4">
                            <h2 className="text-lg font-black tracking-tight">PREDICTOR CONTROLS</h2>
                            <button
                                onClick={runPredictor}
                                disabled={predicting || loading}
                                className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-xs font-bold transition disabled:opacity-50"
                            >
                                {predicting ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                                RUN AI PREDICTOR
                            </button>
                        </div>

                        <div className="flex items-center gap-2 px-1">
                            <input
                                type="checkbox"
                                id="autoPredict"
                                checked={autoPredict}
                                onChange={(e) => setAutoPredict(e.target.checked)}
                                className="w-4 h-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-600 cursor-pointer"
                            />
                            <label htmlFor="autoPredict" className="text-xs font-bold text-muted-foreground cursor-pointer select-none">
                                Auto-check on next candle
                            </label>
                        </div>

                        {prediction && (
                            <div className="space-y-6 animate-in fade-in duration-500">
                                <div className="flex justify-between items-center bg-muted/30 p-4 rounded-xl border border-primary/10">
                                    <div>
                                        <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">AI Signal</p>
                                        <p className={`text-2xl font-black italic tracking-tighter ${prediction.prediction.signal === 'CALL' ? 'text-green-500' : prediction.prediction.signal === 'PUT' ? 'text-red-500' : 'text-yellow-500'}`}>
                                            {prediction.prediction.signal === 'CALL' ? 'BULLISH' : prediction.prediction.signal === 'PUT' ? 'BEARISH' : 'NEUTRAL'}
                                        </p>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Confidence</p>
                                        <p className="text-2xl font-black tabular-nums">{(prediction.prediction.confidence * 100).toFixed(1)}%</p>
                                    </div>
                                </div>

                                <div className="border rounded-xl overflow-hidden bg-background">
                                    <div className="bg-muted px-3 py-1 text-[10px] font-bold text-muted-foreground uppercase flex justify-between items-center">
                                        <span>Signal Visualization</span>
                                        <span>15m Interval</span>
                                    </div>
                                    <div className="h-[200px]">
                                        <PredictionChart
                                            data={historyData.slice(Math.max(0, currentIndex - 50), currentIndex + 1)}
                                            prediction={{
                                                ...prediction.prediction,
                                                // Ensure we have entry/tp/sl from the prediction or calculate them if missing
                                                entry_price: prediction.prediction.entry_price || historyData[currentIndex].Close,
                                                tp_price: prediction.prediction.tp_price,
                                                sl_price: prediction.prediction.sl_price
                                            }}
                                            height={200}
                                        />
                                    </div>
                                </div>

                                {prediction.option && (
                                    <div className="space-y-4">
                                        <div className="bg-card border border-dashed p-4 rounded-xl space-y-3">
                                            <div className="flex items-center gap-2 mb-2">
                                                <Target className="w-4 h-4 text-primary" />
                                                <span className="text-xs font-black uppercase tracking-widest">Recommended OTM Strike</span>
                                            </div>
                                            <div className="grid grid-cols-2 gap-4">
                                                <div className="bg-muted/50 p-2 rounded-lg">
                                                    <p className="text-[10px] uppercase font-bold text-muted-foreground">Strike Price</p>
                                                    <p className="text-lg font-black">${prediction.option.strike}</p>
                                                </div>
                                                <div className="bg-muted/50 p-2 rounded-lg">
                                                    <p className="text-[10px] uppercase font-bold text-muted-foreground">Premium</p>
                                                    <p className="text-lg font-black text-indigo-500">${prediction.option.premium.toFixed(2)}</p>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="space-y-4">
                                            <h3 className="text-xs font-black uppercase tracking-widest flex items-center gap-2">
                                                <Shield className="w-4 h-4 text-primary" />
                                                Manual Risk Settings
                                            </h3>
                                            <div className="grid grid-cols-2 gap-4">
                                                <div className="space-y-1">
                                                    <label className="text-[10px] font-bold uppercase text-muted-foreground">TP Target (%)</label>
                                                    <input
                                                        type="number"
                                                        name="take_profit_pct"
                                                        value={tradeInputs.take_profit_pct}
                                                        onChange={handleInputChange}
                                                        className="w-full bg-muted border rounded-lg p-2 font-black text-green-500"
                                                    />
                                                </div>
                                                <div className="space-y-1">
                                                    <label className="text-[10px] font-bold uppercase text-muted-foreground">SL Target (%)</label>
                                                    <input
                                                        type="number"
                                                        name="stop_loss_pct"
                                                        value={tradeInputs.stop_loss_pct}
                                                        onChange={handleInputChange}
                                                        className="w-full bg-muted border rounded-lg p-2 font-black text-red-500"
                                                    />
                                                </div>
                                            </div>

                                            <button
                                                onClick={startTrade}
                                                disabled={activeTrade !== null}
                                                className="w-full py-4 bg-primary text-primary-foreground font-black rounded-xl hover:scale-105 transition shadow-lg flex items-center justify-center gap-3 disabled:opacity-50"
                                            >
                                                {activeTrade ? 'TRADE IN PROGRESS' : 'ENTER MANUAL POSITION'}
                                                <ChevronRight className="w-5 h-5" />
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Manual Test History */}
                    <div className="bg-card border rounded-2xl p-6 shadow-sm flex flex-col min-h-[400px]">
                        <div className="flex items-center justify-between border-b pb-4 mb-4">
                            <h2 className="text-lg font-black tracking-tight flex items-center gap-2">
                                <ArrowRight className="w-5 h-5 text-primary" />
                                JOURNAL HISTORY
                            </h2>
                            <button
                                onClick={clearManualHistory}
                                className="p-2 hover:bg-red-500/10 text-muted-foreground hover:text-red-500 rounded-lg transition-colors flex items-center gap-2 text-xs font-bold"
                                title="Clear History"
                            >
                                <Trash2 className="w-4 h-4" />
                                CLEAR
                            </button>
                        </div>

                        {/* Performance Stats */}
                        <div className="grid grid-cols-2 gap-2 mb-6">
                            <div className="bg-muted/30 p-3 rounded-xl border border-primary/5">
                                <p className="text-[9px] font-black uppercase tracking-widest text-muted-foreground">Total Trades</p>
                                <p className="text-lg font-black">{stats.totalTrades}</p>
                            </div>
                            <div className="bg-muted/30 p-3 rounded-xl border border-primary/5">
                                <p className="text-[9px] font-black uppercase tracking-widest text-muted-foreground">Win Rate</p>
                                <p className={`text-lg font-black ${stats.winRate >= 50 ? 'text-green-500' : 'text-red-500'}`}>{stats.winRate.toFixed(1)}%</p>
                            </div>
                            <div className="bg-green-500/5 p-3 rounded-xl border border-green-500/10">
                                <p className="text-[9px] font-black uppercase tracking-widest text-green-500/70">Total Wins</p>
                                <p className="text-lg font-black text-green-500">{stats.totalWins}</p>
                            </div>
                            <div className="bg-red-500/5 p-3 rounded-xl border border-red-500/10">
                                <p className="text-[9px] font-black uppercase tracking-widest text-red-500/70">Total Losses</p>
                                <p className="text-lg font-black text-red-500">{stats.totalLosses}</p>
                            </div>
                        </div>

                        <div className="space-y-3 overflow-y-auto pr-2 max-h-[400px]">
                            {manualHistory.length === 0 ? (
                                <div className="text-center py-10 text-muted-foreground opacity-50 italic">
                                    No manual records yet.
                                </div>
                            ) : (
                                manualHistory.map((trade, idx) => (
                                    <div key={idx} className="bg-muted/50 p-4 rounded-xl border flex flex-col gap-2 hover:bg-muted transition cursor-default group">
                                        <div className="flex justify-between items-start">
                                            <div>
                                                <p className="text-[10px] font-black uppercase tracking-widest opacity-50">
                                                    {new Date(trade.timestamp).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                                                </p>
                                                <p className={`font-black tracking-tighter ${trade.option_type === 'CALL' ? 'text-green-500' : 'text-red-500'}`}>
                                                    {trade.symbol} {trade.option_type} @ ${trade.option_strike}
                                                </p>
                                            </div>
                                            <div className={`px-2 py-1 rounded text-[10px] font-black tracking-widest ${trade.result === 'TP' ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500'}`}>
                                                {trade.result}
                                            </div>
                                        </div>
                                        <div className="flex justify-between items-end">
                                            <p className="text-xs font-bold text-muted-foreground">Confidence: {(trade.confidence * 100).toFixed(0)}%</p>
                                            <p className={`text-lg font-black italic ${trade.pnl_pct >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                                                {trade.pnl_pct >= 0 ? '+' : ''}{trade.pnl_pct.toFixed(2)}%
                                            </p>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
