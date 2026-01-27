'use client';

import React, { useState } from 'react';
import { Terminal, Activity, Play, BarChart2 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';

const TrainAlgoPage = () => {
    const [loading, setLoading] = useState(false);
    const [btLoading, setBtLoading] = useState(false);

    // Global Config
    const [projectName, setProjectName] = useState('default');

    // Train Config
    const [trainConfig, setTrainConfig] = useState({
        symbol: 'SPY',
        interval: '15m',
        epochs: 20,
        lookback: 60,
        batchSize: 32
    });

    // Backtest Config
    const [btConfig, setBtConfig] = useState({
        initialBalance: 1000,
        riskPct: 20,
        stopLoss: 10,
        takeProfit: 50,
        modelType: 'lstm'
    });

    const [status, setStatus] = useState(null);
    const [logs, setLogs] = useState([]);
    const [backtestResult, setBacktestResult] = useState(null);

    const addLog = (msg) => {
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
    };

    const handleTrain = async () => {
        setLoading(true);
        setStatus('Initializing training...');
        addLog(`Requesting Multi-Timeframe Training for Project: ${projectName}...`);

        try {
            const res = await fetch('http://localhost:8000/api/train/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: trainConfig.symbol,
                    intervals: [trainConfig.interval, '1h'], // Train both 15m and 1h usually
                    epochs: parseInt(trainConfig.epochs),
                    batch_size: parseInt(trainConfig.batchSize),
                    lookback: parseInt(trainConfig.lookback),
                    project_name: projectName,
                    model_type: 'lstm'
                })
            });

            const data = await res.json();

            if (res.ok) {
                setStatus(`Training Started. Check backend logs.`);
                addLog(`Success: Training initiated for ${trainConfig.symbol}`);
                if (data.details) {
                    Object.entries(data.details).forEach(([k, v]) => addLog(`${k}: ${v}`));
                }
            } else {
                setStatus(`Error: ${data.error}`);
                addLog(`Error: ${data.error}`);
            }
        } catch (e) {
            setStatus(`Network Error: ${e.message}`);
            addLog(`Network Error: ${e.message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleBacktest = async () => {
        setBtLoading(true);
        addLog(`Starting Backtest for ${trainConfig.symbol} (${trainConfig.interval})...`);
        setBacktestResult(null);

        try {
            const res = await fetch('http://localhost:8000/api/backtest/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: trainConfig.symbol,
                    interval: trainConfig.interval,
                    initial_balance: parseFloat(btConfig.initialBalance),
                    risk_pct: parseFloat(btConfig.riskPct),
                    stop_loss: parseFloat(btConfig.stopLoss),
                    take_profit: parseFloat(btConfig.takeProfit),
                    project_name: projectName,
                    model_type: btConfig.modelType
                })
            });

            const data = await res.json();

            if (res.ok) {
                addLog(`Backtest Complete. Trades: ${data.total_trades}, Final: $${data.final_balance?.toFixed(2)}`);
                setBacktestResult(data);
            } else {
                addLog(`Backtest Failed: ${data.error}`);
            }
        } catch (e) {
            addLog(`Backtest Net Error: ${e.message}`);
        } finally {
            setBtLoading(false);
        }
    };

    return (
        <div className="p-6 space-y-6 bg-black min-h-screen text-gray-100 font-sans">
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">
                Algo Lab: Train & Backtest
            </h1>

            {/* Global Settings */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 flex items-center gap-4">
                <div className="flex-1">
                    <label className="block text-xs font-bold text-gray-500 uppercase mb-1">Project Name (Namespace)</label>
                    <input
                        value={projectName}
                        onChange={(e) => setProjectName(e.target.value)}
                        className="w-full bg-gray-800 border border-gray-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none"
                        placeholder="e.g. experiment_v1"
                    />
                </div>
                <div className="flex-1">
                    <label className="block text-xs font-bold text-gray-500 uppercase mb-1">Symbol</label>
                    <input
                        value={trainConfig.symbol}
                        onChange={(e) => setTrainConfig({ ...trainConfig, symbol: e.target.value.toUpperCase() })}
                        className="w-full bg-gray-800 border border-gray-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none"
                    />
                </div>
                <div className="flex-1">
                    <label className="block text-xs font-bold text-gray-500 uppercase mb-1">Interval</label>
                    <input
                        value={trainConfig.interval}
                        onChange={(e) => setTrainConfig({ ...trainConfig, interval: e.target.value })}
                        className="w-full bg-gray-800 border border-gray-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none"
                        placeholder="15m"
                    />
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                {/* LEFT COL: Configs */}
                <div className="space-y-6">
                    {/* Training Config */}
                    <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden shadow-lg">
                        <div className="p-4 border-b border-gray-800 flex justify-between items-center">
                            <h2 className="text-lg font-semibold text-gray-100 flex items-center gap-2">
                                <Activity size={18} className="text-blue-400" /> Training Config
                            </h2>
                        </div>
                        <div className="p-6 space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-xs text-gray-400 mb-1">Epochs</label>
                                    <input type="number" value={trainConfig.epochs} onChange={(e) => setTrainConfig({ ...trainConfig, epochs: e.target.value })} className="w-full bg-gray-800 border-gray-700 rounded px-2 py-1" />
                                </div>
                                <div>
                                    <label className="block text-xs text-gray-400 mb-1">Lookback</label>
                                    <input type="number" value={trainConfig.lookback} onChange={(e) => setTrainConfig({ ...trainConfig, lookback: e.target.value })} className="w-full bg-gray-800 border-gray-700 rounded px-2 py-1" />
                                </div>
                            </div>
                            <button
                                onClick={handleTrain}
                                disabled={loading}
                                className={`w-full font-bold py-2 rounded transition-colors ${loading ? 'bg-gray-600' : 'bg-blue-600 hover:bg-blue-500 text-white'}`}
                            >
                                {loading ? "Training..." : "Train Model"}
                            </button>
                        </div>
                    </div>

                    {/* Backtest Config */}
                    <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden shadow-lg">
                        <div className="p-4 border-b border-gray-800 flex justify-between items-center">
                            <h2 className="text-lg font-semibold text-gray-100 flex items-center gap-2">
                                <Play size={18} className="text-green-400" /> Backtest Config
                            </h2>
                        </div>
                        <div className="p-6 space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-xs text-gray-400 mb-1">Start Balance ($)</label>
                                    <input type="number" value={btConfig.initialBalance} onChange={(e) => setBtConfig({ ...btConfig, initialBalance: e.target.value })} className="w-full bg-gray-800 border-gray-700 rounded px-2 py-1" />
                                </div>
                                <div>
                                    <label className="block text-xs text-gray-400 mb-1">Risk Per Trade (%)</label>
                                    <input type="number" value={btConfig.riskPct} onChange={(e) => setBtConfig({ ...btConfig, riskPct: e.target.value })} className="w-full bg-gray-800 border-gray-700 rounded px-2 py-1" />
                                </div>
                                <div>
                                    <label className="block text-xs text-gray-400 mb-1">Stop Loss (%)</label>
                                    <input type="number" value={btConfig.stopLoss} onChange={(e) => setBtConfig({ ...btConfig, stopLoss: e.target.value })} className="w-full bg-gray-800 border-gray-700 rounded px-2 py-1" />
                                </div>
                                <div>
                                    <label className="block text-xs text-gray-400 mb-1">Take Profit (%)</label>
                                    <input type="number" value={btConfig.takeProfit} onChange={(e) => setBtConfig({ ...btConfig, takeProfit: e.target.value })} className="w-full bg-gray-800 border-gray-700 rounded px-2 py-1" />
                                </div>
                            </div>
                            <button
                                onClick={handleBacktest}
                                disabled={btLoading}
                                className={`w-full font-bold py-2 rounded transition-colors ${btLoading ? 'bg-gray-600' : 'bg-green-600 hover:bg-green-500 text-white'}`}
                            >
                                {btLoading ? "Running Backtest..." : "Run Backtest"}
                            </button>
                        </div>
                    </div>

                    {/* Logs */}
                    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 h-48 overflow-y-auto font-mono text-xs text-gray-300">
                        {logs.map((L, i) => <div key={i} className="border-b border-gray-800 pb-1 mb-1">{L}</div>)}
                    </div>

                </div>

                {/* RIGHT COL: Results & Charts */}
                <div className="space-y-6">
                    <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden shadow-lg min-h-[500px] flex flex-col">
                        <div className="p-4 border-b border-gray-800">
                            <h2 className="text-lg font-semibold text-gray-100 flex items-center gap-2">
                                <BarChart2 size={18} className="text-purple-400" /> Results: Equity Curve
                            </h2>
                        </div>
                        <div className="p-4 flex-1 flex flex-col">
                            {backtestResult ? (
                                <>
                                    <div className="grid grid-cols-4 gap-2 mb-4">
                                        <div className="bg-gray-800 p-2 rounded text-center">
                                            <p className="text-xs text-gray-400">Total Trades</p>
                                            <p className="text-xl font-bold text-white">{backtestResult.total_trades}</p>
                                        </div>
                                        <div className="bg-gray-800 p-2 rounded text-center">
                                            <p className="text-xs text-gray-400">Win Rate</p>
                                            <p className="text-xl font-bold text-blue-400">
                                                {backtestResult.summary && backtestResult.summary.win_rate
                                                    ? (backtestResult.summary.win_rate * 100).toFixed(1) + '%'
                                                    : '0%'}
                                            </p>
                                        </div>
                                        <div className="bg-gray-800 p-2 rounded text-center">
                                            <p className="text-xs text-gray-400">Final Balance</p>
                                            <p className={`text-xl font-bold ${backtestResult.final_balance >= btConfig.initialBalance ? 'text-green-400' : 'text-red-400'}`}>
                                                ${backtestResult.final_balance?.toFixed(0)}
                                            </p>
                                        </div>
                                        <div className="bg-gray-800 p-2 rounded text-center">
                                            <p className="text-xs text-gray-400">Max Win</p>
                                            <p className="text-xl font-bold text-green-400">
                                                {backtestResult.summary?.max_win ? (backtestResult.summary.max_win * 100).toFixed(1) + '%' : '0%'}
                                            </p>
                                        </div>
                                    </div>

                                    <div className="flex-1 min-h-[300px]">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <LineChart data={backtestResult.equity_curve}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                                <XAxis
                                                    dataKey="Datetime"
                                                    tick={{ fill: '#666', fontSize: 10 }}
                                                    tickFormatter={(val) => val.split(' ')[0]} // Show date only
                                                />
                                                <YAxis domain={['auto', 'auto']} tick={{ fill: '#666', fontSize: 10 }} />
                                                <Tooltip
                                                    contentStyle={{ backgroundColor: '#111', border: '1px solid #333' }}
                                                />
                                                <Line
                                                    type="monotone"
                                                    dataKey="equity"
                                                    stroke="#8b5cf6"
                                                    strokeWidth={2}
                                                    dot={false}
                                                />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>
                                </>
                            ) : (
                                <div className="h-full flex items-center justify-center text-gray-600">
                                    Run a backtest to see results
                                </div>
                            )}
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
};

export default TrainAlgoPage;
