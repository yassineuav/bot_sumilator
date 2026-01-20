"use client";
import React, { useMemo } from 'react';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceArea
} from 'recharts';

export default function PredictionChart({ data = [], prediction, height = 350 }) {
    // Preprocess data to add "future" space
    const chartData = useMemo(() => {
        if (!data || data.length === 0) return [];

        // Clone data to avoid mutations
        const plotData = data.map(d => ({ ...d }));

        // Add null points for future space (about 20% of visible range)
        // If we have prediction data, we can try to be smart, otherwise just add fixed count
        const futureSteps = 15; // Number of "empty" steps to show on the right
        const lastTime = new Date(plotData[plotData.length - 1].Datetime).getTime();

        // Estimate average interval
        let interval = 15 * 60 * 1000; // default 15m
        if (plotData.length > 1) {
            const time1 = new Date(plotData[plotData.length - 2].Datetime).getTime();
            interval = lastTime - time1;
        }

        for (let i = 1; i <= futureSteps; i++) {
            const futureTime = new Date(lastTime + (i * interval));
            // Format to match existing Datetime string format if possible, or just use ISO
            // The XAxis tickFormatter should handle display
            plotData.push({
                Datetime: futureTime.toISOString(),
                Close: null, // No price line
                isFuture: true
            });
        }

        return plotData;
    }, [data]);

    if (!prediction) {
        return (
            <div style={{ height: `${height}px` }} className="flex items-center justify-center text-muted-foreground border rounded-xl">
                No prediction data
            </div>
        );
    }

    // Determine Zone Colors based on Signal
    const isCall = prediction.signal === 'CALL';
    const tpColor = "#22c55e"; // Green
    const slColor = "#ef4444"; // Red
    const entryColor = "hsl(var(--foreground))"; // Text color for contrast

    // Calculate coordinates for zones
    // The zone starts at the index of the LAST REAL data point
    const lastRealPoint = chartData[data.length - 1];
    const lastFuturePoint = chartData[chartData.length - 1];

    if (!lastRealPoint || !lastFuturePoint) return null;

    const startX = lastRealPoint.Datetime;
    const endX = lastFuturePoint.Datetime;

    // Custom Label Component for Right Axis
    const CustomLabel = ({ viewBox, value, color, fill }) => {
        const { x, y } = viewBox;
        return (
            <text x={x + 5} y={y + 4} fill={fill || color} fontSize={10} fontWeight="bold" textAnchor="start">
                {value}
            </text>
        );
    };

    return (
        <div style={{ height: `${height}px` }} className="w-full">
            <ResponsiveContainer width="100%" height="100%" minHeight={100}>
                <AreaChart data={chartData} margin={{ top: 20, right: 100, left: 0, bottom: 0 }}>
                    <defs>
                        <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.1} />
                            <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />

                    <XAxis
                        dataKey="Datetime"
                        hide={true}
                        range={[0, 'auto']}
                    />

                    <YAxis
                        domain={['auto', 'auto']}
                        orientation="left"
                        tickFormatter={(val) => val.toFixed(2)}
                        fontSize={10}
                        stroke="hsl(var(--muted-foreground))"
                        tickLine={false}
                        axisLine={false}
                    />

                    <Tooltip
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))', fontSize: 10 }}
                        labelFormatter={(label) => new Date(label).toLocaleString()}
                    />

                    {/* ENTRY LINE - Extends across future */}
                    <ReferenceLine
                        segment={[{ x: startX, y: prediction.entry_price }, { x: endX, y: prediction.entry_price }]}
                        stroke={entryColor}
                        strokeDasharray="3 3"
                        strokeWidth={1}
                        ifOverflow="extendDomain"
                        label={{ position: 'right', value: `Entry: ${prediction.entry_price.toFixed(2)}`, fill: entryColor, fontSize: 10, fontWeight: 'bold' }}
                    />

                    {/* TP ZONE */}
                    <ReferenceArea
                        x1={startX}
                        x2={endX}
                        y1={prediction.entry_price}
                        y2={prediction.tp_price}
                        fill={tpColor}
                        fillOpacity={0.2}
                        strokeOpacity={0}
                        ifOverflow="extendDomain"
                    />
                    <ReferenceLine
                        segment={[{ x: startX, y: prediction.tp_price }, { x: endX, y: prediction.tp_price }]}
                        stroke={tpColor}
                        strokeWidth={1}
                        label={{ position: 'right', value: `TP: ${prediction.tp_price.toFixed(2)}`, fill: tpColor, fontSize: 10, fontWeight: 'bold' }}
                    />

                    {/* SL ZONE */}
                    <ReferenceArea
                        x1={startX}
                        x2={endX}
                        y1={prediction.entry_price}
                        y2={prediction.sl_price}
                        fill={slColor}
                        fillOpacity={0.2}
                        strokeOpacity={0}
                        ifOverflow="extendDomain"
                    />
                    <ReferenceLine
                        segment={[{ x: startX, y: prediction.sl_price }, { x: endX, y: prediction.sl_price }]}
                        stroke={slColor}
                        strokeWidth={1}
                        label={{ position: 'right', value: `SL: ${prediction.sl_price.toFixed(2)}`, fill: slColor, fontSize: 10, fontWeight: 'bold' }}
                    />

                    {/* Main Price Line (stops before future) */}
                    <Area
                        type="monotone"
                        dataKey="Close"
                        stroke="hsl(var(--primary))"
                        strokeWidth={2}
                        fill="url(#colorPrice)"
                        fillOpacity={1}
                        connectNulls={false}
                        activeDot={{ r: 4, strokeWidth: 0 }}
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
}
