import React from 'react';
import { ChevronLeft, ChevronRight, Clock } from 'lucide-react';

const TimelineSlider = ({
    data,
    currentIndex,
    onChange,
    interval
}) => {
    if (!data || data.length === 0) return null;

    const currentPoint = data[currentIndex];

    const handleSliderChange = (e) => {
        onChange(parseInt(e.target.value));
    };

    const stepForward = () => {
        if (currentIndex < data.length - 1) {
            onChange(currentIndex + 1);
        }
    };

    const stepBackward = () => {
        if (currentIndex > 0) {
            onChange(currentIndex - 1);
        }
    };

    return (
        <div className="w-full bg-card border rounded-xl p-4 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-primary" />
                    <span className="text-xs font-bold uppercase tracking-widest text-muted-foreground">Historical Timeline</span>
                </div>
                <div className="flex items-center gap-4">
                    <div className="text-right">
                        <p className="text-sm font-black italic">{currentPoint?.Datetime ? new Date(currentPoint.Datetime).toLocaleString() : 'Loading...'}</p>
                        <p className="text-[10px] text-muted-foreground font-medium uppercase">
                            Index: {currentIndex} / {data.length - 1} | Interval: {interval}
                        </p>
                    </div>
                </div>
            </div>

            <div className="flex items-center gap-4">
                <button
                    onClick={stepBackward}
                    disabled={currentIndex === 0}
                    className="p-2 hover:bg-muted rounded-full transition disabled:opacity-30"
                >
                    <ChevronLeft className="w-5 h-5" />
                </button>

                <input
                    type="range"
                    min="0"
                    max={data.length - 1}
                    value={currentIndex}
                    onChange={handleSliderChange}
                    className="flex-1 h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                />

                <button
                    onClick={stepForward}
                    disabled={currentIndex === data.length - 1}
                    className="p-2 hover:bg-muted rounded-full transition disabled:opacity-30"
                >
                    <ChevronRight className="w-5 h-5" />
                </button>
            </div>
        </div>
    );
};

export default TimelineSlider;
