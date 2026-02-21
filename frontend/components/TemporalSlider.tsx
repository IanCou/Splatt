import { Clock } from 'lucide-react';

interface TemporalSliderProps {
    opacity: number;
    onChange: (value: number) => void;
}

export default function TemporalSlider({ opacity, onChange }: TemporalSliderProps) {
    return (
        <div className="absolute bottom-10 left-1/2 -translate-x-1/2 w-full max-w-md z-10">
            <div className="bg-zinc-900/80 backdrop-blur-md border border-zinc-700/50 rounded-2xl p-4 shadow-2xl">
                <div className="flex items-center justify-between text-zinc-400 text-sm font-medium mb-3">
                    <span>Snapshot A (Yesterday)</span>
                    <Clock className="w-4 h-4 text-blue-400" />
                    <span>Snapshot B (Today)</span>
                </div>
                <div className="relative flex items-center">
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={opacity}
                        onChange={(e) => onChange(parseFloat(e.target.value))}
                        className="w-full h-2 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                </div>
                <div className="mt-2 text-center text-xs text-zinc-500">
                    Showing {Math.round(opacity * 100)}% of Snapshot B
                </div>
            </div>
        </div>
    );
}
