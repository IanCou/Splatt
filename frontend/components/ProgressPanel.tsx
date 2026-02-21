import { AlertCircle, CheckCircle2, Navigation } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface EventLog {
    id: string;
    message: string;
    type: 'info' | 'success' | 'alert';
    time: string;
}

interface ProgressPanelProps {
    logs: EventLog[];
    searchResult?: {
        analysis: string;
        confidence: string;
        location: string;
    } | null;
}

export default function ProgressPanel({ logs, searchResult }: ProgressPanelProps) {
    return (
        <div className="absolute top-6 right-6 bottom-6 w-96 flex flex-col gap-4 z-10 pointer-events-none">

            {/* AI Search Result Panel */}
            <AnimatePresence>
                {searchResult && (
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        className="bg-zinc-900/90 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-5 shadow-2xl pointer-events-auto"
                    >
                        <div className="flex items-center gap-2 text-blue-400 mb-3">
                            <Navigation className="w-5 h-5" />
                            <h3 className="font-semibold">AI Analysis</h3>
                        </div>
                        <p className="text-zinc-200 text-sm leading-relaxed mb-4">
                            {searchResult.analysis}
                        </p>
                        <div className="flex gap-2">
                            <span className="px-2 py-1 bg-zinc-800 rounded text-xs text-zinc-400 border border-zinc-700">
                                Conf: {searchResult.confidence}
                            </span>
                            {searchResult.location && (
                                <span className="px-2 py-1 bg-zinc-800 rounded text-xs text-zinc-400 border border-zinc-700">
                                    {searchResult.location}
                                </span>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Change Log Panel */}
            <div className="flex-1 bg-zinc-900/80 backdrop-blur-md border border-zinc-700/50 rounded-2xl flex flex-col overflow-hidden shadow-2xl pointer-events-auto">
                <div className="p-5 border-b border-zinc-800 bg-zinc-900/50">
                    <h2 className="text-lg font-semibold text-zinc-100 flex items-center gap-2">
                        Change Log
                        <span className="px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400 text-xs text-center border border-blue-500/20">
                            Live
                        </span>
                    </h2>
                    <p className="text-sm text-zinc-500 mt-1">AI-detected site updates</p>
                </div>

                <div className="flex-1 overflow-y-auto p-5 pb-8 space-y-4 custom-scrollbar">
                    {logs.map((log) => (
                        <div key={log.id} className="flex gap-3 items-start animate-fade-in">
                            <div className="mt-0.5">
                                {log.type === 'success' ? (
                                    <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                                ) : log.type === 'alert' ? (
                                    <AlertCircle className="w-5 h-5 text-amber-400" />
                                ) : (
                                    <div className="w-2 h-2 mt-1.5 ml-1.5 rounded-full bg-blue-400 ring-4 ring-blue-400/20" />
                                )}
                            </div>
                            <div>
                                <p className="text-sm text-zinc-200">{log.message}</p>
                                <span className="text-xs text-zinc-500">{log.time}</span>
                            </div>
                        </div>
                    ))}
                    {logs.length === 0 && (
                        <div className="text-center text-zinc-500 text-sm mt-10">
                            No recent activity detected.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
