import { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';

interface SemanticSearchBarProps {
    onSearch: (query: string) => void;
    isSearching: boolean;
}

export default function SemanticSearchBar({ onSearch, isSearching }: SemanticSearchBarProps) {
    const [query, setQuery] = useState('');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (query.trim()) {
            onSearch(query);
        }
    };

    return (
        <div className="absolute top-6 left-1/2 -translate-x-1/2 w-full max-w-lg z-10">
            <form onSubmit={handleSubmit} className="relative group">
                <div className="absolute inset-0 bg-blue-500/20 blur-xl group-hover:bg-blue-500/30 transition-all rounded-full" />
                <div className="relative flex items-center bg-zinc-900/90 backdrop-blur-md border border-zinc-700/50 rounded-full h-14 shadow-2xl overflow-hidden">
                    <div className="pl-5 pr-3 text-zinc-400">
                        {isSearching ? <Loader2 className="w-5 h-5 animate-spin text-blue-400" /> : <Search className="w-5 h-5" />}
                    </div>
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Ask about the site (e.g. 'Where is the excavator?')"
                        className="flex-1 bg-transparent border-none outline-none text-zinc-100 placeholder:text-zinc-500 text-lg"
                    />
                    <button
                        type="submit"
                        disabled={isSearching || !query.trim()}
                        className="h-full px-6 bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-800 disabled:text-zinc-500 text-white font-medium transition-colors"
                    >
                        Locate
                    </button>
                </div>
            </form>
        </div>
    );
}
