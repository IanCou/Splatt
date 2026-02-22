"use client"

import { useState, useRef } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Search, Mic, ChevronDown, Loader2 } from "lucide-react"
import { sampleQueries } from "@/lib/mock-data"

interface QueryBarProps {
  onSubmit: (query: string) => void
  isLoading: boolean
}

export function QueryBar({ onSubmit, isLoading }: QueryBarProps) {
  const [query, setQuery] = useState("")
  const [showSuggestions, setShowSuggestions] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      onSubmit(query.trim())
      setShowSuggestions(false)
    }
  }

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion)
    setShowSuggestions(false)
    onSubmit(suggestion)
  }

  return (
    <div className="relative px-6 py-8">
      <form onSubmit={handleSubmit} className="relative group max-w-3xl mx-auto">
        <div className="relative flex items-center transition-all">
          <div className="absolute left-4 z-10">
            <Search className="h-5 w-5 text-white/30 group-focus-within:text-white transition-colors" />
          </div>
          <Input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => setShowSuggestions(true)}
            placeholder="Search spatial memory... &quot;Where is the blue crane?&quot;"
            className="h-14 rounded-2xl border-white/10 bg-white/5 pl-12 pr-28 text-[15px] font-medium text-white placeholder:text-white/20 focus-visible:ring-0 focus-visible:border-white/30 hover:border-white/20 transition-all shadow-[0_4px_20px_-10px_rgba(0,0,0,0.5)]"
          />
          <div className="absolute right-3 flex items-center gap-2">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-10 w-10 text-white/20 hover:text-white hover:bg-white/5 transition-all"
              aria-label="Voice input"
            >
              <Mic className="h-5 w-5" />
            </Button>
            <Button
              type="submit"
              disabled={!query.trim() || isLoading}
              className="h-10 rounded-xl bg-white px-5 text-[13px] font-bold text-black hover:bg-white/90 disabled:bg-white/10 disabled:text-white/20 transition-all hover:scale-[1.02] active:scale-[0.98]"
            >
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "QUERY"}
            </Button>
          </div>
        </div>
      </form>

      {showSuggestions && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setShowSuggestions(false)} />
          <div className="absolute left-6 right-6 z-20 mt-3 overflow-hidden rounded-2xl border border-white/10 bg-black shadow-[0_20px_50px_rgba(0,0,0,1)] max-w-3xl mx-auto backdrop-blur-3xl">
            <div className="flex items-center gap-2 border-b border-white/5 px-5 py-3.5 bg-white/5">
              <span className="text-[10px] font-bold uppercase tracking-widest text-white/40">Suggested Queries</span>
            </div>
            <div className="py-2">
              {sampleQueries.map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="flex w-full items-center gap-4 px-5 py-3 text-left text-[14px] font-medium text-white/60 transition-all hover:bg-white/5 hover:text-white group"
                >
                  <Search className="h-4 w-4 shrink-0 text-white/20 group-hover:text-white transition-colors" />
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
