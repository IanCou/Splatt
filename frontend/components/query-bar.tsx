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
    <div className="relative px-4 py-4 md:px-6">
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative flex items-center">
          <Search className="absolute left-4 h-4.5 w-4.5 text-muted-foreground" />
          <Input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => setShowSuggestions(true)}
            placeholder="Ask about your site... &quot;Where did we leave the scaffolding?&quot;"
            className="h-12 rounded-xl border-border bg-secondary pl-11 pr-24 text-sm text-foreground placeholder:text-muted-foreground focus-visible:ring-primary"
          />
          <div className="absolute right-2 flex items-center gap-1">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-muted-foreground hover:text-foreground"
              aria-label="Voice input (coming soon)"
            >
              <Mic className="h-4 w-4" />
            </Button>
            <Button
              type="submit"
              size="sm"
              disabled={!query.trim() || isLoading}
              className="h-8 rounded-lg bg-primary px-3 text-primary-foreground hover:bg-primary/90"
            >
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Search"}
            </Button>
          </div>
        </div>
      </form>

      {showSuggestions && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setShowSuggestions(false)} />
          <div className="absolute left-4 right-4 z-20 mt-2 overflow-hidden rounded-xl border border-border bg-card shadow-lg md:left-6 md:right-6">
            <div className="flex items-center gap-2 border-b border-border px-4 py-2.5">
              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-xs font-medium text-muted-foreground">Example queries</span>
            </div>
            <div className="py-1">
              {sampleQueries.map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="flex w-full items-center gap-3 px-4 py-2.5 text-left text-sm text-foreground transition-colors hover:bg-secondary"
                >
                  <Search className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
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
