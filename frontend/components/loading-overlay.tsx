"use client"

import { Loader2, ScanSearch, Brain } from "lucide-react"

interface LoadingOverlayProps {
  isVisible: boolean
  query: string
}

export function LoadingOverlay({ isVisible, query }: LoadingOverlayProps) {
  if (!isVisible) return null

  return (
    <div className="absolute inset-0 z-30 flex items-center justify-center bg-background/70 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="flex flex-col items-center gap-4 rounded-2xl border border-border bg-card p-8 shadow-xl">
        <div className="relative">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <ScanSearch className="absolute -right-1.5 -top-1.5 h-4 w-4 text-primary" />
        </div>
        <div className="flex flex-col items-center gap-1.5">
          <p className="text-sm font-semibold text-foreground">Analyzing footage...</p>
          <p className="max-w-[250px] text-center text-xs text-muted-foreground">
            {`"${query}"`}
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Brain className="h-3.5 w-3.5" />
          <span>Querying spatial scene graph</span>
        </div>
      </div>
    </div>
  )
}
