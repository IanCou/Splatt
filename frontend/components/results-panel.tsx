"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import {
  MapPin,
  Clock,
  User,
  Brain,
  Image as ImageIcon,
  Sparkles,
  X,
  ArrowRight,
  Shield,
} from "lucide-react"
import type { QueryResult } from "@/lib/mock-data"

interface ResultsPanelProps {
  result: QueryResult | null
  onClose: () => void
  onRelatedQuery: (query: string) => void
  isVisible: boolean
}

export function ResultsPanel({ result, onClose, onRelatedQuery, isVisible }: ResultsPanelProps) {
  if (!result || !isVisible) return null

  const confidenceColor = {
    High: "border-primary/30 bg-primary/10 text-primary",
    Medium: "border-chart-4/30 bg-chart-4/10 text-chart-4",
    Low: "border-muted-foreground/30 bg-muted/50 text-muted-foreground",
  }

  return (
    <div className="flex w-full flex-col gap-4 overflow-y-auto border-t border-border bg-background p-4 md:w-[420px] md:border-l md:border-t-0 md:p-5 animate-in slide-in-from-right-5 duration-300">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center rounded-lg bg-primary/10 p-1.5">
            <Sparkles className="h-4 w-4 text-primary" />
          </div>
          <h2 className="text-sm font-semibold text-foreground">Query Result</h2>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} className="h-7 w-7 text-muted-foreground hover:text-foreground">
          <X className="h-4 w-4" />
          <span className="sr-only">Close results panel</span>
        </Button>
      </div>

      {/* Confidence */}
      <Badge variant="outline" className={`w-fit gap-1.5 ${confidenceColor[result.confidence]}`}>
        <Shield className="h-3 w-3" />
        {result.confidence} Confidence
      </Badge>

      {/* Location */}
      <Card className="gap-0 border-border bg-secondary p-4">
        <div className="flex items-start gap-3">
          <MapPin className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
          <div>
            <p className="text-xs font-medium text-muted-foreground">Location</p>
            <p className="mt-0.5 text-sm font-medium text-foreground">{result.location}</p>
            <p className="mt-1 font-mono text-xs text-muted-foreground">{result.coordinates}</p>
          </div>
        </div>
      </Card>

      {/* Timestamp & Worker */}
      <div className="grid grid-cols-2 gap-3">
        <Card className="gap-0 border-border bg-secondary p-3">
          <div className="flex items-start gap-2.5">
            <Clock className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
            <div>
              <p className="text-xs text-muted-foreground">Captured</p>
              <p className="mt-0.5 text-xs font-medium text-foreground">{result.timestamp}</p>
            </div>
          </div>
        </Card>
        <Card className="gap-0 border-border bg-secondary p-3">
          <div className="flex items-start gap-2.5">
            <User className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
            <div>
              <p className="text-xs text-muted-foreground">Source</p>
              <p className="mt-0.5 text-xs font-medium text-foreground">{result.worker}</p>
              <p className="text-xs text-muted-foreground">{result.workerRole}</p>
            </div>
          </div>
        </Card>
      </div>

      {/* AI Analysis */}
      <Card className="gap-0 border-border bg-secondary p-4">
        <div className="flex items-start gap-3">
          <Brain className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
          <div>
            <p className="text-xs font-medium text-muted-foreground">AI Analysis</p>
            <p className="mt-1.5 text-sm leading-relaxed text-foreground">{result.description}</p>
          </div>
        </div>
      </Card>

      {/* Thumbnails */}
      <div>
        <div className="mb-2.5 flex items-center gap-2">
          <ImageIcon className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs font-medium text-muted-foreground">Related Frames</span>
        </div>
        <div className="flex gap-2">
          {result.thumbnails.map((thumb, i) => (
            <div
              key={i}
              className="flex flex-1 items-center justify-center rounded-lg border border-border bg-muted p-3 text-center"
            >
              <span className="text-xs text-muted-foreground">{thumb}</span>
            </div>
          ))}
        </div>
      </div>

      <Separator className="bg-border" />

      {/* Related Queries */}
      <div>
        <p className="mb-2.5 text-xs font-medium text-muted-foreground">Related Scenes</p>
        <div className="flex flex-col gap-1.5">
          {result.relatedQueries.map((rq) => (
            <button
              key={rq}
              onClick={() => onRelatedQuery(rq)}
              className="group flex items-center justify-between rounded-lg border border-border bg-secondary px-3 py-2.5 text-left text-sm text-foreground transition-colors hover:border-primary/30 hover:bg-primary/5"
            >
              <span>{rq}</span>
              <ArrowRight className="h-3.5 w-3.5 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-primary" />
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
