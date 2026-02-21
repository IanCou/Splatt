"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"
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
  ArrowLeft,
  Box,
  Move3D,
} from "lucide-react"
import type { QueryResult, Hotspot } from "@/lib/mock-data"

interface ResultsPanelProps {
  result: QueryResult
  hotspots: Hotspot[]
  activeHotspot: string | null
  highlightedHotspots: string[]
  onHotspotClick: (id: string) => void
  onClose: () => void
  onRelatedQuery: (query: string) => void
}

function MiniSceneMap({
  hotspots,
  activeHotspot,
  highlightedHotspots,
  onHotspotClick,
}: {
  hotspots: Hotspot[]
  activeHotspot: string | null
  highlightedHotspots: string[]
  onHotspotClick: (id: string) => void
}) {
  const typeColor: Record<string, string> = {
    material: "bg-primary",
    equipment: "bg-chart-4",
    worker: "bg-chart-2",
    event: "bg-destructive",
  }

  return (
    <div className="relative h-full w-full overflow-hidden rounded-xl border border-border bg-[#0f1724]">
      {/* Grid pattern */}
      <div
        className="absolute inset-0 opacity-[0.06]"
        style={{
          backgroundImage: `linear-gradient(rgba(148,163,184,1) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,1) 1px, transparent 1px)`,
          backgroundSize: "40px 40px",
        }}
      />

      {/* Building wireframe using divs */}
      <div className="absolute inset-[20%] border border-primary/12 rounded" />
      <div className="absolute left-[20%] right-[20%] top-[35%] h-px bg-primary/8" />
      <div className="absolute left-[20%] right-[20%] top-[50%] h-px bg-primary/8" />
      <div className="absolute left-[20%] right-[20%] top-[65%] h-px bg-primary/8" />

      {/* Hotspots */}
      {hotspots.map((spot) => {
        const isActive = activeHotspot === spot.id
        const isHighlighted = highlightedHotspots.includes(spot.id)

        return (
          <button
            key={spot.id}
            onClick={() => onHotspotClick(spot.id)}
            className="group absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${spot.x}%`, top: `${spot.y}%` }}
            aria-label={`Hotspot: ${spot.label}`}
          >
            {(isActive || isHighlighted) && (
              <span
                className={`absolute inset-0 -m-2 animate-ping rounded-full ${typeColor[spot.type]} ${
                  isActive ? "opacity-60" : "opacity-40"
                }`}
                style={{ animationDuration: isActive ? "1s" : "2.5s" }}
              />
            )}
            <span
              className={`absolute inset-0 -m-1.5 rounded-full ${typeColor[spot.type]} ${
                isActive ? "opacity-30" : "opacity-10"
              } transition-opacity group-hover:opacity-30`}
            />
            <span
              className={`relative block h-3 w-3 rounded-full border-2 border-background shadow-lg transition-transform ${typeColor[spot.type]} ${
                isActive ? "scale-125" : "group-hover:scale-110"
              }`}
            />
            <span
              className={`absolute left-1/2 top-full mt-1.5 -translate-x-1/2 whitespace-nowrap rounded-md px-2 py-0.5 text-xs font-medium transition-opacity ${
                isActive
                  ? "bg-primary text-primary-foreground opacity-100"
                  : "bg-card text-card-foreground opacity-0 group-hover:opacity-100"
              } border border-border shadow-md`}
            >
              {spot.label}
            </span>
          </button>
        )
      })}

      {/* Scene label */}
      <div className="absolute left-3 top-3">
        <Badge variant="outline" className="gap-1.5 border-border bg-background/80 text-xs backdrop-blur-sm">
          <Box className="h-3 w-3 text-primary" />
          Site Map
        </Badge>
      </div>
      <div className="absolute bottom-3 right-3">
        <Badge variant="outline" className="border-border bg-background/80 text-xs backdrop-blur-sm">
          <Move3D className="mr-1 h-3 w-3" />
          {highlightedHotspots.length} pin{highlightedHotspots.length !== 1 ? "s" : ""}
        </Badge>
      </div>
    </div>
  )
}

export function ResultsPanel({
  result,
  hotspots,
  activeHotspot,
  highlightedHotspots,
  onHotspotClick,
  onClose,
  onRelatedQuery,
}: ResultsPanelProps) {
  const confidenceColor: Record<string, string> = {
    High: "border-primary/30 bg-primary/10 text-primary",
    Medium: "border-chart-4/30 bg-chart-4/10 text-chart-4",
    Low: "border-muted-foreground/30 bg-muted/50 text-muted-foreground",
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden animate-in fade-in duration-300">
      {/* Top bar with back */}
      <div className="flex items-center justify-between px-4 pb-3 md:px-6">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="gap-1.5 text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
          <Separator orientation="vertical" className="h-5 bg-border" />
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            <h2 className="text-sm font-semibold text-foreground">Search Results</h2>
          </div>
          <Badge variant="outline" className={`gap-1.5 ${confidenceColor[result.confidence]}`}>
            <Shield className="h-3 w-3" />
            {result.confidence} Confidence
          </Badge>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="h-8 w-8 text-muted-foreground hover:text-foreground"
        >
          <X className="h-4 w-4" />
          <span className="sr-only">Close results</span>
        </Button>
      </div>

      {/* Main content: scene map + details side by side */}
      <div className="flex flex-1 flex-col gap-4 overflow-hidden px-4 pb-4 md:flex-row md:px-6 md:pb-6">
        {/* Scene map -- large area */}
        <div className="flex-1 min-h-[240px]">
          <MiniSceneMap
            hotspots={hotspots}
            activeHotspot={activeHotspot}
            highlightedHotspots={highlightedHotspots}
            onHotspotClick={onHotspotClick}
          />
        </div>

        {/* Detail sidebar */}
        <ScrollArea className="w-full shrink-0 md:w-[380px]">
          <div className="flex flex-col gap-4">
            {/* Location */}
            <Card className="gap-0 border-border bg-card p-4">
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
              <Card className="gap-0 border-border bg-card p-3">
                <div className="flex items-start gap-2.5">
                  <Clock className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                  <div>
                    <p className="text-xs text-muted-foreground">Captured</p>
                    <p className="mt-0.5 text-xs font-medium text-foreground">{result.timestamp}</p>
                  </div>
                </div>
              </Card>
              <Card className="gap-0 border-border bg-card p-3">
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
            <Card className="gap-0 border-border bg-card p-4">
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
              <div className="grid grid-cols-3 gap-2">
                {result.thumbnails.map((thumb, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-center rounded-lg border border-border bg-muted p-4 text-center"
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
                    className="group flex items-center justify-between rounded-lg border border-border bg-card px-3 py-2.5 text-left text-sm text-foreground transition-colors hover:border-primary/30 hover:bg-primary/5"
                  >
                    <span>{rq}</span>
                    <ArrowRight className="h-3.5 w-3.5 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-primary" />
                  </button>
                ))}
              </div>
            </div>
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}
