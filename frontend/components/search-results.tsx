"use client"

import { useMemo } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import {
  X,
  MapPin,
  Clock,
  Video,
  Eye,
  Ruler,
  Search,
} from "lucide-react"

interface SearchHit {
  description: string
  score: number
  videoId: string | null
  objectType: string | null
  seconds: number | null
  frameNumber: number | null
  distanceEstimate: number | null
  x: number | null
  y: number | null
  z: number | null
}

interface SearchResultsProps {
  results: SearchHit[]
  query: string
  onClose: () => void
  onVideoClick?: (videoId: string, videoName: string) => void
}

const TYPE_COLORS: Record<string, string> = {
  person: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  vehicle: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  equipment: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  structure: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  material: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  signage: "bg-pink-500/20 text-pink-400 border-pink-500/30",
  animal: "bg-teal-500/20 text-teal-400 border-teal-500/30",
  default: "bg-gray-500/20 text-gray-400 border-gray-500/30",
}

function formatTimestamp(seconds: number | null): string {
  if (seconds == null) return "—"
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

function ScoreBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const color =
    pct >= 80
      ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
      : pct >= 60
        ? "bg-amber-500/20 text-amber-400 border-amber-500/30"
        : "bg-red-500/20 text-red-400 border-red-500/30"
  return (
    <Badge variant="outline" className={`${color} text-xs tabular-nums`}>
      {pct}% match
    </Badge>
  )
}

export function SearchResults({ results, query, onClose, onVideoClick }: SearchResultsProps) {
  // Group results by videoId
  const grouped = useMemo(() => {
    const map = new Map<string, { videoId: string; hits: SearchHit[] }>()
    for (const hit of results) {
      const vid = hit.videoId || "unknown"
      if (!map.has(vid)) map.set(vid, { videoId: vid, hits: [] })
      map.get(vid)!.hits.push(hit)
    }
    // Sort groups by best score in each
    return Array.from(map.values()).sort(
      (a, b) => Math.max(...b.hits.map((h) => h.score)) - Math.max(...a.hits.map((h) => h.score))
    )
  }, [results])

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 pb-3 md:px-6">
        <div className="flex items-center gap-3">
          <Search className="h-4 w-4 text-muted-foreground" />
          <div>
            <p className="text-sm font-medium">
              Results for &ldquo;<span className="text-primary">{query}</span>&rdquo;
            </p>
            <p className="text-xs text-muted-foreground">
              {results.length} match{results.length !== 1 && "es"} across {grouped.length} video
              {grouped.length !== 1 && "s"}
            </p>
          </div>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} className="h-8 w-8">
          <X className="h-4 w-4" />
        </Button>
      </div>

      <Separator />

      {/* Results list */}
      <ScrollArea className="flex-1">
        <div className="space-y-4 p-4 md:px-6">
          {grouped.map((group) => (
            <Card key={group.videoId} className="overflow-hidden">
              {/* Video header */}
              <div className="flex items-center justify-between border-b bg-muted/30 px-4 py-2.5">
                <div className="flex items-center gap-2">
                  <Video className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium truncate">
                    Video {group.videoId.slice(0, 8)}…
                  </span>
                  <Badge variant="secondary" className="text-xs">
                    {group.hits.length} hit{group.hits.length !== 1 && "s"}
                  </Badge>
                </div>
                {onVideoClick && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 gap-1.5 text-xs text-primary"
                    onClick={() =>
                      onVideoClick(group.videoId, `Video ${group.videoId.slice(0, 8)}`)
                    }
                  >
                    <Eye className="h-3 w-3" />
                    View Map
                  </Button>
                )}
              </div>

              {/* Hits */}
              <div className="divide-y">
                {group.hits
                  .sort((a, b) => b.score - a.score)
                  .map((hit, idx) => {
                    const typeClass =
                      TYPE_COLORS[(hit.objectType || "").toLowerCase()] || TYPE_COLORS.default
                    return (
                      <div key={idx} className="flex flex-col gap-2 px-4 py-3">
                        {/* Top row: badges */}
                        <div className="flex flex-wrap items-center gap-2">
                          <ScoreBadge score={hit.score} />
                          {hit.objectType && (
                            <Badge variant="outline" className={`${typeClass} text-xs capitalize`}>
                              {hit.objectType}
                            </Badge>
                          )}
                          {hit.seconds != null && (
                            <span className="flex items-center gap-1 text-xs text-muted-foreground">
                              <Clock className="h-3 w-3" />
                              {formatTimestamp(hit.seconds)}
                            </span>
                          )}
                          {hit.frameNumber != null && (
                            <span className="text-xs text-muted-foreground">
                              Frame {hit.frameNumber}
                            </span>
                          )}
                          {hit.distanceEstimate != null && (
                            <span className="flex items-center gap-1 text-xs text-muted-foreground">
                              <Ruler className="h-3 w-3" />
                              {hit.distanceEstimate.toFixed(1)}m
                            </span>
                          )}
                        </div>

                        {/* Description */}
                        <p className="text-sm leading-relaxed text-foreground/90">
                          {hit.description}
                        </p>

                        {/* Coordinates */}
                        {hit.x != null && hit.y != null && hit.z != null && (
                          <div className="flex items-center gap-1 text-xs text-muted-foreground">
                            <MapPin className="h-3 w-3" />
                            <span className="tabular-nums">
                              ({hit.x.toFixed(2)}, {hit.y.toFixed(2)}, {hit.z.toFixed(2)})
                            </span>
                          </div>
                        )}
                      </div>
                    )
                  })}
              </div>
            </Card>
          ))}
        </div>
      </ScrollArea>
    </div>
  )
}
