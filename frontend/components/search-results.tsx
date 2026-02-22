"use client"

import { useMemo } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  X,
  MapPin,
  Clock,
  Video,
  Eye,
  Ruler,
  Search,
  ExternalLink,
} from "lucide-react"

interface SearchHit {
  description: string
  score: number
  videoId: string | null
  videoName: string | null
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
    const map = new Map<string, { videoId: string; videoName: string; hits: SearchHit[] }>()
    for (const hit of results) {
      const vid = hit.videoId || "unknown"
      if (!map.has(vid)) {
        map.set(vid, {
          videoId: vid,
          videoName: hit.videoName || vid.slice(0, 8),
          hits: [],
        })
      }
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
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/10 shrink-0 md:px-6">
        <div className="flex items-center gap-3">
          <Search className="h-4 w-4 text-white/40" />
          <div>
            <p className="text-sm font-medium text-white">
              Results for &ldquo;<span className="text-white/60">{query}</span>&rdquo;
            </p>
            <p className="text-xs text-white/30">
              {results.length} match{results.length !== 1 && "es"} across {grouped.length} video
              {grouped.length !== 1 && "s"}
            </p>
          </div>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} className="h-8 w-8 text-white/40 hover:text-white hover:bg-white/10">
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Results list — native scrolling */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="space-y-3 p-4 md:px-6">
          {grouped.map((group) => (
            <div key={group.videoId} className="rounded-xl border border-white/10 overflow-hidden bg-white/[0.02]">
              {/* Video header — clickable */}
              <div className="flex items-center justify-between border-b border-white/5 px-4 py-2.5 bg-white/[0.03]">
                <button
                  className="flex items-center gap-2 min-w-0 group"
                  onClick={() => onVideoClick?.(group.videoId, group.videoName)}
                >
                  <Video className="h-4 w-4 shrink-0 text-white/40 group-hover:text-white/70 transition-colors" />
                  <span className="text-sm font-medium text-white/80 truncate group-hover:text-white transition-colors">
                    {group.videoName}
                  </span>
                  <ExternalLink className="h-3 w-3 shrink-0 text-white/20 group-hover:text-white/50 transition-colors" />
                </button>
                <Badge variant="outline" className="text-xs border-white/10 text-white/40 shrink-0 ml-2">
                  {group.hits.length} hit{group.hits.length !== 1 && "s"}
                </Badge>
              </div>

              {/* Hits */}
              <div className="divide-y divide-white/5">
                {group.hits
                  .sort((a, b) => b.score - a.score)
                  .map((hit, idx) => {
                    const typeClass =
                      TYPE_COLORS[(hit.objectType || "").toLowerCase()] || TYPE_COLORS.default
                    return (
                      <div key={idx} className="flex flex-col gap-2 px-4 py-3 hover:bg-white/[0.02] transition-colors">
                        {/* Top row: badges */}
                        <div className="flex flex-wrap items-center gap-2">
                          <ScoreBadge score={hit.score} />
                          {hit.objectType && (
                            <Badge variant="outline" className={`${typeClass} text-xs capitalize`}>
                              {hit.objectType}
                            </Badge>
                          )}
                          {hit.seconds != null && (
                            <span className="flex items-center gap-1 text-xs text-white/30">
                              <Clock className="h-3 w-3" />
                              {formatTimestamp(hit.seconds)}
                            </span>
                          )}
                          {hit.frameNumber != null && (
                            <span className="text-xs text-white/30">
                              Frame {hit.frameNumber}
                            </span>
                          )}
                          {hit.distanceEstimate != null && (
                            <span className="flex items-center gap-1 text-xs text-white/30">
                              <Ruler className="h-3 w-3" />
                              {hit.distanceEstimate.toFixed(1)}m
                            </span>
                          )}
                        </div>

                        {/* Description */}
                        <p className="text-sm leading-relaxed text-white/80">
                          {hit.description}
                        </p>

                        {/* Coordinates */}
                        {hit.x != null && hit.y != null && hit.z != null && (
                          <div className="flex items-center gap-1 text-xs text-white/20">
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
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
