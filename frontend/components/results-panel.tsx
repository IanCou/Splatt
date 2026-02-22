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
  return (
    <div className="relative h-full w-full overflow-hidden rounded-2xl border border-white/10 bg-black shadow-inner">
      {/* Grid pattern */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `linear-gradient(white 1px, transparent 1px), linear-gradient(90deg, white 1px, transparent 1px)`,
          backgroundSize: "30px 30px",
        }}
      />

      {/* Building wireframe using divs */}
      <div className="absolute inset-[25%] border border-white/5 rounded-lg" />
      <div className="absolute left-[25%] right-[25%] top-[45%] h-px bg-white/5" />
      <div className="absolute left-[25%] right-[25%] top-[60%] h-px bg-white/5" />

      {/* Hotspots */}
      {hotspots.map((spot) => {
        const isActive = activeHotspot === spot.id
        const isHighlighted = highlightedHotspots.includes(spot.id)

        if (!isActive && !isHighlighted) return null;

        return (
          <button
            key={spot.id}
            onClick={() => onHotspotClick(spot.id)}
            className="group absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${spot.x}%`, top: `${spot.y}%` }}
          >
            <span className={`absolute inset-0 -m-3 animate-ping rounded-full bg-white ${isActive ? "opacity-40" : "opacity-20"
              }`} />
            <div className={`relative flex h-3.5 w-3.5 items-center justify-center rounded-full border-2 border-black bg-white shadow-[0_0_10px_rgba(255,255,255,0.5)] transition-all ${isActive ? "scale-125" : ""
              }`}>
              <div className="h-1.5 w-1.5 rounded-full bg-black" />
            </div>
          </button>
        )
      })}

      {/* Labels */}
      <div className="absolute left-4 top-4">
        <Badge variant="outline" className="h-6 gap-2 border-white/10 bg-white/5 text-[10px] font-bold uppercase tracking-widest text-white/40">
          <Box className="h-3 w-3" />
          Spatial Index
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
    High: "border-green-500/20 bg-green-500/10 text-green-400",
    Medium: "border-amber-500/20 bg-amber-500/10 text-amber-400",
    Low: "border-white/10 bg-white/5 text-white/40",
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Top bar with back */}
      <div className="flex items-center justify-between border-b border-white/5 bg-black/40 px-6 py-4 backdrop-blur-xl">
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="h-8 gap-2 rounded-lg px-3 text-[11px] font-bold uppercase tracking-widest text-white/40 hover:bg-white/5 hover:text-white transition-all"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Dismiss
          </Button>
          <div className="h-4 w-px bg-white/10" />
          <div className="flex items-center gap-2 text-white">
            <Sparkles className="h-4 w-4 text-blue-400" />
            <h2 className="text-[13px] font-bold uppercase tracking-tight">Neural Recovery Result</h2>
          </div>
          <Badge className={`h-6 gap-2 rounded-full border px-3 text-[10px] font-bold ${confidenceColor[result.confidence]}`}>
            <Shield className="h-3 w-3" />
            {result.confidence} CONFIDENCE
          </Badge>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="h-8 w-8 text-white/20 hover:text-white hover:bg-white/5 transition-all"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Main content */}
      <div className="flex flex-1 flex-col gap-6 overflow-hidden p-6 md:flex-row">
        {/* Large Visual Context */}
        <div className="flex-1 min-h-[300px] group relative">
          <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent opacity-60 z-10 pointer-events-none" />
          <MiniSceneMap
            hotspots={hotspots}
            activeHotspot={activeHotspot}
            highlightedHotspots={highlightedHotspots}
            onHotspotClick={onHotspotClick}
          />
          <div className="absolute bottom-6 left-6 z-20">
            <p className="text-[10px] uppercase font-bold tracking-widest text-white/40 mb-1">Inferred Location</p>
            <h3 className="text-xl font-bold text-white tracking-tight">{result.location}</h3>
          </div>
        </div>

        {/* Detailed Intelligence Sidebar */}
        <ScrollArea className="w-full shrink-0 md:w-[420px]">
          <div className="flex flex-col gap-6 pr-4">
            {/* Metadata Grid */}
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4 transition-colors hover:bg-white/[0.07]">
                <div className="flex items-center gap-2 mb-2 text-white/40">
                  <Clock className="h-3.5 w-3.5" />
                  <span className="text-[10px] font-bold uppercase tracking-widest">Captured</span>
                </div>
                <p className="text-[13px] font-bold text-white tracking-tight">{result.timestamp}</p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4 transition-colors hover:bg-white/[0.07]">
                <div className="flex items-center gap-2 mb-2 text-white/40">
                  <User className="h-3.5 w-3.5" />
                  <span className="text-[10px] font-bold uppercase tracking-widest">Source AI</span>
                </div>
                <p className="text-[13px] font-bold text-white tracking-tight">{result.worker}</p>
                <p className="text-[10px] font-bold text-white/40 uppercase mt-1">{result.workerRole}</p>
              </div>
            </div>

            {/* Analysis */}
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5 transition-colors hover:bg-white/[0.07]">
              <div className="flex items-center gap-2 mb-3 text-blue-400">
                <Brain className="h-4 w-4" />
                <span className="text-[10px] font-bold uppercase tracking-widest">AI Spatial Reconstruction</span>
              </div>
              <p className="text-[14px] leading-relaxed text-white/80 font-medium">
                {result.description}
              </p>
            </div>

            {/* Visual Proof */}
            <div>
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2 text-white/40">
                  <ImageIcon className="h-3.5 w-3.5" />
                  <span className="text-[10px] font-bold uppercase tracking-widest">Reference Frames</span>
                </div>
                <span className="text-[10px] font-bold text-blue-500 uppercase cursor-pointer hover:underline">View All</span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                {result.thumbnails.map((thumb, i) => (
                  <div
                    key={i}
                    className="aspect-square flex items-center justify-center rounded-xl border border-white/10 bg-white/5 text-center transition-all hover:scale-105 active:scale-95 cursor-pointer hover:bg-white/10"
                  >
                    <span className="text-[10px] font-bold text-white/20 uppercase px-2">{thumb}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="h-px bg-white/5" />

            {/* Pivot Intelligence */}
            <div>
              <p className="mb-3 text-[10px] font-bold uppercase tracking-widest text-white/40">Related Retrieval Nodes</p>
              <div className="grid gap-2">
                {result.relatedQueries.map((rq) => (
                  <button
                    key={rq}
                    onClick={() => onRelatedQuery(rq)}
                    className="group flex items-center justify-between rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-left text-[12px] font-bold text-white/60 transition-all hover:bg-white/10 hover:text-white"
                  >
                    <span>{rq}</span>
                    <ArrowRight className="h-3.5 w-3.5 text-white/20 transition-transform group-hover:translate-x-1 group-hover:text-white" />
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
