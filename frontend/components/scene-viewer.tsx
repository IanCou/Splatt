"use client"

import { useEffect, useRef } from "react"
import { Badge } from "@/components/ui/badge"
import { Box, Move3D } from "lucide-react"
import type { Hotspot } from "@/lib/mock-data"

interface SceneViewerProps {
  hotspots: Hotspot[]
  activeHotspot: string | null
  onHotspotClick: (id: string) => void
  highlightedHotspots: string[]
}

function HotspotDot({
  spot,
  isActive,
  isHighlighted,
  onClick,
}: {
  spot: Hotspot
  isActive: boolean
  isHighlighted: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="group absolute -translate-x-1/2 -translate-y-1/2"
      style={{ left: `${spot.x}%`, top: `${spot.y}%` }}
      aria-label={`Hotspot: ${spot.label}`}
    >
      {/* Pulse ring */}
      <span
        className={`absolute inset-0 -m-3 animate-ping rounded-full bg-white ${
          isActive ? "opacity-30" : isHighlighted ? "opacity-20" : "opacity-0"
        }`}
        style={{ animationDuration: isActive ? "1s" : "2.5s" }}
      />

      {/* Outer glow */}
      <span
        className={`absolute inset-0 -m-1.5 rounded-full bg-white ${
          isActive ? "opacity-20" : "opacity-0"
        } transition-opacity group-hover:opacity-30`}
      />

      {/* Dot */}
      <div className={`relative flex h-4 w-4 items-center justify-center rounded-full border-2 border-black bg-white shadow-[0_0_15px_rgba(255,255,255,0.5)] transition-all ${
        isActive ? "scale-125 ring-4 ring-white/20" : "group-hover:scale-110"
      }`}>
        <div className="h-1.5 w-1.5 rounded-full bg-black" />
      </div>

      {/* Label */}
      <span
        className={`absolute left-1/2 top-full mt-3 -translate-x-1/2 whitespace-nowrap rounded-full px-3 py-1 text-[10px] font-bold uppercase tracking-widest transition-all ${
          isActive
            ? "bg-white text-black opacity-100 translate-y-0"
            : "bg-black/80 text-white opacity-0 group-hover:opacity-100 translate-y-1"
        } border border-white/20 backdrop-blur-md shadow-2xl`}
      >
        {spot.label}
      </span>
    </button>
  )
}

export function SceneViewer({ hotspots, activeHotspot, onHotspotClick, highlightedHotspots }: SceneViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const draw = () => {
      const dpr = window.devicePixelRatio || 1
      const rect = canvas.getBoundingClientRect()
      canvas.width = rect.width * dpr
      canvas.height = rect.height * dpr
      ctx.scale(dpr, dpr)

      // Background
      ctx.fillStyle = "#000000"
      ctx.fillRect(0, 0, rect.width, rect.height)

      // Grid
      const gridSize = 60
      ctx.strokeStyle = "rgba(255, 255, 255, 0.05)"
      ctx.lineWidth = 1

      for (let x = 0; x <= rect.width; x += gridSize) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, rect.height)
        ctx.stroke()
      }
      for (let y = 0; y <= rect.height; y += gridSize) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(rect.width, y)
        ctx.stroke()
      }

      // Perspective lines
      const cx = rect.width / 2
      const horizon = rect.height * 0.4
      ctx.strokeStyle = "rgba(255, 255, 255, 0.03)"
      for (let i = -10; i <= 10; i++) {
        ctx.beginPath()
        ctx.moveTo(cx + i * 200, rect.height)
        ctx.lineTo(cx + i * 40, horizon)
        ctx.stroke()
      }

      // Building wireframe
      ctx.strokeStyle = "rgba(255, 255, 255, 0.08)"
      ctx.lineWidth = 1
      const bx = rect.width * 0.3
      const by = rect.height * 0.35
      const bw = rect.width * 0.4
      const bh = rect.height * 0.45
      ctx.strokeRect(bx, by, bw, bh)

      // Floors
      ctx.setLineDash([5, 5])
      for (let i = 1; i <= 4; i++) {
        const floorY = by + (bh * i) / 5
        ctx.beginPath()
        ctx.moveTo(bx, floorY)
        ctx.lineTo(bx + bw, floorY)
        ctx.stroke()
      }
      ctx.setLineDash([])
    }

    draw()

    const observer = new ResizeObserver(draw)
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [])

  return (
    <div className="relative flex-1 overflow-hidden rounded-xl border border-border bg-card">
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />

      {/* Hotspots */}
      <div className="absolute inset-0">
        {hotspots.map((spot) => (
          <HotspotDot
            key={spot.id}
            spot={spot}
            isActive={activeHotspot === spot.id}
            isHighlighted={highlightedHotspots.includes(spot.id)}
            onClick={() => onHotspotClick(spot.id)}
          />
        ))}
      </div>

      {/* Overlay labels */}
      <div className="absolute left-4 top-4 flex items-center gap-2">
        <Badge variant="outline" className="gap-1.5 border-border bg-background/80 text-xs backdrop-blur-sm">
          <Box className="h-3 w-3 text-primary" />
          3D Construction Site Scene
        </Badge>
      </div>

      <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between">
        <div className="flex items-center gap-1.5 rounded-lg bg-background/80 px-3 py-1.5 text-xs text-muted-foreground backdrop-blur-sm">
          <Move3D className="h-3.5 w-3.5" />
          Rotate | Pan | Zoom
        </div>
        <Badge variant="outline" className="border-border bg-background/80 text-xs backdrop-blur-sm">
          Gaussian Splatting - integration in progress
        </Badge>
      </div>
    </div>
  )
}
