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
  const typeColor = {
    material: "bg-primary",
    equipment: "bg-chart-4",
    worker: "bg-chart-2",
    event: "bg-destructive",
  }

  return (
    <button
      onClick={onClick}
      className="group absolute -translate-x-1/2 -translate-y-1/2"
      style={{ left: `${spot.x}%`, top: `${spot.y}%` }}
      aria-label={`Hotspot: ${spot.label}`}
    >
      {/* Pulse ring */}
      <span
        className={`absolute inset-0 -m-2 animate-ping rounded-full opacity-40 ${typeColor[spot.type]} ${
          isActive ? "opacity-60" : isHighlighted ? "opacity-50" : ""
        }`}
        style={{ animationDuration: isActive ? "1s" : "2.5s" }}
      />
      {/* Outer glow */}
      <span
        className={`absolute inset-0 -m-1.5 rounded-full ${typeColor[spot.type]} ${
          isActive ? "opacity-30" : "opacity-15"
        } transition-opacity group-hover:opacity-30`}
      />
      {/* Dot */}
      <span
        className={`relative block h-3.5 w-3.5 rounded-full border-2 border-background shadow-lg transition-transform ${typeColor[spot.type]} ${
          isActive ? "scale-125" : "group-hover:scale-110"
        }`}
      />
      {/* Label */}
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
}

export function SceneViewer({ hotspots, activeHotspot, onHotspotClick, highlightedHotspots }: SceneViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  console.log("Viewer constructed");
  // Log hotspot data when component receives props
  useEffect(() => {
    console.log('SCENE_VIEWER_PROPS_UPDATED:', {
      hotspotsCount: hotspots.length,
      hotspots: hotspots.map(h => ({ id: h.id, label: h.label, x: h.x, y: h.y, type: h.type })),
      activeHotspot,
      highlightedHotspotsCount: highlightedHotspots.length,
      highlightedHotspots
    })
  }, [hotspots, activeHotspot, highlightedHotspots])

  useEffect(() => {
    console.log('CANVAS_EFFECT_RUNNING')
    const canvas = canvasRef.current
    if (!canvas) {
      console.log('CANVAS_ERROR: Canvas ref is null')
      return
    }
    console.log('CANVAS_REF_FOUND:', canvas)

    const ctx = canvas.getContext("2d")
    if (!ctx) {
      console.log('CANVAS_ERROR: Could not get 2d context')
      return
    }
    console.log('CANVAS_CONTEXT_OBTAINED')

    const draw = () => {
      const dpr = window.devicePixelRatio || 1
      const rect = canvas.getBoundingClientRect()
      console.log('CANVAS_DRAW_CALLED:', {
        dpr,
        rectWidth: rect.width,
        rectHeight: rect.height,
        canvasWidth: canvas.width,
        canvasHeight: canvas.height
      })

      if (rect.width === 0 || rect.height === 0) {
        console.log('CANVAS_ERROR: Canvas has zero dimensions!', rect)
        return
      }

      canvas.width = rect.width * dpr
      canvas.height = rect.height * dpr
      ctx.scale(dpr, dpr)

      // Background
      ctx.fillStyle = "#0f1724"
      ctx.fillRect(0, 0, rect.width, rect.height)
      console.log('CANVAS_BACKGROUND_DRAWN')

      // Grid
      const gridSize = 40
      ctx.strokeStyle = "rgba(148, 163, 184, 0.06)"
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

      // Perspective lines for depth illusion
      const cx = rect.width / 2
      const horizon = rect.height * 0.35
      ctx.strokeStyle = "rgba(148, 163, 184, 0.03)"
      ctx.lineWidth = 1
      for (let i = -6; i <= 6; i++) {
        ctx.beginPath()
        ctx.moveTo(cx + i * 120, rect.height)
        ctx.lineTo(cx + i * 20, horizon)
        ctx.stroke()
      }
      for (let i = 0; i <= 8; i++) {
        const y = horizon + ((rect.height - horizon) * i) / 8
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(rect.width, y)
        ctx.stroke()
      }

      // Building wireframe shapes
      ctx.strokeStyle = "rgba(249, 115, 22, 0.12)"
      ctx.lineWidth = 1.5

      // Main building outline
      const bx = rect.width * 0.25
      const by = rect.height * 0.3
      const bw = rect.width * 0.5
      const bh = rect.height * 0.5
      ctx.strokeRect(bx, by, bw, bh)

      // Floors
      for (let i = 1; i <= 3; i++) {
        const floorY = by + (bh * i) / 4
        ctx.beginPath()
        ctx.moveTo(bx, floorY)
        ctx.lineTo(bx + bw, floorY)
        ctx.stroke()
      }

      // Crane arm
      ctx.strokeStyle = "rgba(249, 115, 22, 0.08)"
      ctx.beginPath()
      ctx.moveTo(bx + bw * 0.7, by)
      ctx.lineTo(bx + bw * 0.7, by - rect.height * 0.15)
      ctx.lineTo(bx + bw * 1.1, by - rect.height * 0.15)
      ctx.stroke()

      // Ground plane markers
      ctx.fillStyle = "rgba(148, 163, 184, 0.04)"
      ctx.fillRect(rect.width * 0.1, rect.height * 0.82, rect.width * 0.15, rect.height * 0.08)
      ctx.fillRect(rect.width * 0.7, rect.height * 0.75, rect.width * 0.2, rect.height * 0.12)

      console.log('CANVAS_DRAW_COMPLETE: Graph elements drawn')
    }

    console.log('CANVAS_CALLING_INITIAL_DRAW')
    draw()

    console.log('CANVAS_SETTING_UP_RESIZE_OBSERVER')
    const observer = new ResizeObserver(() => {
      console.log('CANVAS_RESIZE_OBSERVED')
      draw()
    })
    observer.observe(canvas)
    return () => {
      console.log('CANVAS_CLEANUP: Disconnecting observer')
      observer.disconnect()
    }
  }, [])

  console.log("HOTSPOTS: ");
  console.log(hotspots);

  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect()
      const computedStyle = window.getComputedStyle(containerRef.current)
      console.log('SCENE_VIEWER_CONTAINER_DIMENSIONS:', {
        width: rect.width,
        height: rect.height,
        display: computedStyle.display,
        flexGrow: computedStyle.flexGrow,
        flexShrink: computedStyle.flexShrink,
        flexBasis: computedStyle.flexBasis,
        minHeight: computedStyle.minHeight,
        maxHeight: computedStyle.maxHeight
      })

      // Check parent chain
      let element = containerRef.current.parentElement
      let level = 1
      while (element && level <= 5) {
        const parentRect = element.getBoundingClientRect()
        const parentStyle = window.getComputedStyle(element)
        console.log(`PARENT_${level}:`, {
          tagName: element.tagName,
          className: element.className,
          width: parentRect.width,
          height: parentRect.height,
          display: parentStyle.display,
          flexDirection: parentStyle.flexDirection,
          flexGrow: parentStyle.flexGrow,
          overflow: parentStyle.overflow
        })
        element = element.parentElement
        level++
      }
    }
  }, [])

  return (
    <div ref={containerRef} className="relative flex-1 overflow-hidden rounded-xl border border-border bg-card">
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />

      {/* Hotspots */}
      <div className="absolute inset-0">
        {hotspots.map((spot) => {
          console.log('SCENE_VIEWER_RENDERING_HOTSPOT:', {
            id: spot.id,
            label: spot.label,
            isActive: activeHotspot === spot.id,
            isHighlighted: highlightedHotspots.includes(spot.id),
            position: { x: spot.x, y: spot.y }
          })
          return (
            <HotspotDot
              key={spot.id}
              spot={spot}
              isActive={activeHotspot === spot.id}
              isHighlighted={highlightedHotspots.includes(spot.id)}
              onClick={() => {
                console.log('SCENE_VIEWER_HOTSPOT_CLICKED:', { id: spot.id, label: spot.label })
                onHotspotClick(spot.id)
              }}
            />
          )
        })}
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
