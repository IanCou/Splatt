"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import useSWR from "swr"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { MapPin, X, Loader2, Navigation, Eye } from "lucide-react"

interface MapPoint {
  frameNumber: number
  seconds: number
  x: number
  y: number
  z: number
  rx: number
  ry: number
  rz: number
  objectType: string
  description: string
  detections?: Array<{
    objectType: string
    description: string
    distanceEstimate: number | null
  }>
}

interface MapViewerProps {
  videoId: string
  videoName: string
  onClose: () => void
}

const fetcher = (url: string) => fetch(url).then((r) => r.json())

const TYPE_COLORS: Record<string, string> = {
  equipment: "#f59e0b",
  materials: "#3b82f6",
  workers: "#10b981",
  vehicles: "#8b5cf6",
  structures: "#ef4444",
}

export function MapViewer({ videoId, videoName, onClose }: MapViewerProps) {
  const { data, isLoading, error } = useSWR<{ videoId: string; points: MapPoint[] }>(
    `/api/videos/${videoId}/coordinates`,
    fetcher
  )

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [hoveredPoint, setHoveredPoint] = useState<MapPoint | null>(null)
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 })

  // Transform and viewport state
  const [offset, setOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 })
  const [scale, setScale] = useState(1)
  const [isDragging, setIsDragging] = useState(false)
  const lastMouseRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 })
  const initialFitDone = useRef(false)

  const points = data?.points ?? []

  // Compute bounding box
  const getBounds = useCallback(() => {
    if (points.length === 0) return { minX: -5, maxX: 5, minZ: -5, maxZ: 5 }
    // Use x and z for bird's eye view (top-down)
    const xs = points.map((p) => p.x)
    const zs = points.map((p) => p.z)
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minZ = Math.min(...zs)
    const maxZ = Math.max(...zs)
    const padX = Math.max((maxX - minX) * 0.15, 1)
    const padZ = Math.max((maxZ - minZ) * 0.15, 1)
    return { minX: minX - padX, maxX: maxX + padX, minZ: minZ - padZ, maxZ: maxZ + padZ }
  }, [points])

  // Fit view to data on first load
  useEffect(() => {
    if (points.length === 0 || initialFitDone.current) return
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const bounds = getBounds()
    const dataWidth = bounds.maxX - bounds.minX
    const dataHeight = bounds.maxZ - bounds.minZ
    const fitScale = Math.min(rect.width / dataWidth, rect.height / dataHeight) * 0.85
    const centerX = (bounds.minX + bounds.maxX) / 2
    const centerZ = (bounds.minZ + bounds.maxZ) / 2
    setScale(fitScale)
    setOffset({
      x: rect.width / 2 - centerX * fitScale,
      y: rect.height / 2 - centerZ * fitScale
    })
    initialFitDone.current = true
  }, [points, getBounds])

  // Reset fit state when videoId changes
  useEffect(() => {
    initialFitDone.current = false
  }, [videoId])

  // World to screen coordinate conversion
  const worldToScreen = useCallback(
    (wx: number, wz: number) => ({
      x: wx * scale + offset.x,
      y: wz * scale + offset.y,
    }),
    [scale, offset]
  )

  const screenToWorld = useCallback(
    (sx: number, sy: number) => ({
      x: (sx - offset.x) / scale,
      z: (sy - offset.y) / scale,
    }),
    [scale, offset]
  )

  // Draw the map
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    // Background
    ctx.fillStyle = "#0a0f1a"
    ctx.fillRect(0, 0, rect.width, rect.height)

    // Grid
    const gridWorldSize = scale > 80 ? 0.5 : scale > 30 ? 1 : scale > 10 ? 2 : 5
    ctx.strokeStyle = "rgba(148, 163, 184, 0.06)"
    ctx.lineWidth = 1

    const topLeft = screenToWorld(0, 0)
    const bottomRight = screenToWorld(rect.width, rect.height)

    const gridStartX = Math.floor(topLeft.x / gridWorldSize) * gridWorldSize
    const gridStartZ = Math.floor(topLeft.z / gridWorldSize) * gridWorldSize

    for (let wx = gridStartX; wx <= bottomRight.x; wx += gridWorldSize) {
      const sx = wx * scale + offset.x
      ctx.beginPath()
      ctx.moveTo(sx, 0)
      ctx.lineTo(sx, rect.height)
      ctx.stroke()
    }
    for (let wz = gridStartZ; wz <= bottomRight.z; wz += gridWorldSize) {
      const sy = wz * scale + offset.y
      ctx.beginPath()
      ctx.moveTo(0, sy)
      ctx.lineTo(rect.width, sy)
      ctx.stroke()
    }

    // Origin crosshair
    const origin = worldToScreen(0, 0)
    ctx.strokeStyle = "rgba(148, 163, 184, 0.15)"
    ctx.lineWidth = 1
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(origin.x, 0)
    ctx.lineTo(origin.x, rect.height)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(0, origin.y)
    ctx.lineTo(rect.width, origin.y)
    ctx.stroke()
    ctx.setLineDash([])

    if (points.length === 0) return

    // Sort by frame number for path drawing
    const sorted = [...points].sort((a, b) => a.frameNumber - b.frameNumber)

    // Draw camera path
    ctx.strokeStyle = "rgba(249, 115, 22, 0.4)"
    ctx.lineWidth = 2
    ctx.beginPath()
    sorted.forEach((p, i) => {
      const sp = worldToScreen(p.x, p.z)
      if (i === 0) ctx.moveTo(sp.x, sp.y)
      else ctx.lineTo(sp.x, sp.y)
    })
    ctx.stroke()

    // Draw direction arrows along path (every few points)
    const arrowInterval = Math.max(1, Math.floor(sorted.length / 10))
    sorted.forEach((p, i) => {
      if (i === 0 || i % arrowInterval !== 0) return
      const prev = sorted[i - 1]
      const curr = worldToScreen(p.x, p.z)
      const prevS = worldToScreen(prev.x, prev.z)
      const dx = curr.x - prevS.x
      const dy = curr.y - prevS.y
      const len = Math.sqrt(dx * dx + dy * dy)
      if (len < 5) return

      const angle = Math.atan2(dy, dx)
      const arrowSize = 6

      ctx.fillStyle = "rgba(249, 115, 22, 0.6)"
      ctx.beginPath()
      ctx.moveTo(
        curr.x - arrowSize * Math.cos(angle - Math.PI / 6),
        curr.y - arrowSize * Math.sin(angle - Math.PI / 6)
      )
      ctx.lineTo(curr.x, curr.y)
      ctx.lineTo(
        curr.x - arrowSize * Math.cos(angle + Math.PI / 6),
        curr.y - arrowSize * Math.sin(angle + Math.PI / 6)
      )
      ctx.fill()
    })

    // Draw points
    sorted.forEach((p, i) => {
      const sp = worldToScreen(p.x, p.z)
      const isHovered = hoveredPoint?.frameNumber === p.frameNumber
      const isFirst = i === 0
      const isLast = i === sorted.length - 1
      const color = TYPE_COLORS[p.objectType] || "#f97316"

      // Glow for start/end
      if (isFirst || isLast) {
        ctx.beginPath()
        ctx.arc(sp.x, sp.y, 12, 0, Math.PI * 2)
        ctx.fillStyle = isFirst ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)"
        ctx.fill()
      }

      // Outer ring
      ctx.beginPath()
      ctx.arc(sp.x, sp.y, isHovered ? 8 : 5, 0, Math.PI * 2)
      ctx.fillStyle = isHovered ? color : `${color}cc`
      ctx.fill()

      // Inner dot
      ctx.beginPath()
      ctx.arc(sp.x, sp.y, isHovered ? 4 : 2.5, 0, Math.PI * 2)
      ctx.fillStyle = "#0a0f1a"
      ctx.fill()

      // Label for start/end
      if (isFirst || isLast) {
        ctx.font = "bold 10px system-ui, sans-serif"
        ctx.fillStyle = isFirst ? "#10b981" : "#ef4444"
        ctx.textAlign = "center"
        ctx.fillText(isFirst ? "START" : "END", sp.x, sp.y - 14)
      }
    })

    // Draw viewing direction for hovered point
    if (hoveredPoint) {
      const sp = worldToScreen(hoveredPoint.x, hoveredPoint.z)
      const rzRad = (hoveredPoint.rz * Math.PI) / 180
      const dirLen = 20
      ctx.strokeStyle = "rgba(255, 255, 255, 0.5)"
      ctx.lineWidth = 2
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(sp.x, sp.y)
      ctx.lineTo(sp.x + Math.cos(rzRad) * dirLen, sp.y + Math.sin(rzRad) * dirLen)
      ctx.stroke()
      ctx.setLineDash([])

      // FOV cone
      const fovAngle = Math.PI / 6
      ctx.fillStyle = "rgba(255, 255, 255, 0.05)"
      ctx.beginPath()
      ctx.moveTo(sp.x, sp.y)
      ctx.lineTo(
        sp.x + Math.cos(rzRad - fovAngle) * dirLen * 1.5,
        sp.y + Math.sin(rzRad - fovAngle) * dirLen * 1.5
      )
      ctx.lineTo(
        sp.x + Math.cos(rzRad + fovAngle) * dirLen * 1.5,
        sp.y + Math.sin(rzRad + fovAngle) * dirLen * 1.5
      )
      ctx.closePath()
      ctx.fill()
    }
  }, [points, scale, offset, hoveredPoint, worldToScreen, screenToWorld])

  // Resize observer
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const observer = new ResizeObserver(() => {
      // Trigger re-render by updating a dummy state
      setScale((s) => s)
    })
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [])

  // Mouse handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true)
    lastMouseRef.current = { x: e.clientX, y: e.clientY }
  }, [])

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current
      if (!canvas) return
      const rect = canvas.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top

      if (isDragging) {
        const dx = e.clientX - lastMouseRef.current.x
        const dy = e.clientY - lastMouseRef.current.y
        setOffset((prev) => ({ x: prev.x + dx, y: prev.y + dy }))
        lastMouseRef.current = { x: e.clientX, y: e.clientY }
        return
      }

      // Hit test points
      const hitRadius = 12
      let found: MapPoint | null = null
      for (const p of points) {
        const sp = worldToScreen(p.x, p.z)
        const dx = sp.x - mx
        const dy = sp.y - my
        if (dx * dx + dy * dy < hitRadius * hitRadius) {
          found = p
          break
        }
      }
      setHoveredPoint(found)
      setTooltipPos({ x: mx, y: my })
    },
    [isDragging, points, worldToScreen]
  )

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault()
      const canvas = canvasRef.current
      if (!canvas) return
      const rect = canvas.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top

      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1
      const newScale = scale * zoomFactor

      // Zoom toward mouse position
      setOffset({
        x: mx - (mx - offset.x) * zoomFactor,
        y: my - (my - offset.y) * zoomFactor,
      })
      setScale(newScale)
    },
    [scale, offset]
  )

  const handleResetView = useCallback(() => {
    initialFitDone.current = false
    // Trigger refitting
    setScale(1)
    setOffset({ x: 0, y: 0 })
    // Let the useEffect handle re-fitting
    setTimeout(() => {
      initialFitDone.current = false
      const canvas = canvasRef.current
      if (!canvas || points.length === 0) return
      const rect = canvas.getBoundingClientRect()
      const bounds = getBounds()
      const dataWidth = bounds.maxX - bounds.minX
      const dataHeight = bounds.maxZ - bounds.minZ
      const fitScale = Math.min(rect.width / dataWidth, rect.height / dataHeight) * 0.85
      const centerX = (bounds.minX + bounds.maxX) / 2
      const centerZ = (bounds.minZ + bounds.maxZ) / 2
      setScale(fitScale)
      setOffset({
        x: rect.width / 2 - centerX * fitScale,
        y: rect.height / 2 - centerZ * fitScale,
      })
    }, 0)
  }, [points, getBounds])

  if (isLoading) {
    return (
      <div className="relative flex flex-1 flex-col overflow-hidden rounded-xl border border-border bg-card">
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="flex items-center gap-2">
            <MapPin className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold text-foreground">{videoName}</span>
          </div>
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex flex-1 items-center justify-center">
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">Loading camera positions...</p>
          </div>
        </div>
      </div>
    )
  }

  if (error || (data && points.length === 0)) {
    const isNetworkError = error && !data
    return (
      <div className="relative flex flex-1 flex-col overflow-hidden rounded-xl border border-border bg-card">
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="flex items-center gap-2">
            <MapPin className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold text-foreground">{videoName}</span>
          </div>
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex flex-1 items-center justify-center">
          <div className="flex flex-col items-center gap-3 px-6 text-center">
            {isNetworkError ? (
              <>
                <MapPin className="h-8 w-8 text-destructive/60" />
                <p className="text-sm text-muted-foreground">
                  Failed to load coordinate data.
                  <br />
                  <span className="text-xs">Check that the backend is running and try again.</span>
                </p>
              </>
            ) : (
              <>
                <Loader2 className="h-8 w-8 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  No coordinate data available yet.
                  <br />
                  <span className="text-xs">
                    Coordinates appear after the Gaussian Splatting pipeline processes this video.
                    This may take several minutes.
                  </span>
                </p>
              </>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="relative flex flex-1 flex-col overflow-hidden rounded-xl border border-border bg-card">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <MapPin className="h-4 w-4 text-primary" />
          <span className="text-sm font-semibold text-foreground truncate">{videoName}</span>
          <Badge variant="outline" className="gap-1 border-border text-xs">
            {points.length} frame{points.length !== 1 ? "s" : ""}
          </Badge>
        </div>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={handleResetView} title="Reset view">
            <Navigation className="h-3.5 w-3.5" />
          </Button>
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Canvas */}
      <div ref={containerRef} className="relative flex-1 overflow-hidden">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 h-full w-full cursor-grab active:cursor-grabbing"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => {
            setIsDragging(false)
            setHoveredPoint(null)
          }}
          onWheel={handleWheel}
        />

        {/* Tooltip */}
        {hoveredPoint && (
          <div
            className="pointer-events-none absolute z-10 max-w-72 rounded-lg border border-border bg-card/95 px-3 py-2 shadow-lg backdrop-blur-sm"
            style={{
              left: Math.min(tooltipPos.x + 16, (containerRef.current?.clientWidth ?? 300) - 300),
              top: Math.max(tooltipPos.y - 10, 8),
            }}
          >
            <div className="flex items-center gap-2 mb-1.5">
              <Eye className="h-3 w-3 text-primary" />
              <span className="text-xs font-semibold text-foreground">Frame {hoveredPoint.frameNumber}</span>
              <span className="text-xs text-muted-foreground">@ {hoveredPoint.seconds.toFixed(1)}s</span>
            </div>
            {/* All detections for this frame */}
            {hoveredPoint.detections && hoveredPoint.detections.length > 0 ? (
              <div className="flex flex-col gap-1 mb-1.5">
                {hoveredPoint.detections.slice(0, 5).map((det, i) => (
                  <div key={i} className="flex items-start gap-1.5">
                    <span
                      className="mt-1 inline-block h-2 w-2 shrink-0 rounded-full"
                      style={{ backgroundColor: TYPE_COLORS[det.objectType] || "#f97316" }}
                    />
                    <div className="min-w-0">
                      <span className="text-[10px] font-medium text-muted-foreground uppercase">{det.objectType}</span>
                      <p className="text-xs text-foreground line-clamp-1">{det.description}</p>
                      {det.distanceEstimate != null && (
                        <span className="text-[10px] text-muted-foreground">{det.distanceEstimate.toFixed(1)}m away</span>
                      )}
                    </div>
                  </div>
                ))}
                {hoveredPoint.detections.length > 5 && (
                  <span className="text-[10px] text-muted-foreground">+{hoveredPoint.detections.length - 5} more</span>
                )}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground line-clamp-2 mb-1.5">{hoveredPoint.description}</p>
            )}
            <div className="flex flex-wrap gap-1.5 text-[10px] text-muted-foreground border-t border-border pt-1">
              <span>X: {hoveredPoint.x.toFixed(2)}</span>
              <span>Z: {hoveredPoint.z.toFixed(2)}</span>
              <span>Y: {hoveredPoint.y.toFixed(2)}</span>
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="absolute bottom-3 left-3 flex flex-col gap-1 rounded-lg bg-background/80 px-3 py-2 text-xs backdrop-blur-sm">
          <span className="font-medium text-foreground mb-0.5">Camera Path</span>
          <div className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-full bg-emerald-500" />
            <span className="text-muted-foreground">Start</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-full bg-red-500" />
            <span className="text-muted-foreground">End</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-sm border border-orange-500/50 bg-orange-500/20" />
            <span className="text-muted-foreground">Trajectory</span>
          </div>
          <span className="font-medium text-foreground mt-1.5 mb-0.5">Detections</span>
          {Object.entries(TYPE_COLORS).map(([type, color]) => (
            <div key={type} className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-muted-foreground capitalize">{type}</span>
            </div>
          ))}
        </div>

        {/* Controls hint */}
        <div className="absolute bottom-3 right-3 flex items-center gap-1.5 rounded-lg bg-background/80 px-3 py-1.5 text-xs text-muted-foreground backdrop-blur-sm">
          <Navigation className="h-3 w-3" />
          Drag to pan · Scroll to zoom
        </div>
      </div>
    </div>
  )
}
