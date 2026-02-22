"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import useSWR from "swr"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import {
  Upload,
  Video,
  FolderOpen,
  Clock,
  FileVideo,
  Loader2,
  CheckCircle,
  AlertCircle,
  ChevronRight,
  ChevronDown,
  HardDrive,
  Film,
  X,
} from "lucide-react"
import type { VideoGroup, UploadProgress } from "@/lib/types"

function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = seconds % 60
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

function formatBytes(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(0)} MB`
  return `${(bytes / 1_000).toFixed(0)} KB`
}

function formatDate(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })
}

const fetcher = (url: string) => {
  console.log('VIDEO_GROUPS_FETCH_INITIATED:', { url, timestamp: new Date().toISOString() })
  return fetch(url)
    .then((r) => {
      console.log('VIDEO_GROUPS_FETCH_RESPONSE:', { status: r.status, statusText: r.statusText })
      if (!r.ok) {
        console.error('VIDEO_GROUPS_FETCH_ERROR:', { status: r.status, statusText: r.statusText })
      }
      return r.json()
    })
    .then((data) => {
      console.log('VIDEO_GROUPS_FETCH_SUCCESS:', {
        groupCount: data?.groups?.length || 0,
        totalVideos: data?.groups?.reduce((sum: number, g: any) => sum + (g.videoCount || 0), 0) || 0,
        groups: data?.groups?.map((g: any) => ({ id: g.id, name: g.name, videoCount: g.videoCount })) || []
      })
      return data
    })
    .catch((err) => {
      console.error('VIDEO_GROUPS_FETCH_FAILED:', { error: err, message: err.message })
      throw err
    })
}

function VideoCard({ video }: { video: VideoGroup["videos"][number] }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-border bg-background p-3 transition-colors hover:border-primary/30">
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-foreground">{video.originalName}</p>
        <div className="mt-1 flex items-center gap-3 text-xs text-muted-foreground">
          <span>{formatDate(video.uploadedAt)}</span>
        </div>
      </div>
      <Badge
        variant="outline"
        className={
          video.status === "ready"
            ? "border-primary/30 bg-primary/10 text-primary"
            : video.status === "processing"
              ? "border-chart-4/30 bg-chart-4/10 text-chart-4"
              : "border-destructive/30 bg-destructive/10 text-destructive"
        }
      >
        {video.status === "ready" && <CheckCircle className="mr-1 h-3 w-3" />}
        {video.status === "processing" && <Loader2 className="mr-1 h-3 w-3 animate-spin" />}
        {video.status === "error" && <AlertCircle className="mr-1 h-3 w-3" />}
        {video.status}
      </Badge>
    </div>
  )
}

function GroupCard({
  group,
  isExpanded,
  onToggle,
}: {
  group: VideoGroup
  isExpanded: boolean
  onToggle: () => void
}) {
  return (
    <Card className="gap-0 overflow-hidden border-border bg-card">
      <button
        onClick={onToggle}
        className="flex w-full items-center gap-3 p-4 text-left transition-colors hover:bg-secondary/50"
      >
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10">
          <FolderOpen className="h-5 w-5 text-primary" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold text-foreground">{group.name}</p>
          <p className="mt-0.5 truncate text-xs text-muted-foreground">{group.description}</p>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          <div className="hidden items-center gap-3 text-xs text-muted-foreground sm:flex">
            <span className="flex items-center gap-1">
              <FileVideo className="h-3 w-3" />
              {group.videoCount} video{group.videoCount !== 1 ? "s" : ""}
            </span>
          </div>
          {isExpanded ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}
        </div>
      </button>

      {isExpanded && (
        <div className="border-t border-border bg-secondary/30 p-3">
          {/* Mobile stats */}
          <div className="mb-3 flex items-center gap-3 text-xs text-muted-foreground sm:hidden">
            <span className="flex items-center gap-1">
              <FileVideo className="h-3 w-3" />
              {group.videoCount} video{group.videoCount !== 1 ? "s" : ""}
            </span>
          </div>
          <div className="flex flex-col gap-2">
            {group.videos.map((video) => (
              <VideoCard key={video.id} video={video} />
            ))}
            {group.videos.length === 0 && (
              <p className="py-4 text-center text-sm text-muted-foreground">No videos in this group yet</p>
            )}
          </div>
        </div>
      )}
    </Card>
  )
}

function UploadItem({ upload, onDismiss }: { upload: UploadProgress; onDismiss: () => void }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-border bg-card p-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-primary/10">
        {upload.status === "complete" ? (
          <CheckCircle className="h-4 w-4 text-primary" />
        ) : upload.status === "error" ? (
          <AlertCircle className="h-4 w-4 text-destructive" />
        ) : (
          <Loader2 className="h-4 w-4 animate-spin text-primary" />
        )}
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm text-foreground">{upload.filename}</p>
        <div className="mt-1.5 flex items-center gap-2">
          <Progress
            value={upload.progress}
            className={`h-1.5 flex-1 ${upload.status === "error" ? "bg-destructive/20" : ""}`}
          />
          <span className="shrink-0 text-xs text-muted-foreground">{upload.progress}%</span>
        </div>
        {(upload.status === "processing" || upload.status === "uploading") && (
          <p className="mt-1 text-xs text-muted-foreground animate-pulse">
            {upload.backendStatus || (upload.status === "uploading" ? "Uploading..." : "Analyzing and grouping footage...")}
          </p>
        )}
        {upload.status === "error" && upload.error && (
          <p className="mt-1 text-xs text-destructive">{upload.error}</p>
        )}
      </div>
      {(upload.status === "complete" || upload.status === "error") && (
        <Button variant="ghost" size="icon" className="h-6 w-6 shrink-0" onClick={onDismiss}>
          <X className="h-3.5 w-3.5" />
          <span className="sr-only">Dismiss</span>
        </Button>
      )}
    </div>
  )
}

export function VideoLibrary() {
  const { data, mutate, isLoading } = useSWR<{ groups: VideoGroup[] }>("http://localhost:8000/api/videos/groups", fetcher)
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set())
  const [uploads, setUploads] = useState<UploadProgress[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Poll for background task status
  useEffect(() => {
    const processingUploads = uploads.filter(u => u.status === "processing" && u.id.startsWith("task-"))

    if (processingUploads.length === 0) return

    const pollInterval = setInterval(async () => {
      const updatedUploads = [...uploads]
      let needsUpdate = false

      for (const upload of processingUploads) {
        try {
          // Task ID is stored in the ID but with a prefix
          const taskId = upload.id.replace("task-", "")
          const res = await fetch(`/api/tasks/${taskId}`)

          if (!res.ok) continue

          const data = await res.json()

          const index = updatedUploads.findIndex(u => u.id === upload.id)
          if (index !== -1) {
            const current = updatedUploads[index]

            // Update if status or progress changed
            if (current.backendStatus !== data.status || current.progress !== data.progress) {
              needsUpdate = true
              updatedUploads[index] = {
                ...current,
                backendStatus: data.status,
                progress: data.progress,
                status: data.progress === 100 ? "complete" : data.progress === -1 ? "error" : "processing",
                error: data.progress === -1 ? data.status : undefined
              }

              if (data.progress === 100) {
                mutate() // Refresh library when a task finishes
              }
            }
          }
        } catch (err) {
          console.error("Polling error for task", upload.id, err)
        }
      }

      if (needsUpdate) {
        setUploads(updatedUploads)
      }
    }, 3000)

    return () => clearInterval(pollInterval)
  }, [uploads, mutate])

  const toggleGroup = useCallback((id: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const uploadFile = useCallback(
    async (file: File) => {
      const uploadId = `upload-${Date.now()}-${Math.random().toString(36).slice(2)}`
      console.log('VIDEO_UPLOAD_INITIATED:', {
        uploadId,
        filename: file.name,
        fileSize: file.size,
        fileType: file.type,
        timestamp: new Date().toISOString()
      })

      const newUpload: UploadProgress = {
        id: uploadId,
        filename: file.name,
        progress: 0,
        status: "uploading",
      }

      setUploads((prev) => [newUpload, ...prev])

      // Track the current ID — it changes once backend assigns a task ID
      let currentId = uploadId

      try {
        // Build FormData and upload directly to the backend via Next.js proxy
        const formData = new FormData()
        formData.append("file", file)

        setUploads((prev) =>
          prev.map((u) => (u.id === currentId ? { ...u, progress: 10, backendStatus: "Uploading to server..." } : u))
        )

        const xhr = new XMLHttpRequest()

        const uploadPromise = new Promise<{ task_id: string }>((resolve, reject) => {
          xhr.upload.addEventListener("progress", (e) => {
            if (e.lengthComputable) {
              // Map upload progress to 10-80% range
              const pct = 10 + Math.round((e.loaded / e.total) * 70)
              setUploads((prev) =>
                prev.map((u) => (u.id === currentId ? { ...u, progress: pct } : u))
              )
            }
          })

          xhr.addEventListener("load", () => {
            if (xhr.status >= 200 && xhr.status < 300) {
              try {
                resolve(JSON.parse(xhr.responseText))
              } catch {
                reject(new Error("Invalid response from server"))
              }
            } else {
              reject(new Error(`Upload failed (${xhr.status})`))
            }
          })

          xhr.addEventListener("error", () => reject(new Error("Network error during upload")))
          xhr.addEventListener("abort", () => reject(new Error("Upload aborted")))

          xhr.open("POST", "/api/videos/upload")
          xhr.send(formData)
        })

        setUploads((prev) =>
          prev.map((u) => (u.id === currentId ? { ...u, progress: 80, status: "processing", backendStatus: "Processing video..." } : u))
        )

        const result = await uploadPromise

        // Switch to backend task tracking
        const backendTaskId = result.task_id
        setUploads((prev) =>
          prev.map((u) => (u.id === currentId ? {
            ...u,
            id: `task-${backendTaskId}`,
            progress: 5,
            status: "processing",
            backendStatus: "Processing started..."
          } : u))
        )
        currentId = `task-${backendTaskId}`

        // The useEffect polling will take it from here
      } catch (err: any) {
        setUploads((prev) =>
          prev.map((u) =>
            u.id === currentId ? { ...u, progress: 0, status: "error", error: err.message || "Upload failed. Try again." } : u
          )
        )
      }
    },
    [mutate]
  )

  const handleFileSelect = useCallback(
    (files: FileList | null) => {
      if (!files) return
      Array.from(files).forEach((file) => {
        if (file.type.startsWith("video/")) {
          uploadFile(file)
        }
      })
    },
    [uploadFile]
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragOver(false)
      handleFileSelect(e.dataTransfer.files)
    },
    [handleFileSelect]
  )

  const dismissUpload = useCallback((id: string) => {
    setUploads((prev) => prev.filter((u) => u.id !== id))
  }, [])

  const groups = data?.groups ?? []
  const totalVideos = groups.reduce((sum, g) => sum + g.videoCount, 0)
  const totalDuration = groups.reduce((sum, g) => sum + g.totalDuration, 0)

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 pb-3 md:px-6">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Video className="h-4 w-4 text-primary" />
            <h2 className="text-sm font-semibold text-foreground">Video Library</h2>
          </div>
          {!isLoading && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>{groups.length} group{groups.length !== 1 ? "s" : ""}</span>
              <span className="text-border">|</span>
              <span>{totalVideos} video{totalVideos !== 1 ? "s" : ""}</span>
              <span className="text-border">|</span>
              <span>{formatDuration(totalDuration)}</span>
            </div>
          )}
        </div>
        <div>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            multiple
            className="hidden"
            onChange={(e) => handleFileSelect(e.target.files)}
          />
          <Button
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            className="gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
          >
            <Upload className="h-4 w-4" />
            Upload Video
          </Button>
        </div>
      </div>

      {/* Active uploads */}
      {uploads.length > 0 && (
        <div className="mx-4 mb-3 flex flex-col gap-2 md:mx-6">
          {uploads.map((upload) => (
            <UploadItem key={upload.id} upload={upload} onDismiss={() => dismissUpload(upload.id)} />
          ))}
        </div>
      )}

      {/* Drop zone + groups list */}
      <div
        className="relative mx-4 flex flex-1 flex-col overflow-hidden md:mx-6"
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragOver(true)
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
      >
        {/* Drag overlay */}
        {isDragOver && (
          <div className="absolute inset-0 z-20 flex items-center justify-center rounded-xl border-2 border-dashed border-primary bg-primary/5 backdrop-blur-sm">
            <div className="flex flex-col items-center gap-3">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
                <Upload className="h-7 w-7 text-primary" />
              </div>
              <div className="text-center">
                <p className="text-sm font-semibold text-foreground">Drop videos here</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  They will be automatically analyzed and grouped
                </p>
              </div>
            </div>
          </div>
        )}

        {isLoading ? (
          <div className="flex flex-1 items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
              <p className="text-sm text-muted-foreground">Loading video library...</p>
            </div>
          </div>
        ) : groups.length === 0 ? (
          /* Empty state */
          <div className="flex flex-1 items-center justify-center rounded-xl border border-dashed border-border">
            <div className="flex flex-col items-center gap-4 px-6 py-12">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-secondary">
                <Video className="h-8 w-8 text-muted-foreground" />
              </div>
              <div className="text-center">
                <p className="text-sm font-semibold text-foreground">No videos yet</p>
                <p className="mt-1 max-w-sm text-xs text-muted-foreground">
                  Upload helmet cam footage and the system will automatically analyze and group it by project context.
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => fileInputRef.current?.click()}
                className="gap-2"
              >
                <Upload className="h-4 w-4" />
                Upload your first video
              </Button>
            </div>
          </div>
        ) : (
          <ScrollArea className="flex-1 pb-4">
            <div className="flex flex-col gap-3">
              {groups.map((group) => (
                <GroupCard
                  key={group.id}
                  group={group}
                  isExpanded={expandedGroups.has(group.id)}
                  onToggle={() => toggleGroup(group.id)}
                />
              ))}
            </div>

            {/* Drop hint at bottom */}
            <Separator className="my-4 bg-border" />
            <div className="flex items-center justify-center gap-2 pb-2 text-xs text-muted-foreground">
              <Upload className="h-3 w-3" />
              <span>Drag and drop videos anywhere to upload</span>
            </div>
          </ScrollArea>
        )}
      </div>
    </div>
  )
}
