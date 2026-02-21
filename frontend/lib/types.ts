export interface Hotspot {
  id: string
  label: string
  x: number
  y: number
  type: "material" | "equipment" | "worker" | "event"
}

export interface QueryResult {
  id: string
  location: string
  coordinates: string
  timestamp: string
  worker: string
  workerRole: string
  description: string
  confidence: "High" | "Medium" | "Low"
  thumbnails: string[]
  relatedQueries: string[]
  relatedGroupIds?: string[]
  videoUrl?: string
}

export interface Video {
  id: string
  filename: string
  originalName: string
  duration: number // seconds
  size: number // bytes
  uploadedAt: string
  status: "processing" | "ready" | "error"
  thumbnail: string | null
  url: string | null
  groupId: string | null
}

export interface VideoGroup {
  id: string
  name: string
  description: string
  videoCount: number
  totalDuration: number // seconds
  createdAt: string
  videos: Video[]
  hotspots: Hotspot[]
  keyframes: string[]
}

export interface UploadProgress {
  id: string
  filename: string
  progress: number // 0–100
  status: "uploading" | "processing" | "complete" | "error"
  error?: string
}
