export interface Video {
  id: string
  filename: string
  originalName: string
  duration: number // seconds
  size: number // bytes
  uploadedAt: string
  status: "processing" | "ready" | "error"
  thumbnailUrl: string | null
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
}

export interface UploadProgress {
  id: string
  filename: string
  progress: number // 0-100
  status: "uploading" | "processing" | "complete" | "error"
  error?: string
}
