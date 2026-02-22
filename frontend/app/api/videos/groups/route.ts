import { NextResponse } from "next/server"
import type { VideoGroup, Video } from "@/lib/types"
import { createClient } from "@supabase/supabase-js"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || ""
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_KEY || ""

let supabase: ReturnType<typeof createClient> | null = null
if (supabaseUrl && supabaseKey) {
  supabase = createClient(supabaseUrl, supabaseKey)
}

export async function GET() {
  if (!supabase) {
    return NextResponse.json({ error: "Supabase not configured" }, { status: 500 })
  }

  try {
    // Fetch all videos from Supabase
    const { data: videos, error } = await supabase
      .from("videos")
      .select("*")
      .order("upload_timestamp", { ascending: false })

    if (error) {
      console.log("Supabase error:", error)
      throw error
    }

    // Group videos by construction objects detected
    const groupsMap = new Map<string, VideoGroup>()

    videos?.forEach((video: any) => {
      const detections = video.detections || []

      // Extract unique object types to determine grouping
      const objectTypes = new Set<string>()
      detections.forEach((d: any) => {
        if (d.object_type) objectTypes.add(d.object_type)
      })

      // Create a group key based on detected objects (simplified grouping)
      const groupKey = objectTypes.size > 0
        ? Array.from(objectTypes).sort().join(", ")
        : "Ungrouped Videos"

      if (!groupsMap.has(groupKey)) {
        groupsMap.set(groupKey, {
          id: `group-${groupsMap.size + 1}`,
          name: groupKey,
          description: `Videos containing: ${groupKey}`,
          videoCount: 0,
          totalDuration: 0,
          createdAt: video.upload_timestamp,
          videos: [],
        })
      }

      const group = groupsMap.get(groupKey)!

      // Convert Supabase video to frontend Video type
      const videoItem: Video = {
        id: video.id,
        filename: video.filename,
        originalName: video.filename,
        duration: 0, // We don't have duration from analysis yet
        size: 0, // We don't have size from analysis yet
        uploadedAt: video.upload_timestamp,
        status: "ready",
        thumbnailUrl: null,
        videoUrl: video.video_url ?? null,
        storagePath: video.storage_path ?? null,
        groupId: group.id,
      }

      group.videos.push(videoItem)
      group.videoCount++
    })

    const groups = Array.from(groupsMap.values())

    return NextResponse.json({ groups })
  } catch (error: any) {
    console.error("Error fetching videos:", error)
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData()
    const file = formData.get("video") as File | null

    if (!file) {
      return NextResponse.json({ error: "No video file provided" }, { status: 400 })
    }

    // Forward the video to the FastAPI backend's background pipeline
    const backendFormData = new FormData()
    backendFormData.append("file", file)

    const backendResponse = await fetch(`${BACKEND_URL}/process-video`, {
      method: "POST",
      body: backendFormData,
    })

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text()
      throw new Error(`Backend processing failed: ${errorText}`)
    }

    const result = await backendResponse.json()

    return NextResponse.json({
      task_id: result.task_id,
      message: result.message,
      video: {
        id: result.task_id,
        filename: file.name,
        status: "processing",
      }
    })

  } catch (error: any) {
    console.error("Upload error:", error)
    return NextResponse.json(
      { error: error.message || "Upload failed" },
      { status: 500 }
    )
  }
}
