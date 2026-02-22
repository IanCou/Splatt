import { NextResponse } from "next/server"
import { createClient } from "@supabase/supabase-js"

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || ""
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_KEY || ""

let supabase: ReturnType<typeof createClient> | null = null
if (supabaseUrl && supabaseKey) {
  supabase = createClient(supabaseUrl, supabaseKey)
}

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params

  if (!supabase) {
    return NextResponse.json({ error: "Supabase not configured" }, { status: 500 })
  }

  try {
    // Fetch detections with coordinates for this video
    const { data, error } = await supabase
      .from("detections")
      .select("id, frame_number, seconds, object_type, description, distance_estimate, x, y, z, rx, ry, rz")
      .eq("video_id", id)
      .not("x", "is", null)
      .order("frame_number", { ascending: true })

    if (error) throw error

    const detections = (data ?? []) as Array<{
      id: string
      frame_number: number
      seconds: number
      object_type: string
      description: string
      distance_estimate: number | null
      x: number
      y: number
      z: number
      rx: number
      ry: number
      rz: number
    }>

    // Group all detections by frame_number for the camera path + per-frame detail
    const frameMap = new Map<number, { camera: (typeof detections)[0]; detections: Array<{ objectType: string; description: string; distanceEstimate: number | null }> }>()
    detections.forEach((det) => {
      if (!frameMap.has(det.frame_number)) {
        frameMap.set(det.frame_number, { camera: det, detections: [] })
      }
      frameMap.get(det.frame_number)!.detections.push({
        objectType: det.object_type,
        description: det.description,
        distanceEstimate: det.distance_estimate,
      })
    })

    const points = Array.from(frameMap.values()).map(({ camera, detections: dets }) => ({
      frameNumber: camera.frame_number,
      seconds: camera.seconds,
      x: camera.x,
      y: camera.y,
      z: camera.z,
      rx: camera.rx,
      ry: camera.ry,
      rz: camera.rz,
      objectType: camera.object_type,
      description: camera.description,
      detections: dets,
    }))

    return NextResponse.json({ videoId: id, points })
  } catch (error: any) {
    console.error("Error fetching coordinates:", error)
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
