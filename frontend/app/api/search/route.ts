import { NextResponse } from "next/server"
import { createClient } from "@supabase/supabase-js"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || ""
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_KEY || ""
let supabase: ReturnType<typeof createClient> | null = null
if (supabaseUrl && supabaseKey) {
  supabase = createClient(supabaseUrl, supabaseKey)
}

export async function POST(request: Request) {
  try {
    const { query, limit = 10 } = await request.json()

    if (!query || typeof query !== "string") {
      return NextResponse.json({ error: "Query is required" }, { status: 400 })
    }

    const res = await fetch(`${BACKEND_URL}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, limit }),
    })

    if (!res.ok) {
      const errText = await res.text()
      throw new Error(`Backend search failed: ${errText}`)
    }

    const data = await res.json()
    const results = data.results || []

    // Look up video filenames for all unique videoIds
    const videoIds = [...new Set(results.map((r: any) => r.videoId).filter(Boolean))] as string[]
    const videoNames: Record<string, string> = {}

    if (supabase && videoIds.length > 0) {
      try {
        const { data: videos } = await supabase
          .from("videos")
          .select("id, filename")
          .in("id", videoIds)
        for (const v of videos || []) {
          videoNames[v.id] = v.filename
        }
      } catch {
        // Non-critical — fall back to truncated IDs
      }
    }

    // Enrich results with video names
    const enriched = results.map((r: any) => ({
      ...r,
      videoName: videoNames[r.videoId] || null,
    }))

    return NextResponse.json({ query: data.query, results: enriched })
  } catch (error: any) {
    console.error("Search proxy error:", error)
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
