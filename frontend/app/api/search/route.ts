import { NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

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
    return NextResponse.json(data)
  } catch (error: any) {
    console.error("Search proxy error:", error)
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
