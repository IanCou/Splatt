import { NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

// Allow large video uploads (up to 500MB)
export const config = {
  api: {
    bodyParser: false,
  },
}

export const maxDuration = 300 // 5 minutes timeout for large uploads

export async function POST(request: Request) {
  try {
    const formData = await request.formData()
    const file = formData.get("file")

    if (!file || !(file instanceof Blob)) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    // Forward the file as multipart/form-data to the FastAPI backend
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
    })
  } catch (error: any) {
    console.error("Upload proxy error:", error)
    return NextResponse.json(
      { error: error.message || "Upload failed" },
      { status: 500 }
    )
  }
}
