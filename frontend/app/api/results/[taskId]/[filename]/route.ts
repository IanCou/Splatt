import { NextRequest, NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

export async function GET(
    request: NextRequest,
    { params }: { params: Promise<{ taskId: string; filename: string }> }
) {
    const { taskId, filename } = await params

    // Only allow known result files
    const allowed = ["baked_splat.ply", "baked_splat_sh.ply", "transforms.json"]
    if (!allowed.includes(filename)) {
        return NextResponse.json({ error: "Not found" }, { status: 404 })
    }

    try {
        const res = await fetch(`${BACKEND_URL}/results/${taskId}/${filename}`, {
            cache: "no-store",
        })

        if (!res.ok) {
            return NextResponse.json(
                { error: "File not found" },
                { status: res.status }
            )
        }

        // Stream the response directly from the backend
        return new NextResponse(res.body, {
            headers: {
                "Content-Type":
                    res.headers.get("Content-Type") || "application/octet-stream",
                "Content-Length": res.headers.get("Content-Length") || "",
                "Cache-Control": "public, max-age=3600",
            },
        })
    } catch (error: unknown) {
        const message = error instanceof Error ? error.message : "Unknown error"
        console.error("Results proxy error:", message)
        return NextResponse.json({ error: message }, { status: 500 })
    }
}
