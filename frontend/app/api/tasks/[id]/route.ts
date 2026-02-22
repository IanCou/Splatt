import { NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

export async function GET(
    request: Request,
    { params }: { params: Promise<{ id: string }> }
) {
    const { id: taskId } = await params

    try {
        const res = await fetch(`${BACKEND_URL}/tasks/${taskId}`, {
            cache: 'no-store'
        })

        if (!res.ok) {
            if (res.status === 404) {
                return NextResponse.json({ error: "Task not found" }, { status: 404 })
            }
            throw new Error("Failed to fetch task status from backend")
        }

        const data = await res.json()
        return NextResponse.json(data)
    } catch (error: any) {
        console.error("Task status proxy error:", error)
        return NextResponse.json({ error: error.message }, { status: 500 })
    }
}
