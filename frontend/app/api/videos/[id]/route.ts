import { NextResponse } from "next/server"

// Placeholder for individual video operations.
// The real backend (FastAPI/Flask) will handle these.

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params
  await new Promise((r) => setTimeout(r, 400))
  return NextResponse.json({ deleted: id })
}
