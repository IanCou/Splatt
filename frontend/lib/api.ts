// Base URL for the Python backend (FastAPI/Flask).
// In production, set NEXT_PUBLIC_API_URL to point at the real backend.
// During development, Next.js API routes act as mock proxies.
const API_BASE = process.env.NEXT_PUBLIC_API_URL || ""

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`)
  }
  return res.json()
}

export async function apiPost<T>(path: string, body?: FormData | Record<string, unknown>): Promise<T> {
  const isFormData = body instanceof FormData
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    ...(isFormData
      ? { body }
      : {
          headers: { "Content-Type": "application/json" },
          body: body ? JSON.stringify(body) : undefined,
        }),
  })
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`)
  }
  return res.json()
}

export async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: "DELETE" })
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`)
  }
  return res.json()
}
