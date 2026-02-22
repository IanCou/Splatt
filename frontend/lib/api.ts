// Base URL for the Python backend (FastAPI/Flask).
// In production, set NEXT_PUBLIC_API_URL to point at the real backend.
// During development, Next.js API routes act as mock proxies.
const API_BASE = process.env.NEXT_PUBLIC_API_URL || ""

export async function apiGet<T>(path: string): Promise<T> {
  const fullUrl = `${API_BASE}${path}`
  console.log(`API_GET_REQUEST: ${fullUrl}`)

  try {
    const res = await fetch(fullUrl)
    console.log(`API_GET_RESPONSE: ${fullUrl} - Status: ${res.status} ${res.statusText}`)

    if (!res.ok) {
      const errorText = await res.text().catch(() => 'Unable to read error response')
      console.error(`API_GET_ERROR: ${fullUrl} - ${res.status} ${res.statusText}`, errorText)
      throw new Error(`API error ${res.status}: ${res.statusText}`)
    }

    const data = await res.json()
    console.log(`API_GET_SUCCESS: ${fullUrl}`, data)
    return data
  } catch (error) {
    console.error(`API_GET_FAILURE: ${fullUrl}`, error)
    throw error
  }
}

export async function apiPost<T>(path: string, body?: FormData | Record<string, unknown>): Promise<T> {
  const fullUrl = `${API_BASE}${path}`
  const isFormData = body instanceof FormData

  console.log(`API_POST_REQUEST: ${fullUrl}`, {
    bodyType: isFormData ? 'FormData' : 'JSON',
    bodyPreview: isFormData ? Array.from((body as FormData).keys()).join(', ') : body
  })

  try {
    const res = await fetch(fullUrl, {
      method: "POST",
      ...(isFormData
        ? { body }
        : {
            headers: { "Content-Type": "application/json" },
            body: body ? JSON.stringify(body) : undefined,
          }),
    })

    console.log(`API_POST_RESPONSE: ${fullUrl} - Status: ${res.status} ${res.statusText}`)

    if (!res.ok) {
      const errorText = await res.text().catch(() => 'Unable to read error response')
      console.error(`API_POST_ERROR: ${fullUrl} - ${res.status} ${res.statusText}`, errorText)
      throw new Error(`API error ${res.status}: ${res.statusText}`)
    }

    const data = await res.json()
    console.log(`API_POST_SUCCESS: ${fullUrl}`, data)
    return data
  } catch (error) {
    console.error(`API_POST_FAILURE: ${fullUrl}`, error)
    throw error
  }
}

export async function apiDelete<T>(path: string): Promise<T> {
  const fullUrl = `${API_BASE}${path}`
  console.log(`API_DELETE_REQUEST: ${fullUrl}`)

  try {
    const res = await fetch(fullUrl, { method: "DELETE" })
    console.log(`API_DELETE_RESPONSE: ${fullUrl} - Status: ${res.status} ${res.statusText}`)

    if (!res.ok) {
      const errorText = await res.text().catch(() => 'Unable to read error response')
      console.error(`API_DELETE_ERROR: ${fullUrl} - ${res.status} ${res.statusText}`, errorText)
      throw new Error(`API error ${res.status}: ${res.statusText}`)
    }

    const data = await res.json()
    console.log(`API_DELETE_SUCCESS: ${fullUrl}`, data)
    return data
  } catch (error) {
    console.error(`API_DELETE_FAILURE: ${fullUrl}`, error)
    throw error
  }
}
