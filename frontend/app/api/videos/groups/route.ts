import { NextResponse } from "next/server"
import type { VideoGroup, Video } from "@/lib/types"

// In-memory mock store. This whole file gets replaced by your FastAPI backend.
// The real backend will persist to a DB and run ML-based grouping.

let nextVideoId = 7
let nextGroupId = 4

const mockVideos: Video[] = [
  {
    id: "v1",
    filename: "helmet-cam-martinez-0219-1400.mp4",
    originalName: "GoPro_0219_1400.mp4",
    duration: 1823,
    size: 524_288_000,
    uploadedAt: "2026-02-19T14:00:00Z",
    status: "ready",
    thumbnailUrl: null,
    groupId: "g1",
  },
  {
    id: "v2",
    filename: "helmet-cam-chen-0219-1100.mp4",
    originalName: "GoPro_0219_1100.mp4",
    duration: 2410,
    size: 712_000_000,
    uploadedAt: "2026-02-19T11:00:00Z",
    status: "ready",
    thumbnailUrl: null,
    groupId: "g1",
  },
  {
    id: "v3",
    filename: "helmet-cam-torres-0219-1500.mp4",
    originalName: "Insta360_0219_1500.mp4",
    duration: 3600,
    size: 1_048_000_000,
    uploadedAt: "2026-02-19T15:00:00Z",
    status: "ready",
    thumbnailUrl: null,
    groupId: "g2",
  },
  {
    id: "v4",
    filename: "helmet-cam-park-0219-0800.mp4",
    originalName: "DJI_0219_0800.mp4",
    duration: 1200,
    size: 356_000_000,
    uploadedAt: "2026-02-19T08:00:00Z",
    status: "ready",
    thumbnailUrl: null,
    groupId: "g2",
  },
  {
    id: "v5",
    filename: "helmet-cam-ruiz-0218-1600.mp4",
    originalName: "GoPro_0218_1600.mp4",
    duration: 2700,
    size: 890_000_000,
    uploadedAt: "2026-02-18T16:00:00Z",
    status: "ready",
    thumbnailUrl: null,
    groupId: "g3",
  },
  {
    id: "v6",
    filename: "helmet-cam-nguyen-0219-1730.mp4",
    originalName: "Insta360_0219_1730.mp4",
    duration: 900,
    size: 278_000_000,
    uploadedAt: "2026-02-19T17:30:00Z",
    status: "ready",
    thumbnailUrl: null,
    groupId: "g3",
  },
]

const mockGroups: { id: string; name: string; description: string; createdAt: string }[] = [
  {
    id: "g1",
    name: "Foundation & Concrete Work",
    description: "Videos covering concrete pours, mixer operation, and foundation prep from Feb 19",
    createdAt: "2026-02-19T11:00:00Z",
  },
  {
    id: "g2",
    name: "Framing & Structural",
    description: "South wall framing, header installation, and structural deliveries",
    createdAt: "2026-02-19T08:00:00Z",
  },
  {
    id: "g3",
    name: "Safety & Inventory",
    description: "Scaffolding inspections, tool inventory checks, and safety walkthroughs",
    createdAt: "2026-02-18T16:00:00Z",
  },
]

function buildGroupResponse(): VideoGroup[] {
  return mockGroups.map((g) => {
    const vids = mockVideos.filter((v) => v.groupId === g.id)
    return {
      ...g,
      videoCount: vids.length,
      totalDuration: vids.reduce((sum, v) => sum + v.duration, 0),
      videos: vids,
    }
  })
}

export async function GET() {
  // Simulate slight network latency
  await new Promise((r) => setTimeout(r, 300))
  return NextResponse.json({ groups: buildGroupResponse() })
}

// POST = upload a new video. The backend would:
// 1. Store the file
// 2. Run frame extraction + ML grouping
// 3. Return the assigned group
export async function POST(request: Request) {
  await new Promise((r) => setTimeout(r, 1500))

  const formData = await request.formData()
  const file = formData.get("video") as File | null

  if (!file) {
    return NextResponse.json({ error: "No video file provided" }, { status: 400 })
  }

  // Simulate ML grouping: randomly assign to an existing group or create a new one
  const rand = Math.random()
  let assignedGroupId: string

  if (rand < 0.6 && mockGroups.length > 0) {
    // Assign to existing group
    assignedGroupId = mockGroups[Math.floor(Math.random() * mockGroups.length)].id
  } else {
    // Create a new group (simulates ML discovering a new cluster)
    const newGroup = {
      id: `g${nextGroupId++}`,
      name: `Auto-grouped: ${file.name.replace(/\.[^.]+$/, "")}`,
      description: `Automatically categorized footage from ${file.name}`,
      createdAt: new Date().toISOString(),
    }
    mockGroups.push(newGroup)
    assignedGroupId = newGroup.id
  }

  const newVideo: Video = {
    id: `v${nextVideoId++}`,
    filename: `processed-${file.name}`,
    originalName: file.name,
    duration: Math.floor(Math.random() * 3600) + 300,
    size: file.size,
    uploadedAt: new Date().toISOString(),
    status: "ready",
    thumbnailUrl: null,
    groupId: assignedGroupId,
  }

  mockVideos.push(newVideo)

  const group = mockGroups.find((g) => g.id === assignedGroupId)!

  return NextResponse.json({
    video: newVideo,
    group: {
      id: group.id,
      name: group.name,
    },
    message: `Video categorized into "${group.name}"`,
  })
}
