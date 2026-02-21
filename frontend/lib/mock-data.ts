export interface Hotspot {
  id: string
  label: string
  x: number
  y: number
  type: "material" | "equipment" | "worker" | "event"
}

export interface QueryResult {
  id: string
  location: string
  coordinates: string
  timestamp: string
  worker: string
  workerRole: string
  description: string
  confidence: "High" | "Medium" | "Low"
  thumbnails: string[]
  relatedQueries: string[]
}

export const hotspots: Hotspot[] = [
  { id: "h1", label: "Lumber Stack", x: 32, y: 55, type: "material" },
  { id: "h2", label: "Concrete Mixer", x: 58, y: 42, type: "equipment" },
  { id: "h3", label: "South Wall Crew", x: 75, y: 65, type: "worker" },
  { id: "h4", label: "Rebar Delivery", x: 20, y: 35, type: "event" },
  { id: "h5", label: "Scaffolding", x: 48, y: 25, type: "equipment" },
  { id: "h6", label: "Tool Storage", x: 85, y: 30, type: "material" },
]

export const queryResults: Record<string, QueryResult> = {
  h1: {
    id: "h1",
    location: "South wall exterior, 12ft from loading zone",
    coordinates: "(X: 145, Y: 22, Z: 8)",
    timestamp: "Feb 19, 2026 - 2:15 PM",
    worker: "John Martinez",
    workerRole: "Foreman",
    description:
      "Lumber stack visible in frame. Approximately 40 2x4s stacked horizontally. Protective tarp partially covering. Concrete mixer visible 6ft to the left.",
    confidence: "High",
    thumbnails: [
      "Frame 1: Wide angle view of lumber stack against south wall",
      "Frame 2: Close-up of tarp covering lumber",
      "Frame 3: Adjacent area showing concrete mixer",
    ],
    relatedQueries: [
      "Show nearby equipment",
      "Who moved lumber last week?",
      "Check tarp condition over time",
    ],
  },
  h2: {
    id: "h2",
    location: "Central work area, next to east foundation",
    coordinates: "(X: 210, Y: 45, Z: 3)",
    timestamp: "Feb 19, 2026 - 11:30 AM",
    worker: "Sarah Chen",
    workerRole: "Equipment Operator",
    description:
      "Concrete mixer positioned for east foundation pour. Hopper loaded with aggregate. Two workers visible preparing forms nearby.",
    confidence: "High",
    thumbnails: [
      "Frame 1: Mixer in operation",
      "Frame 2: Workers preparing forms",
      "Frame 3: Overview of foundation area",
    ],
    relatedQueries: [
      "When was last pour completed?",
      "Show foundation progress",
      "Track mixer usage today",
    ],
  },
  h3: {
    id: "h3",
    location: "South wall, second floor framing",
    coordinates: "(X: 290, Y: 68, Z: 22)",
    timestamp: "Feb 19, 2026 - 3:45 PM",
    worker: "Mike Torres",
    workerRole: "Carpenter",
    description:
      "Three-person crew visible working on south wall framing. Second-floor headers being installed. Safety harnesses properly attached.",
    confidence: "Medium",
    thumbnails: [
      "Frame 1: Crew installing headers",
      "Frame 2: Framing progress overview",
      "Frame 3: Safety equipment check",
    ],
    relatedQueries: [
      "Show framing progress this week",
      "Who else worked on south wall?",
      "Check safety compliance",
    ],
  },
  h4: {
    id: "h4",
    location: "North entrance, delivery staging area",
    coordinates: "(X: 80, Y: 15, Z: 2)",
    timestamp: "Feb 19, 2026 - 8:00 AM",
    worker: "David Park",
    workerRole: "Site Superintendent",
    description:
      "Rebar delivery truck at north gate. Approximately 200 pieces #5 rebar unloaded. Delivery manifest signed by D. Park at 8:00 AM.",
    confidence: "High",
    thumbnails: [
      "Frame 1: Delivery truck at gate",
      "Frame 2: Rebar being unloaded",
      "Frame 3: Manifest signing",
    ],
    relatedQueries: [
      "Track all deliveries this week",
      "Where is rebar stored now?",
      "Check delivery schedule",
    ],
  },
  h5: {
    id: "h5",
    location: "West facade, floors 1-3",
    coordinates: "(X: 180, Y: 30, Z: 35)",
    timestamp: "Feb 18, 2026 - 4:00 PM",
    worker: "Carlos Ruiz",
    workerRole: "Safety Officer",
    description:
      "Scaffolding erected along west facade. Three-tier structure supporting exterior finish work. All guardrails and toe boards in place.",
    confidence: "High",
    thumbnails: [
      "Frame 1: Full scaffolding view",
      "Frame 2: Connection points detail",
      "Frame 3: Guardrail inspection",
    ],
    relatedQueries: [
      "When was scaffolding last inspected?",
      "Show west facade progress",
      "Check scaffold load ratings",
    ],
  },
  h6: {
    id: "h6",
    location: "East lot, secured container #3",
    coordinates: "(X: 340, Y: 28, Z: 4)",
    timestamp: "Feb 19, 2026 - 5:30 PM",
    worker: "Lisa Nguyen",
    workerRole: "Tool Manager",
    description:
      "Tool storage container opened for end-of-day inventory. Power tools returned and logged. Circular saw missing from inventory - flagged.",
    confidence: "Medium",
    thumbnails: [
      "Frame 1: Container interior overview",
      "Frame 2: Tool check-in process",
      "Frame 3: Missing tool alert",
    ],
    relatedQueries: [
      "Who last used the circular saw?",
      "Show full tool inventory",
      "Track missing equipment",
    ],
  },
}

export const sampleQueries = [
  "Where did we leave the scaffolding on Tuesday?",
  "Show me where the rebar was delivered",
  "Who was working on the south wall yesterday?",
  "Find the last time we used the concrete mixer",
  "What deliveries came in this morning?",
  "Where is the tool storage container?",
]

export const projects = [
  { id: "p1", name: "Riverside Tower - Phase 2", footage: 24, hours: 6.2 },
  { id: "p2", name: "Oakwood Mall Renovation", footage: 18, hours: 4.8 },
  { id: "p3", name: "Highway 101 Bridge Repair", footage: 31, hours: 8.1 },
]
