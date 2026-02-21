// Re-export shared types so existing imports of these from mock-data still compile.
export type { Hotspot, QueryResult } from "@/lib/types"

// Sample query suggestions shown in the query bar placeholder.
export const sampleQueries = [
  "Where did we leave the scaffolding on Tuesday?",
  "Show me where the rebar was delivered",
  "Who was working on the south wall yesterday?",
  "Find the last time we used the concrete mixer",
  "What deliveries came in this morning?",
  "Where is the tool storage container?",
]
