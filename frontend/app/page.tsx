"use client"

import { useState, useCallback } from "react"
import useSWR from "swr"
import { ProjectBar } from "@/components/project-bar"
import { QueryBar } from "@/components/query-bar"
import { SceneViewer } from "@/components/scene-viewer"
import { ResultsPanel } from "@/components/results-panel"
import { LoadingOverlay } from "@/components/loading-overlay"
import { VideoLibrary } from "@/components/video-library"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { AlertCircle, Map, Film } from "lucide-react"
import type { VideoGroup, Hotspot, QueryResult } from "@/lib/types"

const fetcher = (url: string) => fetch(url).then((r) => r.json())

export default function SplattPage() {
  const [selectedProject, setSelectedProject] = useState<string | null>(null)
  const [activeHotspot, setActiveHotspot] = useState<string | null>(null)
  const [highlightedHotspots, setHighlightedHotspots] = useState<string[]>([])
  const [currentResult, setCurrentResult] = useState<QueryResult | null>(null)
  const [showResults, setShowResults] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [loadingQuery, setLoadingQuery] = useState("")
  const [noResults, setNoResults] = useState(false)
  const [activeTab, setActiveTab] = useState("videos")

  // Fetch all groups from the backend
  const { data } = useSWR<{ groups: VideoGroup[] }>(
    "http://localhost:8000/api/videos/groups",
    fetcher,
    { refreshInterval: 5000 }
  )

  const groups = data?.groups ?? []

  // Auto-select the first group when groups load
  const effectiveProject =
    selectedProject && groups.find((g) => g.id === selectedProject)
      ? selectedProject
      : groups[0]?.id ?? null

  // Hotspots from the currently selected group (real AI-generated data)
  const activeGroupHotspots: Hotspot[] =
    groups.find((g) => g.id === effectiveProject)?.hotspots ?? []

  // ── Shared query helper ──────────────────────────────────────────────────────

  const runQuery = useCallback(
    async (query: string, hotspotId?: string) => {
      setIsLoading(true)
      setLoadingQuery(query)
      setNoResults(false)
      setShowResults(false)
      setActiveHotspot(null)

      try {
        const formData = new FormData()
        formData.append("query", query)
        if (effectiveProject) formData.append("group_id", effectiveProject)

        const res = await fetch("http://localhost:8000/api/query", {
          method: "POST",
          body: formData,
        })

        if (!res.ok) throw new Error("Query failed")

        const apiData = await res.json()
        setIsLoading(false)

        const buildResult = (overrides: Partial<QueryResult>): QueryResult => ({
          id: hotspotId ?? "gemini",
          location: apiData.location || "Scene Wide",
          coordinates: apiData.coordinates || "N/A",
          timestamp: new Date().toLocaleString(),
          worker: apiData.worker || "Gemini",
          workerRole: apiData.workerRole || "AI Analyst",
          description: apiData.analysis || "No analysis available.",
          confidence: apiData.confidence || "Medium",
          thumbnails: [],
          relatedQueries: [],
          relatedGroupIds: apiData.relatedGroupIds || [],
          ...overrides,
        })

        const returnedHotspots: string[] = apiData.hotspots ?? []

        if (returnedHotspots.length > 0) {
          setHighlightedHotspots(returnedHotspots)
          setActiveHotspot(hotspotId ?? returnedHotspots[0])
          setCurrentResult(buildResult({ id: hotspotId ?? returnedHotspots[0] }))
          setShowResults(true)
          setActiveTab("scene")
        } else if (apiData.analysis) {
          setHighlightedHotspots(hotspotId ? [hotspotId] : [])
          setActiveHotspot(hotspotId ?? null)
          setCurrentResult(buildResult({}))
          setShowResults(true)
          setActiveTab("scene")
        } else {
          setHighlightedHotspots([])
          setNoResults(true)
        }
      } catch (err) {
        console.error(err)
        setIsLoading(false)
        setNoResults(true)
      }
    },
    [effectiveProject]
  )

  // ── Handlers ─────────────────────────────────────────────────────────────────

  const handleSearch = useCallback(
    (query: string) => runQuery(query),
    [runQuery]
  )

  // When a hotspot is clicked, query the API with its label so we get real AI analysis
  const handleHotspotClick = useCallback(
    (id: string) => {
      const spot = activeGroupHotspots.find((h) => h.id === id)
      const query = spot ? `Tell me about ${spot.label}` : `Hotspot ${id}`
      runQuery(query, id)
    },
    [activeGroupHotspots, runQuery]
  )

  const handleCloseResults = useCallback(() => {
    setShowResults(false)
    setActiveHotspot(null)
    setHighlightedHotspots([])
  }, [])

  const handleRelatedQuery = useCallback(
    (query: string) => handleSearch(query),
    [handleSearch]
  )

  return (
    <div className="flex h-dvh flex-col bg-background overflow-hidden">
      <ProjectBar selectedProject={effectiveProject ?? ""} onProjectChange={setSelectedProject} />
      <QueryBar onSubmit={handleSearch} isLoading={isLoading} />

      {/* No results state */}
      {noResults && (
        <div className="mx-4 mb-3 flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 md:mx-6">
          <AlertCircle className="h-4 w-4 shrink-0 text-destructive" />
          <p className="text-sm text-foreground">
            No footage found for this query. Try broader terms or click a hotspot on the scene.
          </p>
        </div>
      )}

      {/* Highlighted count */}
      {highlightedHotspots.length > 0 && !isLoading && (
        <div className="mx-4 mb-3 flex items-center gap-2 md:mx-6">
          <Badge variant="outline" className="gap-1.5 border-primary/30 bg-primary/10 text-primary">
            {highlightedHotspots.length} result{highlightedHotspots.length !== 1 ? "s" : ""} found
          </Badge>
          <span className="text-xs text-muted-foreground">Click a highlighted point to view details</span>
        </div>
      )}

      {/* Main content area */}
      <div className="flex flex-1 flex-col overflow-hidden min-h-0">
        <Tabs
          value={activeTab}
          onValueChange={setActiveTab}
          className="flex flex-1 flex-col overflow-hidden min-h-0"
        >
          <div className="px-4 pb-3 md:px-6 shrink-0">
            <TabsList className="bg-secondary">
              <TabsTrigger value="videos" className="gap-1.5 data-[state=active]:bg-card data-[state=active]:text-foreground">
                <Film className="h-3.5 w-3.5" />
                Videos
              </TabsTrigger>
              <TabsTrigger value="scene" className="gap-1.5 data-[state=active]:bg-card data-[state=active]:text-foreground">
                <Map className="h-3.5 w-3.5" />
                Site Scene
              </TabsTrigger>
            </TabsList>
          </div>

          {/* Videos tab */}
          <TabsContent value="videos" className="flex flex-1 flex-col overflow-hidden min-h-0 mt-0">
            <VideoLibrary />
          </TabsContent>

          {/* Scene tab */}
          <TabsContent value="scene" className="flex flex-1 overflow-hidden min-h-0 mt-0">
            <div className="flex flex-1 overflow-hidden px-4 pb-4 md:px-6 md:pb-6 gap-4">
              {/* Scene canvas wrapper */}
              <div className="relative flex-1 min-h-0">
                <SceneViewer
                  hotspots={activeGroupHotspots}
                  activeHotspot={activeHotspot}
                  onHotspotClick={handleHotspotClick}
                  highlightedHotspots={highlightedHotspots}
                />
                <LoadingOverlay isVisible={isLoading} query={loadingQuery} />
              </div>

              <ResultsPanel
                result={currentResult}
                onClose={handleCloseResults}
                onRelatedQuery={handleRelatedQuery}
                isVisible={showResults}
              />
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
