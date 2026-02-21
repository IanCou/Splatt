"use client"

import { useState, useCallback } from "react"
import { ProjectBar } from "@/components/project-bar"
import { QueryBar } from "@/components/query-bar"
import { SceneViewer } from "@/components/scene-viewer"
import { ResultsPanel } from "@/components/results-panel"
import { LoadingOverlay } from "@/components/loading-overlay"
import { VideoLibrary } from "@/components/video-library"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { AlertCircle, Map, Film } from "lucide-react"
import { hotspots, queryResults } from "@/lib/mock-data"

export default function SplattPage() {
  const [selectedProject, setSelectedProject] = useState("p1")
  const [activeHotspot, setActiveHotspot] = useState<string | null>(null)
  const [highlightedHotspots, setHighlightedHotspots] = useState<string[]>([])
  const [currentResult, setCurrentResult] = useState<(typeof queryResults)[string] | null>(null)
  const [showResults, setShowResults] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [loadingQuery, setLoadingQuery] = useState("")
  const [noResults, setNoResults] = useState(false)
  const [activeTab, setActiveTab] = useState("videos")

  const handleSearch = useCallback(async (query: string) => {
    setIsLoading(true)
    setLoadingQuery(query)
    setNoResults(false)
    setShowResults(false)
    setActiveHotspot(null)

    try {
      const formData = new FormData()
      formData.append("query", query)

      const res = await fetch("http://localhost:8000/api/query", {
        method: "POST",
        body: formData,
      })

      if (!res.ok) throw new Error("Query failed")

      const data = await res.json()

      setIsLoading(false)

      if (data.hotspots && data.hotspots.length > 0) {
        setHighlightedHotspots(data.hotspots)
        const firstMatch = data.hotspots[0]
        setActiveHotspot(firstMatch)

        // Map Gemini response to QueryResult format
        setCurrentResult({
          id: firstMatch,
          location: data.location || "Unknown Location",
          coordinates: data.coordinates || "N/A",
          timestamp: new Date().toLocaleString(),
          worker: data.worker || "System",
          workerRole: data.workerRole || "Assistant",
          description: data.analysis,
          confidence: data.confidence || "Medium",
          thumbnails: [],
          relatedQueries: []
        })

        setShowResults(true)
        setActiveTab("scene")
      } else {
        // Fallback for analysis with no specific hotspots
        if (data.analysis) {
          setHighlightedHotspots([])
          setCurrentResult({
            id: "gemini",
            location: "Scene Wide",
            coordinates: "N/A",
            timestamp: new Date().toLocaleString(),
            worker: "Gemini",
            workerRole: "AI Analyst",
            description: data.analysis,
            confidence: data.confidence || "High",
            thumbnails: [],
            relatedQueries: ["Show overview", "What else is here?"]
          })
          setShowResults(true)
          setActiveTab("scene")
        } else {
          setHighlightedHotspots([])
          setNoResults(true)
        }
      }
    } catch (err) {
      console.error(err)
      setIsLoading(false)
      setNoResults(true)
    }
  }, [])

  const handleHotspotClick = useCallback((id: string) => {
    setActiveHotspot(id)
    setCurrentResult(queryResults[id] || null)
    setShowResults(true)
    setNoResults(false)
  }, [])

  const handleCloseResults = useCallback(() => {
    setShowResults(false)
    setActiveHotspot(null)
    setHighlightedHotspots([])
  }, [])

  const handleRelatedQuery = useCallback(
    (query: string) => {
      handleSearch(query)
    },
    [handleSearch]
  )

  return (
    <div className="flex h-dvh flex-col bg-background">
      <ProjectBar selectedProject={selectedProject} onProjectChange={setSelectedProject} />
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

      {/* Tabs to switch between videos and scene */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex flex-1 flex-col overflow-hidden">
          <div className="px-4 pb-3 md:px-6">
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

          <TabsContent value="videos" className="flex flex-1 flex-col overflow-hidden pb-4">
            <VideoLibrary />
          </TabsContent>

          <TabsContent value="scene" className="flex flex-1 flex-col overflow-hidden">
            <div className="relative flex flex-1 flex-col overflow-hidden px-4 pb-4 md:flex-row md:px-6 md:pb-6">
              <div className="relative flex-1">
                <SceneViewer
                  hotspots={hotspots}
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
