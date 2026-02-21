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

  const simulateSearch = useCallback((query: string) => {
    setIsLoading(true)
    setLoadingQuery(query)
    setNoResults(false)
    setShowResults(false)
    setActiveHotspot(null)

    const lowerQuery = query.toLowerCase()

    setTimeout(() => {
      const matches: string[] = []

      if (lowerQuery.includes("lumber") || lowerQuery.includes("wood") || lowerQuery.includes("2x4")) {
        matches.push("h1")
      }
      if (lowerQuery.includes("concrete") || lowerQuery.includes("mixer") || lowerQuery.includes("pour")) {
        matches.push("h2")
      }
      if (
        lowerQuery.includes("south wall") ||
        lowerQuery.includes("crew") ||
        lowerQuery.includes("framing") ||
        lowerQuery.includes("working")
      ) {
        matches.push("h3")
      }
      if (lowerQuery.includes("rebar") || lowerQuery.includes("delivery") || lowerQuery.includes("deliver")) {
        matches.push("h4")
      }
      if (lowerQuery.includes("scaffold")) {
        matches.push("h5")
      }
      if (lowerQuery.includes("tool") || lowerQuery.includes("storage") || lowerQuery.includes("circular saw")) {
        matches.push("h6")
      }
      if (lowerQuery.includes("yesterday") || lowerQuery.includes("last")) {
        matches.push("h5", "h1")
      }
      if (lowerQuery.includes("morning") || lowerQuery.includes("today")) {
        matches.push("h4", "h2")
      }

      const uniqueMatches = [...new Set(matches)]

      setIsLoading(false)

      if (uniqueMatches.length > 0) {
        setHighlightedHotspots(uniqueMatches)
        const firstMatch = uniqueMatches[0]
        setActiveHotspot(firstMatch)
        setCurrentResult(queryResults[firstMatch] || null)
        setShowResults(true)
        // Switch to scene tab to show results on the map
        setActiveTab("scene")
      } else {
        setHighlightedHotspots([])
        setNoResults(true)
      }
    }, 1800)
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
      simulateSearch(query)
    },
    [simulateSearch]
  )

  return (
    <div className="flex h-dvh flex-col bg-background">
      <ProjectBar selectedProject={selectedProject} onProjectChange={setSelectedProject} />
      <QueryBar onSubmit={simulateSearch} isLoading={isLoading} />

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
