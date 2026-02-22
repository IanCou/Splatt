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
    <div className="flex h-dvh flex-col bg-background selection:bg-white selection:text-black">
      <ProjectBar selectedProject={selectedProject} onProjectChange={setSelectedProject} />

      <main className="flex flex-1 flex-col overflow-hidden max-w-[1400px] mx-auto w-full">
        <QueryBar onSubmit={simulateSearch} isLoading={isLoading} />

        {/* No results state */}
        {noResults && !showResults && (
          <div className="mx-4 mb-4 flex items-center gap-3 rounded-lg border border-red-500/20 bg-red-500/5 px-4 py-4 md:mx-6">
            <AlertCircle className="h-5 w-5 shrink-0 text-red-500" />
            <p className="text-sm font-medium text-red-200">
              No footage found. Try broader terms or select a site point manually.
            </p>
          </div>
        )}

        {/* Highlighted count */}
        {highlightedHotspots.length > 0 && !isLoading && !showResults && (
          <div className="mx-4 mb-4 flex items-center gap-3 md:mx-6">
            <Badge variant="outline" className="h-7 gap-1.5 border-white/10 bg-white/5 px-3 text-[11px] font-bold uppercase tracking-wider text-white">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-white opacity-75"></span>
                <span className="relative inline-flex h-2 w-2 rounded-full bg-white"></span>
              </span>
              {highlightedHotspots.length} match{highlightedHotspots.length !== 1 ? "es" : ""} found
            </Badge>
          </div>
        )}

        <div className="flex flex-1 flex-col overflow-hidden px-4 md:px-6">
          <div className="flex flex-1 flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#0a0a0a] shadow-[0_0_50px_-12px_rgba(0,0,0,0.5)]">
            {showResults && currentResult ? (
              <ResultsPanel
                result={currentResult}
                hotspots={hotspots}
                activeHotspot={activeHotspot}
                highlightedHotspots={highlightedHotspots}
                onHotspotClick={handleHotspotClick}
                onClose={handleCloseResults}
                onRelatedQuery={handleRelatedQuery}
              />
            ) : (
              <Tabs value={activeTab} onValueChange={setActiveTab} className="flex flex-1 flex-col overflow-hidden">
                <div className="flex items-center justify-between border-b border-white/5 bg-black/40 px-6 py-3 backdrop-blur-xl">
                  <TabsList className="h-9 bg-white/5 p-1">
                    <TabsTrigger value="videos" className="h-7 gap-2 px-4 text-[13px] font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-black">
                      <Film className="h-3.5 w-3.5" />
                      Videos
                    </TabsTrigger>
                    <TabsTrigger value="scene" className="h-7 gap-2 px-4 text-[13px] font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-black">
                      <Map className="h-3.5 w-3.5" />
                      Scene
                    </TabsTrigger>
                  </TabsList>

                  <div className="hidden items-center gap-4 md:flex">
                    <div className="h-1 w-24 rounded-full bg-white/5 overflow-hidden">
                      <div className="h-full bg-white/20 w-3/4"></div>
                    </div>
                    <span className="text-[10px] uppercase tracking-widest text-white/40 font-bold">Storage: 75%</span>
                  </div>
                </div>

                <div className="flex-1 overflow-hidden relative">
                  <TabsContent value="videos" className="m-0 flex h-full flex-col overflow-hidden">
                    <VideoLibrary />
                  </TabsContent>

                  <TabsContent value="scene" className="m-0 flex h-full flex-col overflow-hidden">
                    <SceneViewer
                      hotspots={hotspots}
                      activeHotspot={activeHotspot}
                      onHotspotClick={handleHotspotClick}
                      highlightedHotspots={highlightedHotspots}
                    />
                    <LoadingOverlay isVisible={isLoading} query={loadingQuery} />
                  </TabsContent>
                </div>
              </Tabs>
            )}
          </div>
        </div>

        <footer className="py-6 px-6 flex items-center justify-between opacity-40 hover:opacity-100 transition-opacity">
          <div className="flex items-center gap-4 text-[10px] uppercase tracking-widest font-bold text-white">
            <span className="hover:text-blue-500 cursor-pointer transition-colors">Safety Protocol v4.2</span>
            <span className="w-1 h-1 rounded-full bg-white/20"></span>
            <span className="hover:text-blue-500 cursor-pointer transition-colors">Neural Index v1.0.8</span>
          </div>
          <p className="text-[10px] uppercase tracking-widest font-bold text-white">
            Splatt &copy; 2026
          </p>
        </footer>
      </main>
    </div>
  )
}
