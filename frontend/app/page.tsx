"use client"

import { useState, useCallback, useEffect } from "react"
import dynamic from "next/dynamic"
import { ProjectBar } from "@/components/project-bar"
import { QueryBar } from "@/components/query-bar"
import { LoadingOverlay } from "@/components/loading-overlay"
import { VideoLibrary } from "@/components/video-library"
import { MapViewer } from "@/components/map-viewer"
import { SearchResults } from "@/components/search-results"
import { Badge } from "@/components/ui/badge"
import { AlertCircle, Map, Box, Search, X } from "lucide-react"

// Dynamic import — Three.js requires browser globals
const PlyViewer = dynamic(
  () => import("@/components/ply-viewer").then((mod) => mod.PlyViewer),
  { ssr: false }
)

interface SearchHit {
  description: string
  score: number
  videoId: string | null
  videoName: string | null
  objectType: string | null
  seconds: number | null
  frameNumber: number | null
  distanceEstimate: number | null
  x: number | null
  y: number | null
  z: number | null
}

export default function SplattPage() {
  const [selectedProject, setSelectedProject] = useState("p1")
  const [isLoading, setIsLoading] = useState(false)
  const [loadingQuery, setLoadingQuery] = useState("")
  const [noResults, setNoResults] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [selectedVideo, setSelectedVideo] = useState<{ id: string; name: string } | null>(null)
  const [searchResults, setSearchResults] = useState<SearchHit[] | null>(null)
  const [searchQuery, setSearchQuery] = useState("")
  const [viewMode, setViewMode] = useState<"splat" | "map">("splat")

  const handleSearch = useCallback(async (query: string) => {
    setIsLoading(true)
    setLoadingQuery(query)
    setNoResults(false)
    setSearchError(null)
    setSearchResults(null)
    setSelectedVideo(null)
    setSearchQuery(query)

    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, limit: 15 }),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ error: "Search failed" }))
        throw new Error(errData.error || "Search failed")
      }

      const data = await res.json()
      const hits: SearchHit[] = data.results || []

      setIsLoading(false)

      if (hits.length > 0) {
        setSearchResults(hits)
      } else {
        setNoResults(true)
      }
    } catch (err: any) {
      console.error("Search error:", err)
      setIsLoading(false)
      setSearchError(err.message || "Search failed. Is the backend running?")
    }
  }, [])

  const handleCloseResults = useCallback(() => {
    setSearchResults(null)
    setSearchQuery("")
    setNoResults(false)
    setSearchError(null)
  }, [])

  const handleVideoClick = useCallback((videoId: string, videoName: string) => {
    setSelectedVideo({ id: videoId, name: videoName })
  }, [])

  const handleCloseMap = useCallback(() => {
    setSelectedVideo(null)
  }, [])

  // Escape key to close panels
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (selectedVideo) {
          setSelectedVideo(null)
        } else if (searchResults) {
          setSearchResults(null)
          setSearchQuery("")
        }
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [selectedVideo, searchResults])

  return (
    <div className="flex h-dvh flex-col bg-background selection:bg-white selection:text-black">
      <ProjectBar selectedProject={selectedProject} onProjectChange={setSelectedProject} />

      <main className="flex flex-1 flex-col overflow-hidden max-w-[1400px] mx-auto w-full">
        <QueryBar onSubmit={handleSearch} isLoading={isLoading} />

        {/* No results / error state */}
        {noResults && !searchResults && (
          <div className="mx-4 mb-4 flex items-center gap-3 rounded-lg border border-white/10 bg-white/5 px-4 py-4 md:mx-6">
            <Search className="h-5 w-5 shrink-0 text-white/40" />
            <p className="text-sm font-medium text-white/60">
              No matching footage found for &ldquo;{searchQuery}&rdquo;. Try broader terms or different keywords.
            </p>
            <button onClick={handleCloseResults} className="ml-auto shrink-0 p-1 rounded hover:bg-white/10 transition-colors">
              <X className="h-4 w-4 text-white/40" />
            </button>
          </div>
        )}
        {searchError && !searchResults && (
          <div className="mx-4 mb-4 flex items-center gap-3 rounded-lg border border-red-500/20 bg-red-500/5 px-4 py-4 md:mx-6">
            <AlertCircle className="h-5 w-5 shrink-0 text-red-500" />
            <p className="text-sm font-medium text-red-200">
              {searchError}
            </p>
            <button onClick={handleCloseResults} className="ml-auto shrink-0 p-1 rounded hover:bg-white/10 transition-colors">
              <X className="h-4 w-4 text-white/40" />
            </button>
          </div>
        )}

        {/* Main content area */}
        <div className="flex flex-1 flex-col overflow-hidden px-4 md:px-6">
          <div className="flex flex-1 flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#0a0a0a] shadow-[0_0_50px_-12px_rgba(0,0,0,0.5)]">
            {isLoading ? (
              <div className="relative flex flex-1 items-center justify-center">
                <LoadingOverlay isVisible={true} query={loadingQuery} />
              </div>
            ) : searchResults ? (
              /* Search results view */
              <SearchResults
                results={searchResults}
                query={searchQuery}
                onClose={handleCloseResults}
                onVideoClick={handleVideoClick}
              />
            ) : selectedVideo ? (
              /* Video detail view — Map / 3D Splat toggle */
              <div className="flex flex-1 flex-col overflow-hidden">
                {/* Tab bar */}
                <div className="flex items-center gap-1 px-4 pt-4 pb-2">
                  <button
                    onClick={() => setViewMode("splat")}
                    className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                      viewMode === "splat"
                        ? "bg-white text-black"
                        : "text-white/40 hover:text-white/70"
                    }`}
                  >
                    <Box className="h-3.5 w-3.5" />
                    3D Splat
                  </button>
                  <button
                    onClick={() => setViewMode("map")}
                    className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                      viewMode === "map"
                        ? "bg-white text-black"
                        : "text-white/40 hover:text-white/70"
                    }`}
                  >
                    <Map className="h-3.5 w-3.5" />
                    Map
                  </button>
                </div>

                <div className="flex flex-1 overflow-hidden p-4 pt-2">
                  {viewMode === "map" ? (
                    <MapViewer
                      videoId={selectedVideo.id}
                      videoName={selectedVideo.name}
                      onClose={handleCloseMap}
                    />
                  ) : (
                    <PlyViewer
                      videoId={selectedVideo.id}
                      videoName={selectedVideo.name}
                      onClose={handleCloseMap}
                    />
                  )}
                </div>
              </div>
            ) : (
              /* Video Library -- default view */
              <div className="flex flex-1 flex-col overflow-hidden py-4">
                <VideoLibrary onVideoClick={handleVideoClick} />
              </div>
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
