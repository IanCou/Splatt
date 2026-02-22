"use client"

import { useState, useCallback, useEffect } from "react"
import { ProjectBar } from "@/components/project-bar"
import { QueryBar } from "@/components/query-bar"
import { SceneViewer } from "@/components/scene-viewer"
import { ResultsPanel } from "@/components/results-panel"
import { LoadingOverlay } from "@/components/loading-overlay"
import { VideoLibrary } from "@/components/video-library"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { AlertCircle, Map as MapIcon, Film } from "lucide-react"
import { hotspots, queryResults } from "@/lib/mock-data"
import type { VideoDescriptor } from "@/lib/types"
import useSWR from "swr"

export default function SplattPage() {
  const [selectedProject, setSelectedProject] = useState("p1")
  const [activeHotspot, setActiveHotspot] = useState<string | null>(null)
  const [highlightedHotspots, setHighlightedHotspots] = useState<string[]>([])
  const [currentResult, setCurrentResult] = useState<(typeof queryResults)[string] | null>(null)
  const [showResults, setShowResults] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [loadingQuery, setLoadingQuery] = useState("")
  const [noResults, setNoResults] = useState(false)
  const [activeTab, setActiveTab] = useState("scene") // FIX #1: Start with scene tab active

  // Store video descriptors: Map<video_id, VideoDescriptor>
  const [videoDescriptors, setVideoDescriptors] = useState<Map<string, VideoDescriptor>>(new Map())

  // Fetch groups and extract descriptors
  const { data: groupsData } = useSWR<{ groups: any[] }>(
    "http://localhost:8000/api/videos/groups",
    (url) => fetch(url).then((r) => r.json())
  )

  // Extract and store descriptors when groups data changes
  useEffect(() => {
    if (groupsData?.groups) {
      const descriptors = new Map<string, VideoDescriptor>()
      let totalDescriptors = 0

      for (const group of groupsData.groups) {
        for (const video of group.videos || []) {
          if (video.descriptor) {
            descriptors.set(video.id, video.descriptor)
            totalDescriptors++
          }
        }
      }

      setVideoDescriptors(descriptors)
      console.log('VIDEO_DESCRIPTORS_LOADED:', {
        totalDescriptors,
        videoIds: Array.from(descriptors.keys()),
        sampleDescriptor: totalDescriptors > 0 ? descriptors.values().next().value : null
      })
    }
  }, [groupsData])

  console.log('PAGE_STATE:', {
    activeTab,
    hotspotsAvailable: hotspots.length,
    activeHotspot,
    highlightedHotspotsCount: highlightedHotspots.length,
    showResults,
    isLoading
  })

  const handleSearch = useCallback(async (query: string) => {
    console.log('SCENE_QUERY_INITIATED:', {
      query,
      timestamp: new Date().toISOString(),
      descriptorsInState: videoDescriptors.size
    })

    setIsLoading(true)
    setLoadingQuery(query)
    setNoResults(false)
    setShowResults(false)
    setActiveHotspot(null)

    const startTime = performance.now()

    try {
      // Gather relevant descriptors (all descriptors, per Option B - send all for current group/all if no group)
      const descriptorsArray = Array.from(videoDescriptors.values())
      console.log('SCENE_QUERY_DESCRIPTORS:', {
        descriptorCount: descriptorsArray.length,
        videoIds: Array.from(videoDescriptors.keys()),
        hasDescriptors: descriptorsArray.length > 0
      })

      // Warn if no descriptors available
      if (descriptorsArray.length === 0) {
        console.warn('SCENE_QUERY_NO_DESCRIPTORS:', {
          message: 'No video descriptors found in state - query will use basic context only',
          videoDescriptorsSize: videoDescriptors.size
        })
      }

      // Build request body
      const requestBody = {
        query,
        group_id: null, // Could be set if we add group filtering UI
        descriptors: descriptorsArray
      }

      console.log('SCENE_QUERY_SENDING:', {
        url: 'http://localhost:8000/api/query',
        method: 'POST',
        query,
        descriptorCount: descriptorsArray.length
      })

      // Log full request payload for debugging
      console.log('SCENE_QUERY_REQUEST_BODY:', {
        query: requestBody.query,
        group_id: requestBody.group_id,
        descriptors_count: requestBody.descriptors?.length || 0,
        descriptors_sample: requestBody.descriptors?.[0] ? {
          video_id: requestBody.descriptors[0].video_id,
          filename: requestBody.descriptors[0].filename,
          scene_type: requestBody.descriptors[0].scene_type,
          people_count: requestBody.descriptors[0].people?.length || 0,
          zones_count: requestBody.descriptors[0].spatial_zones?.length || 0,
          snapshots_count: requestBody.descriptors[0].frame_snapshots?.length || 0
        } : null
      })

      // Check if descriptors can be stringified (JSON serialization test)
      try {
        const testJson = JSON.stringify(requestBody)
        console.log('SCENE_QUERY_JSON_SIZE:', {
          bytes: testJson.length,
          kilobytes: (testJson.length / 1024).toFixed(2) + 'KB'
        })
      } catch (jsonErr) {
        console.error('SCENE_QUERY_JSON_SERIALIZATION_ERROR:', jsonErr)
      }

      const res = await fetch("http://localhost:8000/api/query", {
        method: "POST",
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody),
      })

      const endTime = performance.now()
      console.log('SCENE_QUERY_RESPONSE_RECEIVED:', {
        status: res.status,
        statusText: res.statusText,
        responseTime: `${(endTime - startTime).toFixed(2)}ms`
      })

      if (!res.ok) {
        // Try to parse error as JSON first (FastAPI returns JSON errors)
        let errorDetails: any = null
        let errorText = ''

        try {
          const errorBody = await res.text()
          errorText = errorBody

          // Try parsing as JSON
          try {
            errorDetails = JSON.parse(errorBody)
            console.error('SCENE_QUERY_ERROR_JSON:', {
              status: res.status,
              statusText: res.statusText,
              errorDetails,
              detail: errorDetails.detail // FastAPI puts error details here
            })
          } catch {
            // Not JSON, just log as text
            console.error('SCENE_QUERY_ERROR_TEXT:', {
              status: res.status,
              statusText: res.statusText,
              errorText: errorText.substring(0, 500) // First 500 chars
            })
          }
        } catch (readErr) {
          console.error('SCENE_QUERY_ERROR_UNABLE_TO_READ:', {
            status: res.status,
            statusText: res.statusText,
            readError: readErr
          })
        }

        // Log what we were trying to send when error occurred
        console.error('SCENE_QUERY_ERROR_CONTEXT:', {
          query,
          descriptorsSent: descriptorsArray.length,
          requestBodyKeys: Object.keys(requestBody)
        })

        throw new Error(`Query failed: ${res.status} ${res.statusText}`)
      }

      const data = await res.json()
      console.log('SCENE_QUERY_DATA_PARSED:', {
        hasHotspots: !!data.hotspots,
        hotspotsCount: data.hotspots?.length || 0,
        hasAnalysis: !!data.analysis,
        analysisLength: data.analysis?.length || 0,
        confidence: data.confidence,
        location: data.location,
        coordinates: data.coordinates,
        worker: data.worker,
        workerRole: data.workerRole
      })

      setIsLoading(false)

      if (data.hotspots && data.hotspots.length > 0) {
        console.log('SCENE_QUERY_HOTSPOTS_FOUND:', { hotspots: data.hotspots, firstMatch: data.hotspots[0] })
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
        console.log('SCENE_QUERY_UI_UPDATED:', { activeTab: 'scene', showResults: true })
      } else {
        // Fallback for analysis with no specific hotspots
        if (data.analysis) {
          console.log('SCENE_QUERY_NO_HOTSPOTS_BUT_HAS_ANALYSIS:', { analysisPreview: data.analysis.substring(0, 100) })
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
          console.log('SCENE_QUERY_UI_UPDATED:', { activeTab: 'scene', showResults: true, mode: 'scene-wide' })
        } else {
          console.warn('SCENE_QUERY_NO_RESULTS:', { data })
          setHighlightedHotspots([])
          setNoResults(true)
        }
      }

      console.log('SCENE_QUERY_COMPLETED:', {
        totalTime: `${(performance.now() - startTime).toFixed(2)}ms`,
        success: true
      })
    } catch (err) {
      const endTime = performance.now()
      console.error('SCENE_QUERY_FAILED:', {
        error: err,
        message: err instanceof Error ? err.message : 'Unknown error',
        stack: err instanceof Error ? err.stack : undefined,
        totalTime: `${(endTime - startTime).toFixed(2)}ms`,
        query,
        descriptorCount: descriptorsArray.length
      })

      // Additional debugging: check if it's a network error vs backend error
      if (err instanceof TypeError) {
        console.error('SCENE_QUERY_NETWORK_ERROR:', {
          message: 'Possible network/CORS issue or backend not running',
          error: err.message
        })
      } else if (err instanceof Error && err.message.includes('Query failed')) {
        console.error('SCENE_QUERY_BACKEND_ERROR:', {
          message: 'Backend returned an error - check logs above for details'
        })
      }

      setIsLoading(false)
      setNoResults(true)
    }
  }, [videoDescriptors])

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
    <div className="flex h-dvh flex-col bg-background selection:bg-white selection:text-black">
      <ProjectBar selectedProject={selectedProject} onProjectChange={setSelectedProject} />

      <main className="flex flex-1 flex-col overflow-hidden max-w-[1400px] mx-auto w-full">
        <QueryBar onSubmit={handleSearch} isLoading={isLoading} />

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
                      <MapIcon className="h-3.5 w-3.5" />
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
