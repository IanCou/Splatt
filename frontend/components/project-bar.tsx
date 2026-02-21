"use client"

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { HardHat, Video, Clock, CheckCircle } from "lucide-react"
import useSWR from "swr"

interface ProjectBarProps {
  selectedProject: string
  onProjectChange: (id: string) => void
}

const fetcher = (url: string) => fetch(url).then((r) => r.json())

export function ProjectBar({ selectedProject, onProjectChange }: ProjectBarProps) {
  const { data } = useSWR<{ groups: { id: string; name: string; videoCount: number; totalDuration: number }[] }>(
    "http://localhost:8000/api/videos/groups",
    fetcher,
    { refreshInterval: 5000 }
  )

  const groups = data?.groups ?? []
  const totalVideos = groups.reduce((sum, g) => sum + g.videoCount, 0)
  const totalHours = groups.reduce((sum, g) => sum + g.totalDuration, 0)
  const totalHoursDisplay = (totalHours / 3600).toFixed(1)

  const projectOptions = groups.length > 0
    ? groups.map((g) => ({ id: g.id, name: g.name }))
    : [{ id: "none", name: "No projects yet – upload a video" }]

  return (
    <header className="flex flex-col gap-3 border-b border-border bg-card px-4 py-3 md:flex-row md:items-center md:justify-between md:px-6">
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center rounded-lg bg-primary/10 p-2">
          <HardHat className="h-5 w-5 text-primary" />
        </div>
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-bold tracking-tight text-foreground">Splatt</h1>
          <span className="hidden text-muted-foreground md:inline">|</span>
          <Select value={selectedProject} onValueChange={onProjectChange}>
            <SelectTrigger className="h-8 w-auto min-w-[200px] border-border bg-secondary text-secondary-foreground">
              <SelectValue placeholder="Select a project…" />
            </SelectTrigger>
            <SelectContent>
              {projectOptions.map((p) => (
                <SelectItem key={p.id} value={p.id}>
                  {p.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3 text-sm">
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <Video className="h-3.5 w-3.5" />
          <span>{totalVideos} helmet cam video{totalVideos !== 1 ? "s" : ""}</span>
          <span className="text-border">|</span>
          <Clock className="h-3.5 w-3.5" />
          <span>{totalHoursDisplay}h total</span>
        </div>
        <Badge variant="outline" className="gap-1.5 border-primary/30 bg-primary/5 text-primary">
          <CheckCircle className="h-3 w-3" />
          {groups.length > 0 ? "Scene graph up to date" : "No footage loaded"}
        </Badge>
      </div>
    </header>
  )
}
