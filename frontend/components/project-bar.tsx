"use client"

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { HardHat, Video, Clock, CheckCircle } from "lucide-react"
import { projects } from "@/lib/mock-data"

interface ProjectBarProps {
  selectedProject: string
  onProjectChange: (id: string) => void
}

export function ProjectBar({ selectedProject, onProjectChange }: ProjectBarProps) {
  const project = projects.find((p) => p.id === selectedProject) ?? projects[0]

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
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {projects.map((p) => (
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
          <span>{project.footage} helmet cam videos</span>
          <span className="text-border">|</span>
          <Clock className="h-3.5 w-3.5" />
          <span>{project.hours} hours total</span>
        </div>
        <Badge variant="outline" className="gap-1.5 border-primary/30 bg-primary/5 text-primary">
          <CheckCircle className="h-3 w-3" />
          Scene graph up to date
        </Badge>
      </div>
    </header>
  )
}
