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
  return (
    <header className="flex flex-col gap-3 border-b border-border bg-card px-4 py-3 md:flex-row md:items-center md:justify-between md:px-6">
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center rounded-lg bg-primary/10 p-2">
          <HardHat className="h-5 w-5 text-primary" />
        </div>
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-bold tracking-tight text-foreground">Splatt</h1>
        </div>
      </div>
    </header>
  )
}
