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
    <header className="flex flex-col gap-4 border-b border-white/10 bg-black px-6 py-4 md:flex-row md:items-center md:justify-between">
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3 group cursor-pointer">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-black transition-transform group-hover:scale-105">
            <HardHat className="h-6 w-6" />
          </div>
          <h1 className="text-xl font-bold tracking-tighter text-white">SPLATT</h1>
        </div>

        <div className="h-6 w-px bg-white/10 hidden md:block" />

        <Select value={selectedProject} onValueChange={onProjectChange}>
          <SelectTrigger className="h-9 w-auto min-w-[240px] border-white/10 bg-white/5 text-[13px] font-medium text-white hover:bg-white/10 transition-colors">
            <div className="flex items-center gap-2">
              <span className="text-white/40 font-bold uppercase tracking-wider text-[10px]">Project:</span>
              <SelectValue />
            </div>
          </SelectTrigger>
          <SelectContent className="bg-[#0a0a0a] border-white/10 text-white">
            {projects.map((p) => (
              <SelectItem key={p.id} value={p.id} className="focus:bg-white focus:text-black">
                {p.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex flex-wrap items-center gap-6 text-[11px] font-bold uppercase tracking-widest text-white/40">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 group cursor-default">
            <Video className="h-3.5 w-3.5 group-hover:text-blue-500 transition-colors" />
            <span className="text-white group-hover:text-white transition-colors">{project.footage} Cam Feeds</span>
          </div>
          <div className="w-1 h-1 rounded-full bg-white/20" />
          <div className="flex items-center gap-2 group cursor-default">
            <Clock className="h-3.5 w-3.5 group-hover:text-amber-500 transition-colors" />
            <span className="text-white group-hover:text-white transition-colors">{project.hours}H Total</span>
          </div>
        </div>

        <Badge className="h-6 gap-2 rounded-full border-white/10 bg-green-500/10 px-3 text-[10px] font-bold text-green-400">
          <CheckCircle className="h-3 w-3" />
          Neural Index Active
        </Badge>
      </div>
    </header>
  )
}
