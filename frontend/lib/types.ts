// Video Descriptor Types
export interface PersonAppearance {
  person_id: string | null
  role_inferred: string | null
  description: string
  activities: string[]
  interactions: string[]
  location_description: string
  timestamp_range: { start: number; end: number }
  notable_behaviors: string[]
  objects_used: string[]
  confidence: string
}

export interface SpatialZone {
  zone_name: string
  zone_description: string
  zone_type: string
  importance: string
  visible_timeranges: Array<{ start: number; end: number }>
  people_in_zone: string[]
  objects_in_zone: string[]
  activities_in_zone: string[]
  accessibility: string
  state_changes: string[]
}

export interface SceneCharacteristics {
  setting_type: string
  location_description: string
  space_characteristics: string
  lighting: {
    type: string
    quality: string
    changes: string | null
  }
  visibility: string
  camera: {
    quality: string
    view_angle: string
  }
  activity_level: string
  noise_level_inferred: string | null
  overall_mood: string
  weather: string | null
  temperature_inferred: string | null
  time_of_day: string | null
  season_inferred: string | null
  distinctive_features: string[]
  background_elements: string[]
  environmental_impact: string | null
}

export interface FrameSnapshot {
  timestamp: number
  scene_summary: string
  objects: {
    furniture: string[]
    equipment: string[]
    materials: string[]
    personal_items: string[]
    decorative_items: string[]
    other_objects: string[]
    object_states: Record<string, string>
    object_locations: Record<string, string>
    recently_moved: string[]
    in_use: string[]
    newly_appeared: string[]
    disappeared: string[]
    organization_level: string
    focal_objects: string[]
  }
  people: {
    visible: string[]
    descriptions: string[]
    locations: string[]
    primary_activity_by_person: Record<string, string>
  }
  zones: {
    visible: string[]
    descriptions: string[]
    primary_zone: string | null
  }
  scene_state: {
    lighting: string
    visibility: string
    activity_level: string
    camera: string
  }
}

export interface ObjectTimeline {
  object_name: string
  object_description: string
  first_appearance: number
  last_appearance: number
  total_visible_duration: number
  locations: Array<{ timestamp: number; location: string }>
  movement_pattern: string | null
  interactions: Array<{ timestamp: number; person: string; action: string }>
  state_changes: Array<{ timestamp: number; state: string }>
}

export interface VideoDescriptor {
  video_id: string
  filename: string
  duration_seconds: number
  upload_timestamp: string
  analysis_timestamp: string
  gemini_model_used: string
  overall_summary: string
  scene_type: string
  primary_activities: string[]
  key_locations: string[]
  people: PersonAppearance[]
  num_people_estimate: number | null
  spatial_zones: SpatialZone[]
  spatial_organization: string | null
  scene_characteristics: SceneCharacteristics | null
  frame_snapshots: FrameSnapshot[]
  object_timelines: ObjectTimeline[]
  tags: string[]
  notable_moments: Array<{ timestamp: number; description: string }>
  anomalies_detected: string[]
}

export interface Video {
  id: string
  filename: string
  originalName: string
  duration: number // seconds
  size: number // bytes
  uploadedAt: string
  status: "processing" | "ready" | "error"
  thumbnailUrl: string | null
  groupId: string | null
  descriptor: VideoDescriptor | null // NEW: Full video descriptor
}

export interface VideoGroup {
  id: string
  name: string
  description: string
  videoCount: number
  totalDuration: number // seconds
  createdAt: string
  videos: Video[]
}

export interface UploadProgress {
  id: string
  filename: string
  progress: number // 0-100
  status: "uploading" | "processing" | "complete" | "error"
  error?: string
}
