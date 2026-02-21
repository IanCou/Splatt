"use client";

import { useState } from 'react';
import SplatViewer from '@/components/SplatViewer';
import SemanticSearchBar from '@/components/SemanticSearchBar';
import TemporalSlider from '@/components/TemporalSlider';
import ProgressPanel from '@/components/ProgressPanel';

// Mock Logs
const initialLogs = [
  { id: '1', message: 'Excavator moved to Sector B', type: 'info' as const, time: '2 mins ago' },
  { id: '2', message: 'Drywall installation completed on Floor 2', type: 'success' as const, time: '1 hr ago' },
  { id: '3', message: 'Unauthorized equipment in loading bay', type: 'alert' as const, time: '3 hrs ago' },
];

export default function Home() {
  const [opacity, setOpacity] = useState(1.0);
  const [isSearching, setIsSearching] = useState(false);
  const [targetCoords, setTargetCoords] = useState<{ x: number, y: number, z: number } | undefined>();
  const [searchResult, setSearchResult] = useState<{ analysis: string, confidence: string, location: string } | null>(null);

  const handleSearch = async (query: string) => {
    setIsSearching(true);
    setSearchResult(null);
    try {
      // Mock API call based on Scaffold
      const formData = new FormData();
      formData.append('query', query);

      const res = await fetch('http://localhost:8000/api/search', {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setSearchResult({
          analysis: data.analysis,
          confidence: data.confidence,
          location: data.location || 'Unknown',
        });

        // Mock coordinates for camera animation
        if (data.hotspots && data.hotspots.length > 0) {
          // just assigning a random coordinate jump for demo purposes since actual XYZ might not be parsed
          setTargetCoords({ x: (Math.random() - 0.5) * 5, y: Math.random() * 2, z: (Math.random() - 0.5) * 5 });
        }
      }
    } catch (error) {
      console.error(error);
      setSearchResult({
        analysis: "Simulated response: The excavator is located near the south wall.",
        confidence: "High",
        location: "South Wall"
      });
      setTargetCoords({ x: 2, y: 0.5, z: -1 });
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <main className="relative w-full h-screen bg-zinc-950 overflow-hidden font-sans">
      {/* 3D Viewport - The Source of Truth */}
      <div className="absolute inset-0">
        <SplatViewer
          splatUrl="/models/mock.splat"
          opacity={opacity}
          targetCoords={targetCoords}
        />
      </div>

      {/* Semantic Search UI */}
      <SemanticSearchBar onSearch={handleSearch} isSearching={isSearching} />

      {/* Temporal Slider for Snapshot Blending */}
      <TemporalSlider opacity={opacity} onChange={setOpacity} />

      {/* Change Log & Progress Panel */}
      <ProgressPanel logs={initialLogs} searchResult={searchResult} />
    </main>
  );
}
