"use client";

import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { Splat, OrbitControls, Stage } from '@react-three/drei';
import { Loader2 } from 'lucide-react';

interface SplatRendererProps {
    url: string;
}

function SplatScene({ url }: { url: string }) {
    return (
        <Suspense fallback={null}>
            <Stage intensity={0.5} environment="city" adjustCamera={1.5}>
                <Splat src={url} />
            </Stage>
            <OrbitControls makeDefault />
        </Suspense>
    );
}

export function SplatRenderer({ url }: SplatRendererProps) {
    return (
        <div className="w-full h-full bg-slate-950 relative">
            <Canvas dpr={[1, 2]} camera={{ position: [0, 0, 5], fov: 45 }}>
                <color attach="background" args={['#020617']} />
                <SplatScene url={url} />
            </Canvas>
            <div className="absolute top-4 left-4 z-10">
                <div className="px-3 py-1 bg-slate-900/80 backdrop-blur border border-slate-700 rounded-full text-[10px] text-slate-400 font-medium">
                    3D GAUSSIAN SPLAT
                </div>
            </div>
        </div>
    );
}
