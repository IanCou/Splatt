"use client";

import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Splat, PerspectiveCamera } from '@react-three/drei';

interface SplatRendererProps {
    url: string;
}

const SplatScene = ({ url }: SplatRendererProps) => {
    // Debug: Check the URL content type and size before Splat loads it
    React.useEffect(() => {
        fetch(url, { method: 'HEAD' })
            .then(res => {
                console.log(`[SplatLoader] Fetching: ${url}`);
                console.log(`[SplatLoader] Status: ${res.status}`);
                console.log(`[SplatLoader] Type: ${res.headers.get('content-type')}`);
                console.log(`[SplatLoader] Size: ${res.headers.get('content-length')} bytes`);
            })
            .catch(err => console.error("[SplatLoader] HEAD request failed:", err));
    }, [url]);

    return (
        <>
            <PerspectiveCamera makeDefault position={[0, 0, 5]} />
            <OrbitControls makeDefault />
            <Splat
                src={url}
                position={[0, 0, 0]}
            />
        </>
    );
};

export function SplatRenderer({ url }: SplatRendererProps) {
    return (
        <div className="w-full h-full bg-slate-950 rounded-lg overflow-hidden relative border border-slate-800">
            <Canvas shadows dpr={[1, 2]}>
                <Suspense fallback={null}>
                    <SplatScene url={url} />
                </Suspense>
            </Canvas>
            <div className="absolute bottom-4 left-4 text-xs text-slate-400 bg-slate-900/80 px-2 py-1 rounded">
                Double click to focus • Click and drag to rotate • Scroll to zoom
            </div>
        </div>
    );
}
