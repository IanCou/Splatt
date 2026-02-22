"use client"

import { Canvas, useThree } from "@react-three/fiber"
import { OrbitControls, Html } from "@react-three/drei"
import * as THREE from "three"
import { PLYLoader } from "three/addons/loaders/PLYLoader.js"
import { useEffect, useState, useRef } from "react"
import { Loader2, X } from "lucide-react"

/* ------------------------------------------------------------------ */
/*  Inner scene component – renders the point cloud from a PLY URL    */
/* ------------------------------------------------------------------ */
function PointCloud({
    url,
    onLoad,
    onError,
    onProgress,
}: {
    url: string
    onLoad?: () => void
    onError?: (msg: string) => void
    onProgress?: (pct: number) => void
}) {
    const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null)
    const pointsRef = useRef<THREE.Points>(null)
    const { camera } = useThree()

    useEffect(() => {
        let disposed = false
        const loader = new PLYLoader()

        loader.load(
            url,
            (geo) => {
                if (disposed) {
                    geo.dispose()
                    return
                }

                // Centre the cloud at the origin
                geo.computeBoundingBox()
                geo.center()

                // If the PLY loader didn't produce a color attribute,
                // fall back to white so vertexColors still works.
                if (!geo.attributes.color) {
                    const count = geo.attributes.position.count
                    const colors = new Float32Array(count * 3)
                    colors.fill(1) // white
                    geo.setAttribute(
                        "color",
                        new THREE.BufferAttribute(colors, 3)
                    )
                }

                // Auto-fit camera
                const box = geo.boundingBox!
                const size = box.getSize(new THREE.Vector3())
                const maxDim = Math.max(size.x, size.y, size.z)
                const dist = maxDim * 1.5
                camera.position.set(dist * 0.4, dist * 0.25, dist)
                camera.lookAt(0, 0, 0)
                camera.updateProjectionMatrix()

                setGeometry(geo)
                onLoad?.()
            },
            (xhr) => {
                if (xhr.total > 0) {
                    onProgress?.(Math.round((xhr.loaded / xhr.total) * 100))
                }
            },
            () => {
                if (!disposed) onError?.("Failed to load PLY file")
            }
        )

        return () => {
            disposed = true
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [url])

    // Clean up geometry on unmount
    useEffect(() => {
        return () => {
            geometry?.dispose()
        }
    }, [geometry])

    if (!geometry) return null

    return (
        <points ref={pointsRef} geometry={geometry}>
            <pointsMaterial
                size={0.006}
                vertexColors
                sizeAttenuation
                transparent
                opacity={0.9}
                depthWrite={false}
            />
        </points>
    )
}

/* ------------------------------------------------------------------ */
/*  Public component                                                   */
/* ------------------------------------------------------------------ */
export function PlyViewer({
    videoId,
    videoName,
    onClose,
}: {
    videoId: string
    videoName?: string
    onClose?: () => void
}) {
    const [loading, setLoading] = useState(true)
    const [progress, setProgress] = useState(0)
    const [error, setError] = useState<string | null>(null)

    const url = `/api/results/${videoId}/baked_splat.ply`

    return (
        <div className="relative w-full h-full min-h-[400px] bg-black rounded-xl overflow-hidden">
            {/* ---- Header overlay ---- */}
            <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-4 py-3 bg-gradient-to-b from-black/80 via-black/40 to-transparent pointer-events-none">
                <div>
                    <h3 className="text-sm font-semibold text-white tracking-wide">
                        {videoName || "Gaussian Splat"}
                    </h3>
                    <p className="text-[10px] text-white/40 uppercase tracking-widest mt-0.5">
                        3D Neural Reconstruction
                    </p>
                </div>
                {onClose && (
                    <button
                        onClick={onClose}
                        className="pointer-events-auto p-1.5 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
                    >
                        <X className="h-4 w-4 text-white" />
                    </button>
                )}
            </div>

            {/* ---- Loading overlay ---- */}
            {loading && !error && (
                <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black">
                    <Loader2 className="h-8 w-8 animate-spin text-white/60 mb-3" />
                    <p className="text-xs text-white/40 uppercase tracking-widest">
                        Loading Point Cloud
                    </p>
                    {progress > 0 && (
                        <div className="mt-3 w-48">
                            <div className="h-1 rounded-full bg-white/10 overflow-hidden">
                                <div
                                    className="h-full bg-white/60 rounded-full transition-all duration-300"
                                    style={{ width: `${progress}%` }}
                                />
                            </div>
                            <p className="text-[10px] text-white/30 text-center mt-1.5">
                                {progress}%
                            </p>
                        </div>
                    )}
                </div>
            )}

            {/* ---- Error state ---- */}
            {error && (
                <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black">
                    <p className="text-sm text-red-400">{error}</p>
                    <p className="text-xs text-white/30 mt-1">
                        The 3D reconstruction may not be available yet.
                    </p>
                </div>
            )}

            {/* ---- Three.js Canvas ---- */}
            <Canvas
                camera={{
                    position: [0, 0, 5],
                    fov: 50,
                    near: 0.001,
                    far: 1000,
                }}
                gl={{ antialias: false, alpha: false }}
                dpr={[1, 2]}
                style={{ background: "#000000" }}
            >
                <PointCloud
                    url={url}
                    onLoad={() => setLoading(false)}
                    onError={(msg) => {
                        setError(msg)
                        setLoading(false)
                    }}
                    onProgress={setProgress}
                />
                <OrbitControls
                    enableDamping
                    dampingFactor={0.12}
                    rotateSpeed={0.5}
                    zoomSpeed={0.8}
                    panSpeed={0.5}
                    minDistance={0.1}
                    maxDistance={200}
                />
            </Canvas>

            {/* ---- Controls hint ---- */}
            {!loading && !error && (
                <div className="absolute bottom-3 left-4 text-[10px] text-white/20 uppercase tracking-widest">
                    Drag to orbit · Scroll to zoom · Right-click to pan
                </div>
            )}
        </div>
    )
}
