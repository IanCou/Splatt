"use client";

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

interface SplatRendererProps {
    url: string;
}

const vertexShader = `
  precision highp float;
  uniform mat4 modelViewMatrix;
  uniform mat4 projectionMatrix;
  
  attribute vec3 position;
  attribute vec3 color;
  attribute vec3 scale;
  
  varying vec3 vColor;

  void main() {
    vColor = color;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    
    // Scale the point size based on average scale and distance
    float avgScale = (scale.x + scale.y + scale.z) / 3.0;
    gl_PointSize = avgScale * 1000.0 / length(mvPosition.xyz);
    
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = `
  precision highp float;
  varying vec3 vColor;

  void main() {
    // Gaussian falloff logic provided by the user
    float dist = length(gl_PointCoord - vec2(0.5));
    float alpha = exp(-dist * dist * 8.0);
    
    // Discard edges to maintain soft circle shape
    if (alpha < 0.05) discard;
    
    gl_FragColor = vec4(vColor, alpha);
  }
`;

function GaussianPoints({ url }: { url: string }) {
    const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);

    useEffect(() => {
        async function loadSplat() {
            const response = await fetch(url);
            const arrayBuffer = await response.arrayBuffer();
            const numGaussians = arrayBuffer.byteLength / 32;

            const positions = new Float32Array(numGaussians * 3);
            const scales = new Float32Array(numGaussians * 3);
            const colors = new Float32Array(numGaussians * 3);

            const view = new DataView(arrayBuffer);

            for (let i = 0; i < numGaussians; i++) {
                const offset = i * 32;

                // Position
                positions[i * 3] = view.getFloat32(offset + 0, true);
                positions[i * 3 + 1] = view.getFloat32(offset + 4, true);
                positions[i * 3 + 2] = view.getFloat32(offset + 8, true);

                // Scale
                scales[i * 3] = view.getFloat32(offset + 12, true);
                scales[i * 3 + 1] = view.getFloat32(offset + 16, true);
                scales[i * 3 + 2] = view.getFloat32(offset + 20, true);

                // Color (RGBA uint8 -> float)
                colors[i * 3] = view.getUint8(offset + 24) / 255;
                colors[i * 3 + 1] = view.getUint8(offset + 25) / 255;
                colors[i * 3 + 2] = view.getUint8(offset + 26) / 255;
            }

            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geo.setAttribute('scale', new THREE.BufferAttribute(scales, 3));
            geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

            setGeometry(geo);
        }

        loadSplat();
    }, [url]);

    if (!geometry) return null;

    return (
        <points geometry={geometry}>
            <rawShaderMaterial
                vertexShader={vertexShader}
                fragmentShader={fragmentShader}
                transparent={true}
                depthWrite={false}
                blending={THREE.NormalBlending}
            />
        </points>
    );
}

export function SplatRenderer({ url }: SplatRendererProps) {
    return (
        <div className="w-full h-full bg-gray-300 relative">
            <Canvas dpr={[1, 2]}>
                <PerspectiveCamera makeDefault position={[0, 0, 5]} fov={45} />
                <color attach="background" args={['#d1d5db']} />
                <ambientLight intensity={0.5} />
                <GaussianPoints url={url} />
                <OrbitControls makeDefault />
            </Canvas>
            <div className="absolute top-4 left-4 z-10">
                <div className="px-3 py-1 bg-slate-900/80 backdrop-blur border border-slate-700 rounded-full text-[10px] text-blue-400 font-bold uppercase tracking-widest">
                    Custom Gaussian Shader
                </div>
            </div>
        </div>
    );
}
