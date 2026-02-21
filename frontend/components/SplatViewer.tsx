import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

interface SplatViewerProps {
    splatUrl: string;
    targetCoords?: { x: number; y: number; z: number };
    opacity?: number;
}

export default function SplatViewer({ splatUrl, targetCoords, opacity = 1.0 }: SplatViewerProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const controlsRef = useRef<OrbitControls | null>(null);

    useEffect(() => {
        if (!containerRef.current) return;

        // 1. Setup Three.js Scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a1a);
        sceneRef.current = scene;

        const camera = new THREE.PerspectiveCamera(75, containerRef.current.clientWidth / containerRef.current.clientHeight, 0.1, 1000);
        camera.position.set(2, 2, 2);
        cameraRef.current = camera;

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        containerRef.current.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controlsRef.current = controls;

        // Add lighting & helpers
        const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
        scene.add(gridHelper);

        const axesHelper = new THREE.AxesHelper(1);
        scene.add(axesHelper);

        // Mock Splat Representation (since gSplat.js requires specific loaders)
        // For scaffolding, we use a point cloud or box to represent the scene limits
        const geometry = new THREE.BoxGeometry(2, 2, 2);
        const material = new THREE.MeshBasicMaterial({ color: 0x4f46e5, wireframe: true, transparent: true, opacity });
        const mockSplat = new THREE.Mesh(geometry, material);
        scene.add(mockSplat);

        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        const handleResize = () => {
            if (!containerRef.current || !camera || !renderer) return;
            camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
        };
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            renderer.dispose();
            if (containerRef.current && renderer.domElement) {
                containerRef.current.removeChild(renderer.domElement);
            }
        };
    }, []);

    // Update Opacity (for Time Slider)
    useEffect(() => {
        if (sceneRef.current) {
            sceneRef.current.traverse((child) => {
                if (child instanceof THREE.Mesh) {
                    child.material.opacity = opacity;
                }
            });
        }
    }, [opacity]);

    // Animate Camera to Target
    useEffect(() => {
        if (targetCoords && cameraRef.current && controlsRef.current) {
            const { x, y, z } = targetCoords;
            // In a real app, use GSAP or Framer Motion for interpolation
            // Mocking manual set for scaffolding
            cameraRef.current.position.set(x + 2, y + 2, z + 2);
            controlsRef.current.target.set(x, y, z);
        }
    }, [targetCoords]);

    return <div ref={containerRef} className="w-full h-full min-h-[500px] rounded-xl overflow-hidden shadow-2xl relative" />;
}
