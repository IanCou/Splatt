"use client";

import React, { useState } from 'react';
import { SplatRenderer } from '@/components/splat-renderer';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Loader2 } from 'lucide-react';

export default function SplatViewPage() {
    const [ptPath, setPtPath] = useState('');
    const [currentUrl, setCurrentUrl] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleLoad = () => {
        if (!ptPath) return;
        setIsLoading(true);

        // Ensure path ends with .splat for the browser to help the loader
        // though the backend will treat it as .pt
        let displayPath = ptPath;
        if (displayPath.endsWith('.pt')) {
            displayPath = displayPath.replace('.pt', '.splat');
        } else if (!displayPath.endsWith('.splat')) {
            displayPath = displayPath + '.splat';
        }

        const backendUrl = `http://localhost:8000/load-splat/${encodeURIComponent(displayPath)}`;
        setCurrentUrl(backendUrl);

        setTimeout(() => setIsLoading(false), 500);
    };

    return (
        <div className="container mx-auto p-8 max-w-6xl space-y-8">
            <div className="flex flex-col space-y-2">
                <h1 className="text-4xl font-bold tracking-tight">Gaussian Splat Viewer</h1>
                <p className="text-muted-foreground">
                    Enter the absolute path to your <code>.pt</code> model file to render it in 3D.
                </p>
            </div>

            <div className="grid gap-8 md:grid-cols-[350px_1fr]">
                <div className="space-y-6">
                    <Card>
                        <CardHeader>
                            <CardTitle>Configuration</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Model Path (.pt)</label>
                                <Input
                                    placeholder="C:\path\to\model.pt"
                                    value={ptPath}
                                    onChange={(e) => setPtPath(e.target.value)}
                                />
                            </div>
                            <Button
                                onClick={handleLoad}
                                className="w-full"
                                disabled={!ptPath || isLoading}
                            >
                                {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                                Load 3D Scene
                            </Button>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle className="text-sm">Information</CardTitle>
                        </CardHeader>
                        <CardContent className="text-sm text-muted-foreground space-y-2">
                            <p>
                                The backend will load the PyTorch state dictionary, extract Gaussian parameters,
                                and stream them in the standard binary <code>.splat</code> format.
                            </p>
                            <div className="space-y-2">
                                Components used:
                                <ul className="list-disc list-inside mt-1">
                                    <li>FastAPI (Conversion)</li>
                                    <li>React Three Fiber</li>
                                    <li>@react-three/drei</li>
                                </ul>
                            </div>
                        </CardContent>
                    </Card>
                </div>

                <div className="aspect-video w-full">
                    {currentUrl ? (
                        <SplatRenderer url={currentUrl} />
                    ) : (
                        <div className="w-full h-full border-2 border-dashed border-slate-800 rounded-lg flex items-center justify-center bg-slate-900/50">
                            <div className="text-center space-y-2">
                                <p className="text-slate-500 font-medium">No model loaded</p>
                                <p className="text-xs text-slate-600">Enter a path on the left to begin</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
