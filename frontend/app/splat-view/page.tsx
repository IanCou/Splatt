"use client";

import React, { useState } from 'react';
import { SplatRenderer } from '@/components/splat-renderer';
import { GaussianInspector } from '@/components/gaussian-inspector';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Loader2, Box, Table as TableIcon } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function SplatViewPage() {
    const [ptPath, setPtPath] = useState('');
    const [currentUrl, setCurrentUrl] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [viewMode, setViewMode] = useState<'3d' | 'data'>('3d');

    const handleLoad = () => {
        if (!ptPath) return;
        setIsLoading(true);

        // Support .ply, .pt, or .splat
        let displayPath = ptPath;
        const lowerPath = displayPath.toLowerCase();

        if (lowerPath.endsWith('.pt')) {
            displayPath = displayPath.replace(/\.pt$/i, '.splat');
        } else if (lowerPath.endsWith('.ply')) {
            displayPath = displayPath.replace(/\.ply$/i, '.splat');
        } else if (!lowerPath.endsWith('.splat')) {
            displayPath = displayPath + '.splat';
        }

        const backendUrl = `http://localhost:8000/load-splat/${encodeURIComponent(displayPath)}`;
        setCurrentUrl(backendUrl);

        setTimeout(() => setIsLoading(false), 500);
    };

    return (
        <div className="container mx-auto p-8 max-w-6xl space-y-8">
            <div className="flex flex-col space-y-2">
                <h1 className="text-4xl font-bold tracking-tight text-white">Gaussian Splat Viewer</h1>
                <p className="text-slate-400">
                    Enter the absolute path to your <code>.pt</code> or <code>.ply</code> Gaussian model.
                </p>
            </div>

            <div className="grid gap-8 lg:grid-cols-[380px_1fr]">
                <div className="space-y-6">
                    <Card className="bg-slate-900 border-slate-800">
                        <CardHeader>
                            <CardTitle className="text-white">Configuration</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-300">Model File Path</label>
                                <Input
                                    placeholder="C:\path\to\model.ply"
                                    value={ptPath}
                                    onChange={(e) => setPtPath(e.target.value)}
                                    className="bg-slate-950 border-slate-800 text-white"
                                />
                            </div>
                            <Button
                                onClick={handleLoad}
                                className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                                disabled={!ptPath || isLoading}
                            >
                                {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                                Load Scene
                            </Button>
                        </CardContent>
                    </Card>

                    <Card className="bg-slate-900 border-slate-800">
                        <CardHeader>
                            <CardTitle className="text-sm text-white">Metadata</CardTitle>
                        </CardHeader>
                        <CardContent className="text-sm text-slate-400 space-y-4">
                            <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as any)} className="w-full">
                                <TabsList className="grid w-full grid-cols-2 bg-slate-950">
                                    <TabsTrigger value="3d" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
                                        <Box className="mr-2 h-4 w-4" /> 3D View
                                    </TabsTrigger>
                                    <TabsTrigger value="data" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
                                        <TableIcon className="mr-2 h-4 w-4" /> Raw Data
                                    </TabsTrigger>
                                </TabsList>
                            </Tabs>
                            <div className="space-y-2">
                                <p>
                                    Viewing raw Gaussian primitives. Each primitive contains position, scale, color, and opacity data.
                                </p>
                            </div>
                        </CardContent>
                    </Card>
                </div>

                <div className="h-[600px] w-full rounded-xl overflow-hidden border border-slate-800 shadow-2xl bg-slate-950">
                    {viewMode === '3d' ? (
                        currentUrl ? (
                            <SplatRenderer url={currentUrl} />
                        ) : (
                            <div className="w-full h-full flex items-center justify-center">
                                <div className="text-center space-y-3">
                                    <Box className="h-12 w-12 text-slate-700 mx-auto" strokeWidth={1} />
                                    <p className="text-slate-500 font-medium">No model loaded</p>
                                    <p className="text-xs text-slate-600">Enter a file path to begin 3D rendering</p>
                                </div>
                            </div>
                        )
                    ) : (
                        ptPath ? (
                            <GaussianInspector ptPath={ptPath} />
                        ) : (
                            <div className="w-full h-full flex items-center justify-center">
                                <div className="text-center space-y-3">
                                    <TableIcon className="h-12 w-12 text-slate-700 mx-auto" strokeWidth={1} />
                                    <p className="text-slate-500 font-medium">No data to inspect</p>
                                    <p className="text-xs text-slate-600">Enter a file path to view raw parameters</p>
                                </div>
                            </div>
                        )
                    )}
                </div>
            </div>
        </div>
    );
}
