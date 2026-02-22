"use client";

import React, { useState, useEffect } from 'react';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table";
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Loader2, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface Gaussian {
    id: number;
    pos: number[];
    scale: number[];
    color: number[];
    opacity: number;
    rot: number[];
}

interface GaussianInspectorProps {
    ptPath: string;
}

export function GaussianInspector({ ptPath }: GaussianInspectorProps) {
    const [gaussians, setGaussians] = useState<Gaussian[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchData = async () => {
        if (!ptPath) return;
        setIsLoading(true);
        setError(null);
        try {
            const response = await fetch(`http://localhost:8000/inspect-gaussians/${encodeURIComponent(ptPath)}?limit=100`);
            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }
            const data = await response.json();
            setGaussians(data.gaussians);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
    }, [ptPath]);

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-64">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    if (error) {
        return (
            <Card className="border-destructive">
                <CardContent className="pt-6">
                    <p className="text-destructive">Failed to load Gaussian data: {error}</p>
                    <Button onClick={fetchData} className="mt-4" variant="outline">
                        <RefreshCw className="mr-2 h-4 w-4" /> Try Again
                    </Button>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="h-full flex flex-col">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-lg font-bold">Raw Gaussian Parameters (Limit 100)</CardTitle>
                <Button size="sm" variant="ghost" onClick={fetchData}>
                    <RefreshCw className="h-4 w-4" />
                </Button>
            </CardHeader>
            <CardContent className="flex-1 overflow-auto">
                <div className="rounded-md border">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead className="w-[50px]">ID</TableHead>
                                <TableHead>Position (XYZ)</TableHead>
                                <TableHead>Scale (XYZ)</TableHead>
                                <TableHead>Color (RGB)</TableHead>
                                <TableHead>Opacity</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {gaussians.map((g) => (
                                <TableRow key={g.id}>
                                    <TableCell className="font-medium text-xs text-muted-foreground">{g.id}</TableCell>
                                    <TableCell className="text-xs font-mono">
                                        {g.pos.map(v => v.toFixed(3)).join(', ')}
                                    </TableCell>
                                    <TableCell className="text-xs font-mono">
                                        {g.scale.map(v => v.toFixed(3)).join(', ')}
                                    </TableCell>
                                    <TableCell className="text-xs font-mono">
                                        {g.color.map(v => Math.round(v)).join(', ')}
                                    </TableCell>
                                    <TableCell className="text-xs font-mono">
                                        {g.opacity.toFixed(3)}
                                    </TableCell>
                                </TableRow>
                            ))}
                            {gaussians.length === 0 && (
                                <TableRow>
                                    <TableCell colSpan={5} className="text-center py-8 text-muted-foreground">
                                        No data available for this path.
                                    </TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </div>
            </CardContent>
        </Card>
    );
}
