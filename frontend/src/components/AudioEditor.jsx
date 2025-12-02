import React, { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';
import { Play, Pause, Scissors, ZoomIn, ZoomOut, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card } from '@/components/ui/card';

export function AudioEditor({ file, onTrimChange }) {
    const containerRef = useRef(null);
    const wavesurferRef = useRef(null);
    const regionsRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [duration, setDuration] = useState(0);
    const [currentTime, setCurrentTime] = useState(0);
    const [zoom, setZoom] = useState(10);
    const [isReady, setIsReady] = useState(false);

    useEffect(() => {
        if (!containerRef.current) return;

        // Initialize WaveSurfer
        const ws = WaveSurfer.create({
            container: containerRef.current,
            waveColor: 'rgba(200, 210, 205, 0.3)', // Muted sage for non-selected regions
            progressColor: 'rgb(46, 125, 85)', // Primary forest green for progress within region
            cursorColor: 'rgb(46, 125, 85)', // Forest green cursor
            barWidth: 2,
            barGap: 3,
            barRadius: 2,
            height: 128,
            normalize: true,
            minPxPerSec: zoom,
        });

        // Initialize Regions Plugin
        const wsRegions = RegionsPlugin.create();
        ws.registerPlugin(wsRegions);
        regionsRef.current = wsRegions;

        wavesurferRef.current = ws;

        // Load audio
        const url = URL.createObjectURL(file);
        ws.load(url);

        // Events
        ws.on('ready', () => {
            setIsReady(true);
            const dur = ws.getDuration();
            setDuration(dur);

            // Create default region (middle 80%)
            wsRegions.addRegion({
                start: dur * 0.1,
                end: dur * 0.9,
                color: 'rgba(46, 125, 85, 0.2)', // Green overlay for selected region
                drag: true,
                resize: true,
                id: 'trim-region'
            });

            onTrimChange(dur * 0.1, dur * 0.9);
        });

        ws.on('play', () => setIsPlaying(true));
        ws.on('pause', () => setIsPlaying(false));
        ws.on('timeupdate', (time) => setCurrentTime(time));

        wsRegions.on('region-updated', (region) => {
            onTrimChange(region.start, region.end);
        });

        return () => {
            try {
                ws.destroy();
            } catch (e) {
                console.warn("Error destroying wavesurfer instance:", e);
            }
            URL.revokeObjectURL(url);
        };
    }, [file]);

    useEffect(() => {
        if (wavesurferRef.current && isReady) {
            try {
                wavesurferRef.current.zoom(zoom);
            } catch (e) {
                console.warn("Error setting zoom:", e);
            }
        }
    }, [zoom, isReady]);

    const togglePlay = () => {
        wavesurferRef.current?.playPause();
    };

    const resetRegion = () => {
        if (regionsRef.current && duration > 0) {
            regionsRef.current.clearRegions();
            regionsRef.current.addRegion({
                start: 0,
                end: duration,
                color: 'rgba(46, 125, 85, 0.2)',
                drag: true,
                resize: true,
                id: 'trim-region'
            });
            onTrimChange(0, duration);
        }
    };

    return (
        <Card className="p-6 space-y-6 bg-card/80 backdrop-blur-sm border-border/60 shadow-sm" data-testid="audio-editor">
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-serif font-medium flex items-center gap-2">
                    <Scissors className="w-4 h-4 text-accent" />
                    Trim Audio
                </h3>
                <div className="text-xs font-mono text-muted-foreground">
                    {currentTime.toFixed(2)}s / {duration.toFixed(2)}s
                </div>
            </div>

            <div
                ref={containerRef}
                className="w-full rounded-lg overflow-hidden bg-background/50 border border-border/50"
                data-testid="waveform-container"
            />

            <div className="flex items-center gap-4 flex-wrap">
                <Button
                    variant="outline"
                    size="icon"
                    onClick={togglePlay}
                    className="h-10 w-10 rounded-full border-primary/20 hover:border-primary hover:bg-primary/5 hover:text-primary transition-colors"
                    data-testid="button-play-pause"
                >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
                </Button>

                <div className="flex items-center gap-2 flex-1 min-w-[120px]">
                    <ZoomOut className="w-4 h-4 text-muted-foreground" />
                    <Slider
                        value={[zoom]}
                        min={10}
                        max={200}
                        step={10}
                        onValueChange={(val) => setZoom(val[0])}
                        className="w-full"
                        data-testid="slider-zoom"
                    />
                    <ZoomIn className="w-4 h-4 text-muted-foreground" />
                </div>

                <Button
                    variant="ghost"
                    size="sm"
                    onClick={resetRegion}
                    className="text-xs text-muted-foreground hover:text-foreground"
                    data-testid="button-reset-trim"
                >
                    <RefreshCw className="w-3 h-3 mr-2" />
                    Reset Trim
                </Button>
            </div>
        </Card>
    );
}
