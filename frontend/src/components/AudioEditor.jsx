import React, { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';
import { Play, Pause, Scissors, ZoomIn, ZoomOut, RefreshCw, AlertCircle } from 'lucide-react';
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
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!containerRef.current) return;

        // Reset state
        setIsReady(false);
        setError(null);
        setDuration(0);
        setCurrentTime(0);

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

        // Pre-fetch duration for fallback
        const media = document.createElement(file.type.startsWith('video') ? 'video' : 'audio');
        media.preload = 'metadata';
        media.onloadedmetadata = () => {
            const dur = media.duration;
            if (dur && dur > 0) {
                // If wavesurfer hasn't set duration yet, or failed, we have this.
                if (!isReady) {
                    setDuration(dur);
                }
            }
        };
        media.src = url;


        // Events
        ws.on('ready', () => {
            setIsReady(true);
            setError(null);
            const dur = ws.getDuration();
            setDuration(dur);

            // Create default region (middle 80%)
            wsRegions.clearRegions();
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

        ws.on('error', (err) => {
            console.warn("WaveSurfer error:", err);
            // If it's a video file, it might fail to decode audio on some mobile browsers.
            // We should still allow the user to proceed if we got the duration from the media element.
            setError("Visual waveform unavailable for this file type.");

            // Fallback: if we have duration from the media element, set up a default trim
            if (media.duration) {
                const dur = media.duration;
                setDuration(dur);
                onTrimChange(0, dur); // Default to full duration if we can't show regions
                setIsReady(true); // Pretend we are ready so user can submit
            }
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
        if (wavesurferRef.current && isReady && !error) {
            try {
                wavesurferRef.current.zoom(zoom);
            } catch (e) {
                console.warn("Error setting zoom:", e);
            }
        }
    }, [zoom, isReady, error]);

    const togglePlay = () => {
        wavesurferRef.current?.playPause();
    };

    const resetRegion = () => {
        if (regionsRef.current && duration > 0 && !error) {
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
        } else if (error && duration > 0) {
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

            <div className="relative w-full rounded-lg overflow-hidden bg-background/50 border border-border/50 min-h-[128px]">
                {error && (
                    <div className="absolute inset-0 flex items-center justify-center text-muted-foreground text-sm bg-background/80 z-10">
                        <AlertCircle className="w-4 h-4 mr-2" />
                        {error} (You can still submit)
                    </div>
                )}
                <div
                    ref={containerRef}
                    className="w-full"
                    data-testid="waveform-container"
                />
            </div>

            <div className="flex items-center gap-4 flex-wrap">
                <Button
                    variant="outline"
                    size="icon"
                    onClick={togglePlay}
                    disabled={!!error}
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
                        disabled={!!error}
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
