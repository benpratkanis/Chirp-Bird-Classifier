import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { cn } from '@/lib/utils';

export function AudioRecorder({ onRecordingComplete }) {
    const [isRecording, setIsRecording] = useState(false);
    const [recordingTime, setRecordingTime] = useState(0);
    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);
    const timerRef = useRef(null);
    const MAX_DURATION = 20; // seconds

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            chunksRef.current = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                }
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/wav' });
                const file = new File([blob], `recording-${Date.now()}.wav`, { type: 'audio/wav' });
                onRecordingComplete(file);

                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsRecording(true);
            setRecordingTime(0);

            timerRef.current = setInterval(() => {
                setRecordingTime(prev => {
                    if (prev >= MAX_DURATION) {
                        stopRecording();
                        return prev;
                    }
                    return prev + 1;
                });
            }, 1000);

        } catch (err) {
            console.error("Error accessing microphone:", err);
            alert("Could not access microphone. Please ensure you have granted permission.");
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            if (timerRef.current) {
                clearInterval(timerRef.current);
            }
        }
    };

    useEffect(() => {
        return () => {
            if (timerRef.current) {
                clearInterval(timerRef.current);
            }
            if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
                mediaRecorderRef.current.stop();
            }
        };
    }, []);

    return (
        <Card className="p-6 border-2 border-dashed border-border bg-card/30 backdrop-blur-sm flex flex-col items-center gap-4 relative overflow-hidden" data-testid="audio-recorder">
            {isRecording && (
                <div className="absolute inset-0 bg-red-500/5 animate-pulse pointer-events-none" />
            )}

            <div className="relative">
                <div className={cn(
                    "w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300",
                    isRecording ? "bg-red-500/20 scale-110" : "bg-primary/10"
                )}>
                    {isRecording ? (
                        <div className="w-8 h-8 bg-red-500 rounded-sm animate-pulse" />
                    ) : (
                        <Mic className="w-8 h-8 text-primary" />
                    )}
                </div>
                {isRecording && (
                    <div className="absolute -inset-1 rounded-full border-2 border-red-500/50 animate-ping" />
                )}
            </div>

            <div className="text-center space-y-1 z-10">
                <h3 className="font-serif font-medium text-foreground">
                    {isRecording ? "Recording in Progress" : "Record Audio"}
                </h3>
                <p className="text-sm text-muted-foreground font-mono">
                    {isRecording
                        ? `00:${recordingTime.toString().padStart(2, '0')} / 00:${MAX_DURATION}`
                        : "Click to start recording (max 20s)"}
                </p>
            </div>

            <Button
                variant={isRecording ? "destructive" : "default"}
                size="lg"
                onClick={isRecording ? stopRecording : startRecording}
                className={cn(
                    "min-w-[160px] transition-all",
                    isRecording ? "hover:bg-red-600" : "hover:bg-primary/90"
                )}
                data-testid={isRecording ? "button-stop-recording" : "button-start-recording"}
            >
                {isRecording ? (
                    <>
                        <Square className="w-4 h-4 mr-2 fill-current" />
                        Stop Recording
                    </>
                ) : (
                    <>
                        <Mic className="w-4 h-4 mr-2" />
                        Start Recording
                    </>
                )}
            </Button>
        </Card>
    );
}
