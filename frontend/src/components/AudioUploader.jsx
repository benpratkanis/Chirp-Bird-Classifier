import React, { useState, useRef } from 'react';
import { Upload, FileVideo, FileAudio } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';

export function AudioUploader({ onFileSelect }) {
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef(null);
    const { toast } = useToast();

    const MAX_SIZE_MB = 50;
    const MAX_DURATION_SEC = 20;

    const validateFile = (file) => {
        // Check size
        if (file.size > MAX_SIZE_MB * 1024 * 1024) {
            toast({
                title: "File too large",
                description: `Please upload a file smaller than ${MAX_SIZE_MB}MB.`,
                variant: "destructive"
            });
            return false;
        }

        // Check duration for audio/video
        const url = URL.createObjectURL(file);
        const media = new Audio(url);

        media.onloadedmetadata = () => {
            URL.revokeObjectURL(url);
            if (media.duration > MAX_DURATION_SEC) {
                toast({
                    title: "File too long",
                    description: `Please upload a recording shorter than ${MAX_DURATION_SEC} seconds.`,
                    variant: "destructive"
                });
            } else {
                onFileSelect(file);
            }
        };

        media.onerror = () => {
            // Fallback if Audio() fails (e.g. some video formats), try to accept it or use Video element
            // For simplicity, if we can't determine duration easily without a video element, we might just pass it 
            // or try creating a video element.
            const video = document.createElement('video');
            video.preload = 'metadata';
            video.onloadedmetadata = () => {
                URL.revokeObjectURL(url);
                if (video.duration > MAX_DURATION_SEC) {
                    toast({
                        title: "File too long",
                        description: `Please upload a recording shorter than ${MAX_DURATION_SEC} seconds.`,
                        variant: "destructive"
                    });
                } else {
                    onFileSelect(file);
                }
            };
            video.onerror = () => {
                URL.revokeObjectURL(url);
                // If we can't read metadata, we might warn or just allow it if it's a valid type
                onFileSelect(file);
            };
            video.src = url;
        };

        return true;
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.type.startsWith('audio/') || file.type.startsWith('video/')) {
                validateFile(file);
            } else {
                toast({
                    title: "Invalid file type",
                    description: "Please upload an audio or video file.",
                    variant: "destructive"
                });
            }
        }
    };

    const handleClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            validateFile(e.target.files[0]);
        }
    };

    return (
        <div
            className={cn(
                "relative group cursor-pointer transition-all duration-300 ease-out",
                "border-2 border-dashed rounded-xl p-12 text-center",
                "bg-card/50 hover:bg-card/80 glass-panel",
                isDragging ? "border-accent bg-accent/5 scale-[1.02]" : "border-border hover:border-primary/50"
            )}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleClick}
            data-testid="uploader-dropzone"
        >
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="audio/*,video/*"
                className="hidden"
                data-testid="input-audio-file"
            />

            <div className="flex flex-col items-center gap-4">
                <div className={cn(
                    "p-4 rounded-full bg-secondary transition-transform duration-500",
                    "group-hover:scale-110 group-hover:bg-primary/10"
                )}>
                    <Upload className="w-8 h-8 text-primary" />
                </div>
                <div className="space-y-2">
                    <h3 className="text-xl font-serif font-medium text-foreground">
                        Upload Recording
                    </h3>
                    <p className="text-sm text-muted-foreground max-w-xs mx-auto">
                        Drag and drop your audio or video file here.
                        <br />
                        <span className="text-xs opacity-70">Max 20s • Max 50MB</span>
                    </p>
                </div>
            </div>

            {/* Decorative corners */}
            <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-primary/30 rounded-tl-lg m-2 transition-all group-hover:m-1" />
            <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-primary/30 rounded-tr-lg m-2 transition-all group-hover:m-1" />
            <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-primary/30 rounded-bl-lg m-2 transition-all group-hover:m-1" />
            <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-primary/30 rounded-br-lg m-2 transition-all group-hover:m-1" />
        </div>
    );
}
