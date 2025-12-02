import React, { useState, useRef } from 'react';
import { Upload } from 'lucide-react';
import { cn } from '@/lib/utils';

export function AudioUploader({ onFileSelect }) {
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef(null);

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
            if (file.type.startsWith('audio/')) {
                onFileSelect(file);
            }
        }
    };

    const handleClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            onFileSelect(e.target.files[0]);
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
                accept="audio/*"
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
                        Upload Bird Recording
                    </h3>
                    <p className="text-sm text-muted-foreground max-w-xs mx-auto">
                        Drag and drop your audio file here, or click to browse.
                        <br />
                        <span className="text-xs opacity-70">Supports MP3, WAV, OGG</span>
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
