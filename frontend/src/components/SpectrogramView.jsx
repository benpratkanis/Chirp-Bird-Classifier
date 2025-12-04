import React from 'react';
import { Card } from '@/components/ui/card';
import { motion } from 'framer-motion';
import { Activity, Waves } from 'lucide-react';

export function SpectrogramView({ spectrogram }) {
    if (!spectrogram) return null;

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="h-full"
        >
            <Card className="h-full overflow-hidden border-none shadow-xl bg-card/90 backdrop-blur-xl p-6">
                <div className="flex items-center gap-2 text-primary mb-4">
                    <Activity className="w-5 h-5" />
                    <span className="text-sm font-medium uppercase tracking-wider">Audio Spectrogram</span>
                </div>

                <div className="relative rounded-lg overflow-hidden bg-black/5 border border-border/50 aspect-[3/1] w-full group">
                    <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                    <img
                        src={spectrogram}
                        alt="Audio Spectrogram"
                        className="w-full h-full object-cover mix-blend-multiply dark:mix-blend-normal dark:opacity-90"
                    />
                    <div className="absolute bottom-2 right-2 px-2 py-1 bg-background/80 backdrop-blur text-xs font-mono text-muted-foreground rounded">
                        Frequency Analysis
                    </div>
                </div>

                <div className="mt-4 flex items-start gap-3 text-sm text-muted-foreground">
                    <Waves className="w-4 h-4 mt-0.5 shrink-0" />
                    <p>
                        Visual representation of the audio frequency spectrum over time.
                        Brighter colors indicate higher intensity at that frequency.
                    </p>
                </div>
            </Card>
        </motion.div>
    );
}
