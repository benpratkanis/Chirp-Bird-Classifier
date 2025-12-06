import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { motion } from 'framer-motion';
import { CheckCircle2, BarChart3, Bird, TrendingDown } from 'lucide-react';

export function ClassificationResult({ species, confidence, allPredictions = [], spectrogram, spectrograms, speciesImage }) {
    // Use the first spectrogram for the main card display if available
    const displaySpectrogram = spectrogram || (spectrograms && spectrograms.length > 0 ? spectrograms[0] : null);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
        >
            <Card className="overflow-hidden border-none shadow-2xl bg-card/90 backdrop-blur-xl" data-testid="classification-result">
                <div className="grid md:grid-cols-2 gap-0">
                    <div className="relative h-64 md:h-full min-h-[300px] overflow-hidden bg-gradient-to-br from-primary/20 via-accent/10 to-primary/5 flex items-center justify-center">
                        {speciesImage ? (
                            <img
                                src={speciesImage}
                                alt={species}
                                className="w-full h-full object-cover"
                            />
                        ) : displaySpectrogram ? (
                            <img
                                src={displaySpectrogram}
                                alt="Spectrogram"
                                className="w-full h-full object-cover opacity-80 mix-blend-multiply"
                            />
                        ) : (
                            <div className="text-center space-y-4">
                                <motion.div
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    transition={{ delay: 0.3, type: "spring", stiffness: 200 }}
                                    className="inline-flex items-center justify-center w-32 h-32 rounded-full bg-primary/10 border-2 border-primary/20"
                                >
                                    <Bird className="w-16 h-16 text-primary" />
                                </motion.div>
                                <div className="md:hidden px-4">
                                    <h2 className="text-2xl font-serif font-bold text-foreground">{species}</h2>
                                </div>
                            </div>
                        )}

                        <div className="absolute top-4 left-4">
                            <Badge variant="outline" className="bg-background/80 backdrop-blur text-primary border-primary/20">
                                AI Identified
                            </Badge>
                        </div>
                    </div>

                    <div className="p-8 flex flex-col justify-center space-y-6 relative">
                        <div className="space-y-2 hidden md:block">
                            <Badge variant="outline" className="bg-primary/10 text-primary border-primary/20 mb-2">
                                Match Found
                            </Badge>
                            <h2 className="text-4xl font-serif font-bold text-foreground tracking-tight break-words leading-tight" data-testid="text-species-name">
                                {species}
                            </h2>
                        </div>

                        <div className="space-y-4 py-6 border-t border-border/50">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2 text-muted-foreground">
                                    <BarChart3 className="w-5 h-5" />
                                    <span className="font-medium">Confidence Score</span>
                                </div>
                                <span className="text-2xl font-mono font-bold text-primary" data-testid="text-confidence">
                                    {(confidence * 100).toFixed(1)}%
                                </span>
                            </div>

                            <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                                <motion.div
                                    className="h-full bg-primary"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${confidence * 100}%` }}
                                    transition={{ duration: 1, delay: 0.2 }}
                                />
                            </div>
                        </div>

                        {allPredictions.length > 1 && (
                            <div className="space-y-3">
                                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                    <TrendingDown className="w-4 h-4" />
                                    <span>Other Possibilities</span>
                                </div>
                                <div className="space-y-2">
                                    {allPredictions.slice(1, 4).map((pred, idx) => (
                                        <div key={idx} className="flex items-center justify-between text-sm">
                                            <span className="text-muted-foreground">{pred.species}</span>
                                            <span className="font-mono text-muted-foreground">
                                                {(pred.probability * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        <div className="bg-muted/50 rounded-lg p-4 border border-border/50">
                            <div className="flex items-start gap-3">
                                <CheckCircle2 className="w-5 h-5 text-primary mt-0.5" />
                                <div>
                                    <h4 className="font-medium text-foreground">Identification Complete</h4>
                                    <p className="text-sm text-muted-foreground mt-1">
                                        The audio signature matches known patterns for this species in our trained model.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </Card>
        </motion.div>
    );
}
