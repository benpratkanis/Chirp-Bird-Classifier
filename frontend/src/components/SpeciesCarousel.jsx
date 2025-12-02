import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { speciesImages } from '@/lib/speciesImages';

export function SpeciesCarousel() {
    const speciesList = Object.keys(speciesImages);
    const [currentIndex, setCurrentIndex] = useState(0);

    useEffect(() => {
        const timer = setInterval(() => {
            setCurrentIndex((prev) => (prev + 1) % speciesList.length);
        }, 3000); // Change every 3 seconds

        return () => clearInterval(timer);
    }, [speciesList.length]);

    // We'll show 3 images at a time on desktop, 1 on mobile
    // But for a simple cycling effect, let's just cycle through them in a row or a single highlight.
    // The user said "cycle through them".

    const visibleSpecies = [
        speciesList[currentIndex],
        speciesList[(currentIndex + 1) % speciesList.length],
        speciesList[(currentIndex + 2) % speciesList.length],
    ];

    return (
        <div className="w-full mt-16 mb-8">
            <h3 className="text-2xl font-serif font-bold text-center mb-8 text-foreground">Discover Local Species</h3>

            <div className="flex justify-center gap-6 overflow-hidden px-4">
                <AnimatePresence mode="popLayout">
                    {visibleSpecies.map((species, idx) => (
                        <motion.div
                            key={`${species}-${currentIndex}`} // Key changes to trigger animation
                            initial={{ opacity: 0, x: 50 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -50 }}
                            transition={{ duration: 0.5 }}
                            className={`relative ${idx === 1 ? 'w-64 h-80 z-10' : 'w-48 h-64 mt-8 opacity-70 hidden md:block'}`}
                        >
                            <Card className="h-full overflow-hidden border-none shadow-lg group cursor-pointer">
                                <img
                                    src={speciesImages[species]}
                                    alt={species}
                                    className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
                                />
                                <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent p-4">
                                    <p className="text-white font-medium text-center truncate">
                                        {species.replace(/_/g, ' ')}
                                    </p>
                                </div>
                            </Card>
                        </motion.div>
                    ))}
                </AnimatePresence>
            </div>
        </div>
    );
}
