import React from 'react';
import { Card } from '@/components/ui/card';
import { motion } from 'framer-motion';
import { Info, MapPin, Music, Feather, Plane, Trees, BookOpen } from 'lucide-react';
import { speciesInfo } from '@/lib/speciesInfo';

export function SpeciesInfoWindow({ species }) {
    const info = speciesInfo[species];

    if (!info) {
        return null;
    }

    const sections = [
        {
            icon: <Feather className="w-5 h-5 text-primary" />,
            title: "Physical Description",
            content: info.physicalDescription
        },
        {
            icon: <Music className="w-5 h-5 text-primary" />,
            title: "Call / Vocalization",
            content: info.callDescription
        },
        {
            icon: <MapPin className="w-5 h-5 text-primary" />,
            title: "Native Range",
            content: info.nativeRange
        },
        {
            icon: <Plane className="w-5 h-5 text-primary" />,
            title: "Migratory Patterns",
            content: info.migratoryPatterns
        },
        {
            icon: <Trees className="w-5 h-5 text-primary" />,
            title: "Typical Habitat",
            content: info.typicalHabitat
        },
        {
            icon: <BookOpen className="w-5 h-5 text-primary" />,
            title: "Additional Info",
            content: info.additionalInfo
        }
    ];

    return (
        <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, ease: "easeOut", delay: 0.2 }}
            className="h-full"
        >
            <Card className="h-full overflow-hidden border-none shadow-2xl bg-card/90 backdrop-blur-xl p-6 md:p-8">
                <div className="space-y-6">
                    <div className="border-b border-border/50 pb-4">
                        <div className="flex items-center gap-2 text-primary mb-2">
                            <Info className="w-5 h-5" />
                            <span className="text-sm font-medium uppercase tracking-wider">Species Information</span>
                        </div>
                        <h2 className="text-3xl font-serif font-bold text-foreground">{info.commonName}</h2>
                        <p className="text-lg text-muted-foreground italic font-serif">{info.scientificName}</p>
                    </div>

                    <div className="grid gap-6">
                        {sections.map((section, idx) => (
                            <motion.div
                                key={idx}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.3 + (idx * 0.1) }}
                                className="group"
                            >
                                <div className="flex items-start gap-3">
                                    <div className="mt-1 p-2 rounded-full bg-primary/10 group-hover:bg-primary/20 transition-colors">
                                        {section.icon}
                                    </div>
                                    <div>
                                        <h3 className="font-medium text-foreground mb-1">{section.title}</h3>
                                        <p className="text-sm text-muted-foreground leading-relaxed">
                                            {section.content}
                                        </p>
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </Card>
        </motion.div>
    );
}
