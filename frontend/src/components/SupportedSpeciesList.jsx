import React from 'react';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { speciesImages } from '@/lib/speciesImages';
import { Bird } from 'lucide-react';

export function SupportedSpeciesList() {
    const SUPPORTED_SPECIES = [
        "American_Crow",
        "American_Robin",
        "Black-capped_Chickadee",
        "Blue_Jay",
        "Dark-eyed_Junco",
        "House_Finch",
        "Northern_Cardinal",
        "Northern_Flicker",
        "Red-bellied_Woodpecker",
        "Red-winged_Blackbird",
        "Song_Sparrow",
        "Tufted_Titmouse",
        "White-breasted_Nuthatch"
    ];

    const speciesList = Object.entries(speciesImages)
        .filter(([name]) => SUPPORTED_SPECIES.includes(name))
        .sort((a, b) => a[0].localeCompare(b[0]));

    return (
        <Card className="h-full bg-card/50 backdrop-blur-sm border-border/50 overflow-hidden flex flex-col">
            <div className="p-4 border-b border-border/50 bg-card/80">
                <h3 className="font-serif font-medium flex items-center gap-2 text-foreground">
                    <Bird className="w-4 h-4 text-primary" />
                    Supported Species
                    <span className="text-xs font-mono text-muted-foreground ml-auto bg-muted px-2 py-0.5 rounded-full">
                        {speciesList.length}
                    </span>
                </h3>
            </div>
            <ScrollArea className="flex-1">
                <div className="p-2 space-y-2">
                    {speciesList.map(([name, image]) => (
                        <div
                            key={name}
                            className="flex items-center gap-3 p-2 rounded-lg hover:bg-accent/50 transition-colors group"
                        >
                            <div className="w-10 h-10 rounded-full overflow-hidden border border-border/50 shrink-0">
                                <img
                                    src={image}
                                    alt={name}
                                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                                />
                            </div>
                            <span className="text-sm text-muted-foreground group-hover:text-foreground transition-colors">
                                {name.replace(/_/g, ' ')}
                            </span>
                        </div>
                    ))}
                </div>
            </ScrollArea>
        </Card>
    );
}
