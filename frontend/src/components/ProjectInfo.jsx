import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import distributionImage from '@/assets/Distribution.png';
import balancedImage from '@/assets/BalancedSet.png';
import confusionMatrixImage from '@/assets/ConfusionMatrix.png';

export function ProjectInfo() {
    const distributionData = [
        { species: "Red-winged Blackbird", count: 33739 },
        { species: "Blue Jay", count: 26905 },
        { species: "Black-capped Chickadee", count: 23295 },
        { species: "American Robin", count: 21746 },
        { species: "American Crow", count: 20109 },
        { species: "Northern Cardinal", count: 18954 },
        { species: "Song Sparrow", count: 16834 },
        { species: "Northern Flicker", count: 15204 },
        { species: "Tufted Titmouse", count: 14796 },
        { species: "House Finch", count: 13467 },
        { species: "White-breasted Nuthatch", count: 12293 },
        { species: "Dark-eyed Junco", count: 12003 },
        { species: "Red-bellied Woodpecker", count: 10765 },
        { species: "American Goldfinch", count: 9799 },
        { species: "House Sparrow", count: 9524 },
        { species: "Common Grackle", count: 9285 },
        { species: "Downy Woodpecker", count: 7446 },
        { species: "European Starling", count: 7169 },
        { species: "Eastern Bluebird", count: 3930 },
        { species: "Mourning Dove", count: 1293 },
    ];

    const performanceData = [
        { species: "American Crow", precision: 0.9794, recall: 0.9755, f1: 0.9775, support: 2000 },
        { species: "American Robin", precision: 0.9859, recall: 0.9790, f1: 0.9824, support: 2000 },
        { species: "Black-capped Chickadee", precision: 0.9734, recall: 0.9700, f1: 0.9717, support: 2000 },
        { species: "Blue Jay", precision: 0.9823, recall: 0.9705, f1: 0.9764, support: 2000 },
        { species: "Dark-eyed Junco", precision: 0.9822, recall: 0.9945, f1: 0.9883, support: 2000 },
        { species: "House Finch", precision: 0.9925, recall: 0.9910, f1: 0.9917, support: 2000 },
        { species: "Northern Cardinal", precision: 0.9810, recall: 0.9800, f1: 0.9805, support: 2000 },
        { species: "Northern Flicker", precision: 0.9865, recall: 0.9845, f1: 0.9855, support: 2000 },
        { species: "Red-bellied Woodpecker", precision: 0.9930, recall: 0.9905, f1: 0.9917, support: 2000 },
        { species: "Red-winged Blackbird", precision: 0.9801, recall: 0.9845, f1: 0.9823, support: 2000 },
        { species: "Song Sparrow", precision: 0.9763, recall: 0.9895, f1: 0.9829, support: 2000 },
        { species: "Tufted Titmouse", precision: 0.9860, recall: 0.9840, f1: 0.9850, support: 2000 },
        { species: "White-breasted Nuthatch", precision: 0.9831, recall: 0.9880, f1: 0.9855, support: 2000 },
    ];

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-12 max-w-6xl mx-auto pb-12"
        >
            <div className="text-center space-y-4">
                <h2 className="text-3xl font-bold tracking-tight">Project Report</h2>
                <p className="text-muted-foreground max-w-2xl mx-auto">
                    An overview of the dataset, training process, and performance metrics for the Chirp bird classifier.
                </p>
            </div>

            <div className="grid gap-8 md:grid-cols-2">
                <Card className="md:col-span-2">
                    <CardHeader>
                        <CardTitle>Project Overview</CardTitle>
                        <CardDescription>
                            Automated bird species identification from raw audio field recordings.
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <p className="text-sm text-muted-foreground">
                            The primary objective of Project "Chirp" is to develop a deep-learning-based system capable of identifying the most common birds native to Erie, PA. By transforming audio into visual spectrograms, we leverage the spatial pattern recognition capabilities of Convolutional Neural Networks (CNNs) to distinguish unique acoustic features like trills, pitch, and rhythm.
                        </p>
                        <div className="grid sm:grid-cols-2 gap-4 pt-4">
                            <div className="bg-muted/30 p-4 rounded-lg">
                                <h4 className="font-semibold mb-2 flex items-center gap-2">
                                    <span className="w-2 h-2 rounded-full bg-primary" />
                                    Objective
                                </h4>
                                <p className="text-sm text-muted-foreground">
                                    Replicate the high-fidelity of tools like Merlin Bird ID on a focused local dataset, creating a transparent and explainable pipeline.
                                </p>
                            </div>
                            <div className="bg-muted/30 p-4 rounded-lg">
                                <h4 className="font-semibold mb-2 flex items-center gap-2">
                                    <span className="w-2 h-2 rounded-full bg-primary" />
                                    Approach
                                </h4>
                                <p className="text-sm text-muted-foreground">
                                    Supervised multi-class classification using EfficientNet-B4 on high-resolution (384x384) Mel Spectrograms treated as 3-channel RGB images.
                                </p>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card className="md:col-span-2">
                    <CardHeader>
                        <CardTitle>Data Acquisition</CardTitle>
                        <CardDescription>
                            Sourcing high-quality audio data to build a robust foundation.
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="grid sm:grid-cols-3 gap-4 text-center">
                            <div className="bg-muted/30 p-4 rounded-lg">
                                <h4 className="font-semibold mb-1">Source</h4>
                                <p className="text-sm text-muted-foreground">
                                    Macaulay Library at the Cornell Lab of Ornithology
                                </p>
                            </div>
                            <div className="bg-muted/30 p-4 rounded-lg">
                                <h4 className="font-semibold mb-1">Volume</h4>
                                <p className="text-sm text-muted-foreground">
                                    ~50,000 Audio Files<br />
                                    <strong>422 GB</strong> Raw Data
                                </p>
                            </div>
                            <div className="bg-muted/30 p-4 rounded-lg">
                                <h4 className="font-semibold mb-1">Compliance</h4>
                                <p className="text-sm text-muted-foreground">
                                    Formal media request approved for educational use
                                </p>
                            </div>
                        </div>
                        <p className="text-sm text-muted-foreground mt-4 text-center">
                            The raw dataset included diverse audio formats (.wav, .mp3, .m4a) which were robustly handled using ffmpeg to ensure no data was discarded due to container incompatibility.
                        </p>
                    </CardContent>
                </Card>

                <Card className="md:col-span-2">
                    <CardHeader>
                        <CardTitle>Methodology: "RGB" Audio Processing</CardTitle>
                        <CardDescription>
                            Converting raw audio into high-fidelity 3-channel images for deep learning.
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="grid md:grid-cols-3 gap-6">
                            <div className="space-y-2">
                                <div className="h-2 w-full bg-red-500/20 rounded-full overflow-hidden">
                                    <div className="h-full w-3/4 bg-red-500" />
                                </div>
                                <h4 className="font-semibold">Channel 1: Energy (Red)</h4>
                                <p className="text-xs text-muted-foreground">
                                    <strong>PCEN (Per-Channel Energy Normalization)</strong> adapts to background noise (wind, rain) and suppresses it while highlighting rapid transient events like chirps.
                                </p>
                            </div>
                            <div className="space-y-2">
                                <div className="h-2 w-full bg-green-500/20 rounded-full overflow-hidden">
                                    <div className="h-full w-3/4 bg-green-500" />
                                </div>
                                <h4 className="font-semibold">Channel 2: Velocity (Green)</h4>
                                <p className="text-xs text-muted-foreground">
                                    The <strong>Delta</strong> feature, representing the rate of change in energy over time, capturing the "speed" of the call.
                                </p>
                            </div>
                            <div className="space-y-2">
                                <div className="h-2 w-full bg-blue-500/20 rounded-full overflow-hidden">
                                    <div className="h-full w-3/4 bg-blue-500" />
                                </div>
                                <h4 className="font-semibold">Channel 3: Acceleration (Blue)</h4>
                                <p className="text-xs text-muted-foreground">
                                    The <strong>Delta-Delta</strong> feature, capturing complex modulation and second-order temporal dynamics.
                                </p>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card className="md:col-span-2">
                    <CardHeader>
                        <CardTitle>Dataset Processing</CardTitle>
                        <CardDescription>
                            Evolution of the dataset from raw collection to balanced training set.
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-8">
                        <div className="grid md:grid-cols-2 gap-8">
                            <div className="space-y-4">
                                <h3 className="font-semibold text-lg">Original Distribution</h3>
                                <p className="text-sm text-muted-foreground">
                                    Initial dataset containing 20 species with significant class imbalance.
                                    Total spectrograms: {distributionData.reduce((acc, curr) => acc + curr.count, 0).toLocaleString()}
                                </p>
                                <img
                                    src={distributionImage}
                                    alt="Original Dataset Distribution"
                                    className="w-full rounded-lg border shadow-sm"
                                />
                                <div className="h-[560px] overflow-y-auto border rounded-md">
                                    <Table>
                                        <TableHeader>
                                            <TableRow>
                                                <TableHead>Species</TableHead>
                                                <TableHead className="text-right">Original Count</TableHead>
                                            </TableRow>
                                        </TableHeader>
                                        <TableBody>
                                            {distributionData.map((item) => (
                                                <TableRow key={item.species}>
                                                    <TableCell className="font-medium">{item.species}</TableCell>
                                                    <TableCell className="text-right">{item.count.toLocaleString()}</TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                </div>
                            </div>

                            <div className="space-y-4">
                                <h3 className="font-semibold text-lg">Balanced Training Set</h3>
                                <p className="text-sm text-muted-foreground">
                                    Refined dataset filtered to 13 species. Classes were balanced through
                                    downsampling (for abundant species) and upsampling (for rarer species)
                                    to ensure uniform training data.
                                </p>
                                <img
                                    src={balancedImage}
                                    alt="Balanced Dataset Distribution"
                                    className="w-full rounded-lg border shadow-sm"
                                />
                                <div className="bg-muted/30 p-4 rounded-lg text-sm space-y-4">
                                    <div>
                                        <h4 className="font-semibold mb-2">Dataset Balancing</h4>
                                        <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                                            <li>AI-Assisted Cleaning: Used ResNet18 + K-Means clustering to identify and remove "junk" clusters.</li>
                                            <li>Filtered from 20 to 13 core species to ensure quality depth (20k+ samples).</li>
                                            <li>Upsampling: Random Duplication | Downsampling: Random Deletion.</li>
                                            <li>Result: Balanced distribution for optimal model training.</li>
                                        </ul>
                                    </div>
                                    <div className="border-t pt-4 border-border/50">
                                        <h4 className="font-semibold mb-2">Preprocessing Logic (Script 0)</h4>
                                        <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                                            <li><strong>Standardization:</strong> Resampled to 32kHz; rejected files &gt;150MB or silence.</li>
                                            <li><strong>Segmentation:</strong> Intelligent scanner targets 800Hz–13.5kHz band.</li>
                                            <li><strong>Detection:</strong> Dynamic thresholding (Noise Floor + 0.12).</li>
                                            <li><strong>Failsafe:</strong> Captures loudest 3s segment if no peaks found.</li>
                                            <li><strong>Output:</strong> Max 12 segments/file as 384x384 RGB spectrograms.</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card className="md:col-span-2">
                    <CardHeader>
                        <CardTitle>Data Processing Pipeline</CardTitle>
                        <CardDescription>
                            End-to-end workflow from raw audio to trained model, executed via custom Python scripts.
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="relative border-l-2 border-muted ml-4 space-y-8 py-2">
                            <div className="relative pl-8">
                                <div className="absolute -left-[9px] top-1 h-4 w-4 rounded-full bg-primary" />
                                <h4 className="font-semibold text-sm">Step 1: Ingestion & Transformation</h4>
                                <p className="text-xs font-mono text-muted-foreground mb-1">0_bird_audio_preprocess_highres.py</p>
                                <p className="text-sm text-muted-foreground">
                                    Raw audio files are ingested, segmented into 3-second clips, and converted into high-resolution 384x384 RGB spectrograms using PCEN for noise suppression.
                                </p>
                            </div>
                            <div className="relative pl-8">
                                <div className="absolute -left-[9px] top-1 h-4 w-4 rounded-full bg-primary" />
                                <h4 className="font-semibold text-sm">Step 2: Quality Control</h4>
                                <p className="text-xs font-mono text-muted-foreground mb-1">1_spectogramClustering.py</p>
                                <p className="text-sm text-muted-foreground">
                                    To handle 288k+ images, we used a ResNet18 model to extract features and K-Means clustering to group similar images, allowing us to bulk-delete clusters of "junk" (silence, static, human speech).
                                </p>
                            </div>
                            <div className="relative pl-8">
                                <div className="absolute -left-[9px] top-1 h-4 w-4 rounded-full bg-primary" />
                                <h4 className="font-semibold text-sm">Step 3: Class Balancing</h4>
                                <p className="text-xs font-mono text-muted-foreground mb-1">2_datasetBalancer.py</p>
                                <p className="text-sm text-muted-foreground">
                                    The cleaned dataset was filtered to the top 13 species. Majority classes were downsampled and minority classes upsampled to achieve a perfect balance of 20,000 samples per class.
                                </p>
                            </div>
                            <div className="relative pl-8">
                                <div className="absolute -left-[9px] top-1 h-4 w-4 rounded-full bg-primary" />
                                <h4 className="font-semibold text-sm">Step 4: Optimization</h4>
                                <p className="text-xs font-mono text-muted-foreground mb-1">3_pngCompressor.py</p>
                                <p className="text-sm text-muted-foreground">
                                    Lossless PNGs were converted to high-quality JPGs to reduce dataset size from over 100GB to 15.4GB, optimizing I/O throughput during training.
                                </p>
                            </div>
                            <div className="relative pl-8">
                                <div className="absolute -left-[9px] top-1 h-4 w-4 rounded-full bg-primary" />
                                <h4 className="font-semibold text-sm">Step 5: Model Training</h4>
                                <p className="text-xs font-mono text-muted-foreground mb-1">4_pytorch_training_highres.py</p>
                                <p className="text-sm text-muted-foreground">
                                    The optimized dataset was fed into an EfficientNet-B4 model. Training utilized a custom loop with Mixup and SpecAugment regularization to prevent overfitting.
                                </p>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card className="md:col-span-2">
                    <CardHeader>
                        <CardTitle>Confusion Matrix</CardTitle>
                        <CardDescription>
                            Visual representation of the model's classification performance on the test set.
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="grid gap-8 md:grid-cols-3 items-center">
                            <div className="bg-muted/30 p-4 rounded-lg text-sm space-y-2 md:col-span-1 order-2 md:order-1">
                                <h4 className="font-semibold flex items-center gap-2">
                                    <span className="w-2 h-2 rounded-full bg-primary" />
                                    Interpretation
                                </h4>
                                <p className="text-muted-foreground">
                                    The strong diagonal line indicates correct predictions. Off-diagonal squares represent misclassifications.
                                </p>
                                <p className="text-muted-foreground">
                                    The matrix is extremely clean, though there is slight confusion (16 misclassifications) between the
                                    <strong> Black-capped Chickadee</strong> and <strong>Tufted Titmouse</strong>, likely due to their similar calls and shared habitats.
                                </p>
                            </div>
                            <div className="md:col-span-2 order-1 md:order-2 flex justify-center">
                                <img
                                    src={confusionMatrixImage}
                                    alt="Confusion Matrix"
                                    className="w-full rounded-lg border shadow-sm"
                                />
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>

            <Card>
                <CardHeader>
                    <CardTitle>Performance Metrics</CardTitle>
                    <CardDescription>
                        Detailed breakdown of precision, recall, and F1-score for each species.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Species</TableHead>
                                <TableHead className="text-right">Precision</TableHead>
                                <TableHead className="text-right">Recall</TableHead>
                                <TableHead className="text-right">F1-Score</TableHead>
                                <TableHead className="text-right">Support</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {performanceData.map((item) => (
                                <TableRow key={item.species}>
                                    <TableCell className="font-medium">{item.species}</TableCell>
                                    <TableCell className="text-right">{item.precision.toFixed(4)}</TableCell>
                                    <TableCell className="text-right">{item.recall.toFixed(4)}</TableCell>
                                    <TableCell className="text-right">{item.f1.toFixed(4)}</TableCell>
                                    <TableCell className="text-right">{item.support}</TableCell>
                                </TableRow>
                            ))}
                            <TableRow className="font-bold bg-muted/50">
                                <TableCell>Accuracy</TableCell>
                                <TableCell className="text-right" colSpan={4}>0.9832 (26000 samples)</TableCell>
                            </TableRow>
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Technical Implementation</CardTitle>
                    <CardDescription>
                        Details on the audio processing pipeline and machine learning model architecture.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="grid md:grid-cols-2 gap-6">
                        <div className="space-y-4">
                            <h3 className="font-semibold text-lg border-b pb-2">Audio Preprocessing</h3>
                            <ul className="space-y-2 text-sm text-muted-foreground">
                                <li><span className="font-medium text-foreground">Sample Rate:</span> 32,000 Hz</li>
                                <li><span className="font-medium text-foreground">Spectrogram:</span> Mel-scale, 384 bands (High Res)</li>
                                <li><span className="font-medium text-foreground">Input Shape:</span> 384 x 384 x 3 (1:1 Aspect Ratio)</li>
                                <li><span className="font-medium text-foreground">Hop Length:</span> 250 samples</li>
                                <li><span className="font-medium text-foreground">Normalization:</span> PCEN (Per-Channel Energy Normalization)</li>
                                <li><span className="font-medium text-foreground">Features:</span> Stacked PCEN + Delta + Delta-Delta</li>
                            </ul>
                        </div>

                        <div className="space-y-4">
                            <h3 className="font-semibold text-lg border-b pb-2">Model Architecture</h3>
                            <ul className="space-y-2 text-sm text-muted-foreground">
                                <li><span className="font-medium text-foreground">Backbone:</span> EfficientNet-B4 (Pretrained)</li>
                                <li><span className="font-medium text-foreground">Optimizer:</span> AdamW (LR: 0.0005)</li>
                                <li><span className="font-medium text-foreground">Loss Function:</span> CrossEntropy with Label Smoothing (0.1)</li>
                                <li><span className="font-medium text-foreground">Augmentation:</span> Mixup (α=0.4), Gaussian Noise, SpecAugment</li>
                                <li><span className="font-medium text-foreground">Training:</span> 50 Epochs, Batch Size 8, GPU Accelerated</li>
                            </ul>
                        </div>
                    </div>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Future Improvements</CardTitle>
                    <CardDescription>
                        Planned enhancements to increase accuracy and user experience.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="grid sm:grid-cols-2 gap-4">
                        <div className="border p-4 rounded-lg">
                            <h4 className="font-semibold mb-2">Geospatial Context</h4>
                            <p className="text-sm text-muted-foreground">
                                Integrating GPS coordinates and time-of-year metadata to filter out unlikely species (e.g., migratory birds not present in winter), reducing false positives.
                            </p>
                        </div>
                        <div className="border p-4 rounded-lg">
                            <h4 className="font-semibold mb-2">Sliding Window Inference</h4>
                            <p className="text-sm text-muted-foreground">
                                Implementing a sliding window approach to analyze overlapping 3-second segments, enabling continuous detection without manual trimming.
                            </p>
                        </div>
                        <div className="border p-4 rounded-lg">
                            <h4 className="font-semibold mb-2">Multi-Label Classification</h4>
                            <p className="text-sm text-muted-foreground">
                                Transitioning from single-class to multi-label output to identify multiple birds singing simultaneously in the same recording.
                            </p>
                        </div>
                        <div className="border p-4 rounded-lg">
                            <h4 className="font-semibold mb-2">Improved Segmentation</h4>
                            <p className="text-sm text-muted-foreground">
                                Enhancing onset detection to more precisely identify the start and end of bird calls, reducing manual data preparation effort.
                            </p>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </motion.div>
    );
}
