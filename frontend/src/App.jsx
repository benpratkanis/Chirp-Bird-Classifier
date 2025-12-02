import React, { useState } from 'react';
import axios from 'axios';
import { Bird, Loader2, ArrowLeft, Upload, Mic } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { AudioUploader } from './components/AudioUploader';
import { AudioRecorder } from './components/AudioRecorder';
import { AudioEditor } from './components/AudioEditor';
import { ClassificationResult } from './components/ClassificationResult';
import { SpeciesCarousel } from './components/SpeciesCarousel';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Toaster } from '@/components/ui/toaster';
import { useToast } from '@/hooks/use-toast';
import backgroundPattern from '@/assets/background.png';
import { speciesImages } from '@/lib/speciesImages';

function App() {
  const [file, setFile] = useState(null);
  const [trimRegion, setTrimRegion] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [allPredictions, setAllPredictions] = useState([]);
  const [spectrogram, setSpectrogram] = useState(null);
  const { toast } = useToast();

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setResult(null);
    setAllPredictions([]);
    setSpectrogram(null);
    setTrimRegion(null);
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setAllPredictions([]);
    setSpectrogram(null);
    setTrimRegion(null);
  };

  const handleSubmit = async () => {
    if (!file || !trimRegion) return;

    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('start', trimRegion.start);
      formData.append('duration', trimRegion.end - trimRegion.start);

      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = response.data;

      if (data.predictions && data.predictions.length > 0) {
        const topPrediction = data.predictions[0];
        setResult({
          species: topPrediction.species,
          confidence: topPrediction.probability
        });
        setAllPredictions(data.predictions);
        setSpectrogram(data.spectrogram);

        toast({
          title: "Analysis Complete",
          description: `Identified: ${topPrediction.species} (${(topPrediction.probability * 100).toFixed(1)}% confidence)`,
        });
      } else {
        throw new Error('No predictions returned');
      }

    } catch (error) {
      console.error('Classification error:', error);
      toast({
        title: "Classification Failed",
        description: error.message || "Could not analyze the audio. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen w-full relative bg-background text-foreground">
      {/* Background Pattern Overlay */}
      <div
        className="fixed inset-0 pointer-events-none opacity-40 mix-blend-multiply z-0"
        style={{
          backgroundImage: `url(${backgroundPattern})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center'
        }}
      />

      <div className="relative z-10 max-w-5xl mx-auto px-4 py-12 md:py-20">
        <Toaster />

        <div className="text-center mb-16 space-y-4">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center justify-center p-3 bg-primary/10 rounded-full mb-4"
          >
            <Bird className="w-8 h-8 text-primary" />
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-4xl md:text-6xl font-serif font-bold text-foreground tracking-tight"
          >
            AvianID
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-lg text-muted-foreground max-w-xl mx-auto font-light"
          >
            Upload or record audio to identify bird species using advanced machine learning analysis.
          </motion.p>
        </div>

        <div className="space-y-8">
          <AnimatePresence mode="wait">
            {!file ? (
              <motion.div
                key="input-methods"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.3 }}
              >
                <Tabs defaultValue="upload" className="w-full max-w-2xl mx-auto">
                  <TabsList className="grid w-full grid-cols-2 mb-8">
                    <TabsTrigger value="upload" className="flex items-center gap-2">
                      <Upload className="w-4 h-4" />
                      Upload File
                    </TabsTrigger>
                    <TabsTrigger value="record" className="flex items-center gap-2">
                      <Mic className="w-4 h-4" />
                      Record Audio
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="upload" className="mt-0">
                    <AudioUploader onFileSelect={handleFileSelect} />
                  </TabsContent>

                  <TabsContent value="record" className="mt-0">
                    <AudioRecorder onRecordingComplete={handleFileSelect} />
                  </TabsContent>
                </Tabs>
              </motion.div>
            ) : !result ? (
              <motion.div
                key="edit"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-8"
              >
                <div className="flex items-center justify-between">
                  <Button
                    variant="ghost"
                    onClick={handleReset}
                    className="text-muted-foreground hover:text-foreground"
                    disabled={isProcessing}
                  >
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    Back to Selection
                  </Button>
                  <div className="text-sm font-medium text-primary bg-primary/10 px-3 py-1 rounded-full truncate max-w-[200px] md:max-w-xs">
                    {file.name}
                  </div>
                </div>

                <AudioEditor
                  file={file}
                  onTrimChange={(start, end) => setTrimRegion({ start, end })}
                />

                <div className="flex justify-center pt-4">
                  <Button
                    size="lg"
                    onClick={handleSubmit}
                    disabled={isProcessing}
                    className="w-full md:w-auto min-w-[200px] text-lg h-12 shadow-lg shadow-primary/20"
                    data-testid="button-identify"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        Analyzing Audio...
                      </>
                    ) : (
                      <>
                        Identify Species
                        <Bird className="w-5 h-5 ml-2" />
                      </>
                    )}
                  </Button>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="result"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-8"
              >
                <div className="flex justify-start">
                  <Button
                    variant="outline"
                    onClick={handleReset}
                    data-testid="button-analyze-another"
                  >
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    Analyze Another
                  </Button>
                </div>

                <ClassificationResult
                  species={result.species}
                  confidence={result.confidence}
                  allPredictions={allPredictions}
                  spectrogram={spectrogram}
                  speciesImage={speciesImages[result.species]}
                />
              </motion.div>
            )}
          </AnimatePresence>

          {!file && <SpeciesCarousel />}
        </div>
      </div>
    </div>
  );
}

export default App;
