import React, { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';
import { Play, Pause, Scissors } from 'lucide-react';

const WaveformEditor = ({ audioFile, onTrimmed }) => {
    const containerRef = useRef(null);
    const wavesurferRef = useRef(null);
    const regionsRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [duration, setDuration] = useState(0);

    useEffect(() => {
        if (!audioFile || !containerRef.current) return;

        const ws = WaveSurfer.create({
            container: containerRef.current,
            waveColor: '#A8DBA8',
            progressColor: '#3B8686',
            cursorColor: '#0B486B',
            barWidth: 2,
            barRadius: 3,
            responsive: true,
            height: 128,
        });

        const wsRegions = ws.registerPlugin(RegionsPlugin.create());
        regionsRef.current = wsRegions;
        wavesurferRef.current = ws;

        ws.loadBlob(audioFile);

        ws.on('ready', () => {
            setDuration(ws.getDuration());
            // Add a default region covering the whole track or first 5s
            wsRegions.addRegion({
                start: 0,
                end: Math.min(5, ws.getDuration()),
                color: 'rgba(0, 0, 255, 0.1)',
                drag: true,
                resize: true,
            });
        });

        ws.on('play', () => setIsPlaying(true));
        ws.on('pause', () => setIsPlaying(false));

        return () => {
            ws.destroy();
        };
    }, [audioFile]);

    const handlePlayPause = () => {
        if (wavesurferRef.current) {
            wavesurferRef.current.playPause();
        }
    };

    const handleTrimAndAnalyze = async () => {
        if (!wavesurferRef.current || !regionsRef.current) return;

        const regions = regionsRef.current.getRegions();
        let start = 0;
        let end = duration;

        if (regions.length > 0) {
            start = regions[0].start;
            end = regions[0].end;
        }

        // We need to slice the audio file. 
        // Since we can't easily slice the Blob directly with precise audio headers without re-encoding in browser (complex),
        // we will send the original file + start/end times to the backend?
        // OR we can try to slice it if it's simple.
        // BUT the requirement says "allow the user to trim it".
        // The backend `predict` endpoint takes a file.
        // Ideally we send the whole file and the trim parameters.
        // However, `modelEvaluationManual.py` logic (ported to `inference.py`) takes raw bytes and processes them.
        // It has a "Enforce 5 Seconds" step (crop or pad).

        // If we want the USER to select the 5 seconds, we should send the start/end to the backend.
        // But `inference.py` currently just loads the file and takes the first 5s (or pads).
        // I should modify `inference.py` or `main.py` to accept start/end times.

        // For now, let's assume we send the whole file and let the backend handle it?
        // No, the user wants to trim it.
        // I will modify the backend to accept `start_time` and `end_time` query params or form data.

        // Wait, `inference.py` uses `librosa.load`. Librosa can load with offset and duration.
        // `librosa.load(path, offset=start, duration=duration)`

        // So I should pass start/end to the backend.

        onTrimmed(audioFile, start, end - start);
    };

    return (
        <div className="w-full max-w-2xl mx-auto p-6 bg-white rounded-2xl shadow-lg mt-6">
            <h3 className="text-lg font-semibold mb-4 text-gray-700">Trim Audio (Select 5s)</h3>

            <div ref={containerRef} className="mb-4" />

            <div className="flex justify-between items-center">
                <button
                    onClick={handlePlayPause}
                    className="px-4 py-2 bg-gray-200 rounded-lg hover:bg-gray-300 transition-colors flex items-center space-x-2"
                >
                    {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                    <span>{isPlaying ? 'Pause' : 'Play'}</span>
                </button>

                <button
                    onClick={handleTrimAndAnalyze}
                    className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center space-x-2"
                >
                    <Scissors size={20} />
                    <span>Analyze Selection</span>
                </button>
            </div>

            <p className="text-xs text-gray-400 mt-2 text-center">
                Drag the blue region to select the segment to analyze.
            </p>
        </div>
    );
};

export default WaveformEditor;
