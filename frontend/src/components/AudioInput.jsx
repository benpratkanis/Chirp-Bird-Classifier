import React, { useState, useRef } from 'react';
import { Upload, Mic, Square, FileAudio } from 'lucide-react';

const AudioInput = ({ onAudioSelected }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('audio/')) {
            onAudioSelected(file);
        }
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            onAudioSelected(file);
        }
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorderRef.current = new MediaRecorder(stream);
            chunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                }
            };

            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
                const file = new File([blob], "recording.webm", { type: 'audio/webm' });
                onAudioSelected(file);
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorderRef.current.start();
            setIsRecording(true);
        } catch (err) {
            console.error("Error accessing microphone:", err);
            alert("Could not access microphone. Please check permissions.");
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    return (
        <div className="w-full max-w-2xl mx-auto p-6">
            <div
                className={`border-4 border-dashed rounded-2xl p-10 text-center transition-all duration-300 ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
                    }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
            >
                <div className="flex flex-col items-center justify-center space-y-4">
                    <div className="p-4 bg-blue-100 rounded-full text-blue-600">
                        <Upload size={48} />
                    </div>
                    <h3 className="text-xl font-semibold text-gray-700">
                        Drag & Drop Audio File
                    </h3>
                    <p className="text-gray-500">or</p>

                    <div className="flex space-x-4">
                        <label className="px-6 py-2 bg-blue-600 text-white rounded-lg cursor-pointer hover:bg-blue-700 transition-colors flex items-center space-x-2">
                            <FileAudio size={20} />
                            <span>Browse Files</span>
                            <input
                                type="file"
                                accept="audio/*"
                                className="hidden"
                                onChange={handleFileChange}
                            />
                        </label>

                        {!isRecording ? (
                            <button
                                onClick={startRecording}
                                className="px-6 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors flex items-center space-x-2"
                            >
                                <Mic size={20} />
                                <span>Record</span>
                            </button>
                        ) : (
                            <button
                                onClick={stopRecording}
                                className="px-6 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-900 transition-colors flex items-center space-x-2 animate-pulse"
                            >
                                <Square size={20} />
                                <span>Stop Recording</span>
                            </button>
                        )}
                    </div>

                    <p className="text-sm text-gray-400 mt-4">
                        Supports MP3, WAV, OGG, M4A
                    </p>
                </div>
            </div>
        </div>
    );
};

export default AudioInput;
