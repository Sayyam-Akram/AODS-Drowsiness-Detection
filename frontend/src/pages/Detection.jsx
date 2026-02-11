import { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Image, Video, RefreshCw, Loader2 } from 'lucide-react';
import { pageVariants } from '../utils/constants';
import { predictImage, processVideo, resetState, getAlertSoundUrl } from '../utils/api';

import ModelSelector from '../components/ModelSelector';
import ImageUploader from '../components/ImageUploader';
import VideoUploader from '../components/VideoUploader';
import AlertIndicator from '../components/AlertIndicator';
import ResultsDisplay from '../components/ResultsDisplay';
import LoadingSpinner from '../components/LoadingSpinner';

const Detection = ({ metrics }) => {
    const [mode, setMode] = useState('image'); // image or video
    const [approach, setApproach] = useState('approach1');
    const [isProcessing, setIsProcessing] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [videoResults, setVideoResults] = useState(null);

    const audioRef = useRef(null);
    const lastDrowsyRef = useRef(false);

    // Initialize audio
    useEffect(() => {
        audioRef.current = new Audio(getAlertSoundUrl());
        audioRef.current.volume = 0.7;
        return () => {
            if (audioRef.current) {
                audioRef.current.pause();
            }
        };
    }, []);

    // Play alert when drowsy
    useEffect(() => {
        if (result?.is_drowsy && !lastDrowsyRef.current) {
            audioRef.current?.play().catch(console.error);
        }
        lastDrowsyRef.current = result?.is_drowsy || false;
    }, [result?.is_drowsy]);

    // Handle image upload
    const handleImageSelect = useCallback(async (file) => {
        if (!file) {
            setResult(null);
            return;
        }

        try {
            setIsProcessing(true);
            setError(null);
            const response = await predictImage(file, approach);
            setResult(response);
        } catch (err) {
            console.error('Prediction error:', err);
            setError(err.message);
        } finally {
            setIsProcessing(false);
        }
    }, [approach]);

    // Handle video upload
    const handleVideoSelect = useCallback(async (file) => {
        if (!file) {
            setVideoResults(null);
            return;
        }

        try {
            setIsProcessing(true);
            setError(null);
            setVideoResults(null);
            const response = await processVideo(file, approach);
            setVideoResults(response);
        } catch (err) {
            console.error('Video processing error:', err);
            setError(err.message);
        } finally {
            setIsProcessing(false);
        }
    }, [approach]);

    // Reset everything
    const handleReset = useCallback(() => {
        setResult(null);
        setVideoResults(null);
        setError(null);
        resetState().catch(console.error);
    }, []);

    const modes = [
        { id: 'image', label: 'Image', icon: Image },
        { id: 'video', label: 'Video', icon: Video },
    ];

    return (
        <motion.div
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="max-w-5xl mx-auto space-y-8"
        >
            {/* Header */}
            <div className="text-center">
                <h1 className="text-4xl font-bold text-white mb-2">Drowsiness Detection</h1>
                <p className="text-white/60">Upload an image or video to analyze</p>
            </div>

            {/* Mode Selector */}
            <div className="flex justify-center gap-2">
                {modes.map(({ id, label, icon: Icon }) => (
                    <button
                        key={id}
                        onClick={() => {
                            setMode(id);
                            handleReset();
                        }}
                        className={`
                            flex items-center gap-2 px-5 py-3 rounded-xl font-medium transition-all
                            ${mode === id
                                ? 'bg-gradient-to-r from-primary-500 to-accent-500 text-white shadow-lg'
                                : 'bg-white/5 text-white/70 hover:bg-white/10 hover:text-white'
                            }
                        `}
                    >
                        <Icon className="w-5 h-5" />
                        {label}
                    </button>
                ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Left Panel - Input */}
                <div className="space-y-6">
                    <div className="glass-card p-6 space-y-6">
                        {/* Model Selector */}
                        <ModelSelector
                            selectedApproach={approach}
                            onSelect={setApproach}
                            disabled={isProcessing}
                        />

                        {/* Image Mode */}
                        {mode === 'image' && (
                            <div className="space-y-4">
                                <ImageUploader
                                    onImageSelect={handleImageSelect}
                                    disabled={isProcessing}
                                />
                                {isProcessing && (
                                    <div className="text-center py-4">
                                        <LoadingSpinner text="Analyzing image..." />
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Video Mode */}
                        {mode === 'video' && (
                            <div className="space-y-4">
                                <VideoUploader
                                    onVideoSelect={handleVideoSelect}
                                    disabled={isProcessing}
                                />
                                {isProcessing && (
                                    <div className="text-center py-8">
                                        <LoadingSpinner size="lg" text="Processing video frames..." />
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Reset Button */}
                        {(result || videoResults || error) && (
                            <button
                                onClick={handleReset}
                                className="btn-secondary w-full flex items-center justify-center gap-2"
                            >
                                <RefreshCw className="w-5 h-5" />
                                Reset
                            </button>
                        )}
                    </div>
                </div>

                {/* Right Panel - Results */}
                <div className="space-y-6">
                    <AnimatePresence mode="wait">
                        {/* Alert Indicator for Image */}
                        {mode === 'image' && (
                            <AlertIndicator
                                isDrowsy={result?.is_drowsy}
                                confidence={result?.confidence}
                                approach={approach}
                            />
                        )}

                        {/* Image Results */}
                        {mode === 'image' && result && (
                            <ResultsDisplay
                                result={result}
                                annotatedImage={result.annotated_image}
                            />
                        )}

                        {/* Video Results */}
                        {videoResults && (
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="glass-card p-6 space-y-4"
                            >
                                <h3 className="text-xl font-semibold text-white">Video Analysis Results</h3>

                                {/* Video Player */}
                                {videoResults.video_data && (
                                    <div className="relative rounded-xl overflow-hidden bg-black">
                                        <video
                                            src={videoResults.video_data}
                                            controls
                                            autoPlay
                                            className="w-full"
                                            onTimeUpdate={(e) => {
                                                const currentTime = e.target.currentTime;
                                                const drowsyTimes = videoResults.drowsy_timestamps || [];
                                                const nearDrowsy = drowsyTimes.some(
                                                    t => Math.abs(t - currentTime) < 0.2 &&
                                                        !e.target.dataset[`played_${Math.floor(t)}`]
                                                );
                                                if (nearDrowsy && audioRef.current) {
                                                    audioRef.current.currentTime = 0;
                                                    audioRef.current.play().catch(() => { });
                                                    drowsyTimes.forEach(t => {
                                                        if (Math.abs(t - currentTime) < 0.2) {
                                                            e.target.dataset[`played_${Math.floor(t)}`] = true;
                                                        }
                                                    });
                                                }
                                            }}
                                        />
                                    </div>
                                )}

                                {/* Key Frames Fallback */}
                                {!videoResults.video_data && videoResults.key_frames?.length > 0 && (
                                    <div className="relative rounded-xl overflow-hidden bg-black">
                                        <img
                                            src={`data:image/jpeg;base64,${videoResults.key_frames[0]}`}
                                            alt="Key frame"
                                            className="w-full"
                                        />
                                        <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/60 rounded text-white text-xs">
                                            {videoResults.key_frames.length} key frames captured
                                        </div>
                                    </div>
                                )}

                                {/* Stats Grid */}
                                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                    <div className="metric-card text-center">
                                        <p className="text-white/50 text-xs">Duration</p>
                                        <p className="text-lg font-bold text-white">
                                            {videoResults.duration_seconds?.toFixed(1)}s
                                        </p>
                                    </div>
                                    <div className="metric-card text-center">
                                        <p className="text-white/50 text-xs">Frames</p>
                                        <p className="text-lg font-bold text-white">{videoResults.total_frames}</p>
                                    </div>
                                    <div className="metric-card text-center">
                                        <p className="text-white/50 text-xs">Drowsy</p>
                                        <p className="text-lg font-bold text-danger-400">{videoResults.drowsy_frames}</p>
                                    </div>
                                    <div className="metric-card text-center">
                                        <p className="text-white/50 text-xs">Drowsiness</p>
                                        <p className={`text-lg font-bold ${videoResults.drowsy_percentage > 30 ? 'text-danger-400' : 'text-success-400'}`}>
                                            {videoResults.drowsy_percentage?.toFixed(1)}%
                                        </p>
                                    </div>
                                </div>

                                {videoResults.drowsy_percentage > 30 && (
                                    <div className="p-3 rounded-xl bg-danger-500/20 border border-danger-500/30">
                                        <p className="text-danger-400 font-medium text-sm">
                                            ⚠️ High drowsiness detected! Consider taking a break.
                                        </p>
                                    </div>
                                )}
                            </motion.div>
                        )}

                        {/* Error Display */}
                        {error && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="glass-card p-6 border-danger-500/30"
                            >
                                <p className="text-danger-400">❌ {error}</p>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </motion.div>
    );
};

export default Detection;
