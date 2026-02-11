import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Upload, Video, X, AlertCircle, Play } from 'lucide-react';

const VideoUploader = ({ onVideoSelect, disabled = false }) => {
    const [preview, setPreview] = useState(null);
    const [fileName, setFileName] = useState(null);
    const [error, setError] = useState(null);
    const [isDragging, setIsDragging] = useState(false);

    const handleFile = useCallback((file) => {
        setError(null);

        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('video/')) {
            setError('Please select a video file');
            return;
        }

        // Validate file size (max 100MB)
        if (file.size > 100 * 1024 * 1024) {
            setError('Video size must be less than 100MB');
            return;
        }

        // Create preview URL
        const url = URL.createObjectURL(file);
        setPreview(url);
        setFileName(file.name);

        // Pass file to parent
        onVideoSelect(file);
    }, [onVideoSelect]);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        setIsDragging(false);

        const file = e.dataTransfer.files[0];
        handleFile(file);
    }, [handleFile]);

    const handleDragOver = useCallback((e) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const handleInputChange = useCallback((e) => {
        const file = e.target.files[0];
        handleFile(file);
    }, [handleFile]);

    const clearVideo = useCallback(() => {
        if (preview) {
            URL.revokeObjectURL(preview);
        }
        setPreview(null);
        setFileName(null);
        setError(null);
        onVideoSelect(null);
    }, [preview, onVideoSelect]);

    return (
        <div className="w-full">
            <label className="block text-sm font-medium text-white/70 mb-3">
                Upload Video
            </label>

            {preview ? (
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="relative rounded-xl overflow-hidden border-2 border-white/20"
                >
                    <video
                        src={preview}
                        className="w-full h-64 object-contain bg-black/50"
                        controls
                    />

                    <div className="absolute top-3 left-3 px-3 py-1 rounded-full bg-black/70 text-white text-sm flex items-center gap-2">
                        <Play className="w-3 h-3" />
                        {fileName}
                    </div>

                    <button
                        onClick={clearVideo}
                        disabled={disabled}
                        className="absolute top-3 right-3 p-2 rounded-full bg-black/50 hover:bg-danger-500 transition-colors"
                    >
                        <X className="w-4 h-4 text-white" />
                    </button>
                </motion.div>
            ) : (
                <motion.div
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    whileHover={{ scale: disabled ? 1 : 1.01 }}
                    className={`
            relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300
            ${isDragging
                            ? 'border-accent-500 bg-accent-500/10'
                            : 'border-white/20 hover:border-white/40'
                        }
            ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          `}
                >
                    <input
                        type="file"
                        accept="video/*"
                        onChange={handleInputChange}
                        disabled={disabled}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                    />

                    <div className="flex flex-col items-center gap-4">
                        <div className={`
              w-16 h-16 rounded-full flex items-center justify-center
              ${isDragging ? 'bg-accent-500/20' : 'bg-white/10'}
            `}>
                            {isDragging ? (
                                <Upload className="w-8 h-8 text-accent-400" />
                            ) : (
                                <Video className="w-8 h-8 text-white/50" />
                            )}
                        </div>

                        <div>
                            <p className="text-white/80 font-medium">
                                {isDragging ? 'Drop your video here' : 'Drag & drop a video'}
                            </p>
                            <p className="text-white/50 text-sm mt-1">
                                or click to browse (MP4, WebM, max 100MB)
                            </p>
                        </div>
                    </div>
                </motion.div>
            )}

            {error && (
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-3 flex items-center gap-2 text-danger-400 text-sm"
                >
                    <AlertCircle className="w-4 h-4" />
                    <span>{error}</span>
                </motion.div>
            )}
        </div>
    );
};

export default VideoUploader;
