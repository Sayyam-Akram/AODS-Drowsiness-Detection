import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Upload, Image, X, AlertCircle } from 'lucide-react';

const ImageUploader = ({ onImageSelect, disabled = false }) => {
    const [preview, setPreview] = useState(null);
    const [error, setError] = useState(null);
    const [isDragging, setIsDragging] = useState(false);

    const handleFile = useCallback((file) => {
        setError(null);

        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            setError('Please select an image file');
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            setError('Image size must be less than 10MB');
            return;
        }

        // Create preview
        const reader = new FileReader();
        reader.onloadend = () => {
            setPreview(reader.result);
        };
        reader.readAsDataURL(file);

        // Pass file to parent
        onImageSelect(file);
    }, [onImageSelect]);

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

    const clearImage = useCallback(() => {
        setPreview(null);
        setError(null);
        onImageSelect(null);
    }, [onImageSelect]);

    return (
        <div className="w-full">
            <label className="block text-sm font-medium text-white/70 mb-3">
                Upload Image
            </label>

            {preview ? (
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="relative rounded-xl overflow-hidden border-2 border-white/20"
                >
                    <img
                        src={preview}
                        alt="Preview"
                        className="w-full h-64 object-contain bg-black/50"
                    />

                    <button
                        onClick={clearImage}
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
                            ? 'border-primary-500 bg-primary-500/10'
                            : 'border-white/20 hover:border-white/40'
                        }
            ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          `}
                >
                    <input
                        type="file"
                        accept="image/*"
                        onChange={handleInputChange}
                        disabled={disabled}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                    />

                    <div className="flex flex-col items-center gap-4">
                        <div className={`
              w-16 h-16 rounded-full flex items-center justify-center
              ${isDragging ? 'bg-primary-500/20' : 'bg-white/10'}
            `}>
                            {isDragging ? (
                                <Upload className="w-8 h-8 text-primary-400" />
                            ) : (
                                <Image className="w-8 h-8 text-white/50" />
                            )}
                        </div>

                        <div>
                            <p className="text-white/80 font-medium">
                                {isDragging ? 'Drop your image here' : 'Drag & drop an image'}
                            </p>
                            <p className="text-white/50 text-sm mt-1">
                                or click to browse (PNG, JPG, max 10MB)
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

export default ImageUploader;
