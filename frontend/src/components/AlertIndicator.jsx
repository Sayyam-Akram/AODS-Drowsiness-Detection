import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, UserX, Loader2, Eye, Image } from 'lucide-react';
import { pulseVariants } from '../utils/constants';

const AlertIndicator = ({
    isDrowsy,
    confidence = 0,
    faceDetected = true,
    isLoading = false,
    earValue = null,
    approach = 'approach1'
}) => {
    if (isLoading) {
        return (
            <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-4 rounded-2xl bg-white/10 p-6 border border-white/20"
            >
                <Loader2 className="w-12 h-12 text-white/70 animate-spin" />
                <div>
                    <h3 className="text-xl font-bold text-white">Analyzing...</h3>
                    <p className="text-white/60">Processing frame</p>
                </div>
            </motion.div>
        );
    }

    if (!faceDetected) {
        // Show approach-specific message for no face detected
        const isEAR = approach === 'approach3';

        return (
            <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-4 rounded-2xl bg-warning-500/20 p-6 border border-warning-500/30"
            >
                {isEAR ? <UserX className="w-12 h-12 text-warning-400" /> : <Eye className="w-12 h-12 text-warning-400" />}
                <div>
                    <h3 className="text-xl font-bold text-warning-400">
                        {isEAR ? 'No Face Detected' : 'No Eyes Detected'}
                    </h3>
                    <p className="text-warning-400/70 text-sm">
                        {isEAR
                            ? '💡 EAR needs a FULL FACE image to detect eye landmarks'
                            : '💡 Position eyes clearly in frame, or upload a full face image'
                        }
                    </p>
                    {isEAR && (
                        <p className="text-warning-400/50 text-xs mt-1">
                            Tip: For eye-only images, use Custom CNN or EfficientNet
                        </p>
                    )}
                </div>
            </motion.div>
        );
    }

    if (isDrowsy) {
        return (
            <motion.div
                variants={pulseVariants}
                initial="initial"
                animate="animate"
                className="flex items-center gap-4 rounded-2xl bg-gradient-to-r from-danger-500 to-danger-600 p-6 shadow-2xl shadow-danger-500/30 animate-pulse-glow"
            >
                <AlertTriangle className="w-12 h-12 text-white" />
                <div className="flex-1">
                    <h3 className="text-2xl font-bold text-white flex items-center gap-2">
                        ⚠️ DROWSINESS DETECTED
                    </h3>
                    <div className="flex items-center gap-4 mt-1">
                        <p className="text-white/80">Confidence: {(confidence * 100).toFixed(1)}%</p>
                        {earValue !== null && (
                            <p className="text-white/80">EAR: {earValue.toFixed(3)}</p>
                        )}
                    </div>
                </div>

                {/* Visual alert indicator */}
                <motion.div
                    animate={{ scale: [1, 1.3, 1], opacity: [1, 0.7, 1] }}
                    transition={{ duration: 0.5, repeat: Infinity }}
                    className="w-4 h-4 rounded-full bg-white"
                />
            </motion.div>
        );
    }

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-4 rounded-2xl bg-gradient-to-r from-success-500 to-success-600 p-6 shadow-xl shadow-success-500/20"
        >
            <CheckCircle className="w-12 h-12 text-white" />
            <div className="flex-1">
                <h3 className="text-2xl font-bold text-white">✅ DRIVER ALERT</h3>
                <div className="flex items-center gap-4 mt-1">
                    <p className="text-white/80">Confidence: {(confidence * 100).toFixed(1)}%</p>
                    {earValue !== null && (
                        <p className="text-white/80">EAR: {earValue.toFixed(3)}</p>
                    )}
                </div>
            </div>
        </motion.div>
    );
};

export default AlertIndicator;
