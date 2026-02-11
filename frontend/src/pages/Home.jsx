import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
    Play, BarChart3, Upload, Brain, Sparkles, Eye,
    CheckCircle, XCircle, Cpu, ArrowRight
} from 'lucide-react';
import { pageVariants, APPROACHES } from '../utils/constants';

const Home = ({ apiStatus, metrics }) => {
    const features = [
        {
            icon: Brain,
            title: 'Custom CNN',
            description: 'Deep convolutional neural network trained on 84K+ infrared eye images',
            color: 'from-blue-500 to-blue-600',
        },
        {
            icon: Sparkles,
            title: 'Transfer Learning',
            description: 'EfficientNetB1 fine-tuned for optimal drowsiness detection',
            color: 'from-purple-500 to-purple-600',
        },
        {
            icon: Eye,
            title: 'EAR Detection',
            description: 'Real-time Eye Aspect Ratio monitoring with MediaPipe',
            color: 'from-green-500 to-green-600',
        },
    ];

    return (
        <motion.div
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="space-y-12"
        >
            {/* Hero Section */}
            <section className="text-center py-16">
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                >
                    <h1 className="text-5xl md:text-7xl font-bold mb-6">
                        <span className="gradient-text">AI-Powered</span>
                        <br />
                        <span className="text-white">Drowsiness Detection</span>
                    </h1>

                    <p className="text-xl text-white/60 max-w-2xl mx-auto mb-8">
                        Advanced driver safety system using deep learning with three detection approaches
                        for real-time alertness monitoring.
                    </p>

                    <div className="flex flex-wrap justify-center gap-4">
                        <Link to="/detection" className="btn-primary flex items-center gap-2">
                            <Play className="w-5 h-5" />
                            Start Detection
                        </Link>
                        <Link to="/comparison" className="btn-secondary flex items-center gap-2">
                            <BarChart3 className="w-5 h-5" />
                            View Comparison
                        </Link>
                    </div>
                </motion.div>

                {/* Status Cards */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    className="flex flex-wrap justify-center gap-4 mt-12"
                >
                    {/* API Status */}
                    <div className={`
            flex items-center gap-3 px-5 py-3 rounded-xl border
            ${apiStatus?.status === 'healthy'
                            ? 'bg-success-500/10 border-success-500/30'
                            : 'bg-danger-500/10 border-danger-500/30'
                        }
          `}>
                        {apiStatus?.status === 'healthy' ? (
                            <>
                                <CheckCircle className="w-5 h-5 text-success-400" />
                                <span className="text-success-400 font-medium">Backend Connected</span>
                            </>
                        ) : (
                            <>
                                <XCircle className="w-5 h-5 text-danger-400" />
                                <span className="text-danger-400 font-medium">Backend Offline</span>
                            </>
                        )}
                    </div>

                    {/* GPU Status */}
                    {apiStatus?.gpu_available !== undefined && (
                        <div className={`
              flex items-center gap-3 px-5 py-3 rounded-xl border
              ${apiStatus.gpu_available
                                ? 'bg-accent-500/10 border-accent-500/30'
                                : 'bg-warning-500/10 border-warning-500/30'
                            }
            `}>
                            <Cpu className="w-5 h-5 text-accent-400" />
                            <span className={apiStatus.gpu_available ? 'text-accent-400' : 'text-warning-400'}>
                                GPU {apiStatus.gpu_available ? 'Enabled' : 'Not Available'}
                            </span>
                        </div>
                    )}

                    {/* Models Loaded */}
                    {apiStatus?.models_loaded && (
                        <div className="flex items-center gap-3 px-5 py-3 rounded-xl border bg-primary-500/10 border-primary-500/30">
                            <Brain className="w-5 h-5 text-primary-400" />
                            <span className="text-primary-400 font-medium">
                                {apiStatus.models_loaded.length} Models Loaded
                            </span>
                        </div>
                    )}
                </motion.div>
            </section>

            {/* Features Section */}
            <section>
                <h2 className="text-3xl font-bold text-center text-white mb-8">
                    Three Detection Approaches
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {features.map((feature, idx) => {
                        const Icon = feature.icon;
                        const approachKey = `approach${idx + 1}`;
                        const approachMetrics = metrics?.[approachKey];

                        return (
                            <motion.div
                                key={feature.title}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: idx * 0.1 }}
                                whileHover={{ y: -5 }}
                                className="glass-card-hover p-6"
                            >
                                <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4`}>
                                    <Icon className="w-7 h-7 text-white" />
                                </div>

                                <h3 className="text-xl font-semibold text-white mb-2">
                                    {feature.title}
                                </h3>

                                <p className="text-white/60 mb-4">
                                    {feature.description}
                                </p>

                                {approachMetrics && approachMetrics.accuracy > 0 && (
                                    <div className="pt-4 border-t border-white/10">
                                        <div className="flex justify-between text-sm">
                                            <span className="text-white/50">Accuracy</span>
                                            <span className="text-success-400 font-semibold">
                                                {(approachMetrics.accuracy * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    </div>
                                )}
                            </motion.div>
                        );
                    })}
                </div>
            </section>

            {/* Detection Modes */}
            <section>
                <h2 className="text-3xl font-bold text-center text-white mb-8">
                    Detection Modes
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-2xl mx-auto">
                    <motion.div
                        whileHover={{ scale: 1.02 }}
                        className="glass-card p-6 text-center"
                    >
                        <Upload className="w-12 h-12 text-accent-400 mx-auto mb-4" />
                        <h3 className="text-lg font-semibold text-white mb-2">Image Upload</h3>
                        <p className="text-white/60 text-sm">
                            Analyze single photos with eye detection and status visualization
                        </p>
                    </motion.div>

                    <motion.div
                        whileHover={{ scale: 1.02 }}
                        className="glass-card p-6 text-center"
                    >
                        <Play className="w-12 h-12 text-success-400 mx-auto mb-4" />
                        <h3 className="text-lg font-semibold text-white mb-2">Video Processing</h3>
                        <p className="text-white/60 text-sm">
                            Process video files with annotated playback and drowsiness stats
                        </p>
                    </motion.div>
                </div>
            </section>

            {/* CTA */}
            <section className="text-center py-8">
                <Link
                    to="/detection"
                    className="inline-flex items-center gap-2 text-lg font-medium text-primary-400 hover:text-primary-300 transition-colors"
                >
                    Get Started with Detection
                    <ArrowRight className="w-5 h-5" />
                </Link>
            </section>
        </motion.div>
    );
};

export default Home;
