import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { RefreshCw, Info, Brain, Sparkles, Eye } from 'lucide-react';
import { pageVariants } from '../utils/constants';
import { getMetrics } from '../utils/api';
import MetricsTable from '../components/MetricsTable';
import LoadingSpinner from '../components/LoadingSpinner';

const Comparison = ({ metrics: initialMetrics }) => {
    const [metrics, setMetrics] = useState(initialMetrics);
    const [loading, setLoading] = useState(!initialMetrics);
    const [error, setError] = useState(null);

    const fetchMetrics = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await getMetrics();
            if (response.success) {
                setMetrics(response.summary);
            } else {
                setError('Failed to load metrics');
            }
        } catch (err) {
            setError(err.message || 'Failed to load metrics');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (!initialMetrics) {
            fetchMetrics();
        }
    }, [initialMetrics]);

    const approachDetails = [
        {
            id: 'approach1',
            name: 'Custom CNN (DrowsyNet)',
            icon: Brain,
            color: 'from-blue-500 to-blue-600',
            description: 'A custom-designed deep convolutional neural network with 5 convolutional blocks, batch normalization, and dropout for regularization. Trained from scratch on the MRL Eye Dataset.',
            architecture: [
                '5 Convolutional blocks (32→64→128→256→512 filters)',
                'BatchNormalization + Dropout layers',
                'GlobalAveragePooling2D',
                'Dense layers (512→256→2)',
                'Input size: 80×80×3',
            ],
            pros: [
                'Tailored for drowsiness detection',
                'Good balance of accuracy and speed',
                'Efficient memory usage',
            ],
            cons: [
                'Trained from scratch (needs more data)',
                'May not generalize as well as transfer learning',
            ],
        },
        {
            id: 'approach2',
            name: 'EfficientNet Transfer',
            icon: Sparkles,
            color: 'from-purple-500 to-purple-600',
            description: 'EfficientNetB1 pre-trained on ImageNet with custom classification head. Uses two-phase training: frozen base feature extraction followed by fine-tuning top layers.',
            architecture: [
                'EfficientNetB1 base (ImageNet weights)',
                'Custom classification head',
                'Two-phase training strategy',
                'Input size: 96×96×3',
            ],
            pros: [
                'Leverages powerful pretrained features',
                'Best overall accuracy',
                'Handles edge cases well',
            ],
            cons: [
                'Slightly slower inference',
                'Larger model size',
            ],
        },
        {
            id: 'approach3',
            name: 'EAR Detection',
            icon: Eye,
            color: 'from-green-500 to-green-600',
            description: 'Eye Aspect Ratio calculation using MediaPipe Face Mesh. A geometric approach that measures eye openness using facial landmarks. Drowsiness detected when EAR falls below threshold for consecutive frames.',
            architecture: [
                'MediaPipe Face Mesh (468 landmarks)',
                '6 eye landmarks per eye',
                'EAR formula: (V1 + V2) / (2 × H)',
                'Threshold optimization via grid search',
            ],
            pros: [
                'Fastest inference (~5ms)',
                'No GPU required',
                'Interpretable results',
                'Works in real-time',
            ],
            cons: [
                'Sensitive to lighting conditions',
                'May miss subtle drowsiness signs',
                'Requires clear face visibility',
            ],
        },
    ];

    return (
        <motion.div
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="max-w-6xl mx-auto space-y-8"
        >
            {/* Header */}
            <div className="text-center">
                <h1 className="text-4xl font-bold text-white mb-2">Model Comparison</h1>
                <p className="text-white/60">Performance analysis of all three detection approaches</p>
            </div>

            {/* Metrics Table */}
            {loading ? (
                <div className="flex justify-center py-12">
                    <LoadingSpinner size="lg" text="Loading metrics..." />
                </div>
            ) : error ? (
                <div className="glass-card p-8 text-center">
                    <p className="text-danger-400 mb-4">{error}</p>
                    <button onClick={fetchMetrics} className="btn-primary">
                        <RefreshCw className="w-5 h-5 mr-2 inline" />
                        Retry
                    </button>
                </div>
            ) : (
                <MetricsTable metrics={metrics} />
            )}

            {/* Refresh Button */}
            <div className="flex justify-center">
                <button
                    onClick={fetchMetrics}
                    disabled={loading}
                    className="btn-secondary flex items-center gap-2"
                >
                    <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
                    Refresh Metrics
                </button>
            </div>

            {/* Approach Details */}
            <section className="space-y-6">
                <h2 className="text-2xl font-bold text-white text-center">Approach Details</h2>

                <div className="space-y-6">
                    {approachDetails.map((approach) => {
                        const Icon = approach.icon;
                        const approachMetrics = metrics?.[approach.id];

                        return (
                            <motion.div
                                key={approach.id}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                className="glass-card p-6"
                            >
                                <div className="flex flex-col lg:flex-row gap-6">
                                    {/* Header */}
                                    <div className="lg:w-1/3">
                                        <div className="flex items-center gap-3 mb-3">
                                            <div className={`p-3 rounded-xl bg-gradient-to-br ${approach.color}`}>
                                                <Icon className="w-6 h-6 text-white" />
                                            </div>
                                            <h3 className="text-xl font-semibold text-white">{approach.name}</h3>
                                        </div>

                                        <p className="text-white/60 text-sm">{approach.description}</p>

                                        {/* Quick Metrics */}
                                        {approachMetrics && approachMetrics.accuracy > 0 && (
                                            <div className="mt-4 p-4 rounded-xl bg-white/5">
                                                <div className="grid grid-cols-2 gap-3 text-sm">
                                                    <div>
                                                        <span className="text-white/50">Accuracy</span>
                                                        <p className="text-lg font-bold text-success-400">
                                                            {(approachMetrics.accuracy * 100).toFixed(1)}%
                                                        </p>
                                                    </div>
                                                    <div>
                                                        <span className="text-white/50">Speed</span>
                                                        <p className="text-lg font-bold text-primary-400">
                                                            {approachMetrics.inference_time_ms?.toFixed(0) || '—'}ms
                                                        </p>
                                                    </div>
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    {/* Architecture */}
                                    <div className="lg:w-1/3">
                                        <h4 className="text-sm font-medium text-white/70 mb-3 flex items-center gap-2">
                                            <Info className="w-4 h-4" />
                                            Architecture
                                        </h4>
                                        <ul className="space-y-2">
                                            {approach.architecture.map((item, idx) => (
                                                <li key={idx} className="text-white/60 text-sm flex items-start gap-2">
                                                    <span className="text-primary-400 mt-1">•</span>
                                                    {item}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>

                                    {/* Pros & Cons */}
                                    <div className="lg:w-1/3 grid grid-cols-1 gap-4">
                                        <div>
                                            <h4 className="text-sm font-medium text-success-400 mb-2">Advantages</h4>
                                            <ul className="space-y-1">
                                                {approach.pros.map((item, idx) => (
                                                    <li key={idx} className="text-white/60 text-sm flex items-start gap-2">
                                                        <span className="text-success-400">✓</span>
                                                        {item}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>

                                        <div>
                                            <h4 className="text-sm font-medium text-warning-400 mb-2">Limitations</h4>
                                            <ul className="space-y-1">
                                                {approach.cons.map((item, idx) => (
                                                    <li key={idx} className="text-white/60 text-sm flex items-start gap-2">
                                                        <span className="text-warning-400">•</span>
                                                        {item}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </section>
        </motion.div>
    );
};

export default Comparison;
