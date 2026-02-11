import { motion } from 'framer-motion';

const ResultsDisplay = ({ result, annotatedImage }) => {
    if (!result) return null;

    const isEAR = result.approach === 'approach3';

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full space-y-4"
        >
            {/* Annotated Image */}
            {annotatedImage && (
                <div className="rounded-xl overflow-hidden border border-white/20">
                    <img
                        src={`data:image/jpeg;base64,${annotatedImage}`}
                        alt="Annotated result"
                        className="w-full h-auto"
                    />
                </div>
            )}

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard
                    label="Status"
                    value={result.is_drowsy ? 'Drowsy' : 'Awake'}
                    color={result.is_drowsy ? 'danger' : 'success'}
                />

                <MetricCard
                    label="Confidence"
                    value={`${(result.confidence * 100).toFixed(1)}%`}
                    color="primary"
                />

                {/* Show EAR for approach3, show probabilities for CNN */}
                {isEAR ? (
                    <>
                        <MetricCard
                            label="Left EAR"
                            value={result.left_ear?.toFixed(3) || 'N/A'}
                            color="accent"
                        />
                        <MetricCard
                            label="Right EAR"
                            value={result.right_ear?.toFixed(3) || 'N/A'}
                            color="accent"
                        />
                    </>
                ) : (
                    <>
                        <MetricCard
                            label="Awake %"
                            value={`${((result.awake_probability || 0) * 100).toFixed(1)}%`}
                            color="success"
                        />
                        <MetricCard
                            label="Drowsy %"
                            value={`${((result.drowsy_probability || 0) * 100).toFixed(1)}%`}
                            color="danger"
                        />
                    </>
                )}
            </div>

            {/* Processing Time */}
            <div className="text-center text-white/50 text-sm">
                Processed in {result.processing_time_ms?.toFixed(0) || 0}ms
            </div>
        </motion.div>
    );
};

const MetricCard = ({ label, value, color = 'primary' }) => {
    const colorClasses = {
        primary: 'from-primary-500/20 to-primary-500/10 border-primary-500/30',
        success: 'from-success-500/20 to-success-500/10 border-success-500/30',
        danger: 'from-danger-500/20 to-danger-500/10 border-danger-500/30',
        accent: 'from-accent-500/20 to-accent-500/10 border-accent-500/30',
    };

    const textColors = {
        primary: 'text-primary-400',
        success: 'text-success-400',
        danger: 'text-danger-400',
        accent: 'text-accent-400',
    };

    return (
        <div className={`
      p-4 rounded-xl bg-gradient-to-br border
      ${colorClasses[color]}
    `}>
            <p className="text-white/50 text-xs uppercase tracking-wider">{label}</p>
            <p className={`text-xl font-bold mt-1 ${textColors[color]}`}>{value}</p>
        </div>
    );
};

export default ResultsDisplay;
