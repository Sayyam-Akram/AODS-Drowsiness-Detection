import { motion } from 'framer-motion';
import { Brain, Sparkles, Eye, Zap, Award, TrendingUp } from 'lucide-react';
import { APPROACHES, staggerContainer, fadeInUp } from '../utils/constants';

const MetricsTable = ({ metrics }) => {
    if (!metrics) {
        return (
            <div className="w-full p-8 text-center text-white/50">
                <p>No metrics available. Train the models first.</p>
            </div>
        );
    }

    const icons = {
        approach1: Brain,
        approach2: Sparkles,
        approach3: Eye,
    };

    const approaches = Object.entries(APPROACHES).map(([key, approach]) => ({
        ...approach,
        data: metrics[key] || {},
        Icon: icons[key],
    }));

    // Find best performer
    const bestAccuracy = Math.max(...approaches.map(a => a.data.accuracy || 0));

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900/80 to-slate-800/80 backdrop-blur-xl border border-white/10 shadow-2xl"
        >
            {/* Header */}
            <div className="p-6 border-b border-white/10">
                <h2 className="text-2xl font-bold gradient-text flex items-center gap-3">
                    <TrendingUp className="w-7 h-7" />
                    Model Performance Comparison
                </h2>
                <p className="text-white/50 mt-1">Comparing all three detection approaches</p>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
                <table className="w-full text-left">
                    <thead>
                        <tr className="border-b border-white/10">
                            <th className="px-6 py-4 text-white/70 font-medium">Approach</th>
                            <th className="px-6 py-4 text-white/70 font-medium text-center">Accuracy</th>
                            <th className="px-6 py-4 text-white/70 font-medium text-center">Precision</th>
                            <th className="px-6 py-4 text-white/70 font-medium text-center">Recall</th>
                            <th className="px-6 py-4 text-white/70 font-medium text-center">F1-Score</th>
                            <th className="px-6 py-4 text-white/70 font-medium text-center">Speed</th>
                        </tr>
                    </thead>
                    <motion.tbody
                        variants={staggerContainer}
                        initial="initial"
                        animate="animate"
                    >
                        {approaches.map((approach, idx) => {
                            const Icon = approach.Icon;
                            const isBest = approach.data.accuracy === bestAccuracy && bestAccuracy > 0;

                            return (
                                <motion.tr
                                    key={approach.id}
                                    variants={fadeInUp}
                                    className={`
                    border-b border-white/5 transition-colors hover:bg-white/5
                    ${isBest ? 'bg-yellow-500/5' : ''}
                  `}
                                >
                                    {/* Approach Name */}
                                    <td className="px-6 py-5">
                                        <div className="flex items-center gap-3">
                                            <div className={`p-2 rounded-lg bg-gradient-to-br ${approach.color}`}>
                                                <Icon className="w-5 h-5 text-white" />
                                            </div>
                                            <div>
                                                <span className="font-semibold text-white flex items-center gap-2">
                                                    {approach.name}
                                                    {isBest && (
                                                        <Award className="w-4 h-4 text-yellow-400" />
                                                    )}
                                                </span>
                                                <p className="text-xs text-white/40 mt-0.5">{approach.description}</p>
                                            </div>
                                        </div>
                                    </td>

                                    {/* Accuracy */}
                                    <td className="px-6 py-5 text-center">
                                        <motion.span
                                            initial={{ scale: 0, opacity: 0 }}
                                            animate={{ scale: 1, opacity: 1 }}
                                            transition={{ delay: idx * 0.1 + 0.2, type: "spring" }}
                                            className={`
                        text-2xl font-bold
                        ${(approach.data.accuracy || 0) >= 0.9 ? 'text-success-400' :
                                                    (approach.data.accuracy || 0) >= 0.8 ? 'text-yellow-400' : 'text-white/50'}
                      `}
                                        >
                                            {approach.data.accuracy
                                                ? `${(approach.data.accuracy * 100).toFixed(1)}%`
                                                : '—'
                                            }
                                        </motion.span>
                                    </td>

                                    {/* Precision */}
                                    <td className="px-6 py-5 text-center text-white/70">
                                        {approach.data.precision
                                            ? `${(approach.data.precision * 100).toFixed(1)}%`
                                            : '—'
                                        }
                                    </td>

                                    {/* Recall */}
                                    <td className="px-6 py-5 text-center text-white/70">
                                        {approach.data.recall
                                            ? `${(approach.data.recall * 100).toFixed(1)}%`
                                            : '—'
                                        }
                                    </td>

                                    {/* F1-Score */}
                                    <td className="px-6 py-5 text-center text-white/70">
                                        {approach.data.f1_score
                                            ? `${(approach.data.f1_score * 100).toFixed(1)}%`
                                            : '—'
                                        }
                                    </td>

                                    {/* Speed */}
                                    <td className="px-6 py-5 text-center">
                                        {approach.data.inference_time_ms ? (
                                            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary-500/20 text-primary-400 text-sm font-medium">
                                                <Zap className="w-3.5 h-3.5" />
                                                {approach.data.inference_time_ms.toFixed(0)}ms
                                            </span>
                                        ) : (
                                            <span className="text-white/30">—</span>
                                        )}
                                    </td>
                                </motion.tr>
                            );
                        })}
                    </motion.tbody>
                </table>
            </div>

            {/* Best Performer Highlight */}
            {bestAccuracy > 0 && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5 }}
                    className="p-4 m-4 rounded-xl bg-gradient-to-r from-yellow-500/10 to-orange-500/10 border border-yellow-500/20"
                >
                    <p className="flex items-center gap-2 text-yellow-400 font-medium">
                        <Award className="w-5 h-5" />
                        <span>
                            Best Overall: {
                                approaches.find(a => a.data.accuracy === bestAccuracy)?.name
                            } ({(bestAccuracy * 100).toFixed(1)}% accuracy)
                        </span>
                    </p>
                </motion.div>
            )}
        </motion.div>
    );
};

export default MetricsTable;
