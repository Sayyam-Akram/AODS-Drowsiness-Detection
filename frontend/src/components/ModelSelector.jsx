import { motion } from 'framer-motion';
import { Brain, Sparkles, Eye, Check } from 'lucide-react';
import { APPROACHES } from '../utils/constants';

const ModelSelector = ({ selectedApproach, onSelect, disabled = false }) => {
    const icons = {
        approach1: Brain,
        approach2: Sparkles,
        approach3: Eye,
    };

    return (
        <div className="w-full">
            <label className="block text-sm font-medium text-white/70 mb-2">
                Select Detection Model
            </label>

            {/* Info banner */}
            <div className="text-xs text-white/50 mb-3 p-2 rounded-lg bg-white/5 border border-white/10">
                <span className="text-primary-400">💡 Tip:</span> CNN/EfficientNet work best with eye images. EAR needs a full face visible.
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {Object.values(APPROACHES).map((approach) => {
                    const Icon = icons[approach.id];
                    const isSelected = selectedApproach === approach.id;

                    return (
                        <motion.button
                            key={approach.id}
                            whileHover={{ scale: disabled ? 1 : 1.02 }}
                            whileTap={{ scale: disabled ? 1 : 0.98 }}
                            onClick={() => !disabled && onSelect(approach.id)}
                            disabled={disabled}
                            className={`
                relative p-4 rounded-xl border-2 transition-all duration-300
                ${isSelected
                                    ? `bg-gradient-to-r ${approach.color} border-white/30 shadow-lg`
                                    : 'bg-white/5 border-white/10 hover:border-white/20 hover:bg-white/10'
                                }
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
              `}
                        >
                            {/* Selected indicator */}
                            {isSelected && (
                                <motion.div
                                    layoutId="selectedIndicator"
                                    className="absolute top-2 right-2"
                                >
                                    <div className="w-5 h-5 rounded-full bg-white flex items-center justify-center">
                                        <Check className="w-3 h-3 text-gray-800" />
                                    </div>
                                </motion.div>
                            )}

                            <div className="flex flex-col items-center gap-2 text-center">
                                <div className={`
                  w-10 h-10 rounded-lg flex items-center justify-center
                  ${isSelected ? 'bg-white/20' : approach.bgColor + '/20'}
                `}>
                                    <Icon className={`w-5 h-5 ${isSelected ? 'text-white' : 'text-white/70'}`} />
                                </div>

                                <div>
                                    <h4 className={`font-semibold ${isSelected ? 'text-white' : 'text-white/90'}`}>
                                        {approach.name}
                                    </h4>
                                    <p className={`text-xs mt-1 ${isSelected ? 'text-white/80' : 'text-white/50'}`}>
                                        {approach.description}
                                    </p>
                                </div>
                            </div>
                        </motion.button>
                    );
                })}
            </div>
        </div>
    );
};

export default ModelSelector;
