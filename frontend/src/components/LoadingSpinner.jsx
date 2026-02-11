import { motion } from 'framer-motion';

const LoadingSpinner = ({ size = 'md', text = '' }) => {
    const sizes = {
        sm: 'w-5 h-5',
        md: 'w-8 h-8',
        lg: 'w-12 h-12',
        xl: 'w-16 h-16',
    };

    return (
        <div className="flex flex-col items-center justify-center gap-4">
            <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                className={`${sizes[size]} rounded-full border-2 border-white/20 border-t-primary-500`}
            />
            {text && (
                <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-white/60 text-sm"
                >
                    {text}
                </motion.p>
            )}
        </div>
    );
};

export default LoadingSpinner;
