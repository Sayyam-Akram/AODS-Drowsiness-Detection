// Model approach definitions
export const APPROACHES = {
    approach1: {
        id: 'approach1',
        name: 'Custom CNN',
        description: 'Custom deep CNN with 5 convolutional blocks',
        color: 'from-blue-500 to-blue-600',
        bgColor: 'bg-blue-500',
        icon: 'Brain',
    },
    approach2: {
        id: 'approach2',
        name: 'EfficientNet',
        description: 'EfficientNetB1 with transfer learning',
        color: 'from-purple-500 to-purple-600',
        bgColor: 'bg-purple-500',
        icon: 'Sparkles',
    },
    approach3: {
        id: 'approach3',
        name: 'EAR Detection',
        description: 'Eye Aspect Ratio with MediaPipe',
        color: 'from-green-500 to-green-600',
        bgColor: 'bg-green-500',
        icon: 'Eye',
    },
};

// Status types
export const STATUS = {
    AWAKE: 'awake',
    DROWSY: 'drowsy',
    NO_FACE: 'no_face',
    LOADING: 'loading',
    ERROR: 'error',
};

// Animation variants for Framer Motion
export const pageVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.5 } },
    exit: { opacity: 0, y: -20, transition: { duration: 0.3 } },
};

export const cardVariants = {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1, transition: { duration: 0.3 } },
    hover: {
        scale: 1.02,
        transition: { duration: 0.2 }
    },
};

export const pulseVariants = {
    initial: { scale: 1 },
    animate: {
        scale: [1, 1.05, 1],
        opacity: [1, 0.8, 1],
        transition: {
            duration: 1,
            repeat: Infinity,
            ease: "easeInOut"
        }
    }
};

export const staggerContainer = {
    initial: {},
    animate: {
        transition: {
            staggerChildren: 0.1,
        },
    },
};

export const fadeInUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
};
