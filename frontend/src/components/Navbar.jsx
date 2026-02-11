import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Eye, Activity, BarChart3, CheckCircle, XCircle } from 'lucide-react';

const Navbar = ({ apiStatus }) => {
    const location = useLocation();

    const navLinks = [
        { path: '/', label: 'Home', icon: Eye },
        { path: '/detection', label: 'Detection', icon: Activity },
        { path: '/comparison', label: 'Comparison', icon: BarChart3 },
    ];

    const isActive = (path) => location.pathname === path;

    return (
        <motion.nav
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="sticky top-0 z-50 glass-card border-b border-white/10 backdrop-blur-xl"
        >
            <div className="container mx-auto px-4">
                <div className="flex items-center justify-between h-16">
                    {/* Logo */}
                    <Link to="/" className="flex items-center gap-3 group">
                        <motion.div
                            whileHover={{ rotate: 360 }}
                            transition={{ duration: 0.5 }}
                            className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center"
                        >
                            <Eye className="w-6 h-6 text-white" />
                        </motion.div>
                        <span className="text-xl font-bold gradient-text">
                            DrowsyGuard
                        </span>
                    </Link>

                    {/* Navigation Links */}
                    <div className="flex items-center gap-2">
                        {navLinks.map((link) => {
                            const Icon = link.icon;
                            return (
                                <Link
                                    key={link.path}
                                    to={link.path}
                                    className={`nav-link flex items-center gap-2 ${isActive(link.path) ? 'nav-link-active' : ''
                                        }`}
                                >
                                    <Icon className="w-4 h-4" />
                                    <span className="hidden sm:inline">{link.label}</span>
                                </Link>
                            );
                        })}
                    </div>

                    {/* API Status */}
                    <div className="flex items-center gap-2">
                        {apiStatus?.status === 'healthy' ? (
                            <motion.div
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-success-500/20 border border-success-500/30"
                            >
                                <CheckCircle className="w-4 h-4 text-success-400" />
                                <span className="text-sm text-success-400 hidden sm:inline">API Connected</span>
                            </motion.div>
                        ) : (
                            <motion.div
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-danger-500/20 border border-danger-500/30"
                            >
                                <XCircle className="w-4 h-4 text-danger-400" />
                                <span className="text-sm text-danger-400 hidden sm:inline">API Offline</span>
                            </motion.div>
                        )}
                    </div>
                </div>
            </div>
        </motion.nav>
    );
};

export default Navbar;
