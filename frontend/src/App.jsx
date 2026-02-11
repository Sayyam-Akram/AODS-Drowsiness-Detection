import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Detection from './pages/Detection';
import Comparison from './pages/Comparison';
import { checkHealth, getMetrics } from './utils/api';

function App() {
    const [apiStatus, setApiStatus] = useState(null);
    const [metrics, setMetrics] = useState(null);

    useEffect(() => {
        // Check API health on load
        const checkApi = async () => {
            try {
                const health = await checkHealth();
                setApiStatus(health);

                const metricsData = await getMetrics();
                if (metricsData.success) {
                    setMetrics(metricsData.summary);
                }
            } catch (error) {
                console.error('API not available:', error);
                setApiStatus({ status: 'error', error: error.message });
            }
        };

        checkApi();
    }, []);

    return (
        <Router>
            <div className="min-h-screen">
                <Navbar apiStatus={apiStatus} />

                <main className="container mx-auto px-4 py-8">
                    <Routes>
                        <Route path="/" element={<Home apiStatus={apiStatus} metrics={metrics} />} />
                        <Route path="/detection" element={<Detection metrics={metrics} />} />
                        <Route path="/comparison" element={<Comparison metrics={metrics} />} />
                    </Routes>
                </main>
            </div>
        </Router>
    );
}

export default App;
