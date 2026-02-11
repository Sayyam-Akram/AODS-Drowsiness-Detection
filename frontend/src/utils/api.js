import axios from 'axios';

const API_BASE = '/api';

const api = axios.create({
    baseURL: API_BASE,
    timeout: 30000,
});

// Health check
export const checkHealth = async () => {
    const response = await api.get('/health');
    return response.data;
};

// Get metrics
export const getMetrics = async () => {
    const response = await api.get('/metrics');
    return response.data;
};

// Get available models
export const getModels = async () => {
    const response = await api.get('/models');
    return response.data;
};

// Predict from image file
export const predictImage = async (imageFile, approach = 'approach1') => {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('approach', approach);

    const response = await api.post('/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
};

// Predict from base64 image (webcam)
export const predictBase64 = async (base64Image, approach = 'approach1') => {
    const response = await api.post('/predict/base64', {
        image: base64Image,
        approach,
    });
    return response.data;
};

// Process video file
export const processVideo = async (videoFile, approach = 'approach1') => {
    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('approach', approach);

    const response = await api.post('/video/process', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000, // 2 minutes for video processing
    });
    return response.data;
};

// Reset inference state
export const resetState = async () => {
    const response = await api.post('/reset');
    return response.data;
};

// Get alert sound URL
export const getAlertSoundUrl = () => `${API_BASE}/alert/sound`;

export default api;
