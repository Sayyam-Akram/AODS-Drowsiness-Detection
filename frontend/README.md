# Drowsiness Detection System - Frontend

Beautiful React + Tailwind CSS interface for the drowsiness detection system.

## Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Features

### Detection Modes

- **Webcam**: Real-time drowsiness detection with live camera feed
- **Image Upload**: Analyze single photos with eye bounding box visualization
- **Video Upload**: Process video files with consecutive frame detection

### Model Selection

Choose from three detection approaches:

1. **Custom CNN**: Deep CNN trained from scratch
2. **EfficientNet**: Transfer learning with EfficientNetB1
3. **EAR Detection**: Eye Aspect Ratio with MediaPipe

### Visualization

- Real-time status indicators (AWAKE/DROWSY)
- Eye bounding boxes with labels
- EAR values display
- Confidence scores
- Processing time

### Alert System

- Audio alerts when drowsiness is detected
- Toggle sound on/off
- Visual pulsing indicators

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Navbar.jsx
│   │   ├── ModelSelector.jsx
│   │   ├── ImageUploader.jsx
│   │   ├── VideoUploader.jsx
│   │   ├── WebcamCapture.jsx
│   │   ├── AlertIndicator.jsx
│   │   ├── ResultsDisplay.jsx
│   │   ├── MetricsTable.jsx
│   │   └── LoadingSpinner.jsx
│   ├── pages/
│   │   ├── Home.jsx
│   │   ├── Detection.jsx
│   │   └── Comparison.jsx
│   ├── utils/
│   │   ├── api.js
│   │   └── constants.js
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── package.json
├── tailwind.config.js
└── vite.config.js
```

## Requirements

- Node.js 18+
- Backend API running on port 5000

## Build for Production

```bash
npm run build
```

The build output will be in the `dist/` directory.
