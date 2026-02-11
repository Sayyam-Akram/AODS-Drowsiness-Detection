# 🚗 Drowsiness Detection System

AI-powered drowsiness detection system with **three detection approaches**, trained on NVIDIA GTX 1660 Super GPU. Pre-trained models are included — no training required to run.

## ✨ Features

- 🧠 **Three AI Approaches**: Custom Deep CNN (DrowsyNet) · EfficientNet Transfer Learning · EAR Detection (MediaPipe)
- 🖼️ **Detection Modes**: Single image analysis · Video file processing with consecutive frame detection
- 🔔 **Alert System**: Audio alerts · Visual pulsing indicators · Confidence scores
- 📊 **Model Comparison**: Performance metrics dashboard with accuracy, precision, recall, F1-score

## 🚀 Quick Start (Clone & Run)

### Prerequisites

- **Python 3.10+** (tested on 3.13)
- **Node.js 18+**

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/AODS-Drowsiness-Detection.git
cd AODS-Drowsiness-Detection
```

### 2. Setup Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Backend

```bash
cd backend
python -m api.app
```

Backend runs on `http://localhost:5000`

### 4. Setup & Start Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Frontend runs on `http://localhost:5173`

> **Note:** Pre-trained model weights are included in `backend/models/`. No training or GPU required to run inference.

---

## 🏋️ Re-Training Models (Optional)

If you want to retrain the models from scratch, you'll need:
- NVIDIA GPU with CUDA support (GTX 1660 Super or better recommended)
- The [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) placed in `data/` directory:
  ```
  data/
  ├── train/
  │   ├── awake/
  │   └── sleepy/
  ├── val/
  │   ├── awake/
  │   └── sleepy/
  └── test/
      ├── awake/
      └── sleepy/
  ```

```bash
cd backend

# Train all approaches
python -m training.train_all

# Or individually:
# python -m training.train_approach1_cnn
# python -m training.train_approach2_efficientnet
# python -m training.train_approach3_ear
```

---

## 📁 Project Structure

```
AODS-Drowsiness-Detection/
├── backend/
│   ├── api/                 # Flask REST API
│   ├── training/            # Training scripts
│   ├── utils/               # Utility modules
│   ├── models/              # Pre-trained models (included)
│   │   ├── approach1_cnn.h5
│   │   ├── approach2_efficientnet.h5
│   │   ├── ear_detector.pkl
│   │   └── face_landmarker.task
│   ├── assets/              # Alert sounds
│   ├── config.py            # Configuration
│   └── requirements.txt     # Python dependencies
│
└── frontend/
    ├── src/
    │   ├── components/      # React components
    │   ├── pages/           # Page components
    │   └── utils/           # API utilities
    ├── package.json
    └── tailwind.config.js
```

## ⚙️ Configuration

Edit `backend/config.py` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 32 | Reduce to 16 if OOM errors |
| `CNN_EPOCHS` | 50 | Training epochs for CNN |
| `EARLY_STOPPING_PATIENCE` | 10 | Early stopping patience |
| `CONSECUTIVE_FRAMES_THRESHOLD` | 5 | Frames for EAR detection |

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check and loaded models |
| `/api/metrics` | GET | Model performance metrics |
| `/api/predict` | POST | Image prediction |
| `/api/predict/base64` | POST | Base64 image (webcam) |
| `/api/video/process` | POST | Video file processing |
| `/api/reset` | POST | Reset inference state |

## 📈 Expected Performance

| Approach | Accuracy | Inference Time |
|----------|----------|----------------|
| Custom CNN | 93-95% | 15-20ms |
| EfficientNet | 95-97% | 20-30ms |
| EAR Detection | 85-92% | 5-10ms |

## 🛠️ Technologies

- **Backend**: Python, TensorFlow, Flask, MediaPipe, OpenCV
- **Frontend**: React, Tailwind CSS, Framer Motion
- **ML**: Custom CNN, EfficientNetB1, Eye Aspect Ratio

## 📄 License

MIT License
