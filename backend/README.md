# Drowsiness Detection System - Backend

Python backend with three AI approaches for drowsiness detection.

## Setup

### 1. Create Virtual Environment

```bash
cd backend
python -m venv venv

# Windows activation
venv\Scripts\activate

# Linux/Mac activation
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify GPU

```bash
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

## Training Models

### Train All Approaches

```bash
cd backend
python -m training.train_all
```

### Train Individual Approaches

```bash
# Approach 1: Custom CNN
python -m training.train_approach1_cnn

# Approach 2: EfficientNet
python -m training.train_approach2_efficientnet

# Approach 3: EAR Detection
python -m training.train_approach3_ear
```

## Configuration

Edit `config.py` to adjust:

- **BATCH_SIZE**: Reduce to 16 if OOM errors occur
- **CNN_EPOCHS**: Default 50
- **EARLY_STOPPING_PATIENCE**: Default 10
- **CONSECUTIVE_FRAMES_THRESHOLD**: For EAR detection

## Running the API

```bash
cd backend
python -m api.app
```

API will be available at `http://localhost:5000`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check and GPU status |
| `/api/metrics` | GET | Model performance metrics |
| `/api/predict` | POST | Image prediction |
| `/api/predict/base64` | POST | Base64 image prediction |
| `/api/video/process` | POST | Video processing |
| `/api/reset` | POST | Reset EAR state |
| `/api/models` | GET | Available models |

## Project Structure

```
backend/
├── api/                 # Flask REST API
│   ├── app.py          # Main Flask server
│   ├── inference.py    # Unified inference engine
│   └── metrics_loader.py
├── training/           # Training scripts
│   ├── gpu_setup.py
│   ├── train_approach1_cnn.py
│   ├── train_approach2_efficientnet.py
│   ├── train_approach3_ear.py
│   └── train_all.py
├── utils/              # Utility modules
│   ├── preprocessing.py
│   ├── eye_detection.py
│   ├── ear_calculator.py
│   └── visualization.py
├── models/             # Saved models (after training)
├── assets/             # Alert sounds
├── config.py           # Configuration
└── requirements.txt
```

## Alert Sound

Place a 1-second MP3 alert sound at `assets/alert.mp3`. This will play when drowsiness is detected.
