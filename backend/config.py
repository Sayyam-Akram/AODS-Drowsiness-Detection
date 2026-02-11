# Configuration for Drowsiness Detection System
# Easy to modify parameters

# ===========================================
# TRAINING CONFIGURATION
# ===========================================

# Batch size (reduce to 16 if OOM error on GTX 1660 Super)
BATCH_SIZE = 32

# Image sizes for each approach
CNN_IMG_SIZE = (80, 80)
EFFICIENTNET_IMG_SIZE = (96, 96)

# Training epochs
CNN_EPOCHS = 50
EFFICIENTNET_PHASE1_EPOCHS = 15
EFFICIENTNET_PHASE2_EPOCHS = 30

# Early stopping patience
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5

# Learning rates
CNN_LEARNING_RATE = 0.001
EFFICIENTNET_PHASE1_LR = 0.001
EFFICIENTNET_PHASE2_LR = 0.0001

# ===========================================
# MODEL PATHS
# ===========================================
MODELS_DIR = "models"
CNN_MODEL_PATH = "models/approach1_cnn.h5"
EFFICIENTNET_MODEL_PATH = "models/approach2_efficientnet.h5"
EAR_MODEL_PATH = "models/ear_detector.pkl"

# Metrics paths
CNN_METRICS_PATH = "models/approach1_metrics.json"
EFFICIENTNET_METRICS_PATH = "models/approach2_metrics.json"
EAR_METRICS_PATH = "models/ear_metrics.json"

# ===========================================
# DATASET PATHS
# ===========================================
DATA_DIR = "../data"
TRAIN_DIR = "../data/train"
VAL_DIR = "../data/val"
TEST_DIR = "../data/test"

# Class names
CLASSES = ["awake", "sleepy"]

# ===========================================
# EAR DETECTION CONFIG
# ===========================================
EAR_THRESHOLD_MIN = 0.15
EAR_THRESHOLD_MAX = 0.30
EAR_THRESHOLD_STEPS = 100
CONSECUTIVE_FRAMES_THRESHOLD = 5

# ===========================================
# ALERT CONFIG
# ===========================================
ALERT_SOUND_PATH = "assets/alert.mp3"

# ===========================================
# API CONFIG
# ===========================================
API_HOST = "0.0.0.0"
API_PORT = 5000
DEBUG_MODE = True
