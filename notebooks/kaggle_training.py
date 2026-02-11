# =============================================================================
# AODS DROWSINESS DETECTION - KAGGLE TRAINING NOTEBOOK
# =============================================================================
# Dataset: MRL Eye Dataset (85K+ infrared eye images)
# GPU: T4 x2 (Enable in Settings → Accelerator)
#
# HOW TO RUN:
# 1. Create new Kaggle Notebook
# 2. Add "mrl-eye-dataset" from Datasets (already added based on screenshot)
# 3. Copy ALL cells below into notebook
# 4. Enable GPU: Settings → Accelerator → GPU T4 x2
# 5. Run All Cells
# 6. Download models from /kaggle/working/ output
# =============================================================================

# %% [markdown]
# # AODS Drowsiness Detection Training
# 
# Training 3 approaches on MRL Eye Dataset (85K+ infrared images):
# 1. **Custom CNN** (Custom_DrowsyNet)
# 2. **EfficientNet** Transfer Learning
# 3. **EAR** Threshold Detection

# %% [code]
# =============================================================================
# CELL 1: PREFLIGHT CHECKS - RUN THIS FIRST!
# =============================================================================
# This cell verifies everything before training starts

import os
import sys
from pathlib import Path

print("=" * 60)
print("PREFLIGHT CHECKS")
print("=" * 60)

# Check 1: GPU Available
print("\n1️⃣ GPU CHECK:")
import tensorflow as tf
print(f"   TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"   GPUs found: {len(gpus)}")
for gpu in gpus:
    print(f"   → {gpu}")
if len(gpus) == 0:
    print("   ⚠️ WARNING: No GPU! Enable GPU in Settings → Accelerator → GPU T4 x2")
else:
    print("   ✅ GPU Ready!")

# Check 2: Dataset Path
print("\n2️⃣ DATASET CHECK:")
DATA_PATH = "/kaggle/input/mrl-eye-dataset/data"
if os.path.exists(DATA_PATH):
    print(f"   ✅ Dataset found at: {DATA_PATH}")
    for split in ["train", "val", "test"]:
        split_path = os.path.join(DATA_PATH, split)
        if os.path.exists(split_path):
            classes = os.listdir(split_path)
            counts = {c: len(os.listdir(os.path.join(split_path, c))) for c in classes}
            print(f"   → {split}: {counts}")
else:
    print(f"   ❌ Dataset NOT found at {DATA_PATH}")
    print("   Make sure 'mrl-eye-dataset' is added to your notebook inputs!")

# Check 3: Sample Image Info
print("\n3️⃣ IMAGE FORMAT CHECK:")
sample_dir = os.path.join(DATA_PATH, "train", "awake")
if os.path.exists(sample_dir):
    sample_files = os.listdir(sample_dir)[:1]
    if sample_files:
        import cv2
        sample_path = os.path.join(sample_dir, sample_files[0])
        img = cv2.imread(sample_path)
        print(f"   Sample image: {sample_files[0]}")
        print(f"   Shape: {img.shape}")
        print(f"   Dtype: {img.dtype}")
        if len(img.shape) == 2 or img.shape[2] == 1:
            print("   Format: Grayscale")
        else:
            print("   Format: Color (BGR)")

# Check 4: Python & Library Versions
print("\n4️⃣ VERSION CHECK:")
print(f"   Python: {sys.version.split()[0]}")
print(f"   TensorFlow: {tf.__version__}")
import numpy as np
print(f"   NumPy: {np.__version__}")
import cv2
print(f"   OpenCV: {cv2.__version__}")
from sklearn import __version__ as sklearn_version
print(f"   Scikit-learn: {sklearn_version}")

# Check 5: Memory
print("\n5️⃣ MEMORY CHECK:")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free', 
                            '--format=csv,noheader,nounits'], capture_output=True, text=True)
    total, free = result.stdout.strip().split('\n')[0].split(', ')
    print(f"   GPU Memory: {free}MB free / {total}MB total")
except:
    print("   Could not check GPU memory")

print("\n" + "=" * 60)
print("✅ PREFLIGHT COMPLETE - If all checks passed, continue to training!")
print("=" * 60)

# %% [code]
# =============================================================================
# CELL 2: CONFIGURATION & SETUP
# =============================================================================

import json
import pickle
import time
from datetime import datetime

import numpy as np
import cv2
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0  # Using B0 for faster training
from sklearn.metrics import classification_report, confusion_matrix

# Configure GPU memory growth (prevents OOM)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# ===== MULTI-GPU SETUP =====
# Use MirroredStrategy to train on 2 GPUs in parallel
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"✅ Multi-GPU enabled: Using {strategy.num_replicas_in_sync} GPUs")
else:
    strategy = tf.distribute.get_strategy()  # Default strategy for 1 GPU
    print(f"ℹ️ Single GPU mode")

NUM_GPUS = strategy.num_replicas_in_sync

# IMPORTANT: Using float32 for stability (mixed precision caused NaN overflow)
# DO NOT enable mixed precision - it causes numerical instability with this dataset
print("✅ Using float32 for stable training (mixed precision disabled)")

# =============================================================================
# CONFIGURATION - Optimized for MRL Eye Dataset (85K images)
# =============================================================================

# Dataset path - VERIFIED FROM YOUR SCREENSHOT
DATA_PATH = "/kaggle/input/mrl-eye-dataset/data"
TRAIN_DIR = os.path.join(DATA_PATH, "train")
VAL_DIR = os.path.join(DATA_PATH, "val")
TEST_DIR = os.path.join(DATA_PATH, "test")

# Output directory
OUTPUT_DIR = "/kaggle/working"

# Model config - Optimized for 85K images to prevent overfitting
CONFIG = {
    # Image sizes
    "CNN_IMG_SIZE": (80, 80),         # For custom CNN
    "EFFICIENTNET_IMG_SIZE": (96, 96), # For EfficientNet
    
    # Training params - Conservative to prevent overfitting on 85K images
    # Batch size per GPU - will be multiplied by number of GPUs
    "BATCH_SIZE_PER_GPU": 32,          # Per-GPU batch size
    "CNN_EPOCHS": 30,                  # Reduced - large dataset converges faster
    "EFFICIENTNET_PHASE1_EPOCHS": 10,  # Feature extraction phase
    "EFFICIENTNET_PHASE2_EPOCHS": 15,  # Fine-tuning phase
    
    # Regularization
    "EARLY_STOPPING_PATIENCE": 7,      # Stop early to prevent overfitting
    "REDUCE_LR_PATIENCE": 3,
    
    # Learning rates - slightly lower for stability
    "CNN_LEARNING_RATE": 0.0003,       # Reduced for stability
    "EFFICIENTNET_PHASE1_LR": 0.0005,  # Reduced
    "EFFICIENTNET_PHASE2_LR": 0.00003, # Very low for fine-tuning
    
    # Classes
    "CLASSES": ["awake", "sleepy"],
    
    # EAR config
    "EAR_THRESHOLD": 0.22,
    "CONSECUTIVE_FRAMES": 5,
}

# Output paths
CNN_MODEL_PATH = os.path.join(OUTPUT_DIR, "approach1_cnn.h5")
CNN_METRICS_PATH = os.path.join(OUTPUT_DIR, "approach1_metrics.json")
EFFICIENTNET_MODEL_PATH = os.path.join(OUTPUT_DIR, "approach2_efficientnet.h5")
EFFICIENTNET_METRICS_PATH = os.path.join(OUTPUT_DIR, "approach2_metrics.json")
EAR_CONFIG_PATH = os.path.join(OUTPUT_DIR, "ear_detector.pkl")
EAR_METRICS_PATH = os.path.join(OUTPUT_DIR, "ear_metrics.json")

# Calculate global batch size (per_gpu * num_gpus)
GLOBAL_BATCH_SIZE = CONFIG["BATCH_SIZE_PER_GPU"] * NUM_GPUS

print("✅ Configuration loaded")
print(f"   Dataset: {DATA_PATH}")
print(f"   Batch size per GPU: {CONFIG['BATCH_SIZE_PER_GPU']}")
print(f"   Global batch size: {GLOBAL_BATCH_SIZE} (x{NUM_GPUS} GPUs)")
print(f"   CNN epochs: {CONFIG['CNN_EPOCHS']}")
print(f"   Classes: {CONFIG['CLASSES']}")

# %% [code]
# =============================================================================
# CELL 3: APPROACH 1 - CUSTOM CNN (Custom_DrowsyNet)
# =============================================================================
# Architecture optimized for grayscale infrared eye images
# Added more dropout to prevent overfitting on 85K images

def create_cnn_model(input_shape=(80, 80, 3), num_classes=2):
    """
    Custom_DrowsyNet: 4 Convolutional Blocks (optimized for eye images)
    
    Reduced from 5 to 4 blocks to prevent overfitting with 85K images.
    Heavy dropout and batchnorm for regularization.
    """
    inputs = keras.Input(shape=input_shape)
    
    # Block 1: 32 filters
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2: 64 filters
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 3: 128 filters
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.35)(x)
    
    # Block 4: 256 filters
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with heavy regularization
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    
    # Output (float32 for mixed precision)
    x = layers.Dense(num_classes, dtype='float32')(x)
    outputs = layers.Activation('softmax', dtype='float32')(x)
    
    model = keras.Model(inputs, outputs, name="Custom_DrowsyNet")
    return model


def train_approach1_cnn():
    """Train Custom CNN on MRL Eye Dataset."""
    print("\n" + "=" * 60)
    print("APPROACH 1: Custom CNN (Custom_DrowsyNet)")
    print("=" * 60)
    
    # Data generators with augmentation
    # Note: Images are loaded as RGB (3 channels) even if grayscale
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,           # Gentle rotation
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1], # Subtle brightness for infrared
        fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    img_size = CONFIG["CNN_IMG_SIZE"]
    
    print(f"\n📁 Loading data...")
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=img_size, batch_size=GLOBAL_BATCH_SIZE,
        class_mode='categorical', classes=CONFIG["CLASSES"], shuffle=True
    )
    val_gen = val_test_datagen.flow_from_directory(
        VAL_DIR, target_size=img_size, batch_size=GLOBAL_BATCH_SIZE,
        class_mode='categorical', classes=CONFIG["CLASSES"], shuffle=False
    )
    test_gen = val_test_datagen.flow_from_directory(
        TEST_DIR, target_size=img_size, batch_size=GLOBAL_BATCH_SIZE,
        class_mode='categorical', classes=CONFIG["CLASSES"], shuffle=False
    )
    
    print(f"   Train: {train_gen.samples} images")
    print(f"   Val: {val_gen.samples} images")
    print(f"   Test: {test_gen.samples} images")
    print(f"   Batch size: {GLOBAL_BATCH_SIZE} (x{NUM_GPUS} GPUs)")
    
    # Create model inside strategy scope for multi-GPU
    print(f"\n🏗️ Building model (Multi-GPU)...")
    with strategy.scope():
        model = create_cnn_model(input_shape=(*img_size, 3), num_classes=2)
        # Gradient clipping prevents NaN by limiting gradient magnitude
        optimizer = keras.optimizers.Adam(
            learning_rate=CONFIG["CNN_LEARNING_RATE"],
            clipnorm=1.0  # Clip gradients to prevent explosion
        )
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    model.summary()
    print(f"\nTotal params: {model.count_params():,}")
    
    # Callbacks
    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=CONFIG["EARLY_STOPPING_PATIENCE"],
            restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=CNN_MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=CONFIG["REDUCE_LR_PATIENCE"], min_lr=1e-7, verbose=1
        ),
    ]
    
    # Train
    print(f"\n🚀 Training for up to {CONFIG['CNN_EPOCHS']} epochs...")
    start_time = time.time()
    
    history = model.fit(
        train_gen, epochs=CONFIG["CNN_EPOCHS"],
        validation_data=val_gen, callbacks=callback_list, verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
    
    # Evaluate
    print("\n📊 Evaluating...")
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    
    report = classification_report(
        true_classes, predicted_classes,
        target_names=CONFIG["CLASSES"], output_dict=True
    )
    cm = confusion_matrix(true_classes, predicted_classes)
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
    
    # Save metrics
    metrics = {
        "model_name": "Custom_DrowsyNet",
        "approach": "approach1",
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "precision": float(report['weighted avg']['precision']),
        "recall": float(report['weighted avg']['recall']),
        "f1_score": float(report['weighted avg']['f1-score']),
        "confusion_matrix": cm.tolist(),
        "class_report": report,
        "input_shape": list(img_size) + [3],
        "training_epochs": len(history.history['loss']),
        "training_time_minutes": training_time / 60
    }
    
    with open(CNN_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n� RESULTS:")
    print(f"   Test Accuracy: {test_accuracy:.2%}")
    print(f"   Precision: {metrics['precision']:.2%}")
    print(f"   Recall: {metrics['recall']:.2%}")
    print(f"   F1-Score: {metrics['f1_score']:.2%}")
    print(f"\n📁 Saved: {CNN_MODEL_PATH}")
    
    return model, metrics

# %% [code]
# =============================================================================
# CELL 4: APPROACH 2 - EFFICIENTNET TRANSFER LEARNING
# =============================================================================
# Using EfficientNetB0 (smaller, faster) with two-phase training

def create_efficientnet_model(input_shape=(96, 96, 3), num_classes=2):
    """
    EfficientNetB0 with custom head.
    B0 is smaller and faster while still effective.
    """
    base_model = EfficientNetB0(
        weights='imagenet', include_top=False, input_shape=input_shape
    )
    base_model.trainable = False  # Freeze initially
    
    inputs = keras.Input(shape=input_shape)
    x = keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = keras.Model(inputs, outputs, name="EfficientNet_Transfer")
    return model, base_model


def train_approach2_efficientnet():
    """Two-phase transfer learning with EfficientNetB0."""
    print("\n" + "=" * 60)
    print("APPROACH 2: EfficientNet Transfer Learning")
    print("=" * 60)
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.1, horizontal_flip=True, brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator()
    
    img_size = CONFIG["EFFICIENTNET_IMG_SIZE"]
    
    print(f"\n📁 Loading data...")
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=img_size, batch_size=GLOBAL_BATCH_SIZE,
        class_mode='categorical', classes=CONFIG["CLASSES"], shuffle=True
    )
    val_gen = val_test_datagen.flow_from_directory(
        VAL_DIR, target_size=img_size, batch_size=GLOBAL_BATCH_SIZE,
        class_mode='categorical', classes=CONFIG["CLASSES"], shuffle=False
    )
    test_gen = val_test_datagen.flow_from_directory(
        TEST_DIR, target_size=img_size, batch_size=GLOBAL_BATCH_SIZE,
        class_mode='categorical', classes=CONFIG["CLASSES"], shuffle=False
    )
    
    print(f"   Train: {train_gen.samples} images")
    print(f"   Val: {val_gen.samples} images")
    print(f"   Test: {test_gen.samples} images")
    print(f"   Batch size: {GLOBAL_BATCH_SIZE} (x{NUM_GPUS} GPUs)")
    
    # Create model inside strategy scope for multi-GPU
    print(f"\n🏗️ Building model (Multi-GPU)...")
    with strategy.scope():
        model, base_model = create_efficientnet_model(input_shape=(*img_size, 3))
    model.summary()
    
    start_time = time.time()
    
    # ===== PHASE 1: Feature Extraction =====
    print("\n" + "-" * 40)
    print("PHASE 1: Feature Extraction (Base Frozen)")
    print("-" * 40)
    
    with strategy.scope():
        # Using gradient clipping to prevent NaN
        optimizer = keras.optimizers.Adam(
            learning_rate=CONFIG["EFFICIENTNET_PHASE1_LR"],
            clipnorm=1.0
        )
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy', metrics=['accuracy']
        )
    
    phase1_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(filepath=EFFICIENTNET_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    ]
    
    history1 = model.fit(
        train_gen, epochs=CONFIG["EFFICIENTNET_PHASE1_EPOCHS"],
        validation_data=val_gen, callbacks=phase1_callbacks, verbose=1
    )
    
    # ===== PHASE 2: Fine-Tuning =====
    print("\n" + "-" * 40)
    print("PHASE 2: Fine-Tuning (Top 30 layers unfrozen)")
    print("-" * 40)
    
    base_model.trainable = True
    # Freeze all but top 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    with strategy.scope():
        # Using gradient clipping for fine-tuning stability
        optimizer = keras.optimizers.Adam(
            learning_rate=CONFIG["EFFICIENTNET_PHASE2_LR"],
            clipnorm=1.0
        )
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy', metrics=['accuracy']
        )
    
    phase2_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=CONFIG["EARLY_STOPPING_PATIENCE"], restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(filepath=EFFICIENTNET_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]
    
    history2 = model.fit(
        train_gen, epochs=CONFIG["EFFICIENTNET_PHASE2_EPOCHS"],
        validation_data=val_gen, callbacks=phase2_callbacks, verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
    
    # Evaluate
    print("\n📊 Evaluating...")
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    
    report = classification_report(
        true_classes, predicted_classes,
        target_names=CONFIG["CLASSES"], output_dict=True
    )
    cm = confusion_matrix(true_classes, predicted_classes)
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
    
    # Save metrics
    metrics = {
        "model_name": "EfficientNet_Transfer",
        "approach": "approach2",
        "base_model": "EfficientNetB0",
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "precision": float(report['weighted avg']['precision']),
        "recall": float(report['weighted avg']['recall']),
        "f1_score": float(report['weighted avg']['f1-score']),
        "confusion_matrix": cm.tolist(),
        "class_report": report,
        "input_shape": list(img_size) + [3],
        "training_time_minutes": training_time / 60
    }
    
    with open(EFFICIENTNET_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n� RESULTS:")
    print(f"   Test Accuracy: {test_accuracy:.2%}")
    print(f"   Precision: {metrics['precision']:.2%}")
    print(f"   Recall: {metrics['recall']:.2%}")
    print(f"   F1-Score: {metrics['f1_score']:.2%}")
    print(f"\n📁 Saved: {EFFICIENTNET_MODEL_PATH}")
    
    return model, metrics

# %% [code]
# =============================================================================
# CELL 5: APPROACH 3 - EAR THRESHOLD CONFIG
# =============================================================================
# For EAR detection, we save optimal threshold configuration
# The actual EAR calculation uses MediaPipe Face Mesh at inference time

def train_approach3_ear():
    """Save EAR detection configuration."""
    print("\n" + "=" * 60)
    print("APPROACH 3: EAR Detection Configuration")
    print("=" * 60)
    
    # EAR threshold configuration
    # Based on research: EAR < 0.22 indicates closed/drowsy eyes
    ear_config = {
        "threshold": CONFIG["EAR_THRESHOLD"],
        "consecutive_frames": CONFIG["CONSECUTIVE_FRAMES"],
        "description": "Eye Aspect Ratio based detection using MediaPipe Face Mesh"
    }
    
    with open(EAR_CONFIG_PATH, 'wb') as f:
        pickle.dump(ear_config, f)
    
    # Metrics (EAR doesn't need training, uses geometric calculation)
    metrics = {
        "model_name": "EAR_Detector",
        "approach": "approach3",
        "optimal_threshold": ear_config["threshold"],
        "consecutive_frames": ear_config["consecutive_frames"],
        "note": "EAR uses MediaPipe Face Mesh - no neural network training needed"
    }
    
    with open(EAR_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n📊 EAR Configuration:")
    print(f"   Threshold: {ear_config['threshold']}")
    print(f"   Consecutive frames: {ear_config['consecutive_frames']}")
    print(f"\n📁 Saved: {EAR_CONFIG_PATH}")
    
    return ear_config, metrics

# %% [code]
# =============================================================================
# CELL 6: RUN ALL TRAINING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 STARTING TRAINING - ALL 3 APPROACHES")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start = time.time()
    results = {}
    
    # Train Approach 1: Custom CNN
    try:
        cnn_model, cnn_metrics = train_approach1_cnn()
        results["approach1"] = {"status": "✅ SUCCESS", "accuracy": f"{cnn_metrics['test_accuracy']:.2%}"}
    except Exception as e:
        print(f"❌ Approach 1 failed: {e}")
        import traceback
        traceback.print_exc()
        results["approach1"] = {"status": "❌ FAILED", "error": str(e)}
    
    # Train Approach 2: EfficientNet
    try:
        effnet_model, effnet_metrics = train_approach2_efficientnet()
        results["approach2"] = {"status": "✅ SUCCESS", "accuracy": f"{effnet_metrics['test_accuracy']:.2%}"}
    except Exception as e:
        print(f"❌ Approach 2 failed: {e}")
        import traceback
        traceback.print_exc()
        results["approach2"] = {"status": "❌ FAILED", "error": str(e)}
    
    # Configure Approach 3: EAR
    try:
        ear_config, ear_metrics = train_approach3_ear()
        results["approach3"] = {"status": "✅ SUCCESS", "threshold": ear_config["threshold"]}
    except Exception as e:
        print(f"❌ Approach 3 failed: {e}")
        results["approach3"] = {"status": "❌ FAILED", "error": str(e)}
    
    total_time = time.time() - total_start
    
    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\n📊 Results Summary:")
    for approach, result in results.items():
        print(f"   {approach}: {result}")
    
    print(f"\n📦 FILES TO DOWNLOAD:")
    print(f"   (Click the Output tab → /kaggle/working/)")
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"   → {f} ({size/1024/1024:.1f} MB)" if size > 1024*1024 else f"   → {f} ({size/1024:.1f} KB)")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Download all files from /kaggle/working/")
    print(f"   2. Place in your local: backend/models/")
    print(f"   3. Run: python -m api.app")
    print(f"   4. Open: http://localhost:5173")
