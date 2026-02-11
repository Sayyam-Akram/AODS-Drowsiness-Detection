"""
Approach 2: Transfer Learning with EfficientNetB1
Using Keras built-in EfficientNet (simpler than HuggingFace)

Two-Phase Training:
1. Feature Extraction: Freeze base, train head only
2. Fine-Tuning: Unfreeze top layers, train with lower LR

Optimized for: NVIDIA GTX 1660 Super (6GB VRAM)
"""

import os
import sys
import json
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB1
from sklearn.metrics import classification_report, confusion_matrix

# Import config and GPU setup
import config
from training.gpu_setup import setup_gpu


def create_model(input_shape=(96, 96, 3), num_classes=2):
    """
    Create EfficientNetB1 transfer learning model.
    
    Uses pretrained ImageNet weights with custom classification head.
    """
    # Load pretrained EfficientNetB1
    base_model = EfficientNetB1(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=input_shape)
    
    # Preprocessing for EfficientNet
    x = keras.applications.efficientnet.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer (float32 for mixed precision)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = keras.Model(inputs, outputs, name="EfficientNet_Transfer")
    
    return model, base_model


def create_data_generators():
    """
    Create data generators for EfficientNet.
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation/Test - no augmentation
    val_test_datagen = ImageDataGenerator()
    
    # Load data
    img_size = config.EFFICIENTNET_IMG_SIZE
    batch_size = config.BATCH_SIZE
    
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=config.CLASSES,
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=config.CLASSES,
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=config.CLASSES,
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def create_callbacks(phase: str):
    """
    Create training callbacks for each phase.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    callback_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model
        callbacks.ModelCheckpoint(
            filepath=config.EFFICIENTNET_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard
        callbacks.TensorBoard(
            log_dir=f'logs/efficientnet_{phase}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    return callback_list


def evaluate_model(model, test_generator):
    """
    Evaluate model and generate metrics.
    """
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    report = classification_report(
        true_classes, 
        predicted_classes,
        target_names=config.CLASSES,
        output_dict=True
    )
    
    cm = confusion_matrix(true_classes, predicted_classes)
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    
    metrics = {
        "model_name": "EfficientNet_Transfer",
        "approach": "approach2",
        "base_model": "EfficientNetB1",
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "precision": float(report['weighted avg']['precision']),
        "recall": float(report['weighted avg']['recall']),
        "f1_score": float(report['weighted avg']['f1-score']),
        "confusion_matrix": cm.tolist(),
        "class_report": report,
        "input_shape": list(config.EFFICIENTNET_IMG_SIZE) + [3],
        "inference_time_ms": 0
    }
    
    return metrics


def measure_inference_time(model):
    """
    Measure average inference time.
    """
    dummy_input = np.random.rand(1, *config.EFFICIENTNET_IMG_SIZE, 3).astype(np.float32)
    
    # Warm up
    for _ in range(10):
        model.predict(dummy_input, verbose=0)
    
    # Measure
    times = []
    for _ in range(100):
        start = time.time()
        model.predict(dummy_input, verbose=0)
        times.append((time.time() - start) * 1000)
    
    return float(np.mean(times))


def train():
    """
    Two-phase training: Feature extraction + Fine-tuning
    """
    print("=" * 60)
    print("APPROACH 2: EfficientNet Transfer Learning")
    print("=" * 60)
    
    # Setup GPU
    print("\n📊 Setting up GPU...")
    setup_gpu()
    
    # Create model
    print("\n🏗️ Creating model...")
    model, base_model = create_model(
        input_shape=(*config.EFFICIENTNET_IMG_SIZE, 3),
        num_classes=2
    )
    
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    print(f"Base model layers: {len(base_model.layers)}")
    
    # Create data generators
    print("\n📁 Loading data...")
    train_gen, val_gen, test_gen = create_data_generators()
    print(f"   Training samples: {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    print(f"   Test samples: {test_gen.samples}")
    
    start_time = time.time()
    
    # ========================================
    # PHASE 1: Feature Extraction
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 1: Feature Extraction (Frozen Base)")
    print("=" * 60)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.EFFICIENTNET_PHASE1_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    phase1_callbacks = create_callbacks("phase1")
    
    print(f"\n🚀 Training for {config.EFFICIENTNET_PHASE1_EPOCHS} epochs...")
    
    history1 = model.fit(
        train_gen,
        epochs=config.EFFICIENTNET_PHASE1_EPOCHS,
        validation_data=val_gen,
        callbacks=phase1_callbacks,
        verbose=1
    )
    
    phase1_time = time.time() - start_time
    print(f"\n✅ Phase 1 completed in {phase1_time/60:.1f} minutes")
    
    # ========================================
    # PHASE 2: Fine-Tuning
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-Tuning (Unfreezing Top Layers)")
    print("=" * 60)
    
    # Unfreeze top layers of base model
    base_model.trainable = True
    
    # Freeze bottom layers, keep top 50 trainable
    fine_tune_at = max(0, len(base_model.layers) - 50)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    trainable_layers = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"   Trainable layers in base: {trainable_layers}/{len(base_model.layers)}")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.EFFICIENTNET_PHASE2_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    phase2_callbacks = create_callbacks("phase2")
    
    print(f"\n🚀 Fine-tuning for {config.EFFICIENTNET_PHASE2_EPOCHS} epochs...")
    
    phase2_start = time.time()
    
    history2 = model.fit(
        train_gen,
        epochs=config.EFFICIENTNET_PHASE2_EPOCHS,
        validation_data=val_gen,
        callbacks=phase2_callbacks,
        verbose=1
    )
    
    total_time = time.time() - start_time
    print(f"\n✅ Total training completed in {total_time/60:.1f} minutes")
    
    # Evaluate
    print("\n📊 Evaluating model...")
    metrics = evaluate_model(model, test_gen)
    
    # Measure inference time
    print("\n⏱️ Measuring inference time...")
    metrics["inference_time_ms"] = measure_inference_time(model)
    
    # Add training history
    metrics["training_history"] = {
        "phase1_epochs": len(history1.history['loss']),
        "phase2_epochs": len(history2.history['loss']),
        "total_epochs": len(history1.history['loss']) + len(history2.history['loss']),
        "final_train_accuracy": float(history2.history['accuracy'][-1]),
        "final_val_accuracy": float(history2.history['val_accuracy'][-1]),
        "training_time_minutes": total_time / 60,
        "fine_tuned_layers": trainable_layers
    }
    
    # Save metrics
    with open(config.EFFICIENTNET_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📁 Model saved: {config.EFFICIENTNET_MODEL_PATH}")
    print(f"📊 Metrics saved: {config.EFFICIENTNET_METRICS_PATH}")
    print(f"\n📈 Results:")
    print(f"   Test Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"   Precision: {metrics['precision']:.2%}")
    print(f"   Recall: {metrics['recall']:.2%}")
    print(f"   F1-Score: {metrics['f1_score']:.2%}")
    print(f"   Inference Time: {metrics['inference_time_ms']:.1f}ms")
    print("=" * 60)
    
    return model, metrics


if __name__ == "__main__":
    train()
