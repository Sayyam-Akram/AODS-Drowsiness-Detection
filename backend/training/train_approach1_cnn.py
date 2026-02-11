"""
Approach 1: Custom Deep CNN Training Script
Model: Custom_DrowsyNet

Architecture:
- 5 Convolutional Blocks (32→64→128→256→512 filters)
- BatchNormalization + Dropout for regularization
- GlobalAveragePooling + Dense layers
- Input: 80x80x3 (infrared images)
- Output: 2 classes (awake, sleepy)

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
from sklearn.metrics import classification_report, confusion_matrix

# Import config and GPU setup
import config
from training.gpu_setup import setup_gpu


def create_model(input_shape=(80, 80, 3), num_classes=2):
    """
    Create Custom_DrowsyNet architecture.
    
    5 convolutional blocks with increasing filters,
    followed by GlobalAveragePooling and Dense layers.
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
    x = layers.Dropout(0.25)(x)
    
    # Block 3: 128 filters
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 4: 256 filters
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 5: 512 filters
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    
    # Output layer (float32 for mixed precision)
    x = layers.Dense(num_classes, dtype='float32')(x)
    outputs = layers.Activation('softmax', dtype='float32')(x)
    
    model = keras.Model(inputs, outputs, name="Custom_DrowsyNet")
    
    return model


def create_data_generators():
    """
    Create data generators with augmentation for training.
    """
    # Training data augmentation (for infrared images)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation/Test - only rescaling
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load data
    img_size = config.CNN_IMG_SIZE
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


def create_callbacks():
    """
    Create training callbacks.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    callback_list = [
        # Early stopping - restore best weights
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model
        callbacks.ModelCheckpoint(
            filepath=config.CNN_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir=f'logs/cnn_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    return callback_list


def evaluate_model(model, test_generator):
    """
    Evaluate model and generate metrics.
    """
    # Predict on test set
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Classification report
    report = classification_report(
        true_classes, 
        predicted_classes,
        target_names=config.CLASSES,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Calculate metrics
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    
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
        "input_shape": list(config.CNN_IMG_SIZE) + [3],
        "inference_time_ms": 0  # Will be calculated separately
    }
    
    return metrics


def measure_inference_time(model):
    """
    Measure average inference time.
    """
    # Create dummy input
    dummy_input = np.random.rand(1, *config.CNN_IMG_SIZE, 3).astype(np.float32)
    
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
    Main training function.
    """
    print("=" * 60)
    print("APPROACH 1: Custom Deep CNN (Custom_DrowsyNet)")
    print("=" * 60)
    
    # Setup GPU
    print("\n📊 Setting up GPU...")
    setup_gpu()
    
    # Create model
    print("\n🏗️ Creating model...")
    model = create_model(
        input_shape=(*config.CNN_IMG_SIZE, 3),
        num_classes=2
    )
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.CNN_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    
    # Create data generators
    print("\n📁 Loading data...")
    train_gen, val_gen, test_gen = create_data_generators()
    print(f"   Training samples: {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    print(f"   Test samples: {test_gen.samples}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    
    # Create callbacks
    callback_list = create_callbacks()
    
    # Train
    print(f"\n🚀 Starting training for {config.CNN_EPOCHS} epochs...")
    print(f"   Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    
    start_time = time.time()
    
    history = model.fit(
        train_gen,
        epochs=config.CNN_EPOCHS,
        validation_data=val_gen,
        callbacks=callback_list,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
    
    # Evaluate
    print("\n📊 Evaluating model...")
    metrics = evaluate_model(model, test_gen)
    
    # Measure inference time
    print("\n⏱️ Measuring inference time...")
    metrics["inference_time_ms"] = measure_inference_time(model)
    
    # Add training history
    metrics["training_history"] = {
        "epochs": len(history.history['loss']),
        "final_train_accuracy": float(history.history['accuracy'][-1]),
        "final_val_accuracy": float(history.history['val_accuracy'][-1]),
        "final_train_loss": float(history.history['loss'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1]),
        "training_time_minutes": training_time / 60
    }
    
    # Save metrics
    with open(config.CNN_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📁 Model saved: {config.CNN_MODEL_PATH}")
    print(f"📊 Metrics saved: {config.CNN_METRICS_PATH}")
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
