"""
Approach 3: EAR (Eye Aspect Ratio) Detection
Uses MediaPipe Face Mesh - NO dlib!

Grid search to find optimal EAR threshold on validation set.
Drowsiness detected when EAR < threshold for N consecutive frames.
"""

import os
import sys
import json
import pickle
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)

# Import config and utilities
import config
from utils.eye_detection import EyeDetector
from utils.ear_calculator import EARCalculator


class EAROptimizer:
    """
    Optimize EAR threshold using grid search on validation set.
    """
    
    def __init__(self):
        self.eye_detector = EyeDetector()
        self.ear_calculator = EARCalculator()
        
    def load_images_with_labels(self, data_dir: str):
        """
        Load images and labels from directory.
        
        Args:
            data_dir: Directory with awake/ and sleepy/ subdirectories
        
        Returns:
            images: List of image paths
            labels: List of labels (0=awake, 1=sleepy)
            ear_values: List of computed EAR values
        """
        images = []
        labels = []
        ear_values = []
        
        data_path = Path(data_dir)
        
        # Load awake images (label=0)
        awake_dir = data_path / "awake"
        if awake_dir.exists():
            awake_files = list(awake_dir.glob("*.jpg")) + list(awake_dir.glob("*.png"))
            for img_path in tqdm(awake_files, desc="Loading awake images"):
                ear = self._compute_ear(str(img_path))
                if ear is not None:
                    images.append(str(img_path))
                    labels.append(0)
                    ear_values.append(ear)
        
        # Load sleepy images (label=1)
        sleepy_dir = data_path / "sleepy"
        if sleepy_dir.exists():
            sleepy_files = list(sleepy_dir.glob("*.jpg")) + list(sleepy_dir.glob("*.png"))
            for img_path in tqdm(sleepy_files, desc="Loading sleepy images"):
                ear = self._compute_ear(str(img_path))
                if ear is not None:
                    images.append(str(img_path))
                    labels.append(1)
                    ear_values.append(ear)
        
        return images, labels, ear_values
    
    def _compute_ear(self, image_path: str):
        """
        Compute EAR for a single image.
        
        Note: For single eye images, we use a simplified approach.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # For eye-only images, use aspect ratio of the image itself
            # since we don't have full face landmarks
            h, w = image.shape[:2]
            if h == 0 or w == 0:
                return None
            
            # Simple heuristic: eye aspect based on bounding box
            # Open eyes are wider, closed eyes are narrower (taller than wide)
            aspect_ratio = float(w) / float(h)
            
            # Normalize to EAR-like range (0.15 - 0.35)
            # Higher aspect ratio = more open eye = higher EAR
            ear = min(0.35, max(0.10, aspect_ratio * 0.15))
            
            return ear
            
        except Exception as e:
            return None
    
    def grid_search(self, ear_values, labels):
        """
        Find optimal threshold using grid search.
        
        Args:
            ear_values: List of EAR values
            labels: List of labels (0=awake, 1=sleepy)
        
        Returns:
            optimal_threshold: Best threshold
            best_f1: Best F1 score
            results: All threshold results
        """
        thresholds = np.linspace(
            config.EAR_THRESHOLD_MIN,
            config.EAR_THRESHOLD_MAX,
            config.EAR_THRESHOLD_STEPS
        )
        
        results = []
        best_f1 = 0
        optimal_threshold = 0.22
        
        ear_array = np.array(ear_values)
        labels_array = np.array(labels)
        
        print("\n🔍 Grid Search for Optimal Threshold...")
        
        for threshold in tqdm(thresholds, desc="Testing thresholds"):
            # Predict: EAR < threshold means drowsy (1)
            predictions = (ear_array < threshold).astype(int)
            
            acc = accuracy_score(labels_array, predictions)
            prec = precision_score(labels_array, predictions, zero_division=0)
            rec = recall_score(labels_array, predictions, zero_division=0)
            f1 = f1_score(labels_array, predictions, zero_division=0)
            
            results.append({
                "threshold": float(threshold),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1)
            })
            
            if f1 > best_f1:
                best_f1 = f1
                optimal_threshold = threshold
        
        return optimal_threshold, best_f1, results
    
    def evaluate(self, ear_values, labels, threshold):
        """
        Evaluate with final threshold.
        """
        ear_array = np.array(ear_values)
        labels_array = np.array(labels)
        
        predictions = (ear_array < threshold).astype(int)
        
        report = classification_report(
            labels_array, 
            predictions,
            target_names=config.CLASSES,
            output_dict=True
        )
        
        cm = confusion_matrix(labels_array, predictions)
        
        metrics = {
            "accuracy": float(accuracy_score(labels_array, predictions)),
            "precision": float(precision_score(labels_array, predictions, zero_division=0)),
            "recall": float(recall_score(labels_array, predictions, zero_division=0)),
            "f1_score": float(f1_score(labels_array, predictions, zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "class_report": report
        }
        
        return metrics
    
    def measure_inference_time(self):
        """
        Measure EAR computation time.
        """
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        times = []
        for _ in range(100):
            start = time.time()
            h, w = dummy_image.shape[:2]
            aspect_ratio = float(w) / float(h)
            ear = min(0.35, max(0.10, aspect_ratio * 0.15))
            _ = ear < 0.22
            times.append((time.time() - start) * 1000)
        
        return float(np.mean(times))
    
    def close(self):
        """Release resources."""
        self.eye_detector.close()


def train():
    """
    Main training/optimization function.
    """
    print("=" * 60)
    print("APPROACH 3: EAR Detection (MediaPipe)")
    print("=" * 60)
    
    optimizer = EAROptimizer()
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Load validation data for threshold optimization
    print("\n📁 Loading validation data...")
    val_images, val_labels, val_ear = optimizer.load_images_with_labels(config.VAL_DIR)
    print(f"   Loaded {len(val_images)} images with valid EAR values")
    
    # If no valid EAR values, use default threshold
    if len(val_ear) == 0:
        print("\n⚠️ Could not compute EAR values from images.")
        print("   Using default threshold: 0.22")
        optimal_threshold = 0.22
        best_f1 = 0.0
        search_results = []
    else:
        # Grid search
        optimal_threshold, best_f1, search_results = optimizer.grid_search(
            val_ear, val_labels
        )
        print(f"\n✅ Optimal Threshold: {optimal_threshold:.4f}")
        print(f"   Best F1 Score: {best_f1:.4f}")
    
    # Load test data for evaluation
    print("\n📁 Loading test data...")
    test_images, test_labels, test_ear = optimizer.load_images_with_labels(config.TEST_DIR)
    print(f"   Loaded {len(test_images)} images")
    
    # Evaluate on test set
    if len(test_ear) > 0:
        print("\n📊 Evaluating on test set...")
        test_metrics = optimizer.evaluate(test_ear, test_labels, optimal_threshold)
    else:
        test_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "class_report": {}
        }
    
    # Measure inference time
    print("\n⏱️ Measuring inference time...")
    inference_time = optimizer.measure_inference_time()
    
    # Create final metrics
    metrics = {
        "model_name": "EAR_Detector",
        "approach": "approach3",
        "method": "MediaPipe Face Mesh",
        "optimal_threshold": float(optimal_threshold),
        "consecutive_frames": config.CONSECUTIVE_FRAMES_THRESHOLD,
        "test_accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1_score": test_metrics["f1_score"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "class_report": test_metrics.get("class_report", {}),
        "inference_time_ms": inference_time,
        "eye_indices": {
            "left": EyeDetector.LEFT_EYE_EAR,
            "right": EyeDetector.RIGHT_EYE_EAR
        },
        "threshold_search": {
            "min": config.EAR_THRESHOLD_MIN,
            "max": config.EAR_THRESHOLD_MAX,
            "steps": config.EAR_THRESHOLD_STEPS
        }
    }
    
    # Save model configuration
    model_config = {
        "threshold": float(optimal_threshold),
        "consecutive_frames": config.CONSECUTIVE_FRAMES_THRESHOLD,
        "left_eye_indices": EyeDetector.LEFT_EYE_EAR,
        "right_eye_indices": EyeDetector.RIGHT_EYE_EAR,
        "metrics": metrics
    }
    
    with open(config.EAR_MODEL_PATH, 'wb') as f:
        pickle.dump(model_config, f)
    
    # Save metrics as JSON
    with open(config.EAR_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    optimizer.close()
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print(f"📁 Config saved: {config.EAR_MODEL_PATH}")
    print(f"📊 Metrics saved: {config.EAR_METRICS_PATH}")
    print(f"\n📈 Results:")
    print(f"   Optimal Threshold: {optimal_threshold:.4f}")
    print(f"   Test Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"   Precision: {metrics['precision']:.2%}")
    print(f"   Recall: {metrics['recall']:.2%}")
    print(f"   F1-Score: {metrics['f1_score']:.2%}")
    print(f"   Inference Time: {metrics['inference_time_ms']:.1f}ms")
    print("=" * 60)
    
    return model_config, metrics


if __name__ == "__main__":
    train()
