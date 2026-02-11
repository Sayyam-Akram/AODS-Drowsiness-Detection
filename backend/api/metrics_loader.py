"""
Metrics Loader
Load and serve saved model metrics.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def load_all_metrics() -> dict:
    """
    Load metrics for all trained models.
    
    Returns:
        Dictionary with metrics for each approach
    """
    metrics = {
        "approach1": None,
        "approach2": None,
        "approach3": None
    }
    
    # Approach 1: CNN
    cnn_metrics_path = Path(config.CNN_METRICS_PATH)
    if cnn_metrics_path.exists():
        try:
            with open(cnn_metrics_path, 'r') as f:
                metrics["approach1"] = json.load(f)
        except Exception as e:
            print(f"Error loading CNN metrics: {e}")
    
    # Approach 2: EfficientNet
    effnet_metrics_path = Path(config.EFFICIENTNET_METRICS_PATH)
    if effnet_metrics_path.exists():
        try:
            with open(effnet_metrics_path, 'r') as f:
                metrics["approach2"] = json.load(f)
        except Exception as e:
            print(f"Error loading EfficientNet metrics: {e}")
    
    # Approach 3: EAR
    ear_metrics_path = Path(config.EAR_METRICS_PATH)
    if ear_metrics_path.exists():
        try:
            with open(ear_metrics_path, 'r') as f:
                metrics["approach3"] = json.load(f)
        except Exception as e:
            print(f"Error loading EAR metrics: {e}")
    
    return metrics


def get_metrics_summary() -> dict:
    """
    Get simplified metrics summary for frontend display.
    
    Returns:
        Dictionary with key metrics for each approach
    """
    all_metrics = load_all_metrics()
    
    summary = {}
    
    for approach, metrics in all_metrics.items():
        if metrics:
            summary[approach] = {
                "model_name": metrics.get("model_name", "Unknown"),
                "accuracy": metrics.get("test_accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0),
                "inference_time_ms": metrics.get("inference_time_ms", 0)
            }
        else:
            # Default values for untrained models
            summary[approach] = {
                "model_name": f"Approach {approach[-1]} (Not Trained)",
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "inference_time_ms": 0
            }
    
    return summary


def get_model_names() -> dict:
    """
    Get model names for each approach.
    """
    return {
        "approach1": "Custom CNN (DrowsyNet)",
        "approach2": "EfficientNet Transfer",
        "approach3": "EAR Detection (MediaPipe)"
    }
