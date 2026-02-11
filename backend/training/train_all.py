"""
Master Training Script
Runs all three approaches sequentially.
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_approach1_cnn import train as train_cnn
from training.train_approach2_efficientnet import train as train_efficientnet
from training.train_approach3_ear import train as train_ear


def main():
    parser = argparse.ArgumentParser(description="Train Drowsiness Detection Models")
    parser.add_argument(
        "--approach", 
        type=str, 
        choices=["all", "cnn", "efficientnet", "ear"],
        default="all",
        help="Which approach to train (default: all)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("   DROWSINESS DETECTION SYSTEM - TRAINING")
    print("=" * 70)
    
    results = {}
    
    if args.approach in ["all", "cnn"]:
        print("\n\n" + "🔹" * 35)
        print("   TRAINING APPROACH 1: Custom CNN")
        print("🔹" * 35)
        _, metrics = train_cnn()
        results["approach1"] = metrics
    
    if args.approach in ["all", "efficientnet"]:
        print("\n\n" + "🔹" * 35)
        print("   TRAINING APPROACH 2: EfficientNet")
        print("🔹" * 35)
        _, metrics = train_efficientnet()
        results["approach2"] = metrics
    
    if args.approach in ["all", "ear"]:
        print("\n\n" + "🔹" * 35)
        print("   TRAINING APPROACH 3: EAR Detection")
        print("🔹" * 35)
        _, metrics = train_ear()
        results["approach3"] = metrics
    
    # Summary
    print("\n\n" + "=" * 70)
    print("   TRAINING SUMMARY")
    print("=" * 70)
    
    for approach, metrics in results.items():
        print(f"\n{approach.upper()}:")
        print(f"   Model: {metrics.get('model_name', 'N/A')}")
        print(f"   Accuracy: {metrics.get('test_accuracy', 0):.2%}")
        print(f"   F1-Score: {metrics.get('f1_score', 0):.2%}")
        print(f"   Inference: {metrics.get('inference_time_ms', 0):.1f}ms")
    
    print("\n" + "=" * 70)
    print("   ALL TRAINING COMPLETE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
