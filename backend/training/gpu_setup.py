"""
GPU Setup and Configuration for TensorFlow
Optimized for NVIDIA GTX 1660 Super (6GB VRAM)
"""

import os
import tensorflow as tf


def setup_gpu():
    """
    Configure TensorFlow for optimal GPU usage.
    - Enable memory growth (prevent OOM)
    - Enable mixed precision (faster training)
    - Set visible devices
    
    Returns:
        True if GPU is available and configured
    """
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Get available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision for faster training
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            print(f"✅ GPU Configuration Successful!")
            print(f"   Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            print(f"   Memory growth: Enabled")
            print(f"   Mixed precision: Enabled (float16)")
            
            return True
        except RuntimeError as e:
            print(f"❌ GPU Configuration Error: {e}")
            return False
    else:
        print("⚠️ No GPU detected. Training will use CPU (much slower).")
        return False


def get_gpu_info():
    """
    Get detailed GPU information.
    
    Returns:
        Dictionary with GPU details
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    info = {
        "gpu_available": len(gpus) > 0,
        "num_gpus": len(gpus),
        "gpu_names": [gpu.name for gpu in gpus],
        "tensorflow_version": tf.__version__,
        "mixed_precision": tf.keras.mixed_precision.global_policy().name
    }
    
    # Try to get memory info (requires CUDA)
    if gpus:
        try:
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            info["gpu_details"] = gpu_details
        except:
            pass
    
    return info


def verify_gpu():
    """
    Verify GPU is being used by running a simple computation.
    
    Returns:
        True if GPU computation successful
    """
    try:
        with tf.device('/GPU:0'):
            # Simple matrix multiplication
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            
        print("✅ GPU Verification: Matrix computation successful!")
        return True
    except Exception as e:
        print(f"❌ GPU Verification Failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("GPU Setup and Verification")
    print("=" * 50)
    
    # Setup GPU
    gpu_ready = setup_gpu()
    
    # Get info
    print("\nGPU Information:")
    info = get_gpu_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Verify
    if gpu_ready:
        print("\nVerifying GPU...")
        verify_gpu()
    
    print("=" * 50)
