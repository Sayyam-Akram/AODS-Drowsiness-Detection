"""
RGB to Infrared-like Preprocessing Pipeline
Converts RGB webcam/upload images to match infrared training data distribution.
"""

import cv2
import numpy as np


def rgb_to_infrared_preprocessing(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert RGB webcam/upload to infrared-like appearance.
    MUST apply before inference on ANY real-time input.
    
    Args:
        rgb_image: BGR image from OpenCV (webcam/upload)
    
    Returns:
        infrared_like: Preprocessed image matching training distribution
    """
    # Step 1: Convert to grayscale
    if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb_image
    
    # Step 2: Histogram equalization (match infrared contrast)
    equalized = cv2.equalizeHist(gray)
    
    # Step 3: Slight blur (simulate infrared sensor characteristics)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    
    # Step 4: Brightness/contrast adjustment for infrared simulation
    alpha = 1.2  # Contrast control
    beta = 10    # Brightness control
    adjusted = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
    
    # Step 5: Convert back to 3-channel (model expects RGB format)
    infrared_like = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)
    
    return infrared_like


def preprocess_for_model(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Full preprocessing pipeline for CNN model inference.
    
    Args:
        image: Input BGR image
        target_size: (width, height) tuple
    
    Returns:
        Preprocessed image ready for model input
    """
    # Apply RGB to infrared preprocessing
    infrared = rgb_to_infrared_preprocessing(image)
    
    # Resize to target size
    resized = cv2.resize(infrared, target_size)
    
    # Normalize to [0, 1] for CNN
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized


def preprocess_for_efficientnet(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Preprocessing for EfficientNet model.
    
    IMPORTANT: EfficientNet model has preprocess_input layer BUILT-IN.
    So we should NOT apply /255 normalization - the model expects raw pixels [0-255]
    and applies its own preprocessing internally.
    
    Args:
        image: Input BGR image
        target_size: (width, height) tuple
    
    Returns:
        Preprocessed image ready for EfficientNet model input
    """
    # Apply RGB to infrared preprocessing
    infrared = rgb_to_infrared_preprocessing(image)
    
    # Resize to target size
    resized = cv2.resize(infrared, target_size)
    
    # Return as float32 but WITHOUT /255 normalization
    # The EfficientNet model's first layer is preprocess_input which expects [0-255]
    return resized.astype(np.float32)


def preprocess_batch(images: list, target_size: tuple) -> np.ndarray:
    """
    Preprocess a batch of images.
    
    Args:
        images: List of BGR images
        target_size: (width, height) tuple
    
    Returns:
        Batch of preprocessed images with shape (N, H, W, 3)
    """
    processed = [preprocess_for_model(img, target_size) for img in images]
    return np.array(processed)
