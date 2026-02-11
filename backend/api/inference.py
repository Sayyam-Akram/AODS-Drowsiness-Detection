"""
Unified Inference Engine
Loads all models and provides prediction interface.
"""

import os
import sys
import base64
import pickle
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

# Defer TensorFlow import for faster startup
_tf = None
_keras = None
_cnn_model = None
_efficientnet_model = None

def _load_tensorflow():
    """Lazy load TensorFlow."""
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf
        _tf = tf
        _keras = tf.keras
        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    return _tf, _keras

import config
from utils.preprocessing import rgb_to_infrared_preprocessing, preprocess_for_model, preprocess_for_efficientnet
from utils.eye_detection import EyeDetector
from utils.ear_calculator import EARCalculator
from utils.visualization import Visualizer


class InferenceEngine:
    """
    Unified inference engine for all three approaches.
    """
    
    def __init__(self):
        self.models = {}
        self.eye_detector = None
        self.ear_calculator = None
        self.ear_config = None
        self._models_loaded = False
    
    def load_models(self):
        """
        Load all trained models.
        """
        if self._models_loaded:
            return
        
        print("🔄 Loading models...")
        
        # Load TensorFlow models
        tf, keras = _load_tensorflow()
        
        # Approach 1: Custom CNN
        cnn_path = Path(config.CNN_MODEL_PATH)
        if cnn_path.exists():
            try:
                self.models["approach1"] = keras.models.load_model(str(cnn_path))
                print(f"   ✅ Loaded Approach 1: Custom CNN")
            except Exception as e:
                print(f"   ❌ Failed to load CNN: {e}")
        else:
            print(f"   ⚠️ CNN model not found: {cnn_path}")
        
        # Approach 2: EfficientNet
        effnet_path = Path(config.EFFICIENTNET_MODEL_PATH)
        if effnet_path.exists():
            try:
                self.models["approach2"] = keras.models.load_model(str(effnet_path))
                print(f"   ✅ Loaded Approach 2: EfficientNet")
            except Exception as e:
                print(f"   ❌ Failed to load EfficientNet: {e}")
        else:
            print(f"   ⚠️ EfficientNet model not found: {effnet_path}")
        
        # Approach 3: EAR Configuration
        ear_path = Path(config.EAR_MODEL_PATH)
        if ear_path.exists():
            try:
                with open(ear_path, 'rb') as f:
                    self.ear_config = pickle.load(f)
                self.models["approach3"] = "ear"  # Marker
                print(f"   ✅ Loaded Approach 3: EAR Detector")
            except Exception as e:
                print(f"   ❌ Failed to load EAR config: {e}")
        else:
            print(f"   ⚠️ EAR config not found: {ear_path}")
            # Use default config
            self.ear_config = {
                "threshold": 0.22,
                "consecutive_frames": 5
            }
            self.models["approach3"] = "ear"
        
        # Initialize eye detector and EAR calculator
        self.eye_detector = EyeDetector()
        self.ear_calculator = EARCalculator(
            threshold=self.ear_config.get("threshold", 0.22),
            consecutive_frames=self.ear_config.get("consecutive_frames", 5)
        )
        
        self._models_loaded = True
        print(f"✅ Models loaded: {list(self.models.keys())}")
    
    def predict_image(self, image: np.ndarray, approach: str = "approach1") -> dict:
        """
        Predict drowsiness from a single image.
        
        CNN/EfficientNet: Work directly on the image (eye crops or full face)
        EAR: Requires face detection for eye landmarks
        
        Args:
            image: BGR image (from webcam or file)
            approach: Which approach to use
        
        Returns:
            Prediction results dictionary
        """
        start_time = time.time()
        
        if not self._models_loaded:
            self.load_models()
        
        if approach not in self.models:
            return {
                "success": False,
                "error": f"Model {approach} not loaded"
            }
        
        # EAR approach - needs face detection for eye landmarks
        if approach == "approach3":
            return self._predict_with_ear(image, start_time)
        
        # CNN/EfficientNet approach - work directly on image
        return self._predict_with_cnn(image, approach, start_time)
    
    def _predict_with_ear(self, image: np.ndarray, start_time: float) -> dict:
        """
        EAR-based prediction. Requires full face for landmark detection.
        """
        # Detect eyes via MediaPipe
        eye_data = self.eye_detector.process_frame(image)
        
        if not eye_data.get("face_detected"):
            return {
                "success": True,
                "is_drowsy": False,
                "confidence": 0.0,
                "message": "No face detected - EAR needs full face image",
                "face_detected": False,
                "approach": "approach3",
                "processing_time_ms": (time.time() - start_time) * 1000
            }
        
        # Calculate EAR
        left_eye = eye_data.get("left_eye_points", [])
        right_eye = eye_data.get("right_eye_points", [])
        ear_data = self.ear_calculator.process_frame(left_eye, right_eye)
        
        # EAR-based detection
        result = self._predict_ear(ear_data)
        
        # Add EAR-specific data
        result.update({
            "success": True,
            "face_detected": True,
            "approach": "approach3",
            "left_ear": ear_data.get("left_ear", 0),
            "right_ear": ear_data.get("right_ear", 0),
            "avg_ear": ear_data.get("avg_ear", 0),
            "ear_threshold": self.ear_calculator.threshold,
            "processing_time_ms": (time.time() - start_time) * 1000
        })
        
        # Generate annotated image with eye boxes
        annotated = Visualizer.annotate_frame(
            image, eye_data, ear_data, result, "approach3"
        )
        result["annotated_image"] = self._encode_image(annotated)
        
        return result
    
    def _predict_with_cnn(self, image: np.ndarray, approach: str, start_time: float) -> dict:
        """
        CNN/EfficientNet prediction.
        
        LOGIC:
        1. Try to detect face in image
        2. If face found → extract eye regions → predict on eye crops
        3. If no face found → assume image IS an eye → predict directly
        
        This way it works for both:
        - Full face images (webcam/video) → extracts eyes
        - Eye-only images → uses directly
        """
        model = self.models.get(approach)
        
        if model is None:
            return {
                "success": False,
                "error": f"Model {approach} not loaded"
            }
        
        # Get target size and preprocessing function based on approach
        if approach == "approach1":
            target_size = config.CNN_IMG_SIZE
            preprocess_fn = preprocess_for_model  # Standard preprocessing for CNN
        else:
            target_size = config.EFFICIENTNET_IMG_SIZE
            preprocess_fn = preprocess_for_efficientnet  # No /255 for EfficientNet (built-in)
        
        # Try to detect face and extract eye regions
        eye_data = self.eye_detector.process_frame(image)
        
        predictions_list = []
        used_eye_regions = False
        eye_regions_for_display = []
        
        if eye_data.get("face_detected"):
            # Face detected! Extract eye regions and predict on those
            left_region = eye_data.get("left_region")
            right_region = eye_data.get("right_region")
            
            if left_region is not None and left_region.size > 0:
                processed = preprocess_fn(left_region, target_size)
                input_batch = np.expand_dims(processed, axis=0)
                pred = model.predict(input_batch, verbose=0)
                predictions_list.append(pred[0])
                eye_regions_for_display.append(("Left Eye", left_region))
                used_eye_regions = True
            
            if right_region is not None and right_region.size > 0:
                processed = preprocess_fn(right_region, target_size)
                input_batch = np.expand_dims(processed, axis=0)
                pred = model.predict(input_batch, verbose=0)
                predictions_list.append(pred[0])
                eye_regions_for_display.append(("Right Eye", right_region))
                used_eye_regions = True
        
        # If no face/eyes detected, use the full image directly
        # (assume user uploaded an eye-only image)
        if not predictions_list:
            processed = preprocess_fn(image, target_size)
            input_batch = np.expand_dims(processed, axis=0)
            pred = model.predict(input_batch, verbose=0)
            predictions_list.append(pred[0])
        
        # Average predictions from both eyes (or single prediction)
        avg_prediction = np.mean(predictions_list, axis=0)
        
        # predictions shape: [awake_prob, drowsy_prob]
        awake_prob = float(avg_prediction[0])
        drowsy_prob = float(avg_prediction[1])
        
        is_drowsy = drowsy_prob > awake_prob
        confidence = drowsy_prob if is_drowsy else awake_prob
        
        result = {
            "success": True,
            "is_drowsy": is_drowsy,
            "confidence": confidence,
            "awake_probability": awake_prob,
            "drowsy_probability": drowsy_prob,
            "face_detected": used_eye_regions,
            "used_eye_regions": used_eye_regions,
            "approach": approach,
            "processing_time_ms": (time.time() - start_time) * 1000
        }
        
        # Annotate image
        if used_eye_regions and eye_data:
            # Full face with eye detection - annotate with eye boxes
            annotated = Visualizer.annotate_frame(
                image, eye_data, {}, result, approach
            )
        else:
            # Eye-only image - just add status bar
            annotated = self._annotate_cnn_result(image, result)
        
        result["annotated_image"] = self._encode_image(annotated)
        
        return result
    
    def _annotate_cnn_result(self, image: np.ndarray, result: dict) -> np.ndarray:
        """Simple annotation for CNN results - just status bar."""
        annotated = image.copy()
        h, w = annotated.shape[:2]
        
        is_drowsy = result.get("is_drowsy", False)
        confidence = result.get("confidence", 0)
        
        if is_drowsy:
            bg_color = (0, 0, 200)  # Red
            text = f"DROWSY! - {confidence:.0%}"
        else:
            bg_color = (0, 180, 0)  # Green  
            text = f"AWAKE - {confidence:.0%}"
        
        cv2.rectangle(annotated, (0, 0), (w, 40), bg_color, -1)
        cv2.putText(annotated, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2)
        
        return annotated
    
    def _predict_cnn(self, image: np.ndarray, eye_data: dict, approach: str) -> dict:
        """
        Predict using CNN or EfficientNet model.
        Uses eye region crop - the model was trained on eye images!
        """
        model = self.models.get(approach)
        
        if model is None:
            return {"is_drowsy": False, "confidence": 0.0}
        
        # Get image size based on approach
        if approach == "approach1":
            target_size = config.CNN_IMG_SIZE
        else:
            target_size = config.EFFICIENTNET_IMG_SIZE
        
        # IMPORTANT: Use eye region, not full face!
        # The model was trained on eye images
        left_region = eye_data.get("left_region")
        right_region = eye_data.get("right_region")
        
        predictions_list = []
        
        # Predict on left eye if available
        if left_region is not None and left_region.size > 0:
            processed = preprocess_for_model(left_region, target_size)
            input_batch = np.expand_dims(processed, axis=0)
            pred = model.predict(input_batch, verbose=0)
            predictions_list.append(pred[0])
        
        # Predict on right eye if available
        if right_region is not None and right_region.size > 0:
            processed = preprocess_for_model(right_region, target_size)
            input_batch = np.expand_dims(processed, axis=0)
            pred = model.predict(input_batch, verbose=0)
            predictions_list.append(pred[0])
        
        # If no eyes detected, fall back to full image
        if not predictions_list:
            processed = preprocess_for_model(image, target_size)
            input_batch = np.expand_dims(processed, axis=0)
            predictions = model.predict(input_batch, verbose=0)
        else:
            # Average predictions from both eyes
            predictions = [np.mean(predictions_list, axis=0)]
        
        # predictions shape: (1, 2) for [awake, sleepy]
        awake_prob = float(predictions[0][0])
        drowsy_prob = float(predictions[0][1])
        
        is_drowsy = drowsy_prob > awake_prob
        confidence = drowsy_prob if is_drowsy else awake_prob
        
        return {
            "is_drowsy": is_drowsy,
            "confidence": confidence,
            "awake_probability": awake_prob,
            "drowsy_probability": drowsy_prob
        }
    
    def _predict_ear(self, ear_data: dict) -> dict:
        """
        Predict using EAR-based detection.
        For single images, detect based on threshold directly.
        For video/webcam, the consecutive frame counter helps reduce false positives.
        """
        avg_ear = ear_data.get("avg_ear", 0.3)
        threshold = self.ear_calculator.threshold
        
        # For EAR: lower EAR = more closed eyes = more drowsy
        # Detect drowsy if EAR is below threshold
        is_drowsy = avg_ear < threshold
        
        # Calculate confidence: how far from threshold
        # If drowsy (below threshold): how much below
        # If awake (above threshold): how much above
        if is_drowsy:
            # Eyes closed - confidence based on how much below threshold
            confidence = min(1.0, max(0.5, 1.0 - (avg_ear / threshold)))
        else:
            # Eyes open - confidence based on how much above threshold  
            confidence = min(1.0, max(0.5, avg_ear / threshold))
        
        return {
            "is_drowsy": is_drowsy,
            "confidence": confidence,
            "frame_counter": ear_data.get("frame_counter", 0)
        }
    
    def predict_video_frame(self, frame: np.ndarray, approach: str = "approach1") -> dict:
        """
        Predict on a video frame (maintains state for EAR).
        """
        return self.predict_image(frame, approach)
    
    def reset_state(self):
        """
        Reset EAR frame counter for new video/session.
        """
        if self.ear_calculator:
            self.ear_calculator.reset()
    
    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode image to base64 string.
        """
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    def get_loaded_models(self) -> list:
        """
        Get list of loaded model names.
        """
        return list(self.models.keys())
    
    def close(self):
        """
        Release resources.
        """
        if self.eye_detector:
            self.eye_detector.close()


# Singleton instance
_engine = None

def get_engine() -> InferenceEngine:
    """
    Get singleton inference engine.
    """
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
        _engine.load_models()
    return _engine
