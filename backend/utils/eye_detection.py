"""
MediaPipe-based Eye Detection
Uses MediaPipe Face Landmarker (Tasks API) for MediaPipe 0.10.30+
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import os


class EyeDetector:
    """
    Detect and extract eye regions using MediaPipe Face Landmarker.
    Compatible with MediaPipe 0.10.30+ (Tasks API)
    """
    
    # EAR calculation indices (6 points per eye based on Face Landmarker)
    # These map to the 478 face landmarks
    LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
    
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Face Landmarker.
        
        Args:
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        self.min_detection_confidence = min_detection_confidence
        
        # Try to import MediaPipe
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError:
            raise ImportError("MediaPipe not installed. Run: pip install mediapipe")
        
        # Download model file if not exists
        model_path = self._get_model_path()
        
        # Create the face landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.mp = mp
    
    def _get_model_path(self) -> str:
        """Download or locate the face landmarker model."""
        import urllib.request
        
        # Model path in backend/models directory
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        model_path = os.path.join(models_dir, "face_landmarker.task")
        
        if not os.path.exists(model_path):
            print("📥 Downloading Face Landmarker model...")
            os.makedirs(models_dir, exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("✅ Face Landmarker model downloaded")
        
        return model_path
    
    def detect_face(self, image: np.ndarray) -> Optional[object]:
        """
        Detect face and return landmarks.
        
        Args:
            image: BGR image
        
        Returns:
            Face landmark detection result or None if no face detected
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect
        result = self.detector.detect(mp_image)
        
        if result.face_landmarks and len(result.face_landmarks) > 0:
            return result.face_landmarks[0]  # Return first face's landmarks
        return None
    
    def get_eye_landmarks(self, landmarks, image_shape: Tuple[int, int]) -> Tuple[List, List]:
        """
        Extract left and right eye landmark coordinates.
        
        Args:
            landmarks: MediaPipe face landmarks (list of NormalizedLandmark)
            image_shape: (height, width) of image
        
        Returns:
            (left_eye_points, right_eye_points) as pixel coordinates
        """
        h, w = image_shape[:2]
        
        left_eye = []
        for idx in self.LEFT_EYE_EAR:
            lm = landmarks[idx]
            left_eye.append((int(lm.x * w), int(lm.y * h)))
        
        right_eye = []
        for idx in self.RIGHT_EYE_EAR:
            lm = landmarks[idx]
            right_eye.append((int(lm.x * w), int(lm.y * h)))
        
        return left_eye, right_eye
    
    def extract_eye_region(self, image: np.ndarray, eye_points: List, 
                           padding: int = 10) -> Optional[np.ndarray]:
        """
        Extract eye region from image.
        
        Args:
            image: BGR image
            eye_points: List of (x, y) coordinates
            padding: Padding around eye region
        
        Returns:
            Cropped eye region or None
        """
        if not eye_points:
            return None
        
        points = np.array(eye_points)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Add padding
        h, w = image.shape[:2]
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Crop eye region
        eye_region = image[y_min:y_max, x_min:x_max]
        
        if eye_region.size == 0:
            return None
        
        return eye_region
    
    def get_eye_bounding_box(self, eye_points: List, padding: int = 10,
                             image_shape: Tuple = None) -> Tuple[int, int, int, int]:
        """
        Get bounding box for eye region.
        
        Args:
            eye_points: List of (x, y) coordinates
            padding: Padding around eye region
            image_shape: (height, width) to clip coordinates
        
        Returns:
            (x_min, y_min, x_max, y_max)
        """
        points = np.array(eye_points)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        x_min -= padding
        y_min -= padding
        x_max += padding
        y_max += padding
        
        if image_shape:
            h, w = image_shape[:2]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
        
        return int(x_min), int(y_min), int(x_max), int(y_max)
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame and extract all eye information.
        
        Args:
            frame: BGR image
        
        Returns:
            Dictionary with eye data or empty dict if no face detected
        """
        landmarks = self.detect_face(frame)
        
        if landmarks is None:
            return {"face_detected": False}
        
        left_eye, right_eye = self.get_eye_landmarks(landmarks, frame.shape)
        
        left_bbox = self.get_eye_bounding_box(left_eye, padding=15, image_shape=frame.shape)
        right_bbox = self.get_eye_bounding_box(right_eye, padding=15, image_shape=frame.shape)
        
        left_region = self.extract_eye_region(frame, left_eye, padding=15)
        right_region = self.extract_eye_region(frame, right_eye, padding=15)
        
        return {
            "face_detected": True,
            "left_eye_points": left_eye,
            "right_eye_points": right_eye,
            "left_bbox": left_bbox,
            "right_bbox": right_bbox,
            "left_region": left_region,
            "right_region": right_region,
            "landmarks": landmarks
        }
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'detector'):
            self.detector.close()
