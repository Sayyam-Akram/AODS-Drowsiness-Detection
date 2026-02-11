"""
Visualization Utilities
Draw bounding boxes, labels, and annotations on frames.
"""

import cv2
import numpy as np
from typing import Tuple, List


class Visualizer:
    """
    Draw annotations for drowsiness detection visualization.
    """
    
    # Colors (BGR format)
    GREEN = (0, 255, 0)      # Awake
    RED = (0, 0, 255)        # Drowsy
    YELLOW = (0, 255, 255)   # Warning
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (255, 128, 0)     # Info
    
    # Font settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    THICKNESS = 2
    
    @staticmethod
    def draw_eye_bbox(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                      is_drowsy: bool = False, label: str = None) -> np.ndarray:
        """
        Draw bounding box around eye region.
        
        Args:
            frame: BGR image
            bbox: (x_min, y_min, x_max, y_max)
            is_drowsy: If True, draw red box; else green
            label: Optional label to display
        
        Returns:
            Annotated frame
        """
        color = Visualizer.RED if is_drowsy else Visualizer.GREEN
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label if provided
        if label:
            label_size = cv2.getTextSize(label, Visualizer.FONT, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_size[0] + 5, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), Visualizer.FONT, 0.5, 
                       Visualizer.WHITE, 1)
        
        return frame
    
    @staticmethod
    def draw_eye_landmarks(frame: np.ndarray, eye_points: List,
                           color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Draw eye landmark points.
        
        Args:
            frame: BGR image
            eye_points: List of (x, y) coordinates
            color: Optional color override
        
        Returns:
            Annotated frame
        """
        if color is None:
            color = Visualizer.BLUE
        
        for point in eye_points:
            cv2.circle(frame, point, 2, color, -1)
        
        return frame
    
    @staticmethod
    def draw_status(frame: np.ndarray, is_drowsy: bool, 
                    confidence: float = None) -> np.ndarray:
        """
        Draw drowsiness status on frame.
        
        Args:
            frame: BGR image
            is_drowsy: Drowsiness status
            confidence: Optional confidence score
        
        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        
        if is_drowsy:
            status = "DROWSY!"
            color = Visualizer.RED
            bg_color = (50, 50, 200)
        else:
            status = "AWAKE"
            color = Visualizer.GREEN
            bg_color = (50, 200, 50)
        
        # Draw status background
        cv2.rectangle(frame, (10, 10), (200, 60), bg_color, -1)
        cv2.rectangle(frame, (10, 10), (200, 60), color, 2)
        
        # Draw status text
        cv2.putText(frame, status, (20, 45), Visualizer.FONT, 1.0, 
                   Visualizer.WHITE, 2)
        
        # Draw confidence if provided
        if confidence is not None:
            conf_text = f"Conf: {confidence:.1%}"
            cv2.putText(frame, conf_text, (w - 150, 30), Visualizer.FONT, 0.6,
                       Visualizer.WHITE, 2)
        
        return frame
    
    @staticmethod
    def draw_ear_info(frame: np.ndarray, left_ear: float, right_ear: float,
                      avg_ear: float, threshold: float) -> np.ndarray:
        """
        Draw EAR information on frame.
        
        Args:
            frame: BGR image
            left_ear: Left eye EAR
            right_ear: Right eye EAR
            avg_ear: Average EAR
            threshold: Drowsiness threshold
        
        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        
        # Draw info panel
        panel_x = w - 180
        cv2.rectangle(frame, (panel_x, 50), (w - 10, 150), (40, 40, 40), -1)
        cv2.rectangle(frame, (panel_x, 50), (w - 10, 150), Visualizer.WHITE, 1)
        
        # Draw EAR values
        y_offset = 70
        cv2.putText(frame, f"L-EAR: {left_ear:.3f}", (panel_x + 10, y_offset),
                   Visualizer.FONT, 0.5, Visualizer.WHITE, 1)
        
        y_offset += 25
        cv2.putText(frame, f"R-EAR: {right_ear:.3f}", (panel_x + 10, y_offset),
                   Visualizer.FONT, 0.5, Visualizer.WHITE, 1)
        
        y_offset += 25
        ear_color = Visualizer.RED if avg_ear < threshold else Visualizer.GREEN
        cv2.putText(frame, f"AVG: {avg_ear:.3f}", (panel_x + 10, y_offset),
                   Visualizer.FONT, 0.5, ear_color, 1)
        
        y_offset += 25
        cv2.putText(frame, f"Thresh: {threshold:.3f}", (panel_x + 10, y_offset),
                   Visualizer.FONT, 0.5, Visualizer.YELLOW, 1)
        
        return frame
    
    @staticmethod
    def draw_approach_label(frame: np.ndarray, approach: str) -> np.ndarray:
        """
        Draw approach label on frame.
        
        Args:
            frame: BGR image
            approach: Approach name
        
        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        
        cv2.rectangle(frame, (10, h - 40), (250, h - 10), (0, 0, 0), -1)
        cv2.putText(frame, f"Model: {approach}", (15, h - 18), 
                   Visualizer.FONT, 0.5, Visualizer.WHITE, 1)
        
        return frame
    
    @staticmethod
    def annotate_frame(frame: np.ndarray, eye_data: dict, ear_data: dict,
                       prediction: dict, approach: str) -> np.ndarray:
        """
        Full annotation pipeline for a frame.
        
        Args:
            frame: BGR image
            eye_data: Eye detection data
            ear_data: EAR calculation data
            prediction: Model prediction
            approach: Approach name
        
        Returns:
            Fully annotated frame
        """
        annotated = frame.copy()
        
        is_drowsy = prediction.get("is_drowsy", False)
        confidence = prediction.get("confidence", 0.0)
        
        # Draw eye bounding boxes (simple, minimal)
        if eye_data.get("face_detected"):
            left_bbox = eye_data.get("left_bbox")
            right_bbox = eye_data.get("right_bbox")
            
            # Just draw simple colored boxes - no labels
            color = Visualizer.RED if is_drowsy else Visualizer.GREEN
            if left_bbox:
                x1, y1, x2, y2 = left_bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            if right_bbox:
                x1, y1, x2, y2 = right_bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw clean status overlay at top
        h, w = annotated.shape[:2]
        
        if is_drowsy:
            status = "DROWSY!"
            bg_color = (50, 50, 200)
            txt_color = Visualizer.WHITE
        else:
            status = "AWAKE"
            bg_color = (50, 200, 50)
            txt_color = Visualizer.WHITE
        
        # Status bar at top
        cv2.rectangle(annotated, (0, 0), (w, 40), bg_color, -1)
        cv2.putText(annotated, f"{status} - {confidence:.0%}", (10, 28), 
                   Visualizer.FONT, 0.8, txt_color, 2)
        
        return annotated
