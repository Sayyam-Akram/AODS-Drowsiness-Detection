"""
Eye Aspect Ratio (EAR) Calculator
Uses MediaPipe landmarks to compute drowsiness indicator.
"""

import numpy as np
from typing import List, Tuple
from scipy.spatial import distance as dist


class EARCalculator:
    """
    Calculate Eye Aspect Ratio for drowsiness detection.
    """
    
    def __init__(self, threshold: float = 0.22, 
                 consecutive_frames: int = 5):
        """
        Initialize EAR calculator.
        
        Args:
            threshold: EAR threshold below which eyes are considered closed
            consecutive_frames: Number of frames to confirm drowsiness
        """
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self.frame_counter = 0
        self.is_drowsy = False
    
    @staticmethod
    def calculate_ear(eye_points: List[Tuple[int, int]]) -> float:
        """
        Calculate Eye Aspect Ratio.
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            eye_points: 6 landmark points for one eye [p1, p2, p3, p4, p5, p6]
                       p1, p4: horizontal corners
                       p2, p3: upper lid points  
                       p5, p6: lower lid points
        
        Returns:
            EAR value (0.0 to ~0.5 typically)
        """
        if len(eye_points) != 6:
            return 0.0
        
        # Convert to numpy array
        points = np.array(eye_points, dtype=np.float64)
        
        # Compute vertical distances
        vertical_1 = dist.euclidean(points[1], points[5])  # p2-p6
        vertical_2 = dist.euclidean(points[2], points[4])  # p3-p5
        
        # Compute horizontal distance
        horizontal = dist.euclidean(points[0], points[3])  # p1-p4
        
        # Avoid division by zero
        if horizontal < 1e-6:
            return 0.0
        
        # Calculate EAR
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        
        return ear
    
    def calculate_average_ear(self, left_eye: List, right_eye: List) -> Tuple[float, float, float]:
        """
        Calculate EAR for both eyes and return average.
        
        Args:
            left_eye: 6 points for left eye
            right_eye: 6 points for right eye
        
        Returns:
            (left_ear, right_ear, average_ear)
        """
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        return left_ear, right_ear, avg_ear
    
    def update(self, avg_ear: float) -> bool:
        """
        Update drowsiness state based on current EAR.
        
        Args:
            avg_ear: Average EAR of both eyes
        
        Returns:
            True if drowsy, False otherwise
        """
        if avg_ear < self.threshold:
            self.frame_counter += 1
            if self.frame_counter >= self.consecutive_frames:
                self.is_drowsy = True
        else:
            self.frame_counter = 0
            self.is_drowsy = False
        
        return self.is_drowsy
    
    def process_frame(self, left_eye: List, right_eye: List) -> dict:
        """
        Process a single frame.
        
        Args:
            left_eye: 6 points for left eye
            right_eye: 6 points for right eye
        
        Returns:
            Dictionary with EAR values and drowsiness status
        """
        left_ear, right_ear, avg_ear = self.calculate_average_ear(left_eye, right_eye)
        is_drowsy = self.update(avg_ear)
        
        return {
            "left_ear": round(left_ear, 3),
            "right_ear": round(right_ear, 3),
            "avg_ear": round(avg_ear, 3),
            "is_drowsy": is_drowsy,
            "frame_counter": self.frame_counter,
            "threshold": self.threshold
        }
    
    def reset(self):
        """Reset the frame counter and drowsiness state."""
        self.frame_counter = 0
        self.is_drowsy = False
    
    def set_threshold(self, threshold: float):
        """Update the EAR threshold."""
        self.threshold = threshold
    
    def set_consecutive_frames(self, frames: int):
        """Update consecutive frames threshold."""
        self.consecutive_frames = frames
