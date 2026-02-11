"""
Flask REST API for Drowsiness Detection
Endpoints for image/video prediction and metrics.
"""

import os
import sys
import io
import base64
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import config
from api.inference import get_engine
from api.metrics_loader import load_all_metrics, get_metrics_summary, get_model_names

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"])

# Alert sound
ALERT_SOUND_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "alert.mp3")


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    Returns API status and loaded models.
    """
    engine = get_engine()
    
    # Check GPU
    gpu_available = False
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        gpu_available = len(gpus) > 0
    except:
        pass
    
    return jsonify({
        "status": "healthy",
        "gpu_available": gpu_available,
        "models_loaded": engine.get_loaded_models(),
        "model_names": get_model_names()
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """
    Get all model performance metrics.
    Returns metrics for all three approaches.
    """
    try:
        summary = get_metrics_summary()
        full_metrics = load_all_metrics()
        
        return jsonify({
            "success": True,
            "summary": summary,
            "detailed": full_metrics
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict drowsiness from uploaded image.
    
    Input: multipart/form-data with 'image' file and 'approach'
    Output: Prediction results with annotated image
    """
    try:
        # Get image file
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400
        
        file = request.files['image']
        approach = request.form.get('approach', 'approach1')
        
        # Read image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                "success": False,
                "error": "Failed to decode image"
            }), 400
        
        # Get prediction
        engine = get_engine()
        result = engine.predict_image(image, approach)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/predict/base64', methods=['POST'])
def predict_base64():
    """
    Predict drowsiness from base64-encoded image.
    Used for webcam frames.
    
    Input: JSON with 'image' (base64) and 'approach'
    Output: Prediction results
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400
        
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                "success": False,
                "error": "Failed to decode image"
            }), 400
        
        approach = data.get('approach', 'approach1')
        
        # Get prediction
        engine = get_engine()
        result = engine.predict_image(image, approach)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/video/process', methods=['POST'])
def process_video():
    """
    Process uploaded video file.
    Returns annotated video (WebM) or key frames.
    """
    try:
        if 'video' not in request.files:
            return jsonify({
                "success": False,
                "error": "No video provided"
            }), 400
        
        file = request.files['video']
        approach = request.form.get('approach', 'approach1')
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            file.save(tmp.name)
            video_path = tmp.name
        
        try:
            # Process video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return jsonify({
                    "success": False,
                    "error": "Failed to open video"
                }), 400
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            engine = get_engine()
            engine.reset_state()
            
            results = []
            frame_count = 0
            drowsy_timestamps = []
            annotated_frames = []
            
            # Sample every N frames for performance
            sample_interval = max(1, int(fps / 10))  # ~10 predictions per second
            last_result = {"is_drowsy": False, "confidence": 0.5}
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Predict on sampled frames
                if frame_count % sample_interval == 0:
                    result = engine.predict_video_frame(frame, approach)
                    last_result = result
                    results.append({
                        "is_drowsy": result.get("is_drowsy", False),
                        "confidence": result.get("confidence", 0),
                        "timestamp_ms": int((frame_count / fps) * 1000)
                    })
                    
                    if result.get("is_drowsy", False):
                        drowsy_timestamps.append(frame_count / fps)
                
                # Draw label on frame
                h, w = frame.shape[:2]
                is_drowsy = last_result.get("is_drowsy", False)
                confidence = last_result.get("confidence", 0)
                
                if is_drowsy:
                    bg_color = (0, 0, 200)  # Red in BGR
                    text = f"DROWSY! - {confidence:.0%}"
                else:
                    bg_color = (0, 180, 0)  # Green in BGR
                    text = f"AWAKE - {confidence:.0%}"
                
                cv2.rectangle(frame, (0, 0), (w, 40), bg_color, -1)
                cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (255, 255, 255), 2)
                
                # Store every Nth frame for video output (subsample for file size)
                if frame_count % 3 == 0:  # Keep 1/3 of frames
                    # Convert BGR to RGB for imageio
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    annotated_frames.append(rgb_frame)
                
                frame_count += 1
            
            cap.release()
            
            # Create output video using imageio
            video_data = None
            try:
                import imageio
                output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
                
                # Use imageio to create MP4 with proper codec
                writer = imageio.get_writer(output_path, fps=fps/3, codec='libx264', 
                                           pixelformat='yuv420p', macro_block_size=1)
                for frame in annotated_frames:
                    writer.append_data(frame)
                writer.close()
                
                # Read and encode
                with open(output_path, 'rb') as f:
                    video_bytes = f.read()
                video_data = f"data:video/mp4;base64,{base64.b64encode(video_bytes).decode('utf-8')}"
                os.unlink(output_path)
                
            except Exception as e:
                print(f"imageio failed: {e}")
                # Fallback: return key frames as images
                key_frames = []
                for i in range(0, len(annotated_frames), max(1, len(annotated_frames) // 10)):
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_frames[i], cv2.COLOR_RGB2BGR))
                    key_frames.append(base64.b64encode(buffer).decode('utf-8'))
                
                return jsonify({
                    "success": True,
                    "video_data": None,
                    "key_frames": key_frames,
                    "total_frames": frame_count,
                    "processed_frames": len(results),
                    "drowsy_frames": sum(1 for r in results if r.get('is_drowsy', False)),
                    "drowsy_percentage": (sum(1 for r in results if r.get('is_drowsy', False)) / len(results) * 100) if results else 0,
                    "drowsy_timestamps": drowsy_timestamps[:50],
                    "duration_seconds": frame_count / fps if fps > 0 else 0,
                    "fps": fps,
                    "approach": approach
                })
            
            # Calculate summary
            drowsy_frames = sum(1 for r in results if r.get('is_drowsy', False))
            total_processed = len(results)
            
            return jsonify({
                "success": True,
                "video_data": video_data,
                "total_frames": frame_count,
                "processed_frames": total_processed,
                "drowsy_frames": drowsy_frames,
                "drowsy_percentage": (drowsy_frames / total_processed * 100) if total_processed > 0 else 0,
                "drowsy_timestamps": drowsy_timestamps[:50],
                "duration_seconds": frame_count / fps if fps > 0 else 0,
                "fps": fps,
                "approach": approach
            })
        
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/reset', methods=['POST'])
def reset_state():
    """
    Reset inference state (EAR frame counter).
    Call when starting new webcam session or video.
    """
    try:
        engine = get_engine()
        engine.reset_state()
        return jsonify({"success": True, "message": "State reset"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/alert/sound', methods=['GET'])
def get_alert_sound():
    """
    Get alert sound file.
    """
    if os.path.exists(ALERT_SOUND_PATH):
        with open(ALERT_SOUND_PATH, 'rb') as f:
            audio_data = f.read()
        return Response(
            audio_data,
            mimetype='audio/mpeg',
            headers={'Content-Disposition': 'inline; filename=alert.mp3'}
        )
    else:
        return jsonify({"error": "Alert sound not found"}), 404


@app.route('/api/models', methods=['GET'])
def get_models():
    """
    Get available models and their status.
    """
    engine = get_engine()
    
    models = {
        "approach1": {
            "name": "Custom CNN (DrowsyNet)",
            "loaded": "approach1" in engine.get_loaded_models(),
            "input_size": list(config.CNN_IMG_SIZE),
            "description": "Custom deep CNN with 5 convolutional blocks"
        },
        "approach2": {
            "name": "EfficientNet Transfer",
            "loaded": "approach2" in engine.get_loaded_models(),
            "input_size": list(config.EFFICIENTNET_IMG_SIZE),
            "description": "EfficientNetB1 with transfer learning"
        },
        "approach3": {
            "name": "EAR Detection",
            "loaded": "approach3" in engine.get_loaded_models(),
            "input_size": None,
            "description": "Eye Aspect Ratio with MediaPipe"
        }
    }
    
    return jsonify({
        "success": True,
        "models": models
    })


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("   Drowsiness Detection API Server")
    print("=" * 50)
    
    # Create assets directory for alert sound
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    print(f"\n📂 Place alert sound at: {ALERT_SOUND_PATH}")
    print(f"\n🚀 Starting server on http://{config.API_HOST}:{config.API_PORT}")
    print(f"   Debug mode: {config.DEBUG_MODE}")
    print("\n" + "=" * 50 + "\n")
    
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG_MODE
    )
