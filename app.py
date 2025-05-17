from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from VisualSearch import ImageProcessor, CameraConfig
import cv2
import numpy as np
from PIL import Image
import io
from functools import lru_cache
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize ImageProcessor with optimized settings
processor = ImageProcessor(camera_config=CameraConfig(
    width=1280,  # Reduced from 1920
    height=720,  # Reduced from 1080
    fps=30
))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@lru_cache(maxsize=32)
def process_image(image_path):
    """Cached image processing function"""
    analysis = processor.analyze_image(image_path)
    ocr_result = processor.perform_ocr(image_path)
    return analysis, ocr_result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            start_time = time.time()
            
            # Read image file
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Resize image if too large
            max_dimension = 1280
            height, width = img.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            
            # Save temporary file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(filepath, img)
            
            # Process image with caching
            analysis, ocr_result = process_image(filepath)
            
            # Clean up
            os.remove(filepath)
            
            # Prepare response
            response = {
                'text': ocr_result.get('text', ''),
                'quality': analysis.estimated_quality if analysis else 'unknown',
                'objects': analysis.object_detection_results if analysis else [],
                'processing_time': round(time.time() - start_time, 2)
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, threaded=True) 