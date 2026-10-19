"""
Flask backend for Gender & Age Prediction application.
Provides API endpoints for image upload and prediction.
"""

from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from models import GenderAgePrediction
import base64

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize model (loaded once at startup)
print("Loading models...")
try:
    predictor = GenderAgePrediction()
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please run 'python download_models.py' first.")
    predictor = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and return predictions.
    
    Expected: multipart/form-data with 'image' file
    Returns: JSON with predictions for all detected faces
    """
    # Check if models are loaded
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Models not loaded. Please run download_models.py first.'
        }), 500
    
    # Check if file is in request
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided'
        }), 400
    
    file = request.files['image']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run prediction
        result = predictor.predict(filepath)
        
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        # Clean up on error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor is not None
    })

if __name__ == '__main__':
    if predictor is None:
        print("\n" + "="*60)
        print("WARNING: Models not loaded!")
        print("Please run: python download_models.py")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("Gender & Age Prediction Server Starting...")
        print("Navigate to: http://localhost:5000")
        print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
