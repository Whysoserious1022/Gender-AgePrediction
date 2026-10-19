"""
Download pre-trained models for gender and age prediction.
This script downloads the necessary Caffe models and face detection models.
"""

import os
import urllib.request
import sys

# Create models directory if it doesn't exist
MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    print(f"Created directory: {MODELS_DIR}")

# Model URLs - Using reliable Hugging Face and GitHub sources
MODEL_FILES = {
    "age_deploy.prototxt": "https://huggingface.co/AjaySharma/genderDetection/resolve/main/age_deploy.prototxt",
    "age_net.caffemodel": "https://huggingface.co/AjaySharma/genderDetection/resolve/main/age_net.caffemodel",
    "gender_deploy.prototxt": "https://huggingface.co/AjaySharma/genderDetection/resolve/main/gender_deploy.prototxt",
    "gender_net.caffemodel": "https://huggingface.co/AjaySharma/genderDetection/resolve/main/gender_net.caffemodel",
    "opencv_face_detector.pbtxt": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/opencv_face_detector.pbtxt",
    "opencv_face_detector_uint8.pb": "https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/models/opencv_face_detector_uint8.pb"
}

def download_file(url, filename):
    """Download a file with progress tracking."""
    filepath = os.path.join(MODELS_DIR, filename)
    
    # Check if file already exists
    if os.path.exists(filepath):
        print(f"✓ {filename} already exists, skipping download.")
        return
    
    print(f"Downloading {filename}...")
    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r  Progress: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print(f"\n✓ Successfully downloaded {filename}")
    except Exception as e:
        print(f"\n✗ Error downloading {filename}: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        raise

def main():
    print("=" * 60)
    print("Downloading Gender & Age Prediction Models")
    print("=" * 60)
    print()
    
    for filename, url in MODEL_FILES.items():
        download_file(url, filename)
        print()
    
    print("=" * 60)
    print("All models downloaded successfully!")
    print("=" * 60)
    print()
    print(f"Models saved in: {os.path.abspath(MODELS_DIR)}")

if __name__ == "__main__":
    main()
