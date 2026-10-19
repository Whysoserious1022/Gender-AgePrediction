"""
Enhanced model loading and prediction logic for gender and age detection.
Uses OpenCV DNN module with optimized preprocessing and configuration.
"""

import cv2
import numpy as np
import os

class GenderAgePrediction:
    """Enhanced class for face detection, gender prediction, and age prediction with improved accuracy."""
    
    def __init__(self, models_dir="models"):
        """Initialize models with enhanced configuration."""
        self.models_dir = models_dir
        
        # Define age and gender categories
        self.AGE_RANGES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                          '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.GENDER_LIST = ['Male', 'Female']
        
        # Enhanced preprocessing - updated mean values for better normalization
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        
        # Load models with optimized backends
        self._load_face_detector()
        self._load_age_model()
        self._load_gender_model()
        
    def _load_face_detector(self):
        """Load OpenCV DNN face detector with optimized backend."""
        face_proto = os.path.join(self.models_dir, "opencv_face_detector.pbtxt")
        face_model = os.path.join(self.models_dir, "opencv_face_detector_uint8.pb")
        
        if not os.path.exists(face_proto) or not os.path.exists(face_model):
            raise FileNotFoundError(
                "Face detector models not found. Please run download_models.py first."
            )
        
        self.face_net = cv2.dnn.readNet(face_model, face_proto)
        # Use CPU optimizations for better stability
        self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("✓ Face detector loaded successfully")
        
    def _load_age_model(self):
        """Load age prediction model with optimized backend."""
        age_proto = os.path.join(self.models_dir, "age_deploy.prototxt")
        age_model = os.path.join(self.models_dir, "age_net.caffemodel")
        
        if not os.path.exists(age_proto) or not os.path.exists(age_model):
            raise FileNotFoundError(
                "Age prediction models not found. Please run download_models.py first."
            )
        
        self.age_net = cv2.dnn.readNet(age_model, age_proto)
        self.age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("✓ Age prediction model loaded successfully")
        
    def _load_gender_model(self):
        """Load gender prediction model with optimized backend."""
        gender_proto = os.path.join(self.models_dir, "gender_deploy.prototxt")
        gender_model = os.path.join(self.models_dir, "gender_net.caffemodel")
        
        if not os.path.exists(gender_proto) or not os.path.exists(gender_model):
            raise FileNotFoundError(
                "Gender prediction models not found. Please run download_models.py first."
            )
        
        self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)
        self.gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("✓ Gender prediction model loaded successfully")
    
    def _preprocess_image(self, image):
        """Enhanced image preprocessing for better quality."""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
        # Apply histogram equalization to improve contrast
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        # Reduce noise while preserving edges
        image = cv2.bilateralFilter(image, 5, 50, 50)
        
        return image
    
    def detect_faces(self, image, confidence_threshold=0.5):
        """
        Enhanced face detection with lower threshold for better recall.
        
        Args:
            image: Input image (numpy array)
            confidence_threshold: Minimum confidence (lowered from 0.7 to 0.5)
            
        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...] with confidence scores
        """
        frame_height, frame_width = image.shape[:2]
        
        # Prepare image for face detection with enhanced preprocessing
        blob = cv2.dnn.blobFromImage(
            image, 
            scalefactor=1.0, 
            size=(300, 300), 
            mean=[104, 117, 123], 
            swapRB=False, 
            crop=False
        )
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        face_boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_width, x2)
                y2 = min(frame_height, y2)
                
                # Add padding around face for better context (10%)
                face_width = x2 - x1
                face_height = y2 - y1
                padding_x = int(face_width * 0.1)
                padding_y = int(face_height * 0.1)
                
                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(frame_width, x2 + padding_x)
                y2 = min(frame_height, y2 + padding_y)
                
                face_boxes.append((x1, y1, x2, y2, float(confidence)))
        
        return face_boxes
    
    def _enhance_face_region(self, face_image):
        """Enhanced face preprocessing for better prediction accuracy."""
        # Resize to ensure consistent input
        if face_image.shape[0] < 50 or face_image.shape[1] < 50:
            # Upscale small faces
            scale_factor = max(50.0 / face_image.shape[0], 50.0 / face_image.shape[1])
            face_image = cv2.resize(face_image, None, fx=scale_factor, fy=scale_factor, 
                                  interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        face_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Slight Gaussian blur to reduce noise
        face_image = cv2.GaussianBlur(face_image, (3, 3), 0)
        
        return face_image
    
    def predict_gender(self, face_image):
        """
        Enhanced gender prediction with optimized preprocessing.
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Tuple of (gender_label, confidence)
        """
        # Enhanced preprocessing
        face_image = self._enhance_face_region(face_image)
        
        blob = cv2.dnn.blobFromImage(
            face_image, 
            scalefactor=1.0, 
            size=(227, 227), 
            mean=self.MODEL_MEAN_VALUES, 
            swapRB=False
        )
        
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        
        gender_idx = gender_preds[0].argmax()
        confidence = float(gender_preds[0][gender_idx] * 100)
        
        return self.GENDER_LIST[gender_idx], round(confidence, 2)
    
    def predict_age(self, face_image):
        """
        Enhanced age prediction with optimized preprocessing.
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Tuple of (age_range, confidence)
        """
        # Enhanced preprocessing
        face_image = self._enhance_face_region(face_image)
        
        blob = cv2.dnn.blobFromImage(
            face_image, 
            scalefactor=1.0, 
            size=(227, 227), 
            mean=self.MODEL_MEAN_VALUES, 
            swapRB=False
        )
        
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        
        age_idx = age_preds[0].argmax()
        confidence = float(age_preds[0][age_idx] * 100)
        
        return self.AGE_RANGES[age_idx], round(confidence, 2)
    
    def predict(self, image_path):
        """
        Enhanced prediction function with better preprocessing.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with predictions for each detected face
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Enhanced preprocessing
        image = self._preprocess_image(image)
        
        # Detect faces with lowered threshold for better recall
        face_boxes = self.detect_faces(image, confidence_threshold=0.5)
        
        if not face_boxes:
            return {
                "success": False,
                "message": "No faces detected in the image. Try a clearer image with visible faces.",
                "faces": []
            }
        
        # Predict for each face
        predictions = []
        for idx, (x1, y1, x2, y2, detection_conf) in enumerate(face_boxes):
            # Crop face
            face = image[y1:y2, x1:x2].copy()
            
            # Skip if face is too small even after preprocessing
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue
            
            # Predict gender and age
            gender, gender_conf = self.predict_gender(face)
            age_range, age_conf = self.predict_age(face)
            
            predictions.append({
                "face_id": idx + 1,
                "detection_confidence": round(float(detection_conf * 100), 2),
                "bounding_box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },
                "gender": {
                    "prediction": gender,
                    "confidence": gender_conf
                },
                "age": {
                    "prediction": age_range,
                    "confidence": age_conf
                }
            })
        
        return {
            "success": True,
            "total_faces": len(predictions),
            "faces": predictions
        }
