# 🎭 Gender & Age Prediction

An AI-powered web application that predicts gender and age from images using deep learning models. Built with Python, Flask, OpenCV, and modern web technologies.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🔍 **Automatic Face Detection** - Detects multiple faces in a single image
- 👤 **Gender Prediction** - Predicts Male/Female with confidence scores
- 📊 **Age Prediction** - Estimates age range in 8 categories
- 🎨 **Premium UI** - Modern, glassmorphic design with smooth animations
- 📱 **Responsive Design** - Works seamlessly on desktop, tablet, and mobile
- ⚡ **Fast Processing** - Real-time predictions using OpenCV DNN
- 🎯 **High Accuracy** - Uses pre-trained Caffe models from research papers

## 🧠 Technical Details

### Models Used

1. **Face Detection**: OpenCV DNN Face Detector
2. **Gender Classification**: GilLevi's Gender Net (Caffe)
3. **Age Classification**: GilLevi's Age Net (Caffe)

### Age Categories
- (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)

### Gender Categories
- Male, Female

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Clone or Download

```bash
cd Gender&AgePrediction
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Models

Run the model download script to fetch all required pre-trained models:

```bash
python download_models.py
```

This will download ~200MB of model files to the `models/` directory.

## 🎮 Usage

### Starting the Application

```bash
python app.py
```

The server will start on `http://localhost:5000`

### Using the Web Interface

1. **Open your browser** and navigate to `http://localhost:5000`
2. **Upload an image** by:
   - Dragging and dropping an image onto the upload area
   - Clicking "Choose Image" to browse files
3. **Click "Analyze Image"** to run the prediction
4. **View results** showing:
   - Face detection bounding boxes
   - Gender prediction with confidence
   - Age range prediction with confidence
   - Results for all detected faces

### Supported Image Formats

- PNG, JPG, JPEG, GIF, BMP, WebP
- Maximum file size: 10MB

## 📁 Project Structure

```
Gender&AgePrediction/
├── app.py                  # Flask backend application
├── models.py               # Model loading and prediction logic
├── download_models.py      # Script to download pre-trained models
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── models/                # Downloaded model files (created after running download_models.py)
│   ├── age_net.caffemodel
│   ├── age_deploy.prototxt
│   ├── gender_net.caffemodel
│   ├── gender_deploy.prototxt
│   ├── opencv_face_detector.pbtxt
│   └── opencv_face_detector_uint8.pb
│
├── templates/             # HTML templates
│   └── index.html
│
├── static/                # Static assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
└── uploads/               # Temporary upload directory (auto-created)
```

## 🎨 UI Features

- **Glassmorphism Design** - Modern frosted glass effects
- **Gradient Backgrounds** - Vibrant, animated color gradients
- **Smooth Animations** - Micro-interactions and transitions
- **Dark Theme** - Easy on the eyes
- **Interactive Elements** - Hover effects and loading states
- **Canvas Visualization** - Face bounding boxes drawn in real-time

## 🔧 API Endpoints

### POST `/predict`

Upload an image for prediction.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `image` file

**Response:**
```json
{
  "success": true,
  "total_faces": 2,
  "faces": [
    {
      "face_id": 1,
      "bounding_box": {
        "x1": 100,
        "y1": 150,
        "x2": 300,
        "y2": 350
      },
      "gender": {
        "prediction": "Male",
        "confidence": 92.5
      },
      "age": {
        "prediction": "(25-32)",
        "confidence": 78.3
      }
    }
  ]
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

## 🛠️ Technologies Used

### Backend
- **Flask** - Web framework
- **OpenCV** - Computer vision and DNN module
- **NumPy** - Numerical computations
- **Pillow** - Image processing

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling (glassmorphism, animations)
- **JavaScript** - Interactive functionality
- **Canvas API** - Bounding box visualization

### Models
- **Caffe** - Deep learning framework
- **OpenCV DNN** - Model inference

## 📊 Performance

- **Face Detection**: ~100-200ms per image
- **Gender Prediction**: ~50ms per face
- **Age Prediction**: ~50ms per face
- **Total Processing**: <1 second for typical images

## 🤝 Credits

- **Age & Gender Models**: [Gil Levi and Tal Hassner](https://github.com/GilLevi/AgeGenderDeepLearning)
- **Face Detection**: OpenCV DNN Face Detector

## 📝 License

MIT License - Feel free to use this project for learning and development!

## 🐛 Troubleshooting

### Models not loading
- Ensure you've run `python download_models.py`
- Check that all 6 model files exist in the `models/` directory

### No faces detected
- Try uploading a clearer, higher-resolution image
- Ensure faces are clearly visible and not too small
- Good lighting improves detection accuracy

### Server errors
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Ensure port 5000 is not in use by another application
- Check the console for detailed error messages

## 🚀 Future Enhancements

- [ ] Add emotion detection
- [ ] Support for video input
- [ ] Batch processing for multiple images
- [ ] Export results as JSON/CSV
- [ ] API authentication
- [ ] Docker containerization

---

Built with ❤️ using Flask, OpenCV, and Deep Learning
