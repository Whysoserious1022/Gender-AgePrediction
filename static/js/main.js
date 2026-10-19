// ===== Global Variables =====
let selectedFile = null;
let currentImage = null;

// ===== DOM Elements =====
const uploadArea = document.getElementById('uploadArea');
const uploadButton = document.getElementById('uploadButton');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const resultCanvas = document.getElementById('resultCanvas');
const analyzeButton = document.getElementById('analyzeButton');
const closePreview = document.getElementById('closePreview');
const resultsSection = document.getElementById('resultsSection');
const resultsGrid = document.getElementById('resultsGrid');
const faceCountBadge = document.getElementById('faceCountBadge');
const loadingOverlay = document.getElementById('loadingOverlay');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const closeError = document.getElementById('closeError');

// ===== Event Listeners =====

// Upload button click
uploadButton.addEventListener('click', () => {
    fileInput.click();
});

// Upload area click
uploadArea.addEventListener('click', (e) => {
    if (e.target !== uploadButton) {
        fileInput.click();
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and drop events
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// Analyze button click
analyzeButton.addEventListener('click', () => {
    analyzeImage();
});

// Close preview button
closePreview.addEventListener('click', () => {
    resetApp();
});

// Close error button
closeError.addEventListener('click', () => {
    hideError();
});

// ===== Functions =====

/**
 * Handle file selection
 */
function handleFileSelect(file) {
    if (!file) return;
    
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload an image file (PNG, JPG, GIF, WebP).');
        return;
    }
    
    // Validate file size (10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File size exceeds 10MB. Please upload a smaller image.');
        return;
    }
    
    selectedFile = file;
    displayPreview(file);
}

/**
 * Display image preview
 */
function displayPreview(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        currentImage = new Image();
        currentImage.onload = () => {
            previewImage.src = e.target.result;
            previewSection.style.display = 'block';
            resultsSection.style.display = 'none';
            
            // Setup canvas
            resultCanvas.width = currentImage.width;
            resultCanvas.height = currentImage.height;
            
            // Scroll to preview
            previewSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
        };
        currentImage.src = e.target.result;
    };
    
    reader.readAsDataURL(file);
}

/**
 * Analyze image using backend API
 */
async function analyzeImage() {
    if (!selectedFile) {
        showError('No image selected');
        return;
    }
    
    // Show loading
    showLoading();
    
    // Prepare form data
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        hideLoading();
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.message || data.error || 'Prediction failed');
        }
    } catch (error) {
        hideLoading();
        showError('Network error. Please check your connection and try again.');
        console.error('Error:', error);
    }
}

/**
 * Display prediction results
 */
function displayResults(data) {
    // Draw bounding boxes on canvas
    drawBoundingBoxes(data.faces);
    
    // Display results cards
    resultsGrid.innerHTML = '';
    
    data.faces.forEach((face, index) => {
        const card = createResultCard(face, index);
        resultsGrid.appendChild(card);
    });
    
    // Update face count badge
    faceCountBadge.textContent = `${data.total_faces} Face${data.total_faces !== 1 ? 's' : ''} Detected`;
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

/**
 * Draw bounding boxes on canvas
 */
function drawBoundingBoxes(faces) {
    const ctx = resultCanvas.getContext('2d');
    ctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
    
    // Draw image
    ctx.drawImage(currentImage, 0, 0, resultCanvas.width, resultCanvas.height);
    
    // Draw boxes for each face
    faces.forEach((face, index) => {
        const bbox = face.bounding_box;
        
        // Set style
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 3;
        ctx.fillStyle = 'rgba(102, 126, 234, 0.2)';
        
        // Draw rectangle
        ctx.fillRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
        ctx.strokeRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
        
        // Draw label background
        const label = `Face ${index + 1}`;
        ctx.font = 'bold 16px Inter';
        const textWidth = ctx.measureText(label).width;
        
        ctx.fillStyle = '#667eea';
        ctx.fillRect(bbox.x1, bbox.y1 - 30, textWidth + 20, 30);
        
        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, bbox.x1 + 10, bbox.y1 - 10);
    });
}

/**
 * Create result card for a face
 */
function createResultCard(face, index) {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.style.animationDelay = `${index * 0.1}s`;
    
    card.innerHTML = `
        <div class="result-header">
            <span>👤 Face ${face.face_id}</span>
        </div>
        
        <div class="prediction-item">
            <div class="prediction-label">Gender</div>
            <div class="prediction-value">${face.gender.prediction}</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${face.gender.confidence}%"></div>
            </div>
            <div class="confidence-text">Confidence: ${face.gender.confidence.toFixed(1)}%</div>
        </div>
        
        <div class="prediction-item">
            <div class="prediction-label">Age Range</div>
            <div class="prediction-value">${face.age.prediction}</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${face.age.confidence}%"></div>
            </div>
            <div class="confidence-text">Confidence: ${face.age.confidence.toFixed(1)}%</div>
        </div>
    `;
    
    return card;
}

/**
 * Show loading overlay
 */
function showLoading() {
    loadingOverlay.style.display = 'flex';
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    loadingOverlay.style.display = 'none';
}

/**
 * Show error message
 */
function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'block';
}

/**
 * Hide error message
 */
function hideError() {
    errorMessage.style.display = 'none';
}

/**
 * Reset application to initial state
 */
function resetApp() {
    selectedFile = null;
    currentImage = null;
    fileInput.value = '';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    
    // Clear canvas
    const ctx = resultCanvas.getContext('2d');
    ctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
}

// ===== Initialize =====
console.log('Gender & Age Prediction App Initialized');
