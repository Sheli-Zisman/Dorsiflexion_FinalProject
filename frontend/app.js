// API Configuration
const API_BASE_URL = `/api`;
console.log(API_BASE_URL);

// State management
let rangeImageFile1 = null;
let rangeImageFile2 = null;

// ============================================
// RANGE OF MOTION ANALYSIS
// ============================================

const rangeUploadBox1 = document.getElementById('range-upload-1');
const rangeUploadBox2 = document.getElementById('range-upload-2');
const rangeFileInput1 = document.getElementById('range-file-input-1');
const rangeFileInput2 = document.getElementById('range-file-input-2');
const rangePreview1 = document.getElementById('range-preview-1');
const rangePreview2 = document.getElementById('range-preview-2');
const rangePreviewImg1 = document.getElementById('range-preview-img-1');
const rangePreviewImg2 = document.getElementById('range-preview-img-2');
const rangeRemoveBtn1 = document.getElementById('range-remove-1');
const rangeRemoveBtn2 = document.getElementById('range-remove-2');
const rangeAnalyzeBtn = document.getElementById('range-analyze-btn');
const rangeResults = document.getElementById('range-results');
const rangeLoading = document.getElementById('range-loading');
const rangeError = document.getElementById('range-error');

// Setup upload box 1
setupRangeUpload(rangeUploadBox1, rangeFileInput1, rangePreview1, rangePreviewImg1, rangeRemoveBtn1, 1);

// Setup upload box 2
setupRangeUpload(rangeUploadBox2, rangeFileInput2, rangePreview2, rangePreviewImg2, rangeRemoveBtn2, 2);

function setupRangeUpload(uploadBox, fileInput, preview, previewImg, removeBtn, index) {
    // Click to upload
    uploadBox.addEventListener('click', (e) => {
        if (!e.target.classList.contains('remove-btn')) {
            fileInput.click();
        }
    });
    
    // Drag and drop
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });
    
    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('dragover');
    });
    
    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleRangeImageUpload(files[0], index, uploadBox, preview, previewImg);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleRangeImageUpload(e.target.files[0], index, uploadBox, preview, previewImg);
        }
    });
    
    // Remove image
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (index === 1) {
            rangeImageFile1 = null;
        } else {
            rangeImageFile2 = null;
        }
        preview.style.display = 'none';
        uploadBox.querySelector('.upload-placeholder').style.display = 'flex';
        updateRangeAnalyzeButton();
        rangeResults.style.display = 'none';
        rangeError.style.display = 'none';
    });
}

function handleRangeImageUpload(file, index, uploadBox, preview, previewImg) {
    if (!file.type.match('image/(png|jpeg|jpg)')) {
        showError('range', 'Please upload a PNG or JPEG image.');
        return;
    }
    
    if (index === 1) {
        rangeImageFile1 = file;
    } else {
        rangeImageFile2 = file;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadBox.querySelector('.upload-placeholder').style.display = 'none';
        preview.style.display = 'flex';
        updateRangeAnalyzeButton();
        rangeError.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function updateRangeAnalyzeButton() {
    rangeAnalyzeBtn.disabled = !(rangeImageFile1 && rangeImageFile2);
}

// Analyze range of motion
rangeAnalyzeBtn.addEventListener('click', async () => {
    if (!rangeImageFile1 || !rangeImageFile2) return;
    
    rangeLoading.style.display = 'flex';
    rangeResults.style.display = 'none';
    rangeError.style.display = 'none';
    rangeAnalyzeBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('image1', rangeImageFile1);
    formData.append('image2', rangeImageFile2);
    
    try {
        const response = await fetch(`${API_BASE_URL}/analyze-range`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }
        
        // Display results
        document.getElementById('range-angle-1').textContent = data.angle1.toFixed(1);
        document.getElementById('range-angle-2').textContent = data.angle2.toFixed(1);
        document.getElementById('range-rom').textContent = Math.abs(data.range_of_motion).toFixed(1);
        rangeResults.style.display = 'block';
        
        // Display annotated images if available
        if (data.annotated1_exists && data.annotated2_exists) {
            const annotatedImg1 = document.getElementById('range-annotated-img-1');
            const annotatedImg2 = document.getElementById('range-annotated-img-2');
            
            annotatedImg1.src = `${API_BASE_URL}/annotated/${data.filename1.split('.')[0]}_annotated.png`;
            annotatedImg2.src = `${API_BASE_URL}/annotated/${data.filename2.split('.')[0]}_annotated.png`;
            
            document.getElementById('range-annotated').style.display = 'grid';
        }
        
    } catch (error) {
        showError('range', error.message);
    } finally {
        rangeLoading.style.display = 'none';
        updateRangeAnalyzeButton();
    }
});

// ============================================
// UTILITY FUNCTIONS
// ============================================

function showError(type, message) {
    const errorElement = document.getElementById(`${type}-error`);
    errorElement.textContent = message;
    errorElement.style.display = 'block';
}

// Check server health on load
async function checkServerHealth() {
    try {
        console.log(API_BASE_URL);

        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            console.warn('Server health check failed');
        }
    } catch (error) {
        console.error('Cannot connect to server. Please ensure the backend is running.');
    }
}

checkServerHealth();
