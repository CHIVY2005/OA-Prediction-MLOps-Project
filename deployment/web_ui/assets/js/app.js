const API_URL = "http://127.0.0.1:8000/predict";

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadSection = document.getElementById('upload-section');
const loadingContainer = document.getElementById('loading-container');
const resultSection = document.getElementById('result-section');
const resetBtn = document.getElementById('reset-btn');

// --- Event Listeners ---
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');

    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

resetBtn.addEventListener('click', resetApp);

// --- Functions ---
function handleFile(file) {
    if (!file.type.match('image.*')) {
        alert("Please upload an image file (JPEG/PNG)");
        return;
    }

    // Show Image immediately
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('original-img').src = e.target.result;
    };
    reader.readAsDataURL(file);

    uploadImage(file);
}

async function uploadImage(file) {
    // UI State: Loading
    uploadSection.classList.add('hidden');
    resultSection.classList.add('hidden');
    loadingContainer.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        const data = await response.json();
        showResult(data);

    } catch (error) {
        console.error(error);
        alert("Verification Failed: Could not connect to API.\nMake sure 'python -m uvicorn api.main:app' is running!");
        resetApp();
    }
}

function showResult(data) {
    loadingContainer.classList.add('hidden');
    resultSection.classList.remove('hidden');

    // Update Text
    document.getElementById('pred-class').textContent = data.prediction;
    document.getElementById('pred-conf').textContent = data.confidence;

    // Update Heatmap
    if (data.heatmap_base64) {
        const heatmapSrc = `data:image/jpeg;base64,${data.heatmap_base64}`;
        document.getElementById('heatmap-img').src = heatmapSrc;
    } else {
        document.getElementById('heatmap-img').alt = "No Heatmap Available";
    }
}

function resetApp() {
    resultSection.classList.add('hidden');
    loadingContainer.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    fileInput.value = '';
}
