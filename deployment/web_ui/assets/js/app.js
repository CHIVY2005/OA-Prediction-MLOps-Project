const API_URL = "/predict";

let currentRequestId = null;
let currentPrediction = null;

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadSection = document.getElementById('upload-section');
const loadingContainer = document.getElementById('loading-container');
const resultSection = document.getElementById('result-section');
const resetBtn = document.getElementById('reset-btn');

// Feedback elements
const feedbackContainer = document.getElementById('feedback-container');
const btnFeedbackYes = document.getElementById('btn-feedback-yes');
const btnFeedbackNo = document.getElementById('btn-feedback-no');
const correctedSelection = document.getElementById('corrected-selection');
const correctGradeSelect = document.getElementById('correct-grade-select');
const btnSubmitFeedback = document.getElementById('btn-submit-feedback');
const thankYouMsg = document.getElementById('thank-you-msg');

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

// Feedback Event Listeners
btnFeedbackYes.addEventListener('click', async () => {
    btnFeedbackYes.classList.add('active');
    btnFeedbackNo.classList.remove('active');
    btnFeedbackYes.disabled = true;
    btnFeedbackNo.disabled = true;
    correctedSelection.classList.add('hidden');
    
    await submitFeedback('correct', null);
});

btnFeedbackNo.addEventListener('click', () => {
    btnFeedbackNo.classList.add('active');
    btnFeedbackYes.classList.remove('active');
    correctedSelection.classList.remove('hidden');
    thankYouMsg.classList.add('hidden');
});

btnSubmitFeedback.addEventListener('click', async () => {
    const correctedGrade = correctGradeSelect.value;
    btnSubmitFeedback.disabled = true;
    btnFeedbackYes.disabled = true;
    btnFeedbackNo.disabled = true;
    
    await submitFeedback('incorrect', correctedGrade);
    btnSubmitFeedback.disabled = false;
});

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
            const errorData = await response.json().catch(() => null);
            const detail = errorData && errorData.detail ? errorData.detail : response.statusText;
            throw new Error(`API Error (${response.status}): ${detail}`);
        }

        const data = await response.json();
        currentRequestId = data.request_id;
        currentPrediction = data.prediction;
        showResult(data);
        resetFeedbackUI();

    } catch (error) {
        console.error(error);
        alert(`Verification Failed: ${error.message}\nMake sure API is running and file is valid.`);
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

function resetFeedbackUI() {
    correctedSelection.classList.add('hidden');
    thankYouMsg.classList.add('hidden');
    btnFeedbackYes.classList.remove('active');
    btnFeedbackNo.classList.remove('active');
    btnFeedbackYes.disabled = false;
    btnFeedbackNo.disabled = false;
    correctGradeSelect.value = "0";
}

function resetApp() {
    resultSection.classList.add('hidden');
    loadingContainer.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    fileInput.value = '';
    currentRequestId = null;
    currentPrediction = null;
    resetFeedbackUI();
}

async function submitFeedback(feedbackType, correctedGrade) {
    if (!currentRequestId) return;
    
    try {
        const response = await fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                request_id: currentRequestId,
                feedback: feedbackType,
                prediction: currentPrediction,
                corrected_grade: correctedGrade
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            const detail = errorData && errorData.detail ? errorData.detail : response.statusText;
            throw new Error(`API Error (${response.status}): ${detail}`);
        }

        // Show success
        correctedSelection.classList.add('hidden');
        thankYouMsg.classList.remove('hidden');

    } catch (error) {
        console.error(error);
        alert(`Feedback submission failed: ${error.message}`);
        btnFeedbackYes.disabled = false;
        btnFeedbackNo.disabled = false;
        btnFeedbackYes.classList.remove('active');
        btnFeedbackNo.classList.remove('active');
    }
}
