/**
 * Fracture Detection Frontend JavaScript
 * Handles file upload, API communication, and UI interactions
 */

class FractureDetectionApp {
    constructor() {
        this.apiUrl = 'http://localhost:8000';
        this.currentFile = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.setupDragAndDrop();
        this.checkApiHealth();
    }

    bindEvents() {
        // File selection
        document.getElementById('selectFileBtn').addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });

        document.getElementById('fileInput').addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Change file button
        document.getElementById('changeFileBtn').addEventListener('click', () => {
            this.resetUpload();
        });

        // Analyze button
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.analyzeImage();
        });

        // New analysis button
        document.getElementById('newAnalysisBtn').addEventListener('click', () => {
            this.resetUpload();
        });

        // Download report button
        document.getElementById('downloadReportBtn').addEventListener('click', () => {
            this.downloadReport();
        });

        // Modal close buttons
        document.getElementById('closeErrorModal').addEventListener('click', () => {
            this.hideModal('errorModal');
        });

        document.getElementById('retryBtn').addEventListener('click', () => {
            this.hideModal('errorModal');
            this.analyzeImage();
        });

        document.getElementById('cancelBtn').addEventListener('click', () => {
            this.hideModal('errorModal');
        });

        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleNavigation(e.target.getAttribute('href').substring(1));
            });
        });
    }

    setupDragAndDrop() {
        const uploadArea = document.getElementById('uploadArea');

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });

        uploadArea.addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });
    }

    async checkApiHealth() {
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            const data = await response.json();
            
            if (!data.model_loaded) {
                this.showNotification('Warning: Model not loaded. Running in demo mode.', 'warning');
            }
        } catch (error) {
            console.warn('API health check failed:', error);
            this.showNotification('Warning: Backend service may be unavailable.', 'warning');
        }
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file (PNG, JPG, JPEG)');
            return;
        }

        // Validate file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('File size exceeds 10MB limit');
            return;
        }

        this.currentFile = file;
        this.displayFilePreview(file);
    }

    displayFilePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('previewImage').src = e.target.result;
            document.getElementById('fileName').textContent = file.name;
            document.getElementById('fileSize').textContent = this.formatFileSize(file.size);
            document.getElementById('fileType').textContent = file.type;

            document.getElementById('uploadArea').style.display = 'none';
            document.getElementById('uploadPreview').style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async analyzeImage() {
        if (!this.currentFile) {
            this.showError('Please select a file first');
            return;
        }

        this.showLoadingModal();
        this.showResultsSection(true);

        try {
            const formData = new FormData();
            formData.append('file', this.currentFile);

            const response = await fetch(`${this.apiUrl}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayResults(result);

        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError('Failed to analyze image. Please try again.');
        } finally {
            this.hideLoadingModal();
        }
    }

    displayResults(result) {
        const resultsContent = document.getElementById('resultsContent');
        const processingIndicator = document.getElementById('processingIndicator');

        // Hide processing indicator
        processingIndicator.style.display = 'none';

        // Update results
        document.getElementById('diagnosisResult').textContent = 
            result.fracture_detected ? 'Fracture Detected' : 'No Fracture';
        
        document.getElementById('confidenceResult').textContent = 
            `${(result.confidence * 100).toFixed(1)}%`;
        
        document.getElementById('processingTime').textContent = 
            result.processing_time ? `${result.processing_time.toFixed(2)}s` : 'N/A';

        // Update result message
        const resultMessage = document.getElementById('resultMessage');
        resultMessage.innerHTML = `
            <div class="result-message ${result.fracture_detected ? 'fracture-detected' : 'no-fracture'}">
                <i class="fas ${result.fracture_detected ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i>
                <strong>${result.message}</strong>
                ${result.fracture_detected ? 
                    '<p>This requires immediate medical attention. Please consult with a healthcare professional.</p>' : 
                    '<p>No signs of fracture found in this image.</p>'
                }
            </div>
        `;

        // Add CSS for result styling
        if (result.fracture_detected) {
            resultMessage.querySelector('.result-message').style.background = '#ffebee';
            resultMessage.querySelector('.result-message').style.borderColor = '#f44336';
            resultMessage.querySelector('.result-message').style.color = '#c62828';
        } else {
            resultMessage.querySelector('.result-message').style.background = '#e8f5e8';
            resultMessage.querySelector('.result-message').style.borderColor = '#4caf50';
            resultMessage.querySelector('.result-message').style.color = '#2e7d32';
        }

        // Show results content
        resultsContent.style.display = 'block';
    }

    resetUpload() {
        this.currentFile = null;
        document.getElementById('fileInput').value = '';
        document.getElementById('uploadArea').style.display = 'block';
        document.getElementById('uploadPreview').style.display = 'none';
        this.hideResultsSection();
    }

    showResultsSection(show = true) {
        const resultsSection = document.getElementById('resultsSection');
        if (show) {
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        } else {
            resultsSection.style.display = 'none';
        }
    }

    hideResultsSection() {
        this.showResultsSection(false);
    }

    showLoadingModal() {
        document.getElementById('loadingModal').style.display = 'block';
    }

    hideLoadingModal() {
        document.getElementById('loadingModal').style.display = 'none';
    }

    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        document.getElementById('errorModal').style.display = 'block';
    }

    hideModal(modalId) {
        document.getElementById(modalId).style.display = 'none';
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas ${type === 'warning' ? 'fa-exclamation-triangle' : 'fa-info-circle'}"></i>
            <span>${message}</span>
            <button class="notification-close">&times;</button>
        `;

        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: ${type === 'warning' ? '#fff3cd' : '#d1ecf1'};
            color: ${type === 'warning' ? '#856404' : '#0c5460'};
            border: 1px solid ${type === 'warning' ? '#ffeaa7' : '#bee5eb'};
            border-radius: 8px;
            padding: 1rem;
            z-index: 3000;
            max-width: 400px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            animation: slideIn 0.3s ease;
        `;

        // Add close button functionality
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.remove();
        });

        // Add to page
        document.body.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    handleNavigation(sectionId) {
        // Update active nav link
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[href="#${sectionId}"]`).classList.add('active');

        // Smooth scroll to section
        const section = document.getElementById(sectionId);
        if (section) {
            section.scrollIntoView({ behavior: 'smooth' });
        }
    }

    downloadReport() {
        if (!this.currentFile) return;

        const report = {
            fileName: this.currentFile.name,
            analysisDate: new Date().toISOString(),
            diagnosis: document.getElementById('diagnosisResult').textContent,
            confidence: document.getElementById('confidenceResult').textContent,
            processingTime: document.getElementById('processingTime').textContent,
            message: document.getElementById('resultMessage').textContent
        };

        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `fracture-analysis-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    .notification-close {
        background: none;
        border: none;
        font-size: 1.2rem;
        cursor: pointer;
        margin-left: auto;
        opacity: 0.7;
    }

    .notification-close:hover {
        opacity: 1;
    }
`;
document.head.appendChild(style);

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FractureDetectionApp();
}); 