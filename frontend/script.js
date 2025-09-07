document.addEventListener('DOMContentLoaded', function() {
    const inputText = document.getElementById('input-text');
    const sentencesCount = document.getElementById('sentences-count');
    const sentencesValue = document.getElementById('sentences-value');
    const algorithm = document.getElementById('algorithm');
    const summarizeBtn = document.getElementById('summarize-btn');
    const summaryOutput = document.getElementById('summary-output');
    const originalLength = document.getElementById('original-length');
    const summaryLength = document.getElementById('summary-length');
    const reduction = document.getElementById('reduction');
    const loading = document.getElementById('loading');
    const currentAlgorithm = document.getElementById('current-algorithm');
    const wordCount = document.getElementById('word-count');
    const algoDetails = document.querySelectorAll('.algo');
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.getElementById('upload-area');
    const uploadedFile = document.getElementById('uploaded-file');
    const uploadedFileName = document.getElementById('uploaded-file-name');
    const uploadedFileSize = document.getElementById('uploaded-file-size');
    const removeFileBtn = document.getElementById('remove-file-btn');

    let currentFile = null;

    // Update sentence count display
    sentencesCount.addEventListener('input', function() {
        sentencesValue.textContent = `${this.value} sentence${this.value > 1 ? 's' : ''}`;
    });

    // Update word count and clear file if user types
    inputText.addEventListener('input', function() {
        const text = this.value.trim();
        const words = text ? text.split(/\s+/).length : 0;
        wordCount.textContent = words;

        if (text.length > 0 && currentFile) {
            clearFileSelection();
        }
    });

    // Algorithm selection logic
    algorithm.addEventListener('change', function() {
        updateAlgorithmInfo(this.value);
        currentAlgorithm.textContent = document.querySelector(`option[value="${this.value}"]`).textContent;
    });

    function updateAlgorithmInfo(algo) {
        algoDetails.forEach(item => {
            if (item.dataset.algo === algo) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
    }

    // Initialize algorithm info display
    updateAlgorithmInfo(algorithm.value);
    currentAlgorithm.textContent = document.querySelector(`option[value="${algorithm.value}"]`).textContent;

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect(e);
        }
    });

    // Fixed click handler: trigger file input once
    uploadArea.addEventListener('click', function(e) {
        if (!e.target.closest('.file-input-label') && !e.target.closest('#remove-file-btn')) {
            fileInput.click();
        }
    });

    // Remove selected file
    removeFileBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        clearFileSelection();
    });

    async function handleFileSelect(e) {
        const file = e.target.files[0] || e.dataTransfer.files[0];

        if (!file) return;

        const fileType = file.name.split('.').pop().toLowerCase();
        if (!['pdf', 'docx', 'txt'].includes(fileType)) {
            showError('Please select a PDF, DOCX, or TXT file');
            return;
        }

        if (file.size > 16 * 1024 * 1024) {
            showError('File size exceeds 16MB limit');
            return;
        }

        currentFile = file;

        // For text files, read and display content
        if (fileType === 'txt') {
            try {
                const text = await readFileAsText(file);
                const words = text.split(/\s+/).length;
                
                if (words > 1000) {
                    showError(`File exceeds 1000 words limit (${words} words)`);
                    clearFileSelection();
                    return;
                }
                
                inputText.value = text;
                wordCount.textContent = words;
            } catch (error) {
                showError('Error reading file: ' + error.message);
                clearFileSelection();
                return;
            }
        } else {
            // For PDF and DOCX, clear the text area as we can't read them in browser
            inputText.value = '';
            wordCount.textContent = '0';
        }

        uploadedFileName.textContent = file.name;
        uploadedFileSize.textContent = formatFileSize(file.size);
        uploadedFile.style.display = 'flex';
    }

    function readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = e => resolve(e.target.result);
            reader.onerror = e => reject(new Error('Could not read file'));
            reader.readAsText(file);
        });
    }

    function clearFileSelection() {
        currentFile = null;
        fileInput.value = '';
        uploadedFile.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' bytes';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        else return (bytes / 1048576).toFixed(1) + ' MB';
    }

    summarizeBtn.addEventListener('click', async function() {
        const text = inputText.value.trim();

        if (!text && !currentFile) {
            showError('Please enter text or upload a document');
            return;
        }

        // For text input, check word count
        if (text) {
            const words = text.split(/\s+/).length;
            if (words > 1000) {
                showError(`Text exceeds 1000 words limit (${words} words)`);
                return;
            }
        }

        const count = parseInt(sentencesCount.value);
        const algo = algorithm.value;

        if (count < 1 || count > 10) {
            showError('Please enter a valid number of sentences (1-10)');
            return;
        }

        loading.style.display = 'flex';

        try {
            let response;

            if (currentFile) {
                const formData = new FormData();
                formData.append('file', currentFile);
                formData.append('sentences_count', count);
                formData.append('algorithm', algo);

                response = await fetch('http://localhost:5000/summarize-document', {
                    method: 'POST',
                    body: formData
                });
            } else {
                response = await fetch('http://localhost:5000/summarize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: text,
                        sentences_count: count,
                        algorithm: algo
                    })
                });
            }

            const data = await response.json();

            if (response.ok) {
                summaryOutput.innerHTML = `<p>${data.summary}</p>`;
                originalLength.textContent = `${data.original_length} words`;
                summaryLength.textContent = `${data.summary_length} words`;

                const reductionPercent = Math.round(
                    (1 - data.summary_length / data.original_length) * 100
                );
                reduction.textContent = `${reductionPercent}%`;
            } else {
                showError(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error:', error);
            showError('Error connecting to the server. Make sure the backend is running on port 5000.');
        } finally {
            loading.style.display = 'none';
        }
    });

    function showError(message) {
        summaryOutput.innerHTML = `<p class="error">${message}</p>`;
        originalLength.textContent = '- words';
        summaryLength.textContent = '- words';
        reduction.textContent = '-';
    }

    // Sample text for testing
    inputText.value = ``;

    inputText.dispatchEvent(new Event('input'));
});