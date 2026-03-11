document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadContent = document.getElementById('upload-content');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const predictBtn = document.getElementById('predict-btn');
    const modelSelect = document.getElementById('model-select');
    const results = document.getElementById('results');
    const predictionLabel = document.getElementById('prediction-label');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    const loader = document.getElementById('loader');

    let selectedFile = null;

    // Abrir selector de archivos al hacer clic
    dropZone.addEventListener('click', () => {
        if (!selectedFile) fileInput.click();
    });

    // Manejar selección de archivos
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    // Drag and Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary)';
        dropZone.style.background = 'rgba(99, 102, 241, 0.1)';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'var(--border)';
        dropZone.style.background = 'transparent';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        handleFiles(e.dataTransfer.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    uploadContent.classList.add('hidden');
                    previewContainer.classList.remove('hidden');
                    predictBtn.disabled = false;
                    results.classList.add('hidden');
                };
                reader.readAsDataURL(file);
            }
        }
    }

    // Quitar imagen
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = '';
        uploadContent.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        predictBtn.disabled = true;
        results.classList.add('hidden');
    });

    // Enviar a la API
    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('model_id', modelSelect.value);

        loader.classList.remove('hidden');
        results.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Error en el servidor');
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            alert('Error al realizar la predicción: ' + error.message);
        } finally {
            loader.classList.add('hidden');
        }
    });

    function displayResults(data) {
        predictionLabel.textContent = data.prediction;

        // Animación de la barra de confianza
        confidenceBar.style.width = '0%';
        results.classList.remove('hidden');

        setTimeout(() => {
            confidenceBar.style.width = `${data.confidence}%`;
            confidenceText.textContent = `${data.confidence}% confianza`;
        }, 100);

        // Scroll suave a los resultados
        results.scrollIntoView({ behavior: 'smooth' });
    }
});
