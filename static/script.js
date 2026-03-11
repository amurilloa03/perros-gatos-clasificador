document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadContent = document.getElementById('upload-content');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const predictBtn = document.getElementById('predict-btn');
    const results = document.getElementById('results');
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

        loader.classList.remove('hidden');
        results.classList.add('hidden');

        try {
            // Ahora usamos /predict_all para obtener todos los resultados
            const response = await fetch('/predict_all', {
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
            alert('Error al realizar la comparativa: ' + error.message);
        } finally {
            loader.classList.add('hidden');
        }
    });

    function displayResults(dataList) {
        const grid = document.getElementById('comparison-grid');
        grid.innerHTML = ''; // Limpiar resultados anteriores

        dataList.forEach(item => {
            const card = document.createElement('div');
            card.className = 'result-card';

            card.innerHTML = `
                <span class="model-name">${item.name}</span>
                <div class="prediction-label">${item.prediction}</div>
                <div class="confidence-container">
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width: 0%"></div>
                    </div>
                    <span class="confidence-val">${item.confidence}%</span>
                </div>
            `;

            grid.appendChild(card);

            // Animación de la barra después de un pequeño delay
            setTimeout(() => {
                const bar = card.querySelector('.confidence-bar-fill');
                bar.style.width = `${item.confidence}%`;
            }, 100);
        });

        results.classList.remove('hidden');

        // Scroll suave
        results.scrollIntoView({ behavior: 'smooth' });
    }
});
