# Usar una imagen base de Python oficial y moderna
FROM python:3.11-slim

# Evitar que Python genere archivos .pyc y habilitar el log en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos
COPY requirements.txt .

# Instalar las librerías de Python. 
# En Hugging Face podemos usar la versión más reciente de TensorFlow.
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código y los modelos
COPY . .

# Exponer el puerto que usa Hugging Face Spaces (7860)
EXPOSE 7860

# Comando para ejecutar la aplicación
CMD ["python", "main.py"]
