import os
import numpy as np
import tensorflow as tf
import builtins
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io

# Hacer 'tf' disponible globalmente para que las capas Lambda puedan encontrarlo
builtins.tf = tf

app = FastAPI()

# Configurar CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos (CSS, JS) para producción
app.mount("/static", StaticFiles(directory="static"), name="static")

# Diccionario para cachear los modelos
models = {}

MODEL_PATHS = {
    "basico": "modelos/modelo_basico_color.keras",
    "cnn_bn": "modelos/modelo_cnn_bn.keras",
    "data_aug": "modelos/modelo_data_aug_color.keras"
}

# Habilitar la deserialización de capas Lambda (necesario en versiones recientes de Keras/TF)
try:
    import keras
    keras.config.enable_unsafe_deserialization()
except:
    # Si estamos en una versión donde la ruta es distinta
    try:
        tf.keras.config.enable_unsafe_deserialization()
    except:
        pass

def load_model(model_id):
    if model_id not in models:
        path = MODEL_PATHS.get(model_id)
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        print(f"Cargando modelo: {path}...")
        # Intentar cargar con safe_mode=False si la función lo soporta
        try:
            models[model_id] = tf.keras.models.load_model(path, safe_mode=False)
        except TypeError:
            models[model_id] = tf.keras.models.load_model(path)
    return models[model_id]

def preprocess_image(image_bytes):
    # Abrir la imagen con PIL
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convertir a RGB si no lo está
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    # Es recomendable redimensionar en Python antes de pasarlo a TensorFlow 
    # para evitar enviar imágenes excesivamente pesadas a la memoria de la GPU/CPU, 
    # aunque la capa Resizing del modelo también lo haga.
    img = img.resize((180, 180))
    
    # Convertir a array
    img_array = np.array(img).astype(np.float32)
    
    # Añadir dimensión de batch (1, 180, 180, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_id: str = Form("basico")):
    if model_id not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="ID de modelo inválido")
    
    try:
        # Leer el contenido de la imagen
        contents = await file.read()
        
        # Preprocesar la imagen
        input_data = preprocess_image(contents)
        
        # Cargar el modelo deseado
        model = load_model(model_id)
        
        # Realizar la predicción
        prediction = model.predict(input_data)
        
        # En clasificación binaria Perros vs Gatos:
        # Típicamente 0 = Gato, 1 = Perro (o viceversa, depende del entrenamiento)
        # Asumiendo sigmoide en la salida (1 neurona)
        score = float(prediction[0][0])
        
        # Definir la etiqueta basada en el score (umbral 0.5)
        # Adaptar según cómo se entrenó el modelo. 
        # Normalmente: Gato < 0.5, Perro >= 0.5
        label = "Perro" if score >= 0.5 else "Gato"
        confidence = score if score >= 0.5 else 1.0 - score
        
        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "raw_score": score,
            "model_used": model_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    # Puerto 7860 es el estándar requerido por Hugging Face Spaces (Docker)
    uvicorn.run(app, host="0.0.0.0", port=7860)
