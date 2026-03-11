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

import traceback

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

def rgb_to_grayscale_wrapper(x):
    return tf.image.rgb_to_grayscale(x)

def load_model(model_id):
    if model_id not in models:
        path = MODEL_PATHS.get(model_id)
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        
        print(f"--- Intentando cargar modelo: {path} ---")
        try:
            # Definir objetos personalizados comunes
            # El modelo BN tiene una capa Lambda que usa tf.image.rgb_to_grayscale
            custom_objs = {
                'tf': tf,
                'np': np,
                'rgb_to_grayscale': rgb_to_grayscale_wrapper,
                'lambda': rgb_to_grayscale_wrapper  # A veces se busca con este nombre
            }
            
            # Cargamos el modelo permitiendo objetos personalizados y deserialización insegura
            models[model_id] = tf.keras.models.load_model(
                path, 
                custom_objects=custom_objs, 
                safe_mode=False
            )
            print(f"--- Modelo {model_id} cargado con éxito ---")
        except Exception as e:
            print(f"!!! Error CRÍTICO cargando modelo {model_id} !!!")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error cargando modelo {model_id}: {str(e)}")
            
    return models[model_id]

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_id: str = Form("basico")):
    print(f"--- Nueva petición recibida para el modelo: {model_id} ---")
    if model_id not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="ID de modelo inválido")
    
    try:
        # Leer el contenido de la imagen
        contents = await file.read()
        print(f"Imagen leída: {len(contents)} bytes")
        
        # Preprocesar la imagen
        input_data = preprocess_image(contents)
        print(f"Preprocesamiento completado. Input shape: {input_data.shape}")
        
        # Cargar el modelo deseado
        model = load_model(model_id)
        
        # Realizar la predicción
        print("Ejecutando predicción...")
        prediction = model.predict(input_data)
        print("Predicción completada.")
        
        score = float(prediction[0][0])
        label = "Perro" if score >= 0.5 else "Gato"
        confidence = score if score >= 0.5 else 1.0 - score
        
        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "raw_score": score,
            "model_used": model_id
        }
        
    except Exception as e:
        print(f"!!! Error durante la ejecución de la predicción !!!")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    # Puerto 7860 es el estándar requerido por Hugging Face Spaces (Docker)
    uvicorn.run(app, host="0.0.0.0", port=7860)
