import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Diccionario para cachear los modelos
models = {}

MODEL_PATHS = {
    "basico": "modelos/modelo_basico_color.keras",
    "cnn_bn": "modelos/modelo_cnn_bn.keras",
    "cnn_bn_dropout": "modelos/modelo_cnn_bn_dropout.keras",
    "data_aug": "modelos/modelo_data_aug_color.keras"
}

def load_model(model_id):
    if model_id not in models:
        path = MODEL_PATHS.get(model_id)
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Modelo {model_id} no encontrado")
        
        print(f"--- Cargando modelo: {path} ---")
        try:
            # Ahora cargamos de forma normal, ya que no hay capas Lambda problemáticas
            models[model_id] = tf.keras.models.load_model(path, safe_mode=False, compile=False)
            print(f"--- Modelo {model_id} cargado con éxito ---")
        except Exception as e:
            print(f"!!! Error cargando modelo {model_id}: {str(e)} !!!")
            raise HTTPException(status_code=500, detail=f"Error cargando modelo: {str(e)}")
            
    return models[model_id]

def preprocess_image(image_bytes, model_id):
    # Abrir la imagen con PIL
    img = Image.open(io.BytesIO(image_bytes))
    
    # Redimensionar a 180x180 (el tamaño que usan tus modelos)
    img = img.resize((180, 180))

    # TRATAMIENTO SEGÚN EL MODELO
    if model_id in ["cnn_bn", "cnn_bn_dropout"]:
        # Estos modelos esperan 1 canal (Blanco y Negro)
        print(f"Convirtiendo imagen a escala de grises (1 canal) para {model_id}...")
        img = img.convert("L") 
        img_array = np.array(img).astype(np.float32)
        # Añadir dimensiones: (batch, height, width, channels) -> (1, 180, 180, 1)
        img_array = np.expand_dims(img_array, axis=(0, -1))
    else:
        # Los otros modelos esperan 3 canales (Color RGB)
        print("Asegurando formato RGB (3 canales)...")
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img).astype(np.float32)
        # (1, 180, 180, 3)
        img_array = np.expand_dims(img_array, axis=0)
    
    # NOTA: No normalizamos (/255) aquí porque el modelo 
    # tiene su propia capa Rescaling interna.
    
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_id: str = Form("basico")):
    print(f"--- Petición recibida: {model_id} ---")
    if model_id not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="ID de modelo inválido")
    
    try:
        contents = await file.read()
        
        # Preprocesar según el modelo elegido
        input_data = preprocess_image(contents, model_id)
        
        # Obtener el modelo (lo carga de disco si no está en memoria)
        model = load_model(model_id)
        
        # Predicción
        prediction = model.predict(input_data)
        score = float(prediction[0][0])
        
        # Etiqueta (0.5 como umbral estándar)
        label = "Perro" if score >= 0.5 else "Gato"
        confidence = score if score >= 0.5 else 1.0 - score
        
        print(f"Predicción final: {label} ({confidence*100:.2f}%)")
        
        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "model_used": model_id
        }
        
    except Exception as e:
        print(f"!!! Error en predicción: {str(e)} !!!")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_all")
async def predict_all(file: UploadFile = File(...)):
    print("--- Petición de comparación recibida (Todos los modelos) ---")
    
    try:
        contents = await file.read()
        results = []
        
        # Diccionario legible para el frontend
        friendly_names = {
            "basico": "Básico (Color)",
            "cnn_bn": "CNN BN (Gris)",
            "cnn_bn_dropout": "CNN Dropout (Gris)",
            "data_aug": "Data Augmentation (Color)"
        }
        
        for model_id in MODEL_PATHS.keys():
            try:
                # Preprocesar según el modelo
                input_data = preprocess_image(contents, model_id)
                
                # Cargar y predecir
                model = load_model(model_id)
                prediction = model.predict(input_data)
                score = float(prediction[0][0])
                
                label = "Perro" if score >= 0.5 else "Gato"
                confidence = score if score >= 0.5 else 1.0 - score
                
                results.append({
                    "id": model_id,
                    "name": friendly_names.get(model_id, model_id),
                    "prediction": label,
                    "confidence": round(confidence * 100, 2)
                })
            except Exception as inner_e:
                print(f"Error procesando modelo {model_id}: {str(inner_e)}")
                results.append({
                    "id": model_id,
                    "name": friendly_names.get(model_id, model_id),
                    "prediction": "Error",
                    "confidence": 0,
                    "error": str(inner_e)
                })
        
        return results
        
    except Exception as e:
        print(f"!!! Error en predict_all: {str(e)} !!!")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    # Puerto para Hugging Face
    uvicorn.run(app, host="0.0.0.0", port=7860)
