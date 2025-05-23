
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn
import os
from PIL import Image, UnidentifiedImageError
import shutil
import uuid
import openai
import asyncio
from pathlib import Path


# Configura tu clave de OpenAI (recomendado: usar variable de entorno)
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS para producción
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://furrowtech.com"],  # Cambia por tu dominio real
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cultivos válidos y mapeo a carpetas de modelos
CULTIVOS_VALIDOS = ["tomate", "papa", "pimiento", "arroz", "cafe"]
MAPA_MODELOS = {
    "tomate": "tomatepapapimiento",
    "papa": "tomatepapapimiento",
    "pimiento": "tomatepapapimiento",
    "arroz": "arroz",
    "cafe": "cafe"
}

# Recomendaciones por defecto (puedes expandirlas si quieres)
RECOMENDACIONES = {
    "Tomato_Early_blight": "Aplica fungicida a base de cobre y elimina hojas infectadas.",
    "Tomato_Leaf_Mold": "Aumenta ventilación y evita riego nocturno. Usa fungicidas orgánicos.",
    "Pepper__bell___Bacterial_spot": "Elimina hojas enfermas. Usa productos con cobre.",
    "Potato___Late_blight": "Mejora drenaje del suelo. Usa fungicidas preventivos.",
    "Potato___Early_blight": "Aplica tratamiento con mancozeb o similar.",
    "Tomato_healthy": "Tu cultivo está saludable. Mantén prácticas preventivas.",
    "Pepper__bell___healthy": "El pimiento está sano. Continúa el monitoreo regular.",
    "Potato___healthy": "Cultivo en buen estado. No se detectan anomalías.",
    "default": "No hay una recomendación específica para esta clase."
}

async def generar_recomendacion_openai(diagnostico, cultivo):
    prompt = (
        f"Soy un agrónomo experto. El cultivo es '{cultivo}'. "
        f"El diagnóstico de la hoja es: '{diagnostico}'. "
        "Redacta una recomendación clara, profesional y breve para el agricultor, "
        "indicando acciones concretas para tratar o prevenir el problema detectado."
    )
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "No se pudo generar una recomendación personalizada en este momento."

@app.post("/furrowscan")
async def predecir(
    cultivo: str = Form(...),
    imagen: UploadFile = File(...)
):
    cultivo = cultivo.lower()
    if cultivo not in CULTIVOS_VALIDOS:
        return JSONResponse(status_code=400, content={"error": f"Cultivo '{cultivo}' no soportado."})

    modelo_folder = MAPA_MODELOS.get(cultivo)
    BASE_DIR = Path(__file__).resolve().parent
    modelo_path = BASE_DIR / "models" / modelo_folder / "best.pt"
    if not os.path.exists(modelo_path):
        return JSONResponse(status_code=404, content={"error": f"Modelo para '{cultivo}' no encontrado."})

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = f"{uuid.uuid4().hex}_{imagen.filename}"
    image_path = os.path.join(temp_dir, temp_filename)

    try:
        with open(image_path, "wb") as f:
            shutil.copyfileobj(imagen.file, f)

        # Validación de imagen
        try:
            with Image.open(image_path) as img:
                img.verify()
        except UnidentifiedImageError:
            os.remove(image_path)
            return JSONResponse(status_code=400, content={"error": "El archivo subido no es una imagen válida."})

        model = YOLO(str(modelo_path))
        results = model.predict(image_path, conf=0.4)

        clases = results[0].names
        boxes = results[0].boxes

        diagnosticos = []
        if boxes and boxes.data.shape[0] > 0:
            # Llama a OpenAI para cada diagnóstico (en paralelo)
            tasks = []
            for i in range(len(boxes.cls)):
                clase_id = int(boxes.cls[i])
                clase = clases[clase_id]
                diagnostico_legible = clase.replace("_", " ").replace("___", " - ").replace("__", " - ")
                tasks.append(generar_recomendacion_openai(diagnostico_legible, cultivo))
                diagnosticos.append({
                    "diagnostico": diagnostico_legible,
                    "recomendacion": None  # Se llenará después
                })
            recomendaciones_ai = await asyncio.gather(*tasks)
            for i, rec in enumerate(recomendaciones_ai):
                diagnosticos[i]["recomendacion"] = rec
        else:
            recomendacion_ai = await generar_recomendacion_openai("No se detectaron enfermedades.", cultivo)
            diagnosticos.append({
                "diagnostico": "No se detectaron enfermedades.",
                "recomendacion": recomendacion_ai
            })

        os.remove(image_path)
        return {"resultados": diagnosticos}

    except Exception as e:
        if os.path.exists(image_path):
            os.remove(image_path)
        return JSONResponse(status_code=500, content={"error": f"Error al procesar la imagen: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)