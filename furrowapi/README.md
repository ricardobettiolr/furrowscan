# FurrowScan API

**FurrowScan** es un microservicio basado en FastAPI que permite subir imágenes de cultivos, detectar enfermedades con modelos YOLOv8 personalizados y generar recomendaciones usando OpenAI.

---

## 🚀 Endpoints principales

### `POST /furrowscan`

Sube una imagen de un cultivo para:

1. Detectar enfermedad con YOLOv8 (según el tipo de cultivo)
2. Obtener una recomendación agronómica generada por OpenAI

#### Ejemplo con `curl`:

```bash
curl -X POST https://furrowscan.onrender.com/furrowscan \
  -F "file=@ruta/a/imagen_tomate.jpg"
```

---

## 🧠 Estructura de Carpetas

```
furrowapi/
├── app/
│   ├── main.py                # FastAPI app
│   ├── routes/diagnostico.py # Endpoint /furrowscan
│   ├── services/             # Lógica de YOLOv8 y OpenAI
│   ├── utils/file_tools.py   # Utilidades para archivos
│   └── config.py             # Variables de entorno
├── models/                   # Modelos YOLOv8 (.pt)
├── requirements.txt
├── render.yaml
└── README.md
```

---

## 🔐 Variables de entorno necesarias

- `OPENAI_API_KEY`: clave de API para usar OpenAI
- (Opcional) `YOLO_CONFIDENCE`: umbral de confianza para predicciones (default: 0.4)

---

## 📦 Instalación local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## 🌐 Despliegue en Render

1. Crea un servicio web Python
2. Root Directory: `furrowapi`
3. Start Command:  
   ```
   uvicorn app.main:app --host 0.0.0.0 --port 10000
   ```
4. Agrega `OPENAI_API_KEY` como variable de entorno

---

## 🧪 Modelos compatibles

Actualmente disponibles:

- `tomate.pt`
- `papa.pt`
- `pimiento.pt`
- `arroz.pt`
- `cafe.pt`

---

© 2025 Furrow Technologies
