from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import os
import tempfile
import librosa
import soundfile as sf
import noisereduce as nr
from starlette.middleware.cors import CORSMiddleware
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

# Load model
def load_audio_model(model_path):
    try:
        loaded_model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    global model
    model_path = "models/lung_sound_classification_model.keras"
    if os.path.exists(model_path):
        model = load_audio_model(model_path)
    else:
        print(f"Model not found at: {model_path}")

def predict_health(audio_data, sr):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=400)
    mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)
    prediction = model.predict(mfccs_processed)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    status = "Healthy" if predicted_class == 1 else "Unhealthy"
    return {"status": status, "confidence": confidence}

@app.get("/")
async def health_check():
    return {"status": "Healthy"}

@app.post("/predict")
async def predict(audio_file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(await audio_file.read())
        tmp_path = tmp.name

    try:
        audio, sr = librosa.load(tmp_path, sr=None)
        result = predict_health(audio, sr)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)
