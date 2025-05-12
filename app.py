from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc
import noisereduce as nr
import os
import io
import tempfile
import wave
from mangum import Mangum

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

# Load model at startup
def load_audio_model(model_path: str):
    try:
        mdl = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return mdl
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@app.on_event("startup")
def startup_event():
    global model
    model_path = "models/lung_sound_classification_model.keras"
    if os.path.exists(model_path):
        model = load_audio_model(model_path)
    else:
        print(f"Model file not found at {model_path}")

# Prediction logic
def predict_health(audio_data: np.ndarray, sr: int, model) -> tuple[str, float]:
    # Compute MFCCs
    mfcc_feat = mfcc(audio_data, samplerate=sr, numcep=400)
    mfccs_processed = np.mean(mfcc_feat, axis=0).reshape(1, -1)
    prediction = model.predict(mfccs_processed)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    status = "Healthy" if predicted_class == 1 else "Unhealthy"
    return status, confidence

@app.get("/")
def health_check():
    return JSONResponse(content={"status": "Healthy"}, status_code=200)

@app.post("/predict")
async def predict(audio_file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio file selected")

    try:
        # Save to temp file
        suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await audio_file.read())
            temp_path = tmp.name

        # Read WAV
        sr, audio_int = wavfile.read(temp_path)
        if audio_int.ndim > 1:
            audio_int = audio_int.mean(axis=1)
        audio = audio_int.astype(float)
        os.remove(temp_path)

        # Predict
        status, confidence = predict_health(audio, sr, model)
        return {"status": status, "confidence": confidence}

    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/noise-reduction")
async def noise_reduction(
    noise_only: UploadFile = File(...),
    heart_noisy: UploadFile = File(...),
):
    if not noise_only.filename or not heart_noisy.filename:
        raise HTTPException(status_code=400, detail="Both files must be provided and have valid names")

    try:
        # Save uploads
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(noise_only.filename)[1] or ".wav") as tmp_noise:
            tmp_noise.write(await noise_only.read())
            noise_path = tmp_noise.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(heart_noisy.filename)[1] or ".wav") as tmp_heart:
            tmp_heart.write(await heart_noisy.read())
            heart_path = tmp_heart.name

        # Read WAVs
        sr_noise, noise_int = wavfile.read(noise_path)
        sr_heart, heart_int = wavfile.read(heart_path)
        if sr_noise != sr_heart:
            raise ValueError("Sample rates of noise and heart files do not match")

        # Mono conversion and float
        if heart_int.ndim > 1:
            heart_int = heart_int.mean(axis=1)
        if noise_int.ndim > 1:
            noise_int = noise_int.mean(axis=1)
        noisy = heart_int.astype(float)
        noise = noise_int.astype(float)

        # Reduce noise
        clean = nr.reduce_noise(y=noisy, y_noise=noise, sr=sr_heart)

        # Write WAV via wave module
        buffer = io.BytesIO()
        data = (clean * 32767).astype(np.int16)
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr_heart)
            wf.writeframes(data.tobytes())
        buffer.seek(0)

        # Cleanup
        os.remove(noise_path)
        os.remove(heart_path)

        headers = {"Content-Disposition": "attachment; filename=cleaned.wav"}
        return StreamingResponse(buffer, media_type="audio/wav", headers=headers)

    except Exception as e:
        for path in (locals().get('noise_path'), locals().get('heart_path')):
            if path and os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

# AWS Lambda handler
handler = Mangum(app)