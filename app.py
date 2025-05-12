# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
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
TARGET_FRAMES = 25  # expected time steps

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

# Audio reading utility
def read_wav(file_path: str) -> tuple[np.ndarray, int]:
    with wave.open(file_path, 'rb') as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        sample_width = wf.getsampwidth()
        channels = wf.getnchannels()

    # Determine dtype
    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    audio = np.frombuffer(frames, dtype=dtype)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio = audio.astype(np.float32)
    return audio, sr

# Preprocessing to MFCC and shape matching
import python_speech_features as psf

def preprocess_audio(audio_data: np.ndarray, sr: int) -> np.ndarray:
    mfccs = psf.mfcc(audio_data, samplerate=sr, numcep=1)
    mfccs = mfccs  # shape: (time_steps, 1)
    # Pad or truncate
    if mfccs.shape[0] < TARGET_FRAMES:
        pad_amt = TARGET_FRAMES - mfccs.shape[0]
        mfccs = np.vstack([mfccs, np.zeros((pad_amt, mfccs.shape[1]))])
    else:
        mfccs = mfccs[:TARGET_FRAMES, :]
    return mfccs.reshape(1, TARGET_FRAMES, mfccs.shape[1])

# Prediction logic
def predict_health(audio_data: np.ndarray, sr: int, model) -> tuple[str, float]:
    input_tensor = preprocess_audio(audio_data, sr)
    prediction = model.predict(input_tensor)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
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
    if not audio_file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files supported")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(await audio_file.read())
            temp_path = tmp.name

        audio, sr = read_wav(temp_path)
        os.remove(temp_path)
        status, confidence = predict_health(audio, sr, model)
        return {"status": status, "confidence": confidence}
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

@app.post("/noise-reduction")
async def noise_reduction(
    noise_only: UploadFile = File(...),
    heart_noisy: UploadFile = File(...),
):
    if not noise_only.filename.lower().endswith('.wav') or not heart_noisy.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files supported for noise reduction")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_noise:
            tmp_noise.write(await noise_only.read())
            noise_path = tmp_noise.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_heart:
            tmp_heart.write(await heart_noisy.read())
            heart_path = tmp_heart.name

        noise, sr_noise = read_wav(noise_path)
        heart, sr_heart = read_wav(heart_path)
        if sr_noise != sr_heart:
            raise ValueError("Sample rates do not match")

        clean = nr.reduce_noise(y=heart, y_noise=noise, sr=sr_heart)

        # Write back to WAV buffer
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr_heart)
            data = (clean * 32767).astype(np.int16)
            wf.writeframes(data.tobytes())
        buffer.seek(0)

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

# requirements.txt
# ----------------
# fastapi
# mangum
# tensorflow
# numpy
# noisereduce
# python_speech_features
# python-multipart
# aiofiles
# uvicorn
