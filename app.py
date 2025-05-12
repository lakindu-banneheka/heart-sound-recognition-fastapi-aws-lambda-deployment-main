from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import tensorflow as tf
import numpy as np
import os
import io
import tempfile
import librosa
import noisereduce as nr
import soundfile as sf
from mangum import Mangum
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
handler = Mangum(app)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

@app.on_event("startup")
async def load_model():
    global model
    model_path = "./models/lung_sound_classification_model.keras"
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Failed to load model")

def predict_health(audio_data, sr):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=400)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    mfccs_processed = mfccs_processed.reshape(1, -1)

    prediction = model.predict(mfccs_processed)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "status": "Healthy" if predicted_class == 1 else "Unhealthy",
        "confidence": confidence
    }

@app.post("/predict")
async def predict(audio_file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        new_audio, sr = librosa.load(temp_path, sr=None)
        os.remove(temp_path)
        return predict_health(new_audio, sr)

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "Healthy"}

@app.post("/noise-reduction")
async def noise_reduction(
    noise_only: UploadFile = File(...),
    heart_noisy: UploadFile = File(...)
):
    try:
        # Save noise file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_noise:
            noise_content = await noise_only.read()
            tmp_noise.write(noise_content)
            noise_path = tmp_noise.name

        # Save heart file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_heart:
            heart_content = await heart_noisy.read()
            tmp_heart.write(heart_content)
            heart_path = tmp_heart.name

        # Process files
        noisy, sr = librosa.load(heart_path, sr=None)
        noise, _ = librosa.load(noise_path, sr=sr)
        clean = nr.reduce_noise(y=noisy, y_noise=noise, sr=sr)

        # Clean up temp files
        os.remove(noise_path)
        os.remove(heart_path)

        # Create in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, clean, sr, format='WAV')
        buffer.seek(0)

        return FileResponse(
            buffer,
            media_type="audio/wav",
            filename="cleaned.wav"
        )

    except Exception as e:
        # Cleanup if error occurs
        for path in [noise_path, heart_path]:
            if path and os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")