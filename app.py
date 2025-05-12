from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from mangum import Mangum
import tensorflow as tf
import numpy as np
import os
import io
import tempfile
import librosa
import noisereduce as nr
import soundfile as sf

app = FastAPI(title="Lung Sound Classification API")

model = None  # global model


@app.on_event("startup")
def load_model():
    global model
    model_path = os.environ.get("MODEL_PATH", "/opt/models/lung_sound_classification_model.keras")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}")
    model = tf.keras.models.load_model(model_path)


def predict_health(audio_data: np.ndarray, sr: int):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=400)
    mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)
    preds = model.predict(mfccs_processed)
    cls = int(np.argmax(preds))
    conf = float(np.max(preds))
    status = "Healthy" if cls == 1 else "Unhealthy"
    return {"status": status, "confidence": conf}


@app.get("/", summary="API health check")
async def health():
    return {"status": "Healthy"}


@app.post("/predict", summary="Predict health status from a WAV file")
async def predict(audio_file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio_file.read())
        tmp_path = tmp.name

    try:
        audio, sr = librosa.load(tmp_path, sr=None)
        result = predict_health(audio, sr)
    finally:
        os.remove(tmp_path)

    return JSONResponse(result)


@app.post("/noise-reduction", summary="Reduce noise given two WAV files")
async def noise_reduction(
    noise_only: UploadFile = File(...),
    heart_noisy: UploadFile = File(...)
):
    # write both to temp
    paths = {}
    for label, up in (("noise", noise_only), ("heart", heart_noisy)):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await up.read())
            paths[label] = tmp.name

    try:
        noisy, sr = librosa.load(paths["heart"], sr=None)
        noise, _ = librosa.load(paths["noise"], sr=sr)
        clean = nr.reduce_noise(y=noisy, y_noise=noise, sr=sr)

        buffer = io.BytesIO()
        sf.write(buffer, clean, sr, format="WAV")
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="cleaned.wav"'}
        )
    finally:
        for p in paths.values():
            if os.path.exists(p):
                os.remove(p)


# AWS Lambda handler
handler = Mangum(app)
