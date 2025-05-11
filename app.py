from fastapi import FastAPI, File, UploadFile, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from mangum import Mangum
import numpy as np
import tensorflow as tf
import librosa, io, logging
from typing import Optional

# ——— Setup ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Lung Sound Classifier API")
handler = Mangum(app)

MODEL_PATH = "models/lung_sound_classification_model.keras"
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_SAMPLES = EXPECTED_SAMPLE_RATE  # 1 second of audio
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

class ErrorResponse(BaseModel):
    code: int
    message: str
    details: Optional[str] = None

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error: {exc.detail}")
    return JSONResponse(exc.status_code, ErrorResponse(code=exc.status_code, message=str(exc.detail)).dict())

@app.exception_handler(Exception)
async def unhandled_exc(request: Request, exc: Exception):
    logger.error("Unhandled error", exc_info=exc)
    return JSONResponse(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Internal server error",
            details=str(exc)
        ).dict()
    )

# Load model once
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.critical(f"Failed loading model: {e}", exc_info=e)
    model = None

class_names = ["Healthy", "Unhealthy"]

def extract_features_from_waveform(waveform: tf.Tensor) -> tf.Tensor:
    spec = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    mag = tf.abs(spec)
    feats = tf.reduce_mean(mag, axis=1)
    # Pad or crop to 400 frames
    n = tf.shape(feats)[0]
    feats = tf.cond(n < 400,
                    lambda: tf.pad(feats, [[0, 400 - n]]),
                    lambda: feats[:400])
    return tf.reshape(feats, (400, 1))

@app.post(
    "/predict",
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}
)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not available")

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Only .wav files are supported")

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large; must be ≤ 5 MB")

    # --- Resample to 16 kHz using librosa ---
    try:
        y, sr = librosa.load(io.BytesIO(contents), sr=EXPECTED_SAMPLE_RATE, mono=True)
    except Exception as e:
        raise HTTPException(400, f"Failed to read/resample audio: {e}")

    # Trim or pad to EXACTLY 1 second
    if len(y) < EXPECTED_SAMPLES:
        y = np.pad(y, (0, EXPECTED_SAMPLES - len(y)))
    else:
        y = y[:EXPECTED_SAMPLES]

    waveform = tf.convert_to_tensor(y, dtype=tf.float32)

    # --- Feature extraction & inference ---
    features = extract_features_from_waveform(waveform)
    features = tf.expand_dims(features, 0)  # shape (1,400,1)

    logits = model(features, training=False)[0]
    probs = tf.nn.softmax(logits).numpy()
    idx = int(np.argmax(probs))
    label = class_names[idx]
    confidence = round(float(probs[idx]) * 100.0, 2)

    return {"predicted_class": label, "confidence_percent": confidence}

@app.get("/", response_model=dict)
def health_check():
    return {"status": "healthy"}
