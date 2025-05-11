from typing import Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, status
from fastapi.responses import JSONResponse
from mangum import Mangum
from pydantic import BaseModel, Field
import logging

# ——— Startup & Logging ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lung Sound Classifier API")
handler = Mangum(app)

MODEL_PATH = "models/lung_sound_classification_model.keras"
EXPECTED_SAMPLES = 16000
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


# ——— Error Response Model ———
class ErrorResponse(BaseModel):
    code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Short error message")
    details: Optional[str] = Field(None, description="Additional diagnostic info")


# ——— Global Exception Handlers ———
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.detail}")
    payload = ErrorResponse(code=exc.status_code, message=str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=payload.dict())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", exc_info=exc)
    payload = ErrorResponse(
        code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message="Internal server error",
        details=str(exc)
    )
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload.dict())


# ——— Load Model ———
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.critical(f"Model loading failed: {e}", exc_info=e)
    model = None


class_names = ["Healthy", "Unhealthy"]


# ——— Audio Preprocessing ———
def normalize_waveform(waveform: tf.Tensor) -> tf.Tensor:
    waveform = tf.cast(waveform, tf.float32)
    max_val = tf.reduce_max(tf.abs(waveform))
    return waveform / (max_val + 1e-6)


def extract_features_from_waveform(waveform: tf.Tensor) -> tf.Tensor:
    spec = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    mag = tf.abs(spec)
    features = tf.reduce_mean(mag, axis=1)  # (frames,)
    num_frames = tf.shape(features)[0]

    def pad():
        return tf.pad(features, [[0, 400 - num_frames]])
    def crop():
        return features[:400]

    fixed = tf.cond(num_frames < 400, pad, crop)
    return tf.reshape(fixed, (400, 1))  # (400,1)


# ——— Prediction Endpoint ———
@app.post("/predict", response_model=None, responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def predict(file: UploadFile = File(...)):
    # 1. Model loaded?
    if model is None:
        raise HTTPException(status_code=503, detail="Classification model not available")

    # 2. File extension & size
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Unsupported file type; only .wav allowed")
    if file.spool_max_size and file.spool_max_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large; must be ≤ 5 MB")

    # 3. Read & decode
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        audio_tensor, sample_rate = tf.audio.decode_wav(
            tf.constant(contents),
            desired_channels=1,
            desired_samples=EXPECTED_SAMPLES
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode WAV: {e}")

    # 4. Validate sample rate
    if sample_rate.numpy() != 16000:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sample rate: expected 16000 Hz, got {int(sample_rate.numpy())} Hz"
        )

    # 5. Preprocess & infer
    waveform = tf.squeeze(audio_tensor, axis=-1)
    waveform = normalize_waveform(waveform)
    features = extract_features_from_waveform(waveform)
    features = tf.expand_dims(features, axis=0)  # (1, 400, 1)

    logits = model(features, training=False)[0]
    probs = tf.nn.softmax(logits).numpy()
    idx = int(np.argmax(probs))
    label = class_names[idx]
    confidence = round(float(probs[idx]) * 100.0, 2)

    return {
        "predicted_class": label,
        "confidence_percent": confidence
    }


# ——— Health Check ———
@app.get("/", response_model=dict)
def health_check():
    return {"status": "healthy"}
