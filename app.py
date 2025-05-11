from fastapi import FastAPI, File, UploadFile, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from mangum import Mangum
import numpy as np
import tensorflow as tf
import logging
from typing import Optional

# ——— Setup & Logging ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lung Sound Classifier API")
handler = Mangum(app)

MODEL_PATH = "models/lung_sound_classification_model.keras"
EXPECTED_SAMPLES = 16000
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

class ErrorResponse(BaseModel):
    code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Short error message")
    details: Optional[str] = Field(None, description="Additional diagnostic info")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(code=exc.status_code, message=str(exc.detail)).dict()
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Internal server error",
            details=str(exc)
        ).dict()
    )

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.critical(f"Model loading failed: {e}", exc_info=e)
    model = None

class_names = ["Healthy", "Unhealthy"]

def normalize_waveform(waveform: tf.Tensor) -> tf.Tensor:
    waveform = tf.cast(waveform, tf.float32)
    max_val = tf.reduce_max(tf.abs(waveform))
    return waveform / (max_val + 1e-6)

def extract_features_from_waveform(waveform: tf.Tensor) -> tf.Tensor:
    spec = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    mag = tf.abs(spec)
    features = tf.reduce_mean(mag, axis=1)
    num_frames = tf.shape(features)[0]
    def pad():
        return tf.pad(features, [[0, 400 - num_frames]])
    def crop():
        return features[:400]
    fixed = tf.cond(num_frames < 400, pad, crop)
    return tf.reshape(fixed, (400, 1))

@app.post(
    "/predict",
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}
)
async def predict(file: UploadFile = File(...)):
    # 1. Model available?
    if model is None:
        raise HTTPException(status_code=503, detail="Classification model not available")

    # 2. Extension check
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Unsupported file type; only .wav allowed")

    # 3. Read contents, then size-check
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large; must be ≤ 5 MB")

    # 4. Decode WAV
    try:
        audio_tensor, sample_rate = tf.audio.decode_wav(
            tf.constant(contents),
            desired_channels=1,
            desired_samples=EXPECTED_SAMPLES
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode WAV: {e}")

    # 5. Sample rate validation
    sr = int(sample_rate.numpy())
    if sr != 16000:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sample rate: expected 16000 Hz, got {sr} Hz"
        )

    # 6. Preprocess & inference
    waveform = tf.squeeze(audio_tensor, axis=-1)
    waveform = normalize_waveform(waveform)
    features = extract_features_from_waveform(waveform)
    features = tf.expand_dims(features, axis=0)  # (1,400,1)

    logits = model(features, training=False)[0]
    probs = tf.nn.softmax(logits).numpy()
    idx = int(np.argmax(probs))
    label = class_names[idx]
    confidence = round(float(probs[idx]) * 100.0, 2)

    return {"predicted_class": label, "confidence_percent": confidence}

@app.get("/", response_model=dict)
def health_check():
    return {"status": "healthy"}
