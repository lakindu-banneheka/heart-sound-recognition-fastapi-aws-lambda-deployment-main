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
EXPECTED_SAMPLES = 16_000      # exactly 1 second @ 16 kHz
EXPECTED_FRAMES = 25           # model’s time-frame count
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
    # 1) STFT → [frames, freq_bins]
    spec = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    mag = tf.abs(spec)                             # [frames, bins]
    features = tf.reduce_mean(mag, axis=1)         # → [frames]
    # 2) Pad or crop to EXPECTED_FRAMES
    num_frames = tf.shape(features)[0]
    def pad():
        return tf.pad(features, [[0, EXPECTED_FRAMES - num_frames]])
    def crop():
        return features[:EXPECTED_FRAMES]
    fixed = tf.cond(num_frames < EXPECTED_FRAMES, pad, crop)
    # 3) Reshape to (EXPECTED_FRAMES, 1)
    return tf.reshape(fixed, (EXPECTED_FRAMES, 1))

@app.post(
    "/predict",
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}
)
async def predict(file: UploadFile = File(...)):
    # — model availability
    if model is None:
        raise HTTPException(status_code=503, detail="Classification model not available")

    # — file validity checks
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Unsupported file type; only .wav allowed")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large; must be ≤ 5 MB")

    # — decode full-length WAV at native rate
    try:
        audio_tensor, sample_rate = tf.audio.decode_wav(
            tf.constant(contents),
            desired_channels=1,
            desired_samples=None
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode WAV: {e}")

    waveform = tf.squeeze(audio_tensor, axis=-1)
    sr = int(sample_rate.numpy())
    logger.info(f"Received audio @ {sr} Hz, {waveform.shape[0]} samples")

        # — resample or up/down-sample to EXPECTED_SAMPLES (16 kHz)
    try:
        import tensorflow_io as tfio
        # Resample to 16 kHz
        if sr != EXPECTED_SAMPLES:
            logger.info(f"Resampling from {sr} Hz to {EXPECTED_SAMPLES} Hz via tensorflow-io")
            waveform = tfio.audio.resample(waveform, rate_in=sr, rate_out=EXPECTED_SAMPLES)
    except ImportError:
        raise HTTPException(status_code=500, detail="Audio resampling requires tensorflow-io; please install 'tensorflow-io'.")

    # Ensure exactly EXPECTED_SAMPLES length
    num_samples = tf.shape(waveform)[0]
    def pad_samples():
        return tf.pad(waveform, [[0, EXPECTED_SAMPLES - num_samples]])
    def crop_samples():
        return waveform[:EXPECTED_SAMPLES]
    waveform = tf.cond(num_samples < EXPECTED_SAMPLES, pad_samples, crop_samples)

    # — normalize, extract features, predict
    waveform = normalize_waveform(waveform)
    features = extract_features_from_waveform(waveform)
    features = tf.expand_dims(features, axis=0)  # → (1, EXPECTED_FRAMES, 1)

    logits = model(features, training=False)[0]
    probs = tf.nn.softmax(logits).numpy()
    idx = int(np.argmax(probs))
    label = class_names[idx]
    confidence = round(float(probs[idx]) * 100.0, 2)

    return {"predicted_class": label, "confidence_percent": confidence}

@app.get("/", response_model=dict)
def health_check():
    return {"status": "healthy"}
