from fastapi import FastAPI, File, UploadFile, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from mangum import Mangum
import numpy as np
import tensorflow as tf
import librosa
import logging
from typing import Optional

# ——— Setup & Logging ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lung Sound Classifier API")
handler = Mangum(app)

MODEL_PATH = "models/lung_sound_classification_model.keras"
EXPECTED_SAMPLES = 2000      # number of audio samples per file
EXPECTED_FRAMES = 25         # number of time‐frames the model was trained on
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
    logger.critical(f"Model loading failed: {e}")
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model loading failed")

@app.post("/predict/", response_model=dict)
async def predict(file: UploadFile = File(...)):
    """
    Endpoint for lung sound classification.
    """
    # Validate file size
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File size too large")

    try:
        # Load and preprocess audio
        contents = await file.read()
        signal, sr = librosa.load(contents, sr=2000, duration=5, res_type='kaiser_fast')

        # Pad/trim signal to the expected length
        if len(signal) < sr * 5:
            signal = librosa.util.fix_length(signal, sr * 5)

        # Extract MFCCs
        mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=400).T, axis=0)
        mfccs = mfccs.reshape(1, mfccs.shape[0], 1)  # Reshape for model input

        # Make prediction
        prediction = model.predict(mfccs)

        # Get predicted class and confidence
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        # Map class index to label (assuming you have a LabelEncoder)
        class_labels = ['Healthy', 'Unhealthy']  # Replace with your actual labels
        predicted_label = class_labels[predicted_class]

        return {"prediction": predicted_label, "confidence": float(confidence)}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed")