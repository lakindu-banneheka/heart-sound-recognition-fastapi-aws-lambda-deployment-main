import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from mangum import Mangum

# --- initialize ---
app = FastAPI()
handler = Mangum(app)

MODEL_PATH = "models/lung_sound_classification_model.keras"
EXPECTED_SAMPLES = 16000  # Adjust based on training configuration

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

# --- Define class names ---
# Ensure these match the labels used during training
class_names = ["Healthy", "Unhealthy"]

# --- Audio preprocessing functions ---
def normalize_waveform(waveform: tf.Tensor) -> tf.Tensor:
    waveform = tf.cast(waveform, tf.float32)
    max_val = tf.reduce_max(tf.abs(waveform))
    return waveform / (max_val + 1e-6)

def extract_features_from_waveform(waveform: tf.Tensor) -> tf.Tensor:
    # You should replace this with your actual feature extraction used during training
    # Here we use a simple STFT + flattening + padding
    spec = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    mag = tf.abs(spec)
    features = tf.reduce_mean(mag, axis=1)  # Simplified feature example
    features = tf.pad(features, [[0, 400 - tf.shape(features)[0]]], constant_values=0)  # Pad or crop
    return tf.reshape(features[:400], (400, 1))  # Ensure shape (timesteps, features)

# --- Inference Endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    try:
        contents = await file.read()
        audio_tensor, sample_rate = tf.audio.decode_wav(contents, desired_channels=1, desired_samples=EXPECTED_SAMPLES)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted .wav file")

    waveform = tf.squeeze(audio_tensor, axis=-1)
    waveform = normalize_waveform(waveform)
    features = extract_features_from_waveform(waveform)
    features = tf.expand_dims(features, axis=0)  # (1, timesteps, features)

    logits = model(features, training=False)[0]
    probs = tf.nn.softmax(logits).numpy()
    class_id = int(np.argmax(probs))
    predicted_label = class_names[class_id]
    confidence = float(probs[class_id] * 100.0)

    return JSONResponse({
        "predicted_class": predicted_label,
        "confidence_percent": round(confidence, 2)
    })

@app.get("/")
def health_check():
    return {"status": "healthy"}
