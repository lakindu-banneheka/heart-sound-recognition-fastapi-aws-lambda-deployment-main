import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from mangum import Mangum

# Initialize FastAPI app and Lambda handler
app = FastAPI()
handler = Mangum(app)

# Load the model once during cold start
MODEL_PATH = "models/lung_sound_classification_model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Function to extract MFCC using TensorFlow
def extract_mfcc(waveform: tf.Tensor, sample_rate: int, num_mfcc: int = 1, frame_count: int = 25) -> np.ndarray:
    stft = tf.signal.stft(waveform, frame_length=640, frame_step=320, fft_length=1024)
    spectrogram = tf.abs(stft)

    num_spectrogram_bins = spectrogram.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate,
        lower_edge_hertz, upper_edge_hertz
    )
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # Extract MFCCs: get only 1 coefficient per frame
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_mfcc]

    # Pad or trim to exactly 25 frames
    mfccs = mfccs[:frame_count, :]  # Trim if longer
    padding = frame_count - tf.shape(mfccs)[0]
    mfccs = tf.pad(mfccs, [[0, padding], [0, 0]])

    return mfccs.numpy()  # shape: (25, 1)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    contents = await file.read()

    try:
        audio_tensor, sample_rate = tf.audio.decode_wav(contents, desired_channels=1)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid WAV file")

    waveform = tf.squeeze(audio_tensor, axis=-1)

    try:
        features = extract_mfcc(waveform, sample_rate)
        input_tensor = np.expand_dims(features, axis=0)  # Reshape for model: (1, features)
        prediction = model.predict(input_tensor)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        status = "Healthy" if predicted_class == 1 else "Unhealthy"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return JSONResponse({
        "status": status,
        "confidence": round(confidence * 100, 2)
    })

@app.get("/")
def health_check():
    return {"status": "healthy"}
