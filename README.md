# FastAPI Lung Sound Classification with AWS Lambda & ECR

This project is a serverless lung sound classification API built with FastAPI and TensorFlow, deployed on AWS Lambda using a Docker container. The CI/CD pipeline automatically builds and pushes the Docker image to Amazon ECR on every push to the main branch.

---

## Project Structure

```

.
├── .github/workflows/
│   └── ecr-cicd.yml            # GitHub Actions CI/CD pipeline
├── models/
│   └── audio_model.keras       # Pre-trained speech recognition model
├── app.py                      # FastAPI application
├── Dockerfile                  # Lambda-compatible Docker container
└── requirements.txt            # Python dependencies

````

---

## How It Works

1. **FastAPI** handles incoming HTTP requests for audio prediction.
2. The `.wav` file is processed using `tensorflow.audio` and converted to a spectrogram.
3. A trained Keras model predicts the spoken word.
4. **GitHub Actions** automatically:
   - Builds the Docker image
   - Logs in to Amazon ECR
   - Pushes the image to your ECR repository

---

## Endpoints

### `POST /predict-audio`
- **Description:** Predicts the spoken command from a `.wav` audio file.
- **Request:** `multipart/form-data` with field `audio_file`
- **Response:**
```json
{
  "predicted_class": "healthy",
  "confidence_percent": 94.76
}
````

### `GET /`

* Health check endpoint to verify the service is running.

---

## Deployment (CI/CD)

The pipeline in `.github/workflows/ecr-cicd.yml`:

* Triggers on every push to `main`
* Builds Docker image using the `Dockerfile`
* Pushes it to ECR using the credentials in GitHub Secrets

### Required GitHub Secrets

| Name                    | Description                       |
| ----------------------- | --------------------------------- |
| `AWS_REGION`            | e.g. `us-east-1`                  |
| `AWS_ACCESS_KEY_ID`     | Your AWS access key ID            |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret access key        |
| `ECR_REPOSITORY_URI`    | URI of your Amazon ECR repository |

---

## Docker

To build and test locally:

```bash
docker build -t fastapi-lung-classifier .
docker run -p 8080:8080 fastapi-lung-classifier
```

Then test the API at: `http://localhost:8080/predict-audio`

---

## Example Request (cURL)

```bash
curl -X POST "http://localhost:8080/predict-audio" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@sample_lung_sound.wav"
```

---

## Model Info

* Format: `lung_sound_classification_model.keras`
* Input: `.wav` mono audio, 16000 samples
* Preprocessing: Normalization → Spectrogram conversion
* Output Classes: `['healthy','unhealthy']`

---

## Contributing
Feel free to submit issues, fork the repository, and send pull requests. Contributions are welcome!

## Author
- **Lakindu Banneheka**
