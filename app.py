"""NeuroScan AI - stateless brain MRI screening API (MobileNetV2, 4-class).

Uploaded images are decoded from an in-memory buffer and never written to disk
or any database. Nothing from a request is retained after the response is sent.
"""

import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette import formparsers

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # noqa: E402

logger = logging.getLogger("neuroscan")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "mobilenetv2_brisc.keras"
FRONTEND_DIR = BASE_DIR / "frontend"

# Order must match the class indices used during training (alphabetical for
# Keras directory loaders: glioma, meningioma, notumor, pituitary).
CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
INPUT_SIZE = (224, 224)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_IMAGE_PIXELS = 25_000_000

CLINICAL_NOTICE = "Experimental screening prototype \u2013 specialist review required"

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor Detected",
    "pituitary": "Pituitary Tumor",
}
MAX_FILES_PER_BATCH = 20
# A scan is flagged for specialist review when a tumor class is predicted or the
# top-class confidence falls below this value. This is a screening rule, not a
# clinically validated cut-off.
REVIEW_CONFIDENCE_THRESHOLD = float(os.environ.get("NEUROSCAN_REVIEW_THRESHOLD", "0.80"))
GRADCAM_ALPHA = 0.4

# Keep uploads of up to the size limit in RAM instead of letting the multipart
# parser spool anything above 1 MB to a temporary file on disk.
if hasattr(formparsers.MultiPartParser, "spool_max_size"):
    formparsers.MultiPartParser.spool_max_size = MAX_UPLOAD_BYTES + READ_CHUNK_BYTES

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "NEUROSCAN_CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5500,http://127.0.0.1:5500",
    ).split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model weights not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    output_units = model.output_shape[-1]
    if output_units != len(CLASS_NAMES):
        raise RuntimeError(
            f"Model outputs {output_units} classes but {len(CLASS_NAMES)} are configured"
        )
    # Warm-up pass so the first real request does not pay graph-tracing cost.
    model.predict(np.zeros((1, *INPUT_SIZE, 3), dtype=np.float32), verbose=0)
    app.state.model = model
    try:
        app.state.gradcam = GradCam(model)
    except Exception:
        logger.exception("Grad-CAM disabled: could not build it for this model")
        app.state.gradcam = None
    logger.info("Model loaded from %s (Grad-CAM %s)", MODEL_PATH,
                "enabled" if app.state.gradcam else "disabled")
    yield
    app.state.model = None
    app.state.gradcam = None


app = FastAPI(
    title="NeuroScan AI",
    description="Stateless 4-class brain MRI screening prototype.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=600,
)


async def read_limited(upload: UploadFile) -> bytes:
    """Read an upload into memory, aborting as soon as the size limit is exceeded."""
    buffer = io.BytesIO()
    total = 0
    while True:
        chunk = await upload.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="File exceeds the 5 MB limit.",
            )
        buffer.write(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return buffer.getvalue()


def load_rgb(raw: bytes) -> np.ndarray:
    """Decode an image entirely in memory to a 224x224 RGB uint8 array."""
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail="File content is not a valid JPEG or PNG image.",
                )
            rgb = image.convert("RGB").resize(INPUT_SIZE, Image.Resampling.BILINEAR)
    except HTTPException:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="The file could not be decoded as a valid image.",
        )
    return np.asarray(rgb, dtype=np.uint8)


def to_model_input(rgb: np.ndarray) -> np.ndarray:
    """Apply the MobileNetV2 normalization and add the batch dimension."""
    array = rgb.astype(np.float32)
    return preprocess_input(np.expand_dims(array, axis=0))


def decode_image(raw: bytes) -> np.ndarray:
    """Decode and preprocess an image entirely in memory."""
    return to_model_input(load_rgb(raw))


class GradCam:
    """Grad-CAM for the loaded model, computed per image.

    Supports a flat MobileNetV2 graph (uses the "out_relu" layer) and a model
    that nests the MobileNetV2 backbone as a single layer followed by a head.
    The wrapper is checked against the real model at startup; if its output
    does not match, construction fails and Grad-CAM is disabled.
    """

    def __init__(self, model: tf.keras.Model):
        self.model = model
        self._flat = None
        self._pre = []
        self._backbone = None
        self._post = []
        try:
            layer = model.get_layer("out_relu")
        except ValueError:
            layer = None
        if layer is not None:
            self._flat = tf.keras.Model(model.inputs, [layer.output, model.output])
        else:
            self._build_nested()
        probe = np.random.default_rng(0).uniform(-1, 1, (1, *INPUT_SIZE, 3)).astype(np.float32)
        _, wrapped = self._forward(tf.convert_to_tensor(probe))
        expected = model(probe, training=False)
        if not np.allclose(wrapped.numpy(), expected.numpy(), atol=1e-4):
            raise ValueError("Grad-CAM forward pass does not match the model output")

    def _build_nested(self) -> None:
        layers = [l for l in self.model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
        for index, layer in enumerate(layers):
            if isinstance(layer, tf.keras.Model) and len(layer.output_shape) == 4:
                self._pre = layers[:index]
                self._backbone = layer
                self._post = layers[index + 1:]
                return
        raise ValueError("No convolutional backbone found for Grad-CAM")

    def _forward(self, x):
        if self._flat is not None:
            conv, preds = self._flat(x, training=False)
            return conv, preds
        for layer in self._pre:
            x = layer(x, training=False)
        conv = self._backbone(x, training=False)
        y = conv
        for layer in self._post:
            y = layer(y, training=False)
        return conv, y

    def overlay(self, batch: np.ndarray, class_index: int, rgb: np.ndarray) -> str:
        """Return a PNG data URI of the heatmap blended over the given RGB image."""
        x = tf.convert_to_tensor(batch)
        with tf.GradientTape() as tape:
            tape.watch(x)
            conv, preds = self._forward(x)
            score = preds[:, class_index]
        grads = tape.gradient(score, conv)
        if grads is None:
            raise RuntimeError("No gradient available for Grad-CAM")
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(conv * weights, axis=-1))[0].numpy()
        peak = float(cam.max())
        if peak > 0:
            cam = cam / peak
        cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0.0, 1.0)), cv2.COLORMAP_JET)
        base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        blended = cv2.addWeighted(base, 1.0 - GRADCAM_ALPHA, heat, GRADCAM_ALPHA, 0)
        ok, encoded = cv2.imencode(".png", blended)
        if not ok:
            raise RuntimeError("Could not encode Grad-CAM image")
        return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def analyze_scan(scan_id: str, filename: str, raw: bytes, model: tf.keras.Model, gradcam) -> dict:
    """Run the full pipeline for ONE scan: decode, infer, Grad-CAM, review flag."""
    started = time.perf_counter()
    rgb = load_rgb(raw)
    batch = to_model_input(rgb)
    probabilities = run_inference(model, batch)

    top_index = int(np.argmax(probabilities))
    predicted = CLASS_NAMES[top_index]
    confidence = float(probabilities[top_index])

    reasons = []
    if predicted != "notumor":
        reasons.append("A tumor class was predicted")
    if confidence < REVIEW_CONFIDENCE_THRESHOLD:
        reasons.append(f"Confidence is below {REVIEW_CONFIDENCE_THRESHOLD:.0%}")

    heatmap = None
    heatmap_error = None
    if gradcam is None:
        heatmap_error = "Grad-CAM is unavailable for the loaded model."
    else:
        try:
            heatmap = gradcam.overlay(batch, top_index, rgb)
        except Exception:
            logger.exception("Grad-CAM failure for %s", scan_id)
            heatmap_error = "Grad-CAM could not be generated for this scan."

    return {
        "scan_id": scan_id,
        "filename": filename,
        "status": "ok",
        "prediction": DISPLAY_NAMES[predicted],
        "predicted_class": predicted,
        "confidence": round(confidence, 6),
        "probabilities": {n: round(float(p), 6) for n, p in zip(CLASS_NAMES, probabilities)},
        "gradcam": heatmap,
        "gradcam_error": heatmap_error,
        "specialist_review": bool(reasons),
        "review_reasons": reasons,
        "inference_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def run_inference(model: tf.keras.Model, batch: np.ndarray) -> np.ndarray:
    return model.predict(batch, verbose=0)[0]


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
        "gradcam_available": app.state.gradcam is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a JPEG or PNG image.",
        )

    started = time.perf_counter()
    raw = await read_limited(file)
    batch = await run_in_threadpool(decode_image, raw)
    del raw

    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available.")

    try:
        probabilities = await run_in_threadpool(run_inference, model, batch)
    except Exception:
        logger.exception("Inference failure")
        raise HTTPException(status_code=500, detail="Inference failed.")
    finally:
        del batch

    top_index = int(np.argmax(probabilities))
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    return {
        "predicted_class": CLASS_NAMES[top_index],
        "confidence": round(float(probabilities[top_index]), 6),
        "probabilities": {
            name: round(float(p), 6) for name, p in zip(CLASS_NAMES, probabilities)
        },
        "inference_ms": elapsed_ms,
        "notice": CLINICAL_NOTICE,
    }


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)) -> dict:
    """Analyze several scans (one per patient) independently with the same loaded model."""
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    if len(files) > MAX_FILES_PER_BATCH:
        raise HTTPException(
            status_code=413,
            detail=f"At most {MAX_FILES_PER_BATCH} scans can be analyzed per request.",
        )
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available.")
    gradcam = app.state.gradcam

    results = []
    for position, upload in enumerate(files, start=1):
        scan_id = f"MRI_{position:02d}"
        filename = Path(upload.filename or "").name[:120] or f"scan_{position:02d}"
        try:
            if upload.content_type not in ALLOWED_MIME_TYPES:
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported file type. Upload a JPEG or PNG image.",
                )
            raw = await read_limited(upload)
            result = await run_in_threadpool(analyze_scan, scan_id, filename, raw, model, gradcam)
        except HTTPException as exc:
            result = {"scan_id": scan_id, "filename": filename, "status": "error", "error": str(exc.detail)}
        except Exception:
            logger.exception("Analysis failure for %s", scan_id)
            result = {
                "scan_id": scan_id,
                "filename": filename,
                "status": "error",
                "error": "Analysis failed for this scan.",
            }
        finally:
            await upload.close()
        results.append(result)

    succeeded = [r for r in results if r["status"] == "ok"]
    return {
        "results": results,
        "total": len(results),
        "succeeded": len(succeeded),
        "failed": len(results) - len(succeeded),
        "flagged": sum(1 for r in succeeded if r["specialist_review"]),
        "notice": CLINICAL_NOTICE,
    }


@app.middleware("http")
async def no_store_headers(request, call_next):
    response = await call_next(request)
    if request.url.path in ("/predict", "/predict/batch", "/health"):
        response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)