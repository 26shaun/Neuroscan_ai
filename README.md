# NeuroScan AI – AI-Assisted Brain MRI Screening and Referral Support System

NeuroScan AI is a web-based prototype that helps with the preliminary screening of brain MRI scans. A user can upload several MRI scans from different patients in one session, and the system analyses **each scan independently**, producing a separate prediction, confidence score, and referral flag for every patient.

> **NeuroScan AI is an experimental screening-support tool, not a diagnostic device. All results require review by a qualified medical professional.**

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Proposed Solution](#proposed-solution)
3. [Key Features](#key-features)
4. [System Workflow](#system-workflow)
5. [AI/ML Approach](#aiml-approach)
6. [Technology Stack](#technology-stack)
7. [Project Architecture](#project-architecture)
8. [Project Folder Structure](#project-folder-structure)
9. [Installation and Setup](#installation-and-setup)
10. [How to Run the Project](#how-to-run-the-project)
11. [How to Use the System](#how-to-use-the-system)
12. [Example Output](#example-output)
13. [Limitations](#limitations)
14. [Medical Disclaimer](#medical-disclaimer)
15. [Future Enhancements](#future-enhancements)
16. [Contributors](#contributors)
17. [License](#license)

---

## Problem Statement

Brain tumours need to be identified early, but reading an MRI scan requires a trained radiologist or neurologist. In rural and resource-constrained healthcare settings, these specialists may be far away, and scans can wait a long time before anyone looks at them. Health workers at the first point of care often have no quick way to tell which scans are more likely to need urgent specialist attention.

## Proposed Solution

NeuroScan AI is a lightweight screening-support tool that gives a first-pass classification of brain MRI images and clearly marks every result for specialist review. It does not replace a doctor. The aim is to help a health worker organise a queue of scans and decide whom to refer first.

The model is built on MobileNetV2, a small convolutional network that is practical to run on modest hardware. Predictions are shown together with Grad-CAM heatmaps so a reviewer can see which image regions influenced the model's output.

**Each patient's MRI is processed independently.** Scans from different patients are never merged into one input, and one patient's scan has no influence on another patient's result.

## Key Features

The table shows the current implementation status so the repository stays honest about what is working.

| Feature | Description | Status |
|----------|------------|--------|
| Multiple MRI Upload | Upload several scans from different patients in a single session | Implemented |
| Individual Classification | Each scan is validated, preprocessed, and classified on its own into one of four classes | Implemented (`POST /predict/batch`) |
| Grad-CAM Visualization | Heatmap overlay for each scan showing the regions that influenced the prediction | Implemented |
| Referral Support | Every result carries a specialist-review notice; prioritisation rules are to be defined | Implemented (rule-based flag; see Example Output) |
| Results Dashboard | Per-class probability bars, predicted class, and confidence for a scan | Implemented |
| Batch Screening | A session summary of how many scans were screened and how many were flagged | Implemented (session counters) |

Already implemented in the current code:

- Drag-and-drop upload with client-side checks on file type and size
- File validation on the server (MIME type, size under 5 MB, and a check that the content is genuinely a JPEG or PNG)
- Stateless processing: images are decoded from an in-memory buffer and are never written to disk or a database
- Four-class softmax output, predicted label, and confidence score
- Loading indicator during inference and a clear error message when a request fails

## System Workflow

```
Multiple MRI scans
        ↓
Validate each uploaded image
        ↓
Image preprocessing
        ↓
MobileNetV2 inference
        ↓
Individual prediction for each patient
        ↓
Confidence score
        ↓
Grad-CAM heatmap for each scan
        ↓
Results dashboard
        ↓
Specialist-review flag
```

Every step after the upload is run **once per scan**. The results are kept attached to the scan they came from, using the file name and a user-supplied reference label (use an anonymised ID, not a patient's real name).

## AI/ML Approach

### MobileNetV2

MobileNetV2 is a compact convolutional neural network designed for devices with limited computing power. It uses depthwise separable convolutions and inverted residual blocks, which keep the parameter count and computation low. This makes it a sensible choice for settings where powerful hardware may not be available.

### Transfer Learning

Rather than training a network from scratch, the model starts from a MobileNetV2 pretrained on a large general image dataset. The classification head is replaced with a four-unit softmax layer and the network is then fine-tuned on brain MRI images. This usually works better than training from scratch when the medical dataset is relatively small.

- Training dataset: [Add Dataset Name and Link]
- Training notebook: [Add Notebook Link]

### Four-Class Classification

Each scan is assigned to one of four classes:

| Class | Meaning |
|---|---|
| Glioma | Glioma tumour pattern predicted |
| Meningioma | Meningioma tumour pattern predicted |
| Pituitary Tumor | Pituitary tumour pattern predicted |
| No Tumor Detected | No tumour pattern predicted |

The class order used by the API is `glioma`, `meningioma`, `notumor`, `pituitary`. It must match the order used during training (set in `CLASS_NAMES` inside `app.py`).

The confidence score is the highest softmax probability. It shows how strongly the model prefers one class over the others, but it is **not** a calibrated probability that the prediction is correct.

### Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) uses the gradients of the predicted class with respect to the last convolutional feature maps to build a heatmap. The heatmap is overlaid on the original scan, highlighting the regions that contributed most to the decision. It is meant to help a reviewer judge whether the model looked at a sensible area. It is an interpretability aid and should not be treated as an exact tumour segmentation.

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Backend / REST API | FastAPI, Uvicorn |
| Deep learning | TensorFlow / Keras, MobileNetV2, Transfer Learning |
| Image processing | Pillow, OpenCV, NumPy |
| Explainability | Grad-CAM |
| Frontend | HTML, CSS, JavaScript |

OpenCV is used to blend the Grad-CAM heatmap over each scan.

## Project Architecture

```
┌──────────────────────┐   multipart upload    ┌─────────────────────────────┐
│   Browser (HTML/CSS/ │ ───────────────────▶ │  FastAPI backend (app.py)   │
│   JavaScript)        │                       │  1. Validate type and size  │
│                      │                       │  2. Decode in memory        │
│  - Drag-and-drop     │                       │  3. Preprocess 224x224      │
│  - Per-scan results  │ ◀─────────────────── │  4. MobileNetV2 inference   │
│  - Referral notice   │    JSON response      │  5. Return probabilities    │
└──────────────────────┘                       └─────────────────────────────┘
                                                  No database. No disk storage.
```

The frontend and backend communicate through a REST API. For multiple scans, the frontend sends the files in one request to `POST /predict/batch`. The backend loops over them and validates, preprocesses, classifies, and generates a Grad-CAM for each file separately with the same loaded model. Images are never stacked or combined, and the results come back in upload order, so each result stays tied to its scan. A failure on one file is reported for that file only.

**Privacy design:** uploaded images exist only in server memory for the duration of a request. The backend does not store images, does not keep a database, and sends `Cache-Control: no-store` on prediction responses.

## Project Folder Structure

```
neuroscan/
├── app.py                  # FastAPI backend
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
├── model/
│   ├── README.md           # Where to place the weights
│   └── mobilenetv2_brisc.keras   # Not included in the repository
└── frontend/
    ├── index.html          # Dashboard layout
    ├── style.css           # Clinical light theme
    └── script.js           # Upload, API calls, result rendering
```

Planned additions: a Grad-CAM module, a training notebook, and automated tests.

## Installation and Setup

### Prerequisites

- Python 3.10 to 3.13 (TensorFlow does not currently publish wheels for Python 3.14)
- Git
- A trained model file at `model/mobilenetv2_brisc.keras` (not included in this repository)

### Steps

1. Clone the repository:

   ```bash
   git clone [Add Repository Link]
   cd neuroscan
   ```

2. Create and activate a virtual environment.

   Windows (PowerShell):

   ```powershell
   py -3.12 -m venv venv
   venv\Scripts\activate
   ```

   macOS / Linux:

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Place the trained weights at `model/mobilenetv2_brisc.keras`.

## How to Run the Project

Start the server from the project folder:

```bash
uvicorn app:app --port 8000
```

Wait for the log line `Application startup complete`, then open `http://localhost:8000` in a browser. The backend serves the frontend, so no separate web server is needed.

To check that the API is running:

```bash
curl http://localhost:8000/health
```

To test a single prediction from the command line:

```bash
curl -X POST -F "file=@scan.jpg;type=image/jpeg" http://localhost:8000/predict
```

Interactive API documentation is available at `http://localhost:8000/docs`.

If the model file is missing, the server stops at startup with a `Model weights not found` error.

## How to Use the System

1. **Upload multiple MRI scans.** Drag and drop JPEG or PNG images onto the upload area, or click to select files. Each file must be under 5 MB. Use one file per patient scan.
2. **Process each scan independently.** Select *Analyze scan*. Each image is sent and processed on its own, so scans from different patients are never combined.
3. **View individual predictions.** Each scan gets its own predicted class: Glioma, Meningioma, Pituitary Tumor, or No Tumor Detected.
4. **View confidence scores.** The dashboard shows the confidence of the top class and a probability bar for all four classes.
5. **View Grad-CAM visualizations.** The heatmap overlay for each scan shows which regions influenced the prediction.
6. **Check specialist-review flags.** Every result carries the notice *"Experimental screening prototype – specialist review required."* Results, including No Tumor Detected, must be reviewed by a qualified professional.

## Example Output

The values below only show the format of the API response. They are **not** real model results.

Current single-scan response from `POST /predict`:

```json
{
  "predicted_class": "<one of: glioma | meningioma | notumor | pituitary>",
  "confidence": 0.0,
  "probabilities": {
    "glioma": 0.0,
    "meningioma": 0.0,
    "notumor": 0.0,
    "pituitary": 0.0
  },
  "inference_ms": 0.0,
  "notice": "Experimental screening prototype – specialist review required"
}
```

`POST /predict/batch` takes several files in the multipart field `files` (up to 20) and returns `{"results": [...]}` with one entry per scan: `scan_id`, `filename`, `prediction`, `confidence`, `probabilities`, `gradcam` (PNG data URI), `specialist_review`, and `review_reasons`. A scan that fails returns `status: "error"` with a message, and the rest of the batch is still processed.

A scan is flagged for specialist review when a tumor class is predicted or the top confidence is below 0.80 (override with the `NEUROSCAN_REVIEW_THRESHOLD` environment variable). This is a simple screening rule, not a clinically validated threshold.

Layout of the multi-patient results dashboard:

| Scan / Reference | Predicted class | Confidence | Grad-CAM | Review flag |
|---|---|---|---|---|
| [Scan 1] | [class] | [%] | [heatmap] | Specialist review required |
| [Scan 2] | [class] | [%] | [heatmap] | Specialist review required |
| [Scan 3] | [class] | [%] | [heatmap] | Specialist review required |

Screenshots: [Add Screenshots]  
Demo: [Add Demo Link]

## Limitations

- This is a student prototype and has **not** been clinically validated or tested on real patients.
- Model performance figures are not reported here: [Add Evaluation Results After Testing]. Any real-world accuracy should not be assumed from this repository.
- The model works on single 2D image slices, not full 3D MRI volumes or multi-sequence studies.
- It recognises only four classes. Other conditions, such as stroke, haemorrhage, abscess, or other tumour types, are not covered and may be misclassified as one of the four.
- The model may perform worse on images from scanners, sequences, or image-quality conditions that differ from its training data.
- Softmax confidence is not a calibrated measure of correctness. A prediction can be wrong with high confidence.
- Grad-CAM heatmaps are rough visual explanations, not precise tumour boundaries, and they have not been validated against expert annotations.
- Images uploaded as JPEG or PNG lose the metadata and bit depth of the original DICOM format.
- Uploads are limited to 5 MB per image.
- Results are not stored, so they are lost when the page is closed unless the user records them.

## Medical Disclaimer

**NeuroScan AI is an experimental screening-support tool, not a diagnostic device. All results require review by a qualified medical professional.**

This project is intended for research and educational purposes only. It must not be used as a substitute for professional medical evaluation, diagnosis, or treatment. Do not make clinical decisions based on its output. Do not upload identifiable patient data to any deployment of this system.

## Future Enhancements

- Referral prioritisation rules, for example based on predicted class and a confidence threshold [Define Threshold]
- Evaluation on a held-out test set, with per-class precision, recall, and a confusion matrix
- Confidence calibration and handling of out-of-distribution images
- DICOM file support
- Exportable per-patient screening report
- Automated tests for upload validation and the prediction endpoint
- Containerised deployment

## Contributors

- [Add Team Member Name]
- [Add Team Member Name]
- [Add Team Member Name]

## License

[Add License]#   N e u r o s c a n i _ a i  
 