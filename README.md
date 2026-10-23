# CastTrace AI

Industrial defect detection using PatchCore and Anomalib.

## Model Details

* Architecture: PatchCore
* Framework: Anomalib 2.7.0
* Dataset: MVTec Anomaly Detection (MVTec AD)
* Category: metal_nut

## Trained Model

The trained model checkpoint is shared separately through Google Drive.

Filename: `CastTraceAI_model.ckpt`

Place the downloaded checkpoint in the project's `models/` directory.

## Evaluation Results

| Metric         | Result |
| -------------- | -----: |
| Image AUROC    | 99.80% |
| Image F1 Score | 98.38% |
| Pixel AUROC    | 98.71% |
| Pixel F1 Score | 84.08% |

These metrics are from the completed test run. Accuracy, precision, and recall have not yet been separately verified for this checkpoint.

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Status

* [x] Model trained
* [x] Checkpoint saved
* [x] Model tested
* [ ] Inference script prepared and verified
* [ ] Integrated into the application

The checkpoint loader and inference script must be verified against the trained model before integration.
