```python
from pathlib import Path
import argparse
import torch
from anomalib.models import Patchcore
from anomalib.engine import Engine

def predict(image_path, checkpoint="models/CastTraceAI_model.ckpt"):
    image_path = Path(image_path)
    checkpoint = Path(checkpoint)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    engine = Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    results = engine.predict(
        model=Patchcore(),
        data_path=str(image_path),
        ckpt_path=str(checkpoint),
        return_predictions=True,
    )

    if not results:
        raise RuntimeError("No prediction returned.")

    result = results[0]
    score = result.pred_score[0].item()
    label = "DEFECT" if result.pred_label[0].item() else "NORMAL"

    print(f"Image: {image_path}")
    print(f"Anomaly score: {score:.4f}")
    print(f"Prediction: {label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the image to inspect")
    parser.add_argument(
        "--checkpoint",
        default="models/CastTraceAI_model.ckpt",
    )
    args = parser.parse_args()
    predict(args.image, args.checkpoint)
```
