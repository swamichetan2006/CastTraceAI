from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

CHECKPOINT = Path("models/CastTraceAI_model.ckpt")

def predict(image_path: str):
"""Run PatchCore inference using the trained checkpoint."""
if not CHECKPOINT.exists():
raise FileNotFoundError(
f"Checkpoint not found at {CHECKPOINT.resolve()}. "
"Download CastTraceAI_model.ckpt and place it in the models folder."
)

```
# The model was trained using MVTec AD's metal_nut category.
datamodule = MVTecAD(
    root="./datasets",
    category="metal_nut",
    train_batch_size=1,
    eval_batch_size=1,
)

model = Patchcore()
engine = Engine(accelerator="auto", devices=1)

# This is the test/prediction pipeline; exact support for arbitrary
# single-image inputs must be verified against the training configuration.
results = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=str(CHECKPOINT),
)

return results
```

if **name** == "**main**":
import argparse

```
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Image to evaluate")
args = parser.parse_args()

predictions = predict(args.image)
print(predictions)
```
