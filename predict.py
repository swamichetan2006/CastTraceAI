from pathlib import Path
import torch
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

CHECKPOINT = Path("models/CastTraceAI_model.ckpt")


def predict_dataset(dataset_root="./datasets"):
    """Evaluate the metal_nut test dataset."""
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT.resolve()}"
        )

    datamodule = MVTecAD(
        root=dataset_root,
        category="metal_nut",
        train_batch_size=8,
        eval_batch_size=8,
    )

    model = Patchcore()
    engine = Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    return engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(CHECKPOINT),
    )


if __name__ == "__main__":
    print(predict_dataset())
