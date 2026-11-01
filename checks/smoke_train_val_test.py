"""Run one CPU training epoch plus val/test on temporary synthetic data, not a scientific evaluation."""

import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import train as train_entry
    import val as val_entry
    from ultralytics.utils import YAML

    spec = importlib.util.spec_from_file_location("hdfi_test_entry", ROOT / "test.py")
    test_entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_entry)
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix="hdfi-pipeline-", dir=ROOT / "checks") as directory:
        temporary = Path(directory)
        generator = np.random.default_rng(0)
        for split, count in (("train", 2), ("val", 1), ("test", 1)):
            images = temporary / split / "images"
            labels = temporary / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for index in range(count):
                pixels = generator.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(images / f"{index}.jpg")
                (labels / f"{index}.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        data = temporary / "synthetic.yaml"
        YAML.save(
            data,
            {
                "path": str(temporary),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {0: "BLI", 1: "BS", 2: "GS", 3: "RL", 4: "RSM"},
            },
        )
        output = temporary / "outputs"
        # Disable optional external logging and package-update checks, not the training implementation.
        with patch("ultralytics.utils.callbacks.add_integration_callbacks"), patch(
            "ultralytics.utils.checks.check_pip_update_available", return_value=None
        ):
            with patch.multiple(
                train_entry,
                MODEL_YAML_PATH=str(ROOT / "ultralytics/cfg/models/HDFI-Net/HDFI-Net.yaml"),
                DATA_YAML_PATH=str(data),
                OUTPUT_PROJECT=str(output),
                OUTPUT_NAME="train",
                DEVICE="cpu",
                TRAIN_ARGS={**train_entry.TRAIN_ARGS, "workers": 0, "imgsz": 128, "epochs": 1, "batch": 2},
            ):
                train_entry.main()
            weights = output / "train/weights/best.pt"
            assert weights.is_file(), weights
            for split, entry, args_name in (("val", val_entry, "VAL_ARGS"), ("test", test_entry, "TEST_ARGS")):
                options = {**getattr(entry, args_name), "workers": 0, "imgsz": 128, "batch": 1, "plots": False}
                with patch.multiple(
                    entry,
                    WEIGHTS_PATH=str(weights),
                    DATA_YAML_PATH=str(data),
                    OUTPUT_PROJECT=str(output),
                    OUTPUT_NAME=split,
                    DEVICE="cpu",
                    **{args_name: options},
                ):
                    result = entry.main()
                assert entry is not None and options["split"] == split
                assert Path(result.save_dir) == output / split
        print("SYNTHETIC_TRAIN_VAL_TEST_PASSED (not an accuracy measurement)")


if __name__ == "__main__":
    main()
