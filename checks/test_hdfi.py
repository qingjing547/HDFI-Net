"""Offline smoke checks: python -m unittest discover -s checks -p test_hdfi.py -v."""

import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import DBID, DLB, FCFB, GEB, DRConv, reparameterize_hdfi
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "ultralytics/cfg/models/HDFI-Net/HDFI-Net.yaml"


class HDFINetSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def build_model(self):
        torch.manual_seed(0)
        return DetectionModel(str(CFG), verbose=False)

    def test_full_resolution_and_module_positions(self):
        model = self.build_model().eval()
        for index in (2, 4):
            self.assertIsInstance(model.model[index], DLB)
        for index in (6, 8, 22):
            self.assertIsInstance(model.model[index], GEB)
        for index in (13, 16, 19):
            self.assertIsInstance(model.model[index], FCFB)
        for index in (3, 5, 7, 17, 20):
            self.assertIsInstance(model.model[index], DBID)
        self.assertEqual([len(model.model[i].m) for i in (2, 4, 6, 8, 13, 16, 19, 22)], [1, 1, 1, 2, 1, 2, 1, 1])
        self.assertEqual(model.model[8].c, 133)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1794386)
        self.assertFalse(model.model[-1].legacy)
        with torch.no_grad():
            output = model(torch.rand(1, 3, 640, 640))[0]
        self.assertEqual(tuple(output.shape), (1, 9, 8400))

    def test_training_loss_and_backward(self):
        model = self.build_model().train()
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            "img": torch.rand(2, 3, 128, 128),
            "batch_idx": torch.tensor([0, 1]),
            "cls": torch.tensor([[0.0], [1.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25], [0.4, 0.4, 0.2, 0.2]]),
        }
        loss, components = model(batch)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(components).all())
        loss.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))

    def test_standard_fusion_preserves_research_structure(self):
        model = self.build_model().eval()
        image = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            before = model(image)[0]
            model.fuse(verbose=False)
            after = model(image)[0]
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1788424)
        torch.testing.assert_close(after, before, atol=1e-4, rtol=1e-4)

    def test_deployment_reparameterization(self):
        model = self.build_model().eval()
        image = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            before = model(image)[0]
            reparameterize_hdfi(model)
            after = model(image)[0]
            reparameterize_hdfi(model)
            repeated = model(image)[0]
        torch.testing.assert_close(after, before, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(repeated, after, atol=0, rtol=0)
        self.assertTrue(all(module.deploy for module in model.modules() if isinstance(module, DRConv)))
        with torch.no_grad():
            model.fuse(verbose=False)
            fused = model(image)[0]
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1784464)
        torch.testing.assert_close(fused, before, atol=1e-4, rtol=1e-4)
        with self.assertRaises(ValueError):
            reparameterize_hdfi(self.build_model().train())

    def test_new_checkpoint_round_trip(self):
        model = YOLO(str(CFG), task="detect")
        with tempfile.TemporaryDirectory(prefix="hdfi-check-", dir=ROOT / "checks") as directory:
            weights = Path(directory) / "roundtrip.pt"
            model.save(str(weights))
            restored = YOLO(str(weights), task="detect")
            self.assertEqual(set(restored.model.state_dict()), set(model.model.state_dict()))
            restored.model.eval()
            with torch.no_grad():
                output = restored.model(torch.rand(1, 3, 128, 128))[0]
            self.assertEqual(tuple(output.shape), (1, 9, 336))


class HDFIEntryPointTests(unittest.TestCase):
    def load_entry(self, name):
        spec = importlib.util.spec_from_file_location(f"hdfi_{name}_entry", ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_blank_paths_stop_before_model_loading(self):
        for name in ("train", "val", "test"):
            with self.subTest(entry=name):
                entry = self.load_entry(name)
                self.assertEqual(entry.DATA_YAML_PATH, "")
                self.assertEqual(entry.OUTPUT_PROJECT, "")
                self.assertEqual(entry.OUTPUT_NAME, "")
                input_name = "MODEL_YAML_PATH" if name == "train" else "WEIGHTS_PATH"
                self.assertEqual(getattr(entry, input_name), "")
                with patch.object(entry, "YOLO") as loader, self.assertRaisesRegex(ValueError, "OUTPUT_PROJECT"):
                    entry.main()
                loader.assert_not_called()

    def test_validation_and_test_use_separate_splits(self):
        self.assertEqual(self.load_entry("val").VAL_ARGS["split"], "val")
        self.assertEqual(self.load_entry("test").TEST_ARGS["split"], "test")

    def test_test_metrics_handle_zero_values(self):
        entry = self.load_entry("test")
        result = SimpleNamespace(box=SimpleNamespace(mp=0.0, mr=0.0, map50=0.0, map=0.0), speed={})
        with redirect_stdout(io.StringIO()) as output:
            entry.print_metrics(result, 0.0)
        self.assertIn("Pipeline FPS", output.getvalue())


if __name__ == "__main__":
    unittest.main()
