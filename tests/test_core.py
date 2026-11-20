"""Run: python -m unittest discover -s tests -v."""
from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from freshness.common import CLASSES, configure, image_hash, letterbox, read_rgb, write_csv
from freshness.data import Transform, audit, prepare
from freshness.evaluate import run_batch
from freshness.inference import Predictor
from freshness.metrics import classification_metrics, timing_stats
from freshness.model import build_model, load_checkpoint, save_checkpoint
from freshness.smoke import make_synthetic_data


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(threads=2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, groups=6):
        raw = self.base / "raw"
        make_synthetic_data(raw, groups_per_class=groups, views=2)
        return raw

    def prepare_fixture(self, groups=6):
        raw, output = self.fixture(groups), self.base / "prepared"
        prepare(raw, output)
        return raw, output

    def checkpoint(self):
        path = self.base / "model.pt"
        model = build_model().eval()
        save_checkpoint(path, model, {"smoke_test": True})
        return path, model

    def test_letterbox_preserves_aspect(self):
        image = np.zeros((50, 100, 3), np.uint8)
        image[:, :, 0] = 255
        result = letterbox(image)
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertTrue(np.all(result[:56] == 114))
        self.assertEqual(int(result[100, 100, 0]), 255)

    def test_eval_and_fallback_transforms(self):
        image = np.zeros((250, 170, 3), np.uint8)
        for train in (False, True):
            result = Transform(train, "torchvision")(image)
            self.assertEqual(tuple(result.shape), (3, 224, 224))
            self.assertEqual(result.dtype, torch.float32)
            self.assertTrue(torch.isfinite(result).all())

    def test_grayscale_unicode_and_exif(self):
        path = self.base / "фото серое.png"
        Image.new("L", (12, 20), 100).save(path)
        self.assertEqual(read_rgb(path).shape, (20, 12, 3))
        rotated = self.base / "oriented.jpg"
        image = Image.new("RGB", (12, 20))
        exif = image.getexif()
        exif[274] = 6
        image.save(rotated, exif=exif)
        self.assertEqual(read_rgb(rotated).shape, (12, 20, 3))

    def test_transparency_uses_neutral_background(self):
        path = self.base / "transparent.png"
        Image.new("RGBA", (12, 20), (0, 0, 0, 0)).save(path)
        self.assertTrue(np.all(read_rgb(path) == 114))

    def test_invalid_image_rejected(self):
        path = self.base / "bad.jpg"
        path.write_bytes(b"not an image")
        with self.assertRaises(ValueError):
            read_rgb(path)

    def test_stratified_groups_and_repeatability(self):
        raw, prepared = self.prepare_fixture()
        first = audit(prepared)
        other = self.base / "again"
        prepare(raw, other)
        second = audit(other)
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(first["images"], 36)
        memberships = {}
        for row in first["manifest"]:
            memberships.setdefault(row["group_id"], set()).add(row["split"])
        self.assertTrue(all(len(v) == 1 for v in memberships.values()))
        self.assertTrue(all(n > 0 for split in first["counts"].values() for n in split.values()))

    def test_minimum_three_groups_per_class(self):
        _, prepared = self.prepare_fixture(groups=3)
        checked = audit(prepared)
        self.assertTrue(all(n > 0 for split in checked["counts"].values() for n in split.values()))

    def test_groups_spanning_multiple_labels(self):
        raw = self.fixture()
        for label in CLASSES:
            for path in list((raw / label).glob("*.png")):
                specimen = path.stem.split("__")[0].rsplit("_", 1)[1]
                view = path.stem.split("__")[1]
                path.rename(path.with_name(f"same_specimen_{specimen}__{view}.png"))
        output = self.base / "prepared"
        prepare(raw, output)
        result = audit(output)
        self.assertEqual(result["groups"], 6)
        memberships = {}
        for row in result["manifest"]:
            memberships.setdefault(row["group_id"], set()).add(row["split"])
        self.assertTrue(all(len(v) == 1 for v in memberships.values()))

    def test_exact_duplicates_removed(self):
        raw = self.fixture()
        original = next((raw / "FRESH").glob("*.png"))
        shutil.copy2(original, raw / "FRESH" / "duplicate.png")
        result = prepare(raw, self.base / "prepared")
        self.assertEqual(result["removed_exact_duplicates"], 1)
        self.assertEqual(result["unique_images"], 36)

    def test_conflicting_duplicate_labels_rejected(self):
        raw = self.fixture()
        original = next((raw / "FRESH").glob("*.png"))
        shutil.copy2(original, raw / "SPOILED" / "conflict.png")
        with self.assertRaises(ValueError):
            prepare(raw, self.base / "prepared")

    def test_missing_class_rejected(self):
        raw = self.fixture()
        shutil.rmtree(raw / "MINOR_DEFECT")
        with self.assertRaises(ValueError):
            prepare(raw, self.base / "prepared")

    def test_existing_output_not_overwritten(self):
        raw, output = self.prepare_fixture()
        with self.assertRaises(ValueError):
            prepare(raw, output)

    def test_modified_file_detected(self):
        _, output = self.prepare_fixture()
        path = next((output / "train/FRESH").glob("*.png"))
        Image.new("RGB", (20, 20), "red").save(path)
        with self.assertRaises(ValueError):
            audit(output)

    def test_group_leakage_rejected(self):
        _, output = self.prepare_fixture()
        manifest = output / "manifest.csv"
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        train = next(r for r in rows if r["split"] == "train")
        test = next(r for r in rows if r["split"] == "test")
        test["group_id"] = train["group_id"]
        write_csv(manifest, rows, list(rows[0]))
        with self.assertRaises(ValueError):
            audit(output)

    def test_extra_files_detected(self):
        _, output = self.prepare_fixture()
        path = output / "val/UNKNOWN/extra.png"
        path.parent.mkdir()
        Image.new("RGB", (20,20), "green").save(path)
        with self.assertRaises(ValueError):
            audit(output)

    def test_checkpoint_roundtrip_and_no_download(self):
        path, original = self.checkpoint()
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("NETWORK")):
            restored, metadata = load_checkpoint(path, torch.device("cpu"))
            tensor = torch.rand(1, 3, 224, 224)
            with torch.inference_mode():
                self.assertTrue(torch.equal(original(tensor), restored(tensor)))
        self.assertTrue(metadata["smoke_test"])

    def test_local_initial_weights_branch(self):
        from torchvision.models import mobilenet_v3_small
        original = mobilenet_v3_small(weights=None)
        path = self.base / "local_initialization.pth"
        torch.save(original.state_dict(), path)
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("NETWORK")):
            initialized = build_model("imagenet", initial_weights=path)
        self.assertEqual(initialized.classifier[-1].out_features, 3)
        self.assertTrue(torch.equal(original.features[0][0].weight,
                                    initialized.features[0][0].weight))

    def test_inference_without_network_socket(self):
        path, _ = self.checkpoint()
        with patch("socket.socket", side_effect=AssertionError("NETWORK")):
            predictor = Predictor(path, threads=2)
            result = predictor.predict_rgb(np.zeros((150, 180, 3), np.uint8))
        self.assertEqual(set(result["probabilities"]), set(CLASSES))

    @unittest.skipUnless(importlib.util.find_spec("albumentations"), "Albumentations is not installed")
    def test_albumentations_transform(self):
        image = np.random.default_rng(5).integers(0, 256, (260, 300, 3), dtype=np.uint8)
        first = Transform(True, "albumentations", seed=42)(image)
        second = Transform(True, "albumentations", seed=42)(image)
        self.assertEqual(tuple(first.shape), (3, 224, 224))
        self.assertTrue(torch.equal(first, second))

    def test_checkpoint_class_order_checked(self):
        path, _ = self.checkpoint()
        value = torch.load(path, weights_only=True)
        value["classes"] = ["SPOILED", "MINOR_DEFECT", "FRESH"]
        torch.save(value, path)
        with self.assertRaises(ValueError):
            load_checkpoint(path, torch.device("cpu"))

    def test_predictions_and_batch_artifacts(self):
        path, _ = self.checkpoint()
        predictor = Predictor(path, threads=2, threshold=1.0)
        folder = self.base / "photos"
        folder.mkdir()
        Image.new("RGB", (160, 120), "green").save(folder / "apple & pear.png")
        (folder / "broken.jpg").write_bytes(b"broken")
        output = self.base / "report"
        summary = run_batch(predictor, folder, output)
        self.assertEqual(summary["processed"], 1)
        self.assertEqual(len(summary["errors"]), 1)
        predictions = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(sum(predictions[0]["probabilities"].values()), 1., places=6)
        self.assertTrue(predictions[0]["needs_review"])
        self.assertTrue(predictions[0]["smoke_test_model"])
        self.assertIn("apple &amp; pear.png", (output / "report.html").read_text(encoding="utf-8"))
        self.assertTrue(list((output / "annotated").glob("*.jpg")))

    def test_metrics(self):
        result = classification_metrics([0, 0, 1, 2], [0, 1, 1, 2])
        self.assertEqual(result["accuracy"], .75)
        self.assertEqual(result["per_class"]["FRESH"]["recall"], .5)
        self.assertEqual(result["confusion_matrix"], [[1,1,0], [0,1,0], [0,0,1]])
        self.assertEqual(timing_stats([10.,20.,30.])["median_ms"], 20.)

    def test_missing_checkpoint_rejected(self):
        with self.assertRaises(FileNotFoundError):
            Predictor(self.base / "absent.pt")

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            Predictor(self.base / "absent.pt", threshold=2.0)


if __name__ == "__main__":
    unittest.main()
