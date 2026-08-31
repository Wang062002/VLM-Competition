from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.package_qwen35_training_artifacts import main


class PackageQwen35TrainingArtifactsTest(unittest.TestCase):
    def test_packages_completed_run_with_processor_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_root = root / "run"
            adapter_dir = run_root / "adapter-final"
            output_dir = root / "transfer"
            adapter_dir.mkdir(parents=True)

            summary = {
                "status": "completed",
                "epochs": 5,
                "eval_loss": 0.2403998877382303,
                "adapter_dir": str(adapter_dir),
                "history_path": str(run_root / "train_history.jsonl"),
            }
            (run_root / "training_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            (run_root / "train_history.jsonl").write_text(
                '{"epoch": 5}\n', encoding="utf-8"
            )
            for filename in (
                "adapter_config.json",
                "tokenizer_config.json",
                "processor_config.json",
            ):
                (adapter_dir / filename).write_text("{}\n", encoding="utf-8")
            (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")

            argv = [
                "package_qwen35_training_artifacts.py",
                "--run-root",
                str(run_root),
                "--output-dir",
                str(output_dir),
            ]
            with patch.object(sys, "argv", argv):
                main()

            archive_path = output_dir / "qwen35-lora-e5-final.tar.gz"
            checksum_path = output_dir / "qwen35-lora-e5-final.tar.gz.sha256"
            manifest_path = output_dir / "artifact_manifest.json"
            self.assertTrue(archive_path.is_file())
            self.assertTrue(manifest_path.is_file())

            expected_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            self.assertEqual(
                checksum_path.read_text(encoding="ascii"),
                f"{expected_digest}  {archive_path.name}\n",
            )
            with tarfile.open(archive_path, "r:gz") as archive:
                self.assertIn("adapter_model.safetensors", archive.getnames())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["training"]["status"], "completed")
            self.assertEqual(len(manifest["files"]), 3)


if __name__ == "__main__":
    unittest.main()
