from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts.check_qwen35_submission_resources import main


class CheckQwen35SubmissionResourcesTest(unittest.TestCase):
    def test_accepts_processor_config_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            algorithm_root = Path(temp_dir) / "segment-algorithm"
            base_dir = algorithm_root / "resources" / "qwen35-4b"
            adapter_dir = algorithm_root / "resources" / "qwen35-lora"
            base_dir.mkdir(parents=True)
            adapter_dir.mkdir(parents=True)

            for path in (
                algorithm_root / "inference.py",
                algorithm_root / "answer_utils.py",
                algorithm_root / "requirements.txt",
                base_dir / "tokenizer_config.json",
                base_dir / "processor_config.json",
                adapter_dir / "adapter_model.safetensors",
                adapter_dir / "tokenizer_config.json",
                adapter_dir / "processor_config.json",
            ):
                path.write_text("placeholder\n", encoding="utf-8")
            (base_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                    }
                ),
                encoding="utf-8",
            )
            (base_dir / "model.safetensors").write_bytes(b"weights")
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA"}), encoding="utf-8"
            )

            argv = [
                "check_qwen35_submission_resources.py",
                "--algorithm-root",
                str(algorithm_root),
            ]
            output = StringIO()
            with patch.object(sys, "argv", argv), redirect_stdout(output):
                main()

            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["symlink_count"], 0)


if __name__ == "__main__":
    unittest.main()
