from __future__ import annotations

import unittest

from submission.segment_qwen35_lora.answer_utils import (
    clean_generated_answer,
    generation_eos_token_ids,
)


class FakeTokenizer:
    eos_token_id = 10
    unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return 11 if token == "<|im_end|>" else self.unk_token_id


class Qwen35AnswerUtilsTest(unittest.TestCase):
    def test_stops_timestamp_before_leaked_chat_tokens(self) -> None:
        raw = (
            "00:18:26\nuser\nassistant\n<think>\nassistant\n<think>"
        )
        self.assertEqual(clean_generated_answer(raw), "00:18:26")

    def test_keeps_first_concise_non_time_answer(self) -> None:
        self.assertEqual(
            clean_generated_answer("Answer: Surgical Sponge\nExtra explanation"),
            "Surgical Sponge",
        )

    def test_prefers_text_after_closed_thinking_block(self) -> None:
        self.assertEqual(
            clean_generated_answer("<think>reasoning</think>00:01:02<|im_end|>"),
            "00:01:02",
        )

    def test_returns_model_and_chat_eos_ids(self) -> None:
        self.assertEqual(generation_eos_token_ids(FakeTokenizer()), [10, 11])


if __name__ == "__main__":
    unittest.main()
